"""Build the single fast-mode Het-Infer input from completed DOPS snapshots."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import heapq
import json
from pathlib import Path
import re
from types import SimpleNamespace

import config as runtime_config
from plan_label import PlanLabel
from hetinfer_capture import route_time_s

NPU = "Ascend_910B_NPU0"
DEVICES = (NPU, "PIM0", "PIM1")
PARAMETERS = (
    "SCHED_JOINT_LK_ENABLE", "SCHED_JOINT_LK_H", "SCHED_JOINT_LK_GAMMA",
    "SCHED_JOINT_LK_CONSIST_LAMBDA", "SCHED_JOINT_LK_PLAN_HINT_MAX",
    "SCHED_WEIGHT_BIAS_ETA", "SCHED_DECODE_AMORT_ENABLE",
    "SCHED_DECODE_AMORT_ALPHA", "SCHED_DECODE_AMORT_RMIN",
    "SCHED_DECODE_AMORT_REUSE_PROB",
)


def _op_role(name, attrs):
    if attrs.get("expert") is not None:
        return "EXPERT"
    token = re.sub(r"[^A-Z0-9]+", "_", str(attrs.get("op_role") or name).upper()).strip("_")
    roles = {
        "LN": "LAYERNORM", "LN1": "LAYERNORM", "LN2": "LAYERNORM",
        "LAYERNORM": "LAYERNORM", "RMSNORM": "LAYERNORM",
        "WQ": "Q_PROJ", "Q": "Q_PROJ", "Q_PROJ": "Q_PROJ",
        "WK": "K_PROJ", "K": "K_PROJ", "K_PROJ": "K_PROJ",
        "WV": "V_PROJ", "V": "V_PROJ", "V_PROJ": "V_PROJ",
        "K_WRITE": "KV_WRITE", "V_WRITE": "KV_WRITE", "KV_WRITE": "KV_WRITE",
        "QK": "QK", "SCORE": "QK", "SOFTMAX": "SOFTMAX", "SV": "SV",
        "WO": "O_PROJ", "O": "O_PROJ", "O_PROJ": "O_PROJ",
        "FFN_W1": "FFN_UP", "FFN_W3": "FFN_UP", "FFN_UP": "FFN_UP",
        "FFN_GATE": "FFN_UP", "SWIGLU": "ACTIVATION", "SILU": "ACTIVATION",
        "GELU": "ACTIVATION", "ACTIVATION": "ACTIVATION",
        "FFN_W2": "FFN_DOWN", "FFN_DOWN": "FFN_DOWN",
        "MOE_ROUTER": "ROUTER", "ROUTER": "ROUTER",
        "MOE_COMBINE": "COMBINE", "COMBINE": "COMBINE",
        "ALLREDUCE": "COLLECTIVE", "ALL_REDUCE": "COLLECTIVE",
        "REDUCE": "COLLECTIVE", "GATHER": "COLLECTIVE", "SCATTER": "COLLECTIVE",
        "TRANSFER": "COPY", "COPY": "COPY",
    }
    return "EXPERT" if "EXPERT" in token else roles.get(token, "OTHER")


def _service(cost, node, device, batch, sequence, phase):
    label = PlanLabel(kv_in_pim=True, kv_place="pim")
    if node.weight_size:
        return float(cost.weighted_compute_stage(
            node, device, label, batch, sequence, phase,
            resident_weight_fmt=cost.weight_resident_format("ND", device)).total_s)
    return float(cost.node_device_cost(node, device, label, batch, sequence, phase))


def _expert_lut(cost, graph, phase, maximum):
    families = {node.name.lower(): node for node in graph.nodes.values()
                if node.attrs.get("expert_id") is not None}
    result = {}
    for family, original in families.items():
        node = deepcopy(original)
        node.attrs["moe_token_fraction"] = 1.0
        buckets = []
        for count in range(1, maximum + 1):
            batch, sequence = (1, count) if phase == "prefill" else (count, 1)
            buckets.append({
                "min_tokens": count, "max_tokens": count,
                "activation_bytes": count * int(node.attrs["dim"]) * 2,
                "service_time_s": {device: _service(cost, node, cost.cluster.devices[device],
                                                    batch, sequence, phase) for device in DEVICES},
                "timing_source": {device: "dops_fast" for device in DEVICES},
            })
        result[family] = buckets
    return result


def _order(operators):
    by_id = {op["op_id"]: op for op in operators}
    position = {op["op_id"]: index for index, op in enumerate(operators)}
    pending = {op_id: set(op["dependencies"]) for op_id, op in by_id.items()}
    successors = defaultdict(list)
    for op_id, dependencies in pending.items():
        for dependency in dependencies:
            successors[dependency].append(op_id)
    ready = [(by_id[op_id]["operator_index"], position[op_id], op_id)
             for op_id, dependencies in pending.items() if not dependencies]
    heapq.heapify(ready)
    result = []
    while ready:
        _, _, op_id = heapq.heappop(ready)
        result.append(op_id)
        for consumer in successors[op_id]:
            pending[consumer].remove(op_id)
            if not pending[consumer]:
                heapq.heappush(ready, (by_id[consumer]["operator_index"], position[consumer], consumer))
    if len(result) != len(operators):
        raise ValueError("DOPS graph is not a DAG")
    return result


def build_experiment_bundle(*, cfg, snapshots, graph, shape, cluster, cost):
    """Preserve DOPS costs and TP communication, projecting whole-expert placement."""
    if cfg.get("npu_backend", "fast") != "fast" or not cfg.get("pim_fast_mode", True):
        raise ValueError("Het-Infer export requires fast mode on NPU and PIM")
    is_moe = cfg["model_family"] == "mixtral"
    luts = {phase: _expert_lut(cost, graph, phase, maximum)
            for phase, maximum in (("prefill", int(cfg["batch"]) * int(cfg["prefill_len"])),
                                   ("decode", int(cfg["batch"])))} if is_moe else {}
    movements = [dict(route) for snapshot in snapshots for route in snapshot["routes"]]
    move_keys = {tuple(row[key] for key in ("tensor_id", "source_device_id",
                 "destination_device_id", "bytes", "layout")) for row in movements}

    def add_route(tensor, source, destination, size, layout, duration):
        key = (tensor, source, destination, size, layout)
        if key not in move_keys:
            move_keys.add(key)
            movements.append(dict(tensor_id=tensor, source_device_id=source,
                destination_device_id=destination, bytes=size, layout=layout, duration_s=duration))

    weights, networks = {}, []
    route_context = SimpleNamespace(cost=cost, cluster=cluster)
    for index, snapshot in enumerate(snapshots):
        phase = snapshot["phase"]
        first = snapshot["operators"][0]["network_metadata"]
        batch, sequence = int(first["batch"]), int(first["seq_len"])
        past, query = (0, sequence) if phase == "prefill" else (sequence, 1)
        inputs = defaultdict(list)
        for entry in snapshot["inputs"]:
            inputs[entry["consumer_op_id"]].append({k: v for k, v in entry.items() if k != "consumer_op_id"})
        collectives = {row["op_id"]: dict(row) for row in snapshot["collective_contexts"]}
        kv_homes = {}
        for raw in snapshot["operators"]:
            meta = raw["network_metadata"]
            attrs = meta["node_attrs"]
            slot = attrs["canonical_op_slot"]
            shard = int(slot.rsplit("_s", 1)[1]) if "_s" in slot else 0
            if _op_role(meta["name"], attrs) == "KV_WRITE":
                kv_homes[(attrs.get("layer_index", attrs.get("layer")), shard)] = raw["expert_device"]
        operators = []
        for position, raw in enumerate(snapshot["operators"]):
            op_id, meta = raw["op_id"], raw["network_metadata"]
            attrs = meta["node_attrs"]
            slot = attrs["canonical_op_slot"]
            layer = attrs.get("layer_index", attrs.get("layer"))
            shard = int(slot.rsplit("_s", 1)[1]) if "_s" in slot else 0
            role = _op_role(meta["name"], attrs)
            node = (SimpleNamespace(name=slot, attrs={}, weight_id=None, weight_size=0)
                    if role == "KV_WRITE" else graph.nodes[op_id.split(":", 2)[2]])
            family, expert = node.name.lower(), node.attrs.get("expert_id")
            if node.weight_size:
                wid, size = str(node.weight_id), int(node.weight_size)
                load = {}
                for device_id in raw["legal_devices"]:
                    device = cluster.devices[device_id]
                    comm = float(cost.comm_cost(cluster.devices["CPU0"], device, size))
                    local = (float(cost.pim_local_weight_load_cost(size, "ND", dev=device).total_s)
                             if device.type == "pim" else float(cost.npu_local_weight_load_cost(
                                 size, "ND", cost.weight_resident_format("ND", device), dev=device).total_s))
                    load[device_id] = {"transfer_s": comm, "local_write_format_s": local,
                                       "total_s": comm + local}
                    add_route(f"weight:{wid}", "CPU0", device_id, size, "ND", comm + local)
                weights[wid] = {"bytes": size, "load": load}
                inputs[op_id].append({"producer_op_id": None, "tensor_id": f"weight:{wid}",
                    "semantics": "data", "bytes": size,
                    "source_residencies": [{"device_id": "CPU0", "layout": "ND"}],
                    "destination_devices": list(raw["legal_devices"])})
            operators.append({
                "op_id": op_id, "dependencies": list(raw["dependencies"]), "op_role": role,
                "layer_index": layer, "operator_index": position, "operator_family": family,
                "legal_devices": sorted(raw["legal_devices"]), "default_device": raw["expert_device"],
                "native_device": raw["expert_device"],
                "placement_supernode": f"{phase}:{index}:L{layer}:expert:{expert}" if expert else op_id,
                "parallel_group_hint": f"{phase}:{index}:L{layer}:experts" if expert else None,
                "weight_home": "CPU0" if node.weight_size else None, "weight_id": node.weight_id,
                "kv_home": kv_homes[(layer, shard)] if slot.split("_s", 1)[0] in
                           {"k", "v", "qk", "sv", "k_write", "v_write"} else None,
                "kv_shard_index": shard, "expert_id": expert,
                "service_s": dict(raw["service_s"]), "inputs": inputs[op_id],
                "collective_context": collectives.get(op_id),
                "timing_source": node.attrs.get("timing_source", "dops_fast"),
            })
        # Preserve the existing input order: it also fixes transfer submission order.
        for op in operators:
            for entry in op["inputs"]:
                entry["source_residencies"] = sorted(entry["source_residencies"],
                    key=lambda row: (row["device_id"], row["layout"]))
                entry["destination_devices"] = sorted(entry["destination_devices"])
            op["inputs"].sort(key=lambda entry: (entry["producer_op_id"] or "", entry["tensor_id"],
                entry["semantics"], entry["bytes"],
                tuple((r["device_id"], r["layout"]) for r in entry["source_residencies"]),
                tuple(entry["destination_devices"])))
            if op["collective_context"] is not None:
                for field in ("participant_device_ids", "output_device_ids", "resource_device_ids"):
                    op["collective_context"][field] = sorted(op["collective_context"][field])
        groups = defaultdict(list)
        by_id = {op["op_id"]: op for op in operators}
        for op in operators:
            if op["expert_id"]:
                groups[op["placement_supernode"]].append(op)
        for members in groups.values():
            selected = min(members[0]["legal_devices"],
                           key=lambda device: sum(op["service_s"][device] for op in members))
            for op in members:
                op["default_device"] = selected
        if is_moe:
            for consumer in operators:
                for entry in consumer["inputs"]:
                    producer = by_id.get(entry["producer_op_id"])
                    if producer is None:
                        continue
                    expert_op = (consumer if producer["op_role"] == "ROUTER" and consumer["op_role"] == "EXPERT"
                                 else producer if producer["op_role"] == "EXPERT" and consumer["op_role"] == "COMBINE"
                                 else None)
                    if expert_op is None:
                        continue
                    for bucket in luts[phase][expert_op["operator_family"]]:
                        size = bucket["activation_bytes"]
                        for source in entry["source_residencies"]:
                            for destination in entry["destination_devices"]:
                                duration = route_time_s(route_context,
                                    cluster.devices[source["device_id"]], cluster.devices[destination],
                                    size, source_layout=source["layout"])
                                add_route(entry["tensor_id"], source["device_id"], destination,
                                          size, source["layout"], duration)
        reference = deepcopy(next(node for node in graph.nodes.values() if node.name.upper() == "FFN_W1"))
        reference.attrs["moe_token_fraction"] = 1.0
        flops = float(cost.estimate_flops(reference, batch, sequence, phase))
        capabilities = {domain: {
            "effective_compute_flops_per_s": sum(flops / _service(cost, reference, cluster.devices[d],
                                                  batch, sequence, phase) for d in devices),
            "effective_bandwidth_bytes_per_s": sum(float(cluster.devices[d].mem_bw_GBs) * 1e9 for d in devices),
            "queue_count": len(devices),
        } for domain, devices in (("NPU", (NPU,)), ("PIM", DEVICES[1:]))}
        order = _order(operators)
        networks.append({"phase": phase, "batch_size": batch, "sequence_length": sequence,
            "past_kv_len": past, "query_len": query, "layer_class": "moe" if is_moe else "dense",
            "router_top_k": 2 if is_moe else None, "shape_bucket": f"b{batch}-past{past}-q{query}",
            "default_order": order, "operators": [by_id[op_id] for op_id in order],
            "capability_basis": "compute", "domain_capabilities": capabilities})
    return {"graph_id": cfg["hetinfer_graph_id"], "workload_id": cfg["hetinfer_workload_id"],
        "device_domains": {NPU: "NPU", "PIM0": "PIM", "PIM1": "PIM"},
        "model": cfg["model_family"], "layer_count": int(shape.layer_num),
        "batch": cfg["batch"], "prefill": cfg["prefill_len"], "decode_rounds": int(cfg["decode_len"]),
        "weights": weights, "weight_capacity_bytes": {device: int(cluster.devices[device].mem_capacity_GB
                                             * 1024 ** 3 * .95) for device in DEVICES},
        "bifocal_parameters": {name: cfg.get(name, getattr(runtime_config, name)) for name in PARAMETERS},
        "scheduler_seed": cfg["scheduler_seed"], "expert_service_buckets": luts,
        "movements": movements, "networks": networks}


def export_experiment_bundle(*, output, **kwargs):
    payload = build_experiment_bundle(**kwargs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return path
