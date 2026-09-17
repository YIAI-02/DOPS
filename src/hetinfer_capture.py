"""Capture placement, service, and movement costs after a DOPS schedule."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from comm_primitives import normalize_topology, ring_allreduce
from hardware import DeviceSpec
from task_graph import TaskNode


def _legal_devices(
    scheduler, g: Any, nid: str, node: TaskNode
) -> Tuple[DeviceSpec, ...]:
    """Return the capability/legal candidate set without timing filters."""

    if scheduler._is_comm_node(node):
        # Communication primitives are not ordinary placement choices.
        # Their committed canonical output device is the scheduler's
        # concrete placement; REDUCE/GATHER/SCATTER may additionally leave
        # copies on devices recorded in _collective_output_devs.
        committed = scheduler._node_placement.get(str(nid))
        if committed in scheduler.cluster.devices:
            return (scheduler.cluster.devices[str(committed)],)
        return (scheduler.cost.get_host_device(),)
    name_up = str(getattr(node, "name", "") or "").upper()
    if name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
        pinned = scheduler._preferred_kv_write_device(g, nid)
        if pinned is not None and scheduler._node_allowed_on(node, pinned):
            return (pinned,)
        # Match Bifocal's long-standing fallback when the KV pin cannot be
        # resolved: fall through to the ordinary executor candidates.
    candidates: List[DeviceSpec] = []
    for device_type in scheduler._executor_device_types():
        for device in scheduler.cluster.devices_by_type(device_type):
            try:
                allowed = scheduler._node_allowed_on(node, device)
            except Exception:
                # Preserve the scheduler's existing conservative behavior:
                # an unavailable optional legality hint must not remove an
                # otherwise capability-compatible executor.
                allowed = True
            if allowed:
                candidates.append(device)
    return tuple(candidates)

def _kv_source_device(scheduler, node: TaskNode) -> DeviceSpec:
    kv_place = str(scheduler._kv_place(scheduler.label) or "host").lower()
    if kv_place == "pim":
        source = scheduler._kv_pim_for_node(node)
        if source is not None:
            return source
    elif kv_place == "npu":
        source = scheduler._kv_npu_device(scheduler.label)
        if source is not None:
            return source
    return scheduler.cost.get_host_device()

def _comm_devices(scheduler, g: Any, nid: str) -> Tuple[DeviceSpec, ...]:
    node = g.nodes[nid]
    prim = str(scheduler._comm_primitive(node) or "").upper()
    names: set[str] = set()
    for predecessor in g.predecessors(nid):
        placed = scheduler._node_placement.get(str(predecessor))
        if placed in scheduler.cluster.devices:
            names.add(str(placed))
    if prim == "SCATTER":
        names.update(scheduler._collective_output_devs.get(str(nid), set()))
    elif prim == "TRANSFER":
        attrs = getattr(node, "attrs", {}) or {}
        for key in ("src", "dst"):
            name = attrs.get(key)
            if name in scheduler.cluster.devices:
                names.add(str(name))
    committed = scheduler._node_placement.get(str(nid))
    if committed in scheduler.cluster.devices:
        names.add(str(committed))
    return tuple(
        scheduler.cluster.devices[name]
        for name in sorted(names)
        if name in scheduler.cluster.devices
    )

def _comm_service_s(
    scheduler, g: Any, nid: str, phase: str
) -> Optional[float]:
    """Return an availability-free communication primitive duration.

    The duration comes from the same DOPS topology and communication model
    used by scheduling, including collectives whose participants include
    PIM devices.
    """

    node = g.nodes[nid]
    devices = _comm_devices(scheduler, g, nid)
    phase_eff = scheduler._node_phase(g, nid, phase)
    batch = scheduler._node_batch(g, nid, phase_eff)
    seq_len = scheduler._node_seq_len(g, nid, phase_eff)
    read_bytes, write_bytes = scheduler.cost.estimate_activation_bytes(
        node, batch, seq_len, phase_eff
    )
    tensor_bytes = int(max(int(read_bytes), int(write_bytes), 0))
    prim = str(scheduler._comm_primitive(node) or "").upper()
    host = scheduler.cost.get_host_device()
    host_name = str(host.name)

    def transfer(source_name: str, destination_name: str, bytes_: int) -> float:
        if source_name == destination_name or int(bytes_) == 0:
            return 0.0
        source = scheduler.cluster.devices[str(source_name)]
        destination = scheduler.cluster.devices[str(destination_name)]
        value = float(scheduler.cost.comm_cost(source, destination, int(bytes_)))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(
                "no finite communication primitive for "
                f"{source_name!r}->{destination_name!r}"
            )
        return value

    participant_names = sorted(
        {
            str(scheduler._node_placement[pred])
            for pred in g.predecessors(nid)
            if scheduler._node_placement.get(pred) in scheduler.cluster.devices
        }
    )
    duration = 0.0
    if prim in ("ALLREDUCE", "ALL_REDUCE", "ALL-REDUCE"):
        topology = normalize_topology(getattr(scheduler.cluster, "topology", None))
        if topology == "fc":
            duration = float(
                ring_allreduce(
                    cost=scheduler.cost,
                    cluster=scheduler.cluster,
                    ring=participant_names,
                    tensor_bytes=tensor_bytes,
                    start=0.0,
                )
            )
        else:
            reduce_s = max(
                (
                    transfer(name, host_name, tensor_bytes)
                    for name in participant_names
                ),
                default=0.0,
            )
            try:
                dtype_bytes = float(scheduler.cost._act_dtype_bytes(node, phase_eff))
            except Exception:
                dtype_bytes = 2.0
            elements = float(tensor_bytes) / max(1.0, dtype_bytes)
            accumulation_s = float(
                scheduler.cost.flop_time(
                    max(0, len(participant_names) - 1) * elements, host
                )
            )
            scatter_s = max(
                (
                    transfer(host_name, name, tensor_bytes)
                    for name in participant_names
                ),
                default=0.0,
            )
            duration = reduce_s + accumulation_s + scatter_s
    elif prim in ("REDUCE", "GATHER"):
        duration = max(
            (
                transfer(name, host_name, tensor_bytes)
                for name in participant_names
            ),
            default=0.0,
        )
        if prim == "REDUCE":
            try:
                dtype_bytes = float(scheduler.cost._act_dtype_bytes(node, phase_eff))
            except Exception:
                dtype_bytes = 2.0
            elements = float(tensor_bytes) / max(1.0, dtype_bytes)
            duration += float(
                scheduler.cost.flop_time(
                    max(0, len(participant_names) - 1) * elements, host
                )
            )
    elif prim == "SCATTER":
        attrs = getattr(node, "attrs", {}) or {}
        targets = sorted(
            set(scheduler._collective_output_devs.get(str(nid), set()))
            - {host_name}
        )
        per_target = tensor_bytes
        if str(attrs.get("scatter_mode", "broadcast")).lower() in (
            "partition",
            "shard",
            "split",
        ):
            per_target = int(
                math.ceil(float(tensor_bytes) / float(max(1, len(targets))))
            )
        duration = max(
            (transfer(host_name, name, per_target) for name in targets),
            default=0.0,
        )
    elif prim == "TRANSFER":
        attrs = getattr(node, "attrs", {}) or {}
        source_name = str(attrs.get("src") or (participant_names or [host_name])[0])
        destination_name = str(attrs.get("dst") or host_name)
        override = attrs.get("bytes", attrs.get("bytes_nd", tensor_bytes))
        duration = transfer(source_name, destination_name, int(override))
    if not math.isfinite(duration) or duration < 0.0:
        raise RuntimeError(f"invalid communication service for {nid!r}: {duration!r}")
    return float(duration)

def compute_service_s(
    scheduler,
    g: Any,
    nid: str,
    device: DeviceSpec,
    phase: str,
) -> Optional[float]:
    """Return movement-, reload-, and queue-free local execution seconds.

    NPU and PIM values come from the CostModel selected for this DOPS run.
    """

    node = g.nodes[nid]
    if scheduler._is_comm_node(node):
        return _comm_service_s(scheduler, g, nid, phase)
    phase_eff = scheduler._node_phase(g, nid, phase)
    batch = scheduler._node_batch(g, nid, phase_eff)
    seq_len = scheduler._node_seq_len(g, nid, phase_eff)
    name_up = str(getattr(node, "name", "") or "").upper()
    if name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
        _, write_bytes = scheduler.cost.estimate_activation_bytes(
            node, batch, seq_len, phase_eff
        )
        bytes_nd = int(scheduler.cost.format_size(int(write_bytes), "ND"))
        if str(device.name) == str(scheduler.cost.get_host_device().name):
            # The existing DOPS host-KV path models the predecessor-to-host
            # copy as movement and commits a zero-duration host-local
            # primitive.  Preserve that exact separation in the static
            # tables; adding host mem_time here would double count it.
            return 0.0
        if str(getattr(device, "type", "") or "").lower() == "pim":
            return float(scheduler.cost.pim_write_time(bytes_nd, device))
        source_layout = str(scheduler.cost.device_preferred_fmt(device))
        conversion_s = 0.0
        if source_layout != "ND":
            conversion_s = float(
                scheduler.cost.format_conversion_time(
                    bytes_nd, source_layout, "ND", device
                )
            )
        return conversion_s + float(scheduler.cost.mem_time(bytes_nd, device))
    result = float(
        scheduler._weighted_compute_time(
            node,
            device,
            scheduler.label,
            int(batch),
            int(seq_len),
            str(phase_eff),
        )
    )
    if name_up in ("QK", "SV"):
        source = _kv_source_device(scheduler, node)
        if str(source.name) == str(device.name):
            kv_bytes = int(
                scheduler.cost.estimate_kv_cache_read_bytes(
                    node, batch, seq_len, phase_eff
                )
            )
            if kv_bytes > 0:
                size_nd = int(scheduler.cost.format_size(kv_bytes, "ND"))
                result += float(scheduler.cost.mem_time(size_nd, device))
                destination_layout = str(
                    scheduler.cost.device_preferred_fmt(device)
                )
                if destination_layout != "ND":
                    result += float(
                        scheduler.cost.format_conversion_time(
                            size_nd,
                            "ND",
                            destination_layout,
                            device,
                        )
                    )
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(
            f"invalid local service for {(nid, device.name)!r}: {result!r}"
        )
    return result

def _edge_data_bytes(
    scheduler, g: Any, source_id: str, destination_id: str, phase: str
) -> int:
    edge_tensor = getattr(g, "edge_tensor", None)
    if callable(edge_tensor):
        metadata = edge_tensor(source_id, destination_id)
        if isinstance(metadata, Mapping) and metadata.get("bytes") is not None:
            return max(0, int(metadata["bytes"]))
    source = g.nodes[source_id]
    source_phase = scheduler._node_phase(g, source_id, phase)
    source_batch = scheduler._node_batch(g, source_id, source_phase)
    source_seq = scheduler._node_seq_len(g, source_id, source_phase)
    _, source_write = scheduler.cost.estimate_activation_bytes(
        source, source_batch, source_seq, source_phase
    )
    return max(0, int(source_write))

def _edge_tensor_id(
    scheduler,
    g: Any,
    source_id: str,
    destination_id: str,
    phase: str,
    schedule_call_index: int,
) -> str:
    edge_tensor = getattr(g, "edge_tensor", None)
    if callable(edge_tensor):
        metadata = edge_tensor(source_id, destination_id)
        if isinstance(metadata, Mapping) and metadata.get("tensor_id"):
            return (
                f"{phase}:{int(schedule_call_index)}:"
                f"{str(metadata['tensor_id'])}"
            )
    attrs = getattr(g.nodes[source_id], "attrs", {}) or {}
    output = attrs.get("hetinfer_output")
    if isinstance(output, Mapping) and output.get("tensor_id"):
        return (
            f"{phase}:{int(schedule_call_index)}:"
            f"{str(output['tensor_id'])}"
        )
    if attrs.get("hetinfer_tensor_id"):
        return (
            f"{phase}:{int(schedule_call_index)}:"
            f"{str(attrs['hetinfer_tensor_id'])}"
        )
    # Tensor identity belongs to the producer output, not an edge.  Forked
    # consumers therefore share one tensor and one residency domain.
    return f"{phase}:{int(schedule_call_index)}:tensor:{source_id}"

def _output_layout(
    scheduler,
    g: Any,
    source_id: str,
    source: DeviceSpec,
    destination_id: Optional[str] = None,
) -> str:
    if str(source.name) == str(scheduler.cost.get_host_device().name):
        return "ND"
    edge_tensor = getattr(g, "edge_tensor", None)
    if destination_id is not None and callable(edge_tensor):
        metadata = edge_tensor(source_id, destination_id)
        if isinstance(metadata, Mapping) and metadata.get("layout"):
            return str(metadata["layout"])
    node = g.nodes[source_id]
    attrs = getattr(node, "attrs", {}) or {}
    output = attrs.get("hetinfer_output")
    if isinstance(output, Mapping) and output.get("layout"):
        return str(output["layout"])
    if scheduler._is_comm_node(node) or str(getattr(node, "name", "")).upper() in (
        "K_WRITE",
        "V_WRITE",
        "KV_WRITE",
    ):
        return "ND"
    return str(scheduler.cost.device_preferred_fmt(source))

def route_time_s(
    scheduler,
    source: DeviceSpec,
    destination: DeviceSpec,
    bytes_nd: int,
    *,
    source_layout: str,
    include_source_read: bool = False,
) -> float:
    if source.name == destination.name:
        return 0.0
    size_nd = int(scheduler.cost.format_size(int(bytes_nd), "ND"))
    source_size = int(scheduler.cost.format_size(int(bytes_nd), source_layout))
    source_conversion = 0.0
    if source_layout != "ND":
        source_conversion = float(
            scheduler.cost.format_conversion_time(
                source_size, source_layout, "ND", source
            )
        )
    source_read = 0.0
    if str(getattr(source, "type", "") or "").lower() == "pim":
        source_read = float(scheduler.cost.activation_read_time_pim(size_nd))
    elif include_source_read:
        source_read = float(scheduler.cost.mem_time(size_nd, source))
    base = source_conversion + source_read
    destination_layout = str(scheduler.cost.device_preferred_fmt(destination))
    direct_probe = float(scheduler.cost.comm_cost(source, destination, size_nd))
    direct = float("inf")
    if math.isfinite(direct_probe):
        direct = base + float(
            scheduler.cost.combine_transfer_and_convert(
                source,
                destination,
                size_nd,
                "ND",
                destination_layout,
            )
        )
    host = scheduler.cost.get_host_device()
    via_host = float("inf")
    to_host = float(scheduler.cost.comm_cost(source, host, size_nd))
    if math.isfinite(to_host):
        via_host = base + to_host + float(
            scheduler.cost.combine_transfer_and_convert(
                host,
                destination,
                size_nd,
                "ND",
                destination_layout,
            )
        )
    topology = normalize_topology(getattr(scheduler.cluster, "topology", None))
    # The execution path uses the direct FC route when it exists, but it
    # still falls back through the host when that particular pair has no
    # finite direct link.  Keep the offline primitive identical.
    if topology == "fc" and math.isfinite(direct):
        result = direct
    else:
        result = min(direct, via_host)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(
            "no finite movement route for "
            f"{source.name!r}->{destination.name!r}"
        )
    return float(result)

def capture_snapshot(
    scheduler,
    g: Any,
    phase: str,
    *,
    schedule_call_index: int,
    scheduled_order: tuple[str, ...] | None = None,
) -> None:
    """Publish one complete clean snapshot after all placements commit."""

    if not scheduler._hetinfer_prior_capture_enabled:
        return
    graph_nodes = (tuple(scheduled_order) if scheduled_order is not None else
                   tuple(str(nid) for nid in scheduler._get_graph_index(g).nodes))
    if set(scheduler._node_placement) != set(graph_nodes):
        raise RuntimeError(
            "cannot export an incomplete final placement: "
            f"missing={sorted(set(graph_nodes) - set(scheduler._node_placement))}, "
            f"unexpected={sorted(set(scheduler._node_placement) - set(graph_nodes))}"
        )

    devices = [
        {
            "device_id": str(device.name),
            "device_type": str(device.type).lower(),
        }
        for device in sorted(
            scheduler.cluster.devices.values(), key=lambda item: str(item.name)
        )
    ]
    host = scheduler.cost.get_host_device()
    host_name = str(host.name)
    topology = str(normalize_topology(getattr(scheduler.cluster, "topology", None)))
    barrier_edges = {
        (str(source), str(destination))
        for source, destination in (getattr(g, "barrier_edges", ()) or ())
    }
    op_id_by_node = {
        nid: f"{phase}:{int(schedule_call_index)}:{nid}"
        for nid in graph_nodes
    }
    legal_by_node: Dict[str, Tuple[DeviceSpec, ...]] = {}
    for nid in graph_nodes:
        node = g.nodes[nid]
        legal = _legal_devices(scheduler, g, nid, node)
        if not legal:
            raise RuntimeError(f"operator {nid!r} has no legal export devices")
        legal_by_node[nid] = legal
        expert = str(scheduler._node_placement[nid])
        if expert not in {str(device.name) for device in legal}:
            raise RuntimeError(
                f"final placement {(nid, expert)!r} is outside legal candidates"
            )

    # Communication nodes expose one fixed, physical DOPS expert context.
    # Inputs may still originate from every legal upstream residency, but
    # they first stage to these fixed devices.  T_service starts only after
    # staging and atomically includes the primitive's internal transport.
    collective_context_by_node: Dict[str, Dict[str, Any]] = {}
    collective_staging_by_edge: Dict[Tuple[str, str], str] = {}
    for nid in graph_nodes:
        node = g.nodes[nid]
        if not scheduler._is_comm_node(node):
            continue
        primitive = str(scheduler._comm_primitive(node) or "").upper()
        if primitive in ("ALL_REDUCE", "ALL-REDUCE"):
            primitive = "ALLREDUCE"
        data_predecessors = [
            str(pred)
            for pred in g.predecessors(nid)
            if (str(pred), nid) not in barrier_edges
        ]
        if not data_predecessors:
            raise RuntimeError(
                f"collective {nid!r} has no data input to bind to fixed staging"
            )
        attrs = getattr(node, "attrs", {}) or {}
        if primitive == "SCATTER":
            staging_devices = {pred: host_name for pred in data_predecessors}
        elif primitive == "TRANSFER":
            if len(data_predecessors) != 1:
                raise RuntimeError(
                    "TRANSFER export requires exactly one data predecessor: "
                    f"{nid!r} has {data_predecessors!r}"
                )
            effective_source = str(
                attrs.get("src")
                or scheduler._node_placement.get(data_predecessors[0], host_name)
            )
            if effective_source not in scheduler.cluster.devices:
                raise RuntimeError(
                    f"TRANSFER {nid!r} references unknown source {effective_source!r}"
                )
            staging_devices = {
                data_predecessors[0]: effective_source
            }
        else:
            staging_devices = {
                pred: str(scheduler._node_placement[pred])
                for pred in data_predecessors
            }
        participants = sorted(set(staging_devices.values()))
        if any(name not in scheduler.cluster.devices for name in participants):
            raise RuntimeError(
                f"collective {nid!r} has an unknown participant: {participants!r}"
            )
        recorded_outputs = {
            str(name)
            for name in scheduler._collective_output_devs.get(nid, set())
        }
        if primitive == "ALLREDUCE":
            outputs = recorded_outputs or set(participants)
        elif primitive in ("REDUCE", "GATHER"):
            outputs = recorded_outputs or {host_name}
        elif primitive == "SCATTER":
            outputs = recorded_outputs or {host_name}
        elif primitive == "TRANSFER":
            destination = str(
                attrs.get("dst") or scheduler._node_placement.get(nid, host_name)
            )
            outputs = recorded_outputs or {destination}
        else:
            raise RuntimeError(
                f"unsupported communication primitive for export: {primitive!r}"
            )
        canonical = str(scheduler._node_placement[nid])
        if canonical not in outputs:
            raise RuntimeError(
                f"collective {nid!r} canonical device {canonical!r} is not an output"
            )
        unknown_outputs = sorted(outputs - set(scheduler.cluster.devices))
        if unknown_outputs:
            raise RuntimeError(
                f"collective {nid!r} has unknown outputs: {unknown_outputs!r}"
            )
        node_phase = scheduler._node_phase(g, nid, phase)
        node_batch = scheduler._node_batch(g, nid, node_phase)
        node_seq = scheduler._node_seq_len(g, nid, node_phase)
        read_bytes, write_bytes = scheduler.cost.estimate_activation_bytes(
            node, node_batch, node_seq, node_phase
        )
        resources = set(participants) | set(outputs)
        if primitive == "ALLREDUCE" and topology != "fc":
            resources.add(host_name)
        context = {
            "op_id": op_id_by_node[nid],
            "primitive": primitive,
            "topology": topology,
            "canonical_device_id": canonical,
            "participant_device_ids": participants,
            "output_device_ids": sorted(outputs),
            "resource_device_ids": sorted(resources),
            "tensor_bytes": max(0, int(read_bytes), int(write_bytes)),
            "internal_transport": "included_in_t_service",
        }
        collective_context_by_node[nid] = context
        for pred, staging_device in staging_devices.items():
            collective_staging_by_edge[(pred, nid)] = staging_device

    routes_by_key: Dict[Tuple[str, str, str, int, str], Dict[str, Any]] = {}
    tensor_source_descriptions: Dict[Tuple[str, str], Tuple[int, str]] = {}
    inputs: List[Dict[str, Any]] = []
    input_keys: set[Tuple[str, Optional[str], str]] = set()
    tensor_bindings: Dict[str, Tuple[Optional[str], int]] = {}

    def add_input(
        *,
        consumer_id: str,
        producer_id: Optional[str],
        tensor_id: str,
        semantics: str,
        bytes_nd: int,
        source_residencies: List[Dict[str, str]],
        destination_devices: List[str],
    ) -> None:
        consumer_op_id = op_id_by_node[consumer_id]
        producer_op_id = (
            None if producer_id is None else op_id_by_node[producer_id]
        )
        key = (consumer_op_id, producer_op_id, str(tensor_id))
        if key in input_keys:
            raise RuntimeError(f"duplicate static-prior input binding: {key!r}")
        input_keys.add(key)
        binding = (producer_op_id, int(bytes_nd))
        previous = tensor_bindings.setdefault(str(tensor_id), binding)
        if previous != binding:
            raise RuntimeError(
                "one tensor_id cannot change producer or bytes across inputs: "
                f"{tensor_id!r}, previous={previous!r}, new={binding!r}"
            )
        residency_devices = [
            str(entry["device_id"]) for entry in source_residencies
        ]
        if len(residency_devices) != len(set(residency_devices)):
            raise RuntimeError(
                f"input {key!r} has duplicate source residency devices"
            )
        inputs.append(
            {
                "consumer_op_id": consumer_op_id,
                "producer_op_id": producer_op_id,
                "tensor_id": str(tensor_id),
                "semantics": str(semantics),
                "bytes": int(bytes_nd),
                "source_residencies": copy.deepcopy(source_residencies),
                "destination_devices": sorted(
                    {str(name) for name in destination_devices}
                ),
            }
        )

    def add_route(
        *,
        tensor_id: str,
        source: DeviceSpec,
        destination: DeviceSpec,
        bytes_nd: int,
        layout: str,
        include_source_read: bool = False,
    ) -> None:
        source_description_key = (str(tensor_id), str(source.name))
        source_description = (int(bytes_nd), str(layout))
        previous_description = tensor_source_descriptions.get(
            source_description_key
        )
        if (
            previous_description is not None
            and previous_description != source_description
        ):
            raise RuntimeError(
                "one tensor residency cannot change bytes/layout across "
                f"consumers: {source_description_key!r}, "
                f"previous={previous_description!r}, "
                f"new={source_description!r}"
            )
        tensor_source_descriptions[source_description_key] = source_description
        resident = str(source.name) == str(destination.name)
        if resident:
            duration_s = 0.0
        else:
            duration_s = route_time_s(scheduler, source,
                destination,
                int(bytes_nd),
                source_layout=str(layout),
                include_source_read=bool(include_source_read),
            )
        key = (
            str(tensor_id),
            str(source.name),
            str(destination.name),
            int(bytes_nd),
            str(layout),
        )
        entry = {
            "tensor_id": key[0],
            "source_device_id": key[1],
            "destination_device_id": key[2],
            "bytes": key[3],
            "layout": key[4],
            "duration_s": duration_s,
        }
        previous = routes_by_key.get(key)
        if previous is not None and previous != entry:
            raise RuntimeError(f"inconsistent duplicate route {key!r}")
        routes_by_key[key] = entry

    # Root operators consume explicit external inputs when declared.  The
    # compatibility fallback is one ND tensor resident on the host.
    for destination_id in graph_nodes:
        data_predecessors = tuple(
            str(item)
            for item in g.predecessors(destination_id)
            if (str(item), destination_id) not in barrier_edges
        )
        if not data_predecessors:
            raw_inputs: Any = None
            external_inputs_for = getattr(g, "external_inputs_for", None)
            if callable(external_inputs_for):
                raw_inputs = external_inputs_for(destination_id)
            if raw_inputs in (None, (), []):
                attrs = getattr(g.nodes[destination_id], "attrs", {}) or {}
                raw_inputs = attrs.get("hetinfer_external_inputs")
            if raw_inputs in (None, (), []):
                destination_node = g.nodes[destination_id]
                destination_phase = scheduler._node_phase(
                    g, destination_id, phase
                )
                destination_batch = scheduler._node_batch(
                    g, destination_id, destination_phase
                )
                destination_seq = scheduler._node_seq_len(
                    g, destination_id, destination_phase
                )
                read_bytes, _ = scheduler.cost.estimate_activation_bytes(
                    destination_node,
                    destination_batch,
                    destination_seq,
                    destination_phase,
                )
                raw_inputs = [
                    {
                        "tensor_id": f"input:{destination_id}",
                        "source_devices": [str(host.name)],
                        "bytes": max(0, int(read_bytes)),
                        "layout": "ND",
                    }
                ]
            if isinstance(raw_inputs, Mapping) or isinstance(
                raw_inputs, (str, bytes)
            ):
                raise RuntimeError(
                    f"external inputs for {destination_id!r} must be an array"
                )
            for input_index, raw_input in enumerate(raw_inputs):
                if not isinstance(raw_input, Mapping):
                    raise RuntimeError(
                        "external input must be an object: "
                        f"{destination_id!r}[{input_index}]"
                    )
                raw_tensor_id = str(
                    raw_input.get("tensor_id")
                    or f"input:{destination_id}:{input_index}"
                )
                tensor_id = (
                    f"{phase}:{int(schedule_call_index)}:{raw_tensor_id}"
                )
                source_names = raw_input.get(
                    "source_devices", [str(host.name)]
                )
                if isinstance(source_names, (str, bytes)):
                    source_names = [str(source_names)]
                try:
                    source_names = [str(item) for item in source_names]
                except TypeError as exc:
                    raise RuntimeError(
                        "external input source_devices must be an array: "
                        f"{destination_id!r}[{input_index}]"
                    ) from exc
                if not source_names:
                    raise RuntimeError(
                        "external input source_devices cannot be empty: "
                        f"{destination_id!r}[{input_index}]"
                    )
                bytes_nd = int(raw_input.get("bytes", 0))
                if bytes_nd < 0:
                    raise RuntimeError(
                        "external input bytes cannot be negative: "
                        f"{destination_id!r}[{input_index}]"
                    )
                layout = str(raw_input.get("layout", "ND") or "ND")
                source_residencies: List[Dict[str, str]] = []
                destination_names = [
                    str(device.name) for device in legal_by_node[destination_id]
                ]
                for source_name in source_names:
                    if str(source_name) not in scheduler.cluster.devices:
                        raise RuntimeError(
                            "external input references unknown source device "
                            f"{source_name!r}"
                        )
                    source = scheduler.cluster.devices[str(source_name)]
                    source_residencies.append(
                        {"device_id": str(source.name), "layout": layout}
                    )
                    for destination in legal_by_node[destination_id]:
                        add_route(
                            tensor_id=tensor_id,
                            source=source,
                            destination=destination,
                            bytes_nd=bytes_nd,
                            layout=layout,
                        )
                add_input(
                    consumer_id=destination_id,
                    producer_id=None,
                    tensor_id=tensor_id,
                    semantics="data",
                    bytes_nd=bytes_nd,
                    source_residencies=source_residencies,
                    destination_devices=destination_names,
                )

        # QK/SV have an additional cache tensor whose residency is fixed by
        # the DOPS KV plan.  It is independent of ordinary graph edges.
        node = g.nodes[destination_id]
        role = str(getattr(node, "name", "") or "").upper()
        if role in ("QK", "SV"):
            destination_phase = scheduler._node_phase(g, destination_id, phase)
            destination_batch = scheduler._node_batch(
                g, destination_id, destination_phase
            )
            destination_seq = scheduler._node_seq_len(
                g, destination_id, destination_phase
            )
            kv_bytes = max(
                0,
                int(
                    scheduler.cost.estimate_kv_cache_read_bytes(
                        node,
                        destination_batch,
                        destination_seq,
                        destination_phase,
                    )
                ),
            )
            if kv_bytes > 0:
                source = _kv_source_device(scheduler, node)
                tensor_id = (
                    f"{phase}:{int(schedule_call_index)}:"
                    f"kv:{'K' if role == 'QK' else 'V'}:{destination_id}"
                )
                for destination in legal_by_node[destination_id]:
                    add_route(
                        tensor_id=tensor_id,
                        source=source,
                        destination=destination,
                        bytes_nd=kv_bytes,
                        layout="ND",
                        include_source_read=True,
                    )
                add_input(
                    consumer_id=destination_id,
                    producer_id=None,
                    tensor_id=tensor_id,
                    semantics="data",
                    bytes_nd=kv_bytes,
                    source_residencies=[
                        {"device_id": str(source.name), "layout": "ND"}
                    ],
                    destination_devices=[
                        str(device.name)
                        for device in legal_by_node[destination_id]
                    ],
                )

    # Each producer output is one tensor even when it fans out.  Its source
    # domain is every legal producer placement (or every fixed collective
    # output) plus a possible host spill.  Ordinary consumers target all of
    # their legal devices; collective inputs target exactly one fixed
    # staging device.  Internal collective hops never appear as T_move.
    for source_id in graph_nodes:
        successors = tuple(str(item) for item in g.successors(source_id))
        if not successors:
            continue
        source_context = collective_context_by_node.get(source_id)
        if source_context is None:
            source_devices: Dict[str, DeviceSpec] = {
                str(device.name): device
                for device in legal_by_node[source_id]
            }
        else:
            source_devices = {
                str(name): scheduler.cluster.devices[str(name)]
                for name in source_context["output_device_ids"]
            }
        # Ordinary DOPS execution may spill an activation to host.  Host is
        # therefore a legal source residency even when it is not an
        # operator execution candidate.
        source_devices[host_name] = host
        for destination_id in successors:
            if (source_id, destination_id) in barrier_edges:
                add_input(
                    consumer_id=destination_id,
                    producer_id=source_id,
                    tensor_id=(
                        f"{phase}:{int(schedule_call_index)}:"
                        f"barrier:{source_id}->{destination_id}"
                    ),
                    semantics="barrier",
                    bytes_nd=0,
                    source_residencies=[],
                    destination_devices=[],
                )
                continue
            tensor_id = _edge_tensor_id(scheduler, g,
                source_id,
                destination_id,
                phase,
                schedule_call_index,
            )
            bytes_nd = _edge_data_bytes(scheduler, g, source_id, destination_id, phase
            )
            staging_device = collective_staging_by_edge.get(
                (source_id, destination_id)
            )
            if staging_device is None:
                semantics = "data"
                consumer_destinations = {
                    str(device.name): device
                    for device in legal_by_node[destination_id]
                }
            else:
                semantics = "collective_staging"
                consumer_destinations = {
                    str(staging_device): scheduler.cluster.devices[str(staging_device)]
                }
            # DOPS may materialize an activation in host memory before a
            # later consumer reloads it.  Export both halves of that
            # residency transition: producer->host store and
            # host->consumer reload (including host->host resident zero).
            route_destinations = dict(consumer_destinations)
            route_destinations[host_name] = host
            source_residencies: List[Dict[str, str]] = []
            for source in source_devices.values():
                layout = _output_layout(scheduler, g, source_id, source, destination_id
                )
                source_residencies.append(
                    {"device_id": str(source.name), "layout": str(layout)}
                )
                for destination in route_destinations.values():
                    add_route(
                        tensor_id=tensor_id,
                        source=source,
                        destination=destination,
                        bytes_nd=bytes_nd,
                        layout=layout,
                    )
            add_input(
                consumer_id=destination_id,
                producer_id=source_id,
                tensor_id=tensor_id,
                semantics=semantics,
                bytes_nd=bytes_nd,
                source_residencies=source_residencies,
                destination_devices=list(consumer_destinations),
            )

    routes = [routes_by_key[key] for key in sorted(routes_by_key)]

    operators: List[Dict[str, Any]] = []
    for nid in graph_nodes:
        node = g.nodes[nid]
        legal = legal_by_node[nid]
        node_phase = scheduler._node_phase(g, nid, phase)
        node_batch = scheduler._node_batch(g, nid, node_phase)
        node_seq = scheduler._node_seq_len(g, nid, node_phase)
        operators.append(
            {
                "op_id": op_id_by_node[nid],
                "dependencies": [
                    op_id_by_node[str(pred)] for pred in g.predecessors(nid)
                ],
                "legal_devices": [str(device.name) for device in legal],
                "expert_device": str(scheduler._node_placement[nid]),
                "service_s": {
                    str(device.name): compute_service_s(scheduler, g, nid, device, phase
                    )
                    for device in legal
                },
                "network_metadata": {
                    "name": str(getattr(node, "name", "") or "UNKNOWN"),
                    "phase": str(node_phase),
                    "batch": int(node_batch),
                    "seq_len": int(node_seq),
                    "node_attrs": dict(getattr(node, "attrs", {}) or {}),
                },
            }
        )

    snapshot = {
        "schedule_call_index": int(schedule_call_index),
        "phase": str(phase),
        "devices": devices,
        "operators": operators,
        "inputs": sorted(
            inputs,
            key=lambda entry: (
                entry["consumer_op_id"],
                entry["producer_op_id"] or "",
                entry["tensor_id"],
                entry["semantics"],
            ),
        ),
        "collective_contexts": [
            copy.deepcopy(collective_context_by_node[nid])
            for nid in graph_nodes
            if nid in collective_context_by_node
        ],
        "routes": routes,
    }
    scheduler._hetinfer_prior_snapshots.append(copy.deepcopy(snapshot))
    scheduler._hetinfer_schedule_call_index = int(schedule_call_index)

