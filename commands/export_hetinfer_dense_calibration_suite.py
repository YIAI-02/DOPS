#!/usr/bin/env python3
"""Export Qwen Dense bundles with analytical fast timing and Bifocal C026."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "commands"))
from export_hetinfer_full_suite import _config

CASES = {
    "W1": ("1.8b", 1, 16, 32),
    "W2": ("1.8b", 4, 16, 32),
    "W3": ("1.8b", 1, 256, 32),
    "W4": ("1.8b", 1, 16, 128),
    "W5": ("7b", 1, 16, 32),
    "W6": ("7b", 4, 256, 32),
    "W7": ("7b", 4, 256, 24),
    "W8": ("14b", 1, 16, 32),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a Slurm compute allocation")
    variant, batch, prefill, horizon = CASES[args.case]
    output = ROOT / "output" / "dense_fast_c026" / args.case
    output.mkdir(parents=True, exist_ok=True)
    shape_file = ROOT / "configs" / f"qwen_{variant}_shape.json"
    layers = json.loads(shape_file.read_text())["layer_num"]
    cfg = _config(dops_root=ROOT, het_infer_root=ROOT.parent / "v1-cama-work",
                  batch=batch, prefix_length=prefill,
                  workload=f"{args.case}-qwen{variant}-b{batch}-p{prefill}-h{horizon}")
    cfg.update({
        "model_variant": variant,
        "model_revision": f"shape-only:{shape_file.name}",
        "shape_file": str(shape_file),
        "hetinfer_graph_id": f"qwen-{variant}-{layers}layer-1npu2pim",
        "experiment_label": f"dense_fast_c026_{args.case}",
        "evidence_class": "DOPS analytical fast timing; no AIM or NPU LUT",
        "bifocal_preset": "C026",
        "decode_len": horizon, "max_seq_len": prefill + horizon,
        "result_dir": str(output / "native_run"),
        "simulation_log_file": str(output / "pim_simulation.log"),
        "hetinfer_prior_out": str(output / "native" / "prior.json"),
        "hetinfer_network_out": str(output / "native" / "network.json"),
        "hetinfer_tensor_bindings_out": str(output / "native" / "tensor_bindings.json"),
        "npu_backend": "fast", "pim_fast_mode": True,
        "npu_lut_strict": False, "pim_trace_strict": False,
        "scheduler_seed": 7, "tp_qkv": 2, "tp_ffn": 2,
        "weight_source": "host", "weight_format": "ND",
        "weight_load_compute_overlap_ratio": 1.0,
        "pim_weight_load_overlap_ratio": 0.5,
        "decode_sample_stride": 2, "decode_plan_refresh_stride": 2,
    })
    config = output / "config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "native").mkdir(exist_ok=True)
    with (output / f"native_export.{os.environ['SLURM_JOB_ID']}.log").open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "src" / "main.py"),
            "evaluate", "--config", str(config)], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    networks = json.loads(Path(cfg["hetinfer_network_out"]).read_text())["networks"]
    if len(networks) != horizon + 1:
        raise ValueError("Native export is incomplete")
    subprocess.run([sys.executable, str(ROOT / "src" / "hetinfer_experiment_export.py"),
                    "--config", str(config)], cwd=ROOT, check=True)
    (output / "bundle_source.json").write_text(json.dumps({
        "case": args.case, "variant": variant, "bundle": str(output / "bundle"),
        "reused": False, "config": str(config)}, indent=2) + "\n")
    print(f"DENSE_FAST_BUNDLE_OK {args.case} {output / 'bundle'}", flush=True)


if __name__ == "__main__":
    main()
