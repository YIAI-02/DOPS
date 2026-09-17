#!/usr/bin/env python3
"""Export the approved FFN-half Mixtral experiment with effective DOPS parameters."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def _config(
    *,
    dops_root: Path,
    batch: int,
    prefix_length: int,
    workload: str,
) -> dict[str, object]:
    result_root = dops_root / "output" / "full_qwen_1p8b" / workload
    return {
        "experiment_label": f"hetinfer_full_qwen_1p8b_{workload}",
        "evidence_class": "DOPS analytical fast timing",
        "model_family": "qwen",
        "model_variant": "1.8b",
        "model_revision": "shape-only:qwen_1.8b_shape.json;layer_num=28",
        "hetinfer_graph_id": "qwen-1.8b-28layer-1npu2pim",
        "hetinfer_workload_id": workload,
        "shape_file": str(dops_root / "configs" / "qwen_1.8b_shape.json"),
        "dtype": "fp16",
        "batch": batch,
        "max_batch_size": batch,
        "prefill_len": prefix_length,
        "decode_len": 128,
        "max_seq_len": prefix_length + 128,
        "decode_sample_stride": 1,
        "decode_plan_refresh_stride": 0,
        "pim_config_path": str(dops_root / "src" / "aim_simulator" / "PIM_AiM.json"),
        "ramulator_config_path": str(dops_root / "src" / "aim_simulator" / "example.yaml"),
        "pim_ramulator_timeout_s": 300,
        "pim_trace_strict": False,
        "pim_trace_keep_traces": False,
        "result_dir": str(result_root),
        "simulation_log_file": str(result_root / "pim_simulation.log"),
        "hardware_json": str(dops_root / "src" / "examples" / "hardware_1npu_2aim.json"),
        "algo": ["Bifocal"],
        "baselines": [],
        "tp_qkv": 1,
        "tp_ffn": 1,
        "tp_moe": 1,
        "pp": 1,
        "ep": 1,
        "npu_backend": "fast",
        "npu_lut_strict": False,
        "pim_fast_mode": True,
        "scheduler_seed": 0,
        "dump_graph": False,
        "pim_weight_load_overlap_ratio": 0.0,
        "weight_load_compute_overlap_ratio": 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--prefill', type=int, default=128)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parameters', type=Path)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Export requires a Slurm compute allocation')
    args.output.mkdir(parents=True, exist_ok=True)
    workload = f'mixtral-ffn-half-b{args.batch}-p{args.prefill}-h{args.horizon}'
    cfg = _config(dops_root=ROOT,
                  batch=args.batch, prefix_length=args.prefill, workload=workload)
    params = json.loads((ROOT/'configs/mixtral_hotcold_selected_parameters.json').read_text())
    if args.parameters:
        params.update(json.loads(args.parameters.read_text()))
    cfg.update(params)
    cfg.update({'model_family': 'mixtral', 'model_variant': '8x7b-ffn-half',
        'shape_file': str(ROOT/'configs/mixtral_8x7b_shape.json'),
        'model_revision': 'shape-only:mixtral-ffn-half', 'intermediate_dim': 7168,
        'hetinfer_graph_id': 'mixtral-32layer-ffn-half-1npu2pim',
        'experiment_label': 'mixtral_hotcold_fast_20260911',
        'evidence_class': 'analytical fast; no AIM, NPU LUT, tensor or token inference',
        'npu_backend': 'fast', 'pim_fast_mode': True,
        'npu_lut_strict': False, 'pim_trace_strict': False,
        'tp': 2, 'tp_qkv': 2, 'tp_ffn': 2, 'tp_moe': 2,
        'decode_len': args.horizon, 'max_seq_len': args.prefill+args.horizon,
        'scheduler_seed': 7, 'weight_source': 'host', 'weight_format': 'ND',
        'weight_load_compute_overlap_ratio': 1., 'pim_weight_load_overlap_ratio': .5,
        'decode_sample_stride': 1, 'decode_plan_refresh_stride': 2,
        'result_dir': str(args.output/'native_run'),
        'simulation_log_file': str(args.output/'pim_simulation.log'),
        'hetinfer_bundle_out': str(args.output/'bundle.json')})
    path = args.output/'config.json'
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    bundle = args.output/'bundle.json'
    ready = False
    if bundle.exists():
        existing = json.loads(bundle.read_text())
        ready = (existing['bifocal_parameters'] == params and
                 (existing['batch'], existing['prefill'], existing['decode_rounds']) ==
                 (args.batch, args.prefill, args.horizon))
    if not ready:
        with (args.output/'evaluate.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'src/main.py'), 'evaluate', '--config', str(path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('MIXTRAL_FAST_EXPORT_OK '+str(bundle), flush=True)


if __name__ == '__main__':
    main()
