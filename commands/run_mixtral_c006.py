#!/usr/bin/env python3
"""Run one approved C006 workload; Slurm resources are set by the submitter."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(args, matrix):
    root = args.root
    exports = root / 'dops/output/mixtral_hotcold_c006/expanded_bpd'
    results = root / 'het-infer/results/mixtral_hotcold_c006/expanded_bpd'
    summarizer = root / 'het-infer/scripts/summarize_mixtral_hotcold_fast.py'
    if args.mode == 'verify':
        return results / 'validation', [
            (results / 'validation/dops.time.txt', root / 'dops',
             [sys.executable, '-m', 'pytest', '-q', 'tests/test_mixtral_export_resume.py', 'tests/test_mixtral_matrix.py']),
            (results / 'validation/hetinfer.time.txt', root / 'het-infer',
             [sys.executable, '-m', 'pytest', '-q'])]
    if args.mode == 'summarize':
        return results, [(results / 'summary.time.txt', root / 'het-infer',
            [sys.executable, str(summarizer), '--results', str(results), '--matrix', str(args.matrix)])]
    preflight = args.mode == 'preflight'
    workload = matrix['preflight_workloads' if preflight else 'main_workloads'][args.workload_index]
    batch, prefill, decode = (workload[k] for k in ('batch', 'prefill', 'decode'))
    name = f'b{batch}-p{prefill}-h{decode}'
    bundle = exports / name
    calls = []
    if args.mode in ('export', 'preflight'):
        calls.append((bundle / 'export.time.txt', root / 'dops',
            [sys.executable, str(root / 'dops/commands/export_mixtral_hotcold_fast.py'),
             '--batch', str(batch), '--prefill', str(prefill), '--horizon', str(decode),
             '--output', str(bundle)]))
    stage = 'preflight' if preflight else 'main'
    task = bundle if args.mode == 'export' else results / stage / name
    if args.mode in ('replay', 'preflight'):
        if preflight:
            points = [(share, matrix['seeds'][0]) for share in matrix['preflight_hot_shares']]
        else:
            share_index, seed_index = divmod(args.route_index, len(matrix['seeds']))
            points = [(matrix['main_hot_shares'][share_index], matrix['seeds'][seed_index])]
        for share, seed in points:
            output = results / stage / name / f'hot{share:g}' / f'seed{seed}'
            if not preflight:
                task = output
            calls.append((output / 'replay.time.txt', root / 'het-infer',
                [sys.executable, str(root / 'het-infer/scripts/run_mixtral_hotcold_fast.py'),
                 '--bundle', str(bundle / 'bundle.json'), '--output', str(output),
                 '--routes', str(results / ('preflight/routes' if preflight else 'routes')),
                 '--warmup', '0', '--resume', '--hot-share', str(share), '--seeds', str(seed),
                 '--policies', *(['R0', 'R2'] if preflight else matrix['policies']),
                 '--arms', *matrix['arms']]))
    if preflight:
        calls.append((task / 'validation.time.txt', root / 'het-infer',
            [sys.executable, str(summarizer), '--results', str(results), '--matrix', str(args.matrix),
             '--stage', 'preflight', '--workload-index', str(args.workload_index), '--validate-only']))
    return task, calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('export', 'replay', 'preflight', 'summarize', 'verify'))
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workload-index', type=int)
    parser.add_argument('--route-index', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workload_index is None and args.mode not in ('summarize', 'verify'):
        args.workload_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if args.mode == 'replay' and args.route_index is None:
        args.route_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    args.matrix = args.matrix.resolve()
    matrix = json.loads(args.matrix.read_text())
    task, calls = commands(args, matrix)
    if args.dry_run:
        print(json.dumps([{'cwd': str(cwd), 'command': command, 'time_log': str(timing)}
                          for timing, cwd, command in calls], indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Export, replay and validation require a Slurm compute allocation')
    started = datetime.now(timezone.utc).isoformat()
    for timing, cwd, command in calls:
        timing.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'PYTHONPATH': str(args.root / 'het-infer/src'),
               'MIXTRAL_MATRIX': str(args.matrix)}
        print('RUN ' + ' '.join(command), flush=True)
        subprocess.run(['/usr/bin/time', '-v', '-o', str(timing), *command], cwd=cwd, env=env, check=True)
    task.mkdir(parents=True, exist_ok=True)
    (task / 'task.json').write_text(json.dumps({
        'mode': args.mode, 'workload_index': args.workload_index, 'route_index': args.route_index,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
