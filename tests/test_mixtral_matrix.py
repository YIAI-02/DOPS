"""The approved matrix must map to disjoint tasks without changing its scope."""
import argparse
import itertools
import json
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
worker = runpy.run_path(str(ROOT/'commands/run_mixtral_c006.py'))
MATRIX = Path(os.environ.get('MIXTRAL_MATRIX', ROOT.parents[1]/'doc_ppt/data/2026-09-17_Mixtral_C006_扩展BPD矩阵.json'))


def test_approved_capacity_and_task_coverage():
    matrix = json.loads(MATRIX.read_text())
    all_points = set(itertools.product(matrix['batches'], matrix['prefills'], matrix['decode_lengths']))
    valid = {(w['batch'], w['prefill'], w['decode']) for w in matrix['main_workloads']}
    excluded = {(w['batch'], w['prefill'], w['decode']) for w in matrix['capacity_excluded']}
    assert valid == {point for point in all_points if point[0]*(point[1]+point[2]) <= 13107}
    assert valid | excluded == all_points and not valid & excluded
    assert len(valid) == 75 and len(excluded) == 50 and 32 not in matrix['batches']
    args = argparse.Namespace(root=ROOT.parent, matrix=MATRIX, mode='replay', workload_index=0, route_index=0)
    outputs = set()
    for workload_index in range(75):
        args.workload_index = workload_index
        for route_index in range(40):
            args.route_index = route_index
            task, calls = worker['commands'](args, matrix)
            outputs.add(task)
            assert len(calls) == 1
            command = calls[0][2]
            assert command[command.index('--warmup')+1] == '0'
            assert command[command.index('--policies')+1:command.index('--arms')] == matrix['policies']
            assert command[command.index('--arms')+1:] == list(matrix['arms'])
    assert len(outputs) == 3000
    assert len(outputs)*len(matrix['policies'])*len(matrix['arms']) == matrix['total_replays'] == 45000
    args.mode = 'preflight'
    for args.workload_index in range(5):
        _, calls = worker['commands'](args, matrix)
        assert len(calls) == 4  # export, two hot shares, validation
        for _, _, command in calls[1:3]:
            assert command[command.index('--policies')+1:command.index('--arms')] == ['R0', 'R2']
            assert command[command.index('--seeds')+1] == '1001'
