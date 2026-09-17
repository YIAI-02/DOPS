import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commands import export_mixtral_hotcold_fast as exporter


def test_c006_default_bundle_reuse_and_parameter_override(tmp_path):
    calls = []
    selected = json.loads((ROOT / 'configs/mixtral_hotcold_selected_parameters.json').read_text())
    def run(command, **kwargs):
        calls.append(command)
        cfg = json.loads((tmp_path / 'config.json').read_text())
        data = {'bifocal_parameters': {k: cfg[k] for k in selected},
                'batch': 1, 'prefill': 128, 'decode_rounds': 2,
                'networks': [{'default_order': ['x'], 'operators': [
                    {'op_id': 'x', 'default_device': 'PIM0'}]}]}
        Path(cfg['hetinfer_bundle_out']).write_text(json.dumps(data))
        return subprocess.CompletedProcess(command, 0)
    argv = ['export_mixtral_hotcold_fast.py', '--batch', '1', '--horizon', '2', '--output', str(tmp_path)]
    with mock.patch.dict(os.environ, {'SLURM_JOB_ID': 'test'}), \
         mock.patch.object(sys, 'argv', argv), \
        mock.patch.object(exporter.subprocess, 'run', side_effect=run):
        exporter.main()
        exporter.main()
        assert len(calls) == 1
        cfg = json.loads((tmp_path / 'config.json').read_text())
        assert [cfg[key] for key in ('SCHED_JOINT_LK_H', 'SCHED_JOINT_LK_GAMMA',
                                    'SCHED_JOINT_LK_CONSIST_LAMBDA', 'SCHED_DECODE_AMORT_ALPHA')] == [2, .4, .5, 0]
        assert 'bifocal_preset' not in cfg
        override = tmp_path / 'parameters.json'
        override.write_text(json.dumps({'SCHED_DECODE_AMORT_ALPHA': 1}))
        argv.extend(['--parameters', str(override)])
        exporter.main()
        exporter.main()
        assert len(calls) == 2
        bundle = json.loads((tmp_path / 'bundle.json').read_text())
        assert bundle['bifocal_parameters'] == {**selected, 'SCHED_DECODE_AMORT_ALPHA': 1}
    assert not (tmp_path / 'plan_trace.json.gz').exists()
    assert not (tmp_path / 'native').exists()
    assert not (tmp_path / 'bundle').exists()
