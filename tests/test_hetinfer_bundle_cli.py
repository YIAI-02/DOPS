from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mainlib.cli import parse_args


def test_evaluate_accepts_one_bundle_output():
    with mock.patch.object(sys, 'argv', ['main.py', 'evaluate', '--config', 'config.json',
                                        '--hetinfer-bundle-out', 'bundle.json']):
        args = parse_args()
    assert args.hetinfer_bundle_out == 'bundle.json'
