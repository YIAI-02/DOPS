from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hetinfer_experiment_export as experiment


class ExportPerformanceTests(unittest.TestCase):
    def test_order_preserves_ready_priority_and_stable_ties(self):
        operators = [
            {"op_id": "a", "operator_index": 10, "dependencies": []},
            {"op_id": "b", "operator_index": 10, "dependencies": []},
            {"op_id": "c", "operator_index": 0, "dependencies": ["a"]},
        ]
        self.assertEqual(experiment._order(operators), ["a", "c", "b"])

    def test_order_does_not_update_every_pending_node_per_emitted_node(self):
        updates = []

        class CountedSet(set):
            def difference_update(self, values):
                updates.append(len(values))
                return super().difference_update(values)

            def remove(self, value):
                updates.append(1)
                return super().remove(value)

        count = 128
        operators = [
            {"op_id": str(i), "operator_index": i,
             "dependencies": [str(i - 1)] if i else []}
            for i in range(count)
        ]
        with mock.patch.object(experiment, "set", CountedSet, create=True):
            order = experiment._order(operators)
        self.assertEqual(order, [str(i) for i in range(count)])
        self.assertLessEqual(sum(updates), 4 * count)

