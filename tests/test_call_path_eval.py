import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.call_path_eval import confined_path, gate, load_cases, score_case


class CallPathEvalTest(unittest.TestCase):
    def test_case_scores_target_and_real_line_locators(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    return helper()\n", encoding="utf-8")
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            row = {
                "id": "one", "repo_id": "fixture", "entry_path": "entry.py",
                "expected_targets": [{"path": "target.py", "symbol": "helper"}],
            }
            edge = {
                "source_line": 2, "target_path": "target.py", "target_symbol": "helper",
                "target_start_line": 1, "target_end_line": 2,
            }
            complete = score_case(row, {"edges": [edge], "truncated": False}, root)
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["located_target_count"], 1)

            invalid = score_case(
                row, {"edges": [{**edge, "target_start_line": 99}], "truncated": False}, root
            )
            self.assertFalse(invalid["complete"])
            self.assertEqual(invalid["matched_target_count"], 1)
            self.assertEqual(invalid["located_target_count"], 0)
            self.assertEqual(invalid["invalid_edge_count"], 1)

    def test_fixture_rejects_unconfined_paths_and_unknown_repo(self):
        for value in ("../outside.py", "/abs.py", "a\\b.py", "a/../b.py", "C:/outside.py", "."):
            with self.assertRaises(ValueError):
                confined_path(value)
        with TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "cases.json"
            fixture.write_text(json.dumps({
                "schema_version": 1,
                "cases": [{
                    "id": "bad", "repo_id": "unknown", "entry_path": "entry.py",
                    "entry_symbol": "run",
                    "expected_targets": [{"path": "target.py", "symbol": "helper"}],
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown repo_id"):
                load_cases(fixture, {"fixture"})

    def test_gate_rejects_identity_drift_and_metric_regression(self):
        base = {
            "schema_version": 1, "benchmark": "pinned_python_call_path_component",
            "dataset": {"sha256": "fixed"}, "manifest": {"sha256": "fixed"},
            "metrics": {
                "target_recall": 1.0, "locator_rate": 1.0,
                "complete_case_rate": 1.0, "invalid_edge_count": 0,
            },
        }
        policy = {
            "schema_version": 1, "benchmark": base["benchmark"],
            "min_target_recall": 1.0, "min_locator_rate": 1.0,
            "min_complete_case_rate": 1.0, "max_invalid_edge_count": 0,
        }
        self.assertEqual(gate(base, base, policy), [])
        altered = {**base, "dataset": {"sha256": "changed"}, "metrics": {
            **base["metrics"], "target_recall": 0.5,
        }}
        failures = gate(altered, base, policy)
        self.assertIn("baseline dataset mismatch", failures)
        self.assertIn("target_recall regression", failures)


if __name__ == "__main__":
    unittest.main()
