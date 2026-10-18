import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.call_path_challenge_eval import compare, load_cases


class CallPathChallengeEvalTest(unittest.TestCase):
    def test_challenge_requires_both_outcomes_and_confined_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    return helper()\n", encoding="utf-8")
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            path = root / "cases.json"
            rows = [
                {"id": "positive", "repo_id": "repo", "query": "Trace run() through helper()",
                 "entry_path": "entry.py", "entry_symbol": "run", "expected_outcome": "link",
                 "target_path": "target.py", "target_symbol": "helper"},
                {"id": "negative", "repo_id": "repo", "query": "Trace run() through unrelated()",
                 "entry_path": "entry.py", "entry_symbol": "run", "expected_outcome": "abstain",
                 "reason": "run does not call unrelated"},
            ]
            path.write_text(json.dumps({"schema_version": 1, "cases": rows}), encoding="utf-8")
            self.assertEqual(len(load_cases(path, {"repo": root})), 2)
            rows[0]["target_path"] = "../target.py"
            path.write_text(json.dumps({"schema_version": 1, "cases": rows}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path, {"repo": root})

    def test_gate_rejects_identity_drift_and_false_links(self):
        baseline = {
            "schema_version": 1, "benchmark": "challenge", "dataset": {"sha256": "fixed"},
            "manifest": {}, "repositories": [], "config": {}, "retriever": "v6", "policy": "v3",
            "evaluation_mode": "offline", "metrics": {
                "case_count": 9, "positive_count": 4, "negative_count": 5,
                "positive_link_rate": 1.0, "negative_abstain_rate": 1.0,
                "false_link_count": 0,
            },
        }
        self.assertEqual(compare(baseline, baseline), [])
        changed = copy.deepcopy(baseline)
        changed["dataset"]["sha256"] = "drifted"
        changed["metrics"]["negative_abstain_rate"] = 0.8
        changed["metrics"]["false_link_count"] = 1
        self.assertIn("incompatible dataset", compare(changed, baseline))
        self.assertIn("negative_abstain_rate regression", compare(changed, baseline))
        self.assertIn("false_link_count regression", compare(changed, baseline))


if __name__ == "__main__":
    unittest.main()
