import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.call_path_negative_eval import load_cases, score_case


class CallPathNegativeEvalTest(unittest.TestCase):
    def test_exact_forbidden_target_and_truncation_are_reported(self):
        row = {
            "id": "near-miss", "entry_path": "entry.py", "entry_symbol": "run",
            "forbidden_target_path": "target.py", "forbidden_target_symbol": "wrong",
        }
        other = {"edges": [{"target_path": "target.py", "target_symbol": "right"}],
                 "truncated": False}
        self.assertFalse(score_case(row, other)["false_link"])
        matched = {"edges": [{"target_path": "target.py", "target_symbol": "wrong"}],
                   "truncated": True}
        self.assertTrue(score_case(row, matched)["false_link"])
        self.assertTrue(score_case(row, matched)["truncated"])

    def test_manifest_validation_rejects_duplicate_ids_and_traversal(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "entry.py").write_text("def run():\n    pass\n", encoding="utf-8")
            (root / "target.py").write_text("def wrong():\n    pass\n", encoding="utf-8")
            case = {
                "id": "near-miss", "repo_id": "sample", "entry_path": "entry.py",
                "entry_symbol": "run", "forbidden_target_path": "target.py",
                "forbidden_target_symbol": "wrong", "reason": "no call",
            }
            cases = root / "cases.json"
            cases.write_text(json.dumps({"schema_version": 1, "cases": [case, case]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cases(cases, {"sample": root})
            cases.write_text(json.dumps({"schema_version": 1, "cases": [
                {**case, "forbidden_target_path": "../target.py"}
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(cases, {"sample": root})


if __name__ == "__main__":
    unittest.main()
