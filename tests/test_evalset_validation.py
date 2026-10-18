import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.evaluation.retrieval import validate_evalset


class EvalsetValidationTest(unittest.TestCase):
    def test_repository_evalsets_are_valid(self):
        for path, count in (
            ("datasets/eval/sample_evalset.json", 3),
            ("datasets/eval/trace.jsonl", 20),
            ("datasets/eval/trace_v2.jsonl", 20),
        ):
            result = validate_evalset(".", path)
            self.assertTrue(result.valid, result.to_dict())
            self.assertEqual(result.case_count, count)

    def test_current_runtime_evalset_does_not_label_legacy_answer_generator_as_main_path(self):
        cases = [json.loads(line) for line in Path("datasets/eval/trace_v2.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {case["id"]: case for case in cases}
        self.assertEqual(by_id["q012"]["gold_files"], ["coding_rag/agent/graph_runtime.py"])
        self.assertEqual(by_id["q014"]["gold_files"], ["coding_rag/agent/executor.py"])
        self.assertEqual(by_id["q015"]["gold_files"], ["coding_rag/tools/agent_readonly.py"])
        self.assertFalse(any(
            "coding_rag/rag/answer_generator.py" in case["gold_files"] for case in cases
        ))

    def test_reports_duplicate_empty_invalid_and_missing_values(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ok.py").write_text("pass\n", encoding="utf-8")
            cases = root / "cases.json"
            cases.write_text(json.dumps([
                {"id": "same", "query": "", "relevant": []},
                {"id": "same", "query": "q", "relevant": ["../escape.py", "missing.py"]},
            ]), encoding="utf-8")
            result = validate_evalset(root, cases)
            self.assertFalse(result.valid)
            self.assertEqual(result.duplicate_ids, ("same",))
            self.assertEqual(result.empty_queries, ("same",))
            self.assertTrue(result.invalid_paths)
            self.assertTrue(result.missing_gold_files)

    def test_cli_uses_zero_and_two_exit_codes(self):
        valid = subprocess.run([sys.executable, "scripts/validate_evalset.py", ".", "datasets/eval/sample_evalset.json"], check=False)
        self.assertEqual(valid.returncode, 0)
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "bad.json"
            target.write_text("[]", encoding="utf-8")
            invalid = subprocess.run([sys.executable, "scripts/validate_evalset.py", ".", str(target)], check=False)
        self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
