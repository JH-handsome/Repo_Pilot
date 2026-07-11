from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from scripts.agent_eval import evaluate_cases
from scripts.generate_function_map import generate_function_map


class FunctionMapAndAgentEvalTest(unittest.TestCase):
    def test_generate_function_map_contains_key_modules(self):
        text = generate_function_map(["main.py", "coding_rag/agent", "coding_rag/rag"])

        self.assertIn("main.py", text)
        self.assertIn("coding_rag/agent/planner.py", text)
        self.assertIn("coding_rag/rag/trace.py", text)
        self.assertIn("function", text)

    def test_agent_eval_computes_summary(self):
        with TemporaryDirectory() as temp_dir:
            evalset = Path(temp_dir) / "cases.json"
            evalset.write_text(
                json.dumps(
                    [
                        {
                            "id": "case-1",
                            "task": "fix target",
                            "repo_files": {"main.py": "def target():\n    pass\n"},
                            "expected_tools": ["search_code", "read_file"],
                            "expected_sections": ["任务分析", "验证计划"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            rows, summary = evaluate_cases(evalset)

        self.assertEqual(summary["total"], 1)
        self.assertTrue(rows[0]["trace_complete"])
        self.assertGreater(summary["avg_expected_tool_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
