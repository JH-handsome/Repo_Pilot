import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.agent_eval import candidate_file_metrics, evaluate_cases, safe_fixture_path, search_result_files


class AgentEvalMetricsTest(unittest.TestCase):
    def test_candidate_metrics_use_ordered_unique_real_results(self):
        files = search_result_files([
            {"tool": "list_files", "output": ["ignored.py"]},
            {"tool": "search_code", "output": [
                {"path": "other.py"}, {"path": "Target.py"}, {"path": "target.py"},
            ]},
        ])
        self.assertEqual(files, ["other.py", "target.py"])
        metrics = candidate_file_metrics(files, ["target.py", "test_target.py"], 2)
        self.assertTrue(metrics["candidate_file_hit"])
        self.assertEqual(metrics["candidate_recall_at_2"], 0.5)
        self.assertEqual(metrics["candidate_mrr_at_2"], 0.5)

    def test_empty_gold_is_explicit_zero_not_automatic_full_credit(self):
        metrics = candidate_file_metrics(["app.py"], [], 2)
        self.assertEqual(metrics["gold_status"], "empty")
        self.assertFalse(metrics["candidate_file_hit"])
        self.assertEqual(metrics["candidate_recall_at_2"], 0.0)
        self.assertEqual(metrics["candidate_mrr_at_2"], 0.0)

    def test_fixture_path_rejects_absolute_and_parent_traversal(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                safe_fixture_path(root, "../escape.py")
            with self.assertRaises(ValueError):
                safe_fixture_path(root, Path(temp_dir).anchor + "escape.py")

    def test_evaluate_cases_reports_real_results_and_repo_summary(self):
        with TemporaryDirectory() as temp_dir:
            evalset = Path(temp_dir) / "agent.json"
            evalset.write_text(json.dumps([{
                "id": "case-1",
                "repo": "sample/repo",
                "task": "fix unique_target",
                "changed_files": ["src/app.py"],
                "test_files": ["tests/test_app.py", "tests/missing.py"],
                "repo_files": {
                    "src/app.py": "def unique_target():\n    return 1\n",
                    "tests/test_app.py": "def test_unique_target():\n    assert True\n",
                },
                "expected_tools": ["search_code"],
                "expected_sections": ["任务分析"],
            }]), encoding="utf-8")
            rows, summary = evaluate_cases(evalset)
        self.assertEqual(rows[0]["result_files"][0], "src/app.py")
        self.assertTrue(rows[0]["candidate_file_hit"])
        self.assertEqual(rows[0]["gold_file_count"], 3)
        self.assertEqual(rows[0]["available_gold_count"], 2)
        self.assertEqual(rows[0]["missing_gold_count"], 1)
        self.assertEqual(rows[0]["negative_file_count"], 0)
        self.assertFalse(rows[0]["discriminative"])
        self.assertEqual(rows[0]["candidate_recall_at_5"], 1.0)
        diagnostic = summary["fixture_diagnostic"]
        self.assertAlmostEqual(diagnostic["dataset_coverage"], 2 / 3)
        self.assertEqual(diagnostic["cases_with_missing_gold"], ["case-1"])
        self.assertEqual(diagnostic["negative_file_count"], 0)
        self.assertFalse(diagnostic["discriminative"])
        self.assertEqual(summary["evaluation_mode"], "offline_contract")
        self.assertEqual(diagnostic["by_repo"]["sample/repo"]["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
