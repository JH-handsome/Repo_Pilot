import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.call_edge_expansion_ab import main


class CallEdgeExpansionABTest(unittest.TestCase):
    def test_report_records_identity_and_exact_edge_metric(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            (repo / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            evalset = root / "cases.jsonl"
            evalset.write_text(json.dumps({
                "id": "case-1", "repo_id": "fixture", "query": "run calls helper",
                "gold_files": ["entry.py", "target.py"], "task_type": "call_path",
                "difficulty": "easy",
            }) + "\n", encoding="utf-8")
            grounding = root / "grounding.json"
            grounding.write_text(json.dumps({
                "schema_version": 1,
                "cases": [{"id": "case-1", "source_symbol": "run", "target_symbol": "helper"}],
            }), encoding="utf-8")
            output = root / "scorecard.json"
            arguments = [
                "call_edge_expansion_ab.py", "--repo", str(repo), "--evalset", str(evalset),
                "--grounding", str(grounding), "--line-weight", "1", "--out", str(output),
            ]
            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["exact_edge_at_1"], 1)
            self.assertEqual(report["repository_identity"]["status"], "unverified")
            self.assertEqual(len(report["repository_fingerprint"]), 64)
            self.assertEqual(len(report["datasets"][0]["sha256"]), 64)
            anchors = root / "anchors.json"
            anchors.write_text(json.dumps({
                "schema_version": 1,
                "cases": [{"id": "case-1", "source_path": "entry.py", "source_symbol": "run"}],
            }), encoding="utf-8")
            anchored_output = root / "anchored.json"
            with patch("sys.argv", [*arguments[:-1], str(anchored_output), "--anchors", str(anchors)]), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            anchored = json.loads(anchored_output.read_text(encoding="utf-8"))
            self.assertEqual(anchored["summary"]["anchored_source_case_count"], 1)
            self.assertEqual(anchored["summary"]["exact_edge_at_1"], 1)
            self.assertEqual(anchored["config"]["candidate_mode"], "source_anchored")
            self.assertEqual(anchored["config"]["source_files"], 1)
            self.assertEqual(len(anchored["anchors_sha256"]), 64)
            auto_output = root / "auto.json"
            with patch("sys.argv", [*arguments[:-1], str(auto_output), "--auto-anchor"]), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            automatic = json.loads(auto_output.read_text(encoding="utf-8"))
            self.assertEqual(automatic["config"]["candidate_mode"], "query_only_auto_anchor")
            self.assertEqual(automatic["summary"]["auto_anchor_source_file_match_count"], 1)
            self.assertEqual(automatic["summary"]["exact_edge_at_1"], 1)
            self.assertEqual(automatic["cases"][0]["source_anchor"]["origin"],
                             "query_only_first_non_test_file_first_symbol")

    def test_duplicate_case_ids_fail_before_index_build(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entry.py").write_text("def run():\n    pass\n", encoding="utf-8")
            evalset = root / "cases.jsonl"
            evalset.write_text(json.dumps({"id": "duplicate"}) + "\n", encoding="utf-8")
            arguments = [
                "call_edge_expansion_ab.py", "--repo", str(root),
                "--evalset", str(evalset), "--evalset", str(evalset),
            ]
            with patch("sys.argv", arguments):
                with self.assertRaisesRegex(SystemExit, "duplicate case ID"):
                    main()

    def test_unknown_case_filter_fails_before_index_build(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evalset = root / "cases.jsonl"
            evalset.write_text(json.dumps({"id": "known"}) + "\n", encoding="utf-8")
            arguments = [
                "call_edge_expansion_ab.py", "--repo", str(root), "--evalset", str(evalset),
                "--case-id", "missing",
            ]
            with patch("sys.argv", arguments):
                with self.assertRaisesRegex(SystemExit, "case-id values must be unique"):
                    main()

    def test_anchor_must_point_at_labeled_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evalset = root / "cases.jsonl"
            evalset.write_text(json.dumps({"id": "case-1", "gold_files": ["entry.py"]}) + "\n", encoding="utf-8")
            anchors = root / "anchors.json"
            anchors.write_text(json.dumps({
                "schema_version": 1,
                "cases": [{"id": "case-1", "source_path": "wrong.py", "source_symbol": "run"}],
            }), encoding="utf-8")
            arguments = [
                "call_edge_expansion_ab.py", "--repo", str(root), "--evalset", str(evalset),
                "--anchors", str(anchors),
            ]
            with patch("sys.argv", arguments):
                with self.assertRaisesRegex(SystemExit, "invalid source anchor"):
                    main()

    def test_local_only_case_reports_misleading_suggestion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            (repo / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            evalset = root / "cases.jsonl"
            evalset.write_text(json.dumps({
                "id": "local-only", "repo_id": "fixture", "query": "where does run call helper locally",
                "gold_files": ["entry.py"], "task_type": "local_behavior",
                "difficulty": "hard", "expect_no_cross_file": True,
            }) + "\n", encoding="utf-8")
            output = root / "scorecard.json"
            arguments = [
                "call_edge_expansion_ab.py", "--repo", str(repo), "--evalset", str(evalset),
                "--out", str(output),
            ]
            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["negative_case_count"], 1)
            self.assertEqual(report["summary"]["negative_suggestion_at_1"], 1)
            self.assertGreater(report["cases"][0]["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
