"""Source-discovery diagnostic keeps source-file and function evidence distinct."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from scripts.source_discovery_eval import (
    PROJECT_ROOT, evaluate, joint_pair_rank, rank_source, source_span, summarize,
)


class SourceDiscoveryTests(unittest.TestCase):
    def test_transfer_anchors_match_frozen_single_file_questions(self) -> None:
        evalset = PROJECT_ROOT / "datasets/eval/source_discovery_transfer_v1.jsonl"
        first_look = json.loads((PROJECT_ROOT / "benchmarks/source-discovery-transfer-first-look-v1.json")
                                .read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(evalset.read_bytes()).hexdigest(),
                         first_look["evalset_sha256"])
        cases = [json.loads(line) for line in evalset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for repo_id in ("django", "leetcode-python"):
            anchors_path = PROJECT_ROOT / f"datasets/eval/source_discovery_transfer_{'leetcode' if repo_id == 'leetcode-python' else repo_id}_anchors_v1.json"
            self.assertEqual(hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
                             first_look["anchors_sha256"][repo_id])
            anchors = json.loads(anchors_path.read_text(encoding="utf-8"))["cases"]
            expected = {case["id"]: case["gold_files"][0]
                        for case in cases if case["repo_id"] == repo_id}
            self.assertEqual({item["id"]: item["source_path"] for item in anchors}, expected)
        with self.assertRaisesRegex(ValueError, "suite must be"):
            evaluate(PROJECT_ROOT, suite="unknown")

    def test_joint_pair_rank_counts_all_visible_candidates_in_order(self) -> None:
        groups = [
            {"path": "tests/test_source.py", "candidates": [{"name": "test_first"}, {"name": "test_second"}]},
            {"path": "package/source.py", "candidates": [{"name": "Source.run"}]},
        ]
        self.assertEqual(joint_pair_rank(groups, "package/source.py", "Source.run"), 3)
        self.assertIsNone(joint_pair_rank(groups, "package/source.py", "Source.missing"))

    def test_rank_source_requires_function_overlap(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "package" / "source.py"
            other = root / "package" / "other.py"
            results = [
                SimpleNamespace(chunk=SimpleNamespace(file_path=source, start_line=1, end_line=20)),
                SimpleNamespace(chunk=SimpleNamespace(file_path=other, start_line=1, end_line=20)),
                SimpleNamespace(chunk=SimpleNamespace(file_path=source, start_line=40, end_line=60)),
            ]
            self.assertEqual(rank_source(results, root, "package/source.py", (45, 52)), (1, 3, 3))
            self.assertEqual(rank_source(results[:2], root, "package/source.py", (45, 52)), (1, None, None))


    def test_source_span_rejects_missing_or_ambiguous_anchor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            record = SimpleNamespace(parent="Worker", name="run", kind="method", start_line=10, end_line=20)
            file_index = SimpleNamespace(path=source, symbols=[record])
            pipeline = SimpleNamespace(
                config=SimpleNamespace(repo_path=root),
                retriever=SimpleNamespace(repo_index=SimpleNamespace(files={"one": file_index})),
            )
            self.assertEqual(source_span(pipeline, "source.py", "Worker.run"), (10, 20))
            with self.assertRaisesRegex(ValueError, "0 indexed matches"):
                source_span(pipeline, "source.py", "Worker.missing")
            file_index.symbols.append(record)
            with self.assertRaisesRegex(ValueError, "2 indexed matches"):
                source_span(pipeline, "source.py", "Worker.run")


    def test_summary_does_not_count_file_hit_as_function_hit(self) -> None:
        rows = [{"file_rank": 1, "function_span_rank": None, "definition_rank": None,
                 "file_local_symbol_rank": 1, "joint_candidate_hit": True,
                 "joint_first_pair_correct": True, "joint_pair_rank": 1},
                {"file_rank": 8, "function_span_rank": 8, "definition_rank": None,
                 "file_local_symbol_rank": 4, "joint_candidate_hit": False,
                 "joint_first_pair_correct": False, "joint_pair_rank": None}]
        self.assertEqual(summarize(rows), {
            "cases": 2, "source_file_at_5": 0.5, "source_file_at_20": 1,
            "source_function_at_5": 0, "source_function_at_20": 0.5,
            "source_definition_at_5": 0, "source_definition_at_20": 0,
            "file_local_symbol_at_1": 0.5, "file_local_symbol_at_3": 0.5,
            "file_local_symbol_at_5": 1,
            "two_stage_reachable_at_5_files_3_symbols": 0.5,
            "joint_candidate_hit_at_5_files_3_symbols": 0.5,
            "joint_first_pair_accuracy": 0.5,
            "joint_pair_at_3": 0.5, "joint_pair_at_5": 0.5,
            "joint_pair_at_10": 0.5,
        })
