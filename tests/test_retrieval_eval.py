import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk
from scripts.retrieval_eval import (
    EvalCase,
    analyze_bad_case,
    build_stage_file_record,
    default_recall_max_results,
    load_evalset,
    mrr_and_recall_at_k,
    stage_files_path_for_trace,
)


class RetrievalEvalPathMatchingTest(unittest.TestCase):
    def test_load_evalset_accepts_trace_jsonl_format(self):
        with TemporaryDirectory() as temp_dir:
            eval_path = Path(temp_dir) / "trace.jsonl"
            eval_path.write_text(
                '\n'
                '{"id":"q001","question":"Where is the fixture loader?","gold_files":["coding_rag/file_loader.py"]}\n'
                '\n'
                '{"id":"q002","question":"Where is fixture filtering?","gold_files":["coding_rag/file_loader.py"]}\n',
                encoding="utf-8",
            )

            cases = load_evalset(str(eval_path))

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].id, "q001")
        self.assertEqual(cases[0].query, "Where is the fixture loader?")
        self.assertEqual(cases[0].relevant, ["coding_rag/file_loader.py"])

    def test_relative_relevant_path_matches_absolute_result_path(self):
        root = Path.cwd()
        result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "bm25_retriever.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=1.0,
        )

        rr, recall, hits = mrr_and_recall_at_k(
            [result],
            ["coding_rag/bm25_retriever.py"],
            k=1,
            path_roots=[root],
        )

        self.assertEqual(rr, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(hits, 1)

    def test_repo_relative_relevant_path_matches_when_repo_root_is_provided(self):
        root = Path.cwd()
        repo_root = root / "coding_rag"
        result = SearchResult(
            chunk=CodeChunk(
                file_path=repo_root / "context_recaller.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=1.0,
        )

        bad_case = analyze_bad_case(
            seed=[result],
            recalled=[result],
            final=[result],
            relevant=["context_recaller.py"],
            k=1,
            path_roots=[repo_root],
        )

        self.assertTrue(bad_case["seed_hit"])
        self.assertTrue(bad_case["recall_hit"])
        self.assertTrue(bad_case["final_hit"])

    def test_recall_counts_each_relevant_file_once(self):
        root = Path.cwd()
        first_chunk = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "result_filter.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=2.0,
        )
        second_chunk = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "result_filter.py",
                start_line=11,
                end_line=20,
                text="",
            ),
            score=1.0,
        )

        _, recall, hits = mrr_and_recall_at_k(
            [first_chunk, second_chunk],
            ["coding_rag/result_filter.py"],
            k=2,
            path_roots=[root],
        )

        self.assertEqual(recall, 1.0)
        self.assertEqual(hits, 1)

    def test_default_recall_max_results_scales_with_window(self):
        self.assertEqual(default_recall_max_results(top_k=5, recall_window=1), 20)
        self.assertEqual(default_recall_max_results(top_k=5, recall_window=2), 25)

    def test_bad_case_does_not_flag_recall_window_when_final_hits(self):
        root = Path.cwd()
        gold_result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "file_loader.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=2.0,
        )
        wrong_result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "bm25_retriever.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=1.0,
        )

        bad_case = analyze_bad_case(
            seed=[gold_result],
            recalled=[wrong_result, gold_result],
            final=[gold_result],
            relevant=["coding_rag/file_loader.py"],
            k=1,
            path_roots=[root],
        )

        self.assertTrue(bad_case["recall_hit"])
        self.assertFalse(bad_case["recall_top_k_hit"])
        self.assertEqual(bad_case["reasons"], [])

    def test_stage_files_path_is_derived_from_trace_path(self):
        self.assertEqual(
            stage_files_path_for_trace(Path("artifacts/retrieval_trace.jsonl")),
            Path("artifacts/retrieval_trace_stage_files.jsonl"),
        )

    def test_stage_file_record_marks_initial_search_miss(self):
        root = Path.cwd()
        case = EvalCase(id="q001", query="Where is the loader?", relevant=["coding_rag/file_loader.py"])
        wrong_result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "bm25_retriever.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=1.0,
        )

        record = build_stage_file_record(
            case=case,
            seed=[wrong_result],
            recalled=[wrong_result],
            final=[wrong_result],
            k=1,
            path_roots=[root],
        )

        self.assertEqual(record["diagnosis"], "missing_in_initial_search")
        self.assertEqual(record["initial_review"]["files"], ["coding_rag/bm25_retriever.py"])
        self.assertFalse(record["initial_review"]["hit"])

    def test_stage_file_record_marks_post_recall_filter_drop(self):
        root = Path.cwd()
        case = EvalCase(id="q002", query="Where is the loader?", relevant=["coding_rag/file_loader.py"])
        gold_result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "file_loader.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=2.0,
        )
        wrong_result = SearchResult(
            chunk=CodeChunk(
                file_path=root / "coding_rag" / "bm25_retriever.py",
                start_line=1,
                end_line=10,
                text="",
            ),
            score=1.0,
        )

        record = build_stage_file_record(
            case=case,
            seed=[gold_result],
            recalled=[gold_result],
            final=[wrong_result],
            k=1,
            path_roots=[root],
        )

        self.assertEqual(record["diagnosis"], "filtered_after_recall_review")
        self.assertTrue(record["initial_review"]["hit"])
        self.assertTrue(record["recall_review_top_k"]["hit"])
        self.assertFalse(record["final"]["hit"])


if __name__ == "__main__":
    unittest.main()
