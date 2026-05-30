from pathlib import Path
import unittest

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk
from rag.trace import build_retrieval_trace, render_trace_report


class RetrievalTraceTest(unittest.TestCase):
    def test_build_trace_contains_all_stages(self):
        seed = SearchResult(
            CodeChunk(Path("a.py"), 1, 3, "def target():\n    return 1"),
            0.9,
            source="hybrid",
        )
        recall = SearchResult(
            CodeChunk(Path("a.py"), 4, 5, "def helper():\n    return 2"),
            0.4,
            source="recall:seed-1+1",
        )

        trace = build_retrieval_trace(
            query="target 在哪里",
            seed_results=[seed],
            recalled_results=[seed, recall],
            final_results=[seed, recall],
            params={"top_k": 1},
        )

        self.assertEqual(trace["summary"]["seed_count"], 1)
        self.assertEqual(trace["summary"]["recalled_count"], 2)
        self.assertEqual(trace["summary"]["final_count"], 2)
        self.assertEqual(trace["summary"]["context_block_count"], 1)
        self.assertIn("initial_search", trace["stages"])
        self.assertIn("context_compaction", trace["stages"])
        self.assertEqual(trace["stages"]["context_compaction"][0]["chunk_count"], 2)
        self.assertIn("char_count", trace["stages"]["context_compaction"][0])

    def test_render_trace_report_is_readable(self):
        seed = SearchResult(
            CodeChunk(Path("a.py"), 1, 3, "def target():\n    return 1"),
            0.9,
            source="hybrid",
        )
        trace = build_retrieval_trace(
            query="target 在哪里",
            seed_results=[seed],
            recalled_results=[seed],
            final_results=[seed],
        )

        report = render_trace_report(trace)

        self.assertIn("RAG 检索轨迹", report)
        self.assertIn("Initial Search", report)
        self.assertIn("Context Compaction", report)
        self.assertIn("a.py:1-3", report)


if __name__ == "__main__":
    unittest.main()
