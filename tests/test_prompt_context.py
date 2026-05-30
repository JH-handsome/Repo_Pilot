from pathlib import Path
import unittest

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk
from rag.prompt import compact_results_for_context, format_results_as_context


class PromptContextTest(unittest.TestCase):
    def test_compacts_overlapping_chunks_from_same_file(self):
        first = CodeChunk(Path("a.py"), 1, 5, "\n".join(["line1", "line2", "line3", "line4", "line5"]))
        second = CodeChunk(Path("a.py"), 4, 8, "\n".join(["line4", "line5", "line6", "line7", "line8"]))

        blocks = compact_results_for_context(
            [
                SearchResult(first, 0.9, source="hybrid"),
                SearchResult(second, 0.5, source="recall:seed-1+1"),
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].start_line, 1)
        self.assertEqual(blocks[0].end_line, 8)
        self.assertEqual(blocks[0].text.count("line4"), 1)
        self.assertEqual(blocks[0].sources, ("hybrid", "recall:seed-1+1"))

    def test_context_header_reports_merged_range_and_chunk_count(self):
        first = CodeChunk(Path("a.py"), 1, 3, "a\nb\nc")
        second = CodeChunk(Path("a.py"), 4, 5, "d\ne")

        context = format_results_as_context(
            [
                SearchResult(first, 0.8, source="hybrid"),
                SearchResult(second, 0.4, source="recall:seed-1+1"),
            ]
        )

        self.assertIn("a.py:1-5", context)
        self.assertIn("chunks=2", context)
        self.assertIn("sources=hybrid,recall:seed-1+1", context)

    def test_non_contiguous_chunks_stay_separate(self):
        first = CodeChunk(Path("a.py"), 1, 3, "a\nb\nc")
        second = CodeChunk(Path("a.py"), 10, 12, "x\ny\nz")

        blocks = compact_results_for_context(
            [
                SearchResult(first, 0.8, source="hybrid"),
                SearchResult(second, 0.7, source="hybrid"),
            ]
        )

        self.assertEqual(len(blocks), 2)


if __name__ == "__main__":
    unittest.main()
