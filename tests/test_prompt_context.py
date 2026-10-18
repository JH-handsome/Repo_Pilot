from pathlib import Path
import unittest
from unittest.mock import patch

from coding_rag.rag.answer_generator import AnswerGenerator
from coding_rag.rag.llm_client import LLMConfig
from coding_rag.tools.bm25 import SearchResult
from coding_rag.repository.chunks import CodeChunk
from coding_rag.rag.prompt import (
    compact_results_for_context, format_results_as_context,
    format_results_as_context_with_ranges,
)


class PromptContextTest(unittest.TestCase):
    def test_compat_generator_rejects_citation_to_omitted_line(self):
        result = SearchResult(
            CodeChunk(Path("a.py"), 1, 30,
                      "\n".join(f"line_{index:02d} = 123456789" for index in range(1, 31))),
            0.9, source="hybrid",
        )
        config = LLMConfig("test", "https://unused.invalid", "unused", "", "UNUSED_KEY")
        with patch("coding_rag.rag.answer_generator.OpenAICompatibleChatClient") as client_type:
            client_type.return_value.complete.return_value = "见 a.py:30-30。"
            answer = AnswerGenerator(config, max_context_chars=220).generate("Where?", [result])

        self.assertIn("无效引用: a.py:30-30", answer)

    def test_visible_ranges_exclude_cut_off_lines_and_omitted_blocks(self):
        first = SearchResult(
            CodeChunk(Path("a.py"), 1, 30,
                      "\n".join(f"line_{index:02d} = 123456789" for index in range(1, 31))),
            0.9, source="hybrid",
        )
        second = SearchResult(CodeChunk(Path("b.py"), 1, 1, "hidden = True"), 0.8,
                              source="hybrid")
        results = [first, second]

        context, ranges = format_results_as_context_with_ranges(results, max_context_chars=220)

        self.assertEqual(context, format_results_as_context(results, max_context_chars=220))
        self.assertIn("代码已截断", context)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0][0], "a.py")
        self.assertLess(ranges[0][2], 30)
        self.assertNotIn("b.py", context)

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
