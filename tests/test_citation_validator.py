from pathlib import Path
import unittest

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk
from rag.citation_validator import (
    append_citation_validation_report,
    extract_citations,
    validate_answer_citations,
)


class CitationValidatorTest(unittest.TestCase):
    def test_extracts_path_line_range_citations(self):
        citations = extract_citations("参考 coding_rag/file_loader.py:10-20。")

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].path, "coding_rag/file_loader.py")
        self.assertEqual(citations[0].start_line, 10)
        self.assertEqual(citations[0].end_line, 20)

    def test_validates_citation_against_retrieved_context(self):
        result = SearchResult(
            CodeChunk(Path("coding_rag/file_loader.py"), 10, 30, "def load_python_files():\n    pass"),
            1.0,
            source="hybrid",
        )

        validation = validate_answer_citations(
            "相关代码在 coding_rag/file_loader.py:10-30。",
            [result],
        )

        self.assertFalse(validation.has_issues)

    def test_flags_missing_and_unsupported_citations(self):
        result = SearchResult(
            CodeChunk(Path("coding_rag/file_loader.py"), 10, 30, "def load_python_files():\n    pass"),
            1.0,
            source="hybrid",
        )

        missing = validate_answer_citations("这里解释了加载逻辑。", [result])
        invalid = validate_answer_citations("见 coding_rag/file_loader.py:1-5。", [result])

        self.assertTrue(missing.missing_citations)
        self.assertEqual([citation.text for citation in invalid.invalid_citations], ["coding_rag/file_loader.py:1-5"])

    def test_appends_validation_report_only_when_needed(self):
        result = append_citation_validation_report(
            "这里解释了加载逻辑。",
            validate_answer_citations(
                "这里解释了加载逻辑。",
                [
                    SearchResult(
                        CodeChunk(Path("coding_rag/file_loader.py"), 1, 5, "def load_python_files(): pass"),
                        1.0,
                        source="hybrid",
                    )
                ],
            ),
        )

        self.assertIn("## 引用校验", result)
        self.assertIn("未检测到 path:start-end 格式引用", result)


if __name__ == "__main__":
    unittest.main()
