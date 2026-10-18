from pathlib import Path
import unittest

from coding_rag.repository.chunks import CodeChunk
from coding_rag.rag.citation_validator import (
    append_citation_validation_report,
    extract_citations,
    validate_answer_citations,
    validate_answer_citations_against_ranges,
)
from coding_rag.tools.bm25 import SearchResult


class CitationValidatorTest(unittest.TestCase):
    def test_rejects_forged_prefix_and_ambiguous_basename(self):
        ranges = [("pkg1/app.py", 1, 4), ("pkg2/app.py", 1, 4)]

        self.assertFalse(validate_answer_citations_against_ranges(
            "见 pkg1/app.py:2-2", ranges,
        ).has_issues)
        self.assertTrue(validate_answer_citations_against_ranges(
            "见 forged/pkg1/app.py:2-2", ranges,
        ).has_issues)
        self.assertTrue(validate_answer_citations_against_ranges(
            "见 app.py:2-2", ranges,
        ).has_issues)

    def test_relative_citation_may_match_observed_absolute_path(self):
        self.assertFalse(validate_answer_citations_against_ranges(
            "见 pkg/app.py:2-2", [("C:/repo/pkg/app.py", 1, 4)],
        ).has_issues)

    def test_extracts_path_line_range_citations(self):
        citations = extract_citations("See coding_rag/file_loader.py:10-20.")

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
            "Relevant code is in coding_rag/file_loader.py:10-30.",
            [result],
        )

        self.assertFalse(validation.has_issues)

    def test_flags_missing_and_unsupported_citations(self):
        result = SearchResult(
            CodeChunk(Path("coding_rag/file_loader.py"), 10, 30, "def load_python_files():\n    pass"),
            1.0,
            source="hybrid",
        )

        missing = validate_answer_citations("This explains file loading.", [result])
        invalid = validate_answer_citations("See coding_rag/file_loader.py:1-5.", [result])

        self.assertTrue(missing.missing_citations)
        self.assertEqual([citation.text for citation in invalid.invalid_citations], ["coding_rag/file_loader.py:1-5"])

    def test_appends_validation_report_only_when_needed(self):
        result = append_citation_validation_report(
            "This explains file loading.",
            validate_answer_citations(
                "This explains file loading.",
                [
                    SearchResult(
                        CodeChunk(Path("coding_rag/file_loader.py"), 1, 5, "def load_python_files(): pass"),
                        1.0,
                        source="hybrid",
                    )
                ],
            ),
        )

        self.assertIn("##", result)
        self.assertIn("path:start-end", result)


if __name__ == "__main__":
    unittest.main()
