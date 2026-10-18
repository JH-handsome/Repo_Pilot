"""File-local symbol ranking is explicit, bounded, and definition-grounded."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.repository.chunks import split_code_by_lines
from coding_rag.repository.symbol_locator import (
    exact_definition_lookup_should_abstain,
    locate_file_symbols,
    locate_retrieved_files,
    missing_qualified_definitions,
)
from coding_rag.tools.bm25 import BM25Retriever


class SymbolLocatorTests(unittest.TestCase):
    def test_class_mode_keeps_function_default_unchanged(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "backend.py"
            chunks = split_code_by_lines(source, (
                "class ModelBackend:\n"
                "    def authenticate(self, credentials):\n"
                "        return credentials\n"
                "    def get_group_permissions(self, user):\n"
                "        return user.permissions\n"
            ), 3, 1)
            retriever = BM25Retriever(chunks, path_root=root)
            query = "Find the backend that authenticates credentials and computes group permissions"
            functions = locate_file_symbols(retriever, source, query)
            classes = locate_file_symbols(retriever, source, query, kind="class")
            self.assertTrue(functions)
            self.assertTrue(all(item["kind"] != "class" for item in functions))
            self.assertEqual([(item["name"], item["kind"]) for item in classes],
                             [("ModelBackend", "class")])
            self.assertIn("ModelBackend", [item["name"] for item in
                                          locate_file_symbols(retriever, source, query, kind="all")])
            with self.assertRaisesRegex(ValueError, "kind must be"):
                locate_file_symbols(retriever, source, query, kind="module")
            with self.assertRaisesRegex(ValueError, "kind must be"):
                locate_retrieved_files(retriever, root, [], query, kind="module")

    def test_ranks_named_definition_and_refuses_no_match(self) -> None:
        """Only indexed definitions with positive query evidence are candidates."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.py"
            chunks = split_code_by_lines(source, (
                "class Parser:\n"
                "    def parse_date(self, value):\n"
                "        return value.split('-')\n"
                "    def format_number(self, number):\n"
                "        return str(number)\n"
            ), 3, 1)
            retriever = BM25Retriever(chunks, path_root=root)
            found = locate_file_symbols(retriever, source, "Which method parses a date?", limit=2)
            self.assertEqual(found[0]["name"], "Parser.parse_date")
            self.assertEqual(found[0]["start_line"], 2)
            self.assertEqual(locate_file_symbols(retriever, source, "galactic zirconium"), [])
            with self.assertRaisesRegex(ValueError, "not an indexed Python file"):
                locate_file_symbols(retriever, root / "missing.py", "date")
            with self.assertRaisesRegex(ValueError, "limit between 1 and 20"):
                locate_file_symbols(retriever, source, "date", limit=21)
            self.assertEqual(missing_qualified_definitions(
                retriever, "Locate Parser.parse_missing for dates"), ["Parser.parse_missing"])
            self.assertEqual(missing_qualified_definitions(
                retriever, "Locate Parser.parse_date for dates"), [])
            self.assertTrue(exact_definition_lookup_should_abstain(
                "Locate Parser.parse_missing", ["Parser.parse_missing"]))
            self.assertFalse(exact_definition_lookup_should_abstain(
                "What calls Parser.parse_missing?", ["Parser.parse_missing"]))
            self.assertFalse(exact_definition_lookup_should_abstain(
                "Locate Parser.parse_missing and Parser.parse_date", ["Parser.parse_missing"]))

    def test_retrieved_files_preserve_order_and_reject_escape(self) -> None:
        """Grouping never turns local symbol scores into cross-file confidence."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.py"
            second = root / "second.py"
            chunks = split_code_by_lines(first, "def alpha():\n    return 1\n", 3, 1)
            chunks += split_code_by_lines(second, "def beta():\n    return 2\n", 3, 1)
            retriever = BM25Retriever(chunks, path_root=root)
            groups = locate_retrieved_files(
                retriever, root, [second, first, second], "beta alpha", file_limit=2,
            )
            self.assertEqual([(item["path"], item["search_rank"]) for item in groups],
                             [("second.py", 1), ("first.py", 2)])
            with self.assertRaises(ValueError):
                locate_retrieved_files(retriever, root, [root.parent / "outside.py"], "alpha")


if __name__ == "__main__":
    unittest.main()
