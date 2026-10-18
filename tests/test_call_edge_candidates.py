import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.repository.call_edge_candidates import candidate_call_edges, query_only_call_edges
from coding_rag.repository.call_path import PythonCallPathIndex
from coding_rag.repository.chunks import split_python_files
from coding_rag.repository.files import load_python_files
from coding_rag.tools.bm25 import BM25Retriever


class CallEdgeCandidatesTest(unittest.TestCase):
    def test_query_only_source_skips_tests_and_abstains_on_ambiguity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_entry.py").write_text("def run():\n    return 0\n", encoding="utf-8")
            (root / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8",
            )
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            retriever, call_index = self._indexes(root)
            source, rows, reason = query_only_call_edges(
                retriever, call_index, root, "run calls helper",
                [tests / "test_entry.py", root / "entry.py"],
            )
            self.assertIsNone(reason)
            self.assertEqual(source["source_path"], "entry.py")
            self.assertEqual(source["source_search_rank"], 2)
            self.assertEqual(rows[0]["target_path"], "target.py")

            (root / "entry.py").write_text(
                "if flag:\n    def run():\n        return 1\n"
                "else:\n    def run():\n        return 2\n", encoding="utf-8",
            )
            retriever, call_index = self._indexes(root)
            source, rows, reason = query_only_call_edges(
                retriever, call_index, root, "run", [root / "entry.py"],
            )
            self.assertEqual(source["source_symbol"], "run")
            self.assertEqual(rows, [])
            self.assertEqual(reason, "untraceable_source_candidate")

    def test_returns_resolved_direct_edge_with_source_and_target_lines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entry.py").write_text(
                "from target import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
            )
            (root / "target.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            retriever, call_index = self._indexes(root)
            candidates = candidate_call_edges(
                retriever, call_index, root, "run calls helper", [root / "entry.py"]
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["source_line"], 4)
            self.assertEqual(candidates[0]["target_start_line"], 1)
            self.assertEqual(candidates[0]["target_path"], "target.py")

    def test_rejects_unresolved_attribute_and_ambiguous_definition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("class Settings:\n    pass\n", encoding="utf-8")
            (root / "entry.py").write_text(
                "from config import Settings\n"
                "if flag:\n"
                "    def run():\n        return Settings.STORAGES.copy()\n"
                "else:\n"
                "    def run():\n        return Settings.STORAGES.copy()\n",
                encoding="utf-8",
            )
            (root / "unresolved.py").write_text(
                "from config import Settings\n\ndef run():\n    return Settings.STORAGES.copy()\n",
                encoding="utf-8",
            )
            retriever, call_index = self._indexes(root)
            self.assertEqual(
                candidate_call_edges(retriever, call_index, root, "storage copy", [root / "entry.py"]),
                [],
            )
            self.assertEqual(
                candidate_call_edges(retriever, call_index, root, "storage copy", [root / "unresolved.py"]),
                [],
            )
            with self.assertRaises(ValueError):
                candidate_call_edges(
                    retriever, call_index, root, "storage copy", [root / "unresolved.py"],
                    line_weight=float("nan"),
                )

    def test_explicit_source_symbol_only_ranks_its_outgoing_calls(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entry.py").write_text(
                "from first import one\nfrom second import two\n"
                "\ndef run_one():\n    return one()\n"
                "\ndef run_two():\n    return two()\n",
                encoding="utf-8",
            )
            (root / "first.py").write_text("def one():\n    return 1\n", encoding="utf-8")
            (root / "second.py").write_text("def two():\n    return 2\n", encoding="utf-8")
            retriever, call_index = self._indexes(root)
            rows = candidate_call_edges(
                retriever, call_index, root, "which call reaches one", [root / "entry.py"],
                source_symbol="run_one",
            )
            self.assertEqual([row["target_path"] for row in rows], ["first.py"])
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                candidate_call_edges(
                    retriever, call_index, root, "one", [root / "entry.py"],
                    source_symbol="missing",
                )

    @staticmethod
    def _indexes(root):
        chunks = split_python_files(load_python_files(root))
        retriever = BM25Retriever(chunks, path_root=root)
        return retriever, PythonCallPathIndex(chunks, retriever.repo_index)


if __name__ == "__main__":
    unittest.main()
