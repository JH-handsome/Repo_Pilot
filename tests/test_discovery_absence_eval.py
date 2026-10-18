"""Exact-definition absence labels and gates reject self-confirming fixtures."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from coding_rag.repository.index import FileIndex, RepoIndex, SymbolRecord, normalize_path
from scripts.discovery_absence_eval import compare, validate_negative_case


class DiscoveryAbsenceEvalTests(unittest.TestCase):
    def test_negative_label_needs_existing_class_and_missing_direct_method(self) -> None:
        """The evaluator must prove its expected absence from AST records."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "entry.py"
            source.write_text("class Parser:\n    def parse(self):\n        pass\n", encoding="utf-8")
            indexed = FileIndex(path=source, module="entry", directories=(), symbols=(
                SymbolRecord("Parser", "class", 1, 3, "class Parser"),
                SymbolRecord("parse", "function", 2, 3, "def parse(self)", "Parser"),
            ))
            pipeline = SimpleNamespace(
                retriever=SimpleNamespace(repo_index=RepoIndex({normalize_path(source): indexed})),
            )
            row = {"id": "missing", "exact_symbol": "Parser.parse_missing",
                   "query": "Locate Parser.parse_missing", "audit_path": "entry.py"}
            validate_negative_case(row, root, pipeline)
            with self.assertRaisesRegex(ValueError, "exact definition is present"):
                validate_negative_case({**row, "exact_symbol": "Parser.parse",
                                        "query": "Locate Parser.parse"}, root, pipeline)
            with self.assertRaisesRegex(ValueError, "unsafe audit path"):
                validate_negative_case({**row, "audit_path": "../outside.py"}, root, pipeline)

    def test_gate_rejects_identity_drift_and_false_abstention(self) -> None:
        """A matching score on a different dataset cannot pass this gate."""
        baseline = {"schema_version": 1, "benchmark": "absence", "evaluation_mode": "offline",
                    "retriever": "v7", "symbol_locator": "v1", "guard_version": "direct-v1",
                    "dataset": {"sha256": "one"}, "repositories": ["aider"],
                    "metrics": {"negative_count": 8, "positive_control_count": 20,
                                "negative_abstention_rate": 1.0,
                                "negative_missing_warning_rate": 1.0,
                                "positive_false_abstention_rate": 0.0}}
        self.assertEqual(compare(baseline, baseline), [])
        changed = {**baseline, "dataset": {"sha256": "two"}}
        self.assertIn("incompatible dataset", compare(changed, baseline))
        changed = {**baseline, "metrics": {**baseline["metrics"],
                                           "positive_false_abstention_rate": 0.05}}
        self.assertIn("positive_false_abstention_rate regression", compare(changed, baseline))


if __name__ == "__main__":
    unittest.main()
