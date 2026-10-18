"""Tests for the real-source patch outcome harness and its safety boundaries."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_rag.evaluation.patch_outcome import (
    CandidateRequest,
    MAX_OUTPUT_BYTES,
    OraclePatchProvider,
    PatchRepository,
    PatchTask,
    StaticPatchProvider,
    _run,
    _test_environment,
    compare_patch_scorecards,
    evaluate_patch_task,
    gate_exit_code,
    load_and_validate_patch_suite,
    validate_candidate_patch_paths,
    validate_patch_gate_preflight,
    validate_patch_gate_policy,
)


FAULT = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def value():
-    return 2
+    return 1
"""
ORACLE = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""
WRONG = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 0
"""
OVERREACH = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-fixture
+tampered
"""


class RecordingProvider:
    name = "recording"

    def __init__(self, patch_text: str):
        self.patch_text = patch_text
        self.requests = []

    def provide(self, request, repo_path):
        self.requests.append(request)
        return self.patch_text


class PatchOutcomeTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.template = root / "template"
        (self.template / "tests").mkdir(parents=True)
        (self.template / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
        (self.template / "README.md").write_text("fixture\n", encoding="utf-8")
        (self.template / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (self.template / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import value\n"
            "class AppTest(unittest.TestCase):\n"
            "    def test_value(self): self.assertEqual(value(), 2)\n",
            encoding="utf-8",
        )
        self.fault = root / "fault.patch"
        self.oracle = root / "oracle.patch"
        self.fault.write_text(FAULT, encoding="utf-8")
        self.oracle.write_text(ORACLE, encoding="utf-8")
        self.repository = PatchRepository(
            "fixture", "fixture", root, "example.invalid/repo", "0" * 40,
            "Test", "https://example.invalid/license", ("app.py", "tests"),
        )
        self.task = PatchTask(
            "fix-value", "Restore value().", ("app.py",), ("README.md", "tests/test_app.py"),
            ("python", "-m", "unittest", "tests.test_app"), 5,
            self.fault, "0" * 64, self.oracle, "0" * 64,
            "app.py", "0" * 64, "tests/test_app.py", "0" * 64,
        )

    def materialize(self, repository, destination):
        shutil.copytree(self.template, destination)

    def evaluate(self, provider, task=None):
        with patch(
            "coding_rag.evaluation.patch_outcome.materialize_pinned_archive",
            side_effect=self.materialize,
        ):
            return evaluate_patch_task(self.repository, task or self.task, provider)

    def test_oracle_is_full_score_and_candidate_request_has_no_oracle_fields(self):
        provider = RecordingProvider(ORACLE)
        row = self.evaluate(provider)
        self.assertEqual(row["score"], 100.0)
        self.assertTrue(row["pristine_pass"] and row["initial_failure_proven"])
        self.assertTrue(row["patch_applied"] and row["tests_pass"])
        self.assertEqual(tuple(CandidateRequest.__dataclass_fields__), (
            "task_id", "requirement", "allowed_paths", "forbidden_paths", "test_command",
        ))
        self.assertEqual(len(provider.requests), 1)

    def test_noop_wrong_and_overreach_drop_without_touching_forbidden_file(self):
        noop = self.evaluate(StaticPatchProvider({}))
        wrong = self.evaluate(StaticPatchProvider({self.task.id: WRONG}))
        overreach = self.evaluate(StaticPatchProvider({self.task.id: OVERREACH}))
        self.assertLessEqual(noop["score"], 10.0)
        self.assertLessEqual(wrong["score"], 25.0)
        self.assertLessEqual(overreach["score"], 10.0)
        self.assertFalse(overreach["patch_applied"])
        self.assertIn("not allowed", overreach["rejection"])
        self.assertTrue(overreach["forbidden_untouched"])

    def test_invalid_harness_never_calls_provider(self):
        harmless_fault = self.fault.parent / "harmless.patch"
        harmless_fault.write_text(
            FAULT.replace("-    return 2\n+    return 1", "-    return 2\n+    return 2  # unchanged behavior"),
            encoding="utf-8",
        )
        task = copy.copy(self.task)
        object.__setattr__(task, "fault_patch_path", harmless_fault)
        provider = RecordingProvider(ORACLE)
        row = self.evaluate(provider, task)
        self.assertFalse(row["harness_valid"])
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(provider.requests, [])

    def test_patch_path_parser_rejects_delete_rename_traversal_and_mismatches(self):
        self.assertIsNone(validate_candidate_patch_paths(ORACLE, ("app.py",))[1])
        samples = (
            OVERREACH,
            ORACLE.replace("a/app.py b/app.py", "a/../app.py b/../app.py"),
            ORACLE.replace("--- a/app.py", "--- a/other.py"),
            ORACLE.replace("--- a/app.py", "deleted file mode 100644\n--- a/app.py"),
            ORACLE.replace("--- a/app.py", "rename from app.py\nrename to other.py\n--- a/app.py"),
            ORACLE.replace("a/app.py b/app.py", "C:/app.py C:/app.py"),
        )
        for candidate in samples:
            with self.subTest(candidate=candidate[:40]):
                self.assertIsNotNone(validate_candidate_patch_paths(candidate, ("app.py",))[1])

    def test_subprocess_output_timeout_and_environment_are_bounded(self):
        result = _run(
            [sys.executable, "-c", f"print('x' * {MAX_OUTPUT_BYTES * 2})"],
            cwd=Path(self.temp.name), timeout=5,
        )
        self.assertTrue(result["output_truncated"])
        self.assertGreater(result["output_bytes"], MAX_OUTPUT_BYTES)
        self.assertLess(len(result["output"]), MAX_OUTPUT_BYTES + 100)
        timed = _run(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            cwd=Path(self.temp.name), timeout=1,
        )
        self.assertTrue(timed["timed_out"])
        guard = Path(self.temp.name) / "guard"
        repo = Path(self.temp.name) / "repo-env"
        repo.mkdir()
        with patch.dict(os.environ, {
            "API_TOKEN": "secret", "COOKIE": "secret", "AUTH_HEADER": "secret",
            "PASSWORD": "secret", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows"),
        }, clear=True):
            env = _test_environment(guard, repo)
        serialized = json.dumps(env)
        for name in ("API_TOKEN", "COOKIE", "AUTH_HEADER", "PASSWORD", "secret"):
            self.assertNotIn(name, serialized)

    def test_gate_exit_semantics_and_strict_policy(self):
        scorecard = {
            "schema_version": 1, "benchmark": "patch_outcome_harness_not_llm_intelligence",
            "evaluator_version": "v", "dataset": {"sha256": "x", "suite_id": "s", "case_count": 1},
            "repository": {"commit": "c"}, "task_repositories": [{"id": "t"}],
            "candidate": {"provider": "oracle"}, "config": {"shell": False}, "score": 100,
            "metrics": {
                "harness_valid_rate": 1.0, "pristine_pass_rate": 1.0,
                "initial_failure_rate": 1.0, "patch_applied_rate": 1.0,
                "test_pass_rate": 1.0, "allowed_path_precision": 1.0,
                "forbidden_untouched_rate": 1.0, "minimality_score": 1.0,
                "timeout_rate": 0.0, "changed_files_avg": 1.0,
            },
        }
        policy = {
            "max_drop": {
                "score": 0, "metrics.harness_valid_rate": 0,
                "metrics.pristine_pass_rate": 0, "metrics.initial_failure_rate": 0,
                "metrics.patch_applied_rate": 0, "metrics.test_pass_rate": 0,
                "metrics.allowed_path_precision": 0,
                "metrics.forbidden_untouched_rate": 0, "metrics.minimality_score": 0,
            },
            "max_increase": {"metrics.timeout_rate": 0, "metrics.changed_files_avg": 0},
        }
        self.assertEqual(gate_exit_code(compare_patch_scorecards(scorecard, scorecard, policy)), 0)
        regressed = copy.deepcopy(scorecard)
        regressed["candidate"] = {"provider": "static"}
        regressed["metrics"]["test_pass_rate"] = 0.0
        self.assertEqual(gate_exit_code(compare_patch_scorecards(regressed, scorecard, policy)), 1)
        incompatible = copy.deepcopy(scorecard)
        incompatible["evaluator_version"] = "changed"
        self.assertEqual(gate_exit_code(compare_patch_scorecards(incompatible, scorecard, policy)), 2)
        for invalid in (
            {}, {"max_drop": {}, "max_increase": {}},
            {"max_drop": {}, "max_increase": {}, "extra": {}},
            {"max_drop": {"score": -1}, "max_increase": {}},
            {"max_drop": {"score": float("inf")}, "max_increase": {}},
            {"max_drop": {"unknown": 0}, "max_increase": {}},
        ):
            with self.assertRaises(ValueError):
                validate_patch_gate_policy(invalid)

    def test_real_manifest_validates_pinned_source_when_checkout_is_available(self):
        root = Path(__file__).resolve().parents[1]
        checkout = root / "data" / "repos" / "django"
        if not checkout.is_dir():
            self.skipTest("pinned Django checkout not prepared")
        legacy_suite = load_and_validate_patch_suite(
            root, root / "datasets" / "eval" / "django_patch_outcome_v1.json",
        )
        suite = load_and_validate_patch_suite(
            root, root / "datasets" / "eval" / "django_patch_outcome_v2.json",
        )
        self.assertEqual(suite.repository.commit, "09fdc06346b167ff6231a4cb80b85ae67b53c715")
        self.assertEqual(len(legacy_suite.tasks), 2)
        self.assertEqual(len(suite.tasks), 3)
        self.assertEqual(suite.tasks[-1].id, "termcolors-background-options")
        legacy_baseline = json.loads((root / "benchmarks" / "patch-outcome-baseline-v1.json").read_text(encoding="utf-8"))
        current_baseline = json.loads((root / "benchmarks" / "patch-outcome-baseline-v2.json").read_text(encoding="utf-8"))
        policy = json.loads((root / "benchmarks" / "patch-outcome-gate-v2.json").read_text(encoding="utf-8"))
        validate_patch_gate_preflight(suite, current_baseline, policy)
        with self.assertRaisesRegex(ValueError, "incompatible patch outcome baseline"):
            validate_patch_gate_preflight(suite, legacy_baseline, policy)


if __name__ == "__main__":
    unittest.main()
