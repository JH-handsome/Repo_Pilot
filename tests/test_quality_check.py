"""Offline aggregate gate runner must not imply an LLM repair score."""

from contextlib import redirect_stderr
from io import StringIO
from subprocess import CompletedProcess
import unittest
from unittest.mock import patch

from scripts.quality_check import gate_specs, run_quality_check, summarize


class QualityCheckTest(unittest.TestCase):
    def test_extended_suite_includes_aider_and_only_fixed_offline_commands(self):
        specs = gate_specs(core_only=False)
        self.assertEqual(len(specs), 7)
        self.assertEqual(len(gate_specs(core_only=True)), 5)
        self.assertEqual(specs[0][0], "unit_tests")
        self.assertEqual(specs[-2][0], "aider_django_multistep")
        self.assertEqual(specs[-1][0], "explicit_absence")
        commands = " ".join(" ".join(argv) for _, _, argv in specs)
        for unsafe in ("--allow-network-llm", "--accept-provider-charges", "agent-live"):
            self.assertNotIn(unsafe, commands)
        self.assertIn("--candidate-mode oracle", commands)

    def test_oracle_score_requires_explicit_self_check_label(self):
        payload = {
            "benchmark": "patch_outcome_harness_not_llm_intelligence",
            "candidate": {"provider": "oracle_self_check_v1"},
            "score": 100, "metrics": {"case_count": 3},
        }
        self.assertEqual(
            summarize("patch_outcome", payload),
            {"case_count": 3, "harness_score": 100},
        )
        payload["candidate"]["provider"] = "agent-live"
        with self.assertRaisesRegex(ValueError, "self-check"):
            summarize("patch_outcome", payload)

    def test_failed_gate_is_reported_without_masking_other_checks(self):
        with patch(
            "scripts.quality_check.subprocess.run",
            return_value=CompletedProcess([], 1, stdout="gate failed", stderr=""),
        ) as run:
            with redirect_stderr(StringIO()):
                report = run_quality_check(core_only=True)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["offline_only"])
        self.assertEqual(len(report["checks"]), 5)
        self.assertEqual(run.call_count, 5)
        self.assertTrue(all(item["status"] == "fail" for item in report["checks"]))
        self.assertTrue(all(item["error"] == "gate failed" for item in report["checks"]))

    def test_missing_scorecard_fails_closed_and_unit_count_is_reported(self):
        def run(command, **kwargs):
            if "unittest" in command:
                return CompletedProcess(command, 0, stdout="", stderr="Ran 456 tests in 1.0s\nOK")
            return CompletedProcess(command, 0, stdout="PASS", stderr="")

        with patch("scripts.quality_check.subprocess.run", side_effect=run):
            with redirect_stderr(StringIO()):
                report = run_quality_check(core_only=True)
        self.assertEqual(report["checks"][0]["metrics"]["test_count"], 456)
        self.assertEqual(report["checks"][1]["status"], "fail")
        self.assertIn("missing or invalid scorecard", report["checks"][1]["error"])


if __name__ == "__main__":
    unittest.main()
