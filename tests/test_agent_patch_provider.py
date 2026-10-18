"""Tests for isolated live-Agent patch candidate generation without network access."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_rag.evaluation.agent_patch_provider import (
    AgentCandidateProvider,
    AgentCandidateRejected,
)
from coding_rag.evaluation.patch_outcome import (
    CandidateRequest,
    PatchSuite,
    evaluate_patch_suite,
    load_and_validate_patch_suite,
)
from coding_rag.repository.search_index import RepositoryIndexRegistry
from scripts import patch_outcome_eval as patch_cli


PATCH = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""
OVERREACH = """diff --git a/tests/test_app.py b/tests/test_app.py
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1 +1 @@
-ORIGINAL
+TAMPERED
"""


def decision(action: str, *, tool=None, arguments=None, answer=None) -> str:
    """Build one strict unified-runtime decision JSON response."""
    return json.dumps({
        "action": action,
        "reason": "bounded test decision",
        "answer": answer,
        "tool": tool,
        "arguments": arguments or {},
        "expected_observation": "test evidence" if tool else None,
    })


class SequenceClient:
    """Generic offline ChatClient that records inputs and returns fixed decisions."""

    def __init__(self, responses: list[str]):
        """Store deterministic responses without provider configuration or network access."""
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Record a detached message copy and return the next response."""
        self.messages.append([dict(item) for item in messages])
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


class AgentPatchProviderTest(unittest.TestCase):
    """Exercise graph, approval, isolation, and redaction boundaries offline."""

    def setUp(self) -> None:
        """Create one already-faulted candidate repository."""
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "faulted"
        (self.repo / "tests").mkdir(parents=True)
        (self.repo / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
        (self.repo / "tests" / "test_app.py").write_text("ORIGINAL\n", encoding="utf-8")
        self.request = CandidateRequest(
            "private-task-id",
            "Restore value behavior.",
            ("app.py",),
            ("tests/test_app.py",),
            ("python", "-m", "unittest", "PRIVATE_TEST_TARGET"),
        )

    def provider(self, client: SequenceClient) -> AgentCandidateProvider:
        """Build an offline provider with safe public labels."""
        return AgentCandidateProvider(
            client, provider_label="offline-test", model_label="sequence-v1",
            max_steps=5, max_approvals=1, execution_mode="offline_scripted_contract",
        )

    def test_real_graph_search_read_apply_resume_is_isolated_and_diskless(self):
        client = SequenceClient([
            decision("tool", tool="search_code", arguments={"query": "value", "top_k": 5}),
            decision("tool", tool="read_file", arguments={"path": "app.py"}),
            decision("tool", tool="apply_patch", arguments={"diff": PATCH}),
            decision("answer", answer="Updated the implementation [app.py:1-2]."),
        ])
        provider = self.provider(client)
        with patch(
            "coding_rag.evaluation.agent_patch_provider.RepositoryIndexRegistry",
            wraps=RepositoryIndexRegistry,
        ) as registry_factory:
            candidate = provider.provide(self.request, self.repo)

        self.assertIn("+    return 2", candidate)
        self.assertEqual(self.repo.joinpath("app.py").read_text(encoding="utf-8"),
                         "def value():\n    return 1\n")
        self.assertEqual(self.repo.joinpath("tests/test_app.py").read_text(encoding="utf-8"),
                         "ORIGINAL\n")
        self.assertFalse((self.repo / ".git").exists())
        registry_kwargs = registry_factory.call_args.kwargs
        self.assertFalse(registry_kwargs["disk_cache"])
        self.assertEqual(Path(registry_kwargs["cache_dir"]).name, "index-cache")
        self.assertFalse(Path(registry_kwargs["cache_dir"]).exists())
        first = client.messages[0]
        self.assertEqual(first[-1], {"role": "user", "content": self.request.requirement})
        first_payload = json.dumps(first)
        for private in ("PRIVATE_TEST_TARGET", "tests/test_app.py", "private-task-id"):
            self.assertNotIn(private, first_payload)
        metadata = provider.scorecard_metadata
        self.assertEqual(metadata["approval_count"], 1)
        self.assertEqual(metadata["llm_call_count"], 4)
        serialized = json.dumps(metadata)
        for secret in (self.request.requirement, PATCH, "Authorization", "api_key", "base_url"):
            self.assertNotIn(secret, serialized)

    def test_run_command_is_never_auto_approved(self):
        client = SequenceClient([
            decision(
                "tool", tool="run_command",
                arguments={
                    "cmd": "python -m unittest",
                    "affected_files": ["app.py"],
                },
            ),
        ])
        provider = self.provider(client)
        self.assertEqual(provider.provide(self.request, self.repo), "")
        metadata = provider.scorecard_metadata
        self.assertEqual(metadata["approval_count"], 0)
        self.assertEqual(metadata["rejection_count"], 1)
        self.assertEqual(metadata["terminal_outcomes"], {"approval_required": 1})
        self.assertEqual(self.repo.joinpath("app.py").read_text(encoding="utf-8"),
                         "def value():\n    return 1\n")

    def test_private_scope_rejection_records_the_model_call(self):
        provider = self.provider(SequenceClient([
            decision("tool", tool="apply_patch", arguments={"diff": OVERREACH}),
        ]))
        with self.assertRaises(AgentCandidateRejected):
            provider.provide(self.request, self.repo)
        metadata = provider.scorecard_metadata
        self.assertEqual(metadata["llm_call_count"], 1)
        self.assertEqual(metadata["rejection_count"], 1)
        self.assertEqual(metadata["terminal_outcomes"], {"approval_required": 1})
        self.assertEqual(self.repo.joinpath("tests/test_app.py").read_text(encoding="utf-8"),
                         "ORIGINAL\n")

    def test_real_pinned_django_case_is_an_offline_scripted_contract(self):
        root = Path(__file__).resolve().parents[1]
        checkout = root / "data" / "repos" / "django"
        if not checkout.is_dir():
            self.skipTest("pinned Django checkout not prepared")
        suite = load_and_validate_patch_suite(
            root, root / "datasets" / "eval" / "django_patch_outcome_v1.json",
        )
        task = suite.tasks[0]
        oracle = task.oracle_patch_path.read_text(encoding="utf-8")
        client = SequenceClient([
            decision("tool", tool="read_file", arguments={"path": task.source_path}),
            decision("tool", tool="apply_patch", arguments={"diff": oracle}),
            decision("answer", answer=f"Applied the requested fix [{task.source_path}:1-40]."),
        ])
        provider = AgentCandidateProvider(
            client,
            provider_label="offline-scripted",
            model_label="django-contract-v1",
            max_steps=4,
            max_approvals=1,
            execution_mode="offline_scripted_contract",
        )
        one_case = PatchSuite(
            suite.suite_id + "-offline-contract",
            suite.evaluation_mode,
            suite.manifest_name,
            suite.manifest_sha256,
            suite.repository,
            (task,),
        )
        scorecard = evaluate_patch_suite(one_case, provider)

        self.assertEqual(scorecard["score"], 100.0)
        self.assertEqual(scorecard["metrics"]["test_pass_rate"], 1.0)
        self.assertEqual(scorecard["candidate"]["provider"], "agent_offline_contract_v1")
        self.assertEqual(
            scorecard["candidate"]["runtime"]["mode"], "offline_scripted_contract",
        )
        self.assertEqual(client.messages[0][-1]["content"], task.requirement)
        first_payload = json.dumps(client.messages[0])
        for private in (
            task.test_command[-1], task.fault_patch_path.name,
            task.oracle_patch_path.name, task.fault_patch_sha256, task.oracle_patch_sha256,
        ):
            self.assertNotIn(private, first_payload)

    def test_live_cli_rejects_gate_and_unbounded_tokens_before_client_setup(self):
        with patch.object(patch_cli, "build_llm_config") as build:
            self.assertEqual(patch_cli.main(["--candidate-mode", "agent-live", "--gate"]), 2)
            self.assertEqual(patch_cli.main([
                "--candidate-mode", "agent-live",
                "--allow-local-code-execution", "--allow-network-llm",
                "--accept-provider-charges", "--llm-max-tokens", "8193",
            ]), 2)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
