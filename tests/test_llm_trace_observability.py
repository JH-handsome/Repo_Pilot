"""End-to-end tests for sanitized LLM evidence in unified traces and SQLite."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query
from coding_rag.rag.llm_client import (
    LLMConfig,
    LLMRequestError,
    LLMRetryPolicy,
    OpenAICompatibleChatClient,
)
from coding_rag.storage.sqlite import SQLiteTraceStore
from tests.test_unified_runtime import SequenceClient, decision


def client() -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(LLMConfig(
        provider="test-provider",
        base_url="https://provider.invalid/v1",
        model="test-model",
        api_key="never-persist-this-api-key",
        api_key_env="TEST_API_KEY",
        retry_policy=LLMRetryPolicy(
            max_attempts=3, base_delay_seconds=0, max_delay_seconds=0, jitter_ratio=0,
        ),
    ))


def response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class LLMTraceObservabilityTest(unittest.TestCase):
    def run_with(self, chat_client, *, db: Path | None = None):
        return run_unified_query(
            "private prompt cookie=session-secret token=token-secret",
            UnifiedRunConfig(trace_db_path=db),
            chat_client,
        )

    def test_success_and_sqlite_roundtrip_are_sanitized(self):
        with TemporaryDirectory() as folder:
            db = Path(folder) / "traces.sqlite3"
            observed = client()
            with patch.object(observed, "_request_once", return_value=response(decision(answer="ok"))):
                run = self.run_with(observed, db=db)
            event = run.trace["events"][0]
            self.assertEqual(event["artifacts"]["llm_call"]["outcome"], "success")
            self.assertEqual(run.trace["summary"]["llm_observed_call_count"], 1)
            rows = SQLiteTraceStore(db).get_llm_calls(run.trace["run"]["run_id"])
            summary = SQLiteTraceStore(db).get_llm_summary(run.trace["run"]["run_id"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(summary["call_count"], 1)
            self.assertEqual(summary["attempt_count"], 1)
            serialized = json.dumps({
                "event_observation": event["artifacts"]["llm_call"], "rows": rows,
            })
            for secret in ("private prompt", "session-secret", "token-secret", "never-persist"):
                self.assertNotIn(secret, serialized)

    def test_transient_retry_is_one_logical_call_with_two_attempts(self):
        observed = client()
        temporary = LLMRequestError("safe", error_class="http_429", transient=True)
        with patch.object(
            observed, "_request_once",
            side_effect=[temporary, response(decision(answer="recovered"))],
        ):
            run = self.run_with(observed)
        item = run.trace["events"][0]["artifacts"]["llm_call"]
        self.assertEqual(item["attempts"], 2)
        self.assertEqual(run.trace["summary"]["llm_call_count"], 1)
        self.assertEqual(run.trace["summary"]["llm_retry_count"], 1)

    def test_permanent_failure_is_attached_to_failed_model_event(self):
        with TemporaryDirectory() as folder:
            db = Path(folder) / "traces.sqlite3"
            observed = client()
            failure = LLMRequestError("safe", error_class="http_401", transient=False)
            with patch.object(observed, "_request_once", side_effect=failure):
                run = self.run_with(observed, db=db)
            self.assertEqual(run.status, "failed")
            item = run.trace["events"][0]["artifacts"]["llm_call"]
            self.assertEqual(item["outcome"], "error")
            self.assertEqual(item["error_class"], "http_401")
            self.assertEqual(run.trace["summary"]["llm_error_classes"], {"http_401": 1})
            stored = SQLiteTraceStore(db).get_llm_calls(run.trace["run"]["run_id"])
            self.assertEqual(stored[0]["outcome"], "error")
            self.assertEqual(stored[0]["error_class"], "http_401")
            stored_summary = SQLiteTraceStore(db).get_llm_summary(
                run.trace["run"]["run_id"]
            )
            self.assertEqual(stored_summary["outcomes"], {"error": 1})
            self.assertEqual(stored_summary["error_classes"], {"http_401": 1})
            llm_evidence = json.dumps({
                "trace": item,
                "calls": stored,
                "summary": stored_summary,
            })
            for forbidden in (
                "private prompt", "session-secret", "token-secret",
                "never-persist-this-api-key", "prompt", "response", "body", "cookie",
            ):
                self.assertNotIn(forbidden, llm_evidence)

    def test_repair_calls_have_distinct_observations_and_no_stale_value(self):
        observed = client()
        with patch.object(
            observed, "_request_once",
            side_effect=[response("not-json"), response(decision(answer="fixed"))],
        ):
            run = self.run_with(observed)
        calls = [event["artifacts"]["llm_call"] for event in run.trace["events"]
                 if event["step"] == "model_decision"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(run.trace["summary"]["llm_observed_call_count"], 2)

    def test_generic_client_remains_compatible_without_observation(self):
        run = self.run_with(SequenceClient([decision(answer="generic")]))
        self.assertEqual(run.status, "success")
        self.assertNotIn("llm_call", run.trace["events"][0]["artifacts"])
        self.assertEqual(run.trace["summary"]["llm_observed_call_count"], 0)

    def test_invalid_optional_observation_is_ignored_by_runtime(self):
        class InvalidObservedClient(SequenceClient):
            def take_call_observation(self):
                return {"prompt": "must-not-enter-trace"}

        run = self.run_with(InvalidObservedClient([decision(answer="ok")]))
        self.assertNotIn("llm_call", run.trace["events"][0]["artifacts"])

    def test_broken_optional_observer_cannot_break_successful_run(self):
        class BrokenObservedClient(SequenceClient):
            def take_call_observation(self):
                raise RuntimeError("observer bug")

        run = self.run_with(BrokenObservedClient([decision(answer="still works")]))
        self.assertEqual(run.status, "success")
        self.assertEqual(run.answer, "still works")
        self.assertNotIn("llm_call", run.trace["events"][0]["artifacts"])

    def test_database_failure_does_not_retry_or_change_model_result(self):
        observed = client()
        with TemporaryDirectory() as folder, patch.object(
            observed, "_request_once", return_value=response(decision(answer="ok"))
        ), patch(
            "coding_rag.storage.persistence.SQLiteTraceStore.save_trace",
            side_effect=sqlite3.OperationalError,
        ):
            run = self.run_with(observed, db=Path(folder) / "not-written.sqlite3")
        self.assertEqual(run.status, "success")
        self.assertEqual(run.answer, "ok")
        self.assertEqual(len(observed.observations), 1)
        self.assertEqual(run.storage["status"], "failed")


if __name__ == "__main__":
    unittest.main()
