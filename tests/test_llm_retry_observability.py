import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from coding_rag.rag.llm_client import (
    LLMConfig,
    LLMRequestError,
    LLMRetryPolicy,
    OpenAICompatibleChatClient,
)


class FakeResponse:
    """Minimal context-managed urllib response for deterministic client tests."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def http_error(status: int, detail: str = "provider-detail") -> urllib.error.HTTPError:
    """Build an HTTPError whose response body must never reach logs or public errors."""
    return urllib.error.HTTPError(
        "https://provider.test/v1/chat/completions",
        status,
        "failed",
        hdrs=None,
        fp=io.BytesIO(detail.encode("utf-8")),
    )


def config(policy: LLMRetryPolicy | None = None) -> LLMConfig:
    """Return a test client configuration containing a recognizable secret key."""
    return LLMConfig(
        provider="test-provider",
        base_url="https://provider.test/v1",
        model="test-model",
        api_key="api-key-must-not-log",
        api_key_env="TEST_API_KEY",
        retry_policy=policy or LLMRetryPolicy(base_delay_seconds=0, max_delay_seconds=0, jitter_ratio=0),
    )


class LLMRetryObservabilityTest(unittest.TestCase):
    def test_success_records_one_sanitized_logical_call(self):
        emitted = []
        client = OpenAICompatibleChatClient(config(), observer=emitted.append)
        with patch(
            "coding_rag.rag.llm_client.urllib.request.urlopen",
            return_value=FakeResponse({"choices": [{"message": {"content": "answer"}}]}),
        ):
            answer = client.complete([{"role": "user", "content": "private prompt"}])
        self.assertEqual(answer, "answer")
        self.assertEqual(len(client.observations), 1)
        self.assertEqual(client.last_observation.attempts, 1)
        self.assertEqual(client.last_observation.outcome, "success")
        self.assertIsNone(client.last_observation.error_class)
        self.assertEqual(emitted[0]["provider"], "test-provider")
        self.assertEqual(client.take_call_observation()["outcome"], "success")
        self.assertIsNone(client.take_call_observation())

    def test_transient_failure_retries_then_succeeds_with_testable_backoff(self):
        sleeps = []
        policy = LLMRetryPolicy(
            max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=2, jitter_ratio=0.2
        )
        client = OpenAICompatibleChatClient(
            config(policy), sleep=sleeps.append, random_value=lambda: 0.5
        )
        transient = http_error(429)
        with patch(
            "coding_rag.rag.llm_client.urllib.request.urlopen",
            side_effect=[
                transient,
                FakeResponse({"choices": [{"message": {"content": "recovered"}}]}),
            ],
        ):
            self.assertEqual(client.complete([]), "recovered")
        self.assertTrue(transient.fp.closed)
        self.assertEqual(sleeps, [0.5])
        self.assertEqual(client.last_observation.attempts, 2)
        self.assertEqual(client.last_observation.outcome, "success")

    def test_transient_failure_exhaustion_stops_at_strict_attempt_limit(self):
        sleeps = []
        policy = LLMRetryPolicy(
            max_attempts=3, base_delay_seconds=0.25, max_delay_seconds=1, jitter_ratio=0
        )
        client = OpenAICompatibleChatClient(config(policy), sleep=sleeps.append)
        failures = [http_error(503, "secret-body") for _ in range(3)]
        with patch(
            "coding_rag.rag.llm_client.urllib.request.urlopen",
            side_effect=failures,
        ):
            with self.assertRaises(LLMRequestError) as raised:
                client.complete([])
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertNotIn("secret-body", str(raised.exception))
        self.assertEqual(client.last_observation.attempts, 3)
        self.assertEqual(client.last_observation.error_class, "http_503")
        self.assertTrue(all(error.fp.closed for error in failures))

    def test_non_transient_failure_fails_immediately(self):
        sleeps = []
        client = OpenAICompatibleChatClient(config(), sleep=sleeps.append)
        failure = http_error(400)
        with patch(
            "coding_rag.rag.llm_client.urllib.request.urlopen",
            side_effect=failure,
        ):
            with self.assertRaises(LLMRequestError):
                client.complete([])
        self.assertEqual(sleeps, [])
        self.assertEqual(client.last_observation.attempts, 1)
        self.assertEqual(client.last_observation.error_class, "http_400")
        self.assertTrue(failure.fp.closed)

    def test_observation_log_excludes_prompt_key_cookie_and_provider_error_body(self):
        client = OpenAICompatibleChatClient(config())
        messages = [{"role": "user", "content": "prompt-secret cookie-secret"}]
        with self.assertLogs("coding_rag.rag.llm_client", level="INFO") as captured:
            with patch(
                "coding_rag.rag.llm_client.urllib.request.urlopen",
                side_effect=http_error(401, "response-secret token-secret"),
            ):
                with self.assertRaises(LLMRequestError):
                    client.complete(messages)
        logged = "\n".join(captured.output)
        for secret in (
            "prompt-secret", "cookie-secret", "api-key-must-not-log",
            "response-secret", "token-secret", "TEST_API_KEY",
        ):
            self.assertNotIn(secret, logged)
        self.assertIn('"attempts": 1', logged)
        self.assertIn('"error_class": "http_401"', logged)


if __name__ == "__main__":
    unittest.main()
