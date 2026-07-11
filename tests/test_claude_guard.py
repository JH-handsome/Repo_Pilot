import argparse
import unittest

from scripts.run_claude_guarded import (
    ClaudeUsage,
    guard_reason,
    update_usage_from_event,
)


class ClaudeGuardTest(unittest.TestCase):
    def test_updates_usage_from_result_model_usage(self):
        usage = ClaudeUsage()

        update_usage_from_event(
            usage,
            {
                "type": "result",
                "modelUsage": {
                    "claude-sonnet": {
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "cacheReadInputTokens": 300,
                        "cacheCreationInputTokens": 50,
                    }
                },
            },
        )

        self.assertEqual(usage.task_tokens, 470)
        self.assertEqual(usage.cache_miss_tokens, 150)
        self.assertAlmostEqual(usage.cache_miss_ratio, 150 / 450)

    def test_updates_usage_from_stream_message_usage(self):
        usage = ClaudeUsage()

        update_usage_from_event(
            usage,
            {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 2,
                    }
                },
            },
        )

        self.assertEqual(usage.task_tokens, 46)
        self.assertEqual(usage.cache_miss_tokens, 12)

    def test_stops_on_task_token_limit(self):
        args = argparse.Namespace(
            max_task_tokens=100,
            max_cache_miss_tokens=1000,
            max_cache_miss_ratio=1.0,
            min_cache_input_tokens=0,
        )
        usage = ClaudeUsage(input_tokens=80, output_tokens=30)

        self.assertIn("task tokens", guard_reason(args, usage))

    def test_stops_on_cache_miss_token_limit(self):
        args = argparse.Namespace(
            max_task_tokens=1000,
            max_cache_miss_tokens=100,
            max_cache_miss_ratio=1.0,
            min_cache_input_tokens=0,
        )
        usage = ClaudeUsage(input_tokens=80, cache_creation_input_tokens=30)

        self.assertIn("cache miss tokens", guard_reason(args, usage))

    def test_stops_on_cache_miss_ratio_after_minimum(self):
        args = argparse.Namespace(
            max_task_tokens=1000,
            max_cache_miss_tokens=1000,
            max_cache_miss_ratio=0.30,
            min_cache_input_tokens=100,
        )
        usage = ClaudeUsage(input_tokens=40, cache_read_input_tokens=60)

        self.assertIn("cache miss ratio", guard_reason(args, usage))

    def test_does_not_apply_ratio_before_minimum(self):
        args = argparse.Namespace(
            max_task_tokens=1000,
            max_cache_miss_tokens=1000,
            max_cache_miss_ratio=0.30,
            min_cache_input_tokens=100,
        )
        usage = ClaudeUsage(input_tokens=40, cache_read_input_tokens=10)

        self.assertIsNone(guard_reason(args, usage))


if __name__ == "__main__":
    unittest.main()
