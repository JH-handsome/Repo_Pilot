import os
import unittest
from unittest.mock import patch

from coding_rag.llm_client import build_llm_config, chat_completions_url


class LLMConfigTest(unittest.TestCase):
    def test_deepseek_preset_uses_provider_env(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True):
            config = build_llm_config("deepseek")

        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.api_key_env, "DEEPSEEK_API_KEY")

    def test_custom_provider_can_use_generic_env(self):
        env = {
            "LLM_API_KEY": "test-secret",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "local-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_llm_config("custom")

        self.assertEqual(config.base_url, "http://localhost:11434/v1")
        self.assertEqual(config.model, "local-model")
        self.assertEqual(config.api_key_env, "LLM_API_KEY")

    def test_missing_key_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                build_llm_config("qwen")

        self.assertIn("DASHSCOPE_API_KEY", str(context.exception))

    def test_chat_completions_url(self):
        self.assertEqual(
            chat_completions_url("https://example.test/v1"),
            "https://example.test/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
