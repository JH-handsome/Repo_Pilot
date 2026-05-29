import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_rag.env_loader import ensure_dotenv, load_dotenv, parse_env_line


class EnvLoaderTest(unittest.TestCase):
    def test_parse_env_line_supports_quotes_and_export(self):
        self.assertEqual(parse_env_line("export FOO=\"bar baz\""), ("FOO", "bar baz"))
        self.assertEqual(parse_env_line("EMPTY="), ("EMPTY", ""))
        self.assertIsNone(parse_env_line("# comment"))

    def test_load_dotenv_does_not_override_existing_values_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("DEEPSEEK_API_KEY=file-value\n", encoding="utf-8")

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "shell-value"}, clear=True):
                self.assertTrue(load_dotenv(path))
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "shell-value")

    def test_ensure_dotenv_creates_deepseek_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"

            self.assertTrue(ensure_dotenv(path))
            text = path.read_text(encoding="utf-8")

            self.assertIn("DEEPSEEK_API_KEY=", text)
            self.assertIn("LLM_API_KEY_ENV=DEEPSEEK_API_KEY", text)
            self.assertFalse(ensure_dotenv(path))


if __name__ == "__main__":
    unittest.main()
