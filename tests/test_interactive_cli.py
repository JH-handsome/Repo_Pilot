from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import unittest

import main as cli


class InteractiveCliTest(unittest.TestCase):
    def test_parse_args_without_positionals_enters_interactive_shape(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = cli.parse_args()

        self.assertIsNone(args.repo_path)
        self.assertIsNone(args.query)

    def test_parse_agent_exec_dry_run_flags(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", ".", "修复 ASK 模式没有调用大模型回答的问题", "--agent-exec", "--dry-run"],
        ):
            args = cli.parse_args()

        self.assertTrue(args.agent_exec)
        self.assertTrue(args.dry_run)
        self.assertFalse(args.safe_mode)

    def test_parse_agent_policy_flag(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", ".", "修复问题", "--agent-exec", "--agent-policy", ".repopilot/policy.json"],
        ):
            args = cli.parse_args()

        self.assertEqual(args.agent_policy, ".repopilot/policy.json")

    def test_agent_exec_routes_to_code_agent(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", ".", "修复 ASK 模式没有调用大模型回答的问题", "--agent-exec", "--dry-run"],
        ):
            args = cli.parse_args()

        with patch.object(cli, "run_agent_cli", return_value=0) as run_agent:
            status = cli.run_single_turn(args)

        self.assertEqual(status, 0)
        run_agent.assert_called_once_with(args)

    def test_agent_exec_dry_run_llm_does_not_create_dotenv(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", ".", "修复 ASK 模式没有调用大模型回答的问题", "--agent-exec", "--dry-run", "--llm"],
        ):
            with patch.object(cli, "ensure_dotenv") as ensure_dotenv:
                with patch.object(cli, "load_dotenv") as load_dotenv:
                    with patch.object(cli, "run_agent_cli", return_value=0):
                        status = cli.main()

        self.assertEqual(status, 0)
        ensure_dotenv.assert_not_called()
        load_dotenv.assert_called_once()

    def test_interactive_cli_can_exit_immediately(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(sys, "argv", ["main.py", temp_dir]):
                args = cli.parse_args()
            output = StringIO()

            with patch("builtins.input", side_effect=["n", ":q"]), redirect_stdout(output):
                status = cli.run_interactive_cli(args)

        self.assertEqual(status, 0)
        self.assertIn("RepoPilot interactive mode", output.getvalue())
        self.assertIn("LLM answers: off", output.getvalue())

    def test_interactive_cli_runs_one_ask_turn(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )
            with patch.object(sys, "argv", ["main.py", temp_dir, "--top-k", "1", "--recall-window", "0"]):
                args = cli.parse_args()
            output = StringIO()

            with patch("builtins.input", side_effect=["n", "load_data 在哪里读取文件？", ":q"]), redirect_stdout(output):
                status = cli.run_interactive_cli(args)

        self.assertEqual(status, 0)
        self.assertIn("load_data", output.getvalue())

    def test_ask_cli_trace_out_uses_unified_shape(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )
            trace_path = root / "trace.json"
            with patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    str(root),
                    "load_data 在哪里读取文件？",
                    "--top-k",
                    "1",
                    "--recall-window",
                    "0",
                    "--trace-out",
                    str(trace_path),
                ],
            ):
                args = cli.parse_args()
            output = StringIO()

            with redirect_stdout(output):
                status = cli.run_single_turn(args)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(trace["trace"]["trace_version"], "1.0")
        self.assertIn("events", trace["trace"])

    def test_prompt_yes_no_accepts_yes(self):
        with patch("builtins.input", return_value="y"):
            self.assertTrue(cli.prompt_yes_no("Call LLM"))


if __name__ == "__main__":
    unittest.main()
