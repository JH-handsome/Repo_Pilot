from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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

    def test_old_workflow_flags_are_rejected(self):
        old_flags = [
            "--workflow-mode",
            "--agent",
            "--agent-exec",
            "--llm",
            "--dry-run",
            "--execute-tools",
            "--safe-mode",
            "--legacy-judge",
        ]
        for flag in old_flags:
            argv = ["main.py", ".", "问题", flag]
            if flag == "--workflow-mode":
                argv.append("ask")
            with self.subTest(flag=flag), patch.object(sys, "argv", argv), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parse_args()

    def test_single_turn_uses_unified_runtime(self):
        with patch.object(sys, "argv", ["main.py", ".", "解释 target"]):
            args = cli.parse_args()
        fake_run = build_fake_run(answer="target 位于 app.py:1-2")

        with patch.object(cli, "build_chat_client", return_value=object()):
            with patch.object(cli, "run_unified_query", return_value=fake_run) as run_unified:
                output = StringIO()
                with redirect_stdout(output):
                    status = cli.run_single_turn(args)

        self.assertEqual(status, 0)
        self.assertIn("target 位于", output.getvalue())
        self.assertEqual(run_unified.call_args.args[0], "解释 target")
        self.assertNotIn("execute_tools", run_unified.call_args.kwargs)

    def test_single_turn_approves_one_pending_write(self):
        with patch.object(sys, "argv", ["main.py", ".", "修复问题"]):
            args = cli.parse_args()
        pending = build_fake_run(status="approval_required")
        completed = build_fake_run(execution_enabled=True)

        with patch.object(cli, "build_chat_client", return_value=object()):
            with patch.object(cli, "run_unified_query", return_value=pending):
                with patch.object(cli, "resume_unified_query", return_value=completed) as resume:
                    with patch("builtins.input", return_value="y"), redirect_stdout(StringIO()):
                        status = cli.run_single_turn(args)

        self.assertEqual(status, 0)
        self.assertEqual(resume.call_args.kwargs["approval_fingerprint"], "approved-call")

    def test_single_turn_denies_pending_write_without_resume(self):
        with patch.object(sys, "argv", ["main.py", ".", "修复问题"]):
            args = cli.parse_args()
        pending = build_fake_run(status="approval_required")

        with patch.object(cli, "build_chat_client", return_value=object()):
            with patch.object(cli, "run_unified_query", return_value=pending):
                with patch.object(cli, "resume_unified_query") as resume:
                    with patch("builtins.input", return_value="n"), redirect_stdout(StringIO()):
                        status = cli.run_single_turn(args)

        self.assertEqual(status, 4)
        resume.assert_not_called()

    def test_interactive_cli_can_exit_immediately(self):
        with patch.object(sys, "argv", ["main.py", "."]):
            args = cli.parse_args()
        output = StringIO()

        with patch.object(cli, "build_chat_client", return_value=object()):
            with patch("builtins.input", return_value=":q"), redirect_stdout(output):
                status = cli.run_interactive_cli(args)

        self.assertEqual(status, 0)
        self.assertIn("RepoPilot 交互模式", output.getvalue())
        self.assertIn("逐次申请批准", output.getvalue())

    def test_interactive_cli_runs_one_unified_turn(self):
        with patch.object(sys, "argv", ["main.py", "."]):
            args = cli.parse_args()

        with patch.object(cli, "build_chat_client", return_value=object()):
            with patch.object(cli, "run_single_turn", return_value=0) as run_turn:
                with patch("builtins.input", side_effect=["解释 target", ":q"]), redirect_stdout(StringIO()):
                    status = cli.run_interactive_cli(args)

        self.assertEqual(status, 0)
        self.assertEqual(run_turn.call_args.args[0].query, "解释 target")

    def test_trace_out_writes_unified_shape_at_top_level(self):
        with TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            with patch.object(
                sys,
                "argv",
                ["main.py", ".", "解释 target", "--trace-out", str(trace_path)],
            ):
                args = cli.parse_args()
            fake_run = build_fake_run()
            with patch.object(cli, "build_chat_client", return_value=object()):
                with patch.object(cli, "run_unified_query", return_value=fake_run):
                    with redirect_stdout(StringIO()):
                        status = cli.run_single_turn(args)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(trace["trace_version"], "1.0")
        self.assertEqual(trace["run"]["mode"], "unified")

    def test_rendered_success_shows_snapshot_and_operation_diff(self):
        run = build_fake_run(execution_enabled=True)
        run.trace["events"] = [
            {
                "step": "apply_patch",
                "artifacts": {
                    "patch": {
                        "snapshot_id": "a" * 24,
                        "post_change_diff": {"text": "-value = 1\n+value = 2\n"},
                    }
                },
            }
        ]

        rendered = cli.render_unified_run(run)

        self.assertIn("回滚快照", rendered)
        self.assertIn("修改后 diff", rendered)
        self.assertIn("+value = 2", rendered)

    def test_rendered_approval_shows_files_and_exact_command(self):
        run = build_fake_run(status="approval_required")
        run.approval = {
            "tool": "run_command",
            "files": ["artifacts/result.txt"],
            "fingerprint": "approved-call",
            "arguments": {
                "cmd": "python -m unittest",
                "affected_files": ["artifacts/result.txt"],
            },
        }

        rendered = cli.render_unified_run(run)

        self.assertIn("artifacts/result.txt", rendered)
        self.assertIn("python -m unittest", rendered)


def build_fake_run(answer="完成", execution_enabled=False, status="success"):
    trace = {
        "trace_version": "1.0",
        "run": {"mode": "unified", "status": "success", "summary": {}},
        "events": [],
        "artifacts": {"retrieval": {"searches": []}, "agent": {}, "tools": []},
        "summary": {"llm_call_count": 1, "tool_call_count": 0, "observation_count": 0},
    }
    return SimpleNamespace(
        status=status,
        answer=answer,
        trace=trace,
        execution_enabled=execution_enabled,
        approval=(
            {"tool": "apply_patch", "files": ["app.py"], "fingerprint": "approved-call"}
            if status == "approval_required"
            else None
        ),
        observations=[],
    )


if __name__ == "__main__":
    unittest.main()
