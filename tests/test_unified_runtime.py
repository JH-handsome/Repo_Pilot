import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query
from coding_rag.rag.trace import render_trace_report


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def decision(**payload):
    base = {
        "action": "answer",
        "reason": "done",
        "answer": "完成",
        "tool": None,
        "arguments": {},
        "expected_observation": None,
    }
    base.update(payload)
    return json.dumps(base, ensure_ascii=False)


class UnifiedRuntimeTest(unittest.TestCase):
    def test_direct_answer_does_not_call_tools(self):
        client = SequenceClient([decision(answer="你好")])

        run = run_unified_query("你好", UnifiedRunConfig(), client)

        self.assertEqual(run.status, "success")
        self.assertEqual(run.answer, "你好")
        self.assertEqual(run.observations, [])
        self.assertEqual([event["step"] for event in run.trace["events"]], ["model_decision", "final_answer"])
        self.assertIn("model_decision", render_trace_report(run.trace))

    def test_search_then_answer_preserves_hybrid_trace_and_citations(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def target():\n    return True\n",
                encoding="utf-8",
            )
            client = SequenceClient(
                [
                    decision(
                        action="tool",
                        answer=None,
                        tool="search_code",
                        arguments={"query": "target", "top_k": 1},
                        expected_observation="找到定义",
                    ),
                    decision(answer="定义位于 app.py:1-2。"),
                ]
            )

            run = run_unified_query(
                "target 在哪里定义？",
                UnifiedRunConfig(repo_path=root, top_k=1, recall_window=0),
                client,
            )

        self.assertEqual(run.status, "success")
        self.assertIn("app.py:1-2", run.answer)
        self.assertEqual(len(run.trace["artifacts"]["retrieval"]["searches"]), 1)
        self.assertEqual(
            [event["step"] for event in run.trace["events"]],
            ["model_decision", "search_code", "model_decision", "final_answer"],
        )
        self.assertFalse(run.trace["events"][-1]["output_summary"]["citation_issues"])

    def test_patch_is_dry_run_by_default(self):
        patch_text = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            client = SequenceClient(
                [
                    decision(
                        action="tool",
                        answer=None,
                        tool="apply_patch",
                        arguments={"diff": patch_text},
                        expected_observation="补丁通过检查",
                    ),
                    decision(answer="补丁已通过 dry-run 检查。"),
                ]
            )

            run = run_unified_query("修改 value", UnifiedRunConfig(repo_path=root), client)

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(run.status, "dry_run")
        patch_event = next(event for event in run.trace["events"] if event["step"] == "apply_patch")
        self.assertFalse(patch_event["artifacts"]["patch"]["applied"])

    def test_patch_can_apply_when_execution_is_enabled(self):
        patch_text = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            client = SequenceClient(
                [
                    decision(
                        action="tool",
                        answer=None,
                        tool="apply_patch",
                        arguments={"diff": patch_text},
                    ),
                    decision(answer="修改已应用。"),
                ]
            )

            run = run_unified_query(
                "修改 value",
                UnifiedRunConfig(repo_path=root),
                client,
                execute_tools=True,
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
        self.assertEqual(run.status, "success")
        self.assertTrue(run.execution_enabled)

    def test_safety_rejection_is_recorded_and_can_recover_to_answer(self):
        client = SequenceClient(
            [
                decision(
                    action="tool",
                    answer=None,
                    tool="run_command",
                    arguments={"cmd": 'python -c "print(1)"'},
                ),
                decision(answer="该命令被安全策略拒绝。"),
            ]
        )

        run = run_unified_query("运行危险命令", UnifiedRunConfig(), client, execute_tools=True)

        self.assertEqual(run.status, "partial")
        command_event = next(event for event in run.trace["events"] if event["step"] == "run_command")
        self.assertEqual(command_event["status"], "failed")
        self.assertFalse(command_event["artifacts"]["safety"]["allowed"])

    def test_invalid_json_gets_one_repair(self):
        client = SequenceClient(["not json", decision(answer="修复后的回答")])

        run = run_unified_query("hello", UnifiedRunConfig(), client)

        self.assertEqual(run.status, "success")
        self.assertEqual(run.trace["events"][0]["status"], "failed")
        self.assertEqual(run.trace["summary"]["llm_call_count"], 2)

    def test_unknown_tool_gets_one_repair(self):
        client = SequenceClient(
            [
                decision(action="tool", answer=None, tool="delete_everything"),
                decision(answer="改用允许的流程。"),
            ]
        )

        run = run_unified_query("test", UnifiedRunConfig(), client)

        self.assertEqual(run.status, "success")
        self.assertIn("unknown tool", run.trace["events"][0]["error"]["message"])

    def test_tool_step_limit_returns_partial(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("value = 1\n", encoding="utf-8")
            tool_call = decision(
                action="tool",
                answer=None,
                tool="search_code",
                arguments={"query": "value", "top_k": 1},
            )
            client = SequenceClient([tool_call, tool_call])

            run = run_unified_query(
                "find value",
                UnifiedRunConfig(repo_path=root, top_k=1, recall_window=0, max_steps=1),
                client,
            )

        self.assertEqual(run.status, "partial")
        self.assertEqual(run.trace["events"][-1]["step"], "tool_limit")


if __name__ == "__main__":
    unittest.main()
