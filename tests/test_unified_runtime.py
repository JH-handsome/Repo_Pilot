import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent.runtime import (
    UnifiedRunConfig, build_unified_messages, compact_tool_result, resume_unified_query, run_unified_query,
)
from coding_rag.agent.snapshots import SnapshotStore
from coding_rag.agent.safety import AgentSafetyPolicy
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
    def test_unified_prompt_infers_intent_without_keyword_routing(self):
        messages = build_unified_messages("解释这个模块", UnifiedRunConfig())
        system = messages[0]["content"]
        self.assertIn("根据用户的完整请求和当前工具观察判断意图", system)
        self.assertIn("问答或定位只读取证据", system)
        self.assertIn("写工具总是按调用逐次申请批准", system)

    def test_tool_compaction_never_exceeds_character_budget(self):
        payload = {"output": ('\\"' * 3000) + ("数据" * 3000)}
        for budget in (1, 2, 20, 80, 300):
            with self.subTest(budget=budget):
                compact, metadata = compact_tool_result("run_command", payload, max_chars=budget)
                actual = len(json.dumps(compact, ensure_ascii=False))
                self.assertLessEqual(actual, budget)
                self.assertEqual(metadata["returned_chars"], actual)
                self.assertTrue(metadata["output_truncated"])
        with self.assertRaises(ValueError):
            compact_tool_result("run_command", payload, max_chars=0)

    def test_compacted_read_cannot_validate_unseen_line(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "\n".join(f"value_{index} = '{'x' * 80}'" for index in range(80)) + "\n",
                encoding="utf-8",
            )
            client = SequenceClient([
                decision(action="tool", answer=None, tool="read_file",
                         arguments={"path": "app.py", "start_line": 1, "end_line": 80}),
                decision(answer="见 app.py:1-1。"),
            ])
            run = run_unified_query(
                "读取代码", UnifiedRunConfig(repo_path=root, max_context_chars=1000,
                                            trace_db_path=None), client,
            )

        self.assertTrue(run.observations[0]["output_truncated"])
        self.assertEqual(run.observations[0]["output"]["text"], "")
        self.assertIn("无效引用: app.py:1-1", run.answer)

    def test_direct_answer_does_not_call_tools(self):
        client = SequenceClient([decision(answer="你好")])

        run = run_unified_query("你好", UnifiedRunConfig(), client)

        self.assertEqual(run.status, "success")
        self.assertEqual(run.answer, "你好")
        self.assertEqual(run.observations, [])
        self.assertEqual([event["step"] for event in run.trace["events"]], ["model_decision", "final_answer"])
        self.assertEqual(run.trace["params"]["runtime"], "langgraph")
        self.assertEqual(
            run.trace["artifacts"]["agent"]["graph_steps"],
            ["model", "finalize"],
        )
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

    def test_separate_agent_runs_reuse_repository_index_and_trace_cache_status(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def target():\n    return True\n",
                encoding="utf-8",
            )
            config = UnifiedRunConfig(repo_path=root, top_k=1, recall_window=0)

            def run_once():
                client = SequenceClient([
                    decision(
                        action="tool",
                        answer=None,
                        tool="search_code",
                        arguments={"query": "target", "top_k": 1},
                    ),
                    decision(answer="定义位于 app.py:1-2。"),
                ])
                return run_unified_query("target 在哪里定义？", config, client)

            first = run_once()
            second = run_once()

        first_index = first.trace["artifacts"]["retrieval"]["searches"][0]["artifacts"]["index"]
        second_index = second.trace["artifacts"]["retrieval"]["searches"][0]["artifacts"]["index"]
        self.assertEqual(first_index["status"], "miss")
        self.assertGreater(first_index["index_build_ms"], 0)
        self.assertEqual(second_index["status"], "hit")
        self.assertEqual(second_index["index_build_ms"], 0)

    def test_patch_requires_approval_and_lists_files_by_default(self):
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
                    decision(answer="补丁等待批准。"),
                ]
            )

            run = run_unified_query("修改 value", UnifiedRunConfig(repo_path=root), client)

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(run.status, "approval_required")
        self.assertEqual(run.approval["tool"], "apply_patch")
        self.assertEqual(run.approval["files"], ["app.py"])
        self.assertEqual(
            run.trace["artifacts"]["agent"]["graph_steps"],
            ["model", "request_approval"],
        )
        self.assertEqual(run.trace["events"][-1]["step"], "approval_required")

    def test_approved_patch_applies_once_and_can_rollback(self):
        patch_text = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as snapshot_dir:
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

            config = UnifiedRunConfig(repo_path=root, snapshot_root=Path(snapshot_dir))
            pending = run_unified_query("修改 value", config, client)
            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

            run = resume_unified_query(
                pending,
                config,
                client,
                approval_fingerprint=pending.approval["fingerprint"],
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")
            write_result = run.observations[-1]["output"]
            self.assertIn("-value = 1", write_result["post_change_diff"]["text"])
            self.assertIn("+value = 2", write_result["post_change_diff"]["text"])
            rollback = SnapshotStore(root, snapshot_dir).rollback(write_result["snapshot_id"])
            self.assertEqual(rollback["remaining_diff"]["text"], "")
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(run.status, "success")
        self.assertTrue(run.execution_enabled)

    def test_project_policy_can_reject_write_approval(self):
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
                    decision(answer="补丁未执行。"),
                ]
            )

            run = run_unified_query(
                "修改 value",
                UnifiedRunConfig(repo_path=root),
                client,
                safety_policy=AgentSafetyPolicy(safe_mode=True),
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(run.status, "failed")
        self.assertIsNone(run.approval)
        self.assertFalse(run.execution_requested)
        self.assertFalse(run.execution_enabled)

    def test_pending_approval_cannot_resume_against_another_repository(self):
        patch_text = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        )
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            (first / "app.py").write_text("value = 1\n", encoding="utf-8")
            (second / "app.py").write_text("value = 1\n", encoding="utf-8")
            client = SequenceClient([
                decision(
                    action="tool",
                    answer=None,
                    tool="apply_patch",
                    arguments={"diff": patch_text},
                )
            ])
            pending = run_unified_query("修改 value", UnifiedRunConfig(repo_path=first), client)

            with self.assertRaisesRegex(ValueError, "different repository"):
                resume_unified_query(
                    pending,
                    UnifiedRunConfig(repo_path=second),
                    client,
                    approval_fingerprint=pending.approval["fingerprint"],
                )

            self.assertEqual((second / "app.py").read_text(encoding="utf-8"), "value = 1\n")

    def test_safety_rejection_stops_before_approval(self):
        client = SequenceClient(
            [
                decision(
                    action="tool",
                    answer=None,
                    tool="run_command",
                    arguments={
                        "cmd": 'python -c "print(1)"',
                        "affected_files": ["artifacts/result.txt"],
                    },
                ),
                decision(answer="该命令被安全策略拒绝。"),
            ]
        )

        run = run_unified_query("运行危险命令", UnifiedRunConfig(), client)

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.trace["events"][-1]["step"], "approval_rejected")
        self.assertIn("denied by safety policy", run.trace["events"][-1]["error"]["message"])

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
