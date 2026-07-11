from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from coding_rag.agent import CodeAgentConfig, analyze_task, agent_run_to_dict, render_agent_run, run_code_agent


class DummyClient:
    def complete(self, messages):
        self.messages = messages
        return "## 实施计划\n- 修改 app.py\n\n## 测试计划\n- 运行 unittest"


class CodeAgentTest(unittest.TestCase):
    def test_offline_agent_builds_plan_and_memory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text()\n",
                encoding="utf-8",
            )
            memory_path = root / "agent_memory.jsonl"

            run = run_code_agent(
                "load_data 需要支持缺省编码",
                CodeAgentConfig(repo_path=str(root), top_k=1, memory_path=memory_path),
            )

            self.assertIn("Agent 工作流设计", run.plan)
            self.assertIn("未启用 LLM", run.implementation)
            self.assertEqual(run.memory.status, "planned")
            self.assertEqual(run.task_profile.kind, "feature")
            self.assertIn("python -m unittest", run.task_profile.recommended_checks)
            self.assertTrue(memory_path.exists())
            self.assertTrue(run.final_results)
            self.assertEqual(run.agent_trace[0]["step"], "receive_task")
            self.assertEqual(run.agent_trace[-1]["step"], "remember")

            rendered = render_agent_run(run, show_trace=False)
            self.assertIn("Agent 运行轨迹", rendered)
            self.assertIn("retrieve_context", rendered)
            self.assertFalse(run.run_log_written)

    def test_agent_uses_relevant_memory_and_llm_client(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def run_agent(task):\n    return task\n",
                encoding="utf-8",
            )
            memory_path = root / "agent_memory.jsonl"
            config = CodeAgentConfig(repo_path=str(root), top_k=1, memory_path=memory_path)
            run_code_agent("agent 记忆功能", config)

            client = DummyClient()
            run = run_code_agent("继续优化 agent 记忆功能", config, client=client)

            self.assertEqual(run.memory.status, "drafted")
            self.assertIn("修改 app.py", run.implementation)
            self.assertTrue(run.memories)
            self.assertIn("历史记忆", client.messages[1]["content"])
            self.assertIn("任务画像", client.messages[1]["content"])
            self.assertEqual(run.agent_trace[4]["status"], "drafted")

            payload = agent_run_to_dict(run)
            self.assertIn("task_profile", payload)
            self.assertIn("agent_trace", payload)
            self.assertIn("trace_unified", payload)
            self.assertEqual(payload["agent_trace"][-1]["step"], "remember")
            self.assertEqual(payload["trace_unified"]["trace_version"], "1.0")
            self.assertEqual(payload["trace_unified"]["run"]["mode"], "agent_exec")
            self.assertEqual(payload["trace_unified"]["events"][0]["step"], "receive_task")

    def test_agent_writes_run_log_to_separate_file(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("def patch_target():\n    return True\n", encoding="utf-8")
            memory_path = root / "agent_memory.jsonl"
            log_path = root / "agent_run_log.jsonl"

            run = run_code_agent(
                "记录 agent 运行环节",
                CodeAgentConfig(repo_path=str(root), top_k=1, memory_path=memory_path, run_log_path=log_path),
            )

            self.assertTrue(run.run_log_written)
            self.assertEqual(run.run_log_path, str(log_path))
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["log_type"], "agent_run")
            self.assertTrue(rows[0]["run_log_written"])
            self.assertEqual(rows[0]["agent_trace"][0]["step"], "receive_task")
            self.assertEqual(rows[0]["agent_trace"][-1]["step"], "remember")

    def test_agent_dry_run_skips_memory_and_run_log_writes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("def patch_target():\n    return True\n", encoding="utf-8")
            memory_path = root / "agent_memory.jsonl"
            log_path = root / "agent_run_log.jsonl"

            run = run_code_agent(
                "修复 patch_target 的返回值",
                CodeAgentConfig(
                    repo_path=str(root),
                    top_k=1,
                    memory_path=memory_path,
                    run_log_path=log_path,
                    dry_run=True,
                ),
            )

            self.assertFalse(run.memory_written)
            self.assertFalse(run.run_log_written)
            self.assertFalse(memory_path.exists())
            self.assertFalse(log_path.exists())
            self.assertEqual(run.agent_trace[-1]["status"], "skipped")
            self.assertTrue(run.agent_trace[-1]["artifacts"]["dry_run"])

    def test_agent_skips_duplicate_memory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("def feature():\n    return True\n", encoding="utf-8")
            memory_path = root / "agent_memory.jsonl"
            config = CodeAgentConfig(repo_path=str(root), top_k=1, memory_path=memory_path)

            first = run_code_agent("新增 feature 功能", config)
            second = run_code_agent("新增 feature 功能", config)

            self.assertTrue(first.memory_written)
            self.assertFalse(second.memory_written)
            self.assertEqual(second.agent_trace[-1]["status"], "skipped")
            self.assertEqual(len(memory_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_task_profile_for_evaluation_recommends_eval(self):
        profile = analyze_task("优化 trace 评测轨迹")

        self.assertEqual(profile.kind, "evaluation")
        self.assertIn(
            "python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --top-k 5",
            profile.recommended_checks,
        )


if __name__ == "__main__":
    unittest.main()
