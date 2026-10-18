from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from coding_rag.agent.planner import (
    AgentPlanConfig,
    ReActAgentInterface,
    WorkflowMode,
    classify_task,
    default_agent_tools,
    render_agent_plan_prompt,
    render_agent_plan_run,
    run_agent_plan_mode,
)


class DummyPlanClient:
    def complete(self, messages):
        self.messages = messages
        return """## 任务分析
- goal: fix the requested behavior
- constraints: use repository evidence and declared tools

## ReAct 工作流计划
1. Thought: locate the relevant entry point
   Action: search_code
   Action Input: {"query": "task", "top_k": 5}
   Expected Observation: candidate files are found
   Fallback: read_file main entry files
2. Thought: inspect candidate files
   Action: read_file
   Action Input: {"path": "main.py"}
   Expected Observation: target function is visible
   Fallback: inspect_symbol
3. Thought: verify the plan
   Action: run_checks
   Action Input: {"command": "python -m unittest"}
   Expected Observation: tests pass
   Fallback: narrow the test scope

## 候选文件
- main.py: CLI or interactive entry
## 验证计划
- python -m unittest: regression tests

## 风险与交付物
- risks: entry point could be misclassified
- deliverables: plan, candidate files, and checks
"""


class TaskPlannerTest(unittest.TestCase):
    def test_question_uses_unified_agent(self):
        plan = classify_task("Where does load_data read files?")

        self.assertEqual(plan.mode, WorkflowMode.AGENT)
        self.assertEqual(plan.intent, "unified_request")
        self.assertIn("检索", " ".join(plan.steps))

    def test_classify_change_request_as_agent(self):
        plan = classify_task("fix load_data encoding bug")

        self.assertEqual(plan.mode, WorkflowMode.AGENT)
        self.assertEqual(plan.intent, "unified_request")
        self.assertIn("审批", " ".join(plan.steps))

    def test_removed_ask_mode_is_not_accepted(self):
        with self.assertRaises(ValueError):
            classify_task("Where is the function?", requested_mode="ask")

    def test_react_compatibility_interface_is_plan_only(self):
        interface = ReActAgentInterface()

        self.assertFalse(hasattr(interface, "run"))
        self.assertEqual(interface.plan("implement a new feature").mode, WorkflowMode.AGENT)

    def test_agent_plan_mode_builds_prompt_and_llm_plan(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text(
                "def run_ask_cli(args):\n    return args.query\n",
                encoding="utf-8",
            )
            client = DummyPlanClient()

            run = run_agent_plan_mode(
                "fix ASK mode not calling the LLM",
                AgentPlanConfig(repo_path=root, top_k=1, recall_window=0),
                client=client,
            )

        self.assertEqual(run.plan.mode, WorkflowMode.AGENT)
        self.assertEqual(run.final_results, [])
        self.assertEqual(run.seed_results, [])
        self.assertEqual(run.trace["summary"]["planning_retrieval"], "disabled")
        self.assertIn("search_code", client.messages[1]["content"])
        self.assertIn("计划", client.messages[1]["content"])
        self.assertIn("ReAct 工作流计划", run.plan_text or "")
        self.assertIn("Agent Plan Prompt", render_agent_plan_prompt(run.messages))
        rendered = render_agent_plan_run(run)
        self.assertIn("search_code", rendered)
        self.assertIn("LLM", rendered)

    def test_default_agent_tools_are_declared_not_executed(self):
        tools = default_agent_tools()
        names = [tool.name for tool in tools]

        self.assertIn("search_code", names)
        self.assertIn("read_file", names)
        self.assertIn("list_files", names)
        self.assertIn("run_checks", names)
        self.assertIn("apply_patch", names)
        self.assertIn("run_command", names)
        self.assertIn("inspect_diff", names)
        command_tool = next(tool for tool in tools if tool.name == "run_command")
        self.assertIn("timeout_seconds", command_tool.input_schema)

    def test_agent_plan_mode_can_include_readonly_observations(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text(
                "def run_ask_cli(args):\n    return args.query\n",
                encoding="utf-8",
            )

            run = run_agent_plan_mode(
                "fix run_ask_cli not calling the LLM",
                AgentPlanConfig(
                    repo_path=root,
                    top_k=1,
                    recall_window=0,
                    execute_readonly_tools=True,
                ),
            )

        self.assertTrue(run.observations)
        self.assertIn("readonly_observations", run.trace)
        prompt = render_agent_plan_prompt(run.messages)
        self.assertIn("Read-only observations", prompt)
        self.assertIn("search_code", prompt)
        self.assertIn("list_files", prompt)
        rendered = render_agent_plan_run(run)
        self.assertIn("Read-only observations", rendered)
        self.assertIn("run_ask_cli", rendered)

    def test_agent_plan_eval_cases_have_reasonable_plan_shape(self):
        cases_path = Path("datasets/eval/agent_plan_cases.json")
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 50)
        for case in cases:
            with self.subTest(case=case["id"]):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    for relative_path, content in case["repo_files"].items():
                        path = root / relative_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding="utf-8")

                    run = run_agent_plan_mode(
                        case["task"],
                        AgentPlanConfig(repo_path=root, top_k=2, recall_window=0),
                        client=DummyPlanClient(),
                    )

                plan_text = run.plan_text or ""
                tool_names = [tool.name for tool in run.tools]
                self.assertEqual(run.final_results, [])
                for tool in case["expected_tools"]:
                    self.assertIn(tool, tool_names)
                    self.assertIn(tool, run.messages[1]["content"])
                for section in case["expected_sections"]:
                    self.assertIn(section, plan_text)


if __name__ == "__main__":
    unittest.main()
