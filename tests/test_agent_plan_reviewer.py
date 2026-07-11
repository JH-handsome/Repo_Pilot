from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from coding_rag.agent.plan_reviewer import (
    build_agent_plan_review_messages,
    parse_agent_plan_review,
    render_agent_plan_review,
    review_agent_plan,
)
from coding_rag.agent.planner import AgentPlanConfig, run_agent_plan_mode
from tests.test_task_planner import DummyPlanClient


class DummyReviewClient:
    def complete(self, messages):
        self.messages = messages
        return json.dumps(
            {
                "verdict": "pass",
                "score": 0.86,
                "issues": [],
                "suggestions": ["add candidate file rationale"],
            },
            ensure_ascii=False,
        )


class AgentPlanReviewerTest(unittest.TestCase):
    def test_parse_review_json(self):
        review = parse_agent_plan_review(
            '{"verdict":"pass","score":0.8,"issues":["minor"],"suggestions":["add tests"]}'
        )

        self.assertTrue(review.passed)
        self.assertEqual(review.issues, ["minor"])

    def test_review_agent_plan_calls_llm(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("def run_ask_cli(args):\n    return args.query\n", encoding="utf-8")
            plan_run = run_agent_plan_mode(
                "修复 ASK 模式没有调用大模型回答的问题",
                AgentPlanConfig(repo_path=root, top_k=1, recall_window=0),
                client=DummyPlanClient(),
            )
            client = DummyReviewClient()

            review = review_agent_plan(plan_run, client)

        self.assertTrue(review.passed)
        self.assertIn("search_code", client.messages[1]["content"])
        self.assertIn("search_code", client.messages[1]["content"])
        self.assertIn("Agent 计划审查", render_agent_plan_review(review))

    def test_review_prompt_contains_candidate_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("def target():\n    pass\n", encoding="utf-8")
            plan_run = run_agent_plan_mode(
                "修复 target 函数",
                AgentPlanConfig(repo_path=root, top_k=1, recall_window=0),
                client=DummyPlanClient(),
            )

            messages = build_agent_plan_review_messages(plan_run)

        self.assertIn("candidate_files", messages[1]["content"])
        self.assertIn("main.py", messages[1]["content"])

    def test_agent_plan_dataset_reviews_pass_with_dummy_llm(self):
        cases = json.loads(Path("datasets/eval/agent_plan_cases.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 50)
        for case in cases:
            with self.subTest(case=case["id"]):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    for relative_path, content in case["repo_files"].items():
                        path = root / relative_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding="utf-8")

                    plan_run = run_agent_plan_mode(
                        case["task"],
                        AgentPlanConfig(repo_path=root, top_k=2, recall_window=0),
                        client=DummyPlanClient(),
                    )
                    review = review_agent_plan(plan_run, DummyReviewClient())

                self.assertTrue(review.passed)


if __name__ == "__main__":
    unittest.main()
