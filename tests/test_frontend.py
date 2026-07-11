from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from frontend import INDEX_HTML, run_frontend_agent_plan, run_frontend_ask


class DummyLLMClient:
    """Fake chat client that records messages and returns a canned response."""

    def __init__(self, response: str = "## ReAct Plan\n1. Thought: search_code..."):
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


class ReviewClient:
    """Fake chat client that returns plan text on first call, review JSON on second."""

    def __init__(self):
        self.call_count = 0
        self.plan_messages = None
        self.review_messages = None

    def complete(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            self.plan_messages = messages
            return "## ReAct Plan\n1. Thought: fix..."
        else:
            self.review_messages = messages
            return '{"verdict": "pass", "score": 0.86, "issues": [], "suggestions": ["add fallback"]}'


class FrontendTest(unittest.TestCase):
    def test_index_html_no_curly_smart_quotes(self):
        cases = [chr(0x201C), chr(0x201D), chr(0x2018), chr(0x2019), chr(0xFFFD)]
        for char in cases:
            with self.subTest(char=repr(char)):
                self.assertNotIn(char, INDEX_HTML)

    def test_agent_mode_controls_present_and_visible(self):
        self.assertIn('<option value="agent">Agent</option>', INDEX_HTML)
        self.assertIn('<option value="ask">ASK</option>', INDEX_HTML)
        self.assertIn('id="reviewToggle" style="display:none"', INDEX_HTML)
        self.assertIn('id="reviewAgentPlan"', INDEX_HTML)
        self.assertIn('id="readonlyToolsToggle" style="display:none"', INDEX_HTML)
        self.assertIn('id="executeReadonlyTools"', INDEX_HTML)
        self.assertIn('"/api/agent-plan"', INDEX_HTML)
        self.assertIn('execute_readonly_tools', INDEX_HTML)
        self.assertIn('"reviewToggle"', INDEX_HTML)
        self.assertIn('"readonlyToolsToggle"', INDEX_HTML)
        self.assertIn('"modeRow"', INDEX_HTML)

    def test_index_contains_main_controls(self):
        self.assertIn('RepoPilot', INDEX_HTML)
        self.assertIn('/api/ask', INDEX_HTML)
        self.assertIn('/api/agent-plan', INDEX_HTML)
        self.assertIn('workflowMode', INDEX_HTML)
        self.assertIn('useLlm', INDEX_HTML)
        self.assertIn('renderTrace', INDEX_HTML)
        self.assertIn('Trace Events', INDEX_HTML)

    def test_run_frontend_ask_returns_results_without_llm(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "load_data 在哪里读取文件？",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": False,
            }
            response = run_frontend_ask(payload)

        self.assertEqual(response["stats"]["final_count"], 1)
        self.assertEqual(response["trace"]["trace_version"], "1.0")
        self.assertIn("events", response["trace"])
        self.assertIsNone(response["answer"])
        self.assertIn("load_data", response["results"][0]["text"])

    # ------------------------------------------------------------------
    # Agent plan endpoint tests
    # ------------------------------------------------------------------

    def test_run_frontend_agent_plan_without_llm(self):
        """Agent plan without LLM: plan/tools/stats present; plan_text and review are None."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "实现 load_data 的缓存功能",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": False,
            }
            response = run_frontend_agent_plan(payload)

        self.assertIn("Agent 计划模式", response["plan"])
        self.assertIsNone(response["plan_text"])
        self.assertIsNone(response["review"])
        self.assertIsInstance(response["tools"], list)
        self.assertGreater(len(response["tools"]), 0)
        self.assertIn("name", response["tools"][0])
        self.assertIn("purpose", response["tools"][0])
        self.assertEqual(response["observations"], [])
        self.assertEqual(response["trace"]["trace_version"], "1.0")
        self.assertEqual(response["trace"]["run"]["mode"], "agent_plan")
        stats = response["stats"]
        self.assertEqual(stats["seed_count"], 0)
        self.assertEqual(stats["recalled_count"], 0)
        self.assertEqual(stats["final_count"], 0)

    def test_run_frontend_agent_plan_with_readonly_tools(self):
        """Agent plan can execute first-stage read-only tools and return observations."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "实现 load_data 的缓存功能",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": False,
                "execute_readonly_tools": True,
            }
            response = run_frontend_agent_plan(payload)

        self.assertIsInstance(response["observations"], list)
        self.assertGreaterEqual(len(response["observations"]), 2)
        tool_names = [row["tool"] for row in response["observations"]]
        self.assertIn("search_code", tool_names)
        self.assertIn("list_files", tool_names)
        self.assertIn("Read-only observations", response["plan"])

    def test_run_frontend_agent_plan_with_llm(self):
        """Agent plan with LLM: plan_text is not None, prompt returned, review is None."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "实现 load_data 的缓存功能",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": True,
            }
            dummy = DummyLLMClient("## ReAct Plan\n1. Thought: search_code...")
            with patch("frontend.OpenAICompatibleChatClient", return_value=dummy):
                with patch("frontend.ensure_dotenv"), patch("frontend.load_dotenv"):
                    with patch("frontend.build_llm_config", return_value=None):
                        response = run_frontend_agent_plan(payload)

        self.assertIsNotNone(response["plan_text"])
        self.assertEqual(response["plan_text"], "## ReAct Plan\n1. Thought: search_code...")
        self.assertIsNone(response["review"])
        self.assertIsNotNone(response["prompt"])
        self.assertIn("Agent 计划模式", response["plan"])
        stats = response["stats"]
        self.assertEqual(stats["seed_count"], 0)
        self.assertEqual(stats["recalled_count"], 0)
        self.assertEqual(stats["final_count"], 0)

    def test_review_agent_plan_without_llm_raises(self):
        """review_agent_plan=true with use_llm=false raises ValueError."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "实现 load_data 的缓存功能",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": False,
                "review_agent_plan": True,
            }
            with self.assertRaises(ValueError) as ctx:
                run_frontend_agent_plan(payload)
            self.assertIn("LLM", str(ctx.exception))

    def test_run_frontend_agent_plan_with_review(self):
        """Agent plan with review: review dict is populated with verdict/score/issues/suggestions."""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def load_data(path):\n    return path.read_text(encoding='utf-8')\n",
                encoding="utf-8",
            )

            payload = {
                "repo_path": str(root),
                "query": "实现 load_data 的缓存功能",
                "top_k": 1,
                "recall_window": 0,
                "use_llm": True,
                "review_agent_plan": True,
            }
            dummy = ReviewClient()
            with patch("frontend.OpenAICompatibleChatClient", return_value=dummy):
                with patch("frontend.ensure_dotenv"), patch("frontend.load_dotenv"):
                    with patch("frontend.build_llm_config", return_value=None):
                        response = run_frontend_agent_plan(payload)

        self.assertIsNotNone(response["review"])
        self.assertEqual(response["review"]["verdict"], "pass")
        self.assertEqual(response["review"]["score"], 0.86)
        self.assertTrue(response["review"]["passed"])
        self.assertEqual(response["review"]["issues"], [])
        self.assertEqual(response["review"]["suggestions"], ["add fallback"])
        self.assertIsNotNone(response["plan_text"])
        self.assertEqual(dummy.call_count, 2)


if __name__ == "__main__":
    unittest.main()
