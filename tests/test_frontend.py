import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch
import unittest
import urllib.error
import urllib.request

from frontend import (
    INDEX_HTML,
    PendingApprovalStore,
    RepoPilotHandler,
    RepoPilotServer,
    build_index_html,
    run_frontend_query,
    run_frontend_rollback,
)


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages):
        return self.responses.pop(0)


def answer_decision(answer="完成"):
    return json.dumps(
        {
            "action": "answer",
            "reason": "已有足够信息",
            "answer": answer,
            "tool": None,
            "arguments": {},
            "expected_observation": None,
        },
        ensure_ascii=False,
    )


class FrontendTest(unittest.TestCase):
    def test_index_keeps_unified_workflow_controls_alongside_learning_mode(self):
        self.assertIn("RepoPilot", INDEX_HTML)
        self.assertIn('fetch("/api/run"', INDEX_HTML)
        self.assertNotIn('id="executeTools"', INDEX_HTML)
        self.assertIn("approveWrite", INDEX_HTML)
        self.assertIn("rollbackWrite", INDEX_HTML)
        self.assertIn("serverAllowsExecution", INDEX_HTML)
        self.assertIn("Trace Events", INDEX_HTML)
        self.assertIn("项目学习", INDEX_HTML)
        self.assertIn('/api/learning/session', INDEX_HTML)
        self.assertNotIn("useLlm", INDEX_HTML)
        self.assertNotIn("/api/ask", INDEX_HTML)
        self.assertNotIn("/api/agent-plan", INDEX_HTML)

    def test_execution_checkbox_reflects_server_capability(self):
        disabled_html = build_index_html(False)
        enabled_html = build_index_html(True)

        self.assertIn("const serverAllowsExecution = false", disabled_html)
        self.assertIn("const serverAllowsExecution = true", enabled_html)

    def test_index_html_has_no_replacement_character(self):
        self.assertNotIn(chr(0xFFFD), INDEX_HTML)

    def test_run_frontend_query_returns_unified_response(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("def target():\n    return True\n", encoding="utf-8")
            payload = {
                "repo_path": str(root),
                "query": "你好",
                "provider": "deepseek",
            }
            client = SequenceClient([answer_decision("你好")])
            with patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()):
                with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                    response = run_frontend_query(payload)

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["answer"], "你好")
        self.assertEqual(response["trace"]["run"]["mode"], "unified")
        self.assertFalse(response["execution"]["enabled"])
        self.assertFalse(response["execution"]["server_allowed"])

    def test_legacy_blanket_execution_request_is_rejected(self):
        payload = {"repo_path": ".", "query": "修改代码", "execute_tools": True}

        with self.assertRaises(ValueError):
            run_frontend_query(payload, server_allows_execution=False)

    def test_browser_write_requires_server_capability_and_one_call_approval(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            diff = (
                "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
            )
            payload = {
                "repo_path": temp_dir,
                "query": "修改 value",
                "provider": "deepseek",
            }
            client = SequenceClient([
                json.dumps({
                    "action": "tool", "reason": "修改", "answer": None,
                    "tool": "apply_patch", "arguments": {"diff": diff},
                    "expected_observation": "修改完成",
                }, ensure_ascii=False),
                answer_decision("修改完成"),
            ])
            store = PendingApprovalStore()
            with patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()), patch(
                "coding_rag.agent.snapshots.Path.home", return_value=root
            ):
                with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                    pending = run_frontend_query(
                        payload,
                        server_allows_execution=True,
                        approval_store=store,
                    )
                    content_before_approval = target.read_text(encoding="utf-8")
                    response = run_frontend_query(
                        {"approval_id": pending["approval_id"]},
                        server_allows_execution=True,
                        approval_store=store,
                    )
                    content_after_approval = target.read_text(encoding="utf-8")
                    patch_event = next(
                        event for event in response["trace"]["events"] if event["step"] == "apply_patch"
                    )
                    snapshot_id = patch_event["artifacts"]["patch"]["snapshot_id"]
                    rollback = run_frontend_rollback(
                        {"repo_path": str(root), "snapshot_id": snapshot_id},
                        server_allows_execution=True,
                    )
                    content_after_rollback = target.read_text(encoding="utf-8")
            self.assertEqual(pending["status"], "approval_required")
            self.assertEqual(pending["approval"]["files"], ["app.py"])
            self.assertTrue(pending["approval_id"])
            self.assertEqual(content_before_approval, "value = 1\n")

        self.assertTrue(response["execution"]["requested"])
        self.assertTrue(response["execution"]["server_allowed"])
        self.assertTrue(response["execution"]["enabled"])
        self.assertEqual(content_after_approval, "value = 2\n")
        self.assertEqual(content_after_rollback, "value = 1\n")
        self.assertEqual(rollback["remaining_diff"]["text"], "")

    def test_project_policy_can_force_browser_back_to_read_only(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_dir = root / ".repopilot"
            policy_dir.mkdir()
            (policy_dir / "policy.json").write_text('{"safe_mode": true}', encoding="utf-8")
            payload = {
                "repo_path": str(root),
                "query": "你好",
                "provider": "deepseek",
            }
            client = SequenceClient([answer_decision()])
            with patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()):
                with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                    response = run_frontend_query(payload, server_allows_execution=True)

        self.assertFalse(response["execution"]["enabled"])
        self.assertTrue(response["trace"]["run"]["flags"]["safe_mode"])

    def test_browser_rollback_requires_server_capability(self):
        with self.assertRaises(PermissionError):
            run_frontend_rollback(
                {"repo_path": ".", "snapshot_id": "0" * 24},
                server_allows_execution=False,
            )

    def test_old_http_endpoints_return_404(self):
        server = RepoPilotServer(("127.0.0.1", 0), RepoPilotHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            for path in ("/api/ask", "/api/agent-plan"):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.subTest(path=path):
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        urllib.request.urlopen(request, timeout=2)
                    self.assertEqual(raised.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
