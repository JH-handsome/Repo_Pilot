"""Learning Mode 浏览器适配层、HTTP 边界和页面状态测试。"""

from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any, Iterator
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from coding_rag.learning.models import LearningPlan, LearningWorkflowResult
from frontend import (
    INDEX_HTML,
    RepoPilotHandler,
    RepoPilotServer,
    build_index_html,
    run_learning_frontend_request,
)


SIGNING_KEY = b"frontend-signing-secret-32-bytes!!"
LEARNING_GOAL = "从空目录复现这个项目"


class SequenceClient:
    """返回预置模型输出，并保留调用证据。"""

    def __init__(self, responses: list[str | Exception] | None = None):
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def valid_plan_payload() -> dict[str, Any]:
    """构造四步、每步两个验收项的最小有效教学计划。"""
    evidence = {
        "claim": "app.py 暴露了项目入口。",
        "path": "app.py",
        "start_line": 1,
        "end_line": 2,
    }
    steps: list[dict[str, Any]] = []
    for index in range(4):
        number = index + 1
        steps.append(
            {
                "step_id": f"step-{number}",
                "title": f"完成第 {number} 个增量",
                "learning_goal": f"理解第 {number} 个增量的职责边界",
                "depends_on": [] if index == 0 else [f"step-{number - 1}"],
                "files_to_create": [f"src/step_{number}.py"],
                "tasks": [f"实现第 {number} 个可独立验证的行为。"],
                "why": "把工作拆小可以清楚定位失败。",
                "benefits": ["每轮只验证一个新增行为。"],
                "verification": [
                    f"第 {number} 步单元测试通过。",
                    f"第 {number} 步烟雾测试返回状态码 0。",
                ],
                "evidence": [evidence],
                "common_pitfalls": ["只描述结果，没有提供测试输出。"],
                "optimization_question": "怎样减少这一步与后续步骤的耦合？",
            }
        )
    return {
        "project_profile": {
            "project_name": "Frontend Session Demo",
            "summary": "用于验证浏览器教学会话的最小项目。",
            "tech_stack": ["Python"],
            "prerequisites": ["Python 基础"],
            "entry_points": [evidence],
            "components": [
                {
                    "name": "入口",
                    "responsibility": "连接四个教学增量。",
                    "evidence": [evidence],
                }
            ],
        },
        "steps": steps,
    }


def workflow_success(plan: LearningPlan) -> LearningWorkflowResult:
    """绕过真实检索，返回结构真实的规划结果。"""
    return LearningWorkflowResult(
        status="success",
        learning_goal=LEARNING_GOAL,
        learner_level="beginner",
        project_profile=plan.project_profile,
        steps=plan.steps,
        first_step=plan.steps[0],
        graph_steps=["collect_evidence", "analyze_and_plan", "present_step"],
        trace={"source": "frontend-offline-test"},
        error=None,
    )


def review_json(step: Any, *, passed: bool) -> str:
    findings = [
        {
            "verification_index": index,
            "satisfied": passed,
            "reason": f"验收项 {index} 的学习者报告已审查。",
        }
        for index, _ in enumerate(step.verification)
    ]
    return json.dumps(
        {
            "passed": passed,
            "evidence_sufficient": passed,
            "findings": findings,
            "gaps": [] if passed else ["测试证据不足。"],
            "hint": None if passed else "补充逐项测试结果后重试。",
            "verification_scope": "learner_reported_evidence",
        },
        ensure_ascii=False,
    )


def outer_payload(repo_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo_path": str(repo_path),
        "provider": "deepseek",
        "top_k": 8,
        "recall_window": 2,
        "request": request,
    }


@contextmanager
def running_server() -> Iterator[tuple[RepoPilotServer, str]]:
    server = RepoPilotServer(("127.0.0.1", 0), RepoPilotHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server, base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    content_type: str = "application/json",
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            decoded = {"raw": body}
        return error.code, decoded
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


class LearningFrontendAdapterTest(unittest.TestCase):
    """覆盖 start -> submit -> reflect 的单回合适配行为。"""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "app.py").write_text(
            'def main():\n    return "ok"\n',
            encoding="utf-8",
        )
        self.plan = LearningPlan.model_validate(valid_plan_payload())

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def call(
        self,
        request: dict[str, Any],
        *,
        client: SequenceClient | None = None,
        patch_planning: bool = False,
    ) -> tuple[dict[str, Any], SequenceClient]:
        active_client = client or SequenceClient()
        patches = [
            patch("frontend.load_dotenv"),
            patch("frontend.build_llm_config", return_value=object()),
            patch("frontend.OpenAICompatibleChatClient", return_value=active_client),
        ]
        if patch_planning:
            patches.append(
                patch(
                    "coding_rag.learning.session.run_learning_workflow",
                    return_value=workflow_success(self.plan),
                )
            )
        entered = []
        try:
            for item in patches:
                entered.append(item)
                item.start()
            result = run_learning_frontend_request(
                outer_payload(self.root, request),
                signing_key=SIGNING_KEY,
            )
        finally:
            for item in reversed(entered):
                item.stop()
        return result, active_client

    def start_session(self) -> dict[str, Any]:
        result, client = self.call(
            {
                "action": "start",
                "learning_goal": LEARNING_GOAL,
                "learner_level": "beginner",
            },
            patch_planning=True,
        )
        self.assertEqual(client.calls, [])
        return result

    def test_start_returns_signed_first_waiting_point(self):
        result = self.start_session()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["action"], "start")
        self.assertEqual(result["session"]["phase"], "awaiting_submission")
        self.assertEqual(result["next_action"], "submit")
        self.assertEqual(result["current_step"]["step_id"], "step-1")
        self.assertEqual(len(result["session"]["integrity_token"]), 64)

    def test_empty_submit_fails_precheck_without_calling_model(self):
        started = self.start_session()

        result, client = self.call(
            {
                "action": "submit",
                "session": started["session"],
                "submission": {"implementation_summary": "", "test_output": ""},
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["session"]["phase"], "needs_revision")
        self.assertEqual(result["next_action"], "submit")
        self.assertTrue(result["feedback"]["gaps"])
        self.assertEqual(result["session"]["reviews"][-1]["review_source"], "precheck")
        self.assertEqual(client.calls, [])

    def test_model_passed_submit_waits_for_reflection(self):
        started = self.start_session()
        step = self.plan.steps[0]
        client = SequenceClient([review_json(step, passed=True)])

        result, client = self.call(
            {
                "action": "submit",
                "session": started["session"],
                "submission": {
                    "implementation_summary": "实现了当前增量并逐项检查验收条件。",
                    "test_output": "2 tests passed; process exited with code 0",
                },
            },
            client=client,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["session"]["phase"], "awaiting_reflection")
        self.assertEqual(result["next_action"], "reflect")
        self.assertTrue(result["review"]["passed"])
        self.assertEqual(len(client.calls), 1)

    def test_reflect_advances_exactly_one_step(self):
        started = self.start_session()
        client = SequenceClient([review_json(self.plan.steps[0], passed=True)])
        submitted, _ = self.call(
            {
                "action": "submit",
                "session": started["session"],
                "submission": {
                    "implementation_summary": "实现完成并核对验收项。",
                    "test_output": "all tests passed with exit code 0",
                },
            },
            client=client,
        )

        reflected, reflect_client = self.call(
            {
                "action": "reflect",
                "session": submitted["session"],
                "reflection": "保持接口最小可以减少与后续步骤的耦合。",
            }
        )

        self.assertEqual(reflected["status"], "success")
        self.assertEqual(reflected["session"]["phase"], "awaiting_submission")
        self.assertEqual(reflected["session"]["current_step_index"], 1)
        self.assertEqual(reflected["current_step"]["step_id"], "step-2")
        self.assertEqual(reflected["next_action"], "submit")
        self.assertEqual(reflect_client.calls, [])

    def test_tampered_session_is_rejected(self):
        started = self.start_session()
        started["session"]["learning_goal"] = "被篡改的目标"

        result, client = self.call(
            {
                "action": "submit",
                "session": started["session"],
                "submission": {
                    "implementation_summary": "任意说明",
                    "test_output": "tests passed",
                },
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "invalid_session")
        self.assertIsNone(result["session"])
        self.assertEqual(client.calls, [])

    def test_model_failure_keeps_last_verified_session(self):
        started = self.start_session()
        result, client = self.call(
            {
                "action": "submit",
                "session": started["session"],
                "submission": {
                    "implementation_summary": "已经完成实现。",
                    "test_output": "tests passed with exit code 0",
                },
            },
            client=SequenceClient([RuntimeError("provider secret")]),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "model_error")
        self.assertEqual(result["session"], started["session"])
        self.assertEqual(result["next_action"], "submit")
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("provider secret", json.dumps(result, ensure_ascii=False))

    def test_signing_key_never_appears_in_response_or_trace(self):
        result = self.start_session()
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertNotIn(SIGNING_KEY.decode("ascii"), serialized)
        self.assertNotIn(SIGNING_KEY.hex(), serialized)
        self.assertNotIn(SIGNING_KEY.decode("ascii"), build_index_html(True))

    def test_learning_rejects_execution_flag_and_never_uses_agent_runtime(self):
        payload = outer_payload(
            self.root,
            {
                "action": "start",
                "learning_goal": LEARNING_GOAL,
                "learner_level": "beginner",
            },
        )
        payload["execute_tools"] = True

        with patch("frontend.run_unified_query") as unified:
            with self.assertRaises(ValueError):
                run_learning_frontend_request(payload, signing_key=SIGNING_KEY)

        unified.assert_not_called()

    def test_learning_rejects_float_top_k_before_running_session(self):
        payload = outer_payload(
            self.root,
            {
                "action": "start",
                "learning_goal": LEARNING_GOAL,
                "learner_level": "beginner",
            },
        )
        payload["top_k"] = 8.5

        with patch("frontend.run_learning_session") as run_session:
            with self.assertRaisesRegex(ValueError, "top_k must be an integer"):
                run_learning_frontend_request(payload, signing_key=SIGNING_KEY)

        run_session.assert_not_called()

    def test_command_text_is_review_evidence_and_is_never_executed(self):
        started = self.start_session()
        client = SequenceClient([review_json(self.plan.steps[0], passed=True)])
        command_text = 'Remove-Item -Recurse .; <script>alert("x")</script>'

        with patch("frontend.run_unified_query") as unified:
            with patch("subprocess.run") as process_run:
                result, client = self.call(
                    {
                        "action": "submit",
                        "session": started["session"],
                        "submission": {
                            "implementation_summary": command_text,
                            "test_output": f"未执行，仅作为报告：{command_text}",
                        },
                    },
                    client=client,
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["session"]["phase"], "awaiting_reflection")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Remove-Item -Recurse .;", client.calls[0][-1]["content"])
        self.assertIn("<script>", client.calls[0][-1]["content"])
        unified.assert_not_called()
        process_run.assert_not_called()


class LearningFrontendHttpTest(unittest.TestCase):
    """验证独立端点、业务/HTTP 错误分层和正文上限。"""

    def test_text_plain_is_415_before_either_runner_is_called(self):
        with running_server() as (_, base_url):
            with patch("frontend.run_frontend_query") as run_query:
                with patch("frontend.run_learning_frontend_request") as run_learning:
                    for path in ("/api/run", "/api/learning/session"):
                        with self.subTest(path=path):
                            status, body = post_json(
                                f"{base_url}{path}",
                                {},
                                content_type="text/plain",
                            )
                            self.assertEqual(status, 415)
                            self.assertFalse(body["ok"])

        run_query.assert_not_called()
        run_learning.assert_not_called()

    def test_run_endpoint_keeps_ok_wrapper_and_server_execution_authorization(self):
        captured: dict[str, Any] = {}

        def fake_query(
            payload: dict[str, Any],
            *,
            server_allows_execution: bool,
            approval_store,
        ) -> dict[str, Any]:
            captured["payload"] = payload
            captured["server_allows_execution"] = server_allows_execution
            captured["approval_store"] = approval_store
            return {
                "status": "success",
                "answer": "mock answer",
                "execution": {"enabled": True},
            }

        payload = {"repo_path": ".", "query": "修改代码"}
        with running_server() as (server, base_url):
            server.allow_tool_execution = True
            with patch("frontend.run_frontend_query", side_effect=fake_query):
                status, body = post_json(f"{base_url}/api/run", payload)

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["answer"], "mock answer")
        self.assertEqual(captured["payload"], payload)
        self.assertTrue(captured["server_allows_execution"])
        self.assertIs(captured["approval_store"], server.pending_approvals)

    def test_same_server_runs_real_start_submit_reflect_sequence(self):
        plan = LearningPlan.model_validate(valid_plan_payload())
        client = SequenceClient([review_json(plan.steps[0], passed=True)])

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                'def main():\n    return "ok"\n',
                encoding="utf-8",
            )
            with patch("frontend.load_dotenv"):
                with patch("frontend.build_llm_config", return_value=object()):
                    with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                        with patch(
                            "coding_rag.learning.session.run_learning_workflow",
                            return_value=workflow_success(plan),
                        ):
                            with running_server() as (server, base_url):
                                signing_key = server.learning_signing_key
                                endpoint = f"{base_url}/api/learning/session"

                                start_status, started = post_json(
                                    endpoint,
                                    outer_payload(
                                        root,
                                        {
                                            "action": "start",
                                            "learning_goal": LEARNING_GOAL,
                                            "learner_level": "beginner",
                                        },
                                    ),
                                )
                                submit_status, submitted = post_json(
                                    endpoint,
                                    outer_payload(
                                        root,
                                        {
                                            "action": "submit",
                                            "session": started["session"],
                                            "submission": {
                                                "implementation_summary": "完成当前增量并逐项核对验收条件。",
                                                "test_output": "2 tests passed; exit code 0",
                                            },
                                        },
                                    ),
                                )
                                reflect_status, reflected = post_json(
                                    endpoint,
                                    outer_payload(
                                        root,
                                        {
                                            "action": "reflect",
                                            "session": submitted["session"],
                                            "reflection": "保持接口最小可以降低后续步骤的耦合。",
                                        },
                                    ),
                                )

                                self.assertEqual(server.learning_signing_key, signing_key)

        self.assertEqual((start_status, submit_status, reflect_status), (200, 200, 200))
        self.assertEqual(started["session"]["phase"], "awaiting_submission")
        self.assertEqual(submitted["session"]["phase"], "awaiting_reflection")
        self.assertEqual(reflected["session"]["phase"], "awaiting_submission")
        self.assertEqual(reflected["session"]["current_step_index"], 1)
        self.assertEqual(started["next_action"], "submit")
        self.assertEqual(submitted["next_action"], "reflect")
        self.assertEqual(reflected["next_action"], "submit")
        self.assertEqual(len(client.calls), 1)

    def test_rotated_server_key_rejects_old_session_over_http(self):
        plan = LearningPlan.model_validate(valid_plan_payload())
        client = SequenceClient()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                'def main():\n    return "ok"\n',
                encoding="utf-8",
            )
            with patch("frontend.load_dotenv"):
                with patch("frontend.build_llm_config", return_value=object()):
                    with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                        with patch(
                            "coding_rag.learning.session.run_learning_workflow",
                            return_value=workflow_success(plan),
                        ):
                            with running_server() as (server, base_url):
                                endpoint = f"{base_url}/api/learning/session"
                                start_status, started = post_json(
                                    endpoint,
                                    outer_payload(
                                        root,
                                        {
                                            "action": "start",
                                            "learning_goal": LEARNING_GOAL,
                                            "learner_level": "beginner",
                                        },
                                    ),
                                )
                                server.learning_signing_key = b"r" * 32
                                submit_status, rejected = post_json(
                                    endpoint,
                                    outer_payload(
                                        root,
                                        {
                                            "action": "submit",
                                            "session": started["session"],
                                            "submission": {
                                                "implementation_summary": "已完成当前增量。",
                                                "test_output": "tests passed",
                                            },
                                        },
                                    ),
                                )

        self.assertEqual((start_status, submit_status), (200, 200))
        self.assertEqual(rejected["status"], "failed")
        self.assertEqual(rejected["error"]["code"], "invalid_session")
        self.assertIsNone(rejected["session"])
        self.assertEqual(client.calls, [])

    def test_learning_endpoint_routes_start_with_process_key(self):
        captured: dict[str, Any] = {}

        def fake_run(payload: dict[str, Any], *, signing_key: bytes) -> dict[str, Any]:
            captured["payload"] = payload
            captured["signing_key"] = signing_key
            return {
                "status": "success",
                "action": "start",
                "session": {"integrity_token": "a" * 64},
                "current_step": {"step_id": "step-1"},
                "next_action": "submit",
                "review": None,
                "feedback": None,
                "graph_steps": ["dispatch", "start_session", "finalize"],
                "trace": {},
                "error": None,
            }

        payload = {
            "repo_path": ".",
            "provider": "deepseek",
            "request": {
                "action": "start",
                "learning_goal": LEARNING_GOAL,
                "learner_level": "beginner",
            },
        }
        with running_server() as (_, base_url):
            with patch("frontend.run_learning_frontend_request", side_effect=fake_run):
                status, body = post_json(f"{base_url}/api/learning/session", payload)

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "success")
        self.assertEqual(captured["payload"], payload)
        self.assertIsInstance(captured["signing_key"], bytes)
        self.assertGreaterEqual(len(captured["signing_key"]), 32)

    def test_business_failure_is_http_200_and_keeps_session(self):
        session = {"integrity_token": "b" * 64, "phase": "awaiting_submission"}
        failure = {
            "status": "failed",
            "action": "submit",
            "session": session,
            "current_step": {"step_id": "step-1"},
            "next_action": "submit",
            "review": None,
            "feedback": None,
            "graph_steps": ["dispatch", "review_submission", "finalize"],
            "trace": {},
            "error": {"code": "model_error", "message": "步骤审查失败。"},
        }
        with running_server() as (_, base_url):
            with patch("frontend.run_learning_frontend_request", return_value=failure):
                status, body = post_json(
                    f"{base_url}/api/learning/session",
                    {
                        "repo_path": ".",
                        "request": {
                            "action": "submit",
                            "session": session,
                            "submission": {
                                "implementation_summary": "已实现",
                                "test_output": "tests passed",
                            },
                        },
                    },
                )

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["session"], session)

    def test_body_larger_than_one_mib_returns_413(self):
        with running_server() as (server, _):
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_address[1],
                timeout=3,
            )
            try:
                connection.putrequest("POST", "/api/learning/session")
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(1024 * 1024 + 1))
                connection.endheaders()
                response = connection.getresponse()
                response.read()
            finally:
                connection.close()

        self.assertEqual(response.status, 413)


class LearningFrontendHtmlTest(unittest.TestCase):
    """固定页面内存、安全渲染和并发请求锁契约。"""

    def test_html_contains_learning_mode_and_in_memory_session_only(self):
        self.assertIn("项目学习", INDEX_HTML)
        self.assertIn('/api/learning/session', INDEX_HTML)
        self.assertIn("let learningSession = null", INDEX_HTML)
        self.assertNotIn("localStorage", INDEX_HTML)
        self.assertNotIn("sessionStorage", INDEX_HTML)

    def test_html_has_busy_lock_and_escaped_learning_rendering(self):
        self.assertIn("let requestInFlight = false", INDEX_HTML)
        self.assertIn("requestInFlight", INDEX_HTML)
        self.assertIn("renderLearning", INDEX_HTML)
        self.assertIn("escapeHtml", INDEX_HTML)
        self.assertIn('String(text ?? "")', INDEX_HTML)
        self.assertNotIn("execute_tools: executeTools.checked", learning_script_slice(INDEX_HTML))

    def test_invalid_session_without_replacement_resets_learning_to_start(self):
        learning_script = learning_script_slice(INDEX_HTML)

        self.assertIn('data.error.code === "invalid_session"', learning_script)
        self.assertIn("learningSession = null", learning_script)
        self.assertIn('learningNextAction = "start"', learning_script)

    def test_mode_switch_updates_top_k_limit_and_clamps_learning_value(self):
        self.assertIn('topKInput.max = isLearning ? "20" : "50"', INDEX_HTML)
        self.assertIn(
            'if (isLearning && Number(topKInput.value) > 20) topKInput.value = "20"',
            INDEX_HTML,
        )


def learning_script_slice(html: str) -> str:
    """只截取 Learning 请求函数，避免与原 /api/run 的授权字段混淆。"""
    start = html.find("async function runLearning")
    if start < 0:
        return ""
    end = html.find("function renderLearning", start)
    return html[start : end if end >= 0 else len(html)]


if __name__ == "__main__":
    unittest.main()
