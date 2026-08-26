"""Serve RepoPilot's unified Agent workflow and read-only Learning sessions."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.runtime import (
    UnifiedRunConfig,
    resume_unified_query,
    run_unified_query,
    unified_run_to_dict,
)
from coding_rag.learning import (
    LearningSessionDependencies,
    LearningWorkflowConfig,
    run_learning_session,
)
from coding_rag.rag.llm_client import OpenAICompatibleChatClient, build_llm_config
from coding_rag.rag.prompt import GenerationMode
from coding_rag.tools.env import load_dotenv


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BODY_BYTES = 1024 * 1024


class RequestBodyTooLarge(ValueError):
    """表示 HTTP 请求体在读取前已经超过公开上限。"""


class UnsupportedMediaType(ValueError):
    """表示 API 请求没有使用 application/json。"""


@dataclass
class PendingApproval:
    """Server-only state required to resume one exact write call."""

    run: Any
    config: UnifiedRunConfig
    client: Any


class PendingApprovalStore:
    """Thread-safe, one-use browser approval state store."""

    def __init__(self) -> None:
        """Initialize an empty process-local store."""
        self._items: dict[str, PendingApproval] = {}
        self._lock = Lock()

    def put(self, pending: PendingApproval) -> str:
        """Store one pending run and return an opaque browser handle."""
        approval_id = secrets.token_urlsafe(24)
        with self._lock:
            self._items[approval_id] = pending
        return approval_id

    def pop(self, approval_id: str) -> PendingApproval:
        """Consume one pending run exactly once."""
        with self._lock:
            pending = self._items.pop(approval_id, None)
        if pending is None:
            raise ValueError("approval request is missing, expired, or already used")
        return pending


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回 argparse.Namespace。"""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="RepoPilot browser frontend")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allow-tool-execution",
        action="store_true",
        help="允许网页请求在安全策略内应用补丁和执行命令",
    )
    return parser.parse_args()


class RepoPilotServer(ThreadingHTTPServer):
    """RepoPilot 多线程 HTTP 服务器。"""
    allow_tool_execution: bool = False
    # 每个 Python 服务进程只生成一次，既不注入 HTML，也不进入任何响应或日志。
    learning_signing_key: bytes = secrets.token_bytes(32)

    def __init__(self, *args, **kwargs):
        """Initialize HTTP service state, including one-time approvals."""
        super().__init__(*args, **kwargs)
        self.pending_approvals = PendingApprovalStore()


class RepoPilotHandler(BaseHTTPRequestHandler):
    """RepoPilot HTTP 请求处理器。"""
    def do_GET(self) -> None:
        """处理 GET 请求，返回首页 HTML。"""
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        self.send_html(build_index_html(self.server_allows_execution()))

    def do_POST(self) -> None:
        """分发普通查询与只读教学会话请求。"""
        path = urlparse(self.path).path
        if path not in {"/api/run", "/api/rollback", "/api/learning/session"}:
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            if path == "/api/learning/session":
                response = run_learning_frontend_request(
                    payload,
                    signing_key=self.server_learning_signing_key(),
                )
            elif path == "/api/rollback":
                response = run_frontend_rollback(
                    payload,
                    server_allows_execution=self.server_allows_execution(),
                )
            else:
                response = run_frontend_query(
                    payload,
                    server_allows_execution=self.server_allows_execution(),
                    approval_store=self.server.pending_approvals,
                )
        except RequestBodyTooLarge as error:
            self.send_json({"ok": False, "error": str(error)}, status=413)
            return
        except UnsupportedMediaType as error:
            self.send_json({"ok": False, "error": str(error)}, status=415)
            return
        except PermissionError as error:
            self.send_json({"ok": False, "error": str(error)}, status=403)
            return
        except ValueError as error:
            self.send_json({"ok": False, "error": str(error)}, status=400)
            return
        except RuntimeError as error:
            message = (
                "Learning Mode service is temporarily unavailable"
                if path == "/api/learning/session"
                else str(error)
            )
            self.send_json({"ok": False, "error": message}, status=502)
            return
        except Exception as error:
            message = (
                "Learning Mode request failed"
                if path == "/api/learning/session"
                else str(error)
            )
            self.send_json({"ok": False, "error": message}, status=500)
            return
        if path == "/api/learning/session":
            # 会话状态机的预期失败属于业务结果，始终保留 HTTP 200。
            self.send_json(response)
        else:
            self.send_json({"ok": True, **response})

    def server_allows_execution(self) -> bool:
        """读取服务器是否允许工具执行的开关。"""
        return bool(getattr(self.server, "allow_tool_execution", False))

    def server_learning_signing_key(self) -> bytes:
        """读取仅保存在服务端进程内的教学会话签名密钥。"""
        signing_key = getattr(self.server, "learning_signing_key", None)
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise RuntimeError("Learning Mode signing service is unavailable")
        return signing_key

    def read_json(self) -> dict:
        """只接受 application/json，并在读取前限制请求体大小。"""
        if self.headers.get_content_type() != "application/json":
            # JSON Content-Type 会触发跨站预检，避免普通网页以 simple request
            # 直接驱动本机已授权的工具或产生付费模型调用。
            raise UnsupportedMediaType("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError) as error:
            raise ValueError("Content-Length must be a non-negative integer") from error
        if length < 0:
            raise ValueError("Content-Length must be a non-negative integer")
        if length > MAX_REQUEST_BODY_BYTES:
            raise RequestBodyTooLarge("request body exceeds the 1 MiB limit")
        try:
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def send_html(self, html: str) -> None:
        """以 UTF-8 HTML 响应发送文本。"""
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        """以指定状态码发送 UTF-8 JSON 响应。"""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        """覆盖基类日志钩子以静默 HTTP 请求日志。"""
        return


def run_frontend_query(
    payload: dict,
    *,
    server_allows_execution: bool = False,
    approval_store: PendingApprovalStore | None = None,
) -> dict:
    """Start or approve one browser run without accepting blanket execution permission."""
    if "execute_tools" in payload:
        raise ValueError("execute_tools is no longer supported; approve each write call instead")
    approval_id = str(payload.get("approval_id") or "").strip()
    if approval_id:
        if not server_allows_execution:
            raise PermissionError("服务启动时未授权写工具，请使用 --allow-tool-execution")
        if approval_store is None:
            raise ValueError("browser approval store is unavailable")
        pending = approval_store.pop(approval_id)
        fingerprint = pending.run.approval["fingerprint"]
        run = resume_unified_query(
            pending.run,
            pending.config,
            pending.client,
            approval_fingerprint=fingerprint,
        )
        return prepare_frontend_run_response(
            run,
            config=pending.config,
            client=pending.client,
            server_allows_execution=server_allows_execution,
            approval_store=approval_store,
        )

    repo_path = str(payload.get("repo_path") or ".").strip()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    load_dotenv()
    llm_config = build_llm_config(
        provider=str(payload.get("provider") or "deepseek"),
        model=empty_to_none(payload.get("model")),
        base_url=empty_to_none(payload.get("base_url")),
        api_key_env=empty_to_none(payload.get("api_key_env")),
        timeout=int(payload.get("timeout") or 60),
        max_tokens=int(payload.get("max_tokens") or 2000),
        temperature=None,
    )
    client = OpenAICompatibleChatClient(llm_config)
    config = UnifiedRunConfig(
        repo_path=repo_path,
        top_k=int(payload.get("top_k") or 5),
        recall_window=int(payload.get("recall_window") or 2),
        generation_mode=GenerationMode(str(payload.get("mode") or "judge")),
        max_context_chars=int(payload.get("max_context_chars") or 12000),
        agent_policy_path=empty_to_none(payload.get("agent_policy")),
    )
    run = run_unified_query(query, config, client)
    return prepare_frontend_run_response(
        run,
        config=config,
        client=client,
        server_allows_execution=server_allows_execution,
        approval_store=approval_store,
    )


def prepare_frontend_run_response(
    run,
    *,
    config: UnifiedRunConfig,
    client: Any,
    server_allows_execution: bool,
    approval_store: PendingApprovalStore | None,
) -> dict:
    """Serialize a run and retain resumable state only on the server."""
    response = unified_run_to_dict(run)
    response["execution"]["server_allowed"] = server_allows_execution
    response["approval_id"] = None
    if run.status == "approval_required" and server_allows_execution and approval_store is not None:
        response["approval_id"] = approval_store.put(PendingApproval(run, config, client))
    return response


def run_frontend_rollback(payload: dict, *, server_allows_execution: bool) -> dict:
    """Restore one snapshot only when the server write capability is enabled."""
    if not server_allows_execution:
        raise PermissionError("服务启动时未授权恢复写操作")
    repo_path = str(payload.get("repo_path") or "").strip()
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    if not repo_path or not snapshot_id:
        raise ValueError("repo_path and snapshot_id are required")
    return AgentExecutor(repo_path).rollback_snapshot(snapshot_id)


def run_learning_frontend_request(payload: dict, *, signing_key: bytes) -> dict:
    """把浏览器外层配置适配成一次严格、只读的 Learning Session 调用。"""
    if not isinstance(payload, dict):
        raise ValueError("Learning Mode payload must be a JSON object")
    if "execute_tools" in payload:
        raise ValueError("Learning Mode does not accept execute_tools")

    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("Learning Mode request is required and must be a JSON object")

    timeout = frontend_integer(payload, "timeout", 60)
    max_tokens = frontend_integer(payload, "max_tokens", 6000)
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    workflow_config = LearningWorkflowConfig(
        repo_path=str(payload.get("repo_path") or ".").strip(),
        top_k=frontend_integer(payload, "top_k", 8),
        chunk_size=frontend_integer(payload, "chunk_size", 40),
        overlap=frontend_integer(payload, "overlap", 5),
        recall_window=frontend_integer(payload, "recall_window", 2),
        max_context_chars=frontend_integer(payload, "max_context_chars", 12000),
    )
    load_dotenv()
    llm_config = build_llm_config(
        provider=str(payload.get("provider") or "deepseek").strip(),
        model=empty_to_none(payload.get("model")),
        base_url=empty_to_none(payload.get("base_url")),
        api_key_env=empty_to_none(payload.get("api_key_env")),
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=None,
    )
    dependencies = LearningSessionDependencies(
        workflow_config=workflow_config,
        client=OpenAICompatibleChatClient(llm_config),
        signing_key=signing_key,
    )
    result = run_learning_session(request, dependencies)
    return result.model_dump(mode="json")


def frontend_integer(payload: dict, name: str, default: int) -> int:
    """读取浏览器整数配置，拒绝 bool 与不可解析值。"""
    raw_value = payload.get(name)
    if raw_value is None or raw_value == "":
        return default
    if isinstance(raw_value, bool) or isinstance(raw_value, float):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def empty_to_none(value) -> str | None:
    """把空白值归一化为 None。"""
    text = str(value or "").strip()
    return text or None


def configure_utf8_stdio() -> None:
    """Keep Chinese help text readable in Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def build_index_html(allow_tool_execution: bool = False) -> str:
    """把工具执行许可开关注入首页 HTML 模板。"""
    return INDEX_HTML_TEMPLATE.replace(
        "__ALLOW_TOOL_EXECUTION__",
        "true" if allow_tool_execution else "false",
    )


def main() -> None:
    """按命令行配置启动前端 HTTP 服务，并在退出时关闭服务器。"""
    args = parse_args()
    server = RepoPilotServer((args.host, args.port), RepoPilotHandler)
    server.allow_tool_execution = args.allow_tool_execution
    execution_label = "per-call approval enabled" if args.allow_tool_execution else "approval disabled"
    print(f"RepoPilot frontend: http://{args.host}:{args.port} (tool execution: {execution_label})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RepoPilot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #1b2530;
      --muted: #687584;
      --line: #d8dee5;
      --accent: #146c5a;
      --accent-strong: #0b5042;
      --code: #111a22;
      --danger: #a43d3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 58px;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      height: calc(100vh - 58px);
    }
    section {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .controls, .output { padding: 16px; overflow: auto; }
    label { display: block; margin: 12px 0 6px; font-weight: 600; }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    textarea { min-height: 130px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .hidden { display: none !important; }
    .mode-note {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 6px;
      background: #edf7f4;
      color: var(--accent-strong);
    }
    .compact textarea { min-height: 92px; }
    .toggle { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-weight: 600; }
    .toggle input { width: auto; }
    .toggle.disabled { color: var(--muted); }
    button {
      width: 100%;
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      padding: 11px 12px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .55; cursor: wait; }
    .status, .meta { color: var(--muted); font-size: 13px; }
    .block { border-top: 1px solid var(--line); padding: 14px 0; }
    .block:first-child { border-top: 0; padding-top: 0; }
    h2 { margin: 0 0 8px; font-size: 15px; }
    h3 { margin: 12px 0 6px; font-size: 14px; }
    ul, ol { margin: 7px 0; padding-left: 22px; }
    li { margin: 4px 0; }
    .badge {
      display: inline-block;
      margin-right: 6px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #e7eeec;
      color: var(--accent-strong);
      font-size: 12px;
    }
    pre {
      margin: 8px 0 0;
      padding: 12px;
      overflow: auto;
      border-radius: 6px;
      background: var(--code);
      color: #edf5f1;
      white-space: pre-wrap;
    }
    details.event { border-top: 1px solid var(--line); padding: 9px 0; }
    details.event:first-of-type { border-top: 0; }
    details.event summary { cursor: pointer; font-weight: 600; }
    .failed { color: var(--danger); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; height: auto; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>RepoPilot</h1>
    <div class="status" id="status">就绪</div>
  </header>
  <main>
    <section class="controls">
      <label for="appMode">使用模式</label>
      <select id="appMode">
        <option value="ask">代码问答</option>
        <option value="learning">项目学习</option>
      </select>
      <label for="repo">仓库路径</label>
      <input id="repo" value="." />
      <div id="askControls">
        <label for="query">问题或需求</label>
        <textarea id="query" placeholder="例如：解释 BM25 索引流程，或修复 ASK 没有调用 LLM 的问题"></textarea>
      </div>
      <div id="learningControls" class="hidden">
        <label for="learningGoal">学习目标</label>
        <textarea id="learningGoal" placeholder="例如：从零复现 RepoPilot，并理解每个模块为什么这样设计"></textarea>
        <div class="mode-note">教学模式始终只读。你粘贴的代码、命令和测试输出只作为学习证据，不会被执行。</div>
        <div id="learningSubmitControls" class="compact hidden">
          <label for="implementationSummary">本步实现说明</label>
          <textarea id="implementationSummary" placeholder="说明你改了哪些文件、如何实现，以及为什么这样做"></textarea>
          <label for="testOutput">已运行的测试输出</label>
          <textarea id="testOutput" placeholder="粘贴你自己运行测试后得到的输出"></textarea>
        </div>
        <div id="learningReflectControls" class="compact hidden">
          <label for="reflection">优化问题回答</label>
          <textarea id="reflection" placeholder="结合当前步骤回答右侧显示的优化问题"></textarea>
        </div>
      </div>
      <div class="row">
        <div>
          <label for="topK">top-k</label>
          <input id="topK" type="number" min="1" max="50" value="5" />
        </div>
        <div>
          <label for="recall">recall-window</label>
          <input id="recall" type="number" min="0" max="10" value="2" />
        </div>
      </div>
      <div class="row">
        <div>
          <label for="provider">提供商</label>
          <select id="provider">
            <option>deepseek</option>
            <option>qwen</option>
            <option>kimi</option>
            <option>zhipu</option>
            <option>custom</option>
          </select>
        </div>
        <div id="answerModeField">
          <label for="mode">回答风格</label>
          <select id="mode">
            <option value="judge">judge</option>
            <option value="code-understand">code-understand</option>
            <option value="code-generate">code-generate</option>
            <option value="api">api</option>
            <option value="leetcode">leetcode</option>
          </select>
        </div>
      </div>
      <div class="mode-note" id="writePolicyNote">写工具会逐调用列出文件并等待一次性批准。</div>
      <button id="run">运行</button>
    </section>
    <section class="output" id="output">
      <div class="status">输入问题或需求后运行。</div>
    </section>
  </main>
  <script>
    const serverAllowsExecution = __ALLOW_TOOL_EXECUTION__;
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const runBtn = document.getElementById("run");
    const appMode = document.getElementById("appMode");
    const askControls = document.getElementById("askControls");
    const learningControls = document.getElementById("learningControls");
    const learningSubmitControls = document.getElementById("learningSubmitControls");
    const learningReflectControls = document.getElementById("learningReflectControls");
    const answerModeField = document.getElementById("answerModeField");
    const topKInput = document.getElementById("topK");
    let requestInFlight = false;
    let learningSession = null;
    let learningNextAction = "start";
    let learningLastResult = null;

    if (!serverAllowsExecution) {
      document.getElementById("writePolicyNote").textContent = "服务未启用写能力；写请求只会展示，不可批准执行。";
    }
    appMode.addEventListener("change", updateMode);
    runBtn.addEventListener("click", runTask);
    document.getElementById("query").addEventListener("keydown", event => {
      if (event.ctrlKey && event.key === "Enter") runTask();
    });
    document.getElementById("learningGoal").addEventListener("keydown", event => {
      if (event.ctrlKey && event.key === "Enter") runTask();
    });
    updateMode();

    function runTask() {
      if (appMode.value === "learning") return runLearningTask();
      return runAskTask();
    }

    async function runAskTask() {
      if (requestInFlight) return;
      requestInFlight = true;
      appMode.disabled = true;
      runBtn.disabled = true;
      statusEl.textContent = "处理中...";
      outputEl.innerHTML = '<div class="status">LLM 正在判断是否需要调用工具...</div>';
      const payload = {
        repo_path: value("repo"),
        query: value("query"),
        top_k: Number(value("topK")),
        recall_window: Number(value("recall")),
        provider: value("provider"),
        mode: value("mode"),
      };
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || `请求失败 (${response.status})`);
        if (appMode.value === "ask") {
          renderRun(data);
          statusEl.textContent = data.status === "approval_required"
            ? "等待批准"
            : (data.status === "failed" ? "失败" : "完成");
        }
      } catch (error) {
        if (appMode.value === "ask") {
          outputEl.innerHTML = `<div class="block"><h2>执行失败</h2><pre>${escapeHtml(error.message)}</pre></div>`;
          statusEl.textContent = "失败";
        }
      } finally {
        requestInFlight = false;
        appMode.disabled = false;
        if (appMode.value === "ask") runBtn.disabled = false;
        else syncLearningControls();
      }
    }

    async function approveWrite(approvalId) {
      if (requestInFlight || !approvalId) return;
      requestInFlight = true;
      runBtn.disabled = true;
      statusEl.textContent = "正在执行已批准的这一次写操作...";
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({approval_id: approvalId}),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || `请求失败 (${response.status})`);
        renderRun(data);
        statusEl.textContent = data.status === "approval_required" ? "等待下一次批准" : "完成";
      } catch (error) {
        outputEl.innerHTML = `<div class="block"><h2>执行失败</h2><pre>${escapeHtml(error.message)}</pre></div>`;
        statusEl.textContent = "失败";
      } finally {
        requestInFlight = false;
        runBtn.disabled = false;
      }
    }

    async function rollbackWrite(snapshotId) {
      if (requestInFlight || !snapshotId) return;
      requestInFlight = true;
      statusEl.textContent = "正在恢复修改前快照...";
      try {
        const response = await fetch("/api/rollback", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({repo_path: value("repo"), snapshot_id: snapshotId}),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || `恢复失败 (${response.status})`);
        statusEl.textContent = "已恢复";
        outputEl.insertAdjacentHTML("afterbegin", `<div class="block"><h2>已恢复快照</h2><pre>${escapeHtml(snapshotId)}</pre></div>`);
      } catch (error) {
        statusEl.textContent = "恢复失败";
        outputEl.insertAdjacentHTML("afterbegin", `<div class="block failed"><h2>恢复失败</h2><pre>${escapeHtml(error.message)}</pre></div>`);
      } finally {
        requestInFlight = false;
      }
    }

    window.approveWrite = approveWrite;
    window.rollbackWrite = rollbackWrite;

    async function runLearningTask() {
      if (requestInFlight || learningNextAction === "none") return;
      const action = learningSession ? learningNextAction : "start";
      const request = buildLearningRequest(action);
      const payload = {
        repo_path: value("repo"),
        provider: value("provider"),
        top_k: Number(value("topK")),
        recall_window: Number(value("recall")),
        request,
      };

      requestInFlight = true;
      appMode.disabled = true;
      syncLearningControls();
      statusEl.textContent = "教学处理中...";
      outputEl.innerHTML = '<div class="status">正在只读分析学习证据并推进到下一个等待点...</div>';
      try {
        const response = await fetch("/api/learning/session", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);

        if (data.error && data.error.code === "invalid_session" && !data.session) {
          learningSession = null;
          learningNextAction = "start";
        } else if (data.session) {
          learningSession = data.session;
          learningNextAction = data.next_action || "none";
        } else if (!learningSession) {
          learningNextAction = "start";
        }
        learningLastResult = data;
        clearCompletedLearningInput(action, data);
        if (appMode.value === "learning") {
          renderLearning(data);
          statusEl.textContent = data.status === "success" ? learningStatusText(data) : "需要处理";
        }
      } catch (error) {
        if (appMode.value === "learning") {
          outputEl.innerHTML = `<div class="block"><h2>教学请求失败</h2><pre>${escapeHtml(error.message)}</pre></div>`;
          statusEl.textContent = "失败";
        }
      } finally {
        requestInFlight = false;
        appMode.disabled = false;
        if (appMode.value === "learning") syncLearningControls();
      }
    }

    function buildLearningRequest(action) {
      if (action === "start") {
        return {
          action: "start",
          learning_goal: value("learningGoal"),
          learner_level: "beginner",
        };
      }
      if (action === "submit") {
        return {
          action: "submit",
          session: learningSession,
          submission: {
            implementation_summary: value("implementationSummary"),
            test_output: value("testOutput"),
          },
        };
      }
      return {
        action: "reflect",
        session: learningSession,
        reflection: value("reflection"),
      };
    }

    function clearCompletedLearningInput(action, data) {
      if (data.status !== "success" || !data.session) return;
      if (action === "submit" && data.session.phase === "awaiting_reflection") {
        document.getElementById("implementationSummary").value = "";
        document.getElementById("testOutput").value = "";
      }
      if (action === "reflect") {
        document.getElementById("reflection").value = "";
        document.getElementById("implementationSummary").value = "";
        document.getElementById("testOutput").value = "";
      }
    }

    function updateMode() {
      const isLearning = appMode.value === "learning";
      askControls.classList.toggle("hidden", isLearning);
      learningControls.classList.toggle("hidden", !isLearning);
      answerModeField.classList.toggle("hidden", isLearning);
      document.getElementById("writePolicyNote").classList.toggle("hidden", isLearning);
      topKInput.max = isLearning ? "20" : "50";
      if (isLearning && Number(topKInput.value) > 20) topKInput.value = "20";
      if (isLearning) {
        if (learningLastResult) {
          renderLearning(learningLastResult);
          statusEl.textContent = learningLastResult.status === "success"
            ? learningStatusText(learningLastResult)
            : "需要处理";
        } else {
          outputEl.innerHTML = '<div class="status">填写学习目标，RepoPilot 将先拆解项目并给出第一步。</div>';
          statusEl.textContent = "就绪";
        }
        syncLearningControls();
      } else {
        runBtn.textContent = "运行";
        runBtn.disabled = requestInFlight;
        statusEl.textContent = "就绪";
        outputEl.innerHTML = '<div class="status">输入问题或需求后运行。</div>';
      }
    }

    function syncLearningControls() {
      const action = learningSession ? learningNextAction : "start";
      const showSubmit = action === "submit";
      const showReflect = action === "reflect";
      learningSubmitControls.classList.toggle("hidden", !showSubmit);
      learningReflectControls.classList.toggle("hidden", !showReflect);
      document.getElementById("learningGoal").disabled = Boolean(learningSession);
      runBtn.textContent = ({
        start: "生成学习路线",
        submit: "提交本步证据",
        reflect: "提交优化回答并继续",
        none: "学习路线已完成",
      })[action] || "继续";
      runBtn.disabled = requestInFlight || action === "none";
    }

    function renderRun(data) {
      const summary = data.summary || {};
      const execution = data.execution || {};
      const approval = data.approval || null;
      const approvalButton = approval && data.approval_id ? `
        <button onclick="approveWrite('${escapeHtml(data.approval_id)}')">批准并执行这一次写操作</button>
      ` : (approval ? '<div class="meta">服务未启用写能力，无法批准执行。</div>' : '');
      const approvalArguments = approval && approval.arguments ? approval.arguments : {};
      const approvalDetail = approval && approval.tool === "run_command"
        ? `<h3>命令</h3><pre>${escapeHtml(approvalArguments.cmd || "")}</pre>`
        : (approval && approval.tool === "apply_patch"
          ? `<h3>待应用补丁</h3><pre>${escapeHtml(approvalArguments.diff || "")}</pre>`
          : "");
      const approvalBlock = approval ? `
        <div class="block">
          <h2>等待写操作批准</h2>
          <div class="meta">tool=${escapeHtml(approval.tool || "unknown")}</div>
          <h3>将修改的文件</h3>
          ${renderList(approval.files || [])}
          ${approvalDetail}
          <div class="meta">一次性指纹=${escapeHtml(approval.fingerprint || "missing")}</div>
          ${approvalButton}
        </div>
      ` : "";
      const traceEvents = data.trace && Array.isArray(data.trace.events) ? data.trace.events : [];
      const writeResults = traceEvents.map(event => {
        const artifacts = event && event.artifacts ? event.artifacts : {};
        return artifacts.patch || artifacts.command || null;
      }).filter(result => result && result.snapshot_id);
      const changesBlock = writeResults.map(output => `
        <div class="block">
          <h2>已执行修改</h2>
          <div class="meta">snapshot=${escapeHtml(output.snapshot_id)}</div>
          <h3>修改后 diff</h3>
          <pre>${escapeHtml((output.post_change_diff && output.post_change_diff.text) || "（声明文件未检测到文本变化）")}</pre>
          <button onclick="rollbackWrite('${escapeHtml(output.snapshot_id)}')">恢复到修改前</button>
        </div>
      `).join("");
      outputEl.innerHTML = `
        <div class="block">
          <h2>回答</h2>
          <div class="meta">status=${escapeHtml(data.status)} write_executed=${execution.enabled ? "yes" : "no"}</div>
          <pre>${escapeHtml(data.answer)}</pre>
        </div>
        ${approvalBlock}
        ${changesBlock}
        <div class="block">
          <h2>运行统计</h2>
          <div class="meta">LLM=${escapeHtml(summary.llm_call_count || 0)} tools=${escapeHtml(summary.tool_call_count || 0)} observations=${escapeHtml(summary.observation_count || 0)}</div>
        </div>
        ${renderTrace(data.trace)}
        ${renderSearchResults(data.trace)}
      `;
    }

    function renderTrace(trace) {
      if (!trace || !Array.isArray(trace.events)) return "";
      const events = trace.events.map((event, index) => {
        const failedClass = event.status === "failed" ? "failed" : "";
        const detail = JSON.stringify({
          input: event.input || {},
          output_summary: event.output_summary || {},
          artifacts: event.artifacts || {},
          error: event.error || null,
          duration_ms: event.duration_ms,
        }, null, 2);
        return `<details class="event">
          <summary class="${failedClass}">${escapeHtml(index + 1)}. ${escapeHtml(event.step)} [${escapeHtml(event.status)}]</summary>
          <pre>${escapeHtml(detail)}</pre>
        </details>`;
      }).join("");
      return `<div class="block"><h2>Trace Events</h2>${events}</div>`;
    }

    function renderSearchResults(trace) {
      if (!trace || !Array.isArray(trace.events)) return "";
      const rows = [];
      trace.events.filter(event => event.step === "search_code").forEach(event => {
        const result = event.artifacts && event.artifacts.result;
        if (result && Array.isArray(result.results)) rows.push(...result.results);
      });
      if (!rows.length) return "";
      const html = rows.map(item => `<div class="block">
        <h2>#${escapeHtml(item.rank)} ${escapeHtml(item.path)}:${escapeHtml(item.start_line)}-${escapeHtml(item.end_line)}</h2>
        <div class="meta">score=${escapeHtml(Number(item.score || 0).toFixed(4))} source=${escapeHtml(item.source)}</div>
        <pre>${escapeHtml(item.text)}</pre>
      </div>`).join("");
      return `<div class="block"><h2>代码证据</h2>${html}</div>`;
    }

    function renderLearning(data) {
      const session = data.session || learningSession;
      const currentStep = data.current_step || currentStepFromSession(session);
      const error = data.error || null;
      const errorBlock = error ? `<div class="block failed">
        <h2>本次未推进</h2>
        <div>${escapeHtml(error.message)}</div>
        <div class="meta">code=${escapeHtml(error.code)}</div>
      </div>` : "";
      const phaseBlock = session ? `<div class="block">
        <h2>学习进度</h2>
        <span class="badge">${escapeHtml(session.phase)}</span>
        <span class="meta">第 ${escapeHtml(Number(session.current_step_index || 0) + 1)} / ${escapeHtml(session.plan && session.plan.steps ? session.plan.steps.length : 0)} 步</span>
      </div>` : "";
      const completedBlock = session && session.phase === "completed" ? `<div class="block">
        <h2>路线完成</h2>
        <p>你已经完成全部复现步骤和优化思考。可以刷新页面后输入新目标，开始另一条学习路线。</p>
      </div>` : "";

      outputEl.innerHTML = `
        ${errorBlock}
        ${phaseBlock}
        ${session ? renderProjectProfile(session.plan && session.plan.project_profile) : ""}
        ${session ? renderPlanOverview(session.plan, session.current_step_index, session.phase) : ""}
        ${currentStep ? renderLearningStep(currentStep) : ""}
        ${renderLearningReview(data.review)}
        ${renderLearningFeedback(data.feedback)}
        ${completedBlock}
      ` || '<div class="status">尚未创建学习路线。</div>';
      syncLearningControls();
    }

    function renderProjectProfile(profile) {
      if (!profile) return "";
      const components = Array.isArray(profile.components) ? profile.components.map(component => `
        <div>
          <h3>${escapeHtml(component.name)}</h3>
          <div>${escapeHtml(component.responsibility)}</div>
          ${renderEvidence(component.evidence)}
        </div>
      `).join("") : "";
      return `<div class="block">
        <h2>项目画像：${escapeHtml(profile.project_name)}</h2>
        <p>${escapeHtml(profile.summary)}</p>
        <h3>技术栈</h3>${renderList(profile.tech_stack)}
        <h3>开始前准备</h3>${renderList(profile.prerequisites)}
        <h3>程序入口</h3>${renderEvidence(profile.entry_points)}
        <h3>核心组件</h3>${components}
      </div>`;
    }

    function renderPlanOverview(plan, currentIndex, phase) {
      if (!plan || !Array.isArray(plan.steps)) return "";
      const items = plan.steps.map((step, index) => {
        const marker = phase === "completed" || index < currentIndex
          ? "已完成"
          : (index === currentIndex ? "当前" : "待完成");
        return `<li><span class="badge">${escapeHtml(marker)}</span>${escapeHtml(step.title)}</li>`;
      }).join("");
      return `<div class="block"><h2>复现路线</h2><ol>${items}</ol></div>`;
    }

    function renderLearningStep(step) {
      return `<div class="block">
        <h2>当前步骤：${escapeHtml(step.title)}</h2>
        <div class="meta">step_id=${escapeHtml(step.step_id)} dependencies=${escapeHtml((step.depends_on || []).join(", ") || "无")}</div>
        <h3>学习目标</h3><p>${escapeHtml(step.learning_goal)}</p>
        <h3>需要创建的文件</h3>${renderList(step.files_to_create)}
        <h3>任务</h3>${renderList(step.tasks)}
        <h3>为什么这样做</h3><p>${escapeHtml(step.why)}</p>
        <h3>收益</h3>${renderList(step.benefits)}
        <h3>验收项</h3>${renderList(step.verification)}
        <h3>源码证据</h3>${renderEvidence(step.evidence)}
        <h3>常见问题</h3>${renderList(step.common_pitfalls)}
        <h3>优化问题</h3><p>${escapeHtml(step.optimization_question)}</p>
      </div>`;
    }

    function renderLearningReview(review) {
      if (!review) return "";
      const findings = Array.isArray(review.findings) ? review.findings.map(finding => `
        <li>${escapeHtml(Number(finding.verification_index || 0) + 1)}. ${finding.satisfied ? "满足" : "未满足"}：${escapeHtml(finding.reason)}</li>
      `).join("") : "";
      return `<div class="block">
        <h2>步骤审查：${review.passed ? "通过" : "需要修改"}</h2>
        <div class="meta">范围=${escapeHtml(review.verification_scope)}</div>
        <ul>${findings}</ul>
      </div>`;
    }

    function renderLearningFeedback(feedback) {
      if (!feedback) return "";
      return `<div class="block">
        <h2>最小修改提示</h2>
        <h3>仍缺少</h3>${renderList(feedback.gaps)}
        <h3>提示</h3><p>${escapeHtml(feedback.hint)}</p>
      </div>`;
    }

    function renderEvidence(items) {
      if (!Array.isArray(items) || !items.length) return '<div class="meta">暂无</div>';
      return `<ul>${items.map(item => `<li>
        <strong>${escapeHtml(item.path)}:${escapeHtml(item.start_line)}-${escapeHtml(item.end_line)}</strong> — ${escapeHtml(item.claim)}
      </li>`).join("")}</ul>`;
    }

    function renderList(items) {
      if (!Array.isArray(items) || !items.length) return '<div class="meta">暂无</div>';
      return `<ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function currentStepFromSession(session) {
      if (!session || session.phase === "completed" || !session.plan || !Array.isArray(session.plan.steps)) return null;
      return session.plan.steps[Number(session.current_step_index || 0)] || null;
    }

    function learningStatusText(data) {
      if (data.next_action === "submit") return "等待步骤提交";
      if (data.next_action === "reflect") return "等待优化回答";
      return "学习路线完成";
    }

    function value(id) { return document.getElementById(id).value.trim(); }
    function escapeHtml(text) {
      return String(text ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }
  </script>
</body>
</html>
"""


INDEX_HTML = build_index_html(False)


if __name__ == "__main__":
    main()
