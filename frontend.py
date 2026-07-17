"""RepoPilot browser frontend for the unified LLM tool workflow."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query, unified_run_to_dict
from coding_rag.rag.llm_client import OpenAICompatibleChatClient, build_llm_config
from coding_rag.rag.prompt import GenerationMode
from coding_rag.tools.env import load_dotenv


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


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


class RepoPilotHandler(BaseHTTPRequestHandler):
    """RepoPilot HTTP 请求处理器。"""
    def do_GET(self) -> None:
        """处理 GET 请求，返回首页 HTML。"""
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        self.send_html(build_index_html(self.server_allows_execution()))

    def do_POST(self) -> None:
        """处理 /api/run 的 POST 请求，解析 JSON 并执行前端查询。"""
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        try:
            response = run_frontend_query(
                self.read_json(),
                server_allows_execution=self.server_allows_execution(),
            )
        except PermissionError as error:
            self.send_json({"ok": False, "error": str(error)}, status=403)
            return
        except ValueError as error:
            self.send_json({"ok": False, "error": str(error)}, status=400)
            return
        except RuntimeError as error:
            self.send_json({"ok": False, "error": str(error)}, status=502)
            return
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, status=500)
            return
        self.send_json({"ok": True, **response})

    def server_allows_execution(self) -> bool:
        """读取服务器是否允许工具执行的开关。"""
        return bool(getattr(self.server, "allow_tool_execution", False))

    def read_json(self) -> dict:
        """按 Content-Length 读取并解析 UTF-8 JSON 请求体。"""

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

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


def run_frontend_query(payload: dict, *, server_allows_execution: bool = False) -> dict:
    """校验前端请求、执行统一查询，仅当服务器已授权时才允许请求启用工具执行。"""
    repo_path = str(payload.get("repo_path") or ".").strip()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    execute_tools = bool(payload.get("execute_tools"))
    if execute_tools and not server_allows_execution:
        raise PermissionError("服务启动时未授权工具执行，请使用 --allow-tool-execution")

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
    run = run_unified_query(
        query,
        UnifiedRunConfig(
            repo_path=repo_path,
            top_k=int(payload.get("top_k") or 5),
            recall_window=int(payload.get("recall_window") or 2),
            generation_mode=GenerationMode(str(payload.get("mode") or "judge")),
            max_context_chars=int(payload.get("max_context_chars") or 12000),
            agent_policy_path=empty_to_none(payload.get("agent_policy")),
        ),
        client,
        execute_tools=execute_tools,
    )
    response = unified_run_to_dict(run)
    response["execution"]["server_allowed"] = server_allows_execution
    return response


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
    execution_label = "enabled" if args.allow_tool_execution else "dry-run only"
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
      <label for="repo">仓库路径</label>
      <input id="repo" value="." />
      <label for="query">问题或需求</label>
      <textarea id="query" placeholder="例如：解释 BM25 索引流程，或修复 ASK 没有调用 LLM 的问题"></textarea>
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
        <div>
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
      <label class="toggle" id="executeLabel" title="需要服务启动时同时传入 --allow-tool-execution">
        <input id="executeTools" type="checkbox" /> 允许执行工具
      </label>
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
    const executeTools = document.getElementById("executeTools");
    const executeLabel = document.getElementById("executeLabel");

    if (!serverAllowsExecution) {
      executeTools.disabled = true;
      executeLabel.classList.add("disabled");
      executeLabel.title = "服务未使用 --allow-tool-execution 启动";
    }
    runBtn.addEventListener("click", runTask);
    document.getElementById("query").addEventListener("keydown", event => {
      if (event.ctrlKey && event.key === "Enter") runTask();
    });

    async function runTask() {
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
        execute_tools: executeTools.checked,
      };
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || `请求失败 (${response.status})`);
        renderRun(data);
        statusEl.textContent = data.status === "failed" ? "失败" : "完成";
      } catch (error) {
        outputEl.innerHTML = `<div class="block"><h2>执行失败</h2><pre>${escapeHtml(error.message)}</pre></div>`;
        statusEl.textContent = "失败";
      } finally {
        runBtn.disabled = false;
      }
    }

    function renderRun(data) {
      const summary = data.summary || {};
      const execution = data.execution || {};
      outputEl.innerHTML = `
        <div class="block">
          <h2>回答</h2>
          <div class="meta">status=${escapeHtml(data.status)} execution=${execution.enabled ? "enabled" : "dry-run"}</div>
          <pre>${escapeHtml(data.answer)}</pre>
        </div>
        <div class="block">
          <h2>运行统计</h2>
          <div class="meta">LLM=${summary.llm_call_count || 0} tools=${summary.tool_call_count || 0} observations=${summary.observation_count || 0}</div>
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
          <summary class="${failedClass}">${index + 1}. ${escapeHtml(event.step)} [${escapeHtml(event.status)}]</summary>
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
        <h2>#${item.rank} ${escapeHtml(item.path)}:${item.start_line}-${item.end_line}</h2>
        <div class="meta">score=${Number(item.score || 0).toFixed(4)} source=${escapeHtml(item.source)}</div>
        <pre>${escapeHtml(item.text)}</pre>
      </div>`).join("");
      return `<div class="block"><h2>代码证据</h2>${html}</div>`;
    }

    function value(id) { return document.getElementById(id).value.trim(); }
    function escapeHtml(text) {
      return String(text || "").replace(/[&<>"']/g, char => ({
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
