"""RepoPilot browser frontend.

Run:
    python frontend.py
Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from coding_rag.agent.plan_reviewer import review_agent_plan
from coding_rag.tools.env import ensure_dotenv, load_dotenv
from coding_rag.rag.llm_client import OpenAICompatibleChatClient, build_llm_config
from coding_rag.agent.planner import (
    AgentPlanConfig,
    AskModeConfig,
    ReActAgentInterface,
    WorkflowMode,
    render_agent_plan_prompt,
    render_agent_plan_run,
    render_task_plan,
    run_ask_mode,
)
from coding_rag.rag.prompt import GenerationMode


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoPilot browser frontend")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


class RepoPilotHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        self.send_html(INDEX_HTML)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/ask", "/api/agent-plan"):
            self.send_error(404)
            return
        try:
            payload = self.read_json()
            if path == "/api/agent-plan":
                response = run_frontend_agent_plan(payload)
            else:
                response = run_frontend_ask(payload)
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, status=500)
            return
        self.send_json({"ok": True, **response})

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def run_frontend_ask(payload: dict) -> dict:
    repo_path = str(payload.get("repo_path") or ".").strip()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    use_llm = bool(payload.get("use_llm"))
    client = None
    if use_llm:
        ensure_dotenv()
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

    run = run_ask_mode(
        query=query,
        config=AskModeConfig(
            repo_path=repo_path,
            top_k=int(payload.get("top_k") or 5),
            recall_window=int(payload.get("recall_window") or 2),
            generation_mode=GenerationMode(str(payload.get("mode") or "judge")),
            max_context_chars=int(payload.get("max_context_chars") or 12000),
        ),
        client=client,
        requested_mode=WorkflowMode.ASK,
    )
    return {
        "plan": render_task_plan(run.plan),
        "trace": run.trace,
        "stats": {
            "seed_count": len(run.seed_results),
            "recalled_count": len(run.recalled_results),
            "final_count": len(run.final_results),
        },
        "answer": run.answer,
        "results": serialize_results(run.final_results),
    }


def run_frontend_agent_plan(payload: dict) -> dict:
    repo_path = str(payload.get("repo_path") or ".").strip()
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("task is required")

    use_llm = bool(payload.get("use_llm"))
    client = None
    if use_llm:
        ensure_dotenv()
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

    config = AgentPlanConfig(
        repo_path=repo_path,
        top_k=int(payload.get("top_k") or 5),
        recall_window=int(payload.get("recall_window") or 2),
        max_context_chars=int(payload.get("max_context_chars") or 12000),
        execute_readonly_tools=bool(payload.get("execute_readonly_tools")),
        agent_policy_path=empty_to_none(payload.get("agent_policy")),
    )

    run = ReActAgentInterface().build_plan(task=query, config=config, client=client)

    review = None
    review_agent_plan_flag = bool(payload.get("review_agent_plan"))
    if review_agent_plan_flag:
        if client is None:
            raise ValueError("Agent plan review requires LLM to be enabled")
        review_result = review_agent_plan(run, client)
        review = {
            "verdict": review_result.verdict,
            "score": review_result.score,
            "passed": review_result.passed,
            "issues": review_result.issues,
            "suggestions": review_result.suggestions,
        }

    tools_serialized = [
        {"name": t.name, "purpose": t.purpose, "input_schema": t.input_schema}
        for t in run.tools
    ]

    return {
        "plan": render_agent_plan_run(run),
        "trace": run.trace,
        "prompt": render_agent_plan_prompt(run.messages) if use_llm else None,
        "plan_text": run.plan_text,
        "tools": tools_serialized,
        "observations": run.observations,
        "stats": {
            "seed_count": len(run.seed_results),
            "recalled_count": len(run.recalled_results),
            "final_count": len(run.final_results),
        },
        "review": review,
    }


def empty_to_none(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def serialize_results(results) -> list[dict]:
    rows = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        rows.append(
            {
                "rank": index,
                "score": result.score,
                "source": result.source,
                "path": str(chunk.file_path),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text.rstrip(),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RepoPilotHandler)
    print(f"RepoPilot frontend: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RepoPilot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d2430;
      --muted: #657080;
      --line: #d9dee7;
      --accent: #176b5d;
      --accent-strong: #0e4f45;
      --code: #101820;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      height: calc(100vh - 58px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .controls { padding: 16px; overflow: auto; }
    label { display: block; font-weight: 600; margin: 12px 0 6px; }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--ink);
      font-weight: 600;
    }
    .toggle input { width: auto; }
    button {
      margin-top: 16px;
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 11px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .55; cursor: wait; }
    .output {
      overflow: auto;
      padding: 16px;
    }
    .status { color: var(--muted); font-size: 13px; }
    .answer, .plan, .result {
      border-top: 1px solid var(--line);
      padding: 14px 0;
    }
    .answer:first-child, .plan:first-child, .result:first-child { border-top: 0; }
    h2 { font-size: 15px; margin: 0 0 8px; }
    pre {
      margin: 8px 0 0;
      padding: 12px;
      overflow: auto;
      background: var(--code);
      color: #eef5f2;
      border-radius: 6px;
      white-space: pre-wrap;
    }
    .meta { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; height: auto; }
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
      <label for="workflowMode">工作流模式</label>
      <select id="workflowMode">
        <option value="ask">ASK</option>
        <option value="agent">Agent</option>
      </select>
      <label for="repo">仓库路径</label>
      <input id="repo" value="." />
      <label for="query" id="queryLabel">问题</label>
      <textarea id="query" placeholder="例如：BM25 检索器在哪里建立索引？"></textarea>
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
      <label class="toggle"><input id="useLlm" type="checkbox" /> 调用大模型回答</label>
      <label class="toggle" id="readonlyToolsToggle" style="display:none"><input id="executeReadonlyTools" type="checkbox" /> 执行只读工具观察</label>
      <label class="toggle" id="reviewToggle" style="display:none"><input id="reviewAgentPlan" type="checkbox" /> Agent 计划审查</label>
      <div class="row" id="modeRow">
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
          <label for="mode">模式</label>
          <select id="mode">
            <option value="judge">judge</option>
            <option value="code-understand">code-understand</option>
            <option value="code-generate">code-generate</option>
            <option value="api">api</option>
            <option value="leetcode">leetcode</option>
          </select>
        </div>
      </div>
      <button id="run">运行</button>
    </section>
    <section class="output" id="output">
      <div class="status">输入仓库路径和问题后点击运行。</div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const runBtn = document.getElementById("run");

    runBtn.addEventListener("click", runTask);
    document.getElementById("query").addEventListener("keydown", event => {
      if (event.ctrlKey && event.key === "Enter") runTask();
    });
    document.getElementById("workflowMode").addEventListener("change", onModeChange);
    onModeChange();

    function onModeChange() {
      const isAgent = value("workflowMode") === "agent";
      document.getElementById("queryLabel").textContent = isAgent ? "任务" : "问题";
      document.getElementById("query").placeholder = isAgent
        ? "例如：修复 ASK 模式没有调用大模型回答的问题"
        : "例如：BM25 检索器在哪里建立索引？";
      document.getElementById("modeRow").style.display = isAgent ? "none" : "";
      document.getElementById("readonlyToolsToggle").style.display = isAgent ? "" : "none";
      document.getElementById("reviewToggle").style.display = isAgent ? "" : "none";
      runBtn.textContent = isAgent ? "生成 Agent 计划" : "运行";
    }

    async function runTask() {
      const isAgent = value("workflowMode") === "agent";
      runBtn.disabled = true;
      statusEl.textContent = "处理中...";
      outputEl.innerHTML = isAgent
        ? '<div class="status">正在检索并生成 Agent 计划...</div>'
        : '<div class="status">正在检索并整理 RAG 上下文...</div>';
      const payload = {
        repo_path: value("repo"),
        query: value("query"),
        top_k: Number(value("topK")),
        recall_window: Number(value("recall")),
        use_llm: document.getElementById("useLlm").checked,
        provider: value("provider"),
        mode: value("mode"),
      };
      if (isAgent) {
        payload.execute_readonly_tools = document.getElementById("executeReadonlyTools").checked;
        payload.review_agent_plan = document.getElementById("reviewAgentPlan").checked;
      }
      const endpoint = isAgent ? "/api/agent-plan" : "/api/ask";
      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "请求失败");
        isAgent ? renderAgentPlan(data) : renderAsk(data);
        statusEl.textContent = "完成";
      } catch (error) {
        outputEl.innerHTML = `<div class="answer"><h2>执行失败</h2><pre>${escapeHtml(error.message)}</pre></div>`;
        statusEl.textContent = "失败";
      } finally {
        runBtn.disabled = false;
      }
    }

    function renderAsk(data) {
      const stats = data.stats;
      const answer = data.answer
        ? `<div class="answer"><h2>大模型回答</h2><pre>${escapeHtml(data.answer)}</pre></div>`
        : `<div class="answer"><h2>大模型回答</h2><div class="status">未启用大模型。勾选 调用大模型回答 后会基于 RAG 结果生成答案。</div></div>`;
      const results = data.results.map(item => `
        <div class="result">
          <h2>#${item.rank} ${escapeHtml(item.path)}:${item.start_line}-${item.end_line}</h2>
          <div class="meta">score=${item.score.toFixed(4)} source=${escapeHtml(item.source)}</div>
          <pre>${escapeHtml(item.text)}</pre>
        </div>
      `).join("");
      outputEl.innerHTML = `
        ${answer}
        <div class="plan"><h2>任务计划</h2><pre>${escapeHtml(data.plan)}</pre></div>
        <div class="plan"><h2>检索统计</h2><div class="meta">seed=${stats.seed_count} recalled=${stats.recalled_count} final=${stats.final_count}</div></div>
        ${renderTrace(data.trace)}
        ${results}
      `;
    }

    function renderAgentPlan(data) {
      const stats = data.stats;
      let reviewHtml = "";
      if (data.review) {
        const r = data.review;
        const verdictClass = r.passed ? "color:var(--accent)" : "color:#c44";
        reviewHtml = `
          <div class="plan"><h2>计划审查</h2>
            <div class="meta">
              verdict=<span style="${verdictClass};font-weight:700">${escapeHtml(r.verdict)}</span>
              score=${r.score.toFixed(2)} passed=${r.passed}
            </div>
            ${r.issues.length ? `<div class="meta"><strong>问题:</strong><br/>${r.issues.map(i => "  - " + escapeHtml(i)).join("<br/>")}</div>` : ""}
            ${r.suggestions.length ? `<div class="meta"><strong>建议:</strong><br/>${r.suggestions.map(s => "  - " + escapeHtml(s)).join("<br/>")}</div>` : ""}
          </div>`;
      }
      const llmPlan = data.plan_text
        ? `<div class="plan"><h2>LLM 计划</h2><pre>${escapeHtml(data.plan_text)}</pre></div>`
        : `<div class="plan"><h2>LLM 计划</h2><div class="status">未启用 LLM，未生成计划文本。</div></div>`;
      const promptHtml = data.prompt
        ? `<div class="plan"><h2>LLM Prompt</h2><pre>${escapeHtml(data.prompt)}</pre></div>`
        : "";
      const toolsHtml = data.tools && data.tools.length
        ? `<div class="plan"><h2>工具接口</h2><pre>${escapeHtml(data.tools.map(t => t.name + ": " + t.purpose).join("\n"))}</pre></div>`
        : "";
      const observationsHtml = data.observations && data.observations.length
        ? `<div class="plan"><h2>只读工具观察</h2><pre>${escapeHtml(formatObservations(data.observations))}</pre></div>`
        : "";
      outputEl.innerHTML = `
        ${reviewHtml}
        ${llmPlan}
        ${promptHtml}
        <div class="plan"><h2>Agent 计划</h2><pre>${escapeHtml(data.plan)}</pre></div>
        ${observationsHtml}
        ${toolsHtml}
        <div class="plan"><h2>检索统计</h2><div class="meta">seed=${stats.seed_count} recalled=${stats.recalled_count} final=${stats.final_count}</div></div>
        ${renderTrace(data.trace)}
      `;
    }

    function renderTrace(trace) {
      if (!trace || !Array.isArray(trace.events)) return "";
      const rows = trace.events.map((event, index) => {
        const summary = JSON.stringify(event.output_summary || {});
        const error = event.error ? `\nerror: ${event.error.type || ""}: ${event.error.message || ""}` : "";
        return `${index + 1}. ${event.step} [${event.status}] ${summary}${error}`;
      }).join("\n");
      return `<div class="plan"><h2>Trace Events</h2><pre>${escapeHtml(rows)}</pre></div>`;
    }

    function formatObservations(observations) {
      return observations.map(obs => {
        const head = `${obs.tool} ${JSON.stringify(obs.input || {})}`;
        if (obs.error) return `${head}\n  error: ${obs.error}`;
        if (Array.isArray(obs.output)) {
          const rows = obs.output.slice(0, 8).map(item => {
            if (typeof item === "string") return `  - ${item}`;
            const loc = item.path ? `${item.path}${item.start_line ? ":" + item.start_line : ""}` : "";
            const label = item.name || item.rank || item.path || "item";
            return `  - ${label} ${loc}`.trimEnd();
          });
          return [head, `  count: ${obs.output.length}`, ...rows].join("\n");
        }
        return `${head}\n  ${JSON.stringify(obs.output || {})}`;
      }).join("\n\n");
    }

    function value(id) {
      return document.getElementById(id).value.trim();
    }
    function escapeHtml(text) {
      return String(text || "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
