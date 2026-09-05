"""Regression coverage for real Graph calls, path round trips and durable chat."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import shutil
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from coding_rag.agent.conversation import ConversationStore, history_messages
from coding_rag.agent.runtime import (
    UnifiedRunConfig, build_hybrid_search_provider, compact_tool_result,
    run_unified_query, resume_unified_query, serialize_search_results,
)
from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.langchain_tools import build_agent_langchain_tool_map
from coding_rag.repository.chunks import CodeChunk
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools
from coding_rag.tools.bm25 import SearchResult
from tests.test_unified_runtime import SequenceClient, decision
import frontend
import main as cli
import web_ui


class RecallClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        previous = messages[1:-1]
        knows = any("cobalt-731" in item["content"] for item in previous)
        return decision(answer="cobalt-731" if knows else "unknown")


class ConversationRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "chat.sqlite3"
        self.config = UnifiedRunConfig(repo_path=self.root, session_id=str(uuid4()),
                                       conversation_db_path=self.db)

    def seed(self, config=None):
        return run_unified_query("Remember my project nickname: cobalt-731", config or self.config,
                                 SequenceClient([decision(answer="Nickname saved")]))

    def test_second_turn_actually_receives_first_turn_and_answer(self):
        first = self.seed()
        client = RecallClient()
        second = run_unified_query("What nickname did I give you?", self.config, client)
        self.assertEqual(second.answer, "cobalt-731")
        self.assertEqual(second.trace["conversation"]["loaded_turns"], 1)
        self.assertIn("Nickname saved", client.calls[0][2]["content"])
        self.assertEqual([m["role"] for m in client.calls[0]], ["system", "user", "assistant", "user"])
        self.assertEqual(first.trace["params"]["runtime"], "langgraph")
        self.assertEqual(first.messages[-1]["role"], "assistant")
        self.assertIn("Nickname saved", first.messages[-1]["content"])

    def test_different_session_and_repository_and_stateless_call_are_isolated(self):
        self.seed()
        other = self.root / "other"
        other.mkdir()
        for config in (replace(self.config, session_id=str(uuid4())),
                       replace(self.config, repo_path=other),
                       replace(self.config, session_id=None)):
            client = RecallClient()
            run = run_unified_query("What nickname?", config, client)
            self.assertEqual(run.answer, "unknown")
            self.assertEqual(len(client.calls[0]), 2)

    def test_history_recovers_in_a_fresh_python_process(self):
        self.seed()
        code = """
import sys
from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query
from tests.test_conversation_runtime import RecallClient
config = UnifiedRunConfig(repo_path=sys.argv[1], session_id=sys.argv[2], conversation_db_path=sys.argv[3])
print(run_unified_query('What nickname?', config, RecallClient()).answer)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root), self.config.session_id,
                                 str(self.db)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "cobalt-731")

    def test_frontend_new_clients_and_server_approval_stores_keep_chat(self):
        payload = {"repo_path": str(self.root), "session_id": self.config.session_id,
                   "query": "Remember cobalt-731"}
        with patch.dict(os.environ, {"REPOPILOT_CHAT_DB": str(self.db)}), \
             patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()):
            with patch("frontend.OpenAICompatibleChatClient", return_value=RecallClient()):
                first = frontend.run_frontend_query(payload, approval_store=frontend.PendingApprovalStore())
            # A refreshed request/new client and a recreated server-side approval store.
            client = RecallClient()
            with patch("frontend.OpenAICompatibleChatClient", return_value=client):
                second = frontend.run_frontend_query({**payload, "query": "What nickname?"},
                                                    approval_store=frontend.PendingApprovalStore())
                other = frontend.run_frontend_query({**payload, "session_id": str(uuid4()),
                                                     "query": "What nickname?"})
        self.assertEqual(second["answer"], "cobalt-731")
        self.assertEqual(other["answer"], "unknown")
        self.assertEqual(first["conversation"]["session_id"], second["conversation"]["session_id"])

    def test_interactive_cli_reuses_one_session(self):
        with patch.object(sys, "argv", ["main.py", str(self.root), "--chat-db", str(self.db)]):
            args = cli.parse_args()
        with patch("main.build_chat_client", return_value=RecallClient()), \
             patch("main.run_single_turn", return_value=0) as run, \
             patch("builtins.input", side_effect=["Remember cobalt-731", "What nickname?", ":q"]), \
             patch("builtins.print"):
            cli.run_interactive_cli(args)
        ids = [call.args[0].session_id for call in run.call_args_list]
        self.assertEqual(len(ids), 2)
        self.assertTrue(ids[0])
        self.assertEqual(ids[0], ids[1])

    def test_desktop_llm_uses_graph_without_requiring_python_files(self):
        ui = web_ui.RepoPilotUI.__new__(web_ui.RepoPilotUI)
        values = {"repo_path": str(self.root), "query": "Remember cobalt-731", "use_llm": True,
                  "top_k": 5, "chunk_size": 40, "overlap": 5, "recall_window": 2,
                  "mode": "judge", "provider": "deepseek", "session_id": self.config.session_id}
        for name, value in values.items():
            setattr(ui, name + "_var", SimpleNamespace(get=lambda v=value: v))
        outputs = []
        ui._set_output = outputs.append
        ui._set_status = lambda text: None
        with patch.dict(os.environ, {"REPOPILOT_CHAT_DB": str(self.db)}), \
             patch("web_ui.load_dotenv"), patch("web_ui.build_llm_config", return_value=object()), \
             patch("web_ui.OpenAICompatibleChatClient", return_value=RecallClient()):
            ui._run_pipeline()
            ui.query_var = SimpleNamespace(get=lambda: "What nickname?")
            ui._run_pipeline()
        self.assertIn("cobalt-731", outputs[-1])
        self.assertEqual(ConversationStore(self.db, self.root, self.config.session_id).load()[1], 2)

    def test_whole_turn_budget_keeps_initial_and_latest_and_reports_omissions(self):
        turns = [("anchor", "a"), ("middle" * 10, "m"), ("latest", "z")]
        messages, omitted = history_messages(turns, 14)
        self.assertEqual([m["content"] for m in messages if m["role"] == "user"], ["anchor", "latest"])
        self.assertEqual(omitted, 1)
        self.seed()
        client = RecallClient()
        run = run_unified_query("What nickname?", replace(self.config, max_history_chars=1), client)
        self.assertEqual(run.trace["conversation"]["omitted_turns"], 1)
        self.assertIn("已省略 1", client.calls[0][0]["content"])
        self.assertEqual(ConversationStore(self.db, self.root, self.config.session_id).load()[1], 2)

    def test_failed_model_turn_is_not_saved(self):
        run = run_unified_query("test", self.config, SequenceClient(["bad", "bad"]))
        self.assertEqual(run.status, "failed")
        self.assertEqual(ConversationStore(self.db, self.root, self.config.session_id).load()[1], 0)

    def test_persistence_failure_is_reported_without_repeating_model(self):
        client = RecallClient()
        with patch.object(ConversationStore, "append", side_effect=sqlite3.OperationalError("disk full")):
            run = run_unified_query("hello", self.config, client)
        self.assertEqual(run.trace["conversation"]["status"], "failed")
        self.assertEqual(len(client.calls), 1)

    def test_concurrent_stale_revision_cannot_overwrite_history(self):
        store = ConversationStore(self.db, self.root, self.config.session_id)
        store.append("first", "answer", "one", 0)
        store.append("first", "answer", "one", 0)
        with self.assertRaises(RuntimeError):
            store.append("stale", "answer", "two", 0)
        self.assertEqual(store.load()[1], 1)

    def test_pending_approval_is_not_chat_and_cannot_resume_after_new_turn(self):
        (self.root / "app.py").write_text("value = 1\n", encoding="utf-8")
        diff = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        client = SequenceClient([decision(action="tool", tool="apply_patch", answer=None,
                                          arguments={"diff": diff})])
        pending = run_unified_query("Change value", self.config, client)
        self.assertEqual(pending.status, "approval_required")
        self.assertEqual(ConversationStore(self.db, self.root, self.config.session_id).load()[1], 0)
        self.seed()
        with self.assertRaisesRegex(ValueError, "Conversation changed"):
            resume_unified_query(pending, self.config, client,
                                 approval_fingerprint=pending.approval["fingerprint"])
        self.assertEqual((self.root / "app.py").read_text(), "value = 1\n")

    def test_same_session_parallel_requests_accumulate_in_order(self):
        def ask(query):
            return run_unified_query(query, self.config, RecallClient())
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(ask, ["Remember cobalt-731", "Hello"]))
        self.assertEqual(sorted(run.trace["conversation"]["loaded_turns"] for run in runs), [0, 1])
        self.assertEqual(ConversationStore(self.db, self.root, self.config.session_id).load()[1], 2)

    def test_approved_write_saves_one_chat_turn_without_replaying_tools(self):
        (self.root / "app.py").write_text("value = 1\n", encoding="utf-8")
        diff = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        client = SequenceClient([
            decision(action="tool", tool="apply_patch", answer=None, arguments={"diff": diff}),
            decision(answer="Changed value"),
        ])
        config = replace(self.config, snapshot_root=self.root / "snapshots")
        pending = run_unified_query("Change value", config, client)
        completed = resume_unified_query(pending, config, client,
                                          approval_fingerprint=pending.approval["fingerprint"])
        self.assertEqual(completed.status, "success")
        self.assertEqual(completed.trace["conversation"]["status"], "saved")
        self.assertEqual(ConversationStore(self.db, self.root, config.session_id).load()[1], 1)
        next_client = RecallClient()
        run_unified_query("What did you do?", config, next_client)
        self.assertIn("Changed value", next_client.calls[0][2]["content"])
        self.assertNotIn("diff --git", json.dumps(next_client.calls[0]))
        self.assertEqual((self.root / "app.py").read_text(), "value = 2\n")

    def test_http_server_restart_retains_only_matching_conversation(self):
        def request(payload):
            server = frontend.RepoPilotServer(("127.0.0.1", 0), frontend.RepoPilotHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/run", data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    return json.load(response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        payload = {"repo_path": str(self.root), "session_id": self.config.session_id,
                   "query": "Remember cobalt-731"}
        with patch.dict(os.environ, {"REPOPILOT_CHAT_DB": str(self.db)}), \
             patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()), \
             patch("frontend.OpenAICompatibleChatClient", side_effect=lambda config: RecallClient()):
            first = request(payload)
            second = request({**payload, "query": "What nickname?"})
            third = request({**payload, "session_id": str(uuid4()), "query": "What nickname?"})
        self.assertTrue(first["ok"])
        self.assertEqual(second["answer"], "cobalt-731")
        self.assertEqual(third["answer"], "unknown")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the browser script check")
    def test_browser_script_refresh_new_chat_and_separate_tab(self):
        script = frontend.INDEX_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
        # Parse the complete shipped script and execute its actual chat storage handlers.
        chat = script[script.index("    const chatStorageKey"):script.index("    function runTask()")]
        harness = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {randomUUID, webcrypto} = require('node:crypto');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
new vm.Script(input.script);
function page(storage = new Map()) {
  const elements = {};
  const context = {
    crypto: {randomUUID, getRandomValues: bytes => webcrypto.getRandomValues(bytes)}, requestInFlight: false, statusEl: {}, outputEl: {},
    sessionStorage: {getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v)},
    document: {getElementById: id => elements[id] ||= {value: '.', addEventListener: (event, fn) => {elements[id][event] = fn;}}},
    value: id => elements[id].value,
  };
  vm.createContext(context);
  vm.runInContext(input.chat, context);
  return {context, elements, id: reset => context.chatSessionId(reset)};
}
const storage = new Map();
let first = page(storage);
first.elements.repo.value = 'D:/repo with spaces';
const id = first.id();
assert.equal(first.id(), id);
let refreshed = page(storage);
assert.equal(refreshed.elements.repo.value, 'D:/repo with spaces');
assert.equal(refreshed.id(), id);
refreshed.elements.newConversation.click();
assert.notEqual(refreshed.id(), id);
delete refreshed.context.crypto.randomUUID;
assert.match(refreshed.id(true), /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
const tab = page();
tab.elements.repo.value = 'D:/repo with spaces';
assert.notEqual(tab.id(), id);
assert.deepEqual([...storage.keys()].sort(), ['repopilot.chat.repo', 'repopilot.chat.sessions.v1']);
assert(!JSON.stringify([...storage.values()]).includes('learningSession'));
console.log('browser session checks passed');
"""
        result = subprocess.run([shutil.which("node"), "-e", harness],
                                input=json.dumps({"script": script, "chat": chat}),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_search_json_tool_read_roundtrip_for_nested_unicode_windows_paths(self):
        folder = self.root / "目录 with space"
        folder.mkdir()
        target = folder / "target.py"
        target.write_text("def unique_lookup():\n    return 'cobalt-731'\n", encoding="utf-8")
        executor = AgentExecutor(self.root, search_provider=build_hybrid_search_provider(self.config))
        tools = build_agent_langchain_tool_map(executor)
        rows = json.loads(json.dumps(tools["search_code"].invoke({"query": "unique_lookup"})))["results"]
        self.assertTrue(rows)
        for row in rows:
            self.assertNotIn(chr(92), row["path"])
            result = tools["read_file"].invoke({"path": row["path"]})
            self.assertIn("cobalt-731", result["text"])
        absolute = json.loads(json.dumps({"path": str(target)}))["path"]
        self.assertIn("cobalt-731", tools["read_file"].invoke({"path": absolute})["text"])
        # Feed the actual search result through a model-shaped JSON decision to Graph.
        client = SequenceClient([
            decision(action="tool", tool="search_code", answer=None, arguments={"query": "unique_lookup"}),
            decision(action="tool", tool="read_file", answer=None, arguments={"path": rows[0]["path"]}),
            decision(answer="Found " + rows[0]["path"] + ":1-2"),
        ])
        run = run_unified_query("Find unique_lookup", replace(self.config, session_id=None), client)
        self.assertEqual(run.status, "success")
        self.assertEqual([item["tool"] for item in run.observations], ["search_code", "read_file"])

    def test_outside_absolute_path_is_displayed_consistently_but_read_is_denied(self):
        outside = self.root.parent / "outside.py"
        chunk = CodeChunk(file_path=outside, start_line=1, end_line=1, text="value = 1")
        row = serialize_search_results([SearchResult(chunk, 1)], self.root)[0]
        self.assertEqual(row["path"], outside.resolve().as_posix())
        tools = ReadOnlyAgentTools(self.root)
        self.assertEqual(tools.display_path(outside), row["path"])
        with self.assertRaisesRegex(AgentToolError, "outside repository"):
            tools.read_file(row["path"])

    def test_compacted_search_keeps_complete_readable_paths(self):
        target = self.root / "target.py"
        target.write_text("value = 1\n", encoding="utf-8")
        compact, meta = compact_tool_result("search_code", {
            "results": [{"path": "target.py", "start_line": 1, "end_line": 1, "text": "x" * 5000}],
        }, max_chars=300)
        self.assertTrue(meta["output_truncated"])
        self.assertEqual(compact["results"][0]["path"], "target.py")
        self.assertIn("value = 1", ReadOnlyAgentTools(self.root).read_file(compact["results"][0]["path"])["text"])


if __name__ == "__main__":
    unittest.main()
