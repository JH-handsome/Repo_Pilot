"""Regression tests for trace durability, recovery and authenticated history."""

import copy
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query, resume_unified_query
from coding_rag.rag.trace import build_trace_event
from coding_rag.storage.persistence import configured_trace_db, persist_trace, replay_pending, sanitize_trace
from coding_rag.storage.sqlite import SQLiteTraceStore
from frontend import RepoPilotHandler, RepoPilotServer
from tests.test_unified_runtime import SequenceClient, decision


def snapshot(run_id=None, revision=1, events=None):
    return {
        "trace_version": "1.0",
        "run": {
            "run_id": run_id or str(uuid4()), "mode": "unified", "status": "success",
            "query": "测试", "revision": revision,
            "started_at": "2026-09-04T01:00:00.000000+00:00",
            "updated_at": "2026-09-04T01:00:01.000000+00:00",
            "finished_at": "2026-09-04T01:00:01.000000+00:00",
        },
        "events": events if events is not None else [build_trace_event(step="model_decision")],
        "artifacts": {},
    }


CONTEXT = {"source": "test", "repo_key": "test-repository"}


class SQLiteTraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "traces.sqlite3"
        self.store = SQLiteTraceStore(self.path)

    def counts(self):
        with closing(sqlite3.connect(self.path)) as db:
            return tuple(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                         for table in ("trace_runs", "trace_events"))

    def test_missing_database_reads_do_not_create_files(self):
        self.assertIsNone(self.store.get_trace("missing"))
        self.assertEqual(self.store.list_traces()["items"], [])
        self.assertFalse(self.path.exists())

    def test_revision_and_event_idempotency(self):
        first = snapshot()
        self.assertTrue(self.store.save_trace(first, CONTEXT))
        self.assertFalse(self.store.save_trace(first, CONTEXT))
        second = copy.deepcopy(first)
        second["run"]["revision"] = 2
        second["events"].append(build_trace_event(step="final_answer"))
        self.assertTrue(self.store.save_trace(second, CONTEXT))
        self.assertFalse(self.store.save_trace(first, CONTEXT))
        self.assertEqual(self.counts(), (1, 2))
        self.assertEqual(self.store.get_trace(first["run"]["run_id"]), second)

    def test_transaction_rolls_back_run_and_events_together(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        updated = copy.deepcopy(trace)
        updated["run"]["revision"] = 2
        updated["events"].append({"step": None})  # SQL NOT NULL fails after the run update.
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_trace(updated, CONTEXT)
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), trace)
        self.assertEqual(self.counts(), (1, 1))

    def test_existing_event_history_and_identity_cannot_change(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        changed = copy.deepcopy(trace)
        changed["run"]["revision"] = 2
        changed["events"][0]["step"] = "changed"
        with self.assertRaises(ValueError):
            self.store.save_trace(changed, CONTEXT)
        with self.assertRaises(ValueError):
            self.store.save_trace(trace, {**CONTEXT, "repo_key": "other"})
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), trace)

    def test_concurrent_initialization_and_writers(self):
        traces = [snapshot() for _ in range(12)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(lambda trace: self.store.save_trace(trace, CONTEXT), traces))
        self.assertTrue(all(outcomes))
        self.assertEqual(self.counts(), (12, 12))

    def test_schema_version_is_separate_and_future_schema_is_rejected(self):
        self.store.save_trace(snapshot(), CONTEXT)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            db.execute("PRAGMA user_version=2")
        with self.assertRaises(ValueError):
            self.store.save_trace(snapshot(), CONTEXT)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_conflicting_revision_and_unrelated_database_are_rejected(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        conflict = copy.deepcopy(trace)
        conflict["run"]["status"] = "failed"
        with self.assertRaises(ValueError):
            self.store.save_trace(conflict, CONTEXT)
        unrelated = self.path.parent / "unrelated.sqlite3"
        with closing(sqlite3.connect(unrelated)) as db:
            db.execute("CREATE TABLE user_data (value TEXT)")
        with self.assertRaises(ValueError):
            SQLiteTraceStore(unrelated).save_trace(trace, CONTEXT)
        with closing(sqlite3.connect(unrelated)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("user_data",)])

    def test_filters_and_cursor_do_not_duplicate_tied_timestamps(self):
        for _ in range(5):
            self.store.save_trace(snapshot(), CONTEXT)
        first = self.store.list_traces({"status": "success"}, limit=2)
        second = self.store.list_traces(cursor=first["next_cursor"], limit=2)
        third = self.store.list_traces(cursor=second["next_cursor"], limit=2)
        ids = [row["run_id"] for page in (first, second, third) for row in page["items"]]
        self.assertEqual(len(set(ids)), 5)
        self.assertIsNone(third["next_cursor"])
        self.assertNotIn("trace_json", first["items"][0])
        self.assertEqual(self.store.list_traces({"status": "' OR 1=1 --"})["items"], [])
        for kwargs in ({"limit": 101}, {"cursor": "bad"}, {"filters": {"db_path": "other"}}):
            with self.assertRaises(ValueError):
                self.store.list_traces(**kwargs)

    def test_sanitized_snapshot_and_pending_replay_do_not_mutate_live_data(self):
        trace = snapshot()
        trace["artifacts"] = {
            "approval": {"fingerprint": "must-not-persist"},
            "messages": ["private conversation"],
            "raw_output": "raw model response",
            "result": "password=secret-value Bearer secret-token", "api_key": "secret-value",
        }
        original = copy.deepcopy(trace)
        with patch("coding_rag.storage.persistence.SQLiteTraceStore.save_trace", side_effect=sqlite3.OperationalError):
            outcome = persist_trace(trace, self.path, source="test", repo_path=self.temp.name)
        self.assertEqual(outcome["status"], "failed")
        self.assertTrue(outcome["pending"])
        pending = next((self.path.parent / "trace_pending").glob("*.json"))
        content = pending.read_text(encoding="utf-8")
        for secret in ("secret-value", "secret-token", "must-not-persist", "private conversation", "raw model response"):
            self.assertNotIn(secret, content)
        self.assertEqual(trace, original)
        self.assertEqual(replay_pending(pending, self.path)["status"], "saved")
        self.assertEqual(replay_pending(pending, self.path)["status"], "unchanged")
        self.assertTrue(pending.exists())
        self.assertEqual(self.counts(), (1, 1))

    def test_disk_failure_reports_when_spool_also_fails(self):
        with patch("coding_rag.storage.persistence.SQLiteTraceStore.save_trace", side_effect=OSError), patch("coding_rag.storage.persistence.Path.mkdir", side_effect=OSError):
            outcome = persist_trace(snapshot(), self.path, source="test", repo_path=self.temp.name)
        self.assertEqual(outcome["status"], "failed")
        self.assertFalse(outcome["pending"])

    def test_config_and_known_secret_masking(self):
        with patch.dict("os.environ", {"REPOPILOT_TRACE_DB": "off", "EXAMPLE_API_KEY": "unique-secret-value"}):
            self.assertIsNone(configured_trace_db())
            self.assertEqual(configured_trace_db(self.path), self.path)
            self.assertEqual(sanitize_trace({"text": "unique-secret-value"}), {"text": "[REDACTED]"})


class TraceRuntimeTest(unittest.TestCase):
    def test_approval_resume_keeps_identity_and_only_appends_events(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "app.py").write_text("value = 1\n", encoding="utf-8")
            db_path = root / "trace.sqlite3"
            config = UnifiedRunConfig(repo_path=root, trace_db_path=db_path, snapshot_root=root / "snapshots")
            client = SequenceClient([
                decision(action="tool", answer=None, tool="apply_patch", arguments={"diff":
                    "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"}),
                decision(answer="完成"),
            ])
            pending = run_unified_query("更新值", config, client)
            self.assertEqual(pending.storage["status"], "saved")
            self.assertIsNone(pending.trace["run"]["finished_at"])
            resumed = resume_unified_query(pending, config, client, approval_fingerprint=pending.approval["fingerprint"])
            self.assertEqual(resumed.storage["status"], "saved")
            self.assertEqual(resumed.trace["run"]["run_id"], pending.trace["run"]["run_id"])
            self.assertEqual(resumed.trace["run"]["started_at"], pending.trace["run"]["started_at"])
            self.assertEqual(resumed.trace["run"]["revision"], 2)
            self.assertIsNotNone(resumed.trace["run"]["finished_at"])
            self.assertEqual(len(SQLiteTraceStore(db_path).list_traces()["items"]), 1)
            with closing(sqlite3.connect(db_path)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM trace_events").fetchone()[0], len(resumed.trace["events"]))
            stored = SQLiteTraceStore(db_path).get_trace(resumed.trace["run"]["run_id"])
            self.assertNotIn("approval", stored["artifacts"]["agent"])
            self.assertEqual((root / "app.py").read_text(), "value = 2\n")

    def test_failed_run_is_saved_and_database_failure_never_retries_model(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.sqlite3"
            client = SequenceClient(["invalid JSON", "still invalid"])
            run = run_unified_query("hello", UnifiedRunConfig(trace_db_path=path), client)
            self.assertEqual(run.status, "failed")
            self.assertEqual(run.storage["status"], "saved")
            self.assertEqual(SQLiteTraceStore(path).get_trace(run.trace["run"]["run_id"])["run"]["status"], "failed")
            client = SequenceClient([decision(answer="hello")])
            with patch("coding_rag.storage.persistence.SQLiteTraceStore.save_trace", side_effect=sqlite3.OperationalError):
                run = run_unified_query("hello", UnifiedRunConfig(trace_db_path=path), client)
            self.assertEqual(run.status, "success")
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(run.storage["status"], "failed")


class TraceHistoryHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "trace.sqlite3"
        self.trace = snapshot()
        SQLiteTraceStore(self.db).save_trace(self.trace, CONTEXT)
        self.token = "test-history-access-" + "x" * 32
        self.server = RepoPilotServer(("127.0.0.1", 0), RepoPilotHandler,
                                      trace_db_path=self.db, trace_read_token=self.token)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.server.server_close()

    def get(self, path, token=None):
        headers = {"Authorization": "Bearer " + token} if token else {}
        try:
            with urlopen(Request(self.url + path, headers=headers), timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_history_requires_separate_authentication_and_is_off_without_token(self):
        for path in ("/api/traces", "/api/traces/" + self.trace["run"]["run_id"]):
            self.assertEqual(self.get(path)[0], 401)
            self.assertEqual(self.get(path, "incorrect")[0], 401)
        self.server.trace_read_token = ""
        self.assertEqual(self.get("/api/traces", self.token)[0], 403)

    def test_normal_http_run_uses_server_database_and_preserves_response(self):
        client = SequenceClient([decision(answer="hello")])
        body = json.dumps({"repo_path": self.temp.name, "query": "hello",
                           "trace_db_path": "client-cannot-select.sqlite3"}).encode()
        with patch("frontend.load_dotenv"), patch("frontend.build_llm_config", return_value=object()), patch("frontend.OpenAICompatibleChatClient", return_value=client):
            request = Request(self.url + "/api/run", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                payload = json.load(response)
        self.assertEqual(payload["answer"], "hello")
        self.assertEqual(payload["storage"]["status"], "saved")
        saved = SQLiteTraceStore(self.db).get_trace(payload["trace"]["run"]["run_id"])
        self.assertEqual(saved["run"]["status"], "success")
        self.assertEqual(len(SQLiteTraceStore(self.db).list_traces({"source": "browser"})["items"]), 1)

    def test_authorized_list_detail_and_invalid_filters(self):
        code, payload = self.get("/api/traces?status=success&limit=1", self.token)
        self.assertEqual(code, 200)
        self.assertEqual(len(payload["items"]), 1)
        code, payload = self.get("/api/traces/" + self.trace["run"]["run_id"], self.token)
        self.assertEqual(code, 200)
        self.assertEqual(payload["trace"], self.trace)
        self.assertEqual(self.get("/api/traces/" + str(uuid4()), self.token)[0], 404)
        for query in ("limit=101", "db_path=other", "limit=1&limit=2", "since=not-a-date", "cursor=bad"):
            self.assertEqual(self.get("/api/traces?" + query, self.token)[0], 400)
        self.server.trace_db_path = None
        self.assertEqual(self.get("/api/traces", self.token)[0], 503)


if __name__ == "__main__":
    unittest.main()
