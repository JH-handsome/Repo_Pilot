"""Regression coverage for generated-file archives and legacy DB migration."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_rag.rag.trace import write_trace_json
from coding_rag.storage.artifacts import write_json_artifact
from coding_rag.storage.sqlite import SQLiteTraceStore
from scripts.import_json_artifacts import main as import_artifacts
from tests.test_trace_storage import CONTEXT, snapshot


class JSONArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "traces.sqlite3"
        self.store = SQLiteTraceStore(self.db)
        self.env = patch.dict(os.environ, {"REPOPILOT_TRACE_DB": str(self.db)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_json_versions_deduplicate_and_keep_filtered_content(self):
        path = self.root / "report.json"
        payload = {"query": "中文", "api_key": "must-not-save", "score": 1}
        write_json_artifact(path, payload)
        first = self.store.list_json_files()[0]
        stored = self.store.get_json_file(first["file_id"])
        self.assertEqual(json.loads(stored["content"]), {"query": "中文", "score": 1})
        self.assertEqual(stored["size_bytes"], len(stored["content"].encode("utf-8")))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
        self.assertNotIn("content", first)
        self.assertEqual(self.store.save_json_file(path)["status"], "unchanged")
        write_json_artifact(path, {"score": 2})
        self.assertEqual(len(self.store.list_json_files(source_path=path)), 2)
        self.assertEqual(json.loads(self.store.get_json_file(first["file_id"])["content"])["score"], 1)
        self.assertIsNone(self.store.get_json_file("missing"))

    def test_jsonl_and_empty_file_roundtrip(self):
        path = self.root / "trace.jsonl"
        rows = [{"id": 1, "token": "secret-value"}, {"id": 2, "text": "中文"}]
        write_json_artifact(path, rows, jsonl=True)
        stored = self.store.get_json_file(self.store.list_json_files()[0]["file_id"])
        self.assertEqual(stored["record_count"], 2)
        self.assertEqual([json.loads(line) for line in stored["content"].splitlines()],
                         [{"id": 1}, {"id": 2, "text": "中文"}])
        write_json_artifact(path, [], jsonl=True)
        empty = self.store.get_json_file(self.store.list_json_files()[0]["file_id"])
        self.assertEqual((empty["record_count"], empty["content"]), (0, ""))

    def test_invalid_files_do_not_partially_import(self):
        self.store.initialize()
        for name, content in (("bad.jsonl", '{}\ninvalid'), ("bad.json", '{"score": NaN}'),
                              ("other.txt", '{}')):
            path = self.root / name
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.store.save_json_file(path)
        self.assertEqual(self.store.list_json_files(), [])

    def test_concurrent_import_is_idempotent(self):
        path = self.root / "trace.json"
        path.write_text('{"id": 1}', encoding="utf-8")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.save_json_file(path), range(8)))
        self.assertEqual(sum(row["status"] == "saved" for row in results), 1)
        self.assertEqual(len(self.store.list_json_files()), 1)

    def test_v2_migration_preserves_trace_and_read_does_not_migrate(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("DROP TABLE json_files")
            connection.execute("PRAGMA user_version=2")
            connection.commit()
        self.assertEqual(self.store.list_json_files(), [])
        self.assertIsNone(self.store.get_json_file("missing"))
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), trace)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
        write_json_artifact(self.root / "report.json", {"score": 1})
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), trace)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_failed_or_disabled_archive_preserves_file_and_explicit_trace_db(self):
        path = self.root / "trace.json"
        with patch("coding_rag.storage.artifacts.SQLiteTraceStore.save_json_file",
                   side_effect=sqlite3.OperationalError):
            with self.assertLogs("coding_rag.storage.artifacts", level="WARNING"):
                write_json_artifact(path, {"score": 1})
        self.assertTrue(path.is_file())
        self.assertFalse(self.db.exists())
        write_trace_json(path, {"events": []}, trace_db="off")
        self.assertFalse(self.db.exists())
        other = self.root / "other.sqlite3"
        write_trace_json(path, {"events": []}, trace_db=other)
        self.assertEqual(len(SQLiteTraceStore(other).list_json_files()), 1)
        self.assertFalse(self.db.exists())

    def test_import_cli_recurses_deduplicates_and_reports_errors(self):
        folder = self.root / "files"
        folder.mkdir()
        nested = folder / "nested"
        nested.mkdir()
        (folder / "report.json").write_text('{"score": 1}', encoding="utf-8")
        (nested / "trace.jsonl").write_text('{}\n{}', encoding="utf-8")
        (folder / "ignore.txt").write_text('ignore', encoding="utf-8")
        args = [str(folder), str(folder / "report.json"), "--trace-db", str(self.db)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(import_artifacts(args), 0)
            self.assertEqual(import_artifacts(args), 0)
        summaries = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual((summaries[0]["saved"], summaries[1]["unchanged"]), (2, 2))
        (nested / "bad.json").write_text('invalid', encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(import_artifacts(args), 1)
        self.assertEqual(len(self.store.list_json_files()), 2)

    def test_missing_reads_and_query_validation(self):
        self.assertEqual(self.store.list_json_files(), [])
        self.assertIsNone(self.store.get_json_file("missing"))
        self.assertFalse(self.db.exists())
        for kwargs in ({"limit": 0}, {"limit": True}, {"offset": -1}):
            with self.assertRaises(ValueError):
                self.store.list_json_files(**kwargs)

    def test_evaluation_writers_archive_their_generated_outputs(self):
        from scripts.agent_eval import write_jsonl as write_agent_jsonl
        from scripts.agent_runtime_eval import _write as write_scorecard
        from scripts.retrieval_eval import write_jsonl as write_retrieval_jsonl

        write_agent_jsonl(self.root / "agent.jsonl", [{"query": "agent"}])
        write_retrieval_jsonl(self.root / "retrieval.jsonl", [{"query": "retrieval"}])
        write_scorecard(self.root / "scorecard.json", {"score": 1})
        files = self.store.list_json_files()
        self.assertEqual(len(files), 3)
        self.assertEqual({Path(row["source_path"]).name for row in files},
                         {"agent.jsonl", "retrieval.jsonl", "scorecard.json"})


if __name__ == "__main__":
    unittest.main()
