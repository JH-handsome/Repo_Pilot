"""Legacy trace conversion must be repeatable and preserve live run history."""

from contextlib import closing, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from coding_rag.storage.legacy import import_archived_traces
from coding_rag.storage.sqlite import SQLiteTraceStore
from scripts.import_json_artifacts import main
from tests.test_trace_storage import CONTEXT, snapshot


def retrieval(query="中文查询"):
    return {"query": query, "params": {"repo_path": "."}, "summary": {"final_count": 1},
            "stages": {"initial_search": [], "neighbor_recall": [],
                       "final_filter": [{"file": "app.py"}], "context_compaction": []}}


class LegacyTraceImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = SQLiteTraceStore(self.root / "traces.sqlite3")
        self.store.initialize()

    def archive(self, name, records):
        path = self.root / name
        content = ("\n".join(json.dumps(row, ensure_ascii=False) for row in records)
                   if path.suffix == ".jsonl" else json.dumps(records, ensure_ascii=False))
        path.write_text(content, encoding="utf-8")
        return path, self.store.save_json_file(path)

    def test_legacy_content_deduplicates_across_copies_and_reimports(self):
        record = retrieval()
        self.archive("original.json", record)
        self.archive("copy.json", dict(reversed(list(record.items()))))
        self.archive("rows.jsonl", [record, record, retrieval("不同查询")])
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["unchanged"], result["failed"]), (2, 3, 0))
        again = import_archived_traces(self.store)
        self.assertEqual((again["saved"], again["unchanged"]), (0, 5))
        runs = self.store.list_traces({"source": "legacy_import"})["items"]
        self.assertEqual(len(runs), 2)
        trace = self.store.get_trace(runs[0]["run_id"])
        self.assertIsNone(trace["run"]["started_at"])
        self.assertEqual(trace["run"]["status"], "unknown")
        self.assertEqual(len(trace["events"]), 4)
        self.assertTrue(all(event["occurred_at"] is None for event in trace["events"]))
        with closing(sqlite3.connect(self.store.path)) as connection:
            self.assertEqual(connection.execute("select count(*) from trace_events").fetchone()[0], 8)

    def test_native_existing_identity_and_old_revision_are_preserved(self):
        older = snapshot()
        latest = copy.deepcopy(older)
        latest["run"]["revision"] = 2
        latest["events"].append({"step": "final_answer"})
        self.store.save_trace(latest, CONTEXT)
        self.archive("older.json", older)
        self.archive("current.json", latest)
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["unchanged"], result["failed"]), (0, 2, 0))
        self.assertEqual(self.store.get_trace(older["run"]["run_id"]), latest)
        self.assertEqual(self.store.get_trace_context(older["run"]["run_id"])["source"], "test")

    def test_newer_export_extends_existing_events(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        updated = copy.deepcopy(trace)
        updated["run"]["revision"] = 2
        updated["events"].append({"step": "final_answer"})
        self.archive("updated.json", updated)
        self.assertEqual(import_archived_traces(self.store)["saved"], 1)
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), updated)

    def test_same_revision_conflict_is_reported_without_overwriting(self):
        trace = snapshot()
        self.store.save_trace(trace, CONTEXT)
        conflict = copy.deepcopy(trace)
        conflict["run"]["status"] = "failed"
        self.archive("conflict.json", conflict)
        self.archive("good.json", retrieval())
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["failed"]), (1, 1))
        self.assertIn("conflicting", result["errors"][0]["error"])
        self.assertEqual(self.store.get_trace(trace["run"]["run_id"]), trace)

    def test_pending_wrapper_uses_its_original_identity(self):
        trace = snapshot()
        self.archive("pending.json", {"trace": trace, "context": CONTEXT})
        self.archive("export.json", trace)
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["unchanged"]), (1, 1))
        context = self.store.get_trace_context(trace["run"]["run_id"])
        self.assertEqual(context["source"], CONTEXT["source"])
        self.assertEqual(context["repo_key"], CONTEXT["repo_key"])

    def test_export_before_pending_copy_still_deduplicates(self):
        trace = snapshot()
        self.archive("export.json", trace)
        self.archive("pending.json", {"trace": trace, "context": CONTEXT})
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["unchanged"], result["failed"]), (1, 1, 0))
        self.assertEqual(import_archived_traces(self.store)["unchanged"], 2)

    def test_no_id_unified_trace_and_agent_wrapper_and_eval_rows(self):
        native = {"run": {"mode": "unified", "status": "success"},
                  "events": [{"step": "final_answer"}], "artifacts": {}}
        agent = {"task": "解释代码", "trace": retrieval(),
                 "agent_trace": [{"step": "receive_task", "status": "done", "artifacts": {}}]}
        rows = [{"id": "eval1", "query": "搜索", "seed": [], "recalled": [], "final": []},
                {"id": "eval2", "trace_complete": True, "result_files": ["app.py"], "repo": "org/repo"}]
        self.archive("unified.json", native)
        self.archive("agent.json", agent)
        self.archive("eval.jsonl", rows)
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["failed"]), (4, 0))
        modes = {row["mode"] for row in self.store.list_traces()["items"]}
        self.assertEqual(modes, {"unified", "agent_exec", "retrieval_eval", "agent_eval"})

    def test_summaries_sidecars_and_memory_do_not_become_runs(self):
        self.archive("other.jsonl", [{"total": 10, "metrics": {}},
                     {"id": "c1", "query": "q", "initial_review": [], "final": []},
                     {"task": "memory", "created_at": "2026-09-04", "decisions": []}])
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["skipped"]), (0, 3))

    def test_invalid_record_does_not_stop_other_records_or_leave_partial_run(self):
        broken = snapshot()
        broken["events"].append({"step": "bad", "status": None})
        self.archive("mixed.json", [broken, retrieval()])
        result = import_archived_traces(self.store)
        self.assertEqual((result["saved"], result["failed"]), (1, 1))
        self.assertIsNone(self.store.get_trace(broken["run"]["run_id"]))

    def test_credentials_are_not_copied_back_into_trace_runs(self):
        record = retrieval()
        record["api_key"] = "secret-value"
        record["summary"]["text"] = "password=secret-password"
        self.archive("secret.json", record)
        self.assertEqual(import_archived_traces(self.store)["saved"], 1)
        run_id = self.store.list_traces()["items"][0]["run_id"]
        stored = json.dumps(self.store.get_trace(run_id))
        self.assertNotIn("secret-value", stored)
        self.assertNotIn("secret-password", stored)

    def test_archives_only_cli_does_not_require_the_old_source_file(self):
        path, _ = self.archive("old.json", retrieval())
        path.rename(self.root / "source-moved-away.json")
        args = ["--convert-traces", "--archives-only", "--trace-db", str(self.store.path)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 0)
        summaries = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(summaries[0]["traces"]["saved"], 1)
        self.assertEqual(summaries[1]["traces"]["saved"], 0)
        self.assertEqual(summaries[1]["traces"]["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
