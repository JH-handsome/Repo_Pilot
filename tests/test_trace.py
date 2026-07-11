from pathlib import Path
import unittest

from coding_rag.tools.bm25 import SearchResult
from coding_rag.repository.chunks import CodeChunk
from coding_rag.rag.trace import build_retrieval_trace, build_tool_event, render_trace_report


class RetrievalTraceTest(unittest.TestCase):
    def test_build_trace_contains_all_stages(self):
        seed = SearchResult(
            CodeChunk(Path("a.py"), 1, 3, "def target():\n    return 1"),
            0.9,
            source="hybrid",
        )
        recall = SearchResult(
            CodeChunk(Path("a.py"), 4, 5, "def helper():\n    return 2"),
            0.4,
            source="recall:seed-1+1",
        )

        trace = build_retrieval_trace(
            query="where is target",
            seed_results=[seed],
            recalled_results=[seed, recall],
            final_results=[seed, recall],
            params={"top_k": 1},
        )

        self.assertEqual(trace["summary"]["seed_count"], 1)
        self.assertEqual(trace["summary"]["recalled_count"], 2)
        self.assertEqual(trace["summary"]["final_count"], 2)
        self.assertEqual(trace["summary"]["context_block_count"], 1)
        self.assertEqual(trace["trace_version"], "1.0")
        self.assertEqual(trace["run"]["mode"], "ask")
        self.assertIn("events", trace)
        self.assertIn("artifacts", trace)
        self.assertIn("retrieval", trace["artifacts"])
        self.assertEqual(trace["events"][0]["step"], "initial_search")
        self.assertEqual(trace["events"][0]["artifacts"]["results"][0]["file"], "a.py")
        self.assertIn("initial_search", trace["stages"])
        self.assertIn("context_compaction", trace["stages"])
        self.assertEqual(trace["stages"]["context_compaction"][0]["chunk_count"], 2)
        self.assertIn("char_count", trace["stages"]["context_compaction"][0])

    def test_render_trace_report_is_readable(self):
        seed = SearchResult(
            CodeChunk(Path("a.py"), 1, 3, "def target():\n    return 1"),
            0.9,
            source="hybrid",
        )
        trace = build_retrieval_trace(
            query="where is target",
            seed_results=[seed],
            recalled_results=[seed],
            final_results=[seed],
        )

        report = render_trace_report(trace)

        self.assertIn("RAG", report)
        self.assertIn("Initial Search", report)
        self.assertIn("Context Compaction", report)
        self.assertIn("a.py:1-3", report)

    def test_render_trace_report_accepts_legacy_shape(self):
        legacy = {
            "query": "where is target",
            "summary": {"seed_count": 1, "recalled_count": 1, "final_count": 1, "context_block_count": 0},
            "stages": {
                "initial_search": [
                    {
                        "rank": 1,
                        "file": "a.py",
                        "start_line": 1,
                        "end_line": 3,
                        "score": 0.9,
                        "source": "hybrid",
                        "preview": "def target()",
                    }
                ],
                "neighbor_recall": [],
                "final_filter": [],
                "context_compaction": [],
            },
        }

        report = render_trace_report(legacy)

        self.assertIn("Initial Search", report)
        self.assertIn("a.py:1-3", report)

    def test_render_trace_report_accepts_unified_events_without_stages(self):
        trace = {
            "trace_version": "1.0",
            "run": {"mode": "agent_exec", "status": "dry_run", "task": "fix bug"},
            "events": [
                {
                    "step": "run_command",
                    "status": "failed",
                    "input": {"cmd": "python -c \"print(1)\""},
                    "output_summary": {"returncode": None},
                    "artifacts": {"command": {"safety": {"allowed": False}}},
                    "error": {"type": "AgentToolError", "message": "command is denied by safety policy"},
                    "duration_ms": None,
                }
            ],
            "artifacts": {"tools": []},
        }

        report = render_trace_report(trace)

        self.assertIn("统一 Trace 事件", report)
        self.assertIn("run_command", report)
        self.assertIn("command is denied", report)

    def test_build_tool_event_for_executor_command(self):
        event = build_tool_event(
            tool="run_command",
            input={"cmd": "python -m unittest"},
            result={
                "command": ["python", "-m", "unittest"],
                "returncode": 0,
                "timeout_seconds": 60,
                "stdout_chars": 10,
                "stderr_chars": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "safety": {"allowed": True, "matched_rule": "python -m unittest"},
            },
        )

        self.assertEqual(event["step"], "run_command")
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["artifacts"]["command"]["timeout_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
