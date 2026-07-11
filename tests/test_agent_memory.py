from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent.memory import AgentMemoryStore, build_memory


class AgentMemoryStoreTest(unittest.TestCase):
    def test_append_and_search_memory(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "agent_memory.jsonl"
            store = AgentMemoryStore(memory_path)
            store.append(
                build_memory(
                    task="implement Hybrid Search",
                    status="done",
                    summary="Add Hybrid Search retrieval mode",
                    files=["coding_rag/tools/bm25.py"],
                    decisions=["combine text and metadata scores"],
                )
            )

            results = store.search("How to optimize Hybrid Search retrieval", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].files, ["coding_rag/tools/bm25.py"])

    def test_missing_memory_file_returns_empty_list(self):
        with TemporaryDirectory() as temp_dir:
            store = AgentMemoryStore(Path(temp_dir) / "missing.jsonl")

            self.assertEqual(store.load_all(), [])
            self.assertEqual(store.search("anything"), [])

    def test_load_memory_accepts_utf8_bom(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "agent_memory.jsonl"
            memory_path.write_text(
                '\ufeff{"id":"agent-1","created_at":"2026-06-03T00:00:00+00:00",'
                '"task":"test","status":"planned","summary":"summary","files":[],"decisions":[]}\n',
                encoding="utf-8",
            )
            store = AgentMemoryStore(memory_path)

            memories = store.load_all()

            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].task, "test")

    def test_append_if_new_skips_duplicate_memory(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "agent_memory.jsonl"
            store = AgentMemoryStore(memory_path)
            memory = build_memory(
                task="same task",
                status="planned",
                summary="summary",
                files=["a.py"],
                decisions=["decision"],
            )

            self.assertTrue(store.append_if_new(memory))
            self.assertFalse(store.append_if_new(memory))
            self.assertEqual(len(store.load_all()), 1)


if __name__ == "__main__":
    unittest.main()
