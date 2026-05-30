from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent_memory import AgentMemoryStore, build_memory


class AgentMemoryStoreTest(unittest.TestCase):
    def test_append_and_search_memory(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "agent_memory.jsonl"
            store = AgentMemoryStore(memory_path)
            store.append(
                build_memory(
                    task="实现 Hybrid Search",
                    status="done",
                    summary="增加 Hybrid Search 检索模式",
                    files=["coding_rag/bm25_retriever.py"],
                    decisions=["融合正文和元数据分数"],
                )
            )

            results = store.search("Hybrid Search 检索怎么优化", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].files, ["coding_rag/bm25_retriever.py"])

    def test_missing_memory_file_returns_empty_list(self):
        with TemporaryDirectory() as temp_dir:
            store = AgentMemoryStore(Path(temp_dir) / "missing.jsonl")

            self.assertEqual(store.load_all(), [])
            self.assertEqual(store.search("anything"), [])

    def test_append_if_new_skips_duplicate_memory(self):
        with TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "agent_memory.jsonl"
            store = AgentMemoryStore(memory_path)
            memory = build_memory(
                task="同一个任务",
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
