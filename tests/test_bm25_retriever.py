from pathlib import Path
import unittest

from coding_rag.bm25_retriever import BM25Retriever
from coding_rag.code_splitter import CodeChunk


class BM25RetrieverHybridSearchTest(unittest.TestCase):
    def test_search_uses_hybrid_metadata_for_semantic_file_location(self):
        query = "\u5982\u679c Recall@3 \u4e00\u76f4\u662f 0\uff0c\u4f46\u8089\u773c\u770b\u68c0\u7d22\u7ed3\u679c\u91cc\u6709\u6b63\u786e\u6587\u4ef6"
        chunks = [
            CodeChunk(Path("rag/prompt.py"), 1, 10, "Recall context and answer prompt"),
            CodeChunk(Path("scripts/retrieval_eval.py"), 1, 10, "matched_relevant_index recall_at_k"),
        ]

        results = BM25Retriever(chunks).search(query, top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("scripts/retrieval_eval.py"))
        self.assertEqual(results[0].source, "hybrid")

    def test_search_uses_symbol_metadata_without_path_patch(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "def complete(messages):\n    return send(messages)"),
            CodeChunk(Path("b.py"), 1, 10, "def render_prompt(context):\n    return context"),
        ]

        results = BM25Retriever(chunks).search("complete 函数在哪里", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("a.py"))

    def test_search_deprioritizes_tests_without_test_intent(self):
        chunks = [
            CodeChunk(Path("coding_rag/loader.py"), 1, 10, "def load_data():\n    return []"),
            CodeChunk(Path("tests/test_loader.py"), 1, 10, "def test_load_data():\n    assert load_data() == []"),
        ]

        results = BM25Retriever(chunks).search("load_data 是如何实现的", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("coding_rag/loader.py"))

    def test_search_allows_tests_when_query_asks_for_tests(self):
        chunks = [
            CodeChunk(Path("coding_rag/loader.py"), 1, 10, "def load_data():\n    return []"),
            CodeChunk(Path("tests/test_loader.py"), 1, 10, "def test_load_data():\n    assert load_data() == []"),
        ]

        results = BM25Retriever(chunks).search("load_data 的测试在哪里", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("tests/test_loader.py"))

    def test_search_prefers_file_diversity_before_duplicate_chunks(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "target target target"),
            CodeChunk(Path("a.py"), 11, 20, "target target"),
            CodeChunk(Path("b.py"), 1, 10, "target"),
        ]

        results = BM25Retriever(chunks).search("target", top_k=2)

        self.assertEqual({result.chunk.file_path for result in results}, {Path("a.py"), Path("b.py")})


if __name__ == "__main__":
    unittest.main()
