from pathlib import Path
import unittest

from coding_rag.bm25_retriever import BM25Retriever, query_path_boost
from coding_rag.code_splitter import CodeChunk


class BM25RetrieverRoleBoostTest(unittest.TestCase):
    def test_big_model_api_query_boosts_llm_client(self):
        query = "\u6700\u7ec8\u8c03\u7528\u5927\u6a21\u578b API \u7684\u4ee3\u7801\u5728\u54ea\u91cc"

        self.assertGreater(query_path_boost(query, Path("coding_rag/llm_client.py")), 0)
        self.assertEqual(query_path_boost(query, Path("rag/prompt.py")), 0)

    def test_search_uses_role_boost_for_semantic_file_location(self):
        query = "\u5982\u679c Recall@3 \u4e00\u76f4\u662f 0\uff0c\u4f46\u8089\u773c\u770b\u68c0\u7d22\u7ed3\u679c\u91cc\u6709\u6b63\u786e\u6587\u4ef6"
        chunks = [
            CodeChunk(Path("rag/prompt.py"), 1, 10, "Recall context and answer prompt"),
            CodeChunk(Path("scripts/retrieval_eval.py"), 1, 10, "matched_relevant_index recall_at_k"),
        ]

        results = BM25Retriever(chunks).search(query, top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("scripts/retrieval_eval.py"))

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
