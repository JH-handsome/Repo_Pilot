from pathlib import Path
import unittest

from coding_rag.bm25_retriever import BM25Retriever
from coding_rag.code_splitter import CodeChunk


class BM25RetrieverRepoIndexTest(unittest.TestCase):
    def test_search_uses_ast_calls_for_caller_queries(self):
        chunks = [
            CodeChunk(Path("data/loader.py"), 1, 4, "def load_data(path):\n    return path"),
            CodeChunk(Path("app/service.py"), 1, 5, "from data.loader import load_data\n\ndef run():\n    return load_data('x')"),
        ]

        results = BM25Retriever(chunks).search("哪里调用 load_data", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("app/service.py"))

    def test_search_uses_import_metadata_for_import_queries(self):
        chunks = [
            CodeChunk(Path("app/io.py"), 1, 4, "from pathlib import Path\n\ndef read(name):\n    return Path(name).read_text()"),
            CodeChunk(Path("app/model.py"), 1, 3, "class PathConfig:\n    pass"),
        ]

        results = BM25Retriever(chunks).search("Path 在哪里导入", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("app/io.py"))

    def test_search_uses_signature_metadata_for_method_queries(self):
        chunks = [
            CodeChunk(Path("solutions/python3/1.py"), 1, 4, "class Solution:\n    def twoSum(self, nums, target):\n        return []"),
            CodeChunk(Path("solutions/python3/2.py"), 1, 4, "class Solution:\n    def addTwoNumbers(self, l1, l2):\n        return None"),
        ]

        results = BM25Retriever(chunks).search("Solution twoSum 函数签名", top_k=1)

        self.assertEqual(results[0].chunk.file_path, Path("solutions/python3/1.py"))


if __name__ == "__main__":
    unittest.main()
