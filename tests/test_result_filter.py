from pathlib import Path
import unittest

from coding_rag.bm25_retriever import BM25Retriever, SearchResult
from coding_rag.code_splitter import CodeChunk
from coding_rag.context_recaller import expand_with_neighbor_chunks
from coding_rag.result_filter import filter_recalled_results


class ResultFilterTest(unittest.TestCase):
    def test_filter_recalled_results_keeps_best_query_match(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "helper setup"),
            CodeChunk(Path("a.py"), 11, 20, "def twoSum(nums, target): hash map"),
            CodeChunk(Path("a.py"), 21, 30, "binary tree traversal"),
        ]
        retriever = BM25Retriever(chunks)
        seeds = [SearchResult(chunks[1], 10.0)]
        recalled = expand_with_neighbor_chunks(chunks, seeds, window=1)

        results = filter_recalled_results(
            query="two sum hash map",
            recalled_results=recalled,
            retriever=retriever,
            final_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk, chunks[1])
        self.assertEqual(results[0].source, "bm25")

    def test_filter_recalled_results_can_return_all_sorted_results(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "unrelated"),
            CodeChunk(Path("a.py"), 11, 20, "TreeNode binary tree"),
            CodeChunk(Path("a.py"), 21, 30, "TreeNode left right"),
        ]
        retriever = BM25Retriever(chunks)
        recalled = [
            SearchResult(chunks[0], 1.0, source="recall:seed-1-1"),
            SearchResult(chunks[1], 2.0, source="bm25"),
            SearchResult(chunks[2], 1.0, source="recall:seed-1+1"),
        ]

        results = filter_recalled_results(
            query="binary tree TreeNode",
            recalled_results=recalled,
            retriever=retriever,
            final_k=None,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].chunk, chunks[1])


if __name__ == "__main__":
    unittest.main()
