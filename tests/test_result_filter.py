from pathlib import Path
import unittest

from coding_rag.tools.bm25 import BM25Retriever, SearchResult
from coding_rag.repository.chunks import CodeChunk
from coding_rag.tools.recall import expand_with_neighbor_chunks
from coding_rag.tools.filter import filter_recalled_results


class ResultFilterTest(unittest.TestCase):
    def test_search_scores_are_reused_by_final_filter(self):
        class CountingRetriever(BM25Retriever):
            def __init__(self, chunks):
                super().__init__(chunks)
                self.score_calls = 0

            def hybrid_scores(self, query):
                self.score_calls += 1
                return super().hybrid_scores(query)

        chunks = [
            CodeChunk(Path("a.py"), 1, 5, "def target(): return helper()"),
            CodeChunk(Path("a.py"), 6, 10, "def helper(): return 1"),
            CodeChunk(Path("b.py"), 1, 5, "def other(): return 2"),
        ]
        retriever = CountingRetriever(chunks)
        seeds, scores = retriever.search_with_scores("target helper", top_k=1)
        recalled = expand_with_neighbor_chunks(chunks, seeds, window=1)

        reused = filter_recalled_results(
            "target helper", recalled, retriever, final_k=2, scores=scores
        )
        self.assertEqual(retriever.score_calls, 1)

        recomputed = filter_recalled_results(
            "target helper", recalled, retriever, final_k=2
        )
        self.assertEqual(retriever.score_calls, 2)
        self.assertEqual(reused, recomputed)

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

    def test_finite_results_prefer_distinct_files_before_extra_chunks(self):
        chunks = [
            CodeChunk(Path("dominant.py"), 1, 10, "target target target"),
            CodeChunk(Path("dominant.py"), 11, 20, "target target"),
            CodeChunk(Path("alternative.py"), 1, 10, "target"),
        ]
        retriever = BM25Retriever(chunks)
        recalled = [SearchResult(chunk, 1.0, source="bm25") for chunk in chunks]

        results = filter_recalled_results(
            query="target",
            recalled_results=recalled,
            retriever=retriever,
            final_k=2,
        )

        self.assertEqual(
            [result.chunk.file_path for result in results],
            [Path("dominant.py"), Path("alternative.py")],
        )


if __name__ == "__main__":
    unittest.main()
