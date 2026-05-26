from pathlib import Path
import unittest

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk
from coding_rag.context_recaller import expand_with_neighbor_chunks


class ContextRecallerTest(unittest.TestCase):
    def test_expand_with_neighbor_chunks(self):
        chunks = [
            CodeChunk(Path("a.py"), 1, 10, "before"),
            CodeChunk(Path("a.py"), 11, 20, "hit"),
            CodeChunk(Path("a.py"), 21, 30, "after"),
            CodeChunk(Path("b.py"), 1, 10, "other"),
        ]
        seeds = [SearchResult(chunks[1], 3.0)]

        results = expand_with_neighbor_chunks(chunks, seeds, window=1)

        self.assertEqual([result.chunk.text for result in results], ["before", "hit", "after"])
        self.assertEqual(results[1].source, "bm25")
        self.assertEqual(results[0].source, "recall:seed-1-1")
        self.assertEqual(results[2].source, "recall:seed-1+1")

    def test_window_zero_returns_seeds(self):
        chunk = CodeChunk(Path("a.py"), 1, 10, "hit")
        seeds = [SearchResult(chunk, 3.0)]

        results = expand_with_neighbor_chunks([chunk], seeds, window=0)

        self.assertEqual(results, seeds)


if __name__ == "__main__":
    unittest.main()
