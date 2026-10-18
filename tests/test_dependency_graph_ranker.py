from pathlib import Path
import unittest

from coding_rag.repository.chunks import CodeChunk
from coding_rag.repository.graph import build_dependency_graph, personalized_dependency_scores
from coding_rag.repository.index import build_repo_index


class DependencyGraphRankerTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            CodeChunk(Path("pkg/api.py"), 1, 3, "from pkg.service import run\n\ndef endpoint():\n    return run()"),
            CodeChunk(Path("pkg/service.py"), 1, 2, "def run():\n    return 1"),
            CodeChunk(Path("pkg/isolated.py"), 1, 2, "def alone():\n    return 0"),
        ]
        self.index = build_repo_index(self.chunks)

    def test_builds_links_from_resolved_imports(self):
        graph = build_dependency_graph(self.index)
        self.assertIn("pkg/service.py", graph["pkg/api.py"])
        self.assertIn("pkg/api.py", graph["pkg/service.py"])
        self.assertEqual(graph["pkg/isolated.py"], ())

    def test_diffuses_query_evidence_to_dependency_neighbor(self):
        scores = personalized_dependency_scores(
            self.index,
            [chunk.file_path for chunk in self.chunks],
            [1.0, 0.0, 0.0],
        )
        self.assertGreater(scores[1], scores[2])

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            personalized_dependency_scores(self.index, [Path("pkg/api.py")], [])


if __name__ == "__main__":
    unittest.main()
