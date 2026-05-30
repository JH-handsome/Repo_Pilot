from pathlib import Path
import unittest

from coding_rag.code_splitter import CodeChunk
from coding_rag.repo_index import build_repo_index


class RepoIndexTest(unittest.TestCase):
    def test_build_repo_index_extracts_symbols_imports_and_calls(self):
        chunk = CodeChunk(
            Path("pkg/service.py"),
            1,
            10,
            "\n".join(
                [
                    "from pathlib import Path",
                    "",
                    "class Loader(BaseLoader):",
                    "    def load_data(self, source):",
                    "        return Path(source).read_text()",
                ]
            ),
        )

        index = build_repo_index([chunk])
        file_index = index.for_chunk(chunk)

        self.assertIsNotNone(file_index)
        assert file_index is not None
        self.assertEqual(file_index.module, "pkg.service")
        self.assertIn("pkg", file_index.directories)
        self.assertIn("pathlib.Path", file_index.imports)
        self.assertIn("Path", file_index.calls)
        self.assertIn("Loader", [symbol.name for symbol in file_index.symbols])
        self.assertIn("load_data", [symbol.name for symbol in file_index.symbols])

    def test_metadata_for_chunk_contains_signature_and_graph_terms(self):
        chunk = CodeChunk(
            Path("pkg/service.py"),
            1,
            8,
            "\n".join(
                [
                    "import json",
                    "",
                    "def parse_payload(raw):",
                    "    return json.loads(raw)",
                ]
            ),
        )

        metadata = build_repo_index([chunk]).metadata_for_chunk(chunk)

        self.assertIn("module pkg.service", metadata)
        self.assertIn("signature parse_payload(raw)", metadata)
        self.assertIn("imports json", metadata)
        self.assertIn("calls json.loads", metadata)


if __name__ == "__main__":
    unittest.main()
