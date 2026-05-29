import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.file_loader import load_python_files, should_skip


class FileLoaderIgnoreTest(unittest.TestCase):
    def test_load_python_files_skips_top_level_legacy_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "coding_rag"
            legacy = root / "legacy" / "repo_indexer"
            current.mkdir()
            legacy.mkdir(parents=True)
            (current / "file_loader.py").write_text("current = True\n", encoding="utf-8")
            (legacy / "file_loader.py").write_text("legacy = True\n", encoding="utf-8")

            files = load_python_files(root)

        self.assertEqual([file.path.name for file in files], ["file_loader.py"])
        self.assertIn("current", files[0].text)

    def test_should_skip_allows_legacy_when_it_is_the_repo_root(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "legacy"
            path = root / "repo_indexer" / "file_loader.py"
            path.parent.mkdir(parents=True)
            path.write_text("legacy = True\n", encoding="utf-8")

            self.assertFalse(should_skip(path, root))


if __name__ == "__main__":
    unittest.main()
