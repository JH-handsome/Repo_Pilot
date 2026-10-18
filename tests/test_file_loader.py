import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_rag.repository.files import load_python_files, should_skip
from coding_rag.repository.search_index import RepositoryIndexKey, build_manifest


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

    def test_load_python_files_skips_file_symlink_outside_repository(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            (root / "safe.py").write_text("safe = True\n", encoding="utf-8")
            outside = Path(temp_dir) / "outside.py"
            outside.write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            try:
                (root / "linked.py").symlink_to(outside)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")

            files = load_python_files(root)

        self.assertEqual([file.path.name for file in files], ["safe.py"])

    def test_load_python_files_skips_env_named_python_file_and_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "safe.py").write_text("safe = True\n", encoding="utf-8")
            (root / ".env.py").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")
            env_dir = root / ".env"
            env_dir.mkdir()
            (env_dir / "secret.py").write_text("FAKE_TEST_VALUE\n", encoding="utf-8")

            files = load_python_files(root)

        self.assertEqual([file.path.name for file in files], ["safe.py"])

    def test_loader_and_read_policy_exclude_common_credential_paths(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "safe.py").write_text("safe = True\n", encoding="utf-8")
            (root / "secrets.py").write_text("FAKE_SECRET = 'not-real'\n", encoding="utf-8")
            (root / "credentials.py").write_text("FAKE_SECRET = 'not-real'\n", encoding="utf-8")
            ssh = root / ".ssh"
            ssh.mkdir()
            (ssh / "helper.py").write_text("FAKE_SECRET = 'not-real'\n", encoding="utf-8")

            files = load_python_files(root)

            self.assertEqual([file.path.name for file in files], ["safe.py"])
            self.assertEqual([entry.path for entry in build_manifest(RepositoryIndexKey.create(root))],
                             ["safe.py"])
            for path in ("secrets.py", "credentials.py", ".ssh/helper.py", "private.pem",
                         "token.json", ".npmrc"):
                self.assertTrue(should_skip(root / path, root), path)

    def test_sensitive_directory_as_selected_root_is_not_indexed(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ".ssh"
            root.mkdir()
            (root / "helper.py").write_text("FAKE_SECRET = 'not-real'\n", encoding="utf-8")
            self.assertEqual(load_python_files(root), [])
            self.assertEqual(build_manifest(RepositoryIndexKey.create(root)), ())


if __name__ == "__main__":
    unittest.main()
