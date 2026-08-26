"""Tests for repository-external write snapshots and rollback."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_rag.agent.snapshots import SnapshotStore
from coding_rag.tools.agent_readonly import AgentToolError


class SnapshotStoreTest(unittest.TestCase):
    def test_snapshot_diff_and_rollback_restore_existing_file(self):
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as cache_dir:
            root = Path(temp_dir)
            target = root / "app.py"
            target.write_text("value = 1\n", encoding="utf-8")
            store = SnapshotStore(root, snapshot_root=cache_dir)
            snapshot = store.create(["app.py"], tool="apply_patch")

            target.write_text("value = 2\n", encoding="utf-8")
            diff = store.diff(snapshot["snapshot_id"])
            rollback = store.rollback(snapshot["snapshot_id"])

            self.assertEqual(diff["files"], ["app.py"])
            self.assertIn("-value = 1", diff["text"])
            self.assertIn("+value = 2", diff["text"])
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertEqual(rollback["restored_files"], ["app.py"])
            self.assertEqual(rollback["remaining_diff"]["files"], [])

    def test_rollback_removes_file_created_after_snapshot(self):
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as cache_dir:
            root = Path(temp_dir)
            store = SnapshotStore(root, snapshot_root=cache_dir)
            snapshot = store.create(["new.py"], tool="apply_patch")
            target = root / "new.py"
            target.write_text("created = True\n", encoding="utf-8")

            rollback = store.rollback(snapshot["snapshot_id"])

            self.assertFalse(target.exists())
            self.assertEqual(rollback["removed_files"], ["new.py"])

    def test_snapshot_rejects_escape_and_empty_file_list(self):
        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as cache_dir:
            store = SnapshotStore(temp_dir, snapshot_root=cache_dir)

            with self.assertRaises(AgentToolError):
                store.create([], tool="run_command")
            with self.assertRaises(AgentToolError):
                store.create(["../outside.py"], tool="run_command")
            with self.assertRaises(AgentToolError):
                store.create(["C:/outside.py"], tool="run_command")


if __name__ == "__main__":
    unittest.main()
