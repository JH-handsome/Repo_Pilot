from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import time
import unittest
from unittest.mock import patch

from coding_rag.agent.runtime import UnifiedRunConfig, build_hybrid_search_provider
from coding_rag.repository.search_index import (
    GLOBAL_REPOSITORY_INDEX_REGISTRY,
    RepositoryIndexKey,
    RepositoryIndexRegistry,
    build_repository_index,
    canonical_json,
    clear_repository_index_registry,
    parse_term_frequency_matrix,
    safe_relative_path,
)


def read_cache(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def write_cache(path: Path, envelope: dict) -> None:
    path.write_bytes(gzip.compress(json.dumps(envelope).encode("utf-8"), mtime=0))


class RepositorySearchIndexTest(unittest.TestCase):
    def setUp(self):
        clear_repository_index_registry()

    def tearDown(self):
        clear_repository_index_registry()

    def make_repo(self, root: Path) -> None:
        (root / "app.py").write_text("def target():\n    return 1\n", encoding="utf-8")

    def test_term_frequency_cache_rows_keep_order_and_reject_invalid_entries(self):
        vocabulary = ["alpha", "beta", "gamma"]
        self.assertEqual(
            parse_term_frequency_matrix([[[0, 2], [2, 1]], []], 2, vocabulary),
            ((('alpha', 2), ('gamma', 1)), ()),
        )
        invalid_rows = (
            [[True, 1]], [[0, True]], [[0, 0]], [[3, 1]],
            [[1, 1], [1, 2]], [[2, 1], [0, 1]], [[0]],
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_term_frequency_matrix([row], 1, vocabulary)

    def test_repeated_lookup_hits_one_built_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            registry = RepositoryIndexRegistry(disk_cache=False)
            key = RepositoryIndexKey.create(root)
            first = registry.get(key)
            second = registry.get(key)
        self.assertEqual(first.status, "miss")
        self.assertEqual(second.status, "hit")
        self.assertIs(first.index, second.index)
        self.assertEqual(registry.entry_count(), 1)

    def test_separate_runtime_providers_reuse_process_registry_and_trace_it(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            config = UnifiedRunConfig(repo_path=root, top_k=1, recall_window=0)
            first = build_hybrid_search_provider(config)("target", 1)
            second = build_hybrid_search_provider(config)("target", 1)
        first_index = first["retrieval_trace"]["artifacts"]["index"]
        second_index = second["retrieval_trace"]["artifacts"]["index"]
        self.assertEqual(first_index["status"], "miss")
        self.assertEqual(second_index["status"], "hit")
        self.assertEqual(first_index["file_count"], 1)
        self.assertGreater(first_index["chunk_count"], 0)
        self.assertIn("manifest_scan_ms", second_index)
        self.assertEqual(second["retrieval_trace"]["summary"]["index_status"], "hit")

    def test_add_modify_delete_invalidate_and_non_python_does_not(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            registry = RepositoryIndexRegistry(disk_cache=False)
            key = RepositoryIndexKey.create(root)
            registry.get(key)

            (root / "notes.md").write_text("ignored", encoding="utf-8")
            self.assertEqual(registry.get(key).status, "hit")

            added = root / "added.py"
            added.write_text("VALUE = 2\n", encoding="utf-8")
            after_add = registry.get(key)
            self.assertEqual(after_add.status, "stale")
            self.assertEqual(after_add.changed_file_count, 1)

            added.write_text("VALUE = 200\n", encoding="utf-8")
            after_modify = registry.get(key)
            self.assertEqual(after_modify.status, "stale")
            self.assertEqual(after_modify.changed_file_count, 1)

            added.unlink()
            after_delete = registry.get(key)
            self.assertEqual(after_delete.status, "stale")
            self.assertEqual(after_delete.changed_file_count, 1)

    def test_same_key_concurrent_build_is_single_flight(self):
        calls = 0
        calls_lock = Lock()

        def counting_builder(key, manifest):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return build_repository_index(key, manifest)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            registry = RepositoryIndexRegistry(builder=counting_builder, disk_cache=False)
            key = RepositoryIndexKey.create(root)
            with ThreadPoolExecutor(max_workers=8) as pool:
                accesses = list(pool.map(lambda _: registry.get(key), range(8)))
        self.assertEqual(calls, 1)
        self.assertEqual(sum(access.status == "miss" for access in accesses), 1)
        self.assertEqual(sum(access.status == "hit" for access in accesses), 7)

    def test_failed_stale_build_keeps_old_snapshot_and_can_recover(self):
        fail = False
        calls = 0

        def controllable_builder(key, manifest):
            nonlocal calls
            calls += 1
            if fail:
                raise RuntimeError("synthetic build failure")
            return build_repository_index(key, manifest)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            registry = RepositoryIndexRegistry(builder=controllable_builder, disk_cache=False)
            key = RepositoryIndexKey.create(root)
            original = registry.get(key).index
            (root / "app.py").write_text("def target():\n    return 200\n", encoding="utf-8")
            fail = True
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                registry.get(key)
            self.assertEqual(registry.entry_count(), 1)
            fail = False
            recovered = registry.get(key)
        self.assertEqual(recovered.status, "stale")
        self.assertIsNot(recovered.index, original)
        self.assertEqual(calls, 3)

    def test_default_excludes_corpora_and_weighted_keys_do_not_alias(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root)
            (root / "data").mkdir()
            (root / "data" / "corpus.py").write_text("CORPUS = True\n", encoding="utf-8")
            registry = RepositoryIndexRegistry(max_entries=2, disk_cache=False)
            default = registry.get(RepositoryIndexKey.create(root))
            weighted = registry.get(RepositoryIndexKey.create(root, path_weight=0.5))
        self.assertEqual(default.index.file_count, 1)
        self.assertEqual(weighted.status, "miss")
        self.assertEqual(registry.entry_count(), 2)

    def test_registry_evicts_least_recently_used_snapshot_at_bound(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first_root = base / "first"
            second_root = base / "second"
            first_root.mkdir()
            second_root.mkdir()
            self.make_repo(first_root)
            self.make_repo(second_root)
            registry = RepositoryIndexRegistry(max_entries=1, disk_cache=False)
            first_key = RepositoryIndexKey.create(first_root)
            registry.get(first_key)
            registry.get(RepositoryIndexKey.create(second_root))
            rebuilt = registry.get(first_key)
        self.assertEqual(registry.entry_count(), 1)
        self.assertEqual(rebuilt.status, "miss")

    def test_cold_start_writes_json_and_new_registry_uses_derived_disk_cache(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "repo"
            cache = base / "cache"
            root.mkdir()
            for index in range(80):
                (root / f"module_{index}.py").write_text(
                    f"class Unique{index}:\n    def method_{index}(self):\n        return {index}\n",
                    encoding="utf-8",
                )
            key = RepositoryIndexKey.create(root)
            cold = RepositoryIndexRegistry(cache_dir=cache).get(key)
            with patch(
                "coding_rag.tools.bm25.build_repo_index",
                side_effect=AssertionError("disk hit must not rebuild AST metadata"),
            ):
                warm = RepositoryIndexRegistry(cache_dir=cache).get(key)
            cold_results = [
                (item.chunk.file_path.name, item.score)
                for item in cold.index.retriever.search("Unique17 method_17", top_k=5)
            ]
            warm_results = [
                (item.chunk.file_path.name, item.score)
                for item in warm.index.retriever.search("Unique17 method_17", top_k=5)
            ]
            cold_ties = [
                (item.chunk.file_path.name, item.chunk.start_line, item.chunk.end_line, item.score)
                for item in cold.index.retriever.search("class method return", top_k=20)
            ]
            warm_ties = [
                (item.chunk.file_path.name, item.chunk.start_line, item.chunk.end_line, item.score)
                for item in warm.index.retriever.search("class method return", top_k=20)
            ]
            cache_files = list(cache.glob("*.json.gz"))
        self.assertEqual(cold.lifecycle, "full_rebuild")
        self.assertGreater(cold.index_build_ms, 0)
        self.assertGreater(cold.disk_write_ms, 0)
        self.assertEqual(cold.disk_write_status, "written")
        self.assertEqual(warm.lifecycle, "disk_hit")
        self.assertEqual(warm.disk_write_status, "not_attempted")
        self.assertEqual(warm.index_build_ms, 0)
        self.assertGreater(warm.disk_read_ms, 0)
        self.assertLess(warm.disk_read_ms, cold.index_build_ms)
        self.assertEqual(len(cache_files), 1)
        self.assertEqual(cold_results, warm_results)
        self.assertEqual(cold_ties, warm_ties)
        self.assertEqual(
            [item.file_path.name for item in cold.index.chunks],
            [item.file_path.name for item in warm.index.chunks],
        )

    def test_incremental_modify_reuses_unchanged_file_and_rebuilds_one(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            other = root / "other.py"
            other.write_text("def other():\n    return 2\n", encoding="utf-8")
            key = RepositoryIndexKey.create(root)
            RepositoryIndexRegistry(cache_dir=cache).get(key)
            other.write_text("def other():\n    return 22222\n", encoding="utf-8")
            access = RepositoryIndexRegistry(cache_dir=cache).get(key)
        self.assertEqual(access.lifecycle, "incremental")
        self.assertEqual(access.reused_file_count, 1)
        self.assertEqual(access.rebuilt_file_count, 1)
        self.assertEqual(access.removed_file_count, 0)
        self.assertGreater(access.reused_chunk_count, 0)
        self.assertGreater(access.rebuilt_chunk_count, 0)
        self.assertGreater(access.removed_chunk_count, 0)

    def test_incremental_delete_removes_file_and_chunks(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            removed = root / "removed.py"
            removed.write_text("VALUE = 2\n", encoding="utf-8")
            key = RepositoryIndexKey.create(root)
            RepositoryIndexRegistry(cache_dir=cache).get(key)
            removed.unlink()
            access = RepositoryIndexRegistry(cache_dir=cache).get(key)
        self.assertEqual(access.lifecycle, "incremental")
        self.assertEqual(access.removed_file_count, 1)
        self.assertEqual(access.rebuilt_file_count, 0)
        self.assertGreater(access.removed_chunk_count, 0)
        self.assertEqual(access.index.file_count, 1)

    def test_corrupt_and_incompatible_cache_fall_back_to_full_rebuild(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            first_registry = RepositoryIndexRegistry(cache_dir=cache)
            first_registry.get(key)
            cache_path = first_registry._cache_path(key)
            cache_path.write_text("not-json", encoding="utf-8")
            corrupt = RepositoryIndexRegistry(cache_dir=cache).get(key)
            envelope = read_cache(cache_path)
            envelope["format_version"] = 999
            write_cache(cache_path, envelope)
            incompatible = RepositoryIndexRegistry(cache_dir=cache).get(key)
            other_config = RepositoryIndexRegistry(cache_dir=cache).get(
                RepositoryIndexKey.create(root, path_weight=0.5)
            )
        self.assertEqual(corrupt.lifecycle, "full_rebuild")
        self.assertEqual(incompatible.lifecycle, "full_rebuild")
        self.assertEqual(other_config.lifecycle, "full_rebuild")

    def test_failed_atomic_replace_preserves_last_successful_cache(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            registry.get(key)
            cache_path = registry._cache_path(key)
            before = cache_path.read_bytes()
            (root / "app.py").write_text("def target():\n    return 999\n", encoding="utf-8")
            with patch("coding_rag.repository.search_index.os.replace", side_effect=OSError("disk full")):
                access = registry.get(key)
            after = cache_path.read_bytes()
            recovered = RepositoryIndexRegistry(cache_dir=cache).get(key)
        self.assertEqual(access.lifecycle, "incremental")
        self.assertEqual(before, after)
        self.assertEqual(access.disk_write_status, "failed")
        self.assertEqual(recovered.lifecycle, "incremental")

    def test_optional_serialization_failure_still_publishes_memory_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            with patch(
                "coding_rag.repository.search_index.snapshot_envelope",
                side_effect=ValueError("cannot serialize"),
            ):
                first = registry.get(key)
            second = registry.get(key)
        self.assertEqual(first.lifecycle, "full_rebuild")
        self.assertEqual(first.disk_write_status, "failed")
        self.assertEqual(second.lifecycle, "memory_hit")

    def test_unwritable_cache_attempts_temp_creation_once_and_keeps_memory_index(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            with patch(
                "coding_rag.repository.search_index.os.open",
                side_effect=PermissionError("cache is read-only"),
            ) as create, patch(
                "coding_rag.repository.search_index.snapshot_envelope",
                side_effect=AssertionError("unwritable cache must not serialize"),
            ):
                first = registry.get(key)
            second = registry.get(key)

        self.assertEqual(create.call_count, 1)
        self.assertEqual(first.lifecycle, "full_rebuild")
        self.assertEqual(first.disk_write_status, "failed")
        self.assertEqual(second.lifecycle, "memory_hit")

    def test_cache_size_limits_fall_back_without_blocking_memory_publish(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            with patch("coding_rag.repository.search_index.MAX_CACHE_JSON_BYTES", 1):
                first = registry.get(key)
            self.assertFalse(registry._cache_path(key).exists())
            memory = registry.get(key)
            RepositoryIndexRegistry(cache_dir=cache).get(key)
            with patch("coding_rag.repository.search_index.MAX_CACHE_COMPRESSED_BYTES", 1):
                recovered = RepositoryIndexRegistry(cache_dir=cache).get(key)
        self.assertEqual(first.lifecycle, "full_rebuild")
        self.assertEqual(memory.lifecycle, "memory_hit")
        self.assertEqual(recovered.lifecycle, "full_rebuild")

    def test_file_vanishing_after_manifest_scan_retries_with_fresh_manifest(self):
        calls = 0

        def vanishing_builder(key, manifest):
            nonlocal calls
            calls += 1
            if calls == 1:
                (Path(key.repo_path) / "vanish.py").unlink()
                raise FileNotFoundError("vanished")
            return build_repository_index(key, manifest)

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            (root / "vanish.py").write_text("VALUE = 1\n", encoding="utf-8")
            access = RepositoryIndexRegistry(
                cache_dir=cache, builder=vanishing_builder
            ).get(RepositoryIndexKey.create(root))
        self.assertEqual(calls, 2)
        self.assertEqual(access.index.file_count, 1)

    def test_cache_paths_are_confined_and_payload_has_no_absolute_repo_path(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            registry.get(key)
            payload = gzip.decompress(registry._cache_path(key).read_bytes()).decode("utf-8")
            decoded = json.loads(payload)
            unsafe = RepositoryIndexRegistry(cache_dir=root / "cache")
            with self.assertRaisesRegex(ValueError, "outside the indexed repository"):
                unsafe.get(key)
        self.assertNotIn(str(root), payload)
        self.assertNotIn(str(root).replace("\\", "/"), payload)
        self.assertNotIn(base.name.casefold(), payload.casefold())
        file_index = decoded["payload"]["prepared"]["files"][0]
        self.assertEqual(file_index["module"], "app")
        self.assertEqual(file_index["directories"], [])
        self.assertEqual(len(registry._cache_path(key).name.split(".", 1)[0]), len(hashlib.sha256().hexdigest()))

    def test_cached_paths_require_real_confined_strings_and_file_indexes_match_chunks(self):
        for unsafe in (None, 7, "", ".", "../escape.py", "/absolute.py", "C:drive.py", "C:\\drive.py"):
            with self.assertRaises(ValueError):
                safe_relative_path(unsafe)
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            registry.get(key)
            cache_path = registry._cache_path(key)
            envelope = read_cache(cache_path)
            envelope["payload"]["prepared"]["files"][0]["path"] = "different.py"
            envelope["checksum"] = hashlib.sha256(canonical_json(envelope["payload"])).hexdigest()
            write_cache(cache_path, envelope)
            recovered = RepositoryIndexRegistry(cache_dir=cache).get(key)
        self.assertEqual(recovered.lifecycle, "full_rebuild")

    def test_valid_checksum_does_not_make_boolean_cache_numbers_valid(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root, cache = base / "repo", base / "cache"
            root.mkdir()
            self.make_repo(root)
            key = RepositoryIndexKey.create(root)
            registry = RepositoryIndexRegistry(cache_dir=cache)
            registry.get(key)
            cache_path = registry._cache_path(key)

            for field in ("manifest_size", "path_token_id", "symbol_line"):
                with self.subTest(field=field):
                    envelope = read_cache(cache_path)
                    payload = envelope["payload"]
                    if field == "manifest_size":
                        payload["manifest"][0]["size"] = True
                    elif field == "path_token_id":
                        payload["prepared"]["path_token_sets"][0] = [True]
                    else:
                        payload["prepared"]["files"][0]["symbols"][0]["start_line"] = True
                    envelope["checksum"] = hashlib.sha256(canonical_json(payload)).hexdigest()
                    write_cache(cache_path, envelope)
                    self.assertEqual(
                        RepositoryIndexRegistry(cache_dir=cache).get(key).lifecycle,
                        "full_rebuild",
                    )


if __name__ == "__main__":
    unittest.main()
