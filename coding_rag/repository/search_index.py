"""Process-wide, change-aware repository search index registry."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock, RLock
from time import perf_counter
from typing import Callable
from uuid import uuid4
from weakref import WeakValueDictionary

from coding_rag.repository.chunks import CodeChunk, split_python_files
from coding_rag.repository.files import PythonFile, iter_python_paths, load_python_files
from coding_rag.repository.index import FileIndex, RepoIndex, SymbolRecord, normalize_path, rebuild_source
from coding_rag.tools.bm25 import (
    DEFAULT_GRAPH_WEIGHT,
    DEFAULT_PATH_WEIGHT,
    BM25Retriever,
    PreparedBM25Data,
    prepare_bm25_data,
)


DEFAULT_INDEX_EXCLUDE_ROOTS = ("data", "datasets", "artifacts")
RETRIEVER_VERSION = "hybrid-bm25-v7"
CACHE_FORMAT = "repo-pilot-repository-index"
CACHE_FORMAT_VERSION = 3
BM25_STATE_SCHEMA_VERSION = 1
MAX_CACHE_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_CACHE_JSON_BYTES = 256 * 1024 * 1024


def default_cache_dir() -> Path:
    """Return an OS user-cache location outside source checkouts by default."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "RepoPilot" / "repository_indexes"
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "repo_pilot" / "repository_indexes"
    return Path.home() / ".cache" / "repo_pilot" / "repository_indexes"


DEFAULT_CACHE_DIR = default_cache_dir()


@dataclass(frozen=True, order=True)
class ManifestEntry:
    """Portable file identity used to invalidate an in-memory index."""

    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class RepositoryIndexKey:
    """All repository and retrieval settings that affect index contents or scores."""

    repo_path: str
    chunk_size: int
    overlap: int
    graph_weight: float
    path_weight: float
    retriever_version: str
    exclude_roots: tuple[str, ...]

    @classmethod
    def create(
        cls,
        repo_path: str | Path,
        *,
        chunk_size: int = 40,
        overlap: int = 5,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
        path_weight: float = DEFAULT_PATH_WEIGHT,
        exclude_roots: tuple[str, ...] = DEFAULT_INDEX_EXCLUDE_ROOTS,
    ) -> "RepositoryIndexKey":
        """Normalize an index configuration into a stable registry key."""
        root = Path(repo_path).resolve()
        return cls(
            repo_path=str(root),
            chunk_size=chunk_size,
            overlap=overlap,
            graph_weight=float(graph_weight),
            path_weight=float(path_weight),
            retriever_version=RETRIEVER_VERSION,
            exclude_roots=tuple(sorted(set(exclude_roots))),
        )


@dataclass(frozen=True)
class PreparedRepositoryIndex:
    """An immutable published repository snapshot and its search structures."""

    key: RepositoryIndexKey
    manifest: tuple[ManifestEntry, ...]
    chunks: tuple[CodeChunk, ...]
    retriever: BM25Retriever
    file_count: int
    chunk_count: int


@dataclass(frozen=True)
class RepositoryIndexAccess:
    """One registry lookup paired with cache and build diagnostics."""

    index: PreparedRepositoryIndex
    status: str
    manifest_scan_ms: float
    index_build_ms: float
    changed_file_count: int
    lifecycle: str = "full_rebuild"
    reused_file_count: int = 0
    rebuilt_file_count: int = 0
    removed_file_count: int = 0
    reused_chunk_count: int = 0
    rebuilt_chunk_count: int = 0
    removed_chunk_count: int = 0
    disk_read_ms: float = 0.0
    disk_write_ms: float = 0.0
    disk_write_status: str = "not_attempted"

    def trace_metadata(self) -> dict:
        """Return JSON-safe index lifecycle evidence for retrieval traces."""
        return {
            "status": self.status,
            "manifest_scan_ms": self.manifest_scan_ms,
            "index_build_ms": self.index_build_ms,
            "file_count": self.index.file_count,
            "chunk_count": self.index.chunk_count,
            "changed_file_count": self.changed_file_count,
            "lifecycle": self.lifecycle,
            "reused_file_count": self.reused_file_count,
            "rebuilt_file_count": self.rebuilt_file_count,
            "removed_file_count": self.removed_file_count,
            "reused_chunk_count": self.reused_chunk_count,
            "rebuilt_chunk_count": self.rebuilt_chunk_count,
            "removed_chunk_count": self.removed_chunk_count,
            "disk_read_ms": self.disk_read_ms,
            "disk_write_ms": self.disk_write_ms,
            "disk_write_status": self.disk_write_status,
            "retriever_version": self.index.key.retriever_version,
        }


IndexBuilder = Callable[[RepositoryIndexKey, tuple[ManifestEntry, ...]], PreparedRepositoryIndex]


@dataclass(frozen=True)
class _BuildStats:
    """Internal file and chunk reuse evidence for one candidate build."""

    reused_files: int
    rebuilt_files: int
    removed_files: int
    reused_chunks: int
    rebuilt_chunks: int
    removed_chunks: int


class RepositoryIndexRegistry:
    """Bounded LRU registry with per-key single-flight index construction."""

    def __init__(
        self,
        max_entries: int = 8,
        builder: IndexBuilder | None = None,
        cache_dir: str | Path | None = None,
        disk_cache: bool = True,
    ):
        """Create a bounded registry; builder injection is reserved for deterministic tests."""
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self.max_entries = max_entries
        self._builder = builder or build_repository_index
        self._custom_builder = builder is not None
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else DEFAULT_CACHE_DIR.resolve()
        self.disk_cache = bool(disk_cache)
        self._entries: OrderedDict[RepositoryIndexKey, PreparedRepositoryIndex] = OrderedDict()
        self._key_locks: WeakValueDictionary[RepositoryIndexKey, Lock] = WeakValueDictionary()
        self._lock = RLock()

    def get(self, key: RepositoryIndexKey) -> RepositoryIndexAccess:
        """Return a current index, rebuilding once when its Python manifest changed."""
        self._validate_cache_location(key)
        key_lock = self._key_lock(key)
        with key_lock:
            total_scan_ms = 0.0
            for _ in range(3):
                scan_started = perf_counter()
                manifest = build_manifest(key)
                total_scan_ms += (perf_counter() - scan_started) * 1000
                with self._lock:
                    previous = self._entries.get(key)
                    if previous is not None and previous.manifest == manifest:
                        self._entries.move_to_end(key)
                        return RepositoryIndexAccess(
                            previous, "hit", total_scan_ms, 0.0, 0,
                            lifecycle="memory_hit",
                            reused_file_count=previous.file_count,
                            reused_chunk_count=previous.chunk_count,
                        )

                disk_snapshot = None
                disk_read_ms = 0.0
                if previous is None and self.disk_cache:
                    disk_snapshot, disk_read_ms = self._read_disk_snapshot(key)
                    if disk_snapshot is not None and disk_snapshot.manifest == manifest:
                        self._publish(key, disk_snapshot)
                        return RepositoryIndexAccess(
                            disk_snapshot, "miss", total_scan_ms, 0.0, 0,
                            lifecycle="disk_hit",
                            reused_file_count=disk_snapshot.file_count,
                            reused_chunk_count=disk_snapshot.chunk_count,
                            disk_read_ms=disk_read_ms,
                        )

                baseline = previous or disk_snapshot
                status = "stale" if previous is not None else "miss"
                changed = changed_file_count(baseline.manifest if baseline else (), manifest)
                build_started = perf_counter()
                try:
                    if baseline is not None and not self._custom_builder:
                        candidate, stats = build_incremental_repository_index(key, manifest, baseline)
                        lifecycle = "incremental"
                    else:
                        candidate = self._builder(key, manifest)
                        lifecycle = "full_rebuild"
                        stats = _BuildStats(
                            0, candidate.file_count, 0, 0, candidate.chunk_count, 0
                        )
                except FileNotFoundError:
                    # A source file vanished after the metadata scan; retry from a fresh manifest.
                    continue
                build_ms = (perf_counter() - build_started) * 1000

                verify_started = perf_counter()
                verified_manifest = build_manifest(key)
                total_scan_ms += (perf_counter() - verify_started) * 1000
                if verified_manifest != manifest:
                    continue

                disk_write_ms, disk_write_status = (
                    self._write_disk_snapshot(candidate)
                    if self.disk_cache else (0.0, "disabled")
                )
                self._publish(key, candidate)
                return RepositoryIndexAccess(
                    candidate, status, total_scan_ms, build_ms, changed,
                    lifecycle=lifecycle,
                    reused_file_count=stats.reused_files,
                    rebuilt_file_count=stats.rebuilt_files,
                    removed_file_count=stats.removed_files,
                    reused_chunk_count=stats.reused_chunks,
                    rebuilt_chunk_count=stats.rebuilt_chunks,
                    removed_chunk_count=stats.removed_chunks,
                    disk_read_ms=disk_read_ms,
                    disk_write_ms=disk_write_ms,
                    disk_write_status=disk_write_status,
                )
            raise RuntimeError("repository changed repeatedly while its search index was being built")

    def _publish(self, key: RepositoryIndexKey, candidate: PreparedRepositoryIndex) -> None:
        """Atomically publish one complete snapshot to the bounded in-memory LRU."""
        with self._lock:
            self._entries[key] = candidate
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def _cache_path(self, key: RepositoryIndexKey) -> Path:
        """Return a filename derived only from a one-way hash of repo identity and config."""
        identity = cache_identity(key)
        return self.cache_dir / f"{identity}.json.gz"

    def _validate_cache_location(self, key: RepositoryIndexKey) -> None:
        """Reject disk cache paths that could pollute an indexed repository."""
        if not self.disk_cache:
            return
        repo = Path(key.repo_path).resolve()
        try:
            relative = self.cache_dir.relative_to(repo)
        except ValueError:
            return
        if not relative.parts or relative.parts[0] != "artifacts" or "artifacts" not in key.exclude_roots:
            raise ValueError(
                "index cache must be outside the indexed repository or under its excluded artifacts directory"
            )

    def _read_disk_snapshot(self, key: RepositoryIndexKey) -> tuple[PreparedRepositoryIndex | None, float]:
        """Load and strictly validate safe JSON cache data, falling back on any defect."""
        started = perf_counter()
        path = self._cache_path(key)
        try:
            if path.stat().st_size > MAX_CACHE_COMPRESSED_BYTES:
                raise ValueError("compressed cache exceeds size limit")
            with gzip.open(path, "rb") as handle:
                encoded = handle.read(MAX_CACHE_JSON_BYTES + 1)
            if len(encoded) > MAX_CACHE_JSON_BYTES:
                raise ValueError("decompressed cache exceeds size limit")
            envelope = json.loads(encoded)
            snapshot = snapshot_from_envelope(envelope, key)
        except (OSError, EOFError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            snapshot = None
        return snapshot, (perf_counter() - started) * 1000

    def _write_disk_snapshot(self, index: PreparedRepositoryIndex) -> tuple[float, str]:
        """Atomically replace safe JSON cache data; preserve the old cache on failure."""
        started = perf_counter()
        target = self._cache_path(index.key)
        temporary: Path | None = None
        status = "written"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_dir / f"{target.name}.{uuid4().hex}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                envelope = snapshot_envelope(index)
                encoded = canonical_json(envelope)
                if len(encoded) > MAX_CACHE_JSON_BYTES:
                    raise ValueError("cache JSON exceeds size limit")
                compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
                if len(compressed) > MAX_CACHE_COMPRESSED_BYTES:
                    raise ValueError("compressed cache exceeds size limit")
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        except (OSError, TypeError, ValueError):
            status = "failed"
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return (perf_counter() - started) * 1000, status

    def clear(self) -> None:
        """Clear published entries and per-key locks for tests or explicit lifecycle shutdown."""
        with self._lock:
            self._entries.clear()
            self._key_locks.clear()

    def entry_count(self) -> int:
        """Return the number of currently published snapshots."""
        with self._lock:
            return len(self._entries)

    def _key_lock(self, key: RepositoryIndexKey) -> Lock:
        """Return the single-flight lock assigned to one exact index key."""
        with self._lock:
            return self._key_locks.setdefault(key, Lock())


def build_manifest(key: RepositoryIndexKey) -> tuple[ManifestEntry, ...]:
    """Scan indexable Python files without reading their contents."""
    root = Path(key.repo_path)
    entries: list[ManifestEntry] = []
    for path in iter_python_paths(root, exclude_roots=key.exclude_roots):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append(ManifestEntry(path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(entries))


def build_repository_index(
    key: RepositoryIndexKey,
    manifest: tuple[ManifestEntry, ...],
) -> PreparedRepositoryIndex:
    """Build a complete candidate snapshot without publishing partial state."""
    files = load_python_files(key.repo_path, exclude_roots=key.exclude_roots)
    chunks = split_python_files(files, chunk_size=key.chunk_size, overlap=key.overlap)
    if not chunks:
        raise ValueError("No Python code chunks found in repository")
    retriever = BM25Retriever(
        chunks,
        graph_weight=key.graph_weight,
        path_weight=key.path_weight,
        path_root=Path(key.repo_path),
    )
    return PreparedRepositoryIndex(key, manifest, tuple(chunks), retriever, len(files), len(chunks))


def build_incremental_repository_index(
    key: RepositoryIndexKey,
    manifest: tuple[ManifestEntry, ...],
    previous: PreparedRepositoryIndex,
) -> tuple[PreparedRepositoryIndex, _BuildStats]:
    """Reuse unchanged file chunks and derived data, rebuilding only changed files."""
    root = Path(key.repo_path)
    before = {entry.path: entry for entry in previous.manifest}
    after = {entry.path: entry for entry in manifest}
    unchanged = {path for path in before.keys() & after.keys() if before[path] == after[path]}
    rebuilt_paths = set(after) - unchanged
    removed_paths = set(before) - set(after)

    previous_chunks: dict[str, list[tuple[int, CodeChunk]]] = {}
    for index, chunk in enumerate(previous.chunks):
        relative = chunk.file_path.resolve().relative_to(root.resolve()).as_posix()
        previous_chunks.setdefault(relative, []).append((index, chunk))
    previous_prepared = previous.retriever.prepared_data()
    chunks: list[CodeChunk] = []
    content_terms: list[tuple[tuple[str, int], ...]] = []
    metadata_terms: list[tuple[tuple[str, int], ...]] = []
    path_tokens: list[frozenset[str]] = []
    repo_files: dict[str, FileIndex] = {}
    rebuilt_chunk_count = 0

    for entry in manifest:
        if entry.path in unchanged:
            indexed_chunks = previous_chunks.get(entry.path, [])
            for old_index, chunk in indexed_chunks:
                chunks.append(chunk)
                content_terms.append(previous_prepared.content_term_frequencies[old_index])
                metadata_terms.append(previous_prepared.metadata_term_frequencies[old_index])
                path_tokens.append(previous_prepared.path_token_sets[old_index])
            if indexed_chunks:
                old_file = previous.retriever.repo_index.for_chunk(indexed_chunks[0][1])
                if old_file is not None:
                    repo_files[normalize_path(old_file.path)] = old_file
            continue

        path = root / Path(*entry.path.split("/"))
        text = path.read_text(encoding="utf-8", errors="ignore")
        new_chunks = split_python_files(
            [PythonFile(path=path, text=text)], chunk_size=key.chunk_size, overlap=key.overlap
        )
        file_prepared = prepare_bm25_data(
            new_chunks, previous.retriever.tokenizer, path_root=root
        )
        chunks.extend(new_chunks)
        content_terms.extend(file_prepared.content_term_frequencies)
        metadata_terms.extend(file_prepared.metadata_term_frequencies)
        path_tokens.extend(file_prepared.path_token_sets)
        repo_files.update(file_prepared.repo_index.files)
        rebuilt_chunk_count += len(new_chunks)

    if not chunks:
        raise ValueError("No Python code chunks found in repository")
    prepared = PreparedBM25Data(
        RepoIndex(repo_files), tuple(content_terms), tuple(metadata_terms), tuple(path_tokens)
    )
    retriever = BM25Retriever(
        chunks,
        tokenizer=previous.retriever.tokenizer,
        graph_weight=key.graph_weight,
        path_weight=key.path_weight,
        prepared_data=prepared,
        path_root=root,
    )
    removed_chunk_count = sum(
        len(previous_chunks.get(path, ())) for path in removed_paths | (set(before) & rebuilt_paths)
    )
    stats = _BuildStats(
        reused_files=len(unchanged),
        rebuilt_files=len(rebuilt_paths),
        removed_files=len(removed_paths),
        reused_chunks=len(chunks) - rebuilt_chunk_count,
        rebuilt_chunks=rebuilt_chunk_count,
        removed_chunks=removed_chunk_count,
    )
    return (
        PreparedRepositoryIndex(key, manifest, tuple(chunks), retriever, len(manifest), len(chunks)),
        stats,
    )


def cache_identity(key: RepositoryIndexKey) -> str:
    """Hash repository identity and every index-affecting setting into a cache filename."""
    payload = {
        "repository_id": repository_identity(key),
        "config": cache_config(key),
        "cache_format_version": CACHE_FORMAT_VERSION,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def repository_identity(key: RepositoryIndexKey) -> str:
    """Return a one-way local repository identity without persisting an absolute path."""
    normalized = os.path.normcase(str(Path(key.repo_path).resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_config(key: RepositoryIndexKey) -> dict:
    """Return all portable settings required for strict cache compatibility."""
    return {
        "chunk_size": key.chunk_size,
        "overlap": key.overlap,
        "graph_weight": key.graph_weight,
        "path_weight": key.path_weight,
        "retriever_version": key.retriever_version,
        "bm25_state_schema_version": BM25_STATE_SCHEMA_VERSION,
        "exclude_roots": list(key.exclude_roots),
    }


def canonical_json(payload: object) -> bytes:
    """Encode deterministic JSON bytes for checksums without executable serialization."""
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def snapshot_envelope(index: PreparedRepositoryIndex) -> dict:
    """Serialize a repository snapshot and reusable derived data into checksummed JSON."""
    root = Path(index.key.repo_path).resolve()
    prepared = index.retriever.prepared_data()
    vocabulary = sorted(
        {
            token
            for matrix in (
                prepared.content_term_frequencies,
                prepared.metadata_term_frequencies,
            )
            for row in matrix
            for token, _ in row
        }
        | {token for row in prepared.path_token_sets for token in row}
    )
    token_ids = {token: position for position, token in enumerate(vocabulary)}
    chunks_by_path: dict[str, list[CodeChunk]] = {}
    for chunk in index.chunks:
        relative = chunk.file_path.resolve().relative_to(root).as_posix()
        chunks_by_path.setdefault(relative, []).append(chunk)
    payload = {
        "repository_id": repository_identity(index.key),
        "config": cache_config(index.key),
        "manifest": [entry.__dict__ for entry in index.manifest],
        "sources": [
            {
                "path": path,
                "text": rebuild_source(chunks),
                "line_count": max(chunk.end_line for chunk in chunks),
            }
            for path, chunks in chunks_by_path.items()
        ],
        "prepared": {
            "files": [serialize_file_index(item, root) for _, item in sorted(prepared.repo_index.files.items())],
            "token_vocabulary": vocabulary,
            "content_term_frequencies": serialize_term_frequencies(
                prepared.content_term_frequencies, token_ids
            ),
            "metadata_term_frequencies": serialize_term_frequencies(
                prepared.metadata_term_frequencies, token_ids
            ),
            "path_token_sets": [[token_ids[token] for token in sorted(tokens)] for tokens in prepared.path_token_sets],
        },
    }
    return {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "checksum": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "payload": payload,
    }


def serialize_file_index(item: FileIndex, root: Path) -> dict:
    """Serialize one AST-derived file index using only relative paths and JSON values."""
    return {
        "path": item.path.resolve().relative_to(root).as_posix(),
        "module": item.module,
        "directories": list(item.directories),
        "symbols": [symbol.__dict__ for symbol in item.symbols],
        "imports": list(item.imports),
        "calls": list(item.calls),
    }


def serialize_term_frequencies(
    matrix: tuple[tuple[tuple[str, int], ...], ...], token_ids: dict[str, int]
) -> list[list[list[int]]]:
    """Encode each BM25 document once per distinct token instead of per occurrence."""
    return [
        sorted([[token_ids[token], count] for token, count in row])
        for row in matrix
    ]


def snapshot_from_envelope(envelope: object, key: RepositoryIndexKey) -> PreparedRepositoryIndex:
    """Strictly restore safe JSON data and rebuild only BM25 corpus statistics."""
    if not isinstance(envelope, dict) or set(envelope) != {
        "format", "format_version", "checksum", "payload"
    }:
        raise ValueError("cache envelope must be an object")
    if envelope.get("format") != CACHE_FORMAT or envelope.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("cache format is incompatible")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or set(payload) != {
        "repository_id", "config", "manifest", "sources", "prepared"
    }:
        raise ValueError("cache payload must be an object")
    if envelope.get("checksum") != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise ValueError("cache checksum mismatch")
    if payload.get("repository_id") != repository_identity(key) or payload.get("config") != cache_config(key):
        raise ValueError("cache identity or config mismatch")
    manifest = parse_manifest(payload.get("manifest"))
    root = Path(key.repo_path).resolve()
    chunks = parse_sources(
        payload.get("sources"), root, {entry.path for entry in manifest}, key.chunk_size, key.overlap
    )
    prepared = parse_prepared(payload.get("prepared"), root, chunks)
    retriever = BM25Retriever(
        list(chunks), graph_weight=key.graph_weight, path_weight=key.path_weight,
        prepared_data=prepared, path_root=root,
    )
    return PreparedRepositoryIndex(key, manifest, chunks, retriever, len(manifest), len(chunks))


def safe_relative_path(value: object) -> str:
    """Validate and normalize one non-empty repository-relative cache path."""
    if not isinstance(value, str):
        raise ValueError("cache path must be a string")
    text = value.replace("\\", "/").strip()
    posix = PurePosixPath(text)
    windows = PureWindowsPath(value)
    if (
        not text or text == "." or posix.is_absolute() or ".." in posix.parts
        or bool(windows.drive) or bool(windows.root)
    ):
        raise ValueError("cache contains unsafe path")
    return posix.as_posix()


def confined_path(root: Path, relative: str, *, root_is_resolved: bool = False) -> Path:
    """Resolve a cached relative path and prove it remains beneath the repository root."""
    if not root_is_resolved:
        root = root.resolve()
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("cache path escapes repository") from error
    return candidate


def parse_manifest(value: object) -> tuple[ManifestEntry, ...]:
    """Validate cached manifest structure, ordering, and unique relative paths."""
    if not isinstance(value, list):
        raise ValueError("cache manifest must be an array")
    entries: list[ManifestEntry] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size", "mtime_ns"}:
            raise ValueError("cache manifest entry is invalid")
        path = safe_relative_path(item["path"])
        size, mtime_ns = item["size"], item["mtime_ns"]
        if type(size) is not int or size < 0 or type(mtime_ns) is not int or mtime_ns < 0:
            raise ValueError("cache manifest metadata is invalid")
        entries.append(ManifestEntry(path, size, mtime_ns))
    if entries != sorted(entries) or len({entry.path for entry in entries}) != len(entries):
        raise ValueError("cache manifest ordering or identity is invalid")
    return tuple(entries)


def parse_sources(
    value: object,
    root: Path,
    manifest_paths: set[str],
    chunk_size: int,
    overlap: int,
) -> tuple[CodeChunk, ...]:
    """Validate deduplicated cached sources and deterministically recreate chunks."""
    if not isinstance(value, list) or not value:
        raise ValueError("cache sources must be a non-empty array")
    root = root.resolve()
    chunks: list[CodeChunk] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "text", "line_count"}:
            raise ValueError("cache source is invalid")
        relative = safe_relative_path(item["path"])
        text = item["text"]
        line_count = item["line_count"]
        if (
            relative not in manifest_paths or relative in seen or not isinstance(text, str)
            or not isinstance(line_count, int) or isinstance(line_count, bool) or line_count <= 0
        ):
            raise ValueError("cache source identity is invalid")
        seen.add(relative)
        path = confined_path(root, relative, root_is_resolved=True)
        missing_blank_lines = line_count - len(text.splitlines())
        if missing_blank_lines < 0:
            raise ValueError("cache source line count is invalid")
        restored_text = text + ("\n" * missing_blank_lines)
        chunks.extend(split_python_files(
            [PythonFile(path=path, text=restored_text)], chunk_size=chunk_size, overlap=overlap
        ))
    if not chunks:
        raise ValueError("cache sources produced no chunks")
    return tuple(chunks)


def parse_prepared(value: object, root: Path, chunks: tuple[CodeChunk, ...]) -> PreparedBM25Data:
    """Validate AST and token-derived JSON structures aligned to cached chunks."""
    if not isinstance(value, dict) or set(value) != {
        "files", "token_vocabulary", "content_term_frequencies",
        "metadata_term_frequencies", "path_token_sets"
    }:
        raise ValueError("cache prepared data is invalid")
    root = root.resolve()
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("cache file indexes must be an array")
    repo_files: dict[str, FileIndex] = {}
    for item in files:
        file_index = parse_file_index(item, root, root_is_resolved=True)
        identity = normalize_path(file_index.path)
        if identity in repo_files:
            raise ValueError("cache file index is duplicated")
        repo_files[identity] = file_index
    chunk_file_ids = {normalize_path(chunk.file_path) for chunk in chunks}
    if set(repo_files) != chunk_file_ids:
        raise ValueError("cache file indexes do not match cached chunks")
    vocabulary = value["token_vocabulary"]
    if (
        not isinstance(vocabulary, list)
        or not all(isinstance(token, str) for token in vocabulary)
        or vocabulary != sorted(set(vocabulary))
    ):
        raise ValueError("cache token vocabulary is invalid")
    tokenized_chunks = parse_term_frequency_matrix(
        value["content_term_frequencies"], len(chunks), vocabulary
    )
    tokenized_metadata = parse_term_frequency_matrix(
        value["metadata_term_frequencies"], len(chunks), vocabulary
    )
    path_sets = tuple(
        frozenset(tokens)
        for tokens in parse_token_id_matrix(value["path_token_sets"], len(chunks), vocabulary)
    )
    return PreparedBM25Data(RepoIndex(repo_files), tokenized_chunks, tokenized_metadata, path_sets)


def parse_term_frequency_matrix(
    value: object,
    expected: int,
    vocabulary: list[str],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Validate sparse token/count rows without expanding token occurrences."""
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("cache term-frequency matrix length is invalid")
    matrix: list[tuple[tuple[str, int], ...]] = []
    vocabulary_size = len(vocabulary)
    for row in value:
        if not isinstance(row, list):
            raise ValueError("cache term-frequency row is invalid")
        terms: list[tuple[str, int]] = []
        previous_id = -1
        append_term = terms.append
        for pair in row:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("cache term-frequency entry is invalid")
            token_id, count = pair
            if (
                type(token_id) is not int or type(count) is not int
                or token_id <= previous_id or not 0 <= token_id < vocabulary_size
                or count <= 0
            ):
                raise ValueError("cache term-frequency entry is invalid")
            previous_id = token_id
            append_term((vocabulary[token_id], count))
        matrix.append(tuple(terms))
    return tuple(matrix)


def parse_token_id_matrix(
    value: object,
    expected: int,
    vocabulary: list[str],
) -> tuple[tuple[str, ...], ...]:
    """Validate compact token-id arrays and expand them through one shared vocabulary."""
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("cache token matrix length is invalid")
    matrix: list[tuple[str, ...]] = []
    for row in value:
        if not isinstance(row, list) or not all(
            type(token_id) is int and 0 <= token_id < len(vocabulary) for token_id in row
        ):
            raise ValueError("cache token matrix row is invalid")
        matrix.append(tuple(vocabulary[token_id] for token_id in row))
    return tuple(matrix)


def parse_file_index(value: object, root: Path, *, root_is_resolved: bool = False) -> FileIndex:
    """Validate and reconstruct one cached AST-derived file index."""
    required = {"path", "module", "directories", "symbols", "imports", "calls"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("cache file index is invalid")
    relative = safe_relative_path(value["path"])
    if not isinstance(value["module"], str):
        raise ValueError("cache module is invalid")
    sequences = (value["directories"], value["imports"], value["calls"])
    if not all(isinstance(items, list) and all(isinstance(item, str) for item in items) for items in sequences):
        raise ValueError("cache file index sequences are invalid")
    symbols: list[SymbolRecord] = []
    if not isinstance(value["symbols"], list):
        raise ValueError("cache symbols must be an array")
    for item in value["symbols"]:
        if not isinstance(item, dict) or set(item) != {
            "name", "kind", "start_line", "end_line", "signature", "parent"
        }:
            raise ValueError("cache symbol is invalid")
        if not all(isinstance(item[field], str) for field in ("name", "kind", "signature")):
            raise ValueError("cache symbol strings are invalid")
        if type(item["start_line"]) is not int or type(item["end_line"]) is not int:
            raise ValueError("cache symbol lines are invalid")
        if item["parent"] is not None and not isinstance(item["parent"], str):
            raise ValueError("cache symbol parent is invalid")
        symbols.append(SymbolRecord(**item))
    return FileIndex(
        path=confined_path(root, relative, root_is_resolved=root_is_resolved), module=value["module"],
        directories=tuple(value["directories"]), symbols=tuple(symbols),
        imports=tuple(value["imports"]), calls=tuple(value["calls"]),
    )


def changed_file_count(
    previous: tuple[ManifestEntry, ...],
    current: tuple[ManifestEntry, ...],
) -> int:
    """Count added, removed, or metadata-changed manifest paths."""
    before = {entry.path: (entry.size, entry.mtime_ns) for entry in previous}
    after = {entry.path: (entry.size, entry.mtime_ns) for entry in current}
    return sum(1 for path in before.keys() | after.keys() if before.get(path) != after.get(path))


GLOBAL_REPOSITORY_INDEX_REGISTRY = RepositoryIndexRegistry()


def clear_repository_index_registry() -> None:
    """Clear the process-wide registry through an explicit test/lifecycle hook."""
    GLOBAL_REPOSITORY_INDEX_REGISTRY.clear()
