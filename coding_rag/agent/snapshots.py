"""Repository-external snapshots for approved Agent write operations."""

from __future__ import annotations

import difflib
import hashlib
import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from coding_rag.tools.agent_readonly import AgentToolError


SNAPSHOT_ID_LENGTH = 24


class SnapshotStore:
    """Cache declared file versions before a write and restore them on demand."""

    def __init__(
        self,
        repo_path: str | Path,
        snapshot_root: str | Path | None = None,
    ) -> None:
        """Bind snapshots to one repository and an external cache root."""
        self.repo_path = Path(repo_path).resolve()
        base = Path(snapshot_root).resolve() if snapshot_root else Path.home() / ".repopilot" / "rollback"
        repo_key = hashlib.sha256(str(self.repo_path).encode("utf-8")).hexdigest()[:16]
        self.root = base / repo_key

    def create(self, paths: Iterable[str], *, tool: str) -> dict[str, Any]:
        """Snapshot each declared repository-relative file before an approved write."""
        normalized = normalize_snapshot_paths(paths)
        if not normalized:
            raise AgentToolError("write tools must declare at least one affected file")

        snapshot_id = secrets.token_hex(SNAPSHOT_ID_LENGTH // 2)
        snapshot_dir = self.root / snapshot_id
        files_dir = snapshot_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=False)
        entries: list[dict[str, Any]] = []
        try:
            for index, relative_path in enumerate(normalized):
                source = resolve_repo_file(self.repo_path, relative_path)
                existed = source.exists()
                if existed and not source.is_file():
                    raise AgentToolError(f"snapshot target is not a file: {relative_path}")
                backup_name = f"{index:04d}.bin" if existed else None
                if backup_name:
                    shutil.copy2(source, files_dir / backup_name)
                entries.append(
                    {
                        "path": relative_path,
                        "existed": existed,
                        "backup": backup_name,
                    }
                )
            manifest = {
                "snapshot_id": snapshot_id,
                "repo_path": str(self.repo_path),
                "tool": tool,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": entries,
            }
            (snapshot_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return manifest
        except Exception:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise

    def diff(self, snapshot_id: str) -> dict[str, Any]:
        """Build an operation-specific unified diff against one snapshot."""
        manifest, snapshot_dir = self._load(snapshot_id)
        chunks: list[str] = []
        changed_files: list[str] = []
        for entry in manifest["files"]:
            relative_path = str(entry["path"])
            current = resolve_repo_file(self.repo_path, relative_path)
            before = (
                (snapshot_dir / "files" / str(entry["backup"])).read_bytes()
                if entry["existed"]
                else None
            )
            after = current.read_bytes() if current.exists() and current.is_file() else None
            if before == after:
                continue
            changed_files.append(relative_path)
            chunks.append(render_file_diff(relative_path, before, after))
        return {
            "snapshot_id": snapshot_id,
            "files": changed_files,
            "text": "".join(chunks),
        }

    def rollback(self, snapshot_id: str) -> dict[str, Any]:
        """Restore all cached versions and remove files created by the approved write."""
        manifest, snapshot_dir = self._load(snapshot_id)
        restored: list[str] = []
        removed: list[str] = []
        for entry in manifest["files"]:
            relative_path = str(entry["path"])
            target = resolve_repo_file(self.repo_path, relative_path)
            if entry["existed"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot_dir / "files" / str(entry["backup"]), target)
                restored.append(relative_path)
            elif target.exists():
                if not target.is_file():
                    raise AgentToolError(f"rollback target is not a file: {relative_path}")
                target.unlink()
                removed.append(relative_path)
        return {
            "snapshot_id": snapshot_id,
            "restored_files": restored,
            "removed_files": removed,
            "remaining_diff": self.diff(snapshot_id),
        }

    def _load(self, snapshot_id: str) -> tuple[dict[str, Any], Path]:
        """Load and validate a snapshot manifest bound to this repository."""
        if len(snapshot_id) != SNAPSHOT_ID_LENGTH or any(ch not in "0123456789abcdef" for ch in snapshot_id):
            raise AgentToolError("invalid snapshot id")
        snapshot_dir = self.root / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.is_file():
            raise AgentToolError(f"snapshot not found: {snapshot_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("repo_path") != str(self.repo_path):
            raise AgentToolError("snapshot belongs to a different repository")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise AgentToolError("snapshot manifest has no files")
        return manifest, snapshot_dir


def normalize_snapshot_paths(paths: Iterable[str]) -> list[str]:
    """Normalize and deduplicate repository-relative file paths."""
    normalized: list[str] = []
    for raw_path in paths:
        path = str(raw_path or "").strip().replace("\\", "/")
        parts = PurePosixPath(path).parts
        if (
            not path
            or path.startswith("/")
            or (len(path) >= 2 and path[1] == ":")
            or ".." in parts
            or PurePosixPath(path).is_absolute()
        ):
            raise AgentToolError(f"unsafe affected file path: {raw_path}")
        clean = str(PurePosixPath(*parts))
        if clean not in normalized:
            normalized.append(clean)
    return normalized


def resolve_repo_file(repo_path: Path, relative_path: str) -> Path:
    """Resolve one declared file and keep it inside the bound repository."""
    target = (repo_path / relative_path).resolve()
    try:
        target.relative_to(repo_path)
    except ValueError as error:
        raise AgentToolError(f"affected file escapes repository: {relative_path}") from error
    return target


def render_file_diff(path: str, before: bytes | None, after: bytes | None) -> str:
    """Render a text diff, or a stable binary-change marker when decoding fails."""
    try:
        before_text = "" if before is None else before.decode("utf-8")
        after_text = "" if after is None else after.decode("utf-8")
    except UnicodeDecodeError:
        return f"Binary files a/{path} and b/{path} differ\n"
    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="/dev/null" if before is None else f"a/{path}",
            tofile="/dev/null" if after is None else f"b/{path}",
        )
    )
