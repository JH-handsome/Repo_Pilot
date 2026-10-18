"""Offline, isolated patch outcome evaluation; this is not an LLM intelligence benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import mean
from time import perf_counter, sleep
from typing import Protocol

from coding_rag.agent.safety import AgentSafetyGuard, AgentSafetyPolicy

from .gate import GateResult


EVALUATOR_VERSION = "patch-outcome-v1"
SCHEMA_VERSION = 1
MAX_TASKS = 100
MAX_PATCH_CHARS = 200_000
MAX_OUTPUT_BYTES = 12_000
MAX_TIMEOUT_SECONDS = 60
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_FILES = 20_000
TASK_FIELDS = frozenset({
    "id", "requirement", "allowed_paths", "forbidden_paths", "test_command",
    "timeout_seconds", "fault_patch", "fault_patch_sha256", "oracle_patch",
    "oracle_patch_sha256", "source_path", "source_sha256", "test_path",
    "test_sha256",
})
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SAFE_TEST_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class PatchOutcomeValidationError(ValueError):
    """Report all actionable task-manifest validation errors."""

    def __init__(self, errors: list[str]):
        """Preserve individual findings as well as a concise combined message."""
        super().__init__("; ".join(errors))
        self.errors = tuple(errors)


@dataclass(frozen=True)
class PatchTask:
    """A real pinned-source regression whose oracle is excluded from candidate input."""

    id: str
    requirement: str
    allowed_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    test_command: tuple[str, ...]
    timeout_seconds: int
    fault_patch_path: Path
    fault_patch_sha256: str
    oracle_patch_path: Path
    oracle_patch_sha256: str
    source_path: str
    source_sha256: str
    test_path: str
    test_sha256: str


@dataclass(frozen=True)
class PatchRepository:
    """One locally prepared, immutable upstream checkout identity."""

    id: str
    relative_path: str
    root: Path
    remote: str
    commit: str
    license: str
    license_url: str
    archive_roots: tuple[str, ...]


@dataclass(frozen=True)
class PatchSuite:
    """Portable task suite plus stable manifest and fixture identities."""

    suite_id: str
    evaluation_mode: str
    manifest_name: str
    manifest_sha256: str
    repository: PatchRepository
    tasks: tuple[PatchTask, ...]


@dataclass(frozen=True)
class CandidateRequest:
    """The complete provider-facing request; deliberately excludes gold patch data."""

    task_id: str
    requirement: str
    allowed_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    test_command: tuple[str, ...]


class CandidateProvider(Protocol):
    """Return one unified diff using only public task input and the isolated checkout."""

    name: str

    def provide(self, request: CandidateRequest, repo_path: Path) -> str:
        """Produce a candidate unified diff without receiving oracle fields."""
        ...


class StaticPatchProvider:
    """Offline provider backed by caller-supplied task-id to patch mappings."""

    name = "static_patch_v1"

    def __init__(self, patches: dict[str, str]):
        """Copy caller-owned patch mappings so later mutation cannot affect a run."""
        self.patches = dict(patches)

    def provide(self, request: CandidateRequest, repo_path: Path) -> str:
        """Return the mapped patch or an explicit no-op candidate."""
        return self.patches.get(request.task_id, "")


class OraclePatchProvider:
    """Self-test-only provider; gold paths stay private to this explicit adapter."""

    name = "oracle_self_check_v1"

    def __init__(self, tasks: tuple[PatchTask, ...]):
        """Bind private oracle paths for explicit evaluator self-checks only."""
        self._paths = {task.id: task.oracle_patch_path for task in tasks}

    def provide(self, request: CandidateRequest, repo_path: Path) -> str:
        """Read the oracle patch without exposing its path through CandidateRequest."""
        return self._paths[request.task_id].read_text(encoding="utf-8")


def safe_relative_path(value: object) -> str | None:
    """Accept only actual, nonempty, confined relative POSIX paths."""
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\\", "/")
    pure = PurePosixPath(text)
    if (
        not text or text == "." or pure.is_absolute() or ".." in pure.parts
        or re.match(r"^[A-Za-z]:", text) or Path(text).is_absolute()
    ):
        return None
    return pure.as_posix()


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_identity(root: Path) -> str:
    """Hash relative paths and bytes, rejecting symlinks and special files."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"fixture contains symlink: {path.relative_to(root).as_posix()}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _confined_file(workspace: Path, relative: str) -> Path | None:
    """Resolve a validated relative path and retain workspace confinement."""
    path = (workspace / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        path.relative_to(workspace)
    except ValueError:
        return None
    return path


def _git_text(root: Path, *args: str) -> str | None:
    """Run a bounded read-only Git query and return stripped text on success."""
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=15, shell=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_blob(root: Path, commit: str, relative: str) -> bytes | None:
    """Read one pinned Git blob without consulting working-tree contents."""
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        capture_output=True, timeout=15, shell=False,
    )
    return result.stdout if result.returncode == 0 else None


def _normalized_remote(value: str) -> str:
    """Normalize common HTTPS and SSH Git remote spellings for comparison."""
    text = value.strip().replace("\\", "/").casefold().rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@") and ":" in text:
        host, path = text[4:].split(":", 1)
        return f"{host}/{path}"
    if "://" in text:
        text = text.split("://", 1)[1]
        if "@" in text.split("/", 1)[0]:
            text = text.split("@", 1)[1]
    return text


def load_and_validate_patch_suite(workspace_root: str | Path, manifest_path: str | Path) -> PatchSuite:
    """Strictly validate portable schema, identities, paths, tests, and oracle files."""
    workspace = Path(workspace_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "suite_id", "evaluation_mode", "repository", "tasks",
    }:
        raise PatchOutcomeValidationError([
            "manifest must contain exactly schema_version, suite_id, evaluation_mode, repository, tasks"
        ])
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    suite_id = payload.get("suite_id")
    mode = payload.get("evaluation_mode")
    if not isinstance(suite_id, str) or SAFE_ID.fullmatch(suite_id) is None:
        errors.append("suite_id must be a safe lowercase identifier")
        suite_id = "invalid"
    if mode != "patch_outcome_harness_not_llm_intelligence":
        errors.append("evaluation_mode must explicitly label this as a patch outcome harness")
    repository_payload = payload.get("repository")
    repository_fields = {
        "id", "path", "remote", "commit", "license", "license_url", "archive_roots",
    }
    if not isinstance(repository_payload, dict) or set(repository_payload) != repository_fields:
        raise PatchOutcomeValidationError(errors + [
            "repository must contain exactly id, path, remote, commit, license, license_url, archive_roots"
        ])
    repo_id = repository_payload["id"]
    repo_relative = safe_relative_path(repository_payload["path"])
    remote = repository_payload["remote"]
    commit = repository_payload["commit"]
    license_name = repository_payload["license"]
    license_url = repository_payload["license_url"]
    archive_roots_raw = repository_payload["archive_roots"]
    if not isinstance(repo_id, str) or SAFE_ID.fullmatch(repo_id) is None:
        errors.append("repository id must be a safe lowercase identifier")
    if repo_relative is None:
        errors.append("repository path must be a confined relative path")
        repo_root = workspace
    else:
        repo_root = _confined_file(workspace, repo_relative) or workspace
    if not repo_root.is_dir():
        errors.append("repository checkout is missing")
    if not isinstance(remote, str) or not remote.strip():
        errors.append("repository remote is required")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        errors.append("repository commit must be a full lowercase SHA")
    if not isinstance(license_name, str) or not license_name.strip() or not isinstance(license_url, str) or not license_url.strip():
        errors.append("repository license metadata is required")
    archive_roots = _path_list(archive_roots_raw, "repository", "archive_roots", errors)
    if tuple(archive_roots) != ("django", "tests"):
        errors.append("repository archive_roots must be exactly ['django', 'tests']")
    if repo_root.is_dir() and isinstance(commit, str):
        head = _git_text(repo_root, "rev-parse", "HEAD")
        actual_remote = _git_text(repo_root, "remote", "get-url", "origin")
        if head != commit:
            errors.append(f"repository HEAD mismatch: expected {commit}, found {head}")
        if actual_remote is None or _normalized_remote(actual_remote) != _normalized_remote(str(remote)):
            errors.append("repository origin remote mismatch")
    repository = PatchRepository(
        str(repo_id), str(repo_relative), repo_root, _normalized_remote(str(remote)),
        str(commit), str(license_name), str(license_url), tuple(archive_roots),
    )
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= MAX_TASKS:
        raise PatchOutcomeValidationError(errors + [f"tasks must contain 1..{MAX_TASKS} items"])
    tasks: list[PatchTask] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_tasks, 1):
        label = f"task {index}"
        if not isinstance(item, dict) or set(item) != TASK_FIELDS:
            errors.append(f"{label} must contain exactly: {', '.join(sorted(TASK_FIELDS))}")
            continue
        task_id = item["id"]
        if not isinstance(task_id, str) or SAFE_ID.fullmatch(task_id) is None or task_id in seen:
            errors.append(f"{label} has invalid or duplicate id")
            continue
        seen.add(task_id)
        fault_relative = safe_relative_path(item["fault_patch"])
        oracle_relative = safe_relative_path(item["oracle_patch"])
        source_relative = safe_relative_path(item["source_path"])
        test_relative = safe_relative_path(item["test_path"])
        requirement = item["requirement"]
        timeout = item["timeout_seconds"]
        if None in (fault_relative, oracle_relative, source_relative, test_relative):
            errors.append(f"{label} patch/source/test paths must be confined relative paths")
            continue
        fault = _confined_file(workspace, fault_relative)
        oracle = _confined_file(workspace, oracle_relative)
        if fault is None or not fault.is_file() or oracle is None or not oracle.is_file():
            errors.append(f"{label} fault or oracle patch is missing")
            continue
        if not isinstance(requirement, str) or not requirement.strip() or len(requirement) > 2000:
            errors.append(f"{label} requirement must contain 1..2000 characters")
        allowed = _path_list(item["allowed_paths"], label, "allowed_paths", errors)
        forbidden = _path_list(item["forbidden_paths"], label, "forbidden_paths", errors)
        if set(allowed) & set(forbidden):
            errors.append(f"{label} allowed_paths and forbidden_paths overlap")
        for relative in (*allowed, *forbidden, source_relative, test_relative):
            if not any(relative == root or relative.startswith(root + "/") for root in archive_roots):
                errors.append(f"{label} path is outside trusted archive roots: {relative}")
        if source_relative not in allowed:
            errors.append(f"{label} source_path must be an allowed path")
        for field_name, path, expected in (
            ("fault_patch_sha256", fault, item["fault_patch_sha256"]),
            ("oracle_patch_sha256", oracle, item["oracle_patch_sha256"]),
        ):
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or file_sha256(path) != expected:
                errors.append(f"{label} {field_name} mismatch")
        for field_name, relative, expected in (
            ("source_sha256", source_relative, item["source_sha256"]),
            ("test_sha256", test_relative, item["test_sha256"]),
        ):
            blob = _git_blob(repo_root, str(commit), relative)
            actual = hashlib.sha256(blob).hexdigest() if blob is not None else None
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or actual != expected:
                errors.append(f"{label} {field_name} mismatch")
        command = item["test_command"]
        if not _valid_test_command(command):
            errors.append(f"{label} test_command must be ['python','-m','unittest', safe dotted targets...]")
            command = []
        if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            errors.append(f"{label} timeout_seconds must be 1..{MAX_TIMEOUT_SECONDS}")
            timeout = 1
        if fault.stat().st_size > MAX_PATCH_CHARS or oracle.stat().st_size > MAX_PATCH_CHARS:
            errors.append(f"{label} patch exceeds maximum patch size")
        guard = AgentSafetyGuard(AgentSafetyPolicy(max_patch_chars=MAX_PATCH_CHARS))
        for kind, patch_path in (("fault", fault), ("oracle", oracle)):
            decision = guard.check_patch(patch_path.read_text(encoding="utf-8"))
            if not decision.allowed or set(decision.metadata.get("changed_files") or []) - set(allowed):
                errors.append(f"{label} {kind} patch is unsafe or changes undeclared paths")
        tasks.append(PatchTask(
            task_id, requirement.strip(), tuple(allowed), tuple(forbidden), tuple(command), timeout,
            fault, str(item["fault_patch_sha256"]), oracle, str(item["oracle_patch_sha256"]),
            source_relative, str(item["source_sha256"]), test_relative, str(item["test_sha256"]),
        ))
    if errors:
        raise PatchOutcomeValidationError(errors)
    return PatchSuite(
        str(suite_id), str(mode), manifest_file.name, file_sha256(manifest_file), repository,
        tuple(tasks),
    )


def _path_list(value: object, label: str, field: str, errors: list[str]) -> list[str]:
    """Validate, normalize, and deduplicate one required relative-path list."""
    if not isinstance(value, list) or not value:
        errors.append(f"{label} {field} must be a non-empty list")
        return []
    output: list[str] = []
    for raw in value:
        relative = safe_relative_path(raw)
        if relative is None or relative in output:
            errors.append(f"{label} {field} contains unsafe or duplicate path: {raw!r}")
        else:
            output.append(relative)
    return output


def _valid_test_command(value: object) -> bool:
    """Accept only fixed unittest argv with safe dotted test targets."""
    return (
        isinstance(value, list) and len(value) >= 4
        and value[:3] == ["python", "-m", "unittest"]
        and all(isinstance(item, str) and SAFE_TEST_TARGET.fullmatch(item) for item in value[3:])
    )


def _tree_files(root: Path) -> dict[str, str]:
    """Snapshot regular-file content hashes outside evaluator-owned Git metadata."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"repository contains symlink: {path.relative_to(root).as_posix()}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    return files


def _tree_metadata(root: Path) -> dict[str, tuple[int, int]]:
    """Best-effort trusted-provider side-effect detection without full-tree rehashing."""
    files: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"repository contains symlink: {path.relative_to(root).as_posix()}")
        if path.is_file():
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def _run(command: list[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None,
         stdin: str | None = None, output_limit: int = MAX_OUTPUT_BYTES) -> dict:
    """Run argv without a shell while continuously draining output into a hard cap."""
    if output_limit < 1 or output_limit > MAX_ARCHIVE_BYTES:
        raise ValueError("subprocess output_limit is outside the safe range")
    started = perf_counter()
    process = subprocess.Popen(
        command, cwd=str(cwd), stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, env=env,
    )
    captured = bytearray()
    output_bytes = 0

    def drain() -> None:
        """Continuously drain the pipe while retaining at most the configured bytes."""
        nonlocal output_bytes
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                return
            output_bytes += len(chunk)
            remaining = output_limit + 1 - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        if stdin is not None:
            assert process.stdin is not None
            process.stdin.write(stdin.encode("utf-8"))
            process.stdin.close()
        returncode = process.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        process.kill()
        process.wait(timeout=5)
    finally:
        reader.join(timeout=5)
        if reader.is_alive():
            process.kill()
            raise RuntimeError("subprocess output reader did not terminate")
        if process.stdout is not None:
            process.stdout.close()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    output = bytes(captured[:output_limit]).decode("utf-8", errors="replace")
    original_bytes = output_bytes
    truncated = output_bytes > output_limit
    if truncated:
        output += f"\n...[truncated {original_bytes - output_limit} bytes]"
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "output": output,
        "output_bytes": original_bytes,
        "output_truncated": truncated,
    }


def _test_environment(guard_root: Path, repo_root: Path) -> dict[str, str]:
    """Force local-only proxy settings and load an evaluator-owned socket guard."""
    guard_root.mkdir()
    asgiref = guard_root / "asgiref"
    asgiref.mkdir()
    (asgiref / "__init__.py").write_text("\n", encoding="utf-8")
    (asgiref / "local.py").write_text(
        "import threading\nclass Local(threading.local):\n    pass\n", encoding="utf-8",
    )
    (asgiref / "sync.py").write_text(
        "import inspect\n"
        "iscoroutinefunction = inspect.iscoroutinefunction\n"
        "def markcoroutinefunction(function): return function\n"
        "def sync_to_async(function=None, *args, **kwargs): return function\n"
        "def async_to_sync(function): return function\n",
        encoding="utf-8",
    )
    (guard_root / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*args, **kwargs): raise OSError('network disabled by patch outcome harness')\n"
        "socket.create_connection = _blocked\n"
        "socket.socket.connect = _blocked\n"
        "socket.socket.connect_ex = _blocked\n",
        encoding="utf-8",
    )
    allowed_environment = {
        "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME",
        "USERPROFILE", "LOCALAPPDATA", "LANG", "LC_ALL", "TZ",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed_environment}
    env.update({
        "PYTHONPATH": os.pathsep.join((str(guard_root), str(repo_root), str(repo_root / "tests"))),
        "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
        "REPOPILOT_EVAL_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
    })
    return env


def _ensure_patch_targets_not_symlinks(repo: Path, paths: list[str]) -> None:
    """Reject a patch target if any existing component is a symbolic link."""
    for relative in paths:
        current = repo
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"patch target traverses symlink: {relative}")


def validate_candidate_patch_paths(patch: str, allowed_paths: tuple[str, ...]) -> tuple[list[str], str | None]:
    """Reject every path/header form except consistent in-place edits to allowed files."""
    if not isinstance(patch, str) or not patch.strip():
        return [], "candidate patch is empty"
    paths: list[str] = []
    current: str | None = None
    saw_old = False
    saw_new = False
    forbidden_headers = (
        "new file mode ", "deleted file mode ", "rename from ", "rename to ",
        "copy from ", "copy to ", "old mode ", "new mode ", "GIT binary patch",
    )
    for line in patch.splitlines():
        if line.startswith(forbidden_headers):
            return [], "candidate patch uses forbidden create/delete/rename/copy/mode metadata"
        if line.startswith("diff --git "):
            if current is not None and not (saw_old and saw_new):
                return [], "candidate patch block is missing consistent ---/+++ headers"
            match = re.fullmatch(r"diff --git a/([^\s]+) b/([^\s]+)", line)
            if match is None or match.group(1) != match.group(2):
                return [], "candidate patch diff paths must be identical simple relative paths"
            relative = safe_relative_path(match.group(1))
            if relative is None or relative not in allowed_paths:
                return [], f"candidate patch path is not allowed: {match.group(1)}"
            current = relative
            saw_old = saw_new = False
            if relative not in paths:
                paths.append(relative)
        elif line.startswith("--- "):
            if current is None or line != f"--- a/{current}":
                return [], "candidate patch --- path does not match diff header"
            saw_old = True
        elif line.startswith("+++ "):
            if current is None or line != f"+++ b/{current}":
                return [], "candidate patch +++ path does not match diff header"
            saw_new = True
    if current is None or not (saw_old and saw_new):
        return [], "candidate patch has no complete file block"
    return paths, None


def _invalid_harness_row(task: PatchTask, pristine: dict, initial: dict, reason: str) -> dict:
    """Return a score-zero row without invoking a candidate on an invalid harness state."""
    return {
        "id": task.id, "pristine_pass": pristine.get("returncode") == 0,
        "initial_failure_proven": False, "patch_applied": False, "tests_pass": False,
        "allowed_path_precision": 0.0, "forbidden_untouched": True,
        "minimality_score": 0.0, "changed_file_count": 0, "changed_files": [],
        "timeout": bool(pristine.get("timed_out") or initial.get("timed_out")),
        "pristine_test_latency_ms": pristine.get("latency_ms", 0.0),
        "initial_test_latency_ms": initial.get("latency_ms", 0.0),
        "test_latency_ms": 0.0, "apply_latency_ms": 0.0,
        "output_truncated": bool(pristine.get("output_truncated") or initial.get("output_truncated")),
        "provider_error": None, "rejection": reason, "harness_valid": False, "score": 0.0,
    }


def materialize_pinned_archive(repository: PatchRepository, destination: Path) -> None:
    """Safely materialize only trusted roots from the exact pinned Git commit."""
    with tempfile.TemporaryFile() as archive_file, tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            ["git", "-C", str(repository.root), "archive", "--format=tar",
             repository.commit, *repository.archive_roots],
            stdout=archive_file, stderr=error_file, shell=False,
        )
        started = perf_counter()
        while process.poll() is None:
            if os.fstat(archive_file.fileno()).st_size > MAX_ARCHIVE_BYTES:
                process.kill()
                process.wait(timeout=5)
                raise ValueError("repository archive exceeds maximum size")
            if perf_counter() - started > 30:
                process.kill()
                process.wait(timeout=5)
                raise TimeoutError("git archive timed out")
            sleep(0.02)
        if process.returncode != 0:
            raise RuntimeError("git archive failed for pinned repository")
        if os.fstat(archive_file.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("repository archive exceeds maximum size")
        archive_file.seek(0)
        destination.mkdir()
        _extract_validated_archive(archive_file, repository, destination)


def _extract_validated_archive(archive_file, repository: PatchRepository, destination: Path) -> None:
    """Validate all members before manually extracting regular files."""
    total_size = 0
    file_count = 0
    with tarfile.open(fileobj=archive_file, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError("repository archive exceeds member limit")
        for member in members:
            relative = safe_relative_path(member.name)
            if relative is None or not any(
                relative == root or relative.startswith(root + "/")
                for root in repository.archive_roots
            ):
                raise ValueError(f"unsafe repository archive member: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev() or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported repository archive member: {relative}")
            if member.isfile():
                file_count += 1
                total_size += member.size
                if member.size > MAX_ARCHIVE_FILE_BYTES or total_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("repository archive extracted size exceeds limit")
        if file_count > MAX_ARCHIVE_FILES:
            raise ValueError("repository archive exceeds file limit")
        for member in members:
            relative = safe_relative_path(member.name)
            assert relative is not None
            target = destination / Path(*PurePosixPath(relative).parts)
            resolved_parent = target.parent.resolve()
            resolved_parent.relative_to(destination.resolve())
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive file cannot be read: {relative}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if member.mode & 0o111:
                target.chmod(0o755)


def _apply_patch(repo: Path, patch: str) -> tuple[bool, str | None, float]:
    """Check and apply a validated patch through bounded, shell-free Git argv."""
    checked = _run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=repo, timeout=10, stdin=patch,
    )
    if checked["returncode"] != 0:
        return False, "git apply --check rejected patch", 0.0
    applied = _run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo, timeout=10, stdin=patch,
    )
    return applied["returncode"] == 0, (
        None if applied["returncode"] == 0 else "git apply rejected patch"
    ), applied["latency_ms"]


def evaluate_patch_task(
    repository: PatchRepository, task: PatchTask, provider: CandidateProvider,
) -> dict:
    """Evaluate one candidate only inside an evaluator-owned temporary checkout."""
    with tempfile.TemporaryDirectory(prefix="repopilot-patch-outcome-") as folder:
        temp_root = Path(folder)
        repo = temp_root / "repo"
        materialize_pinned_archive(repository, repo)
        env = _test_environment(temp_root / "network_guard", repo)
        command = [sys.executable, *task.test_command[1:]]
        pristine = _run(command, cwd=repo, timeout=task.timeout_seconds, env=env)
        pristine_pass = not pristine["timed_out"] and pristine["returncode"] == 0
        fault_patch = task.fault_patch_path.read_text(encoding="utf-8")
        fault_applied, fault_error, _ = _apply_patch(repo, fault_patch)
        initial = _run(command, cwd=repo, timeout=task.timeout_seconds, env=env)
        initial_failure_proven = (
            pristine_pass and fault_applied and not initial["timed_out"]
            and initial["returncode"] != 0
        )
        if not pristine_pass:
            return _invalid_harness_row(task, pristine, initial, "pristine upstream test did not pass")
        if not fault_applied or not initial_failure_proven:
            return _invalid_harness_row(
                task, pristine, initial, fault_error or "fault injection did not make the upstream test fail",
            )
        start_metadata = _tree_metadata(repo)
        start_allowed_hashes = {
            path: file_sha256(repo / Path(*PurePosixPath(path).parts))
            for path in task.allowed_paths
        }
        request = CandidateRequest(
            task.id, task.requirement, task.allowed_paths, task.forbidden_paths,
            task.test_command,
        )
        provider_error = None
        try:
            patch = provider.provide(request, repo)
        except Exception as error:
            patch = ""
            provider_error = type(error).__name__
        patch_applied = False
        rejection = None
        apply_latency_ms = 0.0
        provider_changed_tree = _tree_metadata(repo) != start_metadata
        if not isinstance(patch, str):
            rejection = "candidate patch must be a string"
        elif provider_changed_tree:
            rejection = "candidate provider mutated the checkout outside its returned patch"
        else:
            decision = AgentSafetyGuard(AgentSafetyPolicy(max_patch_chars=MAX_PATCH_CHARS)).check_patch(patch)
            if not decision.allowed:
                rejection = decision.reason
            else:
                paths, path_error = validate_candidate_patch_paths(patch, task.allowed_paths)
                if path_error is not None:
                    rejection = path_error
                    paths = []
                if decision.metadata.get("deleted_files"):
                    rejection = "candidate file deletion is forbidden"
                try:
                    if rejection is None:
                        _ensure_patch_targets_not_symlinks(repo, paths)
                except ValueError as error:
                    rejection = str(error)
                if rejection is None:
                    patch_applied, rejection, apply_latency_ms = _apply_patch(repo, patch)
        changed_files = []
        if patch_applied:
            for path, before_hash in start_allowed_hashes.items():
                target = repo / Path(*PurePosixPath(path).parts)
                if not target.is_file() or target.is_symlink() or file_sha256(target) != before_hash:
                    changed_files.append(path)
        tests = (
            _run(command, cwd=repo, timeout=task.timeout_seconds, env=env)
            if patch_applied and initial_failure_proven
            else {"returncode": None, "timed_out": False, "latency_ms": 0.0,
                  "output": "", "output_bytes": 0, "output_truncated": False}
        )
        tests_pass = tests["returncode"] == 0 and not tests["timed_out"]
        allowed_changed = [path for path in changed_files if path in task.allowed_paths]
        precision = len(allowed_changed) / len(changed_files) if changed_files else 0.0
        forbidden_untouched = not any(path in task.forbidden_paths for path in changed_files)
        extra_count = len([path for path in changed_files if path not in task.allowed_paths])
        minimality = 1.0 / (1.0 + extra_count) if changed_files else 0.0
        score = 100 * (
            0.10 * float(initial_failure_proven)
            + 0.15 * float(patch_applied)
            + 0.50 * float(tests_pass)
            + 0.10 * precision
            + 0.10 * float(forbidden_untouched)
            + 0.05 * minimality
        )
        if not tests_pass:
            score = min(score, 25.0)
        if not patch_applied:
            score = min(score, 10.0)
        if not forbidden_untouched:
            score = min(score, 49.0)
        return {
            "id": task.id,
            "harness_valid": True,
            "pristine_pass": pristine_pass,
            "initial_failure_proven": initial_failure_proven,
            "patch_applied": patch_applied,
            "tests_pass": tests_pass,
            "allowed_path_precision": precision,
            "forbidden_untouched": forbidden_untouched,
            "minimality_score": minimality,
            "changed_file_count": len(changed_files),
            "changed_files": changed_files,
            "timeout": bool(pristine["timed_out"] or initial["timed_out"] or tests["timed_out"]),
            "pristine_test_latency_ms": pristine["latency_ms"],
            "initial_test_latency_ms": initial["latency_ms"],
            "test_latency_ms": tests["latency_ms"],
            "apply_latency_ms": apply_latency_ms,
            "output_truncated": bool(
                pristine["output_truncated"] or initial["output_truncated"]
                or tests["output_truncated"]
            ),
            "provider_error": provider_error,
            "rejection": rejection or fault_error,
            "score": round(score, 6),
        }


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile or zero for no observations."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def evaluate_patch_suite(suite: PatchSuite, provider: CandidateProvider) -> dict:
    """Evaluate every task and return a portable quality scorecard."""
    rows = [evaluate_patch_task(suite.repository, task, provider) for task in suite.tasks]
    avg = lambda key: mean(float(row[key]) for row in rows)
    latencies = [row["test_latency_ms"] for row in rows]
    candidate = {"provider": provider.name}
    metadata = getattr(provider, "scorecard_metadata", None)
    if isinstance(metadata, dict):
        candidate["runtime"] = json_safe_candidate_metadata(metadata)
    return {
        **patch_scorecard_identity(suite),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "score": avg("score"),
        "metrics": {
            "case_count": len(rows),
            "harness_valid_rate": avg("harness_valid"),
            "pristine_pass_rate": avg("pristine_pass"),
            "initial_failure_rate": avg("initial_failure_proven"),
            "patch_applied_rate": avg("patch_applied"),
            "test_pass_rate": avg("tests_pass"),
            "allowed_path_precision": avg("allowed_path_precision"),
            "forbidden_untouched_rate": avg("forbidden_untouched"),
            "minimality_score": avg("minimality_score"),
            "timeout_rate": avg("timeout"),
            "changed_files_avg": avg("changed_file_count"),
            "test_latency_ms_p50": _percentile(latencies, 0.50),
            "test_latency_ms_p95": _percentile(latencies, 0.95),
        },
        "cases": rows,
    }


def patch_scorecard_identity(suite: PatchSuite) -> dict:
    """Build fields that must match before any candidate or paid evaluation starts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "patch_outcome_harness_not_llm_intelligence",
        "evaluator_version": EVALUATOR_VERSION,
        "dataset": {
            "name": suite.manifest_name, "sha256": suite.manifest_sha256,
            "suite_id": suite.suite_id, "case_count": len(suite.tasks),
        },
        "task_repositories": [
            {
                "id": task.id, "repository_id": suite.repository.id,
                "source_path": task.source_path, "source_sha256": task.source_sha256,
                "test_path": task.test_path, "test_sha256": task.test_sha256,
                "fault_patch_sha256": task.fault_patch_sha256,
            }
            for task in suite.tasks
        ],
        "repository": {
            "id": suite.repository.id, "path": suite.repository.relative_path,
            "remote": suite.repository.remote, "commit": suite.repository.commit,
            "license": suite.repository.license,
            "license_url": suite.repository.license_url,
            "archive_roots": list(suite.repository.archive_roots),
        },
        "config": {
            "max_patch_chars": MAX_PATCH_CHARS,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "shell": False,
            "execution_trust": "trusted_local_candidate_only",
            "isolation": "workspace_isolation_not_os_sandbox",
            "network": "python_socket_guard_and_proxy_defense_in_depth",
        },
    }


def validate_patch_gate_preflight(suite: PatchSuite, baseline: object, policy: object) -> None:
    """Reject malformed or incompatible gate inputs before candidate evaluation."""
    validate_patch_gate_policy(policy)
    if not isinstance(baseline, dict):
        raise ValueError("patch outcome baseline must be a JSON object")
    expected = patch_scorecard_identity(suite)
    fields = (
        "schema_version", "benchmark", "evaluator_version", "dataset",
        "repository", "task_repositories", "config",
    )
    failures = [field for field in fields if baseline.get(field) != expected.get(field)]
    if failures:
        raise ValueError("incompatible patch outcome baseline: " + ", ".join(failures))


def json_safe_candidate_metadata(metadata: dict) -> dict:
    """Keep only the fixed, non-sensitive Agent candidate metadata contract."""
    allowed = {
        "mode", "policy_version", "provider", "model", "task_count",
        "approval_count", "rejection_count", "incomplete_task_count", "llm_call_count",
        "llm_attempt_count", "llm_retry_count", "terminal_outcomes",
        "logical_call_upper_bound_per_task", "max_output_tokens_per_call",
        "output_token_upper_bound", "billing_measurement",
    }
    if set(metadata) != allowed:
        raise ValueError("candidate runtime metadata does not match the safe schema")
    labels = {key: metadata[key] for key in (
        "mode", "policy_version", "provider", "model", "billing_measurement",
    )}
    if not all(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,128}", value)
        for value in labels.values()
    ):
        raise ValueError("candidate runtime metadata contains an unsafe label")
    counters = {key: metadata[key] for key in (
        "task_count", "approval_count", "rejection_count", "incomplete_task_count", "llm_call_count",
        "llm_attempt_count", "llm_retry_count", "logical_call_upper_bound_per_task",
        "max_output_tokens_per_call", "output_token_upper_bound",
    )}
    if not all(type(value) is int and 0 <= value <= 100_000_000 for value in counters.values()):
        raise ValueError("candidate runtime metadata contains an invalid counter")
    outcomes = metadata["terminal_outcomes"]
    if not isinstance(outcomes, dict) or set(outcomes) - {
        "success", "partial", "failed", "approval_required", "other",
    } or not all(type(value) is int and 0 <= value <= 1_000_000 for value in outcomes.values()):
        raise ValueError("candidate runtime metadata contains invalid outcomes")
    return {**labels, **counters, "terminal_outcomes": dict(sorted(outcomes.items()))}


def _raw(payload: dict, dotted: str):
    """Read one dotted scorecard field, returning None when absent."""
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def compare_patch_scorecards(current: dict, baseline: dict, policy: dict) -> GateResult:
    """Return incompatibility, regression, or pass without gating wall-clock latency."""
    validate_patch_gate_policy(policy)
    fields = (
        "schema_version", "benchmark", "evaluator_version", "dataset.sha256",
        "dataset.suite_id", "dataset.case_count", "repository", "task_repositories", "config",
    )
    incompatibilities = [
        f"incompatible {field}: baseline={_raw(baseline, field)!r}, current={_raw(current, field)!r}"
        for field in fields if _raw(baseline, field) != _raw(current, field)
    ]
    if incompatibilities:
        return GateResult(False, False, tuple(incompatibilities))
    failures: list[str] = []
    for metric, allowed in (policy.get("max_drop") or {}).items():
        before, after = float(_raw(baseline, metric)), float(_raw(current, metric))
        if before - after > float(allowed) + 1e-12:
            failures.append(
                f"{metric}: baseline={before:.6f}, current={after:.6f}, "
                f"drop={before-after:.6f}, allowed={float(allowed):.6f}"
            )
    for metric, allowed in (policy.get("max_increase") or {}).items():
        before, after = float(_raw(baseline, metric)), float(_raw(current, metric))
        if after - before > float(allowed) + 1e-12:
            failures.append(
                f"{metric}: baseline={before:.6f}, current={after:.6f}, "
                f"increase={after-before:.6f}, allowed={float(allowed):.6f}"
            )
    return GateResult(True, not failures, tuple(failures))


def validate_patch_gate_policy(policy: object) -> None:
    """Require a closed, finite, nonnegative regression policy schema."""
    if not isinstance(policy, dict) or set(policy) != {"max_drop", "max_increase"}:
        raise ValueError("patch outcome gate policy requires exactly max_drop and max_increase")
    allowed = {
        "max_drop": {
            "score", "metrics.harness_valid_rate", "metrics.pristine_pass_rate",
            "metrics.initial_failure_rate", "metrics.patch_applied_rate",
            "metrics.test_pass_rate", "metrics.allowed_path_precision",
            "metrics.forbidden_untouched_rate", "metrics.minimality_score",
        },
        "max_increase": {"metrics.timeout_rate", "metrics.changed_files_avg"},
    }
    for section, allowed_keys in allowed.items():
        values = policy[section]
        if not isinstance(values, dict) or set(values) != allowed_keys:
            raise ValueError(
                f"{section} patch outcome policy fields must exactly match the fixed metric contract"
            )
        for key, value in values.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"policy threshold {section}.{key} must be finite and nonnegative")


def gate_exit_code(result: GateResult) -> int:
    """Map pass, regression, and incompatibility to exit codes 0, 1, and 2."""
    if not result.compatible:
        return 2
    return 0 if result.passed else 1
