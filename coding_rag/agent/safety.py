"""Safety checks for executable Agent tools."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

from coding_rag.tools.agent_readonly import AgentToolError


DEFAULT_ALLOWED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "unittest"),
    ("python", "-m", "compileall"),
    ("python", "main.py", "--help"),
    ("python", "scripts/retrieval_eval.py"),
    ("rg",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "show"),
    ("git", "log"),
    ("git", "ls-files"),
)

DEFAULT_DENIED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("rm",),
    ("del",),
    ("rmdir",),
    ("remove-item",),
    ("mv",),
    ("move",),
    ("cp",),
    ("copy",),
    ("pip", "install"),
    ("python", "-c"),
    ("git", "reset"),
    ("git", "checkout"),
    ("git", "clean"),
    ("git", "apply"),
    ("git", "commit"),
    ("git", "push"),
    ("git", "pull"),
    ("git", "merge"),
    ("git", "rebase"),
    ("curl",),
    ("wget",),
    ("powershell",),
    ("pwsh",),
    ("cmd",),
    ("start",),
)

SHELL_META_PATTERN = re.compile(r"&&|\|\||[;&|<>`]|\$\(|\r|\n")
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")
MAX_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    matched_rule: str | None = None
    dry_run: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSafetyPolicy:
    dry_run: bool = False
    safe_mode: bool = False
    command_timeout_seconds: int = 60
    max_output_chars: int = 20000
    max_patch_chars: int = 200000
    allowlist: tuple[tuple[str, ...], ...] = DEFAULT_ALLOWED_COMMAND_PREFIXES
    denylist: tuple[tuple[str, ...], ...] = DEFAULT_DENIED_COMMAND_PREFIXES

    @property
    def no_execute(self) -> bool:
        return self.dry_run or self.safe_mode


class AgentSafetyGuard:
    """Central safety policy for patch and command execution."""

    def __init__(self, policy: AgentSafetyPolicy | None = None):
        self.policy = policy or AgentSafetyPolicy()

    def check_command(self, cmd: str) -> SafetyDecision:
        args = parse_command(cmd)
        normalized = normalize_args(args)
        if not normalized:
            return SafetyDecision(False, "cmd is required")

        if SHELL_META_PATTERN.search(cmd):
            return SafetyDecision(False, "shell composition and redirection are not allowed", "shell-meta")

        denied = matching_prefix(normalized, self.policy.denylist)
        if denied:
            return SafetyDecision(False, "command is denied by safety policy", command_rule_name(denied))

        allowed = matching_prefix(normalized, self.policy.allowlist)
        if not allowed:
            return SafetyDecision(False, "command is not in the allowlist", None)

        return SafetyDecision(
            True,
            "command allowed",
            command_rule_name(allowed),
            dry_run=self.policy.no_execute,
            metadata={"args": args},
        )

    def check_patch(self, diff: str) -> SafetyDecision:
        if not diff.strip():
            return SafetyDecision(False, "diff is required")
        if len(diff) > self.policy.max_patch_chars:
            return SafetyDecision(False, "diff exceeds max patch size", "max-patch-chars")
        if "GIT binary patch" in diff or "Binary files " in diff:
            return SafetyDecision(False, "binary patches are not allowed", "binary-patch")

        changed_files, deleted_files = inspect_patch_paths(diff)
        for path in changed_files:
            if is_unsafe_patch_path(path):
                return SafetyDecision(False, f"unsafe patch path: {path}", "unsafe-path")
            if is_env_path(path):
                return SafetyDecision(False, ".env files cannot be modified by Agent patches", "env-file")
        if len(deleted_files) > 1:
            return SafetyDecision(
                False,
                "Patch deletes multiple files; delete files one by one or ask for confirmation first",
                "multi-file-delete",
                metadata={"deleted_files": deleted_files},
            )

        return SafetyDecision(
            True,
            "patch allowed",
            "patch-policy",
            dry_run=self.policy.no_execute,
            metadata={
                "changed_files": changed_files,
                "deleted_files": deleted_files,
                "deletes_files": bool(deleted_files),
            },
        )

    def resolve_timeout(self, timeout_seconds: int | None = None) -> int:
        timeout = timeout_seconds if timeout_seconds is not None else self.policy.command_timeout_seconds
        timeout = max(1, int(timeout))
        return min(timeout, MAX_TIMEOUT_SECONDS)

    def truncate_output(self, text: str) -> tuple[str, bool, int]:
        char_count = len(text)
        limit = self.policy.max_output_chars
        if char_count <= limit:
            return text, False, char_count
        marker = f"\n...[truncated {char_count - limit} chars]"
        return text[:limit] + marker, True, char_count


def parse_command(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=True)
    except ValueError as exc:
        raise AgentToolError(f"Invalid command syntax: {exc}") from exc


def normalize_args(args: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for index, arg in enumerate(args):
        value = arg.strip().strip("\"'").casefold()
        if index == 0 and value.endswith(".exe"):
            value = value[:-4]
        if index == 0 and value == "py":
            value = "python"
        normalized.append(value.replace("\\", "/"))
    return normalized


def matching_prefix(
    args: list[str],
    prefixes: tuple[tuple[str, ...], ...],
) -> tuple[str, ...] | None:
    for prefix in prefixes:
        if len(args) >= len(prefix) and tuple(args[: len(prefix)]) == prefix:
            return prefix
    return None


def command_rule_name(prefix: tuple[str, ...]) -> str:
    return " ".join(prefix)


def inspect_patch_paths(diff: str) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    deleted: list[str] = []
    current_file: str | None = None
    current_block: list[str] = []

    def finalize_current_file() -> None:
        if current_file and "/dev/null" in "\n".join(current_block):
            deleted.append(current_file)

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            finalize_current_file()
            parts = line.split()
            if len(parts) >= 4:
                for raw_path in (parts[2], parts[3]):
                    path = strip_git_prefix(raw_path)
                    if path != "/dev/null" and path not in changed:
                        changed.append(path)
                current_file = strip_git_prefix(parts[2])
            else:
                current_file = None
            current_block = [line]
        else:
            current_block.append(line)
    finalize_current_file()
    return changed, deleted


def strip_git_prefix(path: str) -> str:
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def is_unsafe_patch_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or WINDOWS_ABSOLUTE_PATTERN.match(path):
        return True
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def is_env_path(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name
    return name == ".env" or name.startswith(".env.")


def load_agent_safety_policy(repo_path: str | Path = ".", policy_path: str | Path | None = None) -> AgentSafetyPolicy:
    """Load an optional project safety policy while keeping denylist defaults."""
    path = Path(policy_path) if policy_path else Path(repo_path) / ".repopilot" / "policy.json"
    if not path.exists():
        return AgentSafetyPolicy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowlist = tuple(parse_rule_list(payload.get("allowlist"), DEFAULT_ALLOWED_COMMAND_PREFIXES))
    denylist = tuple(
        merge_rule_lists(
            DEFAULT_DENIED_COMMAND_PREFIXES,
            parse_rule_list(payload.get("denylist"), ()),
        )
    )
    return AgentSafetyPolicy(
        dry_run=bool(payload.get("dry_run", False)),
        safe_mode=bool(payload.get("safe_mode", False)),
        command_timeout_seconds=int(payload.get("command_timeout_seconds", 60)),
        max_output_chars=int(payload.get("max_output_chars", 20000)),
        max_patch_chars=int(payload.get("max_patch_chars", 200000)),
        allowlist=allowlist,
        denylist=denylist,
    )


def parse_rule_list(value: object, default: tuple[tuple[str, ...], ...]) -> list[tuple[str, ...]]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise AgentToolError("policy rule list must be a list")
    rules: list[tuple[str, ...]] = []
    for item in value:
        if isinstance(item, str):
            parts = tuple(normalize_args(shlex.split(item)))
        elif isinstance(item, list):
            parts = tuple(normalize_args(str(part) for part in item))
        else:
            raise AgentToolError("policy rules must be strings or string lists")
        if parts:
            rules.append(parts)
    return rules


def merge_rule_lists(*rule_lists: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    merged: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for rules in rule_lists:
        for rule in rules:
            if rule and rule not in seen:
                merged.append(rule)
                seen.add(rule)
    return merged
