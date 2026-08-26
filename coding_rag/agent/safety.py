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
    """不可变的安全决策记录，包含允许/拒绝结果及匹配的规则元数据。"""

    allowed: bool
    reason: str
    matched_rule: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSafetyPolicy:
    """不可变的安全限制与允许/拒绝策略配置。"""

    safe_mode: bool = False
    command_timeout_seconds: int = 60
    max_output_chars: int = 20000
    max_patch_chars: int = 200000
    allowlist: tuple[tuple[str, ...], ...] = DEFAULT_ALLOWED_COMMAND_PREFIXES
    denylist: tuple[tuple[str, ...], ...] = DEFAULT_DENIED_COMMAND_PREFIXES


class AgentSafetyGuard:
    """Central safety policy for patch and command execution."""

    def __init__(self, policy: AgentSafetyPolicy | None = None):
        """使用提供的策略或默认策略初始化安全守卫。"""
        self.policy = policy or AgentSafetyPolicy()

    def check_command(self, cmd: str) -> SafetyDecision:
        """解析命令并应用 safe-mode、拒绝与允许规则。"""
        # 检查顺序很重要：先拒绝 shell 组合语法，再看 denylist，最后才看 allowlist。
        args = parse_command(cmd)
        normalized = normalize_args(args)
        if not normalized:
            return SafetyDecision(False, "cmd is required")
        if self.policy.safe_mode:
            return SafetyDecision(False, "safe mode blocks command execution", "safe-mode")

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
            metadata={"args": args},
        )

    def check_patch(self, diff: str) -> SafetyDecision:
        """检查补丁路径、safe-mode 及删除数量是否符合安全策略。"""
        # patch 安全检查只看“能不能被 Agent 尝试”，真正能否应用由 git apply --check 再判断。
        if not diff.strip():
            return SafetyDecision(False, "diff is required")
        if self.policy.safe_mode:
            return SafetyDecision(False, "safe mode blocks patch execution", "safe-mode")
        if len(diff) > self.policy.max_patch_chars:
            return SafetyDecision(False, "diff exceeds max patch size", "max-patch-chars")
        if "GIT binary patch" in diff or "Binary files " in diff:
            return SafetyDecision(False, "binary patches are not allowed", "binary-patch")

        changed_files, deleted_files = inspect_patch_paths(diff)
        if not changed_files:
            return SafetyDecision(
                False,
                "patch must include diff --git headers with identifiable files",
                "missing-patch-files",
            )
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
            metadata={
                "changed_files": changed_files,
                "deleted_files": deleted_files,
                "deletes_files": bool(deleted_files),
            },
        )

    def resolve_timeout(self, timeout_seconds: int | None = None) -> int:
        """将请求/默认超时值限制在策略和硬上限范围内。"""
        timeout = timeout_seconds if timeout_seconds is not None else self.policy.command_timeout_seconds
        timeout = max(1, int(timeout))
        return min(timeout, MAX_TIMEOUT_SECONDS)

    def truncate_output(self, text: str) -> tuple[str, bool, int]:
        """截断文本至策略上限并返回文本、截断标志和原始长度。"""
        char_count = len(text)
        limit = self.policy.max_output_chars
        if char_count <= limit:
            return text, False, char_count
        marker = f"\n...[truncated {char_count - limit} chars]"
        return text[:limit] + marker, True, char_count


def parse_command(cmd: str) -> list[str]:
    """使用 POSIX shlex 规则将命令字符串拆分为参数列表，shlex 语法错误转换为 AgentToolError。"""
    try:
        return shlex.split(cmd, posix=True)
    except ValueError as exc:
        raise AgentToolError(f"Invalid command syntax: {exc}") from exc


def normalize_args(args: Iterable[str]) -> list[str]:
    """规范化可执行文件别名并返回调整后的参数列表。"""
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
    """返回第一个与 args 开头匹配的前缀（按给定顺序），无匹配则返回 None。"""
    for prefix in prefixes:
        if len(args) >= len(prefix) and tuple(args[: len(prefix)]) == prefix:
            return prefix
    return None


def command_rule_name(prefix: tuple[str, ...]) -> str:
    """将命令-规则令牌元组合并为其显示名称。"""
    return " ".join(prefix)


def inspect_patch_paths(diff: str) -> tuple[list[str], list[str]]:
    """解析 diff 头信息，返回变更路径列表和删除路径列表。"""
    changed: list[str] = []
    deleted: list[str] = []
    current_file: str | None = None
    current_block: list[str] = []

    def finalize_current_file() -> None:
        """当前差异块含 /dev/null 时，将当前文件路径追加到已删除路径列表中。"""
        if current_file and "/dev/null" in "\n".join(current_block):
            deleted.append(current_file)

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            finalize_current_file()
            try:
                parts = shlex.split(line, posix=True)
            except ValueError:
                parts = []
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
    """去除diff路径的前缀 "a/" 或 "b/"。"""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def is_unsafe_patch_path(path: str) -> bool:
    """检测绝对路径、路径遍历或其它不允许的补丁路径。"""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or WINDOWS_ABSOLUTE_PATTERN.match(path):
        return True
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def is_env_path(path: str) -> bool:
    """检测路径的 basename 是否为 .env 或以 .env. 开头。"""
    name = PurePosixPath(path.replace("\\", "/")).name
    return name == ".env" or name.startswith(".env.")


def load_agent_safety_policy(repo_path: str | Path = ".", policy_path: str | Path | None = None) -> AgentSafetyPolicy:
    """Load an optional project safety policy while keeping denylist defaults."""
    # 项目配置只允许扩展策略；默认 denylist 会始终合并回来，避免本地配置绕过硬限制。
    path = Path(policy_path) if policy_path else Path(repo_path) / ".repopilot" / "policy.json"
    if not path.exists():
        return AgentSafetyPolicy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "dry_run" in payload:
        raise AgentToolError("policy field 'dry_run' is no longer supported; use safe_mode")
    allowlist = tuple(parse_rule_list(payload.get("allowlist"), DEFAULT_ALLOWED_COMMAND_PREFIXES))
    denylist = tuple(
        merge_rule_lists(
            DEFAULT_DENIED_COMMAND_PREFIXES,
            parse_rule_list(payload.get("denylist"), ()),
        )
    )
    return AgentSafetyPolicy(
        safe_mode=bool(payload.get("safe_mode", False)),
        command_timeout_seconds=int(payload.get("command_timeout_seconds", 60)),
        max_output_chars=int(payload.get("max_output_chars", 20000)),
        max_patch_chars=int(payload.get("max_patch_chars", 200000)),
        allowlist=allowlist,
        denylist=denylist,
    )


def parse_rule_list(value: object, default: tuple[tuple[str, ...], ...]) -> list[tuple[str, ...]]:
    """验证并解析策略规则条目为命令令牌元组。"""
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
    """按顺序合并各规则可迭代对象，丢弃空规则和重复项。"""
    merged: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for rules in rule_lists:
        for rule in rules:
            if rule and rule not in seen:
                merged.append(rule)
                seen.add(rule)
    return merged
