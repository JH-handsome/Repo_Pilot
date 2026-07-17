"""Executable Agent tools bound to a repository workspace."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from coding_rag.agent.safety import AgentSafetyGuard, AgentSafetyPolicy, SafetyDecision
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools


@dataclass(frozen=True)
class CommandResult:
    """子进程命令执行的结果，包含命令、返回码、标准输出和标准错误。"""

    command: str | list[str]
    returncode: int
    stdout: str
    stderr: str


SearchProvider = Callable[[str, int], Any]


class AgentExecutor:
    """Small execution surface for Agent tool calls.

    The executor is rooted at one repository path. File reads and code search reuse the
    existing read-only tools; patch and command execution run with the repository as cwd.
    """

    def __init__(
        self,
        repo_path: str | Path,
        *,
        chunk_size: int = 40,
        overlap: int = 5,
        command_timeout: int = 120,
        dry_run: bool = False,
        safe_mode: bool = False,
        safety_policy: AgentSafetyPolicy | None = None,
        search_provider: SearchProvider | None = None,
    ):
        """初始化 Agent 执行器，绑定仓库路径并配置安全策略与工具集。"""
        self.repo_path = Path(repo_path).resolve()
        self.readonly = ReadOnlyAgentTools(self.repo_path, chunk_size=chunk_size, overlap=overlap)
        self.search_provider = search_provider
        if safety_policy is None:
            safety_policy = AgentSafetyPolicy(
                dry_run=dry_run,
                safe_mode=safe_mode,
                command_timeout_seconds=command_timeout,
            )
        elif dry_run or safe_mode:
            safety_policy = replace(
                safety_policy,
                dry_run=safety_policy.dry_run or dry_run,
                safe_mode=safety_policy.safe_mode or safe_mode,
            )
        self.safety = AgentSafetyGuard(safety_policy)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """根据工具名称分派调用。"""
        # executor 的公开入口保持很窄：所有 Agent 工具调用都先落到这里，再分发到具体方法。
        args = arguments or {}
        if name == "read_file":
            return self.read_file(
                path=str(args.get("path") or ""),
                start_line=optional_int(args.get("start_line")),
                end_line=optional_int(args.get("end_line")),
            )
        if name == "search_code":
            return self.search_code(
                query=str(args.get("query") or ""),
                top_k=int(args.get("top_k") or 5),
            )
        if name == "apply_patch":
            return self.apply_patch(str(args.get("diff") or ""))
        if name == "run_command":
            return self.run_command(
                str(args.get("cmd") or ""),
                timeout_seconds=optional_int(args.get("timeout_seconds")),
            )
        if name == "inspect_diff":
            return self.inspect_diff()
        raise AgentToolError(f"Unknown Agent executor tool: {name}")

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """委托仓库文件读取。"""
        return self.readonly.read_file(path, start_line=start_line, end_line=end_line)

    def search_code(self, query: str, top_k: int = 5) -> Any:
        """优先使用注入的搜索提供者，否则回退到只读后端。"""
        if self.search_provider is not None:
            return self.search_provider(query, top_k)
        return self.readonly.search_code(query, top_k=top_k)

    def apply_patch(self, diff: str) -> dict[str, Any]:
        """执行安全检查与 git apply --check，仅在非 dry-run 时真正应用补丁。"""
        # patch 是最高风险工具之一：先做安全检查，再用 git apply --check 做语法/上下文校验。
        decision = self.safety.check_patch(diff)
        ensure_allowed(decision)

        check = self._run_git(["git", "apply", "--check", "--whitespace=nowarn", "-"], stdin=diff)
        if check.returncode != 0:
            raise AgentToolError(f"Patch check failed: {check.stderr.strip() or check.stdout.strip()}")
        if decision.dry_run:
            # dry-run/safe-mode 只证明 patch 可应用，不写入工作区。
            return {
                "applied": False,
                "dry_run": True,
                "safety": decision_to_dict(decision),
                "stdout": check.stdout,
                "stderr": check.stderr,
                "diff": self.inspect_diff(),
            }

        applied = self._run_git(["git", "apply", "--whitespace=nowarn", "-"], stdin=diff)
        if applied.returncode != 0:
            raise AgentToolError(f"Patch apply failed: {applied.stderr.strip() or applied.stdout.strip()}")

        return {
            "applied": True,
            "dry_run": False,
            "safety": decision_to_dict(decision),
            "stdout": applied.stdout,
            "stderr": applied.stderr,
            "diff": self.inspect_diff(),
        }

    def run_command(self, cmd: str, timeout_seconds: int | None = None) -> dict[str, Any]:
        """安全校验命令并在 dry-run 时跳过子进程执行。"""
        # 命令执行必须经过 allowlist/denylist；通过后也用 shell=False 避免 shell 拼接副作用。
        decision = self.safety.check_command(cmd)
        ensure_allowed(decision)
        timeout = self.safety.resolve_timeout(timeout_seconds)
        if decision.dry_run:
            # dry-run/safe-mode 直接返回“本应执行什么”，不会启动子进程。
            return {
                "command": cmd,
                "args": decision.metadata.get("args", []),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "stdout_chars": 0,
                "stderr_chars": 0,
                "executed": False,
                "dry_run": True,
                "timeout_seconds": timeout,
                "safety": decision_to_dict(decision),
            }
        result = run_subprocess(
            decision.metadata["args"],
            cwd=self.repo_path,
            timeout=timeout,
            shell=False,
        )
        payload = command_result_to_dict(result, self.safety)
        payload.update(
            {
                "executed": True,
                "dry_run": False,
                "timeout_seconds": timeout,
                "safety": decision_to_dict(decision),
            }
        )
        return payload

    def inspect_diff(self) -> dict[str, Any]:
        """检查工作区 diff。"""
        result = self._run_git(["git", "diff", "--no-ext-diff", "--"], stdin=None)
        payload = command_result_to_dict(result, self.safety)
        payload["executed"] = True
        payload["dry_run"] = False
        return payload

    def _run_git(self, command: list[str], stdin: str | None) -> CommandResult:
        """使用策略超时在仓库中运行 git。"""
        return run_subprocess(
            command,
            cwd=self.repo_path,
            timeout=self.safety.resolve_timeout(),
            shell=False,
            stdin=stdin,
        )


def run_subprocess(
    command: str | list[str],
    *,
    cwd: Path,
    timeout: int,
    shell: bool,
    stdin: str | None = None,
) -> CommandResult:
    """运行子进程并将超时转换为 AgentToolError。"""
    try:
        completed = subprocess.run(
            command,
            input=stdin,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout,
            shell=shell,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentToolError(f"Command timed out after {timeout}s: {command}") from exc
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def command_result_to_dict(result: CommandResult, safety: AgentSafetyGuard) -> dict[str, Any]:
    """截断命令输出以构建 payload。"""
    stdout, stdout_truncated, stdout_chars = safety.truncate_output(result.stdout)
    stderr, stderr_truncated, stderr_chars = safety.truncate_output(result.stderr)
    return {
        "command": result.command,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_chars": stdout_chars,
        "stderr_chars": stderr_chars,
    }


def ensure_allowed(decision: SafetyDecision) -> None:
    """拒绝被禁止的安全决策。"""
    if not decision.allowed:
        raise AgentToolError(decision.reason)


def decision_to_dict(decision: SafetyDecision) -> dict[str, Any]:
    """序列化安全决策。"""
    return {
        "allowed": decision.allowed,
        "reason": decision.reason,
        "matched_rule": decision.matched_rule,
        "dry_run": decision.dry_run,
        "metadata": decision.metadata,
    }


def optional_int(value: object) -> int | None:
    """将可选值转换为 int。"""
    if value in (None, ""):
        return None
    return int(value)
