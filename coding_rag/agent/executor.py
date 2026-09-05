"""Executable Agent tools bound to a repository workspace."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from coding_rag.agent.safety import AgentSafetyGuard, AgentSafetyPolicy, SafetyDecision
from coding_rag.agent.snapshots import SnapshotStore, normalize_snapshot_paths, resolve_repo_file
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools


@dataclass(frozen=True)
class CommandResult:
    """子进程命令执行的结果，包含命令、返回码、标准输出和标准错误。"""

    command: str | list[str]
    returncode: int
    stdout: str
    stderr: str


SearchProvider = Callable[[str, int], Any]


class WriteApprovalRequired(AgentToolError):
    """Raised when a real write lacks an exact, unused approval fingerprint."""

    def __init__(self, request: dict[str, Any]):
        """Preserve the approval envelope for CLI/HTTP presentation."""
        self.request = request
        super().__init__("write approval is required")


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
        safe_mode: bool = False,
        safety_policy: AgentSafetyPolicy | None = None,
        search_provider: SearchProvider | None = None,
        write_approval: str | None = None,
        snapshot_root: str | Path | None = None,
    ):
        """初始化 Agent 执行器，绑定仓库路径并配置安全策略与工具集。"""
        self.repo_path = Path(repo_path).resolve()
        self.readonly = ReadOnlyAgentTools(self.repo_path, chunk_size=chunk_size, overlap=overlap)
        self.search_provider = search_provider
        if safety_policy is None:
            safety_policy = AgentSafetyPolicy(
                safe_mode=safe_mode,
                command_timeout_seconds=command_timeout,
            )
        elif safe_mode:
            safety_policy = replace(
                safety_policy,
                safe_mode=safety_policy.safe_mode or safe_mode,
            )
        self.safety = AgentSafetyGuard(safety_policy)
        self.snapshots = SnapshotStore(self.repo_path, snapshot_root=snapshot_root)
        self._write_approval = write_approval
        self._write_approval_consumed = False

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
                affected_files=args.get("affected_files") or [],
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

    def preview_write(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate a write request and return its exact approval envelope."""
        if name == "apply_patch":
            decision = self.safety.check_patch(str(arguments.get("diff") or ""))
            ensure_allowed(decision)
            files = normalize_snapshot_paths(decision.metadata.get("changed_files") or [])
        elif name == "run_command":
            decision = self.safety.check_command(
                str(arguments.get("cmd") or ""),
                repo_path=self.repo_path,
            )
            ensure_allowed(decision)
            files = normalize_snapshot_paths(arguments.get("affected_files") or [])
            if not files:
                raise AgentToolError("run_command must declare affected_files before approval")
        else:
            raise AgentToolError(f"tool does not require write approval: {name}")
        for relative_path in files:
            resolve_repo_file(self.repo_path, relative_path)
        canonical = canonical_write_arguments(name, arguments, files)
        return {
            "tool": name,
            "files": files,
            "arguments": canonical,
            "fingerprint": write_call_fingerprint(name, canonical),
            "safety": decision_to_dict(decision),
        }

    def apply_patch(self, diff: str) -> dict[str, Any]:
        """Check, approve, snapshot and apply one unified diff."""
        # patch 是最高风险工具之一：先做安全检查，再用 git apply --check 做语法/上下文校验。
        decision = self.safety.check_patch(diff)
        ensure_allowed(decision)

        check = self._run_git(["git", "apply", "--check", "--whitespace=nowarn", "-"], stdin=diff)
        if check.returncode != 0:
            raise AgentToolError(f"Patch check failed: {check.stderr.strip() or check.stdout.strip()}")
        request = self.preview_write("apply_patch", {"diff": diff})
        self._consume_write_approval(request)
        snapshot = self.snapshots.create(request["files"], tool="apply_patch")

        applied = self._run_git(["git", "apply", "--whitespace=nowarn", "-"], stdin=diff)
        if applied.returncode != 0:
            self._rollback_failed_write(snapshot["snapshot_id"], "patch apply")
            raise AgentToolError(
                f"Patch apply failed and declared files were restored: "
                f"{applied.stderr.strip() or applied.stdout.strip()}"
            )

        return {
            "applied": True,
            "safety": decision_to_dict(decision),
            "stdout": applied.stdout,
            "stderr": applied.stderr,
            "snapshot_id": snapshot["snapshot_id"],
            "affected_files": request["files"],
            "post_change_diff": self.snapshots.diff(snapshot["snapshot_id"]),
        }

    def run_command(
        self,
        cmd: str,
        affected_files: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Check, approve, snapshot and execute one allowlisted command."""
        # 命令执行必须经过 allowlist/denylist；通过后也用 shell=False 避免 shell 拼接副作用。
        decision = self.safety.check_command(cmd, repo_path=self.repo_path)
        ensure_allowed(decision)
        timeout = self.safety.resolve_timeout(timeout_seconds)
        arguments: dict[str, Any] = {
            "cmd": cmd,
            "affected_files": affected_files or [],
        }
        if timeout_seconds is not None:
            arguments["timeout_seconds"] = timeout_seconds
        request = self.preview_write("run_command", arguments)
        self._consume_write_approval(request)
        snapshot = self.snapshots.create(request["files"], tool="run_command")
        try:
            result = run_subprocess(
                decision.metadata["args"],
                cwd=self.repo_path,
                timeout=timeout,
                shell=False,
            )
        except Exception:
            self._rollback_failed_write(snapshot["snapshot_id"], "command execution")
            raise
        payload = command_result_to_dict(result, self.safety)
        payload.update(
            {
                "executed": True,
                "timeout_seconds": timeout,
                "safety": decision_to_dict(decision),
                "snapshot_id": snapshot["snapshot_id"],
                "affected_files": request["files"],
                "post_change_diff": self.snapshots.diff(snapshot["snapshot_id"]),
            }
        )
        return payload

    def inspect_diff(self) -> dict[str, Any]:
        """检查工作区 diff。"""
        result = self._run_git(["git", "diff", "--no-ext-diff", "--"], stdin=None)
        return command_result_to_dict(result, self.safety)

    def rollback_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore a previously cached approved write."""
        return self.snapshots.rollback(snapshot_id)

    def _consume_write_approval(self, request: dict[str, Any]) -> None:
        """Consume one approval only when the exact tool arguments match."""
        if self._write_approval_consumed or self._write_approval != request["fingerprint"]:
            raise WriteApprovalRequired(request)
        self._write_approval_consumed = True

    def _rollback_failed_write(self, snapshot_id: str, operation: str) -> None:
        """Restore declared files after an execution exception, surfacing rollback failure."""
        try:
            self.snapshots.rollback(snapshot_id)
        except Exception as rollback_error:
            raise AgentToolError(
                f"{operation} failed and automatic rollback failed; "
                f"manual recovery snapshot: {snapshot_id}: {rollback_error}"
            ) from rollback_error

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
        "metadata": decision.metadata,
    }


def canonical_write_arguments(
    name: str,
    arguments: dict[str, Any],
    files: list[str],
) -> dict[str, Any]:
    """Normalize fields covered by one approval fingerprint."""
    if name == "apply_patch":
        return {"diff": str(arguments.get("diff") or "")}
    canonical: dict[str, Any] = {
        "cmd": str(arguments.get("cmd") or ""),
        "affected_files": files,
    }
    timeout = optional_int(arguments.get("timeout_seconds"))
    if timeout is not None:
        canonical["timeout_seconds"] = timeout
    return canonical


def write_call_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Hash one exact write tool call for single-use approval."""
    payload = json.dumps(
        {"tool": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optional_int(value: object) -> int | None:
    """将可选值转换为 int。"""
    if value in (None, ""):
        return None
    return int(value)
