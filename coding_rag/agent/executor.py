"""Executable Agent tools bound to a repository workspace."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from coding_rag.agent.safety import AgentSafetyGuard, AgentSafetyPolicy, SafetyDecision
from coding_rag.tools.agent_readonly import AgentToolError, ReadOnlyAgentTools


@dataclass(frozen=True)
class CommandResult:
    command: str | list[str]
    returncode: int
    stdout: str
    stderr: str


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
    ):
        self.repo_path = Path(repo_path).resolve()
        self.readonly = ReadOnlyAgentTools(self.repo_path, chunk_size=chunk_size, overlap=overlap)
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
        return self.readonly.read_file(path, start_line=start_line, end_line=end_line)

    def search_code(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.readonly.search_code(query, top_k=top_k)

    def apply_patch(self, diff: str) -> dict[str, Any]:
        decision = self.safety.check_patch(diff)
        ensure_allowed(decision)

        check = self._run_git(["git", "apply", "--check", "--whitespace=nowarn", "-"], stdin=diff)
        if check.returncode != 0:
            raise AgentToolError(f"Patch check failed: {check.stderr.strip() or check.stdout.strip()}")
        if decision.dry_run:
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
        decision = self.safety.check_command(cmd)
        ensure_allowed(decision)
        timeout = self.safety.resolve_timeout(timeout_seconds)
        if decision.dry_run:
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
        result = self._run_git(["git", "diff", "--no-ext-diff", "--"], stdin=None)
        payload = command_result_to_dict(result, self.safety)
        payload["executed"] = True
        payload["dry_run"] = False
        return payload

    def _run_git(self, command: list[str], stdin: str | None) -> CommandResult:
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
    if not decision.allowed:
        raise AgentToolError(decision.reason)


def decision_to_dict(decision: SafetyDecision) -> dict[str, Any]:
    return {
        "allowed": decision.allowed,
        "reason": decision.reason,
        "matched_rule": decision.matched_rule,
        "dry_run": decision.dry_run,
        "metadata": decision.metadata,
    }


def optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
