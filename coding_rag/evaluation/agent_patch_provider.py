"""Adapt the approved unified Agent runtime into an isolated patch candidate provider."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from coding_rag.agent.runtime import (
    ChatClient,
    UnifiedRun,
    UnifiedRunConfig,
    resume_unified_query,
    run_unified_query,
)
from coding_rag.agent.safety import AgentSafetyGuard, AgentSafetyPolicy
from coding_rag.rag.prompt import GenerationMode
from coding_rag.repository.search_index import RepositoryIndexRegistry

from .patch_outcome import (
    MAX_PATCH_CHARS,
    CandidateRequest,
    _run,
    validate_candidate_patch_paths,
)


AGENT_CANDIDATE_POLICY_VERSION = "unified-agent-candidate-v1"
SAFE_PUBLIC_LABEL = re.compile(r"^[A-Za-z0-9_.:/+-]{1,128}$")


class AgentCandidateRejected(RuntimeError):
    """Represent a private-policy rejection without exposing candidate content."""


class AgentCandidateProvider:
    """Run one ChatClient in a disposable copy and return only its cumulative diff.

    The model sees the task requirement and observations produced inside the copied
    repository. Private evaluator constraints are used only by the approval broker.
    """

    name = "agent_live_v1"

    def __init__(
        self,
        client: ChatClient,
        *,
        provider_label: str,
        model_label: str,
        max_steps: int = 6,
        max_approvals: int = 2,
        max_context_chars: int = 12_000,
        execution_mode: str = "live_unified_agent",
        max_output_tokens_per_call: int = 0,
    ) -> None:
        """Bind a generic client and strict per-task execution limits."""
        for field, value in (("provider", provider_label), ("model", model_label)):
            if not isinstance(value, str) or SAFE_PUBLIC_LABEL.fullmatch(value) is None:
                raise ValueError(f"agent candidate {field} label is invalid")
        if not 1 <= max_steps <= 12:
            raise ValueError("agent candidate max_steps must be between 1 and 12")
        if not 1 <= max_approvals <= 4:
            raise ValueError("agent candidate max_approvals must be between 1 and 4")
        if not 1000 <= max_context_chars <= 50_000:
            raise ValueError("agent candidate max_context_chars must be between 1000 and 50000")
        if execution_mode not in {"live_unified_agent", "offline_scripted_contract"}:
            raise ValueError("agent candidate execution_mode is invalid")
        if not 0 <= max_output_tokens_per_call <= 8192:
            raise ValueError("agent candidate output token cap must be between 0 and 8192")
        self.client = client
        self.provider_label = provider_label
        self.model_label = model_label
        self.max_steps = max_steps
        self.max_approvals = max_approvals
        self.max_context_chars = max_context_chars
        self.execution_mode = execution_mode
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.name = (
            "agent_live_v1" if execution_mode == "live_unified_agent"
            else "agent_offline_contract_v1"
        )
        self._task_count = 0
        self._approval_count = 0
        self._rejection_count = 0
        self._incomplete_task_count = 0
        self._llm_calls = 0
        self._llm_attempts = 0
        self._llm_retries = 0
        self._outcomes: dict[str, int] = {}

    @property
    def scorecard_metadata(self) -> dict:
        """Return bounded aggregate metadata without prompts, source, patches, or answers."""
        return {
            "mode": self.execution_mode,
            "policy_version": AGENT_CANDIDATE_POLICY_VERSION,
            "provider": self.provider_label,
            "model": self.model_label,
            "task_count": self._task_count,
            "approval_count": self._approval_count,
            "rejection_count": self._rejection_count,
            "incomplete_task_count": self._incomplete_task_count,
            "llm_call_count": self._llm_calls,
            "llm_attempt_count": self._llm_attempts,
            "llm_retry_count": self._llm_retries,
            "logical_call_upper_bound_per_task": self.max_steps + 2,
            "max_output_tokens_per_call": self.max_output_tokens_per_call,
            "output_token_upper_bound": (
                self._task_count * (self.max_steps + 2) * self.max_output_tokens_per_call
            ),
            "terminal_outcomes": dict(sorted(self._outcomes.items())),
            "billing_measurement": "unavailable",
        }

    def provide(self, request: CandidateRequest, repo_path: Path) -> str:
        """Run the real graph in a second workspace and return its validated cumulative diff."""
        self._task_count += 1
        self._reject_links(repo_path)
        with tempfile.TemporaryDirectory(prefix="repopilot-agent-candidate-") as folder:
            temp_root = Path(folder)
            candidate_repo = temp_root / "repo"
            shutil.copytree(repo_path, candidate_repo, symlinks=False)
            self._prepare_git_index(candidate_repo, request.allowed_paths)
            baseline_metadata = self._workspace_metadata(candidate_repo)
            registry = RepositoryIndexRegistry(
                cache_dir=temp_root / "index-cache",
                disk_cache=False,
            )
            policy = AgentSafetyPolicy(max_patch_chars=MAX_PATCH_CHARS)
            config = UnifiedRunConfig(
                repo_path=candidate_repo,
                generation_mode=GenerationMode.CODE_GENERATE,
                max_context_chars=self.max_context_chars,
                max_steps=self.max_steps,
                trace_db_path=None,
                trace_source="patch-outcome-agent-candidate",
                snapshot_root=temp_root / "snapshots",
            )
            try:
                run = run_unified_query(
                    request.requirement,
                    config,
                    self.client,
                    safety_policy=policy,
                    index_registry=registry,
                )
            except Exception:
                self._incomplete_task_count += 1
                self._outcomes["failed"] = self._outcomes.get("failed", 0) + 1
                raise
            approvals = 0
            try:
                while run.status == "approval_required":
                    approval = run.approval or {}
                    if approval.get("tool") != "apply_patch":
                        self._rejection_count += 1
                        break
                    if approvals >= self.max_approvals:
                        self._rejection_count += 1
                        break
                    diff = ((approval.get("arguments") or {}).get("diff"))
                    self._validate_private_patch(diff, request.allowed_paths)
                    fingerprint = approval.get("fingerprint")
                    if not isinstance(fingerprint, str) or not fingerprint:
                        raise AgentCandidateRejected("candidate approval fingerprint was unavailable")
                    approvals += 1
                    self._approval_count += 1
                    try:
                        run = resume_unified_query(
                            run,
                            config,
                            self.client,
                            approval_fingerprint=fingerprint,
                            safety_policy=policy,
                            index_registry=registry,
                        )
                    except Exception:
                        self._incomplete_task_count += 1
                        raise
            finally:
                self._record_run(run)
            changed = self._changed_workspace_paths(baseline_metadata, candidate_repo)
            if changed - set(request.allowed_paths):
                self._rejection_count += 1
                raise AgentCandidateRejected("candidate changed a file outside its private scope")
            patch = self._cumulative_diff(candidate_repo, request.allowed_paths)
            if patch:
                self._validate_private_patch(patch, request.allowed_paths)
            return patch

    def _validate_private_patch(self, patch: object, allowed_paths: tuple[str, ...]) -> None:
        """Apply evaluator-private scope and deletion rules before granting approval."""
        if not isinstance(patch, str):
            self._rejection_count += 1
            raise AgentCandidateRejected("candidate patch was not text")
        decision = AgentSafetyGuard(
            AgentSafetyPolicy(max_patch_chars=MAX_PATCH_CHARS)
        ).check_patch(patch)
        paths, path_error = validate_candidate_patch_paths(patch, allowed_paths)
        if (
            not decision.allowed or decision.metadata.get("deleted_files")
            or path_error is not None or not paths
        ):
            self._rejection_count += 1
            raise AgentCandidateRejected("candidate patch violated private benchmark policy")

    def _record_run(self, run: UnifiedRun) -> None:
        """Aggregate only sanitized counters from a completed or stopped graph run."""
        summary = run.trace.get("summary") or {}
        self._llm_calls += _bounded_counter(summary.get("llm_call_count"))
        self._llm_attempts += _bounded_counter(summary.get("llm_attempt_count"))
        self._llm_retries += _bounded_counter(summary.get("llm_retry_count"))
        status = run.status if run.status in {"success", "partial", "failed", "approval_required"} else "other"
        self._outcomes[status] = self._outcomes.get(status, 0) + 1

    @staticmethod
    def _reject_links(repo_path: Path) -> None:
        """Reject links before copying so the candidate cannot reach outside its source tree."""
        for path in repo_path.rglob("*"):
            if path.is_symlink():
                raise AgentCandidateRejected("candidate source workspace contains a symlink")

    @staticmethod
    def _prepare_git_index(repo: Path, allowed_paths: tuple[str, ...]) -> None:
        """Stage the faulted allowed files as the private baseline for cumulative diffing."""
        initialized = _run(["git", "init", "--quiet"], cwd=repo, timeout=15)
        if initialized["returncode"] != 0:
            raise AgentCandidateRejected("candidate Git index initialization failed")
        staged = _run(["git", "add", "--", *allowed_paths], cwd=repo, timeout=15)
        if staged["returncode"] != 0:
            raise AgentCandidateRejected("candidate baseline staging failed")

    @staticmethod
    def _cumulative_diff(repo: Path, allowed_paths: tuple[str, ...]) -> str:
        """Read a bounded diff while keeping Git diagnostics out of patch bytes."""
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run(
                    [
                        "git", "diff", "--no-ext-diff", "--no-color", "--no-textconv",
                        "--binary", "--", *allowed_paths,
                    ],
                    cwd=str(repo), shell=False, stdout=stdout, stderr=stderr,
                    timeout=15, check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise AgentCandidateRejected("candidate cumulative diff timed out") from error
            stdout_size = stdout.tell()
            stderr_size = stderr.tell()
            if result.returncode != 0 or stdout_size > MAX_PATCH_CHARS or stderr_size:
                raise AgentCandidateRejected("candidate cumulative diff was unavailable or too large")
            stdout.seek(0)
            try:
                return stdout.read(MAX_PATCH_CHARS + 1).decode("utf-8")
            except UnicodeDecodeError as error:
                raise AgentCandidateRejected("candidate cumulative diff was not UTF-8 text") from error

    @staticmethod
    def _workspace_metadata(repo: Path) -> dict[str, tuple[int, int]]:
        """Capture best-effort metadata outside the evaluator-owned Git directory."""
        metadata: dict[str, tuple[int, int]] = {}
        for path in repo.rglob("*"):
            relative = path.relative_to(repo)
            if relative.parts and relative.parts[0] == ".git":
                continue
            if path.is_symlink():
                raise AgentCandidateRejected("candidate workspace contains a symlink")
            if path.is_file():
                stat = path.stat()
                metadata[relative.as_posix()] = (stat.st_size, stat.st_mtime_ns)
        return metadata

    @classmethod
    def _changed_workspace_paths(
        cls, before: dict[str, tuple[int, int]], repo: Path,
    ) -> set[str]:
        """Return added, removed, or metadata-changed paths after Agent execution."""
        after = cls._workspace_metadata(repo)
        return {path for path in set(before) | set(after) if before.get(path) != after.get(path)}


def _bounded_counter(value: object) -> int:
    """Accept only small nonnegative integer counters from trace summaries."""
    return value if type(value) is int and 0 <= value <= 10_000 else 0
