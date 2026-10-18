"""Portable validation, evaluation, and gating for pinned source repositories."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import mean
from urllib.parse import urlparse

from .gate import GateResult
from .retrieval import (
    RETRIEVER_NAME,
    EvalCase,
    PreparedRetrievalPipeline,
    RetrievalEvalConfig,
    dataset_sha256,
    evaluate_retrieval,
    repo_fingerprint,
)


@dataclass(frozen=True)
class RepositorySpec:
    """One validated pinned repository from the portable manifest."""

    id: str
    relative_path: str
    root: Path
    remote: str
    commit: str
    license: str
    license_url: str
    purpose: str


@dataclass(frozen=True)
class MultiRepoCase:
    """One source-grounded retrieval case assigned to a manifest repository."""

    id: str
    repo_id: str
    query: str
    gold_files: tuple[str, ...]
    task_type: str
    difficulty: str


@dataclass(frozen=True)
class MultiRepoSuite:
    """Validated repositories, cases, and portable input identities."""

    repositories: tuple[RepositorySpec, ...]
    cases: tuple[MultiRepoCase, ...]
    manifest_name: str
    manifest_sha256: str
    evalset_name: str
    evalset_sha256: str


class MultiRepoValidationError(ValueError):
    """Raised with every actionable manifest or evalset validation finding."""

    def __init__(self, errors: list[str]):
        """Preserve individual validation findings and a concise combined message."""
        super().__init__("; ".join(errors))
        self.errors = tuple(errors)


def normalize_remote(value: str) -> str:
    """Normalize common HTTPS, SSH, and scp-style Git remotes for comparison."""
    remote = value.strip().replace("\\", "/")
    if remote.startswith("git@") and ":" in remote:
        host, path = remote[4:].split(":", 1)
        remote = f"{host}/{path}"
    elif "://" in remote:
        parsed = urlparse(remote)
        remote = f"{parsed.hostname or ''}/{parsed.path.lstrip('/')}"
    remote = remote.rstrip("/")
    if remote.casefold().endswith(".git"):
        remote = remote[:-4]
    return remote.casefold()


def _git(root: Path, *args: str) -> str | None:
    """Return stripped Git output or None without mutating the checkout."""
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _safe_relative(value: object) -> str | None:
    """Return a normalized safe relative POSIX path, otherwise None."""
    text = str(value).replace("\\", "/").strip()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or Path(text).is_absolute():
        return None
    return pure.as_posix()


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and require every non-empty line to be an object."""
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise MultiRepoValidationError([f"evalset line {line_number} must be an object"])
        rows.append(item)
    return rows


def load_and_validate_suite(workspace_root: str | Path, manifest_path: str | Path, evalset_path: str | Path) -> MultiRepoSuite:
    """Strictly validate inputs, checkout identities, confinement, and gold files."""
    workspace = Path(workspace_root).resolve()
    manifest_file = Path(manifest_path)
    evalset_file = Path(evalset_path)
    errors: list[str] = []
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise MultiRepoValidationError(["manifest schema_version must equal 1"])
    raw_repositories = manifest.get("repositories")
    if not isinstance(raw_repositories, list) or not raw_repositories:
        raise MultiRepoValidationError(["manifest repositories must be a non-empty array"])

    repositories: list[RepositorySpec] = []
    seen_repos: set[str] = set()
    required = {"id", "path", "remote", "commit", "license", "license_url", "purpose"}
    for index, item in enumerate(raw_repositories, 1):
        label = f"manifest repository {index}"
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
            continue
        repo_id = str(item["id"]).strip()
        relative = _safe_relative(item["path"])
        if not repo_id or repo_id in seen_repos:
            errors.append(f"{label} has empty or duplicate id: {repo_id!r}")
            continue
        seen_repos.add(repo_id)
        if relative is None:
            errors.append(f"{label} path must be a confined relative path")
            continue
        root = (workspace / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            root.relative_to(workspace)
        except ValueError:
            errors.append(f"{label} path escapes workspace: {relative}")
            continue
        commit = str(item["commit"]).strip().casefold()
        remote = str(item["remote"]).strip()
        if not all(str(item[field]).strip() for field in ("remote", "commit", "license", "license_url", "purpose")):
            errors.append(f"{label} has empty required metadata")
            continue
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            errors.append(f"{label} commit must be a full 40-character hexadecimal SHA")
            continue
        if not root.is_dir():
            errors.append(
                f"missing repository {repo_id!r} at {relative}; prepare the local checkout at {commit} (no automatic clone)"
            )
            continue
        head = _git(root, "rev-parse", "HEAD")
        actual_remote = _git(root, "remote", "get-url", "origin")
        if head is None:
            errors.append(f"repository {repo_id!r} is not a readable Git checkout")
        elif head.casefold() != commit:
            errors.append(f"repository {repo_id!r} HEAD mismatch: expected {commit}, found {head.casefold()}")
        if actual_remote is None:
            errors.append(f"repository {repo_id!r} has no readable origin remote")
        elif normalize_remote(actual_remote) != normalize_remote(remote):
            errors.append(
                f"repository {repo_id!r} remote mismatch: expected {normalize_remote(remote)}, found {normalize_remote(actual_remote)}"
            )
        repositories.append(RepositorySpec(
            repo_id, relative, root, normalize_remote(remote), commit,
            str(item["license"]).strip(), str(item["license_url"]).strip(), str(item["purpose"]).strip(),
        ))

    repo_map = {repo.id: repo for repo in repositories}
    cases: list[MultiRepoCase] = []
    seen_cases: set[str] = set()
    for index, item in enumerate(_read_jsonl(evalset_file), 1):
        case_id = str(item.get("id") or "").strip()
        repo_id = str(item.get("repo_id") or "").strip()
        query = str(item.get("query") or "").strip()
        task_type = str(item.get("task_type") or "").strip()
        difficulty = str(item.get("difficulty") or "").strip()
        gold = item.get("gold_files")
        if not case_id or case_id in seen_cases:
            errors.append(f"evalset line {index} has empty or duplicate id: {case_id!r}")
        seen_cases.add(case_id)
        if repo_id not in repo_map:
            errors.append(f"case {case_id or index!r} references unknown repo_id {repo_id!r}")
        if not query:
            errors.append(f"case {case_id or index!r} has an empty query")
        if not task_type or not difficulty:
            errors.append(f"case {case_id or index!r} requires task_type and difficulty")
        if not isinstance(gold, list) or not gold:
            errors.append(f"case {case_id or index!r} requires non-empty gold_files")
            gold = []
        normalized_gold: list[str] = []
        for raw_path in gold:
            relative = _safe_relative(raw_path)
            if relative is None:
                errors.append(f"case {case_id or index!r} has unsafe gold path: {raw_path!r}")
                continue
            if relative in normalized_gold:
                errors.append(f"case {case_id or index!r} has duplicate gold path: {relative}")
                continue
            normalized_gold.append(relative)
            repo = repo_map.get(repo_id)
            if repo is not None:
                gold_path = (repo.root / Path(*PurePosixPath(relative).parts)).resolve()
                try:
                    gold_path.relative_to(repo.root)
                except ValueError:
                    errors.append(f"case {case_id or index!r} gold path escapes repository: {repo_id}:{relative}")
                else:
                    if not gold_path.is_file():
                        errors.append(f"case {case_id or index!r} gold file does not exist: {repo_id}:{relative}")
        cases.append(MultiRepoCase(case_id, repo_id, query, tuple(normalized_gold), task_type, difficulty))
    if not cases:
        errors.append("evalset must contain at least one case")
    if errors:
        raise MultiRepoValidationError(errors)
    return MultiRepoSuite(
        tuple(repositories), tuple(cases), manifest_file.name, dataset_sha256(manifest_file),
        evalset_file.name, dataset_sha256(evalset_file),
    )


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _aggregate(rows: list[dict], top_k: int) -> dict:
    """Aggregate retrieval rows into standard metrics and bad-case identities."""
    avg = lambda key: mean(float(row[key]) for row in rows) if rows else 0.0
    return {
        "case_count": len(rows),
        "hit_at_1": avg("hit_at_1"),
        f"hit_at_{top_k}": avg("hit_at_k"),
        f"mrr_at_{top_k}": avg("rr"),
        f"recall_at_{top_k}": avg("recall_at_k"),
        f"all_gold_at_{top_k}": avg("all_gold_at_k"),
        f"ndcg_at_{top_k}": avg("ndcg_at_k"),
        "bad_case_count": sum(1 for row in rows if not row["hit_at_k"]),
        "bad_case_ids": [row["id"] for row in rows if not row["hit_at_k"]],
        "incomplete_case_count": sum(1 for row in rows if not row["all_gold_at_k"]),
        "incomplete_case_ids": [row["id"] for row in rows if not row["all_gold_at_k"]],
    }


def _group(rows: list[dict], key: str, top_k: int) -> dict[str, dict]:
    """Aggregate rows by one portable case dimension."""
    values: dict[str, list[dict]] = {}
    for row in rows:
        values.setdefault(str(row[key]), []).append(row)
    return {name: _aggregate(group, top_k) for name, group in sorted(values.items())}


def evaluate_suite(suite: MultiRepoSuite, config: dict) -> dict:
    """Build one index per repository and aggregate source-grounded metrics."""
    rows: list[dict] = []
    efficiency_by_repo: dict[str, dict] = {}
    repository_metadata: list[dict] = []
    for repo in suite.repositories:
        repo_cases = [case for case in suite.cases if case.repo_id == repo.id]
        retrieval_config = RetrievalEvalConfig(repo_path=repo.root, **config)
        pipeline = PreparedRetrievalPipeline.build(retrieval_config)
        result = evaluate_retrieval(pipeline, [
            EvalCase(case.id, case.query, list(case.gold_files), case.task_type, case.difficulty)
            for case in repo_cases
        ])
        case_map = {case.id: case for case in repo_cases}
        for row in result["cases"]:
            case = case_map[row["id"]]
            rows.append({"repo_id": repo.id, **row, "gold_files": list(case.gold_files)})
        efficiency_by_repo[repo.id] = result["efficiency"]
        repository_metadata.append({
            "id": repo.id,
            "path": repo.relative_path,
            "remote": repo.remote,
            "commit": repo.commit,
            "license": repo.license,
            "license_url": repo.license_url,
            "purpose": repo.purpose,
            "fingerprint": repo_fingerprint(pipeline.files, repo.root),
        })
    top_k = int(config["top_k"])
    metrics = _aggregate(rows, top_k)
    quality_score = 100 * (
        0.375 * metrics[f"mrr_at_{top_k}"]
        + 0.3125 * metrics[f"recall_at_{top_k}"]
        + 0.1875 * metrics["hit_at_1"]
        + 0.125 * metrics[f"ndcg_at_{top_k}"]
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    context_sizes = [float(row["context_chars"]) for row in rows]
    return {
        "schema_version": 1,
        "benchmark": "real_multirepo_source_retrieval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": suite.evalset_name, "sha256": suite.evalset_sha256, "case_count": len(rows)},
        "manifest": {"name": suite.manifest_name, "sha256": suite.manifest_sha256, "schema_version": 1},
        "repositories": repository_metadata,
        "config": dict(config),
        "retriever": RETRIEVER_NAME,
        "quality_score": quality_score,
        "metrics": metrics,
        "by_repo": _group(rows, "repo_id", top_k),
        "by_task_type": _group(rows, "task_type", top_k),
        "by_difficulty": _group(rows, "difficulty", top_k),
        "efficiency": {
            "file_count": sum(value["file_count"] for value in efficiency_by_repo.values()),
            "chunk_count": sum(value["chunk_count"] for value in efficiency_by_repo.values()),
            "index_build_ms": sum(value["index_build_ms"] for value in efficiency_by_repo.values()),
            "query_latency_ms_p50": _percentile(latencies, 0.50),
            "query_latency_ms_p95": _percentile(latencies, 0.95),
            "context_chars_avg": mean(context_sizes) if context_sizes else 0.0,
            "context_chars_p95": _percentile(context_sizes, 0.95),
            "by_repo": efficiency_by_repo,
        },
        "cases": rows,
    }


def _raw(payload: dict, dotted: str):
    """Read a nested scorecard value, returning None when absent."""
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def compare_multirepo_scorecards(current: dict, baseline: dict, policy: dict) -> GateResult:
    """Check strict experiment compatibility and configured regression thresholds."""
    fields = (
        "schema_version", "benchmark", "dataset.sha256", "dataset.case_count",
        "manifest.sha256", "manifest.schema_version", "repositories", "config", "retriever",
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
                f"{metric}: baseline={before:.6f}, current={after:.6f}, drop={before-after:.6f}, allowed={float(allowed):.6f}"
            )
    for metric, allowed in (policy.get("max_ratio") or {}).items():
        before, after = float(_raw(baseline, metric)), float(_raw(current, metric))
        ratio = after / before if before > 0 else (1.0 if after <= 0 else float("inf"))
        if ratio > float(allowed) + 1e-12:
            failures.append(
                f"{metric}: baseline={before:.6f}, current={after:.6f}, ratio={ratio:.3f}, allowed={float(allowed):.3f}"
            )
    return GateResult(True, not failures, tuple(failures))


def gate_exit_code(result: GateResult) -> int:
    """Map compatible pass/regression/incompatibility to process exit codes 0/1/2."""
    if not result.compatible:
        return 2
    return 0 if result.passed else 1
