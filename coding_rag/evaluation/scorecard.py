"""Build a portable capability scorecard from deterministic evaluations."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import RETRIEVER_NAME, RetrievalEvalConfig, dataset_sha256, repo_fingerprint


def _git(repo: Path, *args: str) -> str:
    """Run a read-only Git query and return unknown when unavailable."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_capability_scorecard(
    config: RetrievalEvalConfig,
    evalset: Path,
    agent_evalset: Path,
    pipeline,
    retrieval: dict,
    agent: dict | None = None,
) -> dict:
    """Combine retrieval, efficiency, agent, and experiment metadata."""
    agent_summary = agent or {}
    total = int(agent_summary.get("total") or 0)
    fixture = agent_summary.get("fixture_diagnostic") or {}
    agent_metrics = {
        "evaluation_mode": "offline_contract",
        "case_count": total,
        "offline_contract": {
            "expected_tool_coverage": float(agent_summary.get("avg_expected_tool_coverage") or 0.0),
            "section_coverage": float(agent_summary.get("avg_section_coverage") or 0.0),
            "trace_complete_rate": (float(agent_summary.get("trace_complete_count") or 0) / total) if total else 0.0,
            "safe_tool_interface_rate": (float(agent_summary.get("safe_tool_interface_count") or 0) / total) if total else 0.0,
        },
        "fixture_diagnostic": {
            "candidate_k": int(fixture.get("candidate_k") or 0),
            "candidate_hit_at_5": float(fixture.get("candidate_hit_at_5") or 0.0),
            "candidate_recall_at_5": float(fixture.get("candidate_recall_at_5") or 0.0),
            "candidate_mrr_at_5": float(fixture.get("candidate_mrr_at_5") or 0.0),
            "discriminative": bool(fixture.get("discriminative")),
            "negative_file_count": int(fixture.get("negative_file_count") or 0),
            "empty_gold_count": int(fixture.get("empty_gold_count") or 0),
            "gold_file_count": int(fixture.get("gold_file_count") or 0),
            "available_gold_count": int(fixture.get("available_gold_count") or 0),
            "missing_gold_count": int(fixture.get("missing_gold_count") or 0),
            "cases_with_missing_gold": fixture.get("cases_with_missing_gold") or [],
            "dataset_coverage": float(fixture.get("dataset_coverage") or 0.0),
            "by_repo": fixture.get("by_repo") or {},
        },
    }
    top_k = config.top_k
    metrics = retrieval["metrics"]
    overall = 100 * (
        0.375 * metrics[f"mrr_at_{top_k}"]
        + 0.3125 * metrics[f"recall_at_{top_k}"]
        + 0.1875 * metrics["hit_at_1"]
        + 0.125 * metrics[f"ndcg_at_{top_k}"]
    )
    root = config.repo_path.resolve()
    return {
        "schema_version": 2,
        "experiment": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(_git(root, "status", "--porcelain")),
            "python_version": platform.python_version(),
            "platform": platform.system().casefold(),
            "datasets": {
                "retrieval": {
                    "name": evalset.name,
                    "sha256": dataset_sha256(evalset),
                    "case_count": len(retrieval["cases"]),
                },
                "agent": {
                    "name": agent_evalset.name,
                    "sha256": dataset_sha256(agent_evalset),
                    "case_count": total,
                },
            },
            "repo_fingerprint": repo_fingerprint(pipeline.files, root),
            "config": config.public_dict(),
            "retriever": {
                "name": RETRIEVER_NAME,
                "weights": {
                    "lexical_hybrid": (1.0 - config.path_weight) * (1.0 - config.graph_weight),
                    "path_identifier": config.path_weight * (1.0 - config.graph_weight),
                    "dependency_graph": config.graph_weight,
                },
                "structural_ranker": "personalized-dependency-diffusion-v1" if config.graph_weight else None,
            },
            "seed": None,
        },
        "retrieval": metrics,
        "efficiency": retrieval["efficiency"],
        "agent": agent_metrics,
        "overall_score": overall,
        "cases": retrieval["cases"],
    }
