"""Evaluate source-audited call-path links and near-miss abstention offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query
from coding_rag.evaluation.call_path_runtime import CallPathScriptedPolicy, observed_line
from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.rag.citation_validator import extract_citations
from coding_rag.repository.search_index import RETRIEVER_VERSION
from scripts.call_path_eval import confined_path
from scripts.call_path_runtime_eval import CONFIG


def load_cases(path: Path, roots: dict[str, Path]) -> list[dict]:
    """Validate frozen labels and confined source paths before any graph run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) and payload.get("schema_version") == 1 else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("challenge cases must be a non-empty schema v1 array")
    ids = set()
    outcomes = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("each challenge case needs a non-empty id")
        if row["id"] in ids:
            raise ValueError(f"duplicate challenge id: {row['id']}")
        ids.add(row["id"])
        root = roots.get(row.get("repo_id"))
        if root is None or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"unknown repo or empty query: {row['id']}")
        entry = confined_path(row.get("entry_path"))
        entry_file = root / entry
        if (not entry_file.is_file() or not entry_file.resolve().is_relative_to(root.resolve())
                or not isinstance(row.get("entry_symbol"), str)):
            raise ValueError(f"missing entry source or symbol: {row['id']}")
        outcome = row.get("expected_outcome")
        if outcome not in {"link", "abstain"}:
            raise ValueError(f"invalid expected outcome: {row['id']}")
        outcomes.add(outcome)
        if outcome == "link":
            target = confined_path(row.get("target_path"))
            target_file = root / target
            if (not target_file.is_file() or not target_file.resolve().is_relative_to(root.resolve())
                    or not isinstance(row.get("target_symbol"), str)):
                raise ValueError(f"missing target source or symbol: {row['id']}")
        elif not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"abstention case needs an audited reason: {row['id']}")
    if outcomes != {"link", "abstain"}:
        raise ValueError("challenge must contain both positive and abstention cases")
    return rows


def evaluate_row(row: dict, root: Path) -> dict:
    """Run the unchanged scripted policy; inspect labels only after it finishes."""
    policy = CallPathScriptedPolicy(row["query"])
    run = run_unified_query(row["query"], UnifiedRunConfig(
        repo_path=root, top_k=5, chunk_size=CONFIG["chunk_size"], overlap=CONFIG["overlap"],
        recall_window=CONFIG["recall_window"], max_recall_results=CONFIG["max_recall_results"],
        min_final_score=CONFIG["min_final_score"], graph_weight=CONFIG["graph_weight"],
        path_weight=CONFIG["path_weight"], index_exclude_roots=tuple(CONFIG["exclude_roots"]),
        max_steps=4,
    ), policy)
    tools = [decision.tool for decision in run.decisions if decision.action == "tool"]
    reads = [observation["output"] for observation in run.observations
             if observation.get("tool") == "read_file" and "error" not in observation
             and isinstance(observation.get("output"), dict)]
    citations = extract_citations(run.answer)
    trace_success = any(
        observation.get("tool") == "trace_call_path" and "error" not in observation
        and isinstance(observation.get("output"), dict)
        and isinstance(observation["output"].get("edges"), list)
        for observation in run.observations
    )
    final = next((event for event in reversed(run.trace.get("events", []))
                  if event.get("step") == "final_answer"), {})
    citation_valid = bool(citations) and not (final.get("output_summary") or {}).get("citation_issues", True)
    entry_selected = (policy.entry or {}).get("path") == row["entry_path"]
    target = policy.target or {}
    if row["expected_outcome"] == "link":
        linked = bool(
            entry_selected and target.get("target_path") == row["target_path"]
            and target.get("target_symbol") == row["target_symbol"]
            and tools == ["search_code", "trace_call_path", "read_file", "read_file"]
            and len(reads) == 2 and citation_valid
            and observed_line(reads[0], row["entry_path"], target.get("source_line"))
            and observed_line(reads[1], row["target_path"], target.get("target_start_line"))
            and {citation.path.replace("\\", "/") for citation in citations}
            >= {row["entry_path"], row["target_path"]}
        )
        success = linked
    else:
        linked = bool(target and "read_file" in tools)
        success = bool(entry_selected and trace_success
                       and tools == ["search_code", "trace_call_path"]
                       and not target and not reads and not citations)
    return {
        "id": row["id"], "expected_outcome": row["expected_outcome"],
        "entry_selected": entry_selected, "selected_target_path": target.get("target_path"),
        "selected_target_symbol": target.get("target_symbol"),
        "tool_sequence": tools, "read_paths": [read.get("path") for read in reads],
        "trace_success": trace_success, "citation_valid": citation_valid,
        "linked": linked, "success": success,
    }


def evaluate(workspace: Path, manifest: Path, dataset: Path) -> dict:
    """Validate pinned repositories, then score each fixed label after execution."""
    suite = load_and_validate_suite(
        workspace, manifest, PROJECT_ROOT / "datasets/eval/multirepo_source.jsonl"
    )
    roots = {repo.id: repo.root for repo in suite.repositories}
    rows = load_cases(dataset, roots)
    results = [evaluate_row(row, roots[row["repo_id"]]) for row in rows]
    positives = [item for item in results if item["expected_outcome"] == "link"]
    negatives = [item for item in results if item["expected_outcome"] == "abstain"]
    return {
        "schema_version": 1,
        "benchmark": "pinned_call_path_link_and_abstain_challenge",
        "dataset": {"name": dataset.name, "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "manifest": {"name": suite.manifest_name, "sha256": suite.manifest_sha256},
        "repositories": [{"id": repo.id, "commit": repo.commit, "remote": repo.remote}
                         for repo in suite.repositories],
        "config": CONFIG, "retriever": RETRIEVER_VERSION,
        "policy": "first-query-callable-search-definition-unique-edge-read-both-v3",
        "evaluation_mode": "offline_scripted_runtime_not_model_capability",
        "metrics": {
            "case_count": len(results), "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_link_rate": mean(float(item["success"]) for item in positives),
            "negative_abstain_rate": mean(float(item["success"]) for item in negatives),
            "false_link_count": sum(bool(item["linked"]) for item in negatives),
        },
        "cases": results,
    }


def compare(current: dict, baseline: dict) -> list[str]:
    """Reject benchmark identity drift, lost links, or increased false links."""
    failures = []
    for key in ("schema_version", "benchmark", "dataset", "manifest", "repositories",
                "config", "retriever", "policy", "evaluation_mode"):
        if current.get(key) != baseline.get(key):
            failures.append(f"incompatible {key}")
    for key in ("case_count", "positive_count", "negative_count"):
        if current.get("metrics", {}).get(key) != baseline.get("metrics", {}).get(key):
            failures.append(f"incompatible {key}")
    for key in ("positive_link_rate", "negative_abstain_rate"):
        if current["metrics"][key] < baseline["metrics"][key]:
            failures.append(f"{key} regression")
    if current["metrics"]["false_link_count"] > baseline["metrics"]["false_link_count"]:
        failures.append("false_link_count regression")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Write a challenge scorecard and optionally enforce its fixed v3 baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets/eval/multirepo_manifest.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/eval/call_path_challenge_v1.json")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "benchmarks/call-path-challenge-baseline-v2.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/call-path-challenge-scorecard.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        scorecard = evaluate(args.workspace_root, args.manifest, args.dataset)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, scorecard)
        print(json.dumps(scorecard["metrics"], ensure_ascii=False, indent=2))
        if not args.gate:
            return 0
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        failures = compare(scorecard, baseline)
        print("PASS" if not failures else "FAIL: " + "; ".join(failures))
        return 0 if not failures else (2 if any(item.startswith("incompatible") for item in failures) else 1)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Call-path challenge evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
