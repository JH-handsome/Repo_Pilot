"""Measure explicit-call-path tool use through the real Agent graph, without an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.call_path_runtime import evaluate_case
from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.repository.search_index import RETRIEVER_VERSION


DEFAULT_EVALSETS = (
    "datasets/eval/multirepo_call_path_holdout.jsonl",
    "datasets/eval/multirepo_call_path_transfer.jsonl",
)
CONFIG = {
    "top_k": 5, "chunk_size": 40, "overlap": 5, "recall_window": 2,
    "max_recall_results": None, "min_final_score": 0.0,
    "graph_weight": 0.0, "path_weight": 0.25,
    "exclude_roots": ["data", "datasets", "artifacts"],
}


def evaluate(workspace: Path, manifest: Path, evalsets: tuple[Path, ...]) -> dict:
    """Run fixed cases; the policy sees only query and graph observations."""
    datasets = []
    cases = []
    repositories = None
    seen_ids = set()
    for evalset in evalsets:
        suite = load_and_validate_suite(workspace, manifest, evalset)
        datasets.append({"name": suite.evalset_name, "sha256": suite.evalset_sha256})
        repo_identity = [{"id": repo.id, "commit": repo.commit, "remote": repo.remote}
                         for repo in suite.repositories]
        if repositories is not None and repositories != repo_identity:
            raise ValueError("evaluation sets resolved to different pinned repositories")
        repositories = repo_identity
        roots = {repo.id: repo.root for repo in suite.repositories}
        for case in suite.cases:
            if case.id in seen_ids:
                raise ValueError(f"duplicate case id across evaluation sets: {case.id}")
            seen_ids.add(case.id)
            if case.task_type != "call_path" or len(case.gold_files) < 2:
                raise ValueError(f"call-path case needs at least two gold files: {case.id}")
            cases.append(evaluate_case(case, roots[case.repo_id], CONFIG))
    if not cases:
        raise ValueError("no call-path cases")
    return {
        "schema_version": 1,
        "benchmark": "explicit_call_path_runtime_contract",
        "datasets": datasets,
        "manifest": {"name": suite.manifest_name, "sha256": suite.manifest_sha256},
        "repositories": repositories,
        "config": CONFIG,
        "retriever": RETRIEVER_VERSION,
        "policy": "first-query-callable-search-definition-unique-edge-read-both-v3",
        "evaluation_mode": "offline_scripted_runtime_not_model_capability",
        "metrics": {
            "case_count": len(cases),
            "contract_success_rate": mean(float(row["contract_success"]) for row in cases),
            "all_gold_read_rate": mean(float(row["all_gold_read"]) for row in cases),
            "source_call_observed_rate": mean(float(row["source_call_observed"]) for row in cases),
            "target_definition_observed_rate": mean(float(row["target_definition_observed"]) for row in cases),
            "all_gold_cited_rate": mean(float(row["all_gold_cited"]) for row in cases),
            "citation_valid_rate": mean(float(row["citation_valid"]) for row in cases),
            "terminal_success_rate": mean(float(row["terminal_success"]) for row in cases),
        },
        "cases": cases,
    }


def compare(current: dict, baseline: dict) -> list[str]:
    """Reject identity drift and any regression in evidence-backed contract metrics."""
    failures = []
    for key in ("schema_version", "benchmark", "datasets", "manifest", "repositories",
                "config", "retriever", "policy", "evaluation_mode"):
        if current.get(key) != baseline.get(key):
            failures.append(f"incompatible {key}")
    if current.get("metrics", {}).get("case_count") != baseline.get("metrics", {}).get("case_count"):
        failures.append("incompatible case_count")
    for key in ("contract_success_rate", "all_gold_read_rate", "source_call_observed_rate",
                "target_definition_observed_rate", "all_gold_cited_rate", "citation_valid_rate"):
        if current["metrics"][key] < baseline["metrics"][key]:
            failures.append(f"{key} regression")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Run the portable offline contract and optionally enforce a fixed baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets/eval/multirepo_manifest.json")
    parser.add_argument("--evalset", action="append", type=Path)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "benchmarks/call-path-runtime-baseline-v4.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/call-path-runtime-scorecard.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        evalsets = tuple(args.evalset or (PROJECT_ROOT / value for value in DEFAULT_EVALSETS))
        scorecard = evaluate(args.workspace_root, args.manifest, evalsets)
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
        print(f"Call-path runtime evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
