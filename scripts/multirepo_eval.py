"""Run and gate the pinned real-source multi-repository retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.multirepo import (
    MultiRepoValidationError,
    compare_multirepo_scorecards,
    evaluate_suite,
    gate_exit_code,
    load_and_validate_suite,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable inputs, retrieval config, and baseline controls."""
    parser = argparse.ArgumentParser(description="Evaluate pinned real source repositories")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--manifest", default="datasets/eval/multirepo_manifest.json")
    parser.add_argument("--evalset", default="datasets/eval/multirepo_source.jsonl")
    parser.add_argument("--baseline", default="benchmarks/multirepo-retrieval-baseline-v4.json")
    parser.add_argument("--policy", default="benchmarks/multirepo-retrieval-gate-v1.json")
    parser.add_argument("--out", default="artifacts/eval/multirepo-scorecard.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--recall-window", type=int, default=2)
    parser.add_argument("--graph-weight", type=float, default=0.0)
    parser.add_argument("--path-weight", type=float, default=0.25)
    parser.add_argument("--exclude-root", action="append", dest="exclude_roots")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    """Write a formatted JSON artifact, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, payload)


def main(argv: list[str] | None = None) -> int:
    """Validate, evaluate, and optionally update or enforce the independent gate."""
    args = parse_args(argv)
    if args.gate and args.update_baseline:
        print("Evaluation error: --gate and --update-baseline are mutually exclusive")
        return 2
    try:
        suite = load_and_validate_suite(args.workspace_root, args.manifest, args.evalset)
        config = {
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "recall_window": args.recall_window,
            "max_recall_results": None,
            "min_final_score": 0.0,
            "graph_weight": args.graph_weight,
            "path_weight": args.path_weight,
            "exclude_roots": tuple(args.exclude_roots or ("data", "datasets", "artifacts")),
        }
        scorecard = evaluate_suite(suite, config)
        # JSON represents immutable config sequences as arrays, including before comparison.
        scorecard["config"]["exclude_roots"] = list(scorecard["config"]["exclude_roots"])
        _write_json(Path(args.out), scorecard)
        print(json.dumps({
            "quality_score": scorecard["quality_score"],
            "metrics": scorecard["metrics"],
            "by_repo": scorecard["by_repo"],
            "by_task_type": scorecard["by_task_type"],
            "by_difficulty": scorecard["by_difficulty"],
            "efficiency": scorecard["efficiency"],
        }, ensure_ascii=False, indent=2))
        baseline_path = Path(args.baseline)
        if args.update_baseline:
            _write_json(baseline_path, scorecard)
            print(f"Baseline updated: {baseline_path}")
            return 0
        if not args.gate:
            return 0
        if not baseline_path.is_file():
            print(f"Baseline does not exist: {baseline_path}")
            return 2
        policy_path = Path(args.policy)
        if not policy_path.is_file():
            print(f"Policy does not exist: {policy_path}")
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        result = compare_multirepo_scorecards(scorecard, baseline, policy)
        print("PASS" if result.passed else "FAIL")
        for failure in result.failures:
            print(f"- {failure}")
        return gate_exit_code(result)
    except MultiRepoValidationError as error:
        print("Validation failed:")
        for finding in error.errors:
            print(f"- {finding}")
        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Evaluation error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
