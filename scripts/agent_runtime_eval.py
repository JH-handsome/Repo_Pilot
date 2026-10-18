"""Run and gate the deterministic real-source Agent runtime contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.agent_runtime import (
    compare_agent_runtime_scorecards,
    evaluate_agent_runtime_suite,
    gate_exit_code,
)
from coding_rag.evaluation.multirepo import MultiRepoValidationError, load_and_validate_suite


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable suite, retrieval, output, and gate controls."""
    parser = argparse.ArgumentParser(description="Evaluate the real-source Agent runtime contract")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--manifest", default="datasets/eval/multirepo_manifest.json")
    parser.add_argument("--evalset", default="datasets/eval/multirepo_source.jsonl")
    parser.add_argument("--baseline", default="benchmarks/agent-runtime-baseline-v4.json")
    parser.add_argument("--policy", default="benchmarks/agent-runtime-gate-v1.json")
    parser.add_argument("--out", default="artifacts/eval/agent-runtime-scorecard.json")
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


def _write(path: Path, payload: dict) -> None:
    """Write a formatted JSON artifact after creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, payload)


def main(argv: list[str] | None = None) -> int:
    """Validate, run, optionally update, and gate the runtime contract."""
    args = parse_args(argv)
    if args.gate and args.update_baseline:
        print("Evaluation error: --gate and --update-baseline are mutually exclusive")
        return 2
    if args.top_k != 5:
        print("Evaluation error: this contract fixes top_k at 5")
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
            "exclude_roots": list(args.exclude_roots or ("data", "datasets", "artifacts")),
        }
        scorecard = evaluate_agent_runtime_suite(suite, config)
        _write(Path(args.out), scorecard)
        print(json.dumps({
            "contract_score": scorecard["contract_score"],
            "corpus": scorecard["corpus"],
            "metrics": scorecard["metrics"],
            "by_repo": scorecard["by_repo"],
            "by_task_type": scorecard["by_task_type"],
            "by_difficulty": scorecard["by_difficulty"],
            "warmups": scorecard["warmups"],
        }, ensure_ascii=False, indent=2))
        baseline_path = Path(args.baseline)
        if args.update_baseline:
            _write(baseline_path, scorecard)
            print(f"Baseline updated: {baseline_path}")
            return 0
        if not args.gate:
            return 0
        policy_path = Path(args.policy)
        if not baseline_path.is_file() or not policy_path.is_file():
            print("Baseline or policy does not exist")
            return 2
        result = compare_agent_runtime_scorecards(
            scorecard,
            json.loads(baseline_path.read_text(encoding="utf-8")),
            json.loads(policy_path.read_text(encoding="utf-8")),
        )
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
