"""Run the unified RepoPilot capability scorecard and optional gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.gate import DEFAULT_POLICY, compare_scorecards
from coding_rag.evaluation.retrieval import PreparedRetrievalPipeline, RetrievalEvalConfig, evaluate_retrieval, load_evalset, validate_evalset
from coding_rag.evaluation.scorecard import build_capability_scorecard
from scripts.agent_eval import evaluate_cases


def _write_json(path: Path, payload: dict) -> None:
    """Write a formatted JSON artifact after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, payload)


def parse_args() -> argparse.Namespace:
    """Parse scorecard, baseline, policy, and retrieval configuration options."""
    parser = argparse.ArgumentParser(description="Build and gate the RepoPilot capability scorecard")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--retrieval-evalset", default="datasets/eval/trace_v2.jsonl")
    parser.add_argument("--agent-evalset", default="datasets/eval/agent_plan_cases.json")
    parser.add_argument("--baseline", default="benchmarks/retrieval-baseline-v5.json")
    parser.add_argument("--policy", default="benchmarks/capability-gate-v2.json")
    parser.add_argument("--out", default="artifacts/eval/capability-scorecard.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--recall-window", type=int, default=2)
    parser.add_argument("--graph-weight", type=float, default=0.0)
    parser.add_argument("--path-weight", type=float, default=0.25)
    parser.add_argument("--exclude-root", action="append", dest="exclude_roots")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Build a scorecard and optionally update or enforce a compatible baseline."""
    args = parse_args()
    repo = Path(args.repo).resolve()
    evalset = Path(args.retrieval_evalset)
    agent_evalset = Path(args.agent_evalset)
    try:
        validation = validate_evalset(repo, evalset)
        if not validation.valid:
            print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
            return 2
        config = RetrievalEvalConfig(
            repo_path=repo,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recall_window=args.recall_window,
            graph_weight=args.graph_weight,
            path_weight=args.path_weight,
            exclude_roots=tuple(args.exclude_roots or ("data", "datasets", "artifacts")),
        )
        pipeline = PreparedRetrievalPipeline.build(config)
        retrieval = evaluate_retrieval(pipeline, load_evalset(evalset))
        _, agent = evaluate_cases(agent_evalset)
        scorecard = build_capability_scorecard(config, evalset, agent_evalset, pipeline, retrieval, agent)
        _write_json(Path(args.out), scorecard)
        print(json.dumps({
            "overall_score": scorecard["overall_score"],
            "retrieval": scorecard["retrieval"],
            "efficiency": scorecard["efficiency"],
            "agent": scorecard["agent"],
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
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else DEFAULT_POLICY
        result = compare_scorecards(scorecard, json.loads(baseline_path.read_text(encoding="utf-8")), policy)
        print("PASS" if result.passed else "FAIL")
        for failure in result.failures:
            print(f"- {failure}")
        if not result.compatible:
            return 2
        return 0 if result.passed else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Evaluation error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
