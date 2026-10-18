"""Diagnose query-sensitive import-bound call edges without changing retrieval ranking."""

from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.evaluation.retrieval import (
    PreparedRetrievalPipeline, RetrievalEvalConfig, dataset_sha256, repo_fingerprint,
)
from coding_rag.repository.call_path import PythonCallPathIndex
from coding_rag.repository.call_edge_candidates import (
    AUTO_SOURCE_CANDIDATE_VERSION, CALL_EDGE_CANDIDATE_VERSION,
    candidate_call_edges, query_only_call_edges,
)


def parse_args() -> argparse.Namespace:
    """Parse pinned evaluation inputs and optional call-line ranking weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("data/repos/django"))
    parser.add_argument("--manifest", type=Path, help="validate pinned Git checkouts and gold paths")
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--evalset", action="append", type=Path, required=True)
    parser.add_argument("--source-files", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--anchors", type=Path, help="evaluate separately with source path and symbol supplied")
    parser.add_argument("--auto-anchor", action="store_true", help="diagnose first retrieved non-test file and its top function as an unverified source")
    parser.add_argument("--line-weight", type=float, default=0.0)
    parser.add_argument("--out", type=Path, help="write a reproducible JSON scorecard")
    return parser.parse_args()


def main() -> int:
    """Score candidate edges separately from the unchanged retrieval baseline."""
    args = parse_args()
    if (not 1 <= args.source_files <= 10 or not 1 <= args.top_candidates <= 200
            or not isfinite(args.line_weight) or args.line_weight < 0):
        raise SystemExit("source-files must be 1..10, top-candidates 1..200, line-weight nonnegative")
    if args.auto_anchor and args.anchors:
        raise SystemExit("--auto-anchor and --anchors are mutually exclusive")
    grounding = {}
    if args.grounding:
        payload = json.loads(args.grounding.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
            raise SystemExit("grounding must be a schema v1 object with cases")
        grounding = {item["id"]: item for item in payload["cases"]}
        if len(grounding) != len(payload["cases"]):
            raise SystemExit("grounding case IDs must be unique")
    anchors = {}
    if args.anchors:
        payload = json.loads(args.anchors.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
            raise SystemExit("anchors must be a schema v1 object with cases")
        anchors = {item["id"]: item for item in payload["cases"]}
        if len(anchors) != len(payload["cases"]):
            raise SystemExit("anchor case IDs must be unique")
    repo_root = args.repo.resolve()
    suites = []
    if args.manifest:
        for evalset in args.evalset:
            suite = load_and_validate_suite(args.workspace_root, args.manifest, evalset)
            repo_ids = {case.repo_id for case in suite.cases}
            if len(repo_ids) != 1:
                raise SystemExit(f"one repository per evaluation set is required: {evalset}")
            expected = next(repo for repo in suite.repositories if repo.id in repo_ids)
            if expected.root.resolve() != repo_root:
                raise SystemExit(f"repository path does not match manifest: {evalset}")
            suites.append(suite)
    seen_cases: set[str] = set()
    case_gold: dict[str, list[str]] = {}
    for evalset in args.evalset:
        for line in evalset.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if case["id"] in seen_cases:
                raise SystemExit(f"duplicate case ID across evaluation sets: {case['id']}")
            seen_cases.add(case["id"])
            if args.anchors:
                case_gold[case["id"]] = case["gold_files"]
    if args.case_id and (len(set(args.case_id)) != len(args.case_id)
                         or not set(args.case_id) <= seen_cases):
        raise SystemExit("case-id values must be unique and present in evaluation sets")
    if grounding and not args.case_id and set(grounding) != seen_cases:
        raise SystemExit("grounding IDs must exactly match evaluation case IDs")
    if args.anchors:
        if not args.case_id and set(anchors) != seen_cases:
            raise SystemExit("anchor IDs must exactly match evaluation case IDs")
        for case_id, anchor in anchors.items():
            if (case_id not in case_gold or not isinstance(anchor.get("source_symbol"), str)
                    or not anchor["source_symbol"].strip()
                    or anchor.get("source_path") != case_gold[case_id][0]):
                raise SystemExit(f"invalid source anchor: {case_id}")
    pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(repo_path=repo_root))
    call_index = PythonCallPathIndex(pipeline.chunks, pipeline.retriever.repo_index)
    scored: list[dict] = []
    for evalset in args.evalset:
        for line in evalset.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if args.case_id and case["id"] not in args.case_id:
                continue
            if grounding and (case["id"] not in grounding or len(case["gold_files"]) != 2):
                raise SystemExit(f"missing grounding or non-pair gold: {case['id']}")
            result = pipeline.query(case["query"])
            baseline = [
                found.chunk.file_path.resolve().relative_to(repo_root).as_posix()
                for found in result["final"]
            ]
            anchor = anchors.get(case["id"])
            if args.anchors and anchor is None:
                raise SystemExit(f"missing source anchor: {case['id']}")
            auto_anchor_abstention_reason = None
            if args.auto_anchor:
                anchor, candidates, auto_anchor_abstention_reason = query_only_call_edges(
                    pipeline.retriever, call_index, repo_root,
                    case["query"], [found.chunk.file_path for found in result["final"]],
                    line_weight=args.line_weight,
                )
            else:
                source_paths = (
                    [repo_root / anchor["source_path"]] if anchor else
                    [found.chunk.file_path for found in result["final"]]
                )
                candidates = candidate_call_edges(
                    pipeline.retriever, call_index, pipeline.config.repo_path, case["query"],
                    source_paths,
                    source_files=1 if anchor else args.source_files, line_weight=args.line_weight,
                    source_symbol=anchor["source_symbol"] if anchor else None,
                )
            gold = grounding.get(case["id"])
            edge_rank = next(
                (rank for rank, candidate in enumerate(candidates, 1)
                 if gold and candidate["source_symbol"] == gold["source_symbol"]
                 and candidate["target_symbol"] == gold["target_symbol"]
                 and candidate["source_path"] == case["gold_files"][0]
                 and candidate["target_path"] == case["gold_files"][1]),
                None,
            )
            gold_targets = set(case["gold_files"]) - set(baseline)
            missing_target_rank = next(
                (rank for rank, candidate in enumerate(candidates, 1)
                 if candidate["source_path"] == case["gold_files"][0]
                 and candidate["target_path"] in gold_targets),
                None,
            )
            row = {
                "id": case["id"],
                "baseline": baseline,
                "gold": case["gold_files"],
                "expect_no_cross_file": case.get("expect_no_cross_file", False),
                "source_anchor": anchor,
                "auto_anchor_abstention_reason": auto_anchor_abstention_reason,
                "exact_edge_rank": edge_rank,
                "missing_target_candidate_rank": missing_target_rank,
                "candidate_count": len(candidates),
                "candidates": candidates[:args.top_candidates],
            }
            scored.append(row)
            print(json.dumps(row, ensure_ascii=False))
    negatives = [row for row in scored if row["expect_no_cross_file"]]
    summary = {
        "case_count": len(scored),
        "anchored_source_case_count": sum(row["source_anchor"] is not None for row in scored),
        "auto_anchor_source_file_match_count": sum(
            row["source_anchor"] is not None
            and row["source_anchor"]["source_path"] == row["gold"][0]
            for row in scored
        ) if args.auto_anchor else None,
        "negative_case_count": len(negatives),
        "negative_suggestion_at_1": sum(row["candidate_count"] > 0 for row in negatives),
        "negative_abstention_count": sum(row["candidate_count"] == 0 for row in negatives),
    }
    if grounding:
        positives = [row for row in scored if not row["expect_no_cross_file"]]
        summary.update({
                "grounded_case_count": len(positives),
                "baseline_all_gold_at_5": sum(set(row["gold"]) <= set(row["baseline"]) for row in positives),
                "missing_case_count": sum(not set(row["gold"]) <= set(row["baseline"]) for row in positives),
                "exact_edge_at_1": sum(row["exact_edge_rank"] == 1 for row in positives),
                "exact_edge_at_3": sum(row["exact_edge_rank"] is not None and row["exact_edge_rank"] <= 3 for row in positives),
                "missing_target_at_1": sum(row["missing_target_candidate_rank"] == 1 for row in positives),
                "missing_target_at_3": sum(row["missing_target_candidate_rank"] is not None and row["missing_target_candidate_rank"] <= 3 for row in positives),
        })
    print(json.dumps({"summary": summary, "line_weight": args.line_weight}, ensure_ascii=False))
    if args.out:
        scorecard = {
            "schema_version": 1,
            "candidate_version": CALL_EDGE_CANDIDATE_VERSION,
            "auto_source_version": AUTO_SOURCE_CANDIDATE_VERSION if args.auto_anchor else None,
            "repository_fingerprint": repo_fingerprint(pipeline.files, repo_root),
            "repository_identity": (
                {"status": "pinned_validated", "manifest_sha256": suites[0].manifest_sha256,
                 "repo_id": suites[0].cases[0].repo_id,
                 "commit": next(repo.commit for repo in suites[0].repositories
                                if repo.id == suites[0].cases[0].repo_id)}
                if suites else {"status": "unverified"}
            ),
            "datasets": [
                {"name": path.name, "sha256": dataset_sha256(path)} for path in args.evalset
            ],
            "grounding_sha256": dataset_sha256(args.grounding) if args.grounding else None,
            "anchors_sha256": dataset_sha256(args.anchors) if args.anchors else None,
            "config": {"top_k": 5, "source_files": 1 if args.anchors or args.auto_anchor else args.source_files,
                       "line_weight": args.line_weight, "top_candidates": args.top_candidates,
                       "case_filter": sorted(args.case_id) if args.case_id else None,
                       "candidate_mode": ("query_only_auto_anchor" if args.auto_anchor else
                                          "source_anchored" if args.anchors else "retrieval_seeded")},
            "summary": summary,
            "cases": scored,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, scorecard)
        print(json.dumps({"scorecard": str(args.out), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
