"""Measure whether natural-language retrieval exposes audited source entrypoints.

Development and transfer suites are offline diagnostics, not model-answer or
call-edge success scores. The transfer queries were used in earlier file-retrieval
evaluation and are not blind end-to-end tasks.
"""

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

from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.evaluation.retrieval import PreparedRetrievalPipeline, RetrievalEvalConfig
from coding_rag.repository.search_index import RETRIEVER_VERSION
from coding_rag.repository.symbol_locator import (
    SYMBOL_LOCATOR_VERSION,
    locate_file_symbols,
    locate_retrieved_files,
)


GROUPS = (
    ("aider", "datasets/eval/aider_manifest_v1.json", (
        "datasets/eval/aider_call_edge_positive_v1.jsonl",
        "datasets/eval/aider_call_edge_negative_v1.jsonl",
        "datasets/eval/aider_call_edge_near_negative_v1.jsonl",
    ), "datasets/eval/aider_call_edge_source_anchors_v1.json"),
    ("django", "datasets/eval/multirepo_manifest.json", (
        "datasets/eval/multirepo_call_path_generalization_holdout_v3.jsonl",
        "datasets/eval/multirepo_call_edge_negative_v1.jsonl",
    ), "datasets/eval/django_call_edge_source_anchors_v1.json"),
)
TRANSFER_GROUPS = (
    ("django", "datasets/eval/multirepo_manifest.json", (
        "datasets/eval/source_discovery_transfer_v1.jsonl",
    ), "datasets/eval/source_discovery_transfer_django_anchors_v1.json"),
    ("leetcode-python", "datasets/eval/multirepo_manifest.json", (
        "datasets/eval/source_discovery_transfer_v1.jsonl",
    ), "datasets/eval/source_discovery_transfer_leetcode_anchors_v1.json"),
)


def source_span(pipeline: PreparedRetrievalPipeline, path: str, symbol: str) -> tuple[int, int]:
    """Require an unambiguous indexed function or method matching the audit anchor."""
    matches = []
    for file_index in pipeline.retriever.repo_index.files.values():
        if file_index.path.resolve().relative_to(pipeline.config.repo_path.resolve()).as_posix() != path:
            continue
        for record in file_index.symbols:
            qualified = f"{record.parent}.{record.name}" if record.parent else record.name
            if qualified == symbol and record.kind in {"function", "method", "async_function"}:
                matches.append((record.start_line, record.end_line))
    if len(matches) != 1:
        raise ValueError(f"source anchor {path}:{symbol} has {len(matches)} indexed matches")
    return matches[0]


def rank_source(results: list, root: Path, path: str, span: tuple[int, int]) -> tuple[int | None, int | None, int | None]:
    """Separate file, function-body overlap, and actual definition-line evidence."""
    file_rank = span_rank = definition_rank = None
    for rank, result in enumerate(results, 1):
        chunk = result.chunk
        relative = chunk.file_path.resolve().relative_to(root.resolve()).as_posix()
        if relative != path:
            continue
        if file_rank is None:
            file_rank = rank
        if span_rank is None and chunk.start_line <= span[1] and chunk.end_line >= span[0]:
            span_rank = rank
        if definition_rank is None and chunk.start_line <= span[0] <= chunk.end_line:
            definition_rank = rank
    return file_rank, span_rank, definition_rank


def joint_pair_rank(groups: list[dict], path: str, symbol: str) -> int | None:
    """Count candidate definitions in the order Codex would review them."""
    rank = 0
    for group in groups:
        for candidate in group["candidates"]:
            rank += 1
            if group["path"] == path and candidate["name"] == symbol:
                return rank
    return None


def summarize(rows: list[dict]) -> dict:
    """Keep file discovery separate from evidence inside the source function."""
    return {
        "cases": len(rows),
        "source_file_at_5": mean(row["file_rank"] is not None and row["file_rank"] <= 5 for row in rows),
        "source_file_at_20": mean(row["file_rank"] is not None for row in rows),
        "source_function_at_5": mean(row["function_span_rank"] is not None and row["function_span_rank"] <= 5 for row in rows),
        "source_function_at_20": mean(row["function_span_rank"] is not None for row in rows),
        "source_definition_at_5": mean(row["definition_rank"] is not None and row["definition_rank"] <= 5 for row in rows),
        "source_definition_at_20": mean(row["definition_rank"] is not None for row in rows),
        "file_local_symbol_at_1": mean(row.get("file_local_symbol_rank") == 1 for row in rows),
        "file_local_symbol_at_3": mean(row.get("file_local_symbol_rank") is not None
                                       and row["file_local_symbol_rank"] <= 3 for row in rows),
        "file_local_symbol_at_5": mean(row.get("file_local_symbol_rank") is not None
                                       and row["file_local_symbol_rank"] <= 5 for row in rows),
        "two_stage_reachable_at_5_files_3_symbols": mean(
            row["file_rank"] is not None and row["file_rank"] <= 5
            and row.get("file_local_symbol_rank") is not None
            and row["file_local_symbol_rank"] <= 3 for row in rows
        ),
        "joint_candidate_hit_at_5_files_3_symbols": mean(
            row.get("joint_candidate_hit", False) for row in rows
        ),
        "joint_first_pair_accuracy": mean(row.get("joint_first_pair_correct", False) for row in rows),
        "joint_pair_at_3": mean(row.get("joint_pair_rank") is not None
                                and row["joint_pair_rank"] <= 3 for row in rows),
        "joint_pair_at_5": mean(row.get("joint_pair_rank") is not None
                                and row["joint_pair_rank"] <= 5 for row in rows),
        "joint_pair_at_10": mean(row.get("joint_pair_rank") is not None
                                 and row["joint_pair_rank"] <= 10 for row in rows),
    }


def evaluate(workspace: Path, *, suite: str = "development") -> dict:
    """Run the same retrieval pipeline on pinned positives and local/same-file negatives."""
    if suite not in {"development", "transfer"}:
        raise ValueError("suite must be development or transfer")
    top_k = 20
    rows = []
    identities = []
    groups = GROUPS if suite == "development" else TRANSFER_GROUPS
    for repo_id, manifest_name, evalsets, anchors_name in groups:
        manifest = PROJECT_ROOT / manifest_name
        anchors_path = PROJECT_ROOT / anchors_name
        anchors_data = json.loads(anchors_path.read_text(encoding="utf-8"))
        if anchors_data.get("schema_version") != 1 or not isinstance(anchors_data.get("cases"), list):
            raise ValueError(f"invalid anchors: {anchors_name}")
        anchors = {item["id"]: item for item in anchors_data["cases"]}
        if len(anchors) != len(anchors_data["cases"]):
            raise ValueError(f"duplicate source anchor in {anchors_name}")
        cases = []
        root = None
        for evalset_name in evalsets:
            loaded_suite = load_and_validate_suite(workspace, manifest, PROJECT_ROOT / evalset_name)
            repo = next(item for item in loaded_suite.repositories if item.id == repo_id)
            root = repo.root
            identities.append({"repo": repo_id, "commit": repo.commit, "manifest": loaded_suite.manifest_sha256,
                               "evalset": loaded_suite.evalset_sha256, "anchors": anchors_path.name,
                               "anchors_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest()})
            cases.extend(case for case in loaded_suite.cases if case.repo_id == repo_id)
        assert root is not None
        if set(anchors) != {case.id for case in cases}:
            raise ValueError(f"source anchors and cases disagree for {repo_id}")
        pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(repo_path=root, top_k=top_k))
        for case in cases:
            anchor = anchors[case.id]
            path, symbol = anchor["source_path"], anchor["source_symbol"]
            if path != case.gold_files[0]:
                raise ValueError(f"source anchor is not the first audited gold file: {case.id}")
            span = source_span(pipeline, path, symbol)
            result = pipeline.query(case.query)
            file_rank, span_rank, definition_rank = rank_source(result["final"], root, path, span)
            local_candidates = locate_file_symbols(
                pipeline.retriever, root / path, case.query, limit=20,
            )
            local_rank = next((rank for rank, item in enumerate(local_candidates, 1)
                               if item["name"] == symbol), None)
            candidate_groups = locate_retrieved_files(
                pipeline.retriever, root,
                [item.chunk.file_path for item in result["final"][:5]],
                case.query, file_limit=5, symbols_per_file=3,
            )
            pair_rank = joint_pair_rank(candidate_groups, path, symbol)
            joint_hit = pair_rank is not None
            first_correct = pair_rank == 1
            rows.append({"id": case.id, "repo": repo_id, "task_type": case.task_type,
                         "source_path": path, "source_symbol": symbol,
                         "file_rank": file_rank, "function_span_rank": span_rank,
                         "definition_rank": definition_rank,
                         "file_local_symbol_rank": local_rank,
                         "file_local_top_3": [item["name"] for item in local_candidates[:3]],
                         "joint_candidate_hit": joint_hit,
                         "joint_first_pair_correct": first_correct,
                         "joint_pair_rank": pair_rank,
                         "joint_groups": [{"path": group["path"], "search_rank": group["search_rank"],
                                           "symbols": [item["name"] for item in group["candidates"]]}
                                          for group in candidate_groups],
                         "returned_paths": [item.chunk.file_path.resolve().relative_to(root).as_posix()
                                            for item in result["final"]]})
    return {"schema_version": 1, "benchmark": "natural_language_source_discovery_diagnostic",
            "suite": suite,
            "evaluation_mode": "offline_retrieval_only_no_oracle_in_search",
            "file_local_evaluation_mode": "oracle_source_file_component_not_auto_file_selection",
            "retriever": RETRIEVER_VERSION, "symbol_locator": SYMBOL_LOCATOR_VERSION,
            "top_k": top_k, "inputs": identities,
            "metrics": summarize(rows),
            "by_repo": {name: summarize([row for row in rows if row["repo"] == name])
                        for name in sorted({row["repo"] for row in rows})},
            "by_task_type": {name: summarize([row for row in rows if row["task_type"] == name])
                             for name in sorted({row["task_type"] for row in rows})},
            "cases": rows}


def main() -> int:
    """Write a portable scorecard and return a diagnostic exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--suite", choices=("development", "transfer"), default="development")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/source-discovery.json")
    args = parser.parse_args()
    try:
        result = evaluate(args.workspace_root, suite=args.suite)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Source discovery evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
