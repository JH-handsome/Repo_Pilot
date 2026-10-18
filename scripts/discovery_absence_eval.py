"""Audit exact-definition abstention on pinned Aider and Django source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.evaluation.retrieval import PreparedRetrievalPipeline, RetrievalEvalConfig
from coding_rag.repository.index import normalize_path
from coding_rag.repository.search_index import RETRIEVER_VERSION
from coding_rag.repository.symbol_locator import (
    SYMBOL_LOCATOR_VERSION,
    exact_definition_lookup_should_abstain,
    missing_qualified_definitions,
)
from scripts.source_discovery_eval import GROUPS


EXACT_SYMBOL_RE = re.compile(r"[A-Za-z_]\w*\.[A-Za-z_]\w*\Z")


def validate_negative_case(row: dict, root: Path, pipeline: PreparedRetrievalPipeline) -> None:
    """Reject invented labels unless a class exists and its exact method does not."""
    symbol = row.get("exact_symbol")
    if not isinstance(symbol, str) or not EXACT_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid exact symbol: {row.get('id')}")
    parent, name = symbol.split(".")
    if not isinstance(row.get("query"), str) or symbol not in row["query"]:
        raise ValueError(f"query must name exact symbol: {row.get('id')}")
    raw_path = row.get("audit_path")
    if not isinstance(raw_path, str):
        raise ValueError(f"missing audit path: {row.get('id')}")
    pure = PurePosixPath(raw_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not raw_path.endswith(".py"):
        raise ValueError(f"unsafe audit path: {row.get('id')}")
    path = (root / Path(*pure.parts)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing audit file: {row.get('id')}")
    file_index = pipeline.retriever.repo_index.files.get(normalize_path(path))
    if file_index is None or not any(record.kind == "class" and record.name == parent
                                     for record in file_index.symbols):
        raise ValueError(f"audit class missing: {row.get('id')}")
    for indexed in pipeline.retriever.repo_index.files.values():
        if any(record.parent == parent and record.name == name
               and record.kind in {"function", "async_function"} for record in indexed.symbols):
            raise ValueError(f"exact definition is present: {row.get('id')}")


def evaluate(workspace: Path, dataset: Path) -> dict:
    """Validate source labels, then score the query-only guard on both label types."""
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("missing schema v1 cases")
    negatives = payload["cases"]
    if (not negatives or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                             or not row["id"] for row in negatives)
            or len({row["id"] for row in negatives}) != len(negatives)):
        raise ValueError("negative cases must have unique nonempty ids")
    results = []
    controls = []
    identities = []
    known_repos = {group[0] for group in GROUPS}
    if {row.get("repo_id") for row in negatives} != known_repos:
        raise ValueError("negative set must cover the pinned repositories")
    for repo_id, manifest_name, evalsets, _ in GROUPS:
        suite = load_and_validate_suite(
            workspace, PROJECT_ROOT / manifest_name, PROJECT_ROOT / evalsets[0]
        )
        repo = next(item for item in suite.repositories if item.id == repo_id)
        root = repo.root
        identities.append({"repo": repo_id, "commit": repo.commit,
                           "manifest_sha256": suite.manifest_sha256})
        pipeline = PreparedRetrievalPipeline.build(RetrievalEvalConfig(repo_path=root, top_k=5))
        for row in negatives:
            if row.get("repo_id") != repo_id:
                continue
            validate_negative_case(row, root, pipeline)
            missing = missing_qualified_definitions(pipeline.retriever, row["query"])
            abstained = exact_definition_lookup_should_abstain(row["query"], missing)
            results.append({"id": row["id"], "repo": repo_id,
                            "exact_symbol": row["exact_symbol"],
                            "missing_reported": row["exact_symbol"] in missing,
                            "abstained": abstained})
        for evalset in evalsets:
            positive_suite = load_and_validate_suite(
                workspace, PROJECT_ROOT / manifest_name, PROJECT_ROOT / evalset
            )
            for case in positive_suite.cases:
                missing = missing_qualified_definitions(pipeline.retriever, case.query)
                controls.append({"id": case.id, "repo": repo_id,
                                 "false_abstention": exact_definition_lookup_should_abstain(
                                     case.query, missing),
                                 "missing_references": missing})
    return {
        "schema_version": 1,
        "benchmark": "exact_indexed_definition_absence_guard",
        "evaluation_mode": "offline_ast_guard_not_model_abstention",
        "retriever": RETRIEVER_VERSION,
        "symbol_locator": SYMBOL_LOCATOR_VERSION,
        "guard_version": "exact-direct-definition-v1",
        "dataset": {"name": dataset.name, "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "repositories": identities,
        "metrics": {"negative_count": len(results), "positive_control_count": len(controls),
                    "negative_abstention_rate": sum(row["abstained"] for row in results) / len(results),
                    "negative_missing_warning_rate": sum(row["missing_reported"] for row in results) / len(results),
                    "positive_false_abstention_rate": sum(row["false_abstention"] for row in controls) / len(controls)},
        "negatives": results, "positive_controls": controls,
    }


def compare(current: dict, baseline: dict) -> list[str]:
    """Reject identity drift and regression without silently rewriting the baseline."""
    failures = []
    for key in ("schema_version", "benchmark", "evaluation_mode", "retriever",
                "symbol_locator", "guard_version", "dataset", "repositories"):
        if current.get(key) != baseline.get(key):
            failures.append(f"incompatible {key}")
    present = current.get("metrics") or {}
    expected = baseline.get("metrics") or {}
    for key in ("negative_count", "positive_control_count"):
        if present.get(key) != expected.get(key):
            failures.append(f"incompatible {key}")
    for key in ("negative_abstention_rate", "negative_missing_warning_rate"):
        if present.get(key, -1) < expected.get(key, 1):
            failures.append(f"{key} regression")
    if present.get("positive_false_abstention_rate", 1) > expected.get("positive_false_abstention_rate", 0):
        failures.append("positive_false_abstention_rate regression")
    return failures


def main() -> int:
    """Write one reproducible scorecard without invoking a model or mutating source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "datasets/eval/discovery_explicit_missing_v1.json")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "artifacts/eval/discovery-absence.json")
    parser.add_argument("--baseline", type=Path,
                        default=PROJECT_ROOT / "benchmarks/discovery-absence-baseline-v1.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(args.workspace_root, args.dataset)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        if args.gate:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            failures = compare(result, baseline)
            print("PASS" if not failures else "FAIL: " + "; ".join(failures))
            return 0 if not failures else (2 if any(item.startswith("incompatible") for item in failures) else 1)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Discovery absence evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
