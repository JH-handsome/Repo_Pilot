"""Gate static call-path navigation on pinned source; no model or network calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.tools.agent_readonly import ReadOnlyAgentTools


def confined_path(value: object) -> str:
    """Accept only normalized, relative POSIX paths inside a pinned checkout."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute() or Path(value).drive or Path(value).is_absolute()
        or value == "." or ".." in pure.parts or pure.as_posix() != value
    ):
        raise ValueError(f"unconfined path: {value!r}")
    return value


def load_cases(path: Path, known_repos: set[str]) -> list[dict]:
    """Validate component fixtures independently of runtime tool observations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("call-path fixture schema_version must equal 1")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("call-path cases must be a non-empty array")
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("every case needs a string id")
        if row["id"] in ids or not row["id"]:
            raise ValueError(f"duplicate or empty case id: {row['id']!r}")
        ids.add(row["id"])
        if row.get("repo_id") not in known_repos:
            raise ValueError(f"unknown repo_id in {row['id']}")
        confined_path(row.get("entry_path"))
        if not isinstance(row.get("entry_symbol"), str) or not row["entry_symbol"].strip():
            raise ValueError(f"missing entry_symbol in {row['id']}")
        expected = row.get("expected_targets")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"missing expected_targets in {row['id']}")
        for target in expected:
            if not isinstance(target, dict):
                raise ValueError(f"invalid target in {row['id']}")
            confined_path(target.get("path"))
            if not isinstance(target.get("symbol"), str) or not target["symbol"].strip():
                raise ValueError(f"missing target symbol in {row['id']}")
    return rows


def valid_edge(edge: dict, root: Path, source_path: str) -> bool:
    """Check emitted locators and repository confinement without trusting the tool."""
    try:
        target = root / confined_path(edge.get("target_path"))
        source = root / confined_path(source_path)
        if not target.is_file() or not source.is_file():
            return False
        source_lines = len(source.read_text(encoding="utf-8").splitlines())
        target_lines = len(target.read_text(encoding="utf-8").splitlines())
        call_line = edge.get("source_line")
        start = edge.get("target_start_line")
        end = edge.get("target_end_line")
        return (
            isinstance(call_line, int) and 1 <= call_line <= source_lines
            and isinstance(start, int) and isinstance(end, int)
            and 1 <= start <= end <= target_lines
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def score_case(row: dict, observation: dict, root: Path) -> dict:
    """Score a declared target subset; other returned edges are not labeled false."""
    edges = observation.get("edges", [])
    expected = {(target["path"], target["symbol"]) for target in row["expected_targets"]}
    matched = {
        (edge.get("target_path"), edge.get("target_symbol"))
        for edge in edges if isinstance(edge, dict)
    } & expected
    located = {
        (edge["target_path"], edge["target_symbol"])
        for edge in edges if isinstance(edge, dict)
        and (edge.get("target_path"), edge.get("target_symbol")) in expected
        and valid_edge(edge, root, row["entry_path"])
    }
    invalid = sum(
        not isinstance(edge, dict) or not valid_edge(edge, root, row["entry_path"])
        for edge in edges
    )
    return {
        "id": row["id"],
        "repo_id": row["repo_id"],
        "expected_target_count": len(expected),
        "matched_target_count": len(matched),
        "located_target_count": len(located),
        "emitted_edge_count": len(edges),
        "invalid_edge_count": invalid,
        "complete": matched == expected and located == expected and invalid == 0
                    and not observation.get("truncated", False),
        "missing_targets": [
            {"path": path, "symbol": symbol} for path, symbol in sorted(expected - matched)
        ],
    }


def evaluate(workspace: Path, manifest: Path, cases_path: Path) -> dict:
    """Run one shared read-only tool instance per pinned repository."""
    source_evalset = PROJECT_ROOT / "datasets/eval/multirepo_source.jsonl"
    suite = load_and_validate_suite(workspace, manifest, source_evalset)
    repos = {spec.id: spec.root for spec in suite.repositories}
    rows = load_cases(cases_path, set(repos))
    tools = {repo_id: ReadOnlyAgentTools(root) for repo_id, root in repos.items()}
    results = []
    for row in rows:
        observation = tools[row["repo_id"]].trace_call_path(
            row["entry_path"], row["entry_symbol"]
        )
        results.append(score_case(row, observation, repos[row["repo_id"]]))
    expected_total = sum(item["expected_target_count"] for item in results)
    return {
        "schema_version": 1,
        "benchmark": "pinned_python_call_path_component",
        "dataset": {"name": cases_path.name, "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()},
        "manifest": {"name": manifest.name, "sha256": suite.manifest_sha256},
        "metrics": {
            "case_count": len(results),
            "target_recall": sum(item["matched_target_count"] for item in results) / expected_total,
            "locator_rate": sum(item["located_target_count"] for item in results) / expected_total,
            "complete_case_rate": sum(item["complete"] for item in results) / len(results),
            "invalid_edge_count": sum(item["invalid_edge_count"] for item in results),
        },
        "cases": results,
    }


def gate(scorecard: dict, baseline: dict, policy: dict) -> list[str]:
    """Reject identity drift and metric regressions against an explicit policy."""
    failures = []
    for field in ("schema_version", "benchmark", "dataset", "manifest"):
        if scorecard[field] != baseline.get(field):
            failures.append(f"baseline {field} mismatch")
    if policy.get("schema_version") != 1 or policy.get("benchmark") != scorecard["benchmark"]:
        failures.append("policy identity mismatch")
    metrics = scorecard["metrics"]
    for field, threshold in (
        ("target_recall", "min_target_recall"),
        ("locator_rate", "min_locator_rate"),
        ("complete_case_rate", "min_complete_case_rate"),
    ):
        if metrics[field] < policy[threshold] or metrics[field] < baseline["metrics"][field]:
            failures.append(f"{field} regression")
    if metrics["invalid_edge_count"] > policy["max_invalid_edge_count"]:
        failures.append("invalid_edge_count exceeds policy")
    return failures


def main() -> int:
    """Run the pinned component suite and optionally enforce its gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets/eval/multirepo_manifest.json")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "datasets/eval/call_path_tool_cases.json")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "benchmarks/call-path-tool-baseline-v1.json")
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "benchmarks/call-path-tool-gate-v1.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/call-path-tool-scorecard.json")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    if args.gate and args.update_baseline:
        parser.error("--gate and --update-baseline are mutually exclusive")
    try:
        scorecard = evaluate(args.workspace_root, args.manifest, args.cases)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, scorecard)
        if args.update_baseline:
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            write_json_artifact(args.baseline, scorecard)
        print(json.dumps(scorecard["metrics"], ensure_ascii=False, indent=2))
        if args.gate:
            failures = gate(
                scorecard,
                json.loads(args.baseline.read_text(encoding="utf-8")),
                json.loads(args.policy.read_text(encoding="utf-8")),
            )
            print("PASS" if not failures else "FAIL: " + "; ".join(failures))
            return 0 if not failures else 1
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Call-path evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
