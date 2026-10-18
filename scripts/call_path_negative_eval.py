"""Check source-audited forbidden static call targets without running repository code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.evaluation.multirepo import load_and_validate_suite
from coding_rag.tools.agent_readonly import ReadOnlyAgentTools
from scripts.call_path_eval import confined_path


def load_cases(path: Path, roots: dict[str, Path]) -> list[dict]:
    """Validate unique cases and confined source paths before any index build."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) and payload.get("schema_version") == 1 else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("negative cases must be a non-empty schema v1 array")
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("negative cases require unique non-empty ids")
        seen.add(case_id)
        root = roots.get(row.get("repo_id"))
        if root is None:
            raise ValueError(f"unknown repository: {case_id}")
        for field in ("entry_path", "forbidden_target_path"):
            relative = confined_path(row.get(field))
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
                raise ValueError(f"missing or unsafe {field}: {case_id}")
        for field in ("entry_symbol", "forbidden_target_symbol", "reason"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing {field}: {case_id}")
    return rows


def score_case(row: dict, observation: dict) -> dict:
    """Reject one exact forbidden target while reporting incomplete trace evidence."""
    forbidden = (row["forbidden_target_path"].replace("\\", "/"), row["forbidden_target_symbol"])
    matched = [edge for edge in observation.get("edges", [])
               if (edge.get("target_path", "").replace("\\", "/"), edge.get("target_symbol"))
               == forbidden]
    return {
        "id": row["id"],
        "entry_path": row["entry_path"],
        "entry_symbol": row["entry_symbol"],
        "forbidden_target_path": forbidden[0],
        "forbidden_target_symbol": forbidden[1],
        "observed_edge_count": len(observation.get("edges", [])),
        "false_link": bool(matched),
        "truncated": bool(observation.get("truncated")),
    }


def evaluate(workspace: Path, manifest: Path, cases_path: Path) -> dict:
    """Run every case against a pinned source checkout, reusing one tool index per repo."""
    suite = load_and_validate_suite(workspace, manifest, PROJECT_ROOT / "datasets/eval/multirepo_source.jsonl")
    roots = {spec.id: spec.root for spec in suite.repositories}
    cases = load_cases(cases_path, roots)
    tools = {repo_id: ReadOnlyAgentTools(root) for repo_id, root in roots.items()}
    results = [score_case(row, tools[row["repo_id"]].trace_call_path(
        row["entry_path"], row["entry_symbol"]
    )) for row in cases]
    false_links = sum(item["false_link"] for item in results)
    truncated = sum(item["truncated"] for item in results)
    return {
        "schema_version": 1,
        "evaluation_mode": "pinned_static_component_negative",
        "dataset": {"name": cases_path.name, "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()},
        "manifest": {"name": manifest.name, "sha256": suite.manifest_sha256},
        "metrics": {
            "case_count": len(results),
            "false_link_count": false_links,
            "truncated_count": truncated,
            "negative_abstain_rate": (len(results) - false_links) / len(results),
        },
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Emit an offline JSON report; gate only exact negative-link safety."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets/eval/multirepo_manifest.json")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "datasets/eval/call_path_generalization_negative_v1.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/call-path-negative-scorecard.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.workspace_root, args.manifest, args.cases)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        if args.gate and (result["metrics"]["false_link_count"] or result["metrics"]["truncated_count"]):
            print("FAIL")
            return 1
        if args.gate:
            print("PASS")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Evaluation error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
