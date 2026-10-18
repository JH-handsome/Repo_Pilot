"""Offline gate for exact direct-call verification on pinned Django and Aider."""

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
from scripts.repo_inspect import verify_edge


def evaluate(workspace: Path, cases_path: Path) -> dict:
    """Validate checkout identity and labels, then score one static check per case."""
    manifests = {
        "django": ("multirepo_manifest.json", "multirepo_source.jsonl"),
        "aider": ("aider_manifest_v1.json", "aider_call_edge_positive_v1.jsonl"),
    }
    roots = {}
    identities = {}
    for repo_id, (manifest_name, evalset_name) in manifests.items():
        suite = load_and_validate_suite(
            workspace, PROJECT_ROOT / "datasets/eval" / manifest_name,
            PROJECT_ROOT / "datasets/eval" / evalset_name,
        )
        spec = next(spec for spec in suite.repositories if spec.id == repo_id)
        roots[repo_id] = spec.root
        identities[repo_id] = {"commit": spec.commit, "manifest_sha256": suite.manifest_sha256}
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) and payload.get("schema_version") == 1 else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases must be a non-empty schema v1 array")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("case ids must be unique strings")
        ids.add(row["id"])
        root = roots.get(row.get("repo_id"))
        if root is None or row.get("expected") not in {"supported_static_call", "not_established"}:
            raise ValueError(f"invalid repo or label: {row['id']}")
        for field in ("source_path", "target_path"):
            relative = confined_path(row.get(field))
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
                raise ValueError(f"unsafe or missing {field}: {row['id']}")
        for field in ("source_symbol", "target_symbol"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing {field}: {row['id']}")
    tools = {repo_id: ReadOnlyAgentTools(root) for repo_id, root in roots.items()}
    results = []
    for row in rows:
        checked = verify_edge(tools[row["repo_id"]], row["source_path"], row["source_symbol"],
                              row["target_path"], row["target_symbol"])
        results.append({
            "id": row["id"], "repo_id": row["repo_id"], "expected": row["expected"],
            "status": checked["status"], "correct": checked["status"] == row["expected"],
            "trace_truncated": checked["trace_truncated"],
            "direct_edge_count": checked["direct_edge_count"],
            "evidence_lines": sum(len(item["text"].splitlines()) for item in checked["evidence"]),
            "source_line": checked["edge"]["source_line"] if checked["edge"] else None,
            "target_line": checked["edge"]["target_start_line"] if checked["edge"] else None,
        })
    positive = [item for item in results if item["expected"] == "supported_static_call"]
    negative = [item for item in results if item["expected"] == "not_established"]
    return {
        "schema_version": 1,
        "evaluation_mode": "pinned_static_component_not_runtime_or_model_capability",
        "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "repositories": identities,
        "metrics": {
            "case_count": len(results), "positive_count": len(positive), "negative_count": len(negative),
            "positive_supported": sum(item["correct"] for item in positive),
            "negative_not_established": sum(item["correct"] for item in negative),
            "false_support_count": sum(item["status"] == "supported_static_call" for item in negative),
            "truncated_count": sum(item["trace_truncated"] for item in results),
            "evidence_lines_total": sum(item["evidence_lines"] for item in results),
        },
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Save a reproducible scorecard and optionally require all audited labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "datasets/eval/direct_edge_verification_v1.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts/eval/direct-edge-verification-v1.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.workspace_root, args.cases)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(args.out, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        metrics = result["metrics"]
        passed = (metrics["positive_supported"] == metrics["positive_count"]
                  and metrics["negative_not_established"] == metrics["negative_count"]
                  and metrics["false_support_count"] == 0)
        if args.gate:
            print("PASS" if passed else "FAIL")
        return 0 if not args.gate or passed else 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Edge verification evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
