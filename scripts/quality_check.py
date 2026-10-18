"""Run RepoPilot's offline quality gates from one Codex-friendly command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 240


def gate_specs(core_only: bool) -> list[tuple[str, str, list[str]]]:
    """Keep model/network flags out of this intentionally fixed offline suite."""
    specs = [
        ("unit_tests", "unit_tests", ["-m", "unittest", "discover", "-s", "tests", "-q"]),
        ("capability", "retrieval", ["scripts/capability_eval.py", "--repo", ".", "--gate"]),
        ("multirepo", "retrieval", ["scripts/multirepo_eval.py", "--gate"]),
        ("agent_runtime", "scripted_contract", ["scripts/agent_runtime_eval.py", "--gate"]),
        ("patch_outcome", "oracle_harness_self_check", [
            "scripts/patch_outcome_eval.py", "--candidate-mode", "oracle", "--gate",
        ]),
    ]
    if not core_only:
        specs.extend([
            ("aider_django_multistep", "retrieval", [
                "scripts/multirepo_eval.py",
                "--manifest", "datasets/eval/aider_django_multistep_manifest_v1.json",
                "--evalset", "datasets/eval/aider_django_multistep_v1.jsonl",
                "--baseline", "benchmarks/aider-django-multistep-baseline-v1.json",
                "--policy", "benchmarks/aider-django-multistep-gate-v1.json",
                "--gate",
            ]),
            ("explicit_absence", "offline_ast_guard", [
                "scripts/discovery_absence_eval.py", "--gate",
            ]),
        ])
    return specs


def summarize(name: str, payload: dict) -> dict:
    """Expose comparable facts while preserving each evaluation's distinct meaning."""
    if name == "capability":
        metrics = payload["retrieval"]
        return {
            "case_count": metrics["case_count"],
            "hit_at_5": metrics["hit_at_5"],
            "all_gold_at_5": metrics["all_gold_at_5"],
            "incomplete_case_ids": metrics["incomplete_case_ids"],
            "query_latency_ms_p95": payload["efficiency"]["query_latency_ms_p95"],
        }
    if name in {"multirepo", "aider_django_multistep"}:
        metrics = payload["metrics"]
        return {
            "case_count": metrics["case_count"],
            "hit_at_5": metrics["hit_at_5"],
            "all_gold_at_5": metrics["all_gold_at_5"],
            "incomplete_case_ids": metrics["incomplete_case_ids"],
            "query_latency_ms_p95": payload["efficiency"]["query_latency_ms_p95"],
        }
    if name == "agent_runtime":
        metrics = payload["metrics"]
        return {
            "case_count": metrics["case_count"],
            "selected_file_hit_at_1": metrics["selected_file_hit_at_1"],
            "gold_citation_grounding": metrics["gold_citation_grounding"],
            "tool_sequence_success": metrics["tool_sequence_success"],
        }
    if name == "patch_outcome":
        if (
            payload.get("benchmark") != "patch_outcome_harness_not_llm_intelligence"
            or payload.get("candidate", {}).get("provider") != "oracle_self_check_v1"
        ):
            raise ValueError("patch outcome result is not labelled as a self-check")
        return {"case_count": payload["metrics"]["case_count"], "harness_score": payload["score"]}
    if name == "explicit_absence":
        metrics = payload["metrics"]
        return {
            "negative_count": metrics["negative_count"],
            "negative_abstention_rate": metrics["negative_abstention_rate"],
            "positive_false_abstention_rate": metrics["positive_false_abstention_rate"],
        }
    raise ValueError(f"unknown quality gate: {name}")


def run_quality_check(*, core_only: bool = False) -> dict:
    """Run gates serially so latency checks do not contend with one another."""
    results: list[dict] = []
    with TemporaryDirectory(prefix="repo-pilot-quality-") as temp_dir:
        for name, evaluation_mode, argv in gate_specs(core_only):
            artifact = Path(temp_dir) / f"{name}.json"
            command = [sys.executable, *argv]
            if name != "unit_tests":
                command.extend(("--out", str(artifact)))
            print(f"[quality] {name}...", file=sys.stderr, flush=True)
            try:
                completed = subprocess.run(
                    command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
                    check=False,
                )
                result = {"name": name, "evaluation_mode": evaluation_mode,
                          "status": "pass" if completed.returncode == 0 else "fail"}
                if result["status"] == "pass" and name == "unit_tests":
                    match = re.search(r"\bRan (\d+) tests? in ", completed.stderr)
                    if match is not None:
                        result["metrics"] = {"test_count": int(match.group(1))}
                elif result["status"] == "pass":
                    try:
                        result["metrics"] = summarize(name, json.loads(artifact.read_text(encoding="utf-8")))
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        result["status"] = "fail"
                        result["error"] = f"missing or invalid scorecard: {exc}"[:500]
                elif result["status"] == "fail":
                    result["error"] = (completed.stderr.strip() or completed.stdout.strip())[-800:]
            except subprocess.TimeoutExpired:
                result = {"name": name, "evaluation_mode": evaluation_mode,
                          "status": "fail", "error": f"exceeded {TIMEOUT_SECONDS}s timeout"}
            except OSError as exc:
                result = {"name": name, "evaluation_mode": evaluation_mode,
                          "status": "fail", "error": str(exc)[:500]}
            results.append(result)
            print(f"[quality] {name}: {result['status']}", file=sys.stderr, flush=True)
    return {
        "schema_version": 1,
        "scope": "core" if core_only else "extended",
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "offline_only": True,
        "limitations": [
            "scripted Agent runtime is a tool contract, not general model reasoning",
            "oracle patch score checks the harness, not autonomous repair ability",
            "revealed fixed datasets are regression checks, not fresh blind generalization",
        ],
        "checks": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Print one aggregated scorecard and fail if any required offline gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true", help="omit Aider/Django extended gates")
    args = parser.parse_args(argv)
    report = run_quality_check(core_only=args.core_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
