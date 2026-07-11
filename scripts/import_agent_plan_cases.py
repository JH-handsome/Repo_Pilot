"""Import SWE-bench Verified cases for Agent planning evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_OUTPUT = Path("datasets/eval/agent_plan_cases.json")
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Agent planning cases from SWE-bench Verified")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = fetch_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    cases = [convert_row(item["row"]) for item in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")
    return 0


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    limit: int,
) -> list[dict]:
    query = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": limit,
        }
    )
    with urlopen(f"{DATASET_ROWS_URL}?{query}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["rows"]


def convert_row(row: dict) -> dict:
    changed_files = diff_paths(row.get("patch", ""))
    test_files = diff_paths(row.get("test_patch", ""))
    repo_files = build_repo_files(row, changed_files, test_files)
    return {
        "id": row["instance_id"],
        "source": "swe-bench-verified",
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "difficulty": row.get("difficulty", ""),
        "task": normalize_task(row.get("problem_statement", "")),
        "hints_text": row.get("hints_text", ""),
        "changed_files": changed_files,
        "test_files": test_files,
        "fail_to_pass": parse_json_list(row.get("FAIL_TO_PASS", "")),
        "pass_to_pass": parse_json_list(row.get("PASS_TO_PASS", "")),
        "repo_files": repo_files,
        "expected_tools": ["search_code", "read_file", "run_checks"],
        "expected_sections": ["任务分析", "ReAct 工作流计划", "验证计划", "风险与交付物"],
    }


def diff_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff_text, flags=re.MULTILINE):
        path = match.group(2)
        if path.endswith(".py") and path not in paths:
            paths.append(path)
    return paths


def build_repo_files(row: dict, changed_files: list[str], test_files: list[str]) -> dict[str, str]:
    paths = changed_files[:3] + [path for path in test_files[:3] if path not in changed_files[:3]]
    if not paths:
        paths = ["issue_context.py"]
    repo_files: dict[str, str] = {}
    problem = normalize_task(row.get("problem_statement", ""))
    hints = row.get("hints_text", "")
    for path in paths:
        repo_files[path] = "\n".join(
            [
                f"# SWE-bench instance: {row['instance_id']}",
                f"# Repo: {row['repo']}",
                f"PROBLEM_STATEMENT = {json.dumps(problem[:3000], ensure_ascii=False)}",
                f"HINTS_TEXT = {json.dumps(hints[:1500], ensure_ascii=False)}",
                "",
                "def issue_reproduction_context():",
                "    return PROBLEM_STATEMENT",
                "",
            ]
        )
    return repo_files


def normalize_task(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def parse_json_list(text: str) -> list[str]:
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
