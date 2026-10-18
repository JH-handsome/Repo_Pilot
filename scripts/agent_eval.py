"""Offline Agent plan evaluation for RepoPilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.artifacts import write_json_artifact

from coding_rag.agent.planner import AgentPlanConfig, run_agent_plan_mode


class OfflinePlanClient:
    """离线计划客户端，返回固定的 ReAct 计划文本以支持离线评测。"""

    def complete(self, messages):
        """返回预定义的 ReAct 计划字符串，模拟 LLM 生成计划的过程。"""
        return """## 任务分析
- 目标: inspect the repository task

## ReAct 工作流计划
1. Thought: locate code
   Action: search_code
   Action Input: {"query": "task", "top_k": 5}
   Expected Observation: candidate files
   Fallback: read_file

## 候选文件
- main.py: likely entry

## 验证计划
- python -m unittest: regression tests

## 风险与交付物
- risks: incomplete context
- deliverables: plan and checks
"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数：评测数据集路径、trace 输出路径、汇总输出路径及用例数量限制。"""
    parser = argparse.ArgumentParser(description="Evaluate Agent planning cases")
    parser.add_argument("--evalset", default="datasets/eval/agent_plan_cases.json")
    parser.add_argument("--trace-out", default="artifacts/agent_eval_trace.jsonl")
    parser.add_argument("--summary-out", default="artifacts/agent_eval_summary.json")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def evaluate_cases(evalset: str | Path, limit: int | None = None) -> tuple[list[dict], dict]:
    """运行固定离线规划 contract，并真实评估只读检索候选文件。

    Returns:
        (逐条记录列表, 汇总指标字典)
    """
    cases = json.loads(Path(evalset).read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    rows: list[dict] = []
    candidate_k = 5
    for case in cases:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path, content in case.get("repo_files", {}).items():
                path = safe_fixture_path(root, relative_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            run = run_agent_plan_mode(
                case["task"],
                AgentPlanConfig(
                    repo_path=root,
                    top_k=candidate_k,
                    recall_window=0,
                    execute_readonly_tools=True,
                ),
                client=OfflinePlanClient(),
            )
        plan_text = run.plan_text or ""
        tool_names = {tool.name for tool in run.tools}
        expected_tools = set(case.get("expected_tools") or [])
        expected_sections = set(case.get("expected_sections") or [])
        result_files = search_result_files(run.observations)
        gold_files = normalized_unique_files([
            *(case.get("changed_files") or []),
            *(case.get("test_files") or []),
        ])
        available_files = set(normalized_unique_files((case.get("repo_files") or {}).keys()))
        available_gold_files = [path for path in gold_files if path in available_files]
        missing_gold_files = [path for path in gold_files if path not in available_files]
        negative_files = sorted(available_files - set(available_gold_files))
        candidate_metrics = candidate_file_metrics(result_files, available_gold_files, candidate_k)
        if gold_files and not available_gold_files:
            candidate_metrics["gold_status"] = "unavailable"
        row = {
            "id": case.get("id"),
            "repo": case.get("repo") or "unknown",
            "candidate_k": candidate_k,
            "gold_files": gold_files,
            "available_gold_files": available_gold_files,
            "missing_gold_files": missing_gold_files,
            "gold_file_count": len(gold_files),
            "available_gold_count": len(available_gold_files),
            "missing_gold_count": len(missing_gold_files),
            "negative_file_count": len(negative_files),
            "candidate_evaluation_mode": "fixture_diagnostic",
            "discriminative": bool(negative_files),
            "result_files": result_files,
            **candidate_metrics,
            "expected_tool_coverage": coverage(expected_tools, tool_names),
            "section_coverage": coverage(expected_sections, {section for section in expected_sections if section in plan_text}),
            "trace_complete": bool(run.trace.get("trace_version") and run.trace.get("events")),
            "safe_tool_interfaces": {"apply_patch", "run_command"}.issubset(tool_names),
        }
        rows.append(row)
    gold_file_count = sum(row["gold_file_count"] for row in rows)
    available_gold_count = sum(row["available_gold_count"] for row in rows)
    missing_gold_count = sum(row["missing_gold_count"] for row in rows)
    negative_file_count = sum(row["negative_file_count"] for row in rows)
    summary = {
        "evaluation_mode": "offline_contract",
        "total": len(rows),
        "fixture_diagnostic": {
            "candidate_k": candidate_k,
            "candidate_hit_at_5": average(row["candidate_hit_at_5"] for row in rows),
            "candidate_recall_at_5": average(row["candidate_recall_at_5"] for row in rows),
            "candidate_mrr_at_5": average(row["candidate_mrr_at_5"] for row in rows),
            "discriminative": bool(negative_file_count),
            "negative_file_count": negative_file_count,
            "empty_gold_count": sum(1 for row in rows if row["gold_status"] == "empty"),
            "unavailable_gold_case_count": sum(1 for row in rows if row["gold_status"] == "unavailable"),
            "gold_file_count": gold_file_count,
            "available_gold_count": available_gold_count,
            "missing_gold_count": missing_gold_count,
            "cases_with_missing_gold": [row["id"] for row in rows if row["missing_gold_count"]],
            "dataset_coverage": (available_gold_count / gold_file_count) if gold_file_count else 0.0,
            "by_repo": summarize_by_repo(rows),
        },
        "avg_expected_tool_coverage": average(row["expected_tool_coverage"] for row in rows),
        "avg_section_coverage": average(row["section_coverage"] for row in rows),
        "trace_complete_count": sum(1 for row in rows if row["trace_complete"]),
        "safe_tool_interface_count": sum(1 for row in rows if row["safe_tool_interfaces"]),
    }
    return rows, summary


def normalize_file(path: object) -> str:
    """Normalize a repository-relative result or gold file path."""
    return str(path).replace("\\", "/").strip().casefold()


def safe_fixture_path(root: Path, relative_path: object) -> Path:
    """Resolve one fixture path and reject absolute or parent-traversal writes."""
    candidate = Path(str(relative_path))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"fixture path must stay inside the temporary repository: {relative_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"fixture path must stay inside the temporary repository: {relative_path}"
        ) from error
    return resolved


def normalized_unique_files(paths) -> list[str]:
    """Normalize file paths and preserve their first-seen order."""
    values: list[str] = []
    for path in paths:
        normalized = normalize_file(path)
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def search_result_files(observations: list[dict]) -> list[str]:
    """Extract ordered unique paths from the real search_code observation."""
    for observation in observations:
        if observation.get("tool") != "search_code":
            continue
        output = observation.get("output")
        if not isinstance(output, list):
            return []
        return normalized_unique_files(
            item.get("path") for item in output if isinstance(item, dict) and item.get("path")
        )
    return []


def candidate_file_metrics(result_files: list[str], gold_files: list[str], k: int) -> dict:
    """Compute metrics over available gold; no available gold scores zero explicitly."""
    if not gold_files:
        return {
            "gold_status": "empty",
            "candidate_file_hit": False,
            f"candidate_hit_at_{k}": 0.0,
            f"candidate_recall_at_{k}": 0.0,
            f"candidate_mrr_at_{k}": 0.0,
        }
    gold = set(gold_files)
    ranked = result_files[:k]
    matched = gold & set(ranked)
    first_rank = next((rank for rank, path in enumerate(ranked, 1) if path in gold), None)
    return {
        "gold_status": "valid",
        "candidate_file_hit": bool(matched),
        f"candidate_hit_at_{k}": float(bool(matched)),
        f"candidate_recall_at_{k}": len(matched) / len(gold),
        f"candidate_mrr_at_{k}": (1.0 / first_rank) if first_rank else 0.0,
    }


def summarize_by_repo(rows: list[dict]) -> dict[str, dict]:
    """Aggregate real candidate metrics by upstream repository."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("repo") or "unknown"), []).append(row)
    return {
        repo: {
            "case_count": len(repo_rows),
            "candidate_hit_at_5": average(row["candidate_hit_at_5"] for row in repo_rows),
            "candidate_recall_at_5": average(row["candidate_recall_at_5"] for row in repo_rows),
            "candidate_mrr_at_5": average(row["candidate_mrr_at_5"] for row in repo_rows),
            "empty_gold_count": sum(1 for row in repo_rows if row["gold_status"] == "empty"),
            "gold_file_count": sum(row["gold_file_count"] for row in repo_rows),
            "available_gold_count": sum(row["available_gold_count"] for row in repo_rows),
            "missing_gold_count": sum(row["missing_gold_count"] for row in repo_rows),
            "negative_file_count": sum(row["negative_file_count"] for row in repo_rows),
            "discriminative": bool(sum(row["negative_file_count"] for row in repo_rows)),
            "dataset_coverage": (
                sum(row["available_gold_count"] for row in repo_rows)
                / sum(row["gold_file_count"] for row in repo_rows)
            ) if sum(row["gold_file_count"] for row in repo_rows) else 0.0,
        }
        for repo, repo_rows in sorted(grouped.items())
    }


def coverage(expected: set[str], actual: set[str]) -> float:
    """计算预期项在实际项中的覆盖率；预期为空时返回 1.0。"""
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def average(values) -> float:
    """计算数值序列的算术平均值；序列为空时返回 0.0。"""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """将字典列表写入 JSONL 文件，自动创建父目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(target, rows, jsonl=True)


def main() -> int:
    """主入口：解析参数、执行离线计划评测、写入 trace 和汇总文件并打印结果。"""
    args = parse_args()
    rows, summary = evaluate_cases(args.evalset, limit=args.limit)
    write_jsonl(args.trace_out, rows)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
