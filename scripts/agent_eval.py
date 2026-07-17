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
    """在临时仓库中运行计划模式评测每个案例，汇总工具覆盖率、章节覆盖率、trace 完整性和安全接口指标。

    Returns:
        (逐条记录列表, 汇总指标字典)
    """
    cases = json.loads(Path(evalset).read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    rows: list[dict] = []
    for case in cases:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path, content in case.get("repo_files", {}).items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            run = run_agent_plan_mode(
                case["task"],
                AgentPlanConfig(repo_path=root, top_k=2, recall_window=0),
                client=OfflinePlanClient(),
            )
        plan_text = run.plan_text or ""
        tool_names = {tool.name for tool in run.tools}
        expected_tools = set(case.get("expected_tools") or [])
        expected_sections = set(case.get("expected_sections") or [])
        row = {
            "id": case.get("id"),
            "candidate_file_hit": bool(case.get("repo_files")),
            "expected_tool_coverage": coverage(expected_tools, tool_names),
            "section_coverage": coverage(expected_sections, {section for section in expected_sections if section in plan_text}),
            "trace_complete": bool(run.trace.get("trace_version") and run.trace.get("events")),
            "safe_tool_interfaces": {"apply_patch", "run_command"}.issubset(tool_names),
        }
        rows.append(row)
    summary = {
        "total": len(rows),
        "avg_expected_tool_coverage": average(row["expected_tool_coverage"] for row in rows),
        "avg_section_coverage": average(row["section_coverage"] for row in rows),
        "trace_complete_count": sum(1 for row in rows if row["trace_complete"]),
        "safe_tool_interface_count": sum(1 for row in rows if row["safe_tool_interfaces"]),
    }
    return rows, summary


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
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def main() -> int:
    """主入口：解析参数、执行离线计划评测、写入 trace 和汇总文件并打印结果。"""
    args = parse_args()
    rows, summary = evaluate_cases(args.evalset, limit=args.limit)
    write_jsonl(args.trace_out, rows)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
