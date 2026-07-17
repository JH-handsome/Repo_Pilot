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
    """解析命令行参数。

    Returns:
        包含数据集、配置、分割、限制、偏移和输出路径等参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="Import Agent planning cases from SWE-bench Verified")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """主入口函数。

    从 SWE-bench Verified 数据集获取指定行，转换为评测用例格式，写入 JSON 文件。

    Returns:
        0 表示正常退出。
    """
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
    """通过 Hugging Face datasets-server HTTP API 获取数据集行。

    Args:
        dataset: 数据集名称，如 "princeton-nlp/SWE-bench_Verified"。
        config: 数据集配置名称。
        split: 数据分割名称。
        offset: 起始偏移量。
        limit: 最大返回行数。

    Returns:
        从 API 响应的 "rows" 字段中提取的行字典列表。
    """
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
    """将原始数据集行转换为 Agent 计划评测用例格式。

    从原始行中提取补丁文件、测试文件、任务描述等信息，构造标准化的评测用例字典。

    Args:
        row: 原始行字典，包含 patch、test_patch、problem_statement 等字段。

    Returns:
        包含 id、repo、task、changed_files、repo_files 等字段的用例字典。
    """
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
    """从 Git diff 文本中提取唯一的 Python 目标文件路径。

    仅保留补丁中以 .py 结尾的目标路径，去重后返回。

    Args:
        diff_text: Git diff 格式的补丁文本。

    Returns:
        去重后的 Python 文件路径列表。
    """
    paths: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff_text, flags=re.MULTILINE):
        path = match.group(2)
        if path.endswith(".py") and path not in paths:
            paths.append(path)
    return paths


def build_repo_files(row: dict, changed_files: list[str], test_files: list[str]) -> dict[str, str]:
    """为离线计划评测生成最多若干个上下文文件。

    从变更文件和测试文件各取至多 3 个不重复路径（总计最多 6 个），若均为空则回退为
    issue_context.py；为每个路径构造包含实例元信息、问题描述和提示文本的占位文件内容。

    Args:
        row: 原始行字典。
        changed_files: 变更文件路径列表。
        test_files: 测试文件路径列表。

    Returns:
        路径到文件内容的映射字典。
    """
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
    """规范化任务文本，去除每行尾部空白并压缩首尾空行。

    Args:
        text: 原始任务文本。

    Returns:
        规范化后的文本。
    """
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def parse_json_list(text: str) -> list[str]:
    """解析 JSON 字符串为列表；解析失败或结果非列表时返回空列表。

    Args:
        text: JSON 格式的字符串。

    Returns:
        解析得到的字符串列表；若 text 为空、解析失败或结果不是列表则返回 []。
    """
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
