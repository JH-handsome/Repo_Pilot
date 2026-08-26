"""RAG 检索轨迹序列化和可读报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_rag.rag.prompt import ContextBlock, compact_results_for_context
from coding_rag.tools.bm25 import SearchResult


TRACE_VERSION = "1.0"


def build_retrieval_trace(
    query: str,
    seed_results: list[SearchResult],
    recalled_results: list[SearchResult],
    final_results: list[SearchResult],
    *,
    params: dict[str, Any] | None = None,
    include_text: bool = False,
) -> dict[str, Any]:
    """构建单次 RAG 检索轨迹。"""
    # 统一 trace 的外壳是 run/events/artifacts；旧 summary/stages 仍保留给现有消费者兼容。
    context_blocks = compact_results_for_context(final_results)
    params = params or {}
    summary = {
        "seed_count": len(seed_results),
        "recalled_count": len(recalled_results),
        "final_count": len(final_results),
        "context_block_count": len(context_blocks),
    }
    stages = {
        "initial_search": serialize_results(seed_results, include_text=include_text),
        "neighbor_recall": serialize_results(recalled_results, include_text=include_text),
        "final_filter": serialize_results(final_results, include_text=include_text),
        "context_compaction": serialize_context_blocks(context_blocks, include_text=include_text),
    }
    mode = str(params.get("workflow_mode") or "ask")
    run = build_trace_run(
        mode=mode,
        query=query,
        status="success",
        repo_path=params.get("repo_path"),
        params=params,
        summary=summary,
    )
    events = build_unified_retrieval_events(stages)
    return {
        "trace_version": TRACE_VERSION,
        "run": run,
        "events": events,
        "artifacts": {
            "retrieval": stages,
            "agent": {},
            "tools": [],
        },
        "query": query,
        "params": params,
        "summary": summary,
        "stages": stages,
    }


def build_trace_run(
    *,
    mode: str,
    query: str | None = None,
    task: str | None = None,
    status: str = "success",
    repo_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
    flags: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the run-level envelope shared by all trace producers."""
    # run 只放本次执行的摘要信息；每个具体阶段的细节放在 events/artifacts。
    return {
        "run_id": run_id or str(uuid4()),
        "mode": mode,
        "status": status,
        "query": query if query is not None else task,
        "task": task,
        "repo_path": str(repo_path) if repo_path is not None else None,
        "params": params or {},
        "flags": flags or {"llm": False, "safe_mode": False},
        "summary": summary or {},
    }


def build_trace_event(
    *,
    step: str,
    status: str = "success",
    input: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Build one normalized step/tool event."""
    # event 是定位问题的最小单位：输入、摘要、产物、错误和耗时都挂在同一个结构里。
    return {
        "step": step,
        "status": status,
        "input": input or {},
        "output_summary": output_summary or {},
        "artifacts": artifacts or {},
        "error": error,
        "duration_ms": duration_ms,
    }


def build_unified_retrieval_events(stages: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Convert legacy retrieval stages into normalized events."""
    events: list[dict[str, Any]] = []
    for stage_name in ("initial_search", "neighbor_recall", "final_filter", "context_compaction"):
        rows = stages.get(stage_name, [])
        events.append(
            build_trace_event(
                step=stage_name,
                status="success",
                output_summary={"result_count": len(rows)},
                artifacts={"results": rows},
            )
        )
    return events


def build_unified_retrieval_trace(
    *,
    query: str,
    params: dict[str, Any] | None,
    summary: dict[str, Any],
    stages: dict[str, list[dict[str, Any]]],
    mode: str = "ask",
) -> dict[str, Any]:
    """Build a normalized trace around already-serialized retrieval stages."""
    params = params or {}
    return {
        "trace_version": TRACE_VERSION,
        "run": build_trace_run(
            mode=mode,
            query=query,
            repo_path=params.get("repo_path"),
            params=params,
            summary=summary,
        ),
        "events": build_unified_retrieval_events(stages),
        "artifacts": {"retrieval": stages, "agent": {}, "tools": []},
        "query": query,
        "params": params,
        "summary": summary,
        "stages": stages,
    }


def build_agent_events(agent_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert legacy Agent step trace rows into normalized events."""
    events: list[dict[str, Any]] = []
    for item in agent_trace:
        artifacts = item.get("artifacts") or {}
        events.append(
            build_trace_event(
                step=str(item.get("step") or "agent_step"),
                status=normalize_event_status(str(item.get("status") or "success")),
                output_summary=summarize_artifact_counts(artifacts),
                artifacts=artifacts,
            )
        )
    return events


def build_tool_event(
    *,
    tool: str,
    result: dict[str, Any],
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert executor tool results into a normalized event."""
    # 工具事件会把安全检查、截断信息、patch 文件变更等诊断字段放进 artifacts。
    status = "success" if int(result.get("returncode") or 0) == 0 else "failed"
    artifacts: dict[str, Any] = {"tool": tool}
    if tool == "run_command":
        artifacts["command"] = {
            key: result.get(key)
            for key in (
                "command",
                "args",
                "returncode",
                "safety",
                "timeout_seconds",
                "stdout_chars",
                "stderr_chars",
                "stdout_truncated",
                "stderr_truncated",
                "executed",
                "snapshot_id",
                "affected_files",
                "post_change_diff",
            )
            if key in result
        }
    elif tool == "apply_patch":
        metadata = ((result.get("safety") or {}).get("metadata") or {})
        artifacts["patch"] = {
            "applied": result.get("applied"),
            "snapshot_id": result.get("snapshot_id"),
            "affected_files": result.get("affected_files", []),
            "post_change_diff": result.get("post_change_diff"),
            "changed_files": metadata.get("changed_files", []),
            "deleted_files": metadata.get("deleted_files", []),
            "deletes_files": metadata.get("deletes_files", False),
            "safety": result.get("safety"),
        }
    else:
        artifacts["result"] = result
    return build_trace_event(
        step=tool,
        status=status,
        input=input or {},
        output_summary={"returncode": result.get("returncode")},
        artifacts=artifacts,
        error=None if status == "success" else {"type": "ToolFailed", "message": "tool returned non-zero status"},
    )


def normalize_event_status(status: str) -> str:
    """将 done/drafted/planned 映射为 success，保留 skipped，否则返回原状态。"""
    if status in {"done", "drafted", "planned"}:
        return "success"
    if status in {"skipped"}:
        return "skipped"
    return status


def summarize_artifact_counts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """从工件中提取固定的计数字段形成摘要。"""
    summary: dict[str, Any] = {}
    for key in ("seed_count", "recalled_count", "final_count", "candidate_file_count", "memory_count"):
        if key in artifacts:
            summary[key] = artifacts[key]
    return summary


def serialize_results(results: list[SearchResult], *, include_text: bool = False) -> list[dict[str, Any]]:
    """序列化搜索结果，包含排名和来源元数据。"""
    return [
        serialize_result(result, rank=index, include_text=include_text)
        for index, result in enumerate(results, start=1)
    ]


def serialize_result(
    result: SearchResult,
    *,
    rank: int | None = None,
    include_text: bool = False,
) -> dict[str, Any]:
    """将单个搜索结果序列化为追踪字典。"""
    chunk = result.chunk
    row: dict[str, Any] = {
        "rank": rank,
        "file": str(chunk.file_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": result.score,
        "source": result.source,
        "preview": preview_text(chunk.text),
    }
    if include_text:
        row["text"] = chunk.text
    return row


def serialize_context_blocks(
    blocks: list[ContextBlock],
    *,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """序列化压缩后的上下文块及其来源信息。"""
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        row: dict[str, Any] = {
            "rank": index,
            "file": str(block.file_path),
            "start_line": block.start_line,
            "end_line": block.end_line,
            "score": block.score,
            "sources": list(block.sources),
            "chunk_count": block.chunk_count,
            "first_rank": block.first_rank + 1,
            "char_count": len(block.text),
            "preview": preview_text(block.text),
        }
        if include_text:
            row["text"] = block.text
        rows.append(row)
    return rows


def render_trace_report(trace: dict[str, Any], *, limit: int = 10) -> str:
    """把轨迹渲染成适合直接阅读的文本报告。"""
    if trace.get("trace_version") and not get_retrieval_stages(trace):
        return render_event_trace_report(trace, limit=limit)
    summary = trace.get("summary") or trace.get("run", {}).get("summary") or {}
    query = trace.get("query") or trace.get("run", {}).get("query") or trace.get("run", {}).get("task")
    lines = [
        "RAG 检索轨迹",
        f"Query: {query}",
        (
            "Summary: "
            f"initial={summary.get('seed_count', 0)}, "
            f"recalled={summary.get('recalled_count', 0)}, "
            f"final={summary.get('final_count', 0)}, "
            f"context_blocks={summary.get('context_block_count', 0)}"
        ),
    ]

    stage_titles = {
        "initial_search": "1. Initial Search",
        "neighbor_recall": "2. Neighbor Recall",
        "final_filter": "3. Final Filter",
        "context_compaction": "4. Context Compaction",
    }
    stages = get_retrieval_stages(trace)
    for stage_name, title in stage_titles.items():
        rows = stages.get(stage_name, [])
        lines.extend(["", title])
        if not rows:
            lines.append("- empty")
            continue
        for row in rows[:limit]:
            lines.append(render_trace_row(row, compact=stage_name == "context_compaction"))
        if len(rows) > limit:
            lines.append(f"- ... {len(rows) - limit} more")

    return "\n".join(lines)


def render_event_trace_report(trace: dict[str, Any], *, limit: int = 10) -> str:
    """将事件跟踪数据渲染为 Markdown 报告。"""
    run = trace.get("run") or {}
    events = trace.get("events") or []
    lines = [
        "统一 Trace 事件",
        f"Mode: {run.get('mode')}",
        f"Status: {run.get('status')}",
        f"Task: {run.get('task') or run.get('query')}",
        f"Events: {len(events)}",
    ]
    lines[0] = "统一 Trace 事件"
    for index, event in enumerate(events[:limit], start=1):
        lines.append(
            f"- {index}. {event.get('step')} [{event.get('status')}] "
            f"{json.dumps(event.get('output_summary') or {}, ensure_ascii=False)}"
        )
        error = event.get("error")
        if error:
            lines.append(f"  error: {error.get('type')}: {error.get('message')}")
    if len(events) > limit:
        lines.append(f"- ... {len(events) - limit} more")
    return "\n".join(lines)


def get_retrieval_stages(trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return retrieval stages from either new artifacts or legacy fields."""
    artifacts = trace.get("artifacts") or {}
    retrieval = artifacts.get("retrieval")
    stage_names = {"initial_search", "neighbor_recall", "final_filter", "context_compaction"}
    if isinstance(retrieval, dict) and stage_names.intersection(retrieval):
        return retrieval
    stages = trace.get("stages")
    if isinstance(stages, dict):
        return stages
    return {}


def render_trace_row(row: dict[str, Any], *, compact: bool = False) -> str:
    """渲染单行检索追踪信息，返回格式化字符串。"""
    location = f"{row['file']}:{row['start_line']}-{row['end_line']}"
    if compact:
        source_text = ",".join(row.get("sources", []))
        return (
            f"- #{row['rank']} score={row['score']:.4f} chunks={row['chunk_count']} "
            f"sources={source_text} {location} | {row['preview']}"
        )

    return (
        f"- #{row['rank']} score={row['score']:.4f} source={row['source']} "
        f"{location} | {row['preview']}"
    )


def write_trace_json(path: str | Path, trace: dict[str, Any]) -> None:
    """将 trace 字典写入 JSON 文件，自动创建父目录并确保 UTF-8 编码。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_text(text: str, max_chars: int = 120) -> str:
    """将多行文本压缩为单行预览，超出指定长度时截断并追加省略号。"""
    preview = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."
