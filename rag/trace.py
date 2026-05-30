"""RAG 检索轨迹序列化和可读报告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coding_rag.bm25_retriever import SearchResult
from rag.prompt import ContextBlock, compact_results_for_context


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
    context_blocks = compact_results_for_context(final_results)
    return {
        "query": query,
        "params": params or {},
        "summary": {
            "seed_count": len(seed_results),
            "recalled_count": len(recalled_results),
            "final_count": len(final_results),
            "context_block_count": len(context_blocks),
        },
        "stages": {
            "initial_search": serialize_results(seed_results, include_text=include_text),
            "neighbor_recall": serialize_results(recalled_results, include_text=include_text),
            "final_filter": serialize_results(final_results, include_text=include_text),
            "context_compaction": serialize_context_blocks(context_blocks, include_text=include_text),
        },
    }


def serialize_results(results: list[SearchResult], *, include_text: bool = False) -> list[dict[str, Any]]:
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
    summary = trace["summary"]
    lines = [
        "RAG 检索轨迹",
        f"Query: {trace['query']}",
        (
            "Summary: "
            f"initial={summary['seed_count']}, "
            f"recalled={summary['recalled_count']}, "
            f"final={summary['final_count']}, "
            f"context_blocks={summary['context_block_count']}"
        ),
    ]

    stage_titles = {
        "initial_search": "1. Initial Search",
        "neighbor_recall": "2. Neighbor Recall",
        "final_filter": "3. Final Filter",
        "context_compaction": "4. Context Compaction",
    }
    for stage_name, title in stage_titles.items():
        rows = trace["stages"].get(stage_name, [])
        lines.extend(["", title])
        if not rows:
            lines.append("- empty")
            continue
        for row in rows[:limit]:
            lines.append(render_trace_row(row, compact=stage_name == "context_compaction"))
        if len(rows) > limit:
            lines.append(f"- ... {len(rows) - limit} more")

    return "\n".join(lines)


def render_trace_row(row: dict[str, Any], *, compact: bool = False) -> str:
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_text(text: str, max_chars: int = 120) -> str:
    preview = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."
