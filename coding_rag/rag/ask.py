"""ASK 模式检索流水线：检索、召回、过滤和 prompt 组装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_rag.repository.chunks import split_python_files
from coding_rag.repository.files import load_python_files
from coding_rag.tools.bm25 import BM25Retriever, SearchResult
from coding_rag.tools.filter import filter_recalled_results
from coding_rag.tools.recall import expand_with_neighbor_chunks
from coding_rag.rag.prompt import GenerationMode, build_user_prompt, get_system_prompt
from coding_rag.rag.trace import build_retrieval_trace


@dataclass(frozen=True)
class AskModeConfig:
    repo_path: str | Path = "."
    top_k: int = 5
    candidate_k: int | None = None
    chunk_size: int = 40
    overlap: int = 5
    recall_window: int = 2
    max_recall_results: int | None = None
    final_k: int | None = None
    min_final_score: float | None = None
    no_final_filter: bool = False
    generation_mode: GenerationMode = GenerationMode.JUDGE
    max_context_chars: int = 12000
    include_trace_text: bool = False


@dataclass(frozen=True)
class AskModeRun:
    query: str
    plan: "TaskPlan"
    seed_results: list[SearchResult]
    recalled_results: list[SearchResult]
    final_results: list[SearchResult]
    trace: dict
    messages: list[dict[str, str]]
    answer: str | None


def retrieve_for_ask(
    query: str,
    config: AskModeConfig,
) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], dict]:
    python_files = load_python_files(config.repo_path)
    chunks = split_python_files(python_files, chunk_size=config.chunk_size, overlap=config.overlap)
    if not chunks:
        raise ValueError("未找到 Python 代码块，请检查 repo_path 和文件内容")

    retriever = BM25Retriever(chunks)
    candidate_k = config.candidate_k or config.top_k
    seed_results = retriever.search(query, top_k=candidate_k)
    max_recall_results = resolve_max_recall_results(
        candidate_k=candidate_k,
        recall_window=config.recall_window,
        max_recall_results=config.max_recall_results,
    )
    recalled_results = expand_with_neighbor_chunks(
        chunks=chunks,
        seed_results=seed_results,
        window=config.recall_window,
        max_results=max_recall_results,
    )
    final_results = recalled_results
    if not config.no_final_filter:
        final_k = config.final_k if config.final_k is not None else config.top_k
        final_results = filter_recalled_results(
            query=query,
            recalled_results=recalled_results,
            retriever=retriever,
            final_k=None if final_k == 0 else final_k,
            min_score=config.min_final_score,
        )

    trace = build_retrieval_trace(
        query=query,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        params={
            "repo_path": str(config.repo_path),
            "top_k": config.top_k,
            "candidate_k": candidate_k,
            "chunk_size": config.chunk_size,
            "overlap": config.overlap,
            "recall_window": config.recall_window,
            "max_recall_results": max_recall_results,
            "final_k": config.final_k if config.final_k is not None else config.top_k,
            "min_final_score": config.min_final_score,
            "no_final_filter": config.no_final_filter,
            "workflow_mode": "ask",
        },
        include_text=config.include_trace_text,
    )
    return seed_results, recalled_results, final_results, trace


def resolve_max_recall_results(
    candidate_k: int,
    recall_window: int,
    max_recall_results: int | None,
) -> int | None:
    if max_recall_results is None:
        return max(20, candidate_k * ((2 * max(recall_window, 0)) + 1))
    if max_recall_results <= 0:
        return None
    return max_recall_results


def build_ask_messages(
    query: str,
    context: str,
    generation_mode: GenerationMode = GenerationMode.JUDGE,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": get_system_prompt(generation_mode)},
        {"role": "user", "content": build_user_prompt(generation_mode, query, context)},
    ]
