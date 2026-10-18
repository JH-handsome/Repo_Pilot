"""统一 Agent 复用的代码检索流水线。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_rag.repository.search_index import (
    DEFAULT_INDEX_EXCLUDE_ROOTS,
    GLOBAL_REPOSITORY_INDEX_REGISTRY,
    RepositoryIndexKey,
    RepositoryIndexRegistry,
)
from coding_rag.tools.bm25 import DEFAULT_GRAPH_WEIGHT, DEFAULT_PATH_WEIGHT, SearchResult
from coding_rag.tools.filter import filter_recalled_results
from coding_rag.tools.recall import expand_with_neighbor_chunks
from coding_rag.rag.prompt import GenerationMode
from coding_rag.rag.trace import build_retrieval_trace


@dataclass(frozen=True)
class RetrievalConfig:
    """统一 Agent 的代码检索配置。

    控制代码检索流水线的各项参数，包括 chunk 切分、BM25 初检、
    邻居召回、最终过滤和 generation mode 选择。
    """

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
    graph_weight: float = DEFAULT_GRAPH_WEIGHT
    path_weight: float = DEFAULT_PATH_WEIGHT
    exclude_roots: tuple[str, ...] = DEFAULT_INDEX_EXCLUDE_ROOTS


def retrieve_code(
    query: str,
    config: RetrievalConfig,
    *,
    index_registry: RepositoryIndexRegistry | None = None,
) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], dict]:
    """执行代码检索流水线。

    依次完成文件加载、chunk 切分、BM25 初检、邻居召回和最终过滤，
    返回 seed_results、recalled_results、final_results 及 trace 字典。

    Args:
        query: 用户查询字符串。
        config: 检索配置。

    Returns:
        四元组 (seed_results, recalled_results, final_results, trace)：
        - seed_results: BM25 初检候选列表。
        - recalled_results: 邻居扩展后的召回列表。
        - final_results: 最终过滤后的结果列表。
        - trace: 全链路 trace 信息字典。

    Raises:
        ValueError: 未找到 Python 代码块。
    """
    registry = index_registry or GLOBAL_REPOSITORY_INDEX_REGISTRY
    access = registry.get(RepositoryIndexKey.create(
        config.repo_path,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        graph_weight=config.graph_weight,
        path_weight=config.path_weight,
        exclude_roots=config.exclude_roots,
    ))
    chunks = list(access.index.chunks)
    retriever = access.index.retriever
    candidate_k = config.candidate_k or config.top_k
    seed_results, scores = retriever.search_with_scores(query, top_k=candidate_k)
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
        # final_filter 会重新评分召回结果；如果排查“为什么没命中”，优先看 trace 里的这三个阶段。
        final_k = config.final_k if config.final_k is not None else config.top_k
        final_results = filter_recalled_results(
            query=query,
            recalled_results=recalled_results,
            retriever=retriever,
            final_k=None if final_k == 0 else final_k,
            min_score=config.min_final_score,
            scores=scores,
        )

    # trace 同时保留旧 stages 和新 events，方便前端、评测脚本和调试报告逐步迁移。
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
            "workflow_mode": "retrieval",
            "retriever_version": access.index.key.retriever_version,
            "graph_weight": config.graph_weight,
            "path_weight": config.path_weight,
            "exclude_roots": list(config.exclude_roots),
        },
        include_text=config.include_trace_text,
    )
    index_metadata = access.trace_metadata()
    trace["artifacts"]["index"] = index_metadata
    trace["summary"].update({
        "index_status": index_metadata["status"],
        "index_lifecycle": index_metadata["lifecycle"],
        "index_build_ms": index_metadata["index_build_ms"],
        "index_disk_read_ms": index_metadata["disk_read_ms"],
        "index_disk_write_ms": index_metadata["disk_write_ms"],
        "index_disk_write_status": index_metadata["disk_write_status"],
        "manifest_scan_ms": index_metadata["manifest_scan_ms"],
        "indexed_file_count": index_metadata["file_count"],
        "indexed_chunk_count": index_metadata["chunk_count"],
        "changed_file_count": index_metadata["changed_file_count"],
        "reused_file_count": index_metadata["reused_file_count"],
        "rebuilt_file_count": index_metadata["rebuilt_file_count"],
        "removed_file_count": index_metadata["removed_file_count"],
        "reused_chunk_count": index_metadata["reused_chunk_count"],
        "rebuilt_chunk_count": index_metadata["rebuilt_chunk_count"],
        "removed_chunk_count": index_metadata["removed_chunk_count"],
    })
    return seed_results, recalled_results, final_results, trace


def resolve_max_recall_results(
    candidate_k: int,
    recall_window: int,
    max_recall_results: int | None,
) -> int | None:
    """解析召回阶段的最大结果数。

    若未显式指定，则根据候选数和召回窗口自动推算一个下界（至少 20）；
    若显式指定且为正数则直接使用；若为 0 或负数则返回 None 表示不限制。

    Args:
        candidate_k: 初检候选数量。
        recall_window: 邻居召回窗口大小。
        max_recall_results: 显式指定的最大召回结果数，可为 None。

    Returns:
        最大召回结果数，None 表示不限制。
    """
    if max_recall_results is None:
        return max(20, candidate_k * ((2 * max(recall_window, 0)) + 1))
    if max_recall_results <= 0:
        return None
    return max_recall_results


# 旧检索 API 的兼容别名；不再代表一个可选的 ASK 工作流。
AskModeConfig = RetrievalConfig
retrieve_for_ask = retrieve_code
