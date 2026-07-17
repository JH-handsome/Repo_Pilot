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
    """ASK 模式检索配置。

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


@dataclass(frozen=True)
class AskModeRun:
    """ASK 模式单次检索运行的完整结果。

    记录从查询输入到最终回答的全链路数据，包括初检结果、
    召回结果、最终过滤结果、trace 信息以及组装后的消息和回答。
    """

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
    """执行 ASK 模式检索流水线。

    依次完成文件加载、chunk 切分、BM25 初检、邻居召回和最终过滤，
    返回 seed_results、recalled_results、final_results 及 trace 字典。

    Args:
        query: 用户查询字符串。
        config: ASK 检索配置。

    Returns:
        四元组 (seed_results, recalled_results, final_results, trace)：
        - seed_results: BM25 初检候选列表。
        - recalled_results: 邻居扩展后的召回列表。
        - final_results: 最终过滤后的结果列表。
        - trace: 全链路 trace 信息字典。

    Raises:
        ValueError: 未找到 Python 代码块。
    """
    # ASK 检索的阅读顺序：文件加载 -> 切 chunk -> 初检索 -> 邻居召回 -> 最终过滤 -> trace。
    python_files = load_python_files(config.repo_path)
    chunks = split_python_files(python_files, chunk_size=config.chunk_size, overlap=config.overlap)
    if not chunks:
        raise ValueError("未找到 Python 代码块，请检查 repo_path 和文件内容")

    # seed_results 是 BM25/结构化检索的第一批候选，后续召回和过滤都围绕它展开。
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
        # final_filter 会重新评分召回结果；如果排查“为什么没命中”，优先看 trace 里的这三个阶段。
        final_k = config.final_k if config.final_k is not None else config.top_k
        final_results = filter_recalled_results(
            query=query,
            recalled_results=recalled_results,
            retriever=retriever,
            final_k=None if final_k == 0 else final_k,
            min_score=config.min_final_score,
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


def build_ask_messages(
    query: str,
    context: str,
    generation_mode: GenerationMode = GenerationMode.JUDGE,
) -> list[dict[str, str]]:
    """组装 ASK 模式的 LLM 对话消息。

    根据 generation mode 构建 system prompt 和 user prompt，
    返回标准的 messages 列表供 LLM 调用。

    Args:
        query: 用户查询字符串。
        context: 检索到的代码上下文文本。
        generation_mode: 生成模式，默认为 JUDGE。

    Returns:
        messages 列表，包含 system 和 user 两个角色消息。
    """
    return [
        {"role": "system", "content": get_system_prompt(generation_mode)},
        {"role": "user", "content": build_user_prompt(generation_mode, query, context)},
    ]
