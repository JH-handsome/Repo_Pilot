"""
上下文召回模块
负责扩展检索结果，召回相邻的代码块以提供更多上下文
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

from coding_rag.bm25_retriever import SearchResult
from coding_rag.code_splitter import CodeChunk


def expand_with_neighbor_chunks(
    chunks: list[CodeChunk],
    seed_results: list[SearchResult],
    window: int = 1,
    max_results: int | None = 20,
) -> list[SearchResult]:
    """
    扩展检索种子结果，召回相邻的代码块
    
    检索经常精确命中某一行范围，但会遗漏附近的定义、装饰器、
    辅助函数或返回处理。这个阶段保留种子结果，并添加同一文件中的相邻代码块。
    
    Args:
        chunks: 所有代码块的列表
        seed_results: 检索到的种子结果
        window: 每个种子结果两侧召回的相邻代码块数量
        max_results: 返回的最大结果数量，None 表示不限制
        
    Returns:
        扩展后的 SearchResult 列表
    """
    # 窗口为0或无种子结果时，直接返回种子结果
    if window <= 0 or not seed_results:
        return seed_results[:max_results] if max_results else list(seed_results)

    # 按文件分组代码块
    chunks_by_file = group_chunks_by_file(chunks)
    # 构建代码块到索引的映射
    index_by_key = build_chunk_index(chunks_by_file)
    recalled: OrderedDict[tuple[Path, int, int], SearchResult] = OrderedDict()

    # 遍历每个种子结果，召回其相邻代码块
    for seed_rank, seed in enumerate(seed_results, start=1):
        file_chunks = chunks_by_file.get(seed.chunk.file_path, [])
        seed_index = index_by_key.get(chunk_key(seed.chunk))
        if seed_index is None:
            add_or_upgrade_result(recalled, seed, source=seed.source)
            continue

        # 计算召回的起始和结束索引（不越界）
        start = max(0, seed_index - window)
        end = min(len(file_chunks), seed_index + window + 1)

        for index in range(start, end):
            chunk = file_chunks[index]
            offset = index - seed_index  # 相对于种子结果的偏移量
            # 设置来源标识：种子结果保留原检索来源，召回结果记录种子编号和偏移量。
            source = seed.source if offset == 0 else f"recall:seed-{seed_rank}{offset:+d}"
            result = SearchResult(chunk=chunk, score=seed.score, source=source)
            add_or_upgrade_result(recalled, result, source=source)

        if max_results and len(recalled) >= max_results:
            break

    # 转换为列表并应用结果数量限制
    results = list(recalled.values())
    return results[:max_results] if max_results else results


def group_chunks_by_file(chunks: list[CodeChunk]) -> dict[Path, list[CodeChunk]]:
    """
    按文件路径对代码块进行分组
    
    Args:
        chunks: 代码块列表
        
    Returns:
        以文件路径为键的字典，值为该文件的代码块列表
    """
    grouped: dict[Path, list[CodeChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.file_path].append(chunk)
    return dict(grouped)


def build_chunk_index(
    chunks_by_file: dict[Path, list[CodeChunk]]
) -> dict[tuple[Path, int, int], int]:
    """
    构建代码块索引
    
    为每个文件中的代码块分配索引位置
    
    Args:
        chunks_by_file: 按文件分组的代码块字典
        
    Returns:
        以代码块键为值的字典，值为该代码块在所属文件中的索引
    """
    indexes: dict[tuple[Path, int, int], int] = {}
    for file_chunks in chunks_by_file.values():
        for index, chunk in enumerate(file_chunks):
            indexes[chunk_key(chunk)] = index
    return indexes


def add_or_upgrade_result(
    recalled: OrderedDict[tuple[Path, int, int], SearchResult],
    result: SearchResult,
    source: str,
) -> None:
    """
    添加或升级召回结果
    
    如果结果已存在且当前来源是种子检索结果，则升级为该种子来源。
    
    Args:
        recalled: 已召回的结果字典
        result: 要添加的结果
        source: 结果来源标识
    """
    key = chunk_key(result.chunk)
    existing = recalled.get(key)
    if existing is None:
        recalled[key] = result
        return

    # 如果当前来源是种子检索结果而已存在结果只是相邻召回，则升级。
    if is_seed_source(source) and not is_seed_source(existing.source):
        recalled[key] = SearchResult(chunk=result.chunk, score=result.score, source=source)


def is_seed_source(source: str) -> bool:
    return source in {"hybrid", "bm25"}


def chunk_key(chunk: CodeChunk) -> tuple[Path, int, int]:
    """生成代码块的唯一键"""
    return (chunk.file_path, chunk.start_line, chunk.end_line)
