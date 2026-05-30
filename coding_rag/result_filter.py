"""
结果过滤模块
负责对召回的结果进行重新评分和过滤，保留最相关的上下文
"""

from __future__ import annotations

from coding_rag.bm25_retriever import BM25Retriever, SearchResult


def filter_recalled_results(
    query: str,
    recalled_results: list[SearchResult],
    retriever: BM25Retriever,
    final_k: int | None,
    min_score: float | None = None,
) -> list[SearchResult]:
    """
    对召回的代码块重新评分并保留最相关的最终上下文
    
    召回阶段有意引入相邻的代码块。这个过滤阶段的问题是：
    在扩展之后，哪些代码块仍然看起来与原始查询最相关？
    
    Args:
        query: 原始查询文本
        recalled_results: 召回的结果列表
        retriever: 检索器，用于重新评分
        final_k: 最终保留的结果数量，None 表示不限制
        min_score: 最低分数阈值，低于此值的结果将被过滤
        
    Returns:
        过滤后的 SearchResult 列表，按相关性降序排列
    """
    # 无结果或最终数量为0时返回空列表
    if not recalled_results:
        return []
    if final_k is not None and final_k <= 0:
        return []

    # 基于原始查询重新评分召回的结果
    rescored_results = retriever.rescore_results(query, recalled_results)
    # 为每个结果添加索引，保持召回顺序
    indexed_results = list(enumerate(rescored_results))

    # 应用最低分数过滤
    if min_score is not None:
        indexed_results = [
            (index, result)
            for index, result in indexed_results
            if result.score >= min_score
        ]

    # 按分数、来源优先级和召回顺序排序
    indexed_results.sort(key=sort_key, reverse=True)
    # 提取结果列表
    final_results = [result for _, result in indexed_results]

    if final_k is None:
        return final_results
    return final_results[:final_k]


def sort_key(index_and_result: tuple[int, SearchResult]) -> tuple[float, int, int]:
    """
    生成排序键
    
    排序优先级：
    1. 检索分数（越高越好）
    2. 来源优先级（hybrid/bm25 > recall）
    3. 召回顺序（越早越好，使用负索引实现）
    
    Args:
        index_and_result: (索引, SearchResult) 元组
        
    Returns:
        用于排序的元组键
    """
    index, result = index_and_result  # 解包索引和结果
    return (result.score, source_priority(result.source), -index)


def source_priority(source: str) -> int:
    """
    获取来源优先级
    
    Args:
        source: 来源标识字符串
        
    Returns:
        优先级值，hybrid/bm25 返回 1，其他返回 0
    """
    return 1 if source in {"hybrid", "bm25"} else 0
