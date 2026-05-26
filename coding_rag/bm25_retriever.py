"""
BM25 检索器模块
使用 BM25 算法进行代码块的相似度搜索
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from coding_rag.code_splitter import CodeChunk
from coding_rag.tokenizer import CodeTokenizer


@dataclass(frozen=True)
class SearchResult:
    """
    表示搜索结果的不可变数据类
    
    Attributes:
        chunk: 匹配的代码块
        score: 相关性得分
        source: 检索来源标识
    """
    chunk: CodeChunk
    score: float
    source: str = "bm25"


class BM25Retriever:
    """
    基于 BM25 算法的代码检索器
    
    使用 rank_bm25 库实现 BM25Okapi 算法进行文本相似度搜索
    
    Attributes:
        chunks: 所有可搜索的代码块列表
        tokenizer: 分词器
        tokenized_chunks: 分词后的代码块列表
        bm25: BM25 索引器
        index_by_key: 代码块到索引的映射字典
    """
    def __init__(self, chunks: list[CodeChunk], tokenizer: CodeTokenizer | None = None):
        """
        初始化 BM25 检索器
        
        Args:
            chunks: 代码块列表
            tokenizer: 自定义分词器，默认使用 CodeTokenizer
            
        Raises:
            ValueError: 代码块列表为空时抛出
        """
        if not chunks:
            raise ValueError("chunks must not be empty")

        self.chunks = chunks
        self.tokenizer = tokenizer or CodeTokenizer()  # 使用默认分词器
        # 预先对所有代码块进行分词
        self.tokenized_chunks = [self.tokenizer.tokenize(chunk_to_document(chunk)) for chunk in chunks]
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        # 构建代码块到索引的快速查找字典
        self.index_by_key = {chunk_key(chunk): index for index, chunk in enumerate(chunks)}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        搜索与查询最相关的代码块
        
        Args:
            query: 搜索查询文本
            top_k: 返回的最相关结果数量
            
        Returns:
            SearchResult 对象列表，按相关性降序排列
        """
        if top_k <= 0:
            return []

        # 对查询进行分词
        tokenized_query = self.tokenizer.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 按分数降序排列索引
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

        results: list[SearchResult] = []
        for index in ranked_indexes[:top_k]:
            results.append(SearchResult(chunk=self.chunks[index], score=float(scores[index])))

        return results

    def rescore_results(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """基于原始 BM25 语料库对现有结果重新评分"""
        tokenized_query = self.tokenizer.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        rescored_results: list[SearchResult] = []
        for result in results:
            index = self.index_by_key.get(chunk_key(result.chunk))
            if index is None:
                continue

            rescored_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=float(scores[index]),
                    source=result.source,
                )
            )

        return rescored_results


def chunk_to_document(chunk: CodeChunk) -> str:
    """
    将代码块转换为文档字符串
    
    同时搜索代码文本和文件路径，提高检索准确性
    
    Args:
        chunk: 代码块
        
    Returns:
        包含文件路径和代码内容的字符串
    """
    return f"{chunk.file_path}\n{chunk.text}"


def chunk_key(chunk: CodeChunk) -> tuple[object, int, int]:
    """
    生成代码块的唯一键
    
    Args:
        chunk: 代码块
        
    Returns:
        包含文件路径和行号范围的元组
    """
    return (chunk.file_path, chunk.start_line, chunk.end_line)
