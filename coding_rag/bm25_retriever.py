"""Hybrid code retriever."""

from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

from coding_rag.code_splitter import CodeChunk
from coding_rag.repo_index import RepoIndex, build_repo_index
from coding_rag.tokenizer import CodeTokenizer


SYMBOL_LINE_RE = re.compile(r"^\s*(class|def|async\s+def)\s+([A-Za-z_][A-Za-z_0-9]*)", re.MULTILINE)
TEST_QUERY_TOKENS = {"test", "tests", "testing", "unittest", "pytest", "测试", "单测", "断言"}


@dataclass(frozen=True)
class SearchResult:
    """Search result."""
    chunk: CodeChunk
    score: float
    source: str = "bm25"


class BM25Retriever:
    """Hybrid Search retriever."""
    def __init__(self, chunks: list[CodeChunk], tokenizer: CodeTokenizer | None = None):
        if not chunks:
            raise ValueError("chunks must not be empty")

        self.chunks = chunks
        self.tokenizer = tokenizer or CodeTokenizer()  # 使用默认分词器
        # 预先对所有代码块进行分词，分别建立正文索引和路径/符号索引。
        self.repo_index = build_repo_index(chunks)
        self.tokenized_chunks = [self.tokenizer.tokenize(chunk_to_document(chunk)) for chunk in chunks]
        self.tokenized_metadata = [
            self.tokenizer.tokenize(chunk_to_metadata_document(chunk, self.repo_index))
            for chunk in chunks
        ]
        self.token_sets = [
            set(content_tokens) | set(metadata_tokens)
            for content_tokens, metadata_tokens in zip(self.tokenized_chunks, self.tokenized_metadata)
        ]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.metadata_bm25 = BM25Okapi(self.tokenized_metadata)
        # 构建代码块到索引的快速查找字典
        self.index_by_key = {chunk_key(chunk): index for index, chunk in enumerate(chunks)}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []

        scores = self.hybrid_scores(query)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        selected_indexes = diversify_by_file(ranked_indexes, self.chunks, top_k)

        results: list[SearchResult] = []
        for index in selected_indexes:
            results.append(SearchResult(chunk=self.chunks[index], score=scores[index], source="hybrid"))

        return results

    def hybrid_scores(self, query: str) -> list[float]:
        """计算融合后的检索分数"""
        tokenized_query = self.tokenizer.tokenize(query)
        query_tokens = set(tokenized_query)
        content_scores = normalize_scores(self.bm25.get_scores(tokenized_query))
        metadata_scores = normalize_scores(self.metadata_bm25.get_scores(tokenized_query))
        overlap_scores = [
            token_overlap_score(query_tokens, token_set)
            for token_set in self.token_sets
        ]
        structural_scores = [
            self.repo_index.structural_score(query_tokens, chunk)
            for chunk in self.chunks
        ]

        return [
            adjust_score_for_document_role(
                0.50 * content + 0.30 * metadata + 0.10 * overlap + 0.10 * structural,
                query_tokens,
                chunk.file_path,
            )
            for chunk, content, metadata, overlap, structural in zip(
                self.chunks,
                content_scores,
                metadata_scores,
                overlap_scores,
                structural_scores,
            )
        ]

    def rescore_results(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """基于 Hybrid Search 对现有结果重新评分"""
        scores = self.hybrid_scores(query)

        rescored_results: list[SearchResult] = []
        for result in results:
            index = self.index_by_key.get(chunk_key(result.chunk))
            if index is None:
                continue

            rescored_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=scores[index],
                    source=result.source,
                )
            )

        return rescored_results


def chunk_to_document(chunk: CodeChunk) -> str:
    return f"{chunk.file_path}\n{chunk.text}"


def chunk_to_metadata_document(chunk: CodeChunk, repo_index: RepoIndex | None = None) -> str:
    """Extract path, symbol, import, and call metadata."""
    symbols = " ".join(match.group(2) for match in SYMBOL_LINE_RE.finditer(chunk.text))
    path = normalize_path(chunk.file_path)
    repo_metadata = repo_index.metadata_for_chunk(chunk) if repo_index is not None else ""
    return f"{path} {path.replace('/', ' ').replace('.', ' ')} {symbols} {repo_metadata}"


def normalize_scores(scores) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []
    max_score = max(values)
    if max_score <= 0:
        return [0.0 for _ in values]
    return [score / max_score for score in values]


def token_overlap_score(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def adjust_score_for_document_role(score: float, query_tokens: set[str], path: object) -> float:
    """根据查询意图轻量调整测试文件和实现文件的相对顺序"""
    if not is_test_path(path):
        return score

    if has_test_intent(query_tokens):
        return score * 1.05
    return score * 0.85


def has_test_intent(query_tokens: set[str]) -> bool:
    return bool(query_tokens & TEST_QUERY_TOKENS)


def is_test_path(path: object) -> bool:
    normalized = normalize_path(path)
    filename = normalized.rsplit("/", 1)[-1]
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


def normalize_path(path: object) -> str:
    return str(path).replace("\\", "/").casefold()


def diversify_by_file(ranked_indexes: list[int], chunks: list[CodeChunk], top_k: int) -> list[int]:
    selected: list[int] = []
    selected_files: set[str] = set()
    deferred: list[int] = []

    for index in ranked_indexes:
        file_key = normalize_path(chunks[index].file_path)
        if file_key in selected_files:
            deferred.append(index)
            continue

        selected.append(index)
        selected_files.add(file_key)
        if len(selected) == top_k:
            return selected

    for index in deferred:
        selected.append(index)
        if len(selected) == top_k:
            break

    return selected


def chunk_key(chunk: CodeChunk) -> tuple[object, int, int]:
    return (chunk.file_path, chunk.start_line, chunk.end_line)
