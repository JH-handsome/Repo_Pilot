"""Hybrid code retriever."""

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

from coding_rag.repository.chunks import CodeChunk
from coding_rag.repository.graph import personalized_dependency_scores
from coding_rag.repository.index import CALL_INTENT_TOKENS, RepoIndex, build_repo_index
from coding_rag.tools.tokenizer import CodeTokenizer


SYMBOL_LINE_RE = re.compile(r"^\s*(class|def|async\s+def)\s+([A-Za-z_][A-Za-z_0-9]*)", re.MULTILINE)
QUALIFIED_CALL_RE = re.compile(r"(?<![A-Za-z_0-9])([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
QUALIFIED_SYMBOL_RE = re.compile(r"(?<![A-Za-z_0-9])([A-Za-z_]\w*)\.([A-Za-z_]\w*)(?![A-Za-z_0-9])")
EXPLICIT_SYMBOL_RE = re.compile(r"(?<![A-Za-z_0-9])([A-Za-z_]\w*_[A-Za-z_0-9]+)(?![A-Za-z_0-9])")
DEFINITION_LOCATION_RE = re.compile(r"\b(?:where|defined|definition|implemented|implementation|located)\b|定义|实现|入口|在哪|哪里|位置", re.IGNORECASE)
CALL_SITE_RE = re.compile(r"\b(?:called|calls|caller|callers|used|usage|references)\b|调用处|谁调用|哪里调用|在哪调用|被调用|引用处", re.IGNORECASE)
TEST_QUERY_TOKENS = {"test", "tests", "testing", "unittest", "pytest", "测试", "单测", "断言"}
DEFAULT_GRAPH_WEIGHT = 0.0
DEFAULT_PATH_WEIGHT = 0.25
PATH_MATCH_STOPWORDS = {
    "agent", "code", "coding", "file", "main", "module", "py", "python",
    "rag", "repo", "repository", "script", "scripts", "test", "tests", "tools",
}


@dataclass(frozen=True)
class SearchResult:
    """Search result."""
    chunk: CodeChunk
    score: float
    source: str = "bm25"


@dataclass(frozen=True)
class PreparedBM25Data:
    """Safe, reusable derived data that avoids repeated AST parsing and tokenization."""

    repo_index: RepoIndex
    content_term_frequencies: tuple[tuple[tuple[str, int], ...], ...]
    metadata_term_frequencies: tuple[tuple[tuple[str, int], ...], ...]
    path_token_sets: tuple[frozenset[str], ...]


class BM25Retriever:
    """Hybrid Search retriever."""
    def __init__(
        self,
        chunks: list[CodeChunk],
        tokenizer: CodeTokenizer | None = None,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
        path_weight: float = DEFAULT_PATH_WEIGHT,
        prepared_data: PreparedBM25Data | None = None,
        path_root: Path | None = None,
    ):
        """构建 BM25 索引，初始化分词器、代码块和元数据倒排索引。"""
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("graph_weight must be between 0 and 1")
        if not 0.0 <= path_weight <= 1.0:
            raise ValueError("path_weight must be between 0 and 1")

        self.chunks = chunks
        self.tokenizer = tokenizer or CodeTokenizer()  # 使用默认分词器
        self.graph_weight = graph_weight
        self.path_weight = path_weight
        # 预先对所有代码块进行分词，或从严格校验的安全缓存恢复派生数据。
        prepared = prepared_data or prepare_bm25_data(
            chunks, self.tokenizer, path_root=path_root
        )
        if not (
            len(prepared.content_term_frequencies)
            == len(prepared.metadata_term_frequencies)
            == len(prepared.path_token_sets)
            == len(chunks)
        ):
            raise ValueError("prepared BM25 data does not match chunks")
        self.repo_index = prepared.repo_index
        self.content_term_frequencies = prepared.content_term_frequencies
        self.metadata_term_frequencies = prepared.metadata_term_frequencies
        self.token_sets = [
            {token for token, _ in content_terms} | {token for token, _ in metadata_terms}
            for content_terms, metadata_terms in zip(
                self.content_term_frequencies, self.metadata_term_frequencies
            )
        ]
        self.path_token_sets = [set(tokens) for tokens in prepared.path_token_sets]
        stem_terms_by_path: dict[Path, frozenset[str]] = {}
        self.stem_term_sets: list[frozenset[str]] = []
        for chunk in chunks:
            path = chunk.file_path
            if path not in stem_terms_by_path:
                stem_terms_by_path[path] = frozenset(
                    identifier_variants(set(self.tokenizer.tokenize(path.stem)))
                    - PATH_MATCH_STOPWORDS
                )
            self.stem_term_sets.append(stem_terms_by_path[path])
        self.bm25 = bm25_from_term_frequencies(self.content_term_frequencies)
        self.metadata_bm25 = bm25_from_term_frequencies(self.metadata_term_frequencies)
        # 构建代码块到索引的快速查找字典
        self.index_by_key = {chunk_key(chunk): index for index, chunk in enumerate(chunks)}
        self.chunk_indexes_by_file: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            self.chunk_indexes_by_file.setdefault(normalize_path(chunk.file_path), []).append(index)

    def prepared_data(self) -> PreparedBM25Data:
        """Return immutable safe-to-serialize derived structures aligned with chunks."""
        return PreparedBM25Data(
            repo_index=self.repo_index,
            content_term_frequencies=self.content_term_frequencies,
            metadata_term_frequencies=self.metadata_term_frequencies,
            path_token_sets=tuple(frozenset(tokens) for tokens in self.path_token_sets),
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """对查询进行混合评分并返回 Top-K 多样化搜索结果。"""
        results, _ = self.search_with_scores(query, top_k=top_k)
        return results

    def search_with_scores(self, query: str, top_k: int = 5) -> tuple[list[SearchResult], list[float]]:
        """Return search results and the same full score vector for downstream reranking."""
        if top_k <= 0:
            return [], []

        scores = self._promote_explicit_definitions(query, self.hybrid_scores(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        selected_indexes = diversify_by_file(ranked_indexes, self.chunks, top_k)
        selected_indexes, scores = self._align_qualified_method_chunks(query, selected_indexes, scores)

        results: list[SearchResult] = []
        for index in selected_indexes:
            results.append(SearchResult(chunk=self.chunks[index], score=scores[index], source="hybrid"))

        return results, scores

    def _promote_explicit_definitions(self, query: str, scores: list[float]) -> list[float]:
        """Surface AST definitions for explicit symbol-location requests, not call-site requests."""
        if not DEFINITION_LOCATION_RE.search(query) or CALL_SITE_RE.search(query):
            return scores
        qualified = set(QUALIFIED_SYMBOL_RE.findall(query))
        unqualified = set(EXPLICIT_SYMBOL_RE.findall(query))
        unqualified.difference_update(method for _, method in qualified)
        if (not qualified and not unqualified) or has_test_intent(set(self.tokenizer.tokenize_query(query))):
            return scores

        promoted = list(scores)
        ceiling = max(scores, default=0.0)
        for path_key, file_index in self.repo_index.files.items():
            indexes = self.chunk_indexes_by_file.get(path_key, ())
            for record in file_index.symbols:
                if record.name not in unqualified and (record.parent, record.name) not in qualified:
                    continue
                covering = [index for index in indexes
                            if self.chunks[index].start_line <= record.start_line <= self.chunks[index].end_line]
                if covering:
                    best = max(covering, key=lambda index: scores[index])
                    promoted[best] = max(promoted[best], ceiling + 0.01 + 0.1 * scores[best])
        return promoted

    def _align_qualified_method_chunks(
        self, query: str, selected_indexes: list[int], scores: list[float]
    ) -> tuple[list[int], list[float]]:
        """Use an exact Class.method() hint inside selected files without changing file scores."""
        mentions = list(dict.fromkeys(QUALIFIED_CALL_RE.findall(query)))
        if not mentions:
            return selected_indexes, scores
        aligned = list(selected_indexes)
        adjusted: list[float] | None = None
        seen_files: set[str] = set()
        for position, selected in enumerate(selected_indexes):
            path_key = normalize_path(self.chunks[selected].file_path)
            if path_key in seen_files:
                continue
            seen_files.add(path_key)
            file_index = self.repo_index.files.get(path_key)
            if file_index is None:
                continue
            indexes = self.chunk_indexes_by_file[path_key]
            for parent, method in mentions:
                symbol = next(
                    (record for record in file_index.symbols
                     if record.parent == parent and record.name == method),
                    None,
                )
                if symbol is None:
                    continue
                covering = [
                    index for index in indexes
                    if self.chunks[index].start_line <= symbol.start_line <= self.chunks[index].end_line
                ]
                if not covering:
                    continue
                target = max(covering, key=lambda index: scores[index])
                if target != selected:
                    if adjusted is None:
                        adjusted = list(scores)
                    adjusted[target] = scores[selected]
                    aligned[position] = target
                    if target in selected_indexes:
                        aligned[selected_indexes.index(target)] = selected
                break
        return aligned, adjusted if adjusted is not None else scores

    def hybrid_scores(self, query: str) -> list[float]:
        """计算融合后的检索分数"""
        tokenized_query = self.tokenizer.tokenize_query(query)
        query_tokens = set(tokenized_query)
        query_path_terms = identifier_variants(query_tokens) - PATH_MATCH_STOPWORDS
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

        lexical_scores = [
            0.50 * content + 0.30 * metadata + 0.10 * overlap + 0.10 * structural
            for content, metadata, overlap, structural in zip(
                content_scores, metadata_scores, overlap_scores, structural_scores
            )
        ]
        path_scores = [
            path_identifier_score(
                query_tokens,
                chunk.file_path,
                self.tokenizer,
                path_terms,
                prepared_stem_terms=stem_terms,
                prepared_query_terms=query_path_terms,
            )
            for chunk, path_terms, stem_terms in zip(
                self.chunks, self.path_token_sets, self.stem_term_sets
            )
        ]
        lexical_scores = [
            (1.0 - self.path_weight) * lexical + self.path_weight * path_score
            for lexical, path_score in zip(lexical_scores, path_scores)
        ]
        graph_scores = personalized_dependency_scores(
            self.repo_index,
            [chunk.file_path for chunk in self.chunks],
            lexical_scores,
        ) if self.graph_weight else [0.0 for _ in self.chunks]

        return [
            adjust_score_for_document_role(
                (1.0 - self.graph_weight) * lexical + self.graph_weight * graph,
                query_tokens,
                chunk.file_path,
            )
            for chunk, lexical, graph in zip(self.chunks, lexical_scores, graph_scores)
        ]

    def rescore_results(
        self,
        query: str,
        results: list[SearchResult],
        scores: list[float] | None = None,
    ) -> list[SearchResult]:
        """基于 Hybrid Search 对现有结果重新评分"""
        if scores is None:
            scores = self.hybrid_scores(query)
        elif len(scores) != len(self.chunks):
            raise ValueError("scores must match indexed chunks")

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


def chunk_to_document(chunk: CodeChunk, path_root: Path | None = None) -> str:
    """将代码块转换为包含文件路径和文本内容的文档字符串。"""
    return f"{display_path(chunk.file_path, path_root)}\n{chunk.text}"


def prepare_bm25_data(
    chunks: list[CodeChunk],
    tokenizer: CodeTokenizer | None = None,
    repo_index: RepoIndex | None = None,
    path_root: Path | None = None,
) -> PreparedBM25Data:
    """Build reusable AST metadata and tokenized corpora without creating BM25 statistics."""
    active_tokenizer = tokenizer or CodeTokenizer()
    active_repo_index = repo_index or build_repo_index(chunks, path_root=path_root)
    content_term_frequencies = tuple(
        term_frequencies(active_tokenizer.tokenize(chunk_to_document(chunk, path_root)))
        for chunk in chunks
    )
    metadata_term_frequencies = tuple(
        term_frequencies(active_tokenizer.tokenize(
            chunk_to_metadata_document(chunk, active_repo_index, path_root)))
        for chunk in chunks
    )
    path_token_sets = tuple(
        frozenset(
            identifier_variants(
                set(
                    active_tokenizer.tokenize(
                        display_path(chunk.file_path, path_root).replace("/", " ").replace(".", " ")
                    )
                )
            )
            - PATH_MATCH_STOPWORDS
        )
        for chunk in chunks
    )
    return PreparedBM25Data(
        active_repo_index, content_term_frequencies, metadata_term_frequencies, path_token_sets
    )


def term_frequencies(tokens: list[str]) -> tuple[tuple[str, int], ...]:
    """Return sparse frequencies in deterministic first-occurrence order."""
    return tuple(Counter(tokens).items())


def bm25_from_term_frequencies(
    rows: tuple[tuple[tuple[str, int], ...], ...],
) -> BM25Okapi:
    """Build rank-bm25 numeric state directly from validated sparse rows."""
    if not rows:
        raise ValueError("BM25 corpus must not be empty")
    # Compatibility contract: rank-bm25 0.2.x exposes these numeric fields and
    # _calc_idf; requirements.txt deliberately excludes later internal layouts.
    index = BM25Okapi.__new__(BM25Okapi)
    index.k1 = 1.5
    index.b = 0.75
    index.epsilon = 0.25
    index.corpus_size = len(rows)
    index.doc_freqs = [dict(row) for row in rows]
    index.doc_len = [sum(count for _, count in row) for row in rows]
    index.avgdl = sum(index.doc_len) / index.corpus_size
    index.tokenizer = None
    index.idf = {}
    document_frequency: dict[str, int] = {}
    for row in rows:
        for token, _ in row:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    index._calc_idf(document_frequency)
    return index


def chunk_to_metadata_document(
    chunk: CodeChunk,
    repo_index: RepoIndex | None = None,
    path_root: Path | None = None,
) -> str:
    """Extract path, symbol, import, and call metadata."""
    symbols = " ".join(match.group(2) for match in SYMBOL_LINE_RE.finditer(chunk.text))
    path = display_path(chunk.file_path, path_root)
    repo_metadata = repo_index.metadata_for_chunk(chunk) if repo_index is not None else ""
    return f"{path} {path.replace('/', ' ').replace('.', ' ')} {symbols} {repo_metadata}"


def display_path(path: object, path_root: Path | None) -> str:
    """Render a repository-relative metadata path without changing real chunk paths."""
    candidate = Path(path)
    if path_root is not None:
        try:
            candidate = candidate.relative_to(path_root)
        except ValueError:
            try:
                candidate = candidate.resolve().relative_to(path_root.resolve())
            except ValueError:
                candidate = Path(candidate.name)
    return normalize_path(candidate)


def normalize_scores(scores) -> list[float]:
    """将分数列表归一化到 [0,1] 区间。"""
    values = [float(score) for score in scores]
    if not values:
        return []
    max_score = max(values)
    if max_score <= 0:
        return [0.0 for _ in values]
    return [score / max_score for score in values]


def token_overlap_score(query_tokens: set[str], document_tokens: set[str]) -> float:
    """计算查询词元在文档词元中的覆盖率（交集 / 查询词元数）。"""
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def path_identifier_score(
    query_tokens: set[str],
    path: object,
    tokenizer: CodeTokenizer,
    prepared_path_terms: set[str] | None = None,
    *,
    prepared_stem_terms: frozenset[str] | None = None,
    prepared_query_terms: set[str] | None = None,
) -> float:
    """Score exact identifier matches against path segments and filenames."""
    query_terms = prepared_query_terms
    if query_terms is None:
        query_terms = identifier_variants(query_tokens) - PATH_MATCH_STOPWORDS
    stem_terms = prepared_stem_terms
    if stem_terms is None:
        stem_terms = identifier_variants(set(tokenizer.tokenize(Path(path).stem))) - PATH_MATCH_STOPWORDS
    path_terms = prepared_path_terms
    if path_terms is None:
        path_tokens = set(tokenizer.tokenize(normalize_path(path).replace("/", " ").replace(".", " ")))
        path_terms = identifier_variants(path_tokens) - PATH_MATCH_STOPWORDS
    matches = query_terms & path_terms
    score = min(len(matches) / 2.0, 1.0)
    if stem_terms and stem_terms <= query_terms:
        score = 1.0
    if query_tokens & CALL_INTENT_TOKENS and score < 1.0:
        score *= 0.25
    return score


def identifier_variants(tokens: set[str]) -> set[str]:
    """Add a conservative singular form for ASCII identifier tokens."""
    variants: set[str] = set()
    for token in tokens:
        normalized = token.casefold()
        if len(normalized) < 3 or not normalized.isascii() or not normalized.replace("_", "").isalnum():
            continue
        variants.add(normalized)
        if len(normalized) > 4 and normalized.endswith("s"):
            variants.add(normalized[:-1])
    return variants


def adjust_score_for_document_role(score: float, query_tokens: set[str], path: object) -> float:
    """根据查询意图轻量调整测试文件和实现文件的相对顺序"""
    if not is_test_path(path):
        return score

    if has_test_intent(query_tokens):
        return score * 1.05
    return score * 0.85


def has_test_intent(query_tokens: set[str]) -> bool:
    """检测查询词元是否包含测试意图关键词。"""
    return bool(query_tokens & TEST_QUERY_TOKENS)


def is_test_path(path: object) -> bool:
    """检测路径是否符合测试文件命名惯例。"""
    normalized = normalize_path(path)
    filename = normalized.rsplit("/", 1)[-1]
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    )


def normalize_path(path: object) -> str:
    """统一路径分隔符并转为小写。"""
    return str(path).replace("\\", "/").casefold()


def diversify_by_file(ranked_indexes: list[int], chunks: list[CodeChunk], top_k: int) -> list[int]:
    """优先选择不同文件，不足时用文件内候补补齐。"""
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
    """构建由文件路径和起止行号组成的代码块身份元组。"""
    return (chunk.file_path, chunk.start_line, chunk.end_line)
