"""Rank indexed Python functions inside one already-selected source file."""

from __future__ import annotations

import re
from math import log1p, sqrt
from pathlib import Path

from coding_rag.repository.index import normalize_path, rebuild_source
from coding_rag.tools.bm25 import BM25Retriever, QUALIFIED_SYMBOL_RE, identifier_variants


QUERY_STOPWORDS = {
    "and", "are", "both", "call", "called", "class", "code", "does", "file",
    "find", "for", "from", "function", "helper", "how", "implementation",
    "in", "is", "it", "its", "locate", "method", "of", "or", "the", "to",
    "use", "uses", "when", "where", "which", "with",
}
SYMBOL_LOCATOR_VERSION = "file-local-symbol-v1"
EXACT_DEFINITION_LOOKUP_RE = re.compile(
    r"\b(?:defined|definition|locate|implemented|implementation)\b|定义|实现|定位",
    re.IGNORECASE,
)


def missing_qualified_definitions(retriever: BM25Retriever, query: str) -> list[str]:
    """Find named Class.method references lacking a direct indexed definition."""
    references = list(dict.fromkeys(QUALIFIED_SYMBOL_RE.findall(query)))
    if not references:
        return []
    classes = set()
    definitions = set()
    for file_index in retriever.repo_index.files.values():
        for record in file_index.symbols:
            if record.kind == "class":
                classes.add(record.name)
            elif record.kind in {"function", "async_function"} and record.parent:
                definitions.add((record.parent, record.name))
    return [f"{parent}.{name}" for parent, name in references
            if parent in classes and (parent, name) not in definitions]


def exact_definition_lookup_should_abstain(query: str, missing: list[str]) -> bool:
    """Abstain only on one explicit missing definition request, not general behavior queries."""
    return bool(missing and len(set(QUALIFIED_SYMBOL_RE.findall(query))) == 1
                and EXACT_DEFINITION_LOOKUP_RE.search(query))


def locate_file_symbols(
    retriever: BM25Retriever, path: Path, query: str, *, limit: int | None = 5,
    kind: str = "function",
) -> list[dict]:
    """Rank real definitions by query overlap; never infer a symbol from a chunk alone."""
    if (limit is not None and not 1 <= limit <= 20) or not query.strip():
        raise ValueError("locate requires a query and limit between 1 and 20")
    if kind not in {"function", "class", "all"}:
        raise ValueError("kind must be function, class, or all")
    key = normalize_path(path.resolve())
    file_index = retriever.repo_index.files.get(key)
    chunk_indexes = retriever.chunk_indexes_by_file.get(key)
    if file_index is None or not chunk_indexes:
        raise ValueError("path is not an indexed Python file")
    lines = rebuild_source([retriever.chunks[index] for index in chunk_indexes]).splitlines()
    tokenizer = retriever.tokenizer
    query_terms = identifier_variants(set(tokenizer.tokenize_query(query))) - QUERY_STOPWORDS
    if not query_terms:
        return []

    documents = []
    qualified_names: list[str] = []
    for position, record in enumerate(file_index.symbols):
        containers = [previous for previous in range(position)
                      if file_index.symbols[previous].start_line < record.start_line
                      and record.end_line <= file_index.symbols[previous].end_line]
        parent = min(containers, key=lambda previous:
                     file_index.symbols[previous].end_line - file_index.symbols[previous].start_line
                     ) if containers else None
        qualified = f"{qualified_names[parent]}.{record.name}" if parent is not None else record.name
        qualified_names.append(qualified)
        if not ((kind in {"function", "all"} and record.kind in {"function", "async_function"})
                or (kind in {"class", "all"} and record.kind == "class")):
            continue
        name_terms = identifier_variants(set(tokenizer.tokenize(qualified)))
        signature_terms = identifier_variants(set(tokenizer.tokenize(record.signature)))
        body = "\n".join(lines[record.start_line - 1:record.end_line])
        body_tokens = tokenizer.tokenize(body)
        body_terms = identifier_variants(set(body_tokens))
        documents.append((record, qualified, name_terms, signature_terms, body_terms, len(body_tokens)))
    if not documents:
        return []

    frequency = {term: sum(term in document[4] for document in documents) for term in query_terms}
    weights = {term: log1p((len(documents) + 1) / (count + 1))
               for term, count in frequency.items()}
    results = []
    for record, qualified, name_terms, signature_terms, body_terms, body_length in documents:
        name_score = sum(weights[term] for term in query_terms & name_terms)
        signature_score = sum(weights[term] for term in query_terms & signature_terms)
        body_score = sum(weights[term] for term in query_terms & body_terms)
        # Long functions naturally contain many incidental query words. Bound their
        # body contribution without losing exact identifier/name evidence.
        body_scale = 1.0 / max(1.0, sqrt(body_length / 80))
        score = 4.0 * name_score + 0.5 * signature_score + body_scale * body_score
        if score <= 0:
            continue
        results.append({"name": qualified, "kind": record.kind,
                        "start_line": record.start_line, "end_line": record.end_line,
                        "score": round(score, 6)})
    ranked = sorted(results, key=lambda row: (-row["score"], row["start_line"], row["name"]))
    return ranked if limit is None else ranked[:limit]


def locate_retrieved_files(
    retriever: BM25Retriever,
    root: Path,
    paths: list[Path],
    query: str,
    *,
    file_limit: int = 5,
    symbols_per_file: int = 3,
    kind: str = "function",
) -> list[dict]:
    """Shortlist definitions from retrieved files without changing their file order."""
    if not 1 <= file_limit <= 10 or not 1 <= symbols_per_file <= 20:
        raise ValueError("file_limit must be 1..10 and symbols_per_file 1..20")
    if kind not in {"function", "class", "all"}:
        raise ValueError("kind must be function, class, or all")
    root = root.resolve()
    groups = []
    seen = set()
    for search_rank, raw_path in enumerate(paths, 1):
        path = raw_path.resolve()
        relative = path.relative_to(root).as_posix()
        if path in seen:
            continue
        seen.add(path)
        candidates = locate_file_symbols(
            retriever, path, query, limit=symbols_per_file, kind=kind,
        )
        groups.append({"path": relative, "search_rank": search_rank,
                       "candidates": candidates})
        if len(groups) >= file_limit:
            break
    return groups
