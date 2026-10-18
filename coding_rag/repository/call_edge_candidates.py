"""Query-ranked, source-grounded cross-file call edges for optional investigation."""

from __future__ import annotations

import re
from math import isfinite
from pathlib import Path

from coding_rag.repository.call_path import PythonCallPathIndex
from coding_rag.repository.index import normalize_path, rebuild_source
from coding_rag.repository.symbol_locator import locate_retrieved_files
from coding_rag.tools.bm25 import BM25Retriever, identifier_variants


LINE_STOPWORDS = {
    "a", "and", "as", "file", "from", "helper", "in", "is", "it", "locate",
    "of", "the", "to", "when", "where", "which", "with",
}
FUNCTION_INTENT = re.compile(r"\b(helper|formatter|utility|function)\b", re.I)
CALL_EDGE_CANDIDATE_VERSION = "call-edge-candidates-v1"
AUTO_SOURCE_CANDIDATE_VERSION = "query-only-auto-source-v1"


def first_non_test_source_candidate(
    retriever: BM25Retriever, root: Path, query: str, paths: list[Path],
) -> dict | None:
    """Choose an unverified query-only source hint from at most five retrieved files."""
    groups = locate_retrieved_files(
        retriever, root, paths, query, file_limit=5, symbols_per_file=1,
    )
    selected = next(
        (group for group in groups
         if group["candidates"] and not group["path"].startswith("tests/")),
        None,
    )
    if selected is None:
        return None
    return {
        "source_path": selected["path"],
        "source_symbol": selected["candidates"][0]["name"],
        "source_search_rank": selected["search_rank"],
        "origin": "query_only_first_non_test_file_first_symbol",
    }


def query_only_call_edges(
    retriever: BM25Retriever,
    call_index: PythonCallPathIndex,
    root: Path,
    query: str,
    source_paths: list[Path],
    *,
    line_weight: float = 1.0,
) -> tuple[dict | None, list[dict], str | None]:
    """Trace one unverified source hint, abstaining on absent or ambiguous symbols."""
    source = first_non_test_source_candidate(retriever, root, query, source_paths)
    if source is None:
        return None, [], "no_non_test_function_candidate"
    try:
        edges = candidate_call_edges(
            retriever, call_index, root, query,
            [root / source["source_path"]], source_files=1,
            line_weight=line_weight, source_symbol=source["source_symbol"],
        )
    except ValueError as error:
        if str(error) not in {
            "source symbol is missing or ambiguous in the selected file",
            "source path is not an indexed Python file",
        }:
            raise
        return source, [], "untraceable_source_candidate"
    return source, edges, None


def score_call_edge(
    query: str,
    query_terms: set[str],
    call_name: str,
    target_symbol: str,
    source_symbol: str,
    target_path: Path,
    target_kind: str,
    source_line: str,
    retriever: BM25Retriever,
    line_weight: float,
) -> float:
    """Rank an actual call edge using query words, names, and source line."""
    tokenize = retriever.tokenizer.tokenize
    call_terms = identifier_variants(set(tokenize(call_name)))
    target_terms = identifier_variants(set(tokenize(target_symbol)))
    source_terms = identifier_variants(set(tokenize(source_symbol)))
    path_terms = identifier_variants(set(tokenize(target_path.stem)))
    score = (
        2.0 * len(query_terms & call_terms)
        + len(query_terms & target_terms)
        + 0.5 * len(query_terms & source_terms)
        + 0.75 * len(query_terms & path_terms)
    )
    if target_kind == "class" and FUNCTION_INTENT.search(query):
        score -= 3.0
    if line_weight:
        line_terms = identifier_variants(set(tokenize(source_line)))
        score += line_weight * len((query_terms - LINE_STOPWORDS) & line_terms)
    return score


def candidate_call_edges(
    retriever: BM25Retriever,
    call_index: PythonCallPathIndex,
    root: Path,
    query: str,
    source_paths: list[Path],
    *,
    source_files: int = 3,
    line_weight: float = 1.0,
    source_symbol: str | None = None,
) -> list[dict]:
    """Return ranked resolvable import-bound edges from retrieved source files.

    Ranking is only a navigation hint. A candidate is not evidence that it
    answers the question; callers must read both source locations.
    """
    if not 1 <= source_files <= 10 or not isfinite(line_weight) or line_weight < 0:
        raise ValueError("source_files must be 1..10 and line_weight nonnegative")
    root = root.resolve()
    tokenizer = retriever.tokenizer
    query_terms = identifier_variants(set(tokenizer.tokenize_query(query)))
    selected: list[Path] = []
    for path in source_paths:
        path = path.resolve()
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/"):
            if source_symbol is not None:
                raise ValueError("anchored source must be a non-test Python file")
            continue
        if path in selected:
            continue
        selected.append(path)
        if len(selected) >= source_files:
            break
    if source_symbol is not None and len(selected) != 1:
        raise ValueError("anchored source requires exactly one selected file")

    candidates: dict[tuple[str, str, str, int], dict] = {}
    for source_rank, path in enumerate(selected, 1):
        file_index = retriever.repo_index.files.get(normalize_path(path))
        if file_index is None:
            if source_symbol is not None:
                raise ValueError("source path is not an indexed Python file")
            continue
        source_lines = rebuild_source(
            [call_index.chunks[index] for index in call_index.chunk_indexes[normalize_path(path)]]
        ).splitlines()
        module_calls = call_index._module_calls(file_index.module)
        if source_symbol is not None and (
            source_symbol not in module_calls.functions
            or source_symbol in module_calls.ambiguous_functions
        ):
            raise ValueError("source symbol is missing or ambiguous in the selected file")
        for symbol in module_calls.functions:
            if source_symbol is not None and symbol != source_symbol:
                continue
            if symbol in module_calls.ambiguous_functions:
                continue
            try:
                traced = call_index.trace(path, symbol, max_local_depth=0, max_edges=100)
            except ValueError:
                continue
            for edge in traced["edges"]:
                if edge["kind"] != "import_bound_call" or edge["source_symbol"] != symbol:
                    continue
                target = Path(edge["target_path"]).resolve()
                if target == path:
                    continue
                target_relative = target.relative_to(root).as_posix()
                if target_relative.startswith("tests/"):
                    continue
                target_index = retriever.repo_index.files.get(normalize_path(target))
                target_start = edge["target_start_line"]
                if target_index is None or target_start is None:
                    continue
                target_record = next(
                    (record for record in target_index.symbols
                     if record.start_line == target_start and record.name == edge["target_symbol"]),
                    None,
                )
                if target_record is None:
                    continue
                relevance = score_call_edge(
                    query, query_terms, edge["call"], str(edge["target_symbol"]),
                    symbol, target, target_record.kind,
                    source_lines[edge["source_line"] - 1], retriever, line_weight,
                )
                record = {
                    "target_path": target_relative,
                    "source_path": path.relative_to(root).as_posix(),
                    "source_rank": source_rank,
                    "source_symbol": symbol,
                    "source_line": edge["source_line"],
                    "call": edge["call"],
                    "target_symbol": edge["target_symbol"],
                    "target_start_line": target_start,
                    "target_kind": target_record.kind,
                    "score": relevance / source_rank,
                }
                key = (record["source_path"], target_relative, str(edge["target_symbol"]), edge["source_line"])
                previous = candidates.get(key)
                if previous is None or record["score"] > previous["score"]:
                    candidates[key] = record
    return sorted(
        candidates.values(),
        key=lambda row: (-row["score"], row["target_path"], row["source_line"]),
    )
