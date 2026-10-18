"""Reusable retrieval benchmark core with one index build per experiment."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Protocol, Sequence

from coding_rag.rag.prompt import compact_results_for_context
from coding_rag.repository.chunks import CodeChunk, split_python_files
from coding_rag.repository.files import PythonFile, load_python_files
from coding_rag.tools.bm25 import DEFAULT_GRAPH_WEIGHT, DEFAULT_PATH_WEIGHT, BM25Retriever, SearchResult
from coding_rag.tools.filter import filter_recalled_results
from coding_rag.tools.recall import expand_with_neighbor_chunks


DEFAULT_EXCLUDE_ROOTS = ("data", "datasets", "artifacts")
RETRIEVER_NAME = "hybrid-bm25-path-v7"


class StructuralRanker(Protocol):
    """Extension point for a future graph-ranked A/B implementation."""

    name: str

    def score(self, query: str, chunks: Sequence[CodeChunk]) -> Sequence[float]:
        """Return one structural score per chunk for an experimental ranker."""
        ...


@dataclass(frozen=True)
class EvalCase:
    """Normalized retrieval evaluation case."""
    id: str
    query: str
    relevant: list[str]
    task_type: str = "unspecified"
    difficulty: str = "unspecified"


@dataclass(frozen=True)
class EvalsetValidation:
    """All structural and repository-path validation findings for an evalset."""
    case_count: int
    duplicate_ids: tuple[str, ...]
    empty_queries: tuple[str, ...]
    empty_gold: tuple[str, ...]
    invalid_paths: tuple[str, ...]
    missing_gold_files: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Return whether the evalset is non-empty and has no findings."""
        return not any((self.duplicate_ids, self.empty_queries, self.empty_gold, self.invalid_paths, self.missing_gold_files)) and self.case_count > 0

    def to_dict(self) -> dict:
        """Serialize findings together with the computed validity flag."""
        return {**self.__dict__, "valid": self.valid}


@dataclass(frozen=True)
class RetrievalEvalConfig:
    """Stable parameters that define one retrieval experiment."""
    repo_path: Path
    top_k: int = 5
    chunk_size: int = 40
    overlap: int = 5
    recall_window: int = 2
    max_recall_results: int | None = None
    min_final_score: float = 0.0
    graph_weight: float = DEFAULT_GRAPH_WEIGHT
    path_weight: float = DEFAULT_PATH_WEIGHT
    exclude_roots: tuple[str, ...] = DEFAULT_EXCLUDE_ROOTS

    def public_dict(self) -> dict:
        """Return portable config metadata without an absolute repository path."""
        return {
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "recall_window": self.recall_window,
            "max_recall_results": self.max_recall_results,
            "min_final_score": self.min_final_score,
            "graph_weight": self.graph_weight,
            "path_weight": self.path_weight,
            "exclude_roots": list(self.exclude_roots),
        }


@dataclass
class PreparedRetrievalPipeline:
    """Repository files, chunks, and index shared by every benchmark query."""
    config: RetrievalEvalConfig
    files: list[PythonFile]
    chunks: list[CodeChunk]
    retriever: BM25Retriever
    index_build_ms: float

    @classmethod
    def build(cls, config: RetrievalEvalConfig) -> "PreparedRetrievalPipeline":
        """Scan, chunk, and index the configured repository exactly once."""
        started = time.perf_counter()
        files = load_python_files(config.repo_path, exclude_roots=config.exclude_roots)
        chunks = split_python_files(files, chunk_size=config.chunk_size, overlap=config.overlap)
        if not chunks:
            raise ValueError("repository contains no indexable Python chunks")
        retriever = BM25Retriever(
            chunks,
            graph_weight=config.graph_weight,
            path_weight=config.path_weight,
            path_root=config.repo_path,
        )
        return cls(config, files, chunks, retriever, (time.perf_counter() - started) * 1000)

    def query(self, query: str) -> dict:
        """Run search, neighbor recall, filtering, and context compaction."""
        started = time.perf_counter()
        seed, scores = self.retriever.search_with_scores(query, top_k=self.config.top_k)
        max_results = self.config.max_recall_results
        if max_results is None:
            max_results = max(20, self.config.top_k * ((2 * max(self.config.recall_window, 0)) + 1))
        recalled = expand_with_neighbor_chunks(
            self.chunks, seed, window=self.config.recall_window, max_results=max_results
        )
        final = filter_recalled_results(
            query,
            recalled,
            self.retriever,
            final_k=self.config.top_k,
            min_score=self.config.min_final_score,
            scores=scores,
        )
        context = compact_results_for_context(final)
        return {
            "seed": seed,
            "recalled": recalled,
            "final": final,
            "context": context,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


def _raw_eval_items(path: str | Path) -> list[dict]:
    """Load raw cases from a JSON array or JSONL file."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"{target}: evaluation set must be a JSON array or JSONL")
    return payload


def load_evalset(path: str | Path) -> list[EvalCase]:
    """Load JSON/JSONL aliases into normalized evaluation cases."""
    cases: list[EvalCase] = []
    for index, item in enumerate(_raw_eval_items(path), 1):
        case_id = str(item.get("id") or f"case-{index}")
        query = item.get("query") or item.get("question") or ""
        gold = item.get("relevant") if "relevant" in item else item.get("gold_files")
        cases.append(EvalCase(
            id=case_id,
            query=str(query),
            relevant=[str(value) for value in gold] if isinstance(gold, list) else [],
            task_type=str(item.get("task_type") or "unspecified"),
            difficulty=str(item.get("difficulty") or "unspecified"),
        ))
    return cases


def validate_evalset(repo_path: str | Path, evalset: str | Path) -> EvalsetValidation:
    """Validate identities, questions, gold lists, and safe existing gold paths."""
    root = Path(repo_path).resolve()
    items = _raw_eval_items(evalset)
    seen: set[str] = set()
    duplicates: list[str] = []
    empty_queries: list[str] = []
    empty_gold: list[str] = []
    invalid: list[str] = []
    missing: list[str] = []
    for index, item in enumerate(items, 1):
        case_id = str(item.get("id") or f"case-{index}")
        if case_id in seen:
            duplicates.append(case_id)
        seen.add(case_id)
        query = item.get("query") or item.get("question")
        if not isinstance(query, str) or not query.strip():
            empty_queries.append(case_id)
        gold = item.get("relevant") if "relevant" in item else item.get("gold_files")
        if not isinstance(gold, list) or not gold:
            empty_gold.append(case_id)
            continue
        for raw_path in gold:
            value = str(raw_path).replace("\\", "/")
            pure = PurePosixPath(value)
            label = f"{case_id}:{value}"
            if pure.is_absolute() or ".." in pure.parts or Path(value).is_absolute():
                invalid.append(label)
            elif not (root / Path(*pure.parts)).is_file():
                missing.append(label)
    return EvalsetValidation(len(items), tuple(duplicates), tuple(empty_queries), tuple(empty_gold), tuple(invalid), tuple(missing))


def _normalized(path: object, root: Path) -> str:
    """Normalize a result or gold path into a portable comparison form."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root.resolve())
        except ValueError:
            pass
    return str(candidate).replace("\\", "/").casefold()


def _ranked_gold(results: Sequence[SearchResult], relevant: Sequence[str], root: Path, k: int) -> list[int]:
    """Return binary relevance by rank while counting each gold file once."""
    gold = {_normalized(path, root) for path in relevant}
    seen: set[str] = set()
    ranked: list[int] = []
    for result in results[:k]:
        path = _normalized(result.chunk.file_path, root)
        hit = path in gold and path not in seen
        ranked.append(1 if hit else 0)
        if hit:
            seen.add(path)
    return ranked


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile for deterministic summaries."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def evaluate_retrieval(pipeline: PreparedRetrievalPipeline, cases: Sequence[EvalCase]) -> dict:
    """Evaluate all cases and aggregate quality, cost, and bad-case metrics."""
    root = pipeline.config.repo_path.resolve()
    rows: list[dict] = []
    latencies: list[float] = []
    context_chars: list[float] = []
    for case in cases:
        result = pipeline.query(case.query)
        ranked = _ranked_gold(result["final"], case.relevant, root, pipeline.config.top_k)
        hit_ranks = [index for index, hit in enumerate(ranked, 1) if hit]
        gold_paths = {_normalized(gold, root) for gold in case.relevant}
        result_paths = {_normalized(item.chunk.file_path, root) for item in result["final"][:pipeline.config.top_k]}
        unique_hits = len(gold_paths & result_paths)
        rr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        recall = unique_hits / len(gold_paths) if gold_paths else 0.0
        all_gold = float(bool(gold_paths) and gold_paths <= result_paths)
        dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(ranked, 1))
        ideal_count = min(len(set(case.relevant)), pipeline.config.top_k)
        idcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_count + 1))
        chars = sum(len(block.text) for block in result["context"])
        latencies.append(result["latency_ms"])
        context_chars.append(float(chars))
        rows.append({
            "id": case.id,
            "task_type": case.task_type,
            "difficulty": case.difficulty,
            "hit_at_1": float(bool(ranked[:1] and ranked[0])),
            "hit_at_k": float(bool(hit_ranks)),
            "rr": rr,
            "recall_at_k": recall,
            "all_gold_at_k": all_gold,
            "ndcg_at_k": dcg / idcg if idcg else 0.0,
            "latency_ms": result["latency_ms"],
            "context_chars": chars,
            "result_files": [_normalized(item.chunk.file_path, root) for item in result["final"]],
        })
    metric = lambda key: mean(row[key] for row in rows) if rows else 0.0
    return {
        "metrics": {
            "case_count": len(rows),
            "hit_at_1": metric("hit_at_1"),
            f"hit_at_{pipeline.config.top_k}": metric("hit_at_k"),
            f"mrr_at_{pipeline.config.top_k}": metric("rr"),
            f"recall_at_{pipeline.config.top_k}": metric("recall_at_k"),
            f"all_gold_at_{pipeline.config.top_k}": metric("all_gold_at_k"),
            f"ndcg_at_{pipeline.config.top_k}": metric("ndcg_at_k"),
            "bad_case_count": sum(1 for row in rows if not row["hit_at_k"]),
            "bad_case_ids": [row["id"] for row in rows if not row["hit_at_k"]],
            "incomplete_case_count": sum(1 for row in rows if not row["all_gold_at_k"]),
            "incomplete_case_ids": [row["id"] for row in rows if not row["all_gold_at_k"]],
        },
        "efficiency": {
            "file_count": len(pipeline.files),
            "chunk_count": len(pipeline.chunks),
            "index_build_ms": pipeline.index_build_ms,
            "query_latency_ms_p50": _percentile(latencies, 0.50),
            "query_latency_ms_p95": _percentile(latencies, 0.95),
            "context_chars_avg": mean(context_chars) if context_chars else 0.0,
            "context_chars_p95": _percentile(context_chars, 0.95),
        },
        "cases": rows,
    }


def dataset_sha256(path: str | Path) -> str:
    """Return the byte-level SHA-256 identity of an evaluation set."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_fingerprint(files: Sequence[PythonFile], root: Path) -> str:
    """Hash indexed relative paths and contents into a portable repository identity."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: _normalized(value.path, root)):
        digest.update(_normalized(item.path, root).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.text.encode()).digest())
    return digest.hexdigest()
