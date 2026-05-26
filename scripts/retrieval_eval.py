"""检索评测、Trace 保存、Bad Case 归因、参数优化脚本。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from coding_rag.bm25_retriever import BM25Retriever, SearchResult
from coding_rag.code_splitter import split_python_files
from coding_rag.context_recaller import expand_with_neighbor_chunks
from coding_rag.file_loader import load_python_files
from coding_rag.result_filter import filter_recalled_results


@dataclass
class EvalCase:
    id: str
    query: str
    relevant: list[str]


def load_evalset(path: str) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(id=item["id"], query=item["query"], relevant=item["relevant"]) for item in data]


def pipeline(repo_path: str, query: str, top_k: int, chunk_size: int, overlap: int, recall_window: int) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult]]:
    python_files = load_python_files(repo_path)
    chunks = split_python_files(python_files, chunk_size=chunk_size, overlap=overlap)
    retriever = BM25Retriever(chunks)
    seed = retriever.search(query, top_k=top_k)
    recalled = expand_with_neighbor_chunks(chunks, seed, window=recall_window, max_results=max(20, top_k * 4))
    final = filter_recalled_results(query, recalled, retriever, final_k=top_k)
    return seed, recalled, final


def mrr_and_recall_at_k(results: list[SearchResult], relevant: list[str], k: int) -> tuple[float, float, int]:
    rel = set(relevant)
    top = results[:k]
    hits = [i for i, r in enumerate(top, start=1) if str(r.chunk.file_path) in rel]
    rr = 1.0 / hits[0] if hits else 0.0
    recall = (len(hits) / len(rel)) if rel else 0.0
    return rr, recall, len(hits)


def analyze_bad_case(seed: list[SearchResult], recalled: list[SearchResult], final: list[SearchResult], relevant: list[str], k: int) -> dict:
    rel = set(relevant)
    seed_hit = any(str(r.chunk.file_path) in rel for r in seed[:k])
    recall_hit = any(str(r.chunk.file_path) in rel for r in recalled[:k])
    final_hit = any(str(r.chunk.file_path) in rel for r in final[:k])

    reasons: list[str] = []
    if not seed_hit:
        reasons.append("query_token_mismatch_or_chunking")
    elif seed_hit and not final_hit:
        reasons.append("post_filter_dropped_relevant_chunk")
    if recall_hit and not final_hit:
        reasons.append("rerank_or_topk_cutoff")
    if not recall_hit:
        reasons.append("recall_window_insufficient")

    prompt_hints = [
        "在 query 里补充函数名/类名/文件名关键词",
        "Prompt 明确要求引用 path:start-end，并先给相关块列表再作答",
    ]

    return {
        "seed_hit": seed_hit,
        "recall_hit": recall_hit,
        "final_hit": final_hit,
        "reasons": reasons,
        "prompt_hints": prompt_hints,
    }


def run_eval(args: argparse.Namespace) -> int:
    cases = load_evalset(args.evalset)
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    per_case: list[dict] = []
    for case in cases:
        seed, recalled, final = pipeline(
            repo_path=args.repo_path,
            query=case.query,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recall_window=args.recall_window,
        )
        rr, recall_at_k, hit_count = mrr_and_recall_at_k(final, case.relevant, args.top_k)
        bad = analyze_bad_case(seed, recalled, final, case.relevant, args.top_k)
        record = {
            "id": case.id,
            "query": case.query,
            "metrics": {"rr": rr, "recall_at_k": recall_at_k, "hit_count": hit_count},
            "bad_case": bad,
            "seed": [serialize_result(x) for x in seed],
            "recalled": [serialize_result(x) for x in recalled],
            "final": [serialize_result(x) for x in final],
        }
        per_case.append(record)

    trace_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in per_case), encoding="utf-8")

    mrr = mean(item["metrics"]["rr"] for item in per_case) if per_case else 0.0
    avg_recall = mean(item["metrics"]["recall_at_k"] for item in per_case) if per_case else 0.0
    bad_cases = [x for x in per_case if not x["bad_case"]["final_hit"]]

    print(f"Cases: {len(per_case)}")
    print(f"MRR@{args.top_k}: {mrr:.4f}")
    print(f"Recall@{args.top_k}: {avg_recall:.4f}")
    print(f"Bad cases: {len(bad_cases)}")
    print(f"Trace saved: {trace_path}")

    if args.optimize:
        optimize(args, cases)

    return 0


def optimize(args: argparse.Namespace, cases: list[EvalCase]) -> None:
    print("\n=== 参数搜索（切片 / 检索）===")
    search_space = [
        (30, 5, 1),
        (40, 5, 1),
        (60, 10, 1),
        (60, 10, 2),
    ]
    best = None
    for chunk_size, overlap, recall_window in search_space:
        scores: list[float] = []
        for case in cases:
            _, _, final = pipeline(args.repo_path, case.query, args.top_k, chunk_size, overlap, recall_window)
            rr, _, _ = mrr_and_recall_at_k(final, case.relevant, args.top_k)
            scores.append(rr)
        mrr = mean(scores) if scores else 0.0
        print(f"chunk={chunk_size}, overlap={overlap}, recall_window={recall_window} -> MRR@{args.top_k}={mrr:.4f}")
        if best is None or mrr > best[0]:
            best = (mrr, chunk_size, overlap, recall_window)

    if best:
        print(
            "推荐参数: "
            f"chunk-size={best[1]}, overlap={best[2]}, recall-window={best[3]} (MRR@{args.top_k}={best[0]:.4f})"
        )
        print("Prompt 优化建议: 在系统提示里强制‘先相关块列表，后答案；证据不足时明确拒答’。")


def serialize_result(result: SearchResult) -> dict:
    return {
        "file": str(result.chunk.file_path),
        "start_line": result.chunk.start_line,
        "end_line": result.chunk.end_line,
        "score": result.score,
        "source": result.source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索评测与优化脚本")
    parser.add_argument("repo_path")
    parser.add_argument("evalset", help="评测集 JSON 路径，格式: [{id,query,relevant:[file_path,...]}]")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--recall-window", type=int, default=1)
    parser.add_argument("--trace-out", default="artifacts/retrieval_trace.jsonl")
    parser.add_argument("--optimize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_eval(parse_args()))
