"""Retrieval evaluation, trace export, bad-case analysis, and tuning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.tools.bm25 import BM25Retriever, SearchResult
from coding_rag.repository.chunks import split_python_files
from coding_rag.tools.recall import expand_with_neighbor_chunks
from coding_rag.tools.env import ensure_dotenv, load_dotenv
from coding_rag.repository.files import load_python_files
from coding_rag.rag.llm_client import available_provider_names
from coding_rag.tools.filter import filter_recalled_results
from coding_rag.rag.answer_generator import AnswerGenerator, build_generator
from coding_rag.rag.prompt import GenerationMode
from coding_rag.rag.trace import build_retrieval_trace, get_retrieval_stages


@dataclass
class EvalCase:
    """单个检索评测用例。

    各字段：
        id: 用例唯一标识。
        query: 检索查询字符串。
        relevant: 相关文件路径列表。
    """
    id: str
    query: str
    relevant: list[str]


STAGE_DIAGNOSIS_ORDER = [
    "hit_final_top_k",
    "missing_in_initial_search",
    "lost_during_recall_review",
    "recall_review_pushed_below_top_k",
    "filtered_after_recall_review",
]

OPTIMIZATION_SEARCH_SPACE = [
    (30, 5, 1),
    (40, 5, 1),
    (60, 10, 1),
    (60, 10, 2),
]


def load_evalset(path: str) -> list[EvalCase]:
    """从文件路径加载评测用例集。

    按扩展名判断格式——.jsonl 按行解析 JSON，否则按整个 JSON 解析；
    加载后逐项调用 normalize_eval_case 规范化。
    """
    eval_path = Path(path)
    text = eval_path.read_text(encoding="utf-8")
    if eval_path.suffix.casefold() == ".jsonl":
        data = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(text)

    return [normalize_eval_case(item, index, eval_path) for index, item in enumerate(data, start=1)]


def normalize_eval_case(item: dict, index: int, eval_path: Path) -> EvalCase:
    """将原始字典规范化为 EvalCase。

    兼容 query/question 与 relevant/gold_files 别名；
    校验 query 和 relevant 必填且 relevant 为列表类型；
    缺失或类型错误时抛出 ValueError。
    """
    case_id = str(item.get("id") or f"case-{index}")
    query = item.get("query") or item.get("question")
    relevant = item.get("relevant") or item.get("gold_files")

    if not query:
        raise ValueError(f"{eval_path}: case {case_id} is missing 'query' or 'question'")
    if not relevant:
        raise ValueError(f"{eval_path}: case {case_id} is missing 'relevant' or 'gold_files'")
    if not isinstance(relevant, list):
        raise ValueError(f"{eval_path}: case {case_id} relevant/gold_files must be a list")

    return EvalCase(id=case_id, query=str(query), relevant=[str(path) for path in relevant])


def build_eval_generator(args: argparse.Namespace) -> AnswerGenerator:
    """按命令行参数构建回答生成器。"""
    return build_generator(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        mode=GenerationMode(args.mode),
        max_context_chars=args.llm_context_chars,
    )


def generate_eval_answer(
    generator: AnswerGenerator | None,
    setup_error: str | None,
    query: str,
    final_results: list[SearchResult],
) -> dict[str, str | None]:
    """返回 answer/error 字典，优先返回配置错误，无生成器时返回空状态，并把生成时 RuntimeError 转为错误文本。"""
    if setup_error:
        return {"answer": "", "error": setup_error}
    if generator is None:
        return {"answer": None, "error": None}

    try:
        return {"answer": generator.generate(query, final_results).strip(), "error": None}
    except RuntimeError as error:
        return {"answer": "", "error": str(error)}


def pipeline(repo_path: str, query: str, top_k: int, chunk_size: int, overlap: int, recall_window: int) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult]]:
    """检索流水线，依次执行初搜、邻居召回与最终过滤三个阶段的检索。

    加载仓库 Python 文件并切块，BM25 初搜得到种子；
    按召回窗口扩展邻居块；重排序过滤后截取 top_k。
    返回 (初搜结果, 邻居召回结果, 最终过滤结果) 三元组。
    """
    python_files = load_python_files(repo_path)
    chunks = split_python_files(python_files, chunk_size=chunk_size, overlap=overlap)
    retriever = BM25Retriever(chunks)
    seed = retriever.search(query, top_k=top_k)
    recalled = expand_with_neighbor_chunks(
        chunks,
        seed,
        window=recall_window,
        max_results=default_recall_max_results(top_k, recall_window),
    )
    final = filter_recalled_results(query, recalled, retriever, final_k=top_k)
    return seed, recalled, final


def default_recall_max_results(top_k: int, recall_window: int) -> int:
    """邻居召回的默认上限条数。

    公式：``max(20, top_k * ((2 * max(recall_window, 0)) + 1))``。
    """
    per_seed = (2 * max(recall_window, 0)) + 1
    return max(20, top_k * per_seed)


def mrr_and_recall_at_k(
    results: list[SearchResult],
    relevant: list[str],
    k: int,
    path_roots: list[Path] | None = None,
) -> tuple[float, float, int]:
    """在前 k 项中按等价路径去重 gold 项，计算 MRR 与召回率。

    遍历 top-k 结果，按等价路径判断是否命中相关项；
    同一 gold 项仅首次命中计入。返回 (首个命中倒数排名, 召回率, 命中数)。
    """
    top = results[:k]
    hit_ranks: list[int] = []
    matched_relevant: set[int] = set()
    for rank, result in enumerate(top, start=1):
        relevant_index = matched_relevant_index(result, relevant, path_roots)
        if relevant_index is None:
            continue

        hit_ranks.append(rank)
        matched_relevant.add(relevant_index)

    rr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    recall = (len(matched_relevant) / len(relevant)) if relevant else 0.0
    return rr, recall, len(matched_relevant)


def analyze_bad_case(
    seed: list[SearchResult],
    recalled: list[SearchResult],
    final: list[SearchResult],
    relevant: list[str],
    k: int,
    path_roots: list[Path] | None = None,
) -> dict:
    """比较初搜、召回和最终前 k 命中，生成启发式原因与提示。

    依次检查 seed 前 k、recall 前 k、recall 全量以及 final 前 k 是否命中相关结果，
    推断失败原因（query 不匹配、召回窗口不足、后过滤丢弃或重排序/截断导致），
    返回命中标记、原因列表与提示建议。
    """
    seed_hit = any(is_relevant_result(result, relevant, path_roots) for result in seed[:k])
    recall_top_k_hit = any(is_relevant_result(result, relevant, path_roots) for result in recalled[:k])
    recall_hit = any(is_relevant_result(result, relevant, path_roots) for result in recalled)
    final_hit = any(is_relevant_result(result, relevant, path_roots) for result in final[:k])

    reasons: list[str] = []
    if not seed_hit:
        reasons.append("query_token_mismatch_or_chunking")
    elif not recall_hit:
        reasons.append("recall_window_insufficient")
    elif seed_hit and not final_hit:
        reasons.append("post_filter_dropped_relevant_chunk")
    if recall_hit and not final_hit:
        reasons.append("rerank_or_topk_cutoff")

    prompt_hints = [
        "在 query 里补充函数名/类名/文件名关键词",
        "Prompt 明确要求引用 path:start-end，并先给相关块列表再作答",
    ]

    return {
        "seed_hit": seed_hit,
        "recall_hit": recall_hit,
        "recall_top_k_hit": recall_top_k_hit,
        "final_hit": final_hit,
        "reasons": reasons,
        "prompt_hints": prompt_hints,
    }


def build_stage_file_record(
    case: EvalCase,
    seed: list[SearchResult],
    recalled: list[SearchResult],
    final: list[SearchResult],
    k: int,
    path_roots: list[Path] | None = None,
) -> dict:
    """为各检索阶段构建文件命中快照并按阶段丢失位置生成诊断记录。

    分别对初搜、召回前 k、召回全量及最终前 k 调用 build_stage_snapshot，
    汇总各阶段命中信息并由 diagnose_stage_drop 生成阶段间丢失原因诊断。
    """
    initial_review = build_stage_snapshot(seed, case.relevant, k, path_roots)
    recall_review_top_k = build_stage_snapshot(recalled, case.relevant, k, path_roots)
    recall_review_all = build_stage_snapshot(recalled, case.relevant, len(recalled), path_roots)
    final_review = build_stage_snapshot(final, case.relevant, k, path_roots)

    return {
        "id": case.id,
        "query": case.query,
        "gold_files": case.relevant,
        "top_k": k,
        "initial_review": initial_review,
        "recall_review_top_k": recall_review_top_k,
        "recall_review_all": recall_review_all,
        "final": final_review,
        "diagnosis": diagnose_stage_drop(
            initial_hit=initial_review["hit"],
            recall_top_k_hit=recall_review_top_k["hit"],
            recall_all_hit=recall_review_all["hit"],
            final_hit=final_review["hit"],
        ),
    }


def build_stage_snapshot(
    results: list[SearchResult],
    relevant: list[str],
    k: int,
    path_roots: list[Path] | None = None,
) -> dict:
    """构建阶段快照，包含前 k 个唯一结果文件、命中的 gold 文件列表以及是否命中。"""
    matched_gold_files = matched_relevant_files(results, relevant, k, path_roots)
    return {
        "files": unique_result_files(results, k, path_roots),
        "matched_gold_files": matched_gold_files,
        "hit": bool(matched_gold_files),
    }


def matched_relevant_files(
    results: list[SearchResult],
    relevant: list[str],
    k: int,
    path_roots: list[Path] | None = None,
) -> list[str]:
    """从检索结果的前 k 个中匹配相关的 gold 文件，按检索结果中首次命中的顺序去重，并返回对应的 gold 路径。"""
    matched_indexes: list[int] = []
    for result in results[:k]:
        relevant_index = matched_relevant_index(result, relevant, path_roots)
        if relevant_index is None or relevant_index in matched_indexes:
            continue
        matched_indexes.append(relevant_index)

    return [relevant[index] for index in matched_indexes]


def unique_result_files(
    results: list[SearchResult],
    k: int,
    path_roots: list[Path] | None = None,
) -> list[str]:
    """提取检索结果前 k 个中按首次出现去重后的唯一文件路径。"""
    files: list[str] = []
    for result in results[:k]:
        path = readable_path(result.chunk.file_path, path_roots)
        if path not in files:
            files.append(path)
    return files


def readable_path(path: str | Path, path_roots: list[Path] | None = None) -> str:
    """将路径转为可读字符串：对绝对路径优先尝试相对于给定 roots 或当前目录展示，失败则保留规范化绝对路径，统一使用正斜杠。"""
    candidate = Path(path)
    roots = [*(path_roots or []), Path.cwd()]

    if candidate.is_absolute():
        resolved = candidate.resolve()
        for root in roots:
            try:
                return str(resolved.relative_to(root.resolve())).replace("\\", "/")
            except ValueError:
                continue
        return str(resolved).replace("\\", "/")

    return str(candidate).replace("\\", "/")


def diagnose_stage_drop(
    initial_hit: bool,
    recall_top_k_hit: bool,
    recall_all_hit: bool,
    final_hit: bool,
) -> str:
    """按优先级返回阶段诊断结果。

    按以下优先顺序判断：最终命中 → 初搜缺失 → 召回全量丢失 → 被挤出召回前k → 最终过滤丢失。
    """
    if final_hit:
        return "hit_final_top_k"
    if not initial_hit:
        return "missing_in_initial_search"
    if not recall_all_hit:
        return "lost_during_recall_review"
    if not recall_top_k_hit:
        return "recall_review_pushed_below_top_k"
    return "filtered_after_recall_review"


def aggregate_stage_diagnostics(
    stage_records: list[dict],
    trace_rows: list[dict] | None = None,
    context_char_warning: int = 12000,
) -> dict:
    """聚合阶段诊断记录，统计诊断计数与用例、bad case，以及可选的上下文字符统计。

    统计各诊断类型的计数和对应用例ID，汇总非最终命中的bad case；若提供 trace 数据，计算上下文字符的均值和最大值，并列出超阈值用例。
    """
    counts = {diagnosis: 0 for diagnosis in STAGE_DIAGNOSIS_ORDER}
    cases_by_diagnosis = {diagnosis: [] for diagnosis in STAGE_DIAGNOSIS_ORDER}

    for record in stage_records:
        diagnosis = str(record.get("diagnosis") or "unknown")
        case_id = str(record.get("id") or "")
        if diagnosis not in counts:
            counts[diagnosis] = 0
            cases_by_diagnosis[diagnosis] = []
        counts[diagnosis] += 1
        if case_id:
            cases_by_diagnosis[diagnosis].append(case_id)

    bad_case_ids = [
        case_id
        for diagnosis, case_ids in cases_by_diagnosis.items()
        if diagnosis != "hit_final_top_k"
        for case_id in case_ids
    ]

    context_totals = context_char_totals(trace_rows or [])
    long_context_case_ids: list[str] = []
    if trace_rows:
        for row, total_chars in zip(trace_rows, context_totals):
            if total_chars > context_char_warning:
                long_context_case_ids.append(str(row.get("id") or ""))

    return {
        "total": len(stage_records),
        "diagnosis_counts": counts,
        "cases_by_diagnosis": cases_by_diagnosis,
        "bad_case_ids": bad_case_ids,
        "context": {
            "case_count": len(context_totals),
            "avg_chars": mean(context_totals) if context_totals else 0.0,
            "max_chars": max(context_totals) if context_totals else 0,
            "warning_threshold": context_char_warning,
            "over_threshold_case_ids": [case_id for case_id in long_context_case_ids if case_id],
        },
    }


def context_char_totals(trace_rows: list[dict]) -> list[int]:
    """计算每行上下文字符总数。

    优先使用 row.context；若为 None 则从 trajectory 提取 context_compaction。
    每块优先取 char_count，否则取 text/preview 的长度。
    """
    totals: list[int] = []
    for row in trace_rows:
        context_blocks = row.get("context")
        if context_blocks is None:
            context_blocks = get_retrieval_stages(row.get("trajectory", {})).get("context_compaction", [])

        total = 0
        for block in context_blocks:
            total += int(block.get("char_count") or len(block.get("text") or block.get("preview") or ""))
        totals.append(total)
    return totals


def render_stage_diagnostics_summary(summary: dict) -> str:
    """渲染阶段诊断摘要字符串。

    输出诊断计数、bad_case_ids 和可选的上下文统计信息（含平均值、最大值、阈值及超阈值 case）。
    """
    lines = ["Stage diagnostics:"]
    counts = summary.get("diagnosis_counts") or {}
    for diagnosis in ordered_diagnoses(counts):
        lines.append(f"- {diagnosis}: {counts[diagnosis]}")

    bad_case_ids = summary.get("bad_case_ids") or []
    if bad_case_ids:
        lines.append(f"- bad_case_ids: {', '.join(bad_case_ids)}")

    context = summary.get("context") or {}
    if context.get("case_count"):
        lines.append(
            "- context_chars: "
            f"avg={context.get('avg_chars', 0.0):.1f}, "
            f"max={context.get('max_chars', 0)}, "
            f"threshold={context.get('warning_threshold', 0)}"
        )
        over_threshold = context.get("over_threshold_case_ids") or []
        if over_threshold:
            lines.append(f"- context_over_threshold: {', '.join(over_threshold)}")

    return "\n".join(lines)


def ordered_diagnoses(counts: dict[str, int]) -> list[str]:
    """按固定顺序返回诊断名列表。

    先按 STAGE_DIAGNOSIS_ORDER 给出已知项，再按字典序追加未知项。
    """
    known = [diagnosis for diagnosis in STAGE_DIAGNOSIS_ORDER if diagnosis in counts]
    unknown = sorted(diagnosis for diagnosis in counts if diagnosis not in STAGE_DIAGNOSIS_ORDER)
    return known + unknown


def is_relevant_result(
    result: SearchResult,
    relevant: list[str],
    path_roots: list[Path] | None = None,
) -> bool:
    """判断检索结果是否相关，相关性由等价路径形式集合的交集判定."""
    return matched_relevant_index(result, relevant, path_roots) is not None


def matched_relevant_index(
    result: SearchResult,
    relevant: list[str],
    path_roots: list[Path] | None = None,
) -> int | None:
    """返回第一个与结果路径形式集合有交集的 gold 索引，无匹配则返回 None."""
    result_paths = comparable_path_forms(result.chunk.file_path, path_roots)
    for index, path in enumerate(relevant):
        if result_paths & comparable_path_forms(path, path_roots):
            return index
    return None


def comparable_path_forms(path: str | Path, path_roots: list[Path] | None = None) -> set[str]:
    """构建路径的等价形式集合，包含规范化原始路径、基于当前目录及给定 roots 解析的绝对形式和可相对化形式."""
    roots = [Path.cwd(), *(path_roots or [])]
    raw = normalize_path_text(path)
    forms = {raw}
    candidate = Path(path)

    if candidate.is_absolute():
        absolute_candidates = [candidate.resolve()]
    else:
        absolute_candidates = [(root / candidate).resolve() for root in roots]

    for absolute_path in absolute_candidates:
        forms.add(normalize_path_text(absolute_path))
        for root in roots:
            try:
                forms.add(normalize_path_text(absolute_path.relative_to(root.resolve())))
            except ValueError:
                continue

    return {form for form in forms if form}


def normalize_path_text(path: str | Path) -> str:
    """规范化路径文本：统一正斜杠、去除首尾空白并 casefold."""
    return str(path).replace("\\", "/").strip().casefold()


def stage_files_path_for_trace(trace_path: str | Path) -> Path:
    """在 trace 文件同目录下，基于 stem 生成 ``{stem}_stage_files`` 路径。

    保留原文件后缀，若缺失则默认使用 ``.jsonl``。
    """
    path = Path(trace_path)
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_stage_files{suffix}")


def stage_summary_path_for_trace(trace_path: str | Path) -> Path:
    """在 trace 文件同目录下，基于 stem 生成 ``{stem}_stage_summary.json`` 路径。"""
    path = Path(trace_path)
    return path.with_name(f"{path.stem}_stage_summary.json")


def write_json(path: str | Path, data: dict) -> None:
    """创建父目录并以 UTF-8 写入格式化 JSON。

    使用 ``ensure_ascii=False``，缩进 2 空格。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """创建父目录并以 UTF-8 逐行写入 JSON。

    每行一个 JSON 对象，使用 ``ensure_ascii=False``。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def run_eval(args: argparse.Namespace) -> int:
    """可选初始化 LLM（配置错误记入逐例结果而非立即终止）、运行所有检索评测、写 trace/阶段文件/阶段汇总、打印聚合指标与可选 LLM 输出，并可继续参数搜索；成功返回 0。"""
    generator: AnswerGenerator | None = None
    llm_setup_error: str | None = None
    if args.llm:
        ensure_dotenv(args.env_file)
        load_dotenv(args.env_file)
        try:
            generator = build_eval_generator(args)
        except ValueError as error:
            llm_setup_error = f"LLM 配置错误: {error}"

    cases = load_evalset(args.evalset)
    path_roots = [Path(args.repo_path).resolve(), Path(args.evalset).resolve().parent]
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    stage_files_out = getattr(args, "stage_files_out", None)
    stage_files_path = Path(stage_files_out) if stage_files_out else stage_files_path_for_trace(trace_path)
    stage_summary_path = stage_summary_path_for_trace(trace_path)

    per_case: list[dict] = []
    stage_file_records: list[dict] = []
    for case in cases:
        seed, recalled, final = pipeline(
            repo_path=args.repo_path,
            query=case.query,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recall_window=args.recall_window,
        )
        rr, recall_at_k, hit_count = mrr_and_recall_at_k(final, case.relevant, args.top_k, path_roots)
        bad = analyze_bad_case(seed, recalled, final, case.relevant, args.top_k, path_roots)
        trajectory = build_retrieval_trace(
            query=case.query,
            seed_results=seed,
            recalled_results=recalled,
            final_results=final,
            params={
                "repo_path": args.repo_path,
                "top_k": args.top_k,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "recall_window": args.recall_window,
            },
        )
        stages = get_retrieval_stages(trajectory)
        record = {
            "id": case.id,
            "query": case.query,
            "metrics": {"rr": rr, "recall_at_k": recall_at_k, "hit_count": hit_count},
            "bad_case": bad,
            "trajectory": trajectory,
            "seed": stages["initial_search"],
            "recalled": stages["neighbor_recall"],
            "final": stages["final_filter"],
            "context": stages["context_compaction"],
        }
        if args.llm:
            record["llm"] = generate_eval_answer(generator, llm_setup_error, case.query, final)
        per_case.append(record)
        stage_file_records.append(build_stage_file_record(case, seed, recalled, final, args.top_k, path_roots))

    write_jsonl(trace_path, per_case)
    write_jsonl(stage_files_path, stage_file_records)
    stage_summary = aggregate_stage_diagnostics(stage_file_records, per_case)
    write_json(stage_summary_path, stage_summary)

    mrr = mean(item["metrics"]["rr"] for item in per_case) if per_case else 0.0
    avg_recall = mean(item["metrics"]["recall_at_k"] for item in per_case) if per_case else 0.0
    bad_cases = [x for x in per_case if not x["bad_case"]["final_hit"]]

    print(f"Cases: {len(per_case)}")
    print(f"MRR@{args.top_k}: {mrr:.4f}")
    print(f"Recall@{args.top_k}: {avg_recall:.4f}")
    print(f"Bad cases: {len(bad_cases)}")
    print(f"Trace saved: {trace_path}")
    print(f"Stage files saved: {stage_files_path}")
    print(f"Stage summary saved: {stage_summary_path}")
    print(render_stage_diagnostics_summary(stage_summary))
    if args.llm:
        print_llm_outputs(per_case)

    if args.optimize:
        optimize(args, cases)

    return 0


def optimize(args: argparse.Namespace, cases: list[EvalCase]) -> str:
    """收集网格搜索结果，打印并返回报告。"""
    rows, best = collect_optimization_results(args, cases)
    report = render_optimization_report(rows, best, args.top_k)
    print(f"\n{report}")
    return report


def collect_optimization_results(args: argparse.Namespace, cases: list[EvalCase]) -> tuple[list[dict], dict | None]:
    """在固定 OPTIMIZATION_SEARCH_SPACE 中按各用例 MRR 求均值，严格更高时更新最佳项。"""
    path_roots = [Path(args.repo_path).resolve()]
    rows: list[dict] = []
    best: dict | None = None
    for chunk_size, overlap, recall_window in OPTIMIZATION_SEARCH_SPACE:
        scores: list[float] = []
        for case in cases:
            _, _, final = pipeline(args.repo_path, case.query, args.top_k, chunk_size, overlap, recall_window)
            rr, _, _ = mrr_and_recall_at_k(final, case.relevant, args.top_k, path_roots)
            scores.append(rr)
        mrr = mean(scores) if scores else 0.0
        row = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "recall_window": recall_window,
            "mrr": mrr,
        }
        rows.append(row)
        if best is None or mrr > best["mrr"]:
            best = row

    return rows, best


def render_optimization_report(rows: list[dict], best: dict | None, top_k: int) -> str:
    """格式化各组合及可选推荐，无行时说明无用例。"""
    lines = ["=== 参数搜索（切片 / 检索）==="]
    if not rows:
        lines.append("No eval cases.")
        return "\n".join(lines)

    for row in rows:
        lines.append(
            f"chunk={row['chunk_size']}, overlap={row['overlap']}, "
            f"recall_window={row['recall_window']} -> MRR@{top_k}={row['mrr']:.4f}"
        )

    if best:
        lines.append(
            "推荐参数: "
            f"--chunk-size {best['chunk_size']} "
            f"--overlap {best['overlap']} "
            f"--recall-window {best['recall_window']} "
            f"(MRR@{top_k}={best['mrr']:.4f})"
        )
        lines.append("Prompt suggestion: list relevant chunks first, then answer; refuse when evidence is insufficient.")

    return "\n".join(lines)


def print_llm_outputs(rows: list[dict]) -> None:
    """按用例打印 LLM 答案、错误或空输出提示。"""
    print("\n=== LLM 输出 ===")
    for row in rows:
        llm = row.get("llm") or {}
        answer = llm.get("answer")
        error = llm.get("error")
        print(f"\n[{row['id']}] {row['query']}")
        if error:
            print(f"LLM 错误: {error}")
        elif answer:
            print(answer)
        else:
            print("LLM output is empty.")


def parse_args() -> argparse.Namespace:
    """解析检索、追踪、优化、环境和可选 LLM 生成参数。"""
    parser = argparse.ArgumentParser(description="检索评测与优化脚本")
    parser.add_argument("repo_path")
    parser.add_argument("evalset", help="评测集 JSON 路径，格式 [{id,query,relevant:[file_path,...]}]")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--recall-window", type=int, default=2)
    parser.add_argument("--trace-out", default="artifacts/retrieval_trace.jsonl")
    parser.add_argument("--stage-files-out", help="write per-case stage file diagnostics JSONL")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--llm", action="store_true", help="call the configured LLM for each eval case")
    parser.add_argument("--mode", choices=[mode.value for mode in GenerationMode], default="judge")
    parser.add_argument("--llm-provider", choices=available_provider_names() + ["custom"], default="deepseek")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-context-chars", type=int, default=12000)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_eval(parse_args()))
