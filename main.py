"""
RepoPilot 主程序
基于 RAG（检索增强生成）的代码搜索和理解工具
"""

import argparse
import sys

from coding_rag.agent import CodeAgentConfig, agent_run_to_dict, render_agent_run, run_code_agent
from coding_rag.bm25_retriever import BM25Retriever, SearchResult
from coding_rag.code_splitter import split_python_files
from coding_rag.context_recaller import expand_with_neighbor_chunks
from coding_rag.env_loader import ensure_dotenv, load_dotenv
from coding_rag.file_loader import load_python_files
from coding_rag.llm_client import (
    OpenAICompatibleChatClient,
    available_provider_names,
    build_llm_config,
)
from coding_rag.llm_judge import LLMJudge
from coding_rag.result_filter import filter_recalled_results

from rag.answer_generator import AnswerGenerator, build_generator
from rag.prompt import GenerationMode
from rag.trace import build_retrieval_trace, render_trace_report, write_trace_json


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    provider_choices = available_provider_names() + ["custom"]
    mode_choices = ["judge", "code-understand", "code-generate", "leetcode", "api"]

    parser = argparse.ArgumentParser(description="轻量级代码 RAG 搜索工具")
    parser.add_argument("repo_path", help="要搜索的代码仓库路径")
    parser.add_argument("query", help="搜索查询内容")
    parser.add_argument("--top-k", type=int, default=5, help="初始检索结果数量")
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="召回前的候选结果数量，默认为 --top-k",
    )
    parser.add_argument(
        "--recall-window",
        type=int,
        default=2,
        help="每个检索种子结果两侧召回的相邻代码块数量，设为 0 则禁用",
    )
    parser.add_argument(
        "--max-recall-results",
        type=int,
        help="召回扩展后的最大代码块数量，设为 0 则不限制",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        help="召回过滤后保留的代码块数量，默认为 --top-k，设为 0 则不限制",
    )
    parser.add_argument(
        "--min-final-score",
        type=float,
        help="丢弃最终检索分数低于此值的代码块",
    )
    parser.add_argument(
        "--no-final-filter",
        action="store_true",
        help="跳过召回后的重新评分/过滤阶段",
    )
    parser.add_argument("--chunk-size", type=int, default=40, help="每个代码块的行数")
    parser.add_argument("--overlap", type=int, default=5, help="代码块之间的重叠行数")
    parser.add_argument("--show-tokens", action="store_true", help="在搜索前打印查询的分词结果")
    parser.add_argument("--show-trace", action="store_true", help="打印 RAG 检索轨迹")
    parser.add_argument("--trace-out", help="保存本次 RAG 检索轨迹 JSON")
    parser.add_argument("--trace-include-text", action="store_true", help="Trace JSON 中包含完整代码文本")
    parser.add_argument("--agent", action="store_true", help="以代码 Agent 工作流运行任务")
    parser.add_argument("--agent-memory", default="artifacts/agent_memory.jsonl", help="Agent 记忆 JSONL 路径")
    parser.add_argument("--agent-memory-limit", type=int, default=5, help="本次任务最多读取的相关记忆数量")

    parser.add_argument("--llm", action="store_true", help="使用 LLM 基于检索结果生成答案")
    parser.add_argument(
        "--mode",
        choices=mode_choices,
        default="judge",
        help="生成模式：judge（判断相关性和回答）、code-understand（解释代码）、"
             "code-generate（生成代码）、leetcode（解决 LeetCode 问题）、api（生成 API 使用示例）",
    )
    parser.add_argument(
        "--llm-provider",
        choices=provider_choices,
        default="deepseek",
        help="LLM 服务提供商预设，使用 custom 时需配合 LLM_BASE_URL/LLM_MODEL 环境变量",
    )
    parser.add_argument("--llm-model", help="覆盖提供商预设的模型名称")
    parser.add_argument("--llm-base-url", help="覆盖提供商预设的基础 URL")
    parser.add_argument("--llm-api-key-env", help="存储 API 密钥的环境变量名")
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM 请求超时时间（秒）")
    parser.add_argument("--llm-max-tokens", type=int, default=2000, help="LLM 最大输出 token 数")
    parser.add_argument("--llm-temperature", type=float, help="可选的 LLM 温度参数")
    parser.add_argument(
        "--llm-context-chars",
        type=int,
        default=12000,
        help="发送给 LLM 的最大检索上下文字符数",
    )
    parser.add_argument(
        "--legacy-judge",
        action="store_true",
        help="使用旧版 LLM 判断模式（兼容旧版本行为）",
    )
    return parser.parse_args()


def main() -> int:
    """主函数：执行代码搜索和可选的 LLM 生成"""
    args = parse_args()
    if args.llm:
        ensure_dotenv()
        load_dotenv()

    if args.agent:
        return run_agent_cli(args)

    python_files = load_python_files(args.repo_path)
    chunks = split_python_files(
        python_files,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if not chunks:
        print("未找到 Python 代码块，请检查 repo_path 和文件内容")
        return 1

    retriever = BM25Retriever(chunks)
    if args.show_tokens:
        print("查询分词结果:")
        print(retriever.tokenizer.tokenize(args.query))
        print()

    # 如果未指定候选数量，使用 top-k 的值
    candidate_k = args.candidate_k or args.top_k
    seed_results = retriever.search(args.query, top_k=candidate_k)
    max_recall_results = args.max_recall_results
    if max_recall_results is None:
        max_recall_results = max(20, candidate_k * ((2 * max(args.recall_window, 0)) + 1))
    elif max_recall_results <= 0:
        max_recall_results = None
    recalled_results = expand_with_neighbor_chunks(
        chunks=chunks,
        seed_results=seed_results,
        window=args.recall_window,
        max_results=max_recall_results,
    )
    results = recalled_results  # 初始结果为召回的结果

    if not args.no_final_filter:
        final_k = args.final_k if args.final_k is not None else args.top_k
        results = filter_recalled_results(
            query=args.query,
            recalled_results=recalled_results,
            retriever=retriever,
            final_k=None if final_k == 0 else final_k,
            min_score=args.min_final_score,
        )

    trace = build_retrieval_trace(
        query=args.query,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=results,
        params=trace_params(args, candidate_k, max_recall_results),
        include_text=args.trace_include_text,
    )
    if args.trace_out:
        write_trace_json(args.trace_out, trace)
    if args.show_trace:
        print(render_trace_report(trace))
        print()

    if args.recall_window > 0:
        # 打印召回统计信息
        print(
            f"Hybrid 种子: {len(seed_results)} | "
            f"召回代码块: {len(recalled_results)} | "
            f"最终代码块: {len(results)}"
        )
        print()

    print_search_results(results)

    if args.llm:
        print()
        print("=" * 80)
        if args.legacy_judge:
            print("LLM 判断（旧版模式）")
        else:
            print(f"LLM 生成（模式: {args.mode}）")
        print("-" * 80)
        try:
            if args.legacy_judge:
                answer = run_llm_judge(args, results)
            else:
                answer = run_answer_generator(args, results)
        except ValueError as error:
            print(f"LLM 配置错误: {error}", file=sys.stderr)
            print_llm_setup_hint(args.llm_provider, file=sys.stderr)
            return 2
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 3

        print(answer.strip())

    return 0


def run_agent_cli(args: argparse.Namespace) -> int:
    """运行代码 Agent 工作流。"""
    client = None
    if args.llm:
        config = build_llm_config(
            provider=args.llm_provider,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature,
        )
        client = OpenAICompatibleChatClient(config)

    try:
        run = run_code_agent(
            task=args.query,
            config=CodeAgentConfig(
                repo_path=args.repo_path,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                recall_window=args.recall_window,
                max_recall_results=args.max_recall_results,
                final_k=args.final_k,
                min_final_score=args.min_final_score,
                memory_path=args.agent_memory,
                memory_limit=args.agent_memory_limit,
                max_context_chars=args.llm_context_chars,
            ),
            client=client,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 3

    if args.trace_out:
        write_trace_json(args.trace_out, agent_run_to_dict(run))

    print(render_agent_run(run, show_trace=args.show_trace))
    return 0


def run_llm_judge(args: argparse.Namespace, results: list[SearchResult]) -> str:
    """运行旧的LLM判断模式（兼容旧版本）"""
    config = build_llm_config(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
    )
    # 打印 LLM 配置信息
    print(f"提供商: {config.provider}")
    print(f"模型: {config.model}")
    print(f"API 密钥环境变量: {config.api_key_env}")
    print()

    # 创建 OpenAI 兼容的聊天客户端
    client = OpenAICompatibleChatClient(config)
    judge = LLMJudge(client=client, max_context_chars=args.llm_context_chars)
    # 让 LLM 判断检索结果的相关性并生成答案
    return judge.judge(args.query, results)


def run_answer_generator(args: argparse.Namespace, results: list[SearchResult]) -> str:
    """运行新的答案生成器（支持多种生成模式）"""
    mode = GenerationMode(args.mode)
    generator = build_generator(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        mode=mode,
        max_context_chars=args.llm_context_chars,
    )
    print(f"提供商: {generator.client.config.provider}")
    print(f"模型: {generator.client.config.model}")
    print(f"API 密钥环境变量: {generator.client.config.api_key_env}")
    print(f"生成模式: {mode.value}")
    print()

    # 根据指定模式生成答案
    return generator.generate(args.query, results)


def print_search_results(results: list[SearchResult]) -> None:
    """格式化并打印搜索结果"""
    for index, result in enumerate(results, start=1):
        chunk = result.chunk  # 获取代码块信息
        print("=" * 80)
        print(f"结果 {index} | 分数: {result.score:.4f}")
        print(f"来源标签: {result.source}")
        print(f"来源: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print("-" * 80)
        print(chunk.text.rstrip())


def trace_params(
    args: argparse.Namespace,
    candidate_k: int,
    max_recall_results: int | None,
) -> dict:
    return {
        "repo_path": args.repo_path,
        "top_k": args.top_k,
        "candidate_k": candidate_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "recall_window": args.recall_window,
        "max_recall_results": max_recall_results,
        "final_k": args.final_k if args.final_k is not None else args.top_k,
        "min_final_score": args.min_final_score,
        "no_final_filter": args.no_final_filter,
    }


def print_llm_setup_hint(provider: str, file) -> None:
    """打印 LLM 配置提示信息"""
    examples = {
        "deepseek": "set DEEPSEEK_API_KEY=your_key",
        "qwen": "set DASHSCOPE_API_KEY=your_key",
        "kimi": "set MOONSHOT_API_KEY=your_key",
        "zhipu": "set ZHIPU_API_KEY=your_key",
        "custom": "set LLM_API_KEY=your_key && set LLM_BASE_URL=https://... && set LLM_MODEL=...",
    }
    print("配置提示:", examples.get(provider, examples["custom"]), file=file)


if __name__ == "__main__":
    # 程序入口点
    raise SystemExit(main())
