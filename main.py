"""RepoPilot CLI."""



import argparse
import sys

from coding_rag.agent import CodeAgentConfig, agent_run_to_dict, render_agent_run, run_code_agent
from coding_rag.agent.plan_reviewer import render_agent_plan_review, review_agent_plan
from coding_rag.tools.bm25 import SearchResult
from coding_rag.tools.env import ensure_dotenv, load_dotenv
from coding_rag.rag.llm_client import (
    OpenAICompatibleChatClient,
    available_provider_names,
    build_llm_config,
)
from coding_rag.rag.judge import LLMJudge
from coding_rag.agent.planner import (
    AgentPlanConfig,
    AskModeRun,
    AskModeConfig,
    ReActAgentInterface,
    WorkflowMode,
    render_agent_plan_prompt,
    render_agent_plan_run,
    classify_task,
    render_ask_prompt,
    render_task_plan,
    run_ask_mode,
)
from coding_rag.tools.tokenizer import CodeTokenizer

from coding_rag.rag.prompt import GenerationMode
from coding_rag.rag.trace import render_trace_report, write_trace_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    provider_choices = available_provider_names() + ["custom"]
    mode_choices = ["judge", "code-understand", "code-generate", "leetcode", "api"]

    parser = argparse.ArgumentParser(description="轻量级代码 RAG 检索工具")
    parser.add_argument("repo_path", nargs="?", help="要搜索的代码仓库路径；不填则进入交互模式")
    parser.add_argument("query", nargs="?", help="搜索查询内容；不填则进入交互模式")
    parser.add_argument("--interactive", action="store_true", help="start interactive mode")
    parser.add_argument("--top-k", type=int, default=5, help="initial retrieval result count")
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="召回前的候选结果数量，默认使用 --top-k",
    )
    parser.add_argument(
        "--recall-window",
        type=int,
        default=2,
        help="neighbor chunks recalled around each seed result; 0 disables recall",
    )
    parser.add_argument(
        "--max-recall-results",
        type=int,
        help="召回扩展后的最大代码块数量，设为 0 则不限制",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        help="召回过滤后保留的代码块数量，默认使用 --top-k，设为 0 则不限制",
    )
    parser.add_argument(
        "--min-final-score",
        type=float,
        help="drop final results below this score",
    )
    parser.add_argument(
        "--no-final-filter",
        action="store_true",
        help="跳过召回后的重新评分/过滤阶段",
    )
    parser.add_argument("--chunk-size", type=int, default=40, help="每个代码块的行数")
    parser.add_argument("--overlap", type=int, default=5, help="代码块之间的重叠行数")
    parser.add_argument("--show-tokens", action="store_true", help="print query tokens before search")
    parser.add_argument("--show-trace", action="store_true", help="print RAG retrieval trace")
    parser.add_argument("--trace-out", help="保存本次 RAG 检索轨迹 JSON")
    parser.add_argument("--trace-include-text", action="store_true", help="include full code text in trace JSON")
    parser.add_argument(
        "--workflow-mode",
        choices=[mode.value for mode in WorkflowMode],
        default=WorkflowMode.AUTO.value,
        help="任务工作流：ask 为代码问答，agent 为 ReAct Agent 接口，auto 自动分类",
    )
    parser.add_argument("--show-plan", action="store_true", help="print task classification and plan")
    parser.add_argument("--show-ask-prompt", action="store_true", help="打印 ASK 模式发送给 LLM 的 prompt")
    parser.add_argument("--show-agent-prompt", action="store_true", help="打印 Agent 计划模式发送给 LLM 的 prompt")
    parser.add_argument("--review-agent-plan", action="store_true", help="使用 LLM 审查 Agent 计划是否合理")
    parser.add_argument("--agent", action="store_true", help="以代码 Agent 工作流运行任务（兼容旧参数）")
    parser.add_argument("--agent-exec", action="store_true", help="run the code Agent execution workflow")
    parser.add_argument("--dry-run", action="store_true", help="Agent dry run: skip memory and run-log writes")
    parser.add_argument("--safe-mode", action="store_true", help="Agent safe mode: read-only planning without persistent writes")
    parser.add_argument("--agent-memory", default="artifacts/agent_memory.jsonl", help="Agent 记忆 JSONL 路径")
    parser.add_argument("--agent-memory-limit", type=int, default=5, help="本任务最多读取的相关记忆数量")
    parser.add_argument("--agent-policy", help="Agent executor safety policy JSON path")
    parser.add_argument(
        "--agent-log",
        default="artifacts/agent_run_log.jsonl",
        help="Agent 运行日志 JSONL 路径",
    )
    parser.add_argument("--no-agent-log", action="store_true", help="不写入 Agent 运行日志")

    parser.add_argument("--llm", action="store_true", help="use an LLM to answer from retrieved context")
    parser.add_argument(
        "--mode",
        choices=mode_choices,
        default="judge",
        help="generation mode: judge, code-understand, code-generate, leetcode, or api",
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
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM request timeout in seconds")
    parser.add_argument("--llm-max-tokens", type=int, default=2000, help="LLM max output tokens")
    parser.add_argument("--llm-temperature", type=float, help="可选的 LLM 温度参数")
    parser.add_argument(
        "--llm-context-chars",
        type=int,
        default=12000,
        help="maximum retrieved context characters sent to LLM",
    )
    parser.add_argument(
        "--legacy-judge",
        action="store_true",
        help="use legacy LLM judge mode",
    )
    return parser.parse_args()


def main() -> int:
    """Run the CLI entrypoint."""
    args = parse_args()
    if args.llm:
        if not is_agent_no_write_mode(args):
            ensure_dotenv()
        load_dotenv()

    if args.interactive or not args.repo_path or not args.query:
        return run_interactive_cli(args)

    return run_single_turn(args)


def run_single_turn(args: argparse.Namespace) -> int:
    """Run one ASK or Agent turn."""
    if args.agent or args.agent_exec:
        return run_agent_cli(args)

    requested_mode = WorkflowMode(args.workflow_mode)
    plan = classify_task(args.query, requested_mode=requested_mode, generation_mode=GenerationMode(args.mode))
    if plan.mode == WorkflowMode.AGENT:
        return run_react_agent_cli(args)

    return run_ask_cli(args, requested_mode=plan.mode)


def is_agent_no_write_mode(args: argparse.Namespace) -> bool:
    return bool((args.agent or args.agent_exec) and (args.dry_run or args.safe_mode))


def run_interactive_cli(args: argparse.Namespace) -> int:
    """Run interactive CLI mode."""
    repo_path = args.repo_path or prompt_with_default("仓库路径", ".")
    if not args.llm:
        args.llm = prompt_yes_no("Call LLM for answers", default=False)
        if args.llm:
            if not is_agent_no_write_mode(args):
                ensure_dotenv()
            load_dotenv()

    print("RepoPilot interactive mode")
    print(f"Current repository: {repo_path}")
    print(f"LLM answers: {'on' if args.llm else 'off'}")
    print("Questions use ASK; change/fix/implement requests use Agent planning.")
    print("Type :q, quit, or exit to quit.")
    print()

    while True:
        try:
            query = input("问题/需求> ").strip()
        except EOFError:
            print()
            return 0

        if query.casefold() in {":q", "q", "quit", "exit"}:
            return 0
        if not query:
            continue

        turn_args = argparse.Namespace(**vars(args))
        turn_args.repo_path = repo_path
        turn_args.query = query
        turn_args.interactive = False
        print()
        status = run_single_turn(turn_args)
        print()
        if status not in {0, 1}:
            return status


def prompt_with_default(label: str, default: str) -> str:
    try:
        value = input(f"{label} [{default}]> ").strip()
    except EOFError:
        return default
    return value or default


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    try:
        value = input(f"{label}? [{default_text}]> ").strip().casefold()
    except EOFError:
        return default
    if not value:
        return default
    return value in {"y", "yes", "true", "1"}


def run_ask_cli(args: argparse.Namespace, requested_mode: WorkflowMode) -> int:
    """Run ASK mode."""
    if args.show_tokens:
        print("查询分词结果:")
        print(CodeTokenizer().tokenize(args.query))
        print()

    client = None
    if args.llm and not args.legacy_judge:
        try:
            config = build_llm_config(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
            )
        except ValueError as error:
            print(f"LLM 配置错误: {error}", file=sys.stderr)
            print_llm_setup_hint(args.llm_provider, file=sys.stderr)
            return 2
        client = OpenAICompatibleChatClient(config)

    try:
        run = run_ask_mode(
            query=args.query,
            config=AskModeConfig(
                repo_path=args.repo_path,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                recall_window=args.recall_window,
                max_recall_results=args.max_recall_results,
                final_k=args.final_k,
                min_final_score=args.min_final_score,
                no_final_filter=args.no_final_filter,
                generation_mode=GenerationMode(args.mode),
                max_context_chars=args.llm_context_chars,
                include_trace_text=args.trace_include_text,
            ),
            client=client,
            requested_mode=requested_mode,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 3

    if args.trace_out:
        write_trace_json(args.trace_out, ask_run_to_dict(run))
    if args.show_plan:
        print(render_task_plan(run.plan))
        print()
    if args.show_trace:
        print(render_trace_report(run.trace))
        print()
    if args.show_ask_prompt:
        print(render_ask_prompt(run.messages))
        print()

    if args.recall_window > 0:
        print(
            f"Hybrid 种子: {len(run.seed_results)} | "
            f"召回代码块: {len(run.recalled_results)} | "
            f"最终代码块: {len(run.final_results)}"
        )
        print()

    print_search_results(run.final_results)

    if args.llm:
        print()
        print("=" * 80)
        if args.legacy_judge:
            print("LLM 判断（旧版模式）")
            print("-" * 80)
            try:
                answer = run_llm_judge(args, run.final_results)
            except ValueError as error:
                print(f"LLM 配置错误: {error}", file=sys.stderr)
                print_llm_setup_hint(args.llm_provider, file=sys.stderr)
                return 2
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                return 3
            print(answer.strip())
        else:
            print(f"ASK LLM answer (mode: {args.mode})")
            print("-" * 80)
            if client is not None:
                print(f"提供商: {client.config.provider}")
                print(f"模型: {client.config.model}")
                print(f"API 密钥环境变量: {client.config.api_key_env}")
                print()
            print((run.answer or "").strip())
    else:
        print()
        print("LLM answers are disabled. Use `--llm` to ask the model to summarize retrieved context.")

    return 0


def ask_run_to_dict(run: AskModeRun) -> dict:
    return {
        "query": run.query,
        "plan": {
            "mode": run.plan.mode.value,
            "intent": run.plan.intent,
            "generation_mode": run.plan.generation_mode.value,
            "steps": run.plan.steps,
            "reason": run.plan.reason,
        },
        "trace": run.trace,
        "messages": run.messages,
        "answer": run.answer,
    }


def run_react_agent_cli(args: argparse.Namespace) -> int:
    """Run ReAct Agent planning mode."""
    interface = ReActAgentInterface()
    client = None
    if args.llm:
        try:
            config = build_llm_config(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                temperature=args.llm_temperature,
            )
        except ValueError as error:
            print(f"LLM 配置错误: {error}", file=sys.stderr)
            print_llm_setup_hint(args.llm_provider, file=sys.stderr)
            return 2
        client = OpenAICompatibleChatClient(config)

    try:
        run = interface.build_plan(
            args.query,
            config=AgentPlanConfig(
                repo_path=args.repo_path,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                recall_window=args.recall_window,
                max_recall_results=args.max_recall_results,
                final_k=args.final_k,
                min_final_score=args.min_final_score,
                max_context_chars=args.llm_context_chars,
                agent_policy_path=args.agent_policy,
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
        write_trace_json(args.trace_out, agent_plan_run_to_dict(run))

    if args.show_agent_prompt:
        print(render_agent_plan_prompt(run.messages))
        print()

    print(render_agent_plan_run(run))
    if args.review_agent_plan:
        if client is None:
            print()
            print("LLM is disabled; use `--llm --review-agent-plan` to review the Agent plan.")
        elif not run.plan_text:
            print()
            print("No LLM plan text is available to review.")
        else:
            review = review_agent_plan(run, client)
            print()
            print(render_agent_plan_review(review))
    if not args.llm:
        print()
        print("LLM is disabled; Agent planning prompt has been prepared.")
        print("Add `--llm` to generate a model-written plan.")
    print()
    print("Tool interfaces are declared but not executed in planning mode.")
    return 0


def agent_plan_run_to_dict(run) -> dict:
    return {
        "task": run.task,
        "plan": {
            "mode": run.plan.mode.value,
            "intent": run.plan.intent,
            "generation_mode": run.plan.generation_mode.value,
            "steps": run.plan.steps,
            "reason": run.plan.reason,
        },
        "tools": [
            {"name": tool.name, "purpose": tool.purpose, "input_schema": tool.input_schema}
            for tool in run.tools
        ],
        "trace": run.trace,
        "messages": run.messages,
        "plan_text": run.plan_text,
    }


def run_agent_cli(args: argparse.Namespace) -> int:
    """Run the code Agent workflow."""
    no_persistent_writes = args.dry_run or args.safe_mode
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
                run_log_path=None if args.no_agent_log or no_persistent_writes else args.agent_log,
                max_context_chars=args.llm_context_chars,
                dry_run=args.dry_run,
                safe_mode=args.safe_mode,
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
    """Run legacy LLM judge mode."""
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


def print_search_results(results: list[SearchResult]) -> None:
    """Print search results."""
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print("=" * 80)
        print(f"结果 {index} | 分数: {result.score:.4f}")
        print(f"来源标签: {result.source}")
        print(f"来源: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print("-" * 80)
        print(chunk.text.rstrip())


def print_llm_setup_hint(provider: str, file) -> None:
    """Print LLM setup hints."""
    examples = {
        "deepseek": "set DEEPSEEK_API_KEY=your_key",
        "qwen": "set DASHSCOPE_API_KEY=your_key",
        "kimi": "set MOONSHOT_API_KEY=your_key",
        "zhipu": "set ZHIPU_API_KEY=your_key",
        "custom": "set LLM_API_KEY=your_key && set LLM_BASE_URL=https://... && set LLM_MODEL=...",
    }
    print("配置提示:", examples.get(provider, examples["custom"]), file=file)


if __name__ == "__main__":
    raise SystemExit(main())
