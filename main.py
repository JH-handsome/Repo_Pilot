"""RepoPilot 统一 CLI 入口。"""

from __future__ import annotations

import argparse
import sys

from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.runtime import UnifiedRunConfig, resume_unified_query, run_unified_query
from coding_rag.rag.llm_client import (
    OpenAICompatibleChatClient,
    available_provider_names,
    build_llm_config,
)
from coding_rag.rag.prompt import GenerationMode
from coding_rag.rag.trace import render_trace_report, write_trace_json
from coding_rag.tools.env import load_dotenv
from coding_rag.tools.tokenizer import CodeTokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the unified LLM-directed workflow."""
    configure_utf8_stdio()
    provider_choices = available_provider_names() + ["custom"]
    mode_choices = ["judge", "code-understand", "code-generate", "leetcode", "api"]

    parser = argparse.ArgumentParser(description="RepoPilot：由 LLM 自动决定回答或调用代码工具")
    parser.add_argument("repo_path", nargs="?", help="代码仓库路径；不填则进入交互模式")
    parser.add_argument("query", nargs="?", help="问题或修改需求；不填则进入交互模式")
    parser.add_argument("--interactive", action="store_true", help="进入连续交互模式")
    parser.add_argument("--top-k", type=int, default=5, help="每次代码搜索保留的结果数量")
    parser.add_argument("--candidate-k", type=int, help="邻居召回前的候选结果数量")
    parser.add_argument("--recall-window", type=int, default=2, help="种子代码块两侧的召回窗口")
    parser.add_argument("--max-recall-results", type=int, help="召回后的最大代码块数量，0 表示不限制")
    parser.add_argument("--final-k", type=int, help="最终过滤后保留的代码块数量，0 表示不限制")
    parser.add_argument("--min-final-score", type=float, help="最终结果的最低分数")
    parser.add_argument("--no-final-filter", action="store_true", help="跳过召回后的重新评分和过滤")
    parser.add_argument("--chunk-size", type=int, default=40, help="每个代码块的行数")
    parser.add_argument("--overlap", type=int, default=5, help="相邻代码块的重叠行数")
    parser.add_argument("--show-tokens", action="store_true", help="在执行前打印查询分词")
    parser.add_argument("--show-trace", action="store_true", help="打印统一运行 trace")
    parser.add_argument("--trace-out", help="保存本次统一 trace JSON")
    parser.add_argument("--trace-include-text", action="store_true", help="在检索 trace 中保留完整代码文本")
    parser.add_argument("--agent-policy", help="Agent executor 安全策略 JSON 路径")
    parser.add_argument("--rollback", metavar="SNAPSHOT_ID", help="恢复一次已批准写操作的修改前快照")
    parser.add_argument(
        "--mode",
        choices=mode_choices,
        default="judge",
        help="回答风格：judge、code-understand、code-generate、leetcode 或 api",
    )
    parser.add_argument(
        "--llm-provider",
        choices=provider_choices,
        default="deepseek",
        help="LLM 服务提供商预设",
    )
    parser.add_argument("--llm-model", help="覆盖提供商预设的模型名称")
    parser.add_argument("--llm-base-url", help="覆盖提供商预设的基础 URL")
    parser.add_argument("--llm-api-key-env", help="存储 API 密钥的环境变量名")
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM 请求超时秒数")
    parser.add_argument("--llm-max-tokens", type=int, default=2000, help="LLM 最大输出 token 数")
    parser.add_argument("--llm-temperature", type=float, help="可选的 LLM 温度参数")
    parser.add_argument("--llm-context-chars", type=int, default=12000, help="单次工具 observation 最大字符数")
    return parser.parse_args()


def main() -> int:
    """Run the CLI entrypoint."""
    args = parse_args()
    # 入口只负责加载已有环境配置，并把任务交给统一的 LLM 决策循环。
    load_dotenv()
    if args.rollback:
        return run_rollback(args)
    if args.interactive or not args.repo_path or not args.query:
        return run_interactive_cli(args)
    return run_single_turn(args)


def run_single_turn(
    args: argparse.Namespace,
    client: OpenAICompatibleChatClient | None = None,
) -> int:
    """Run one unified answer/tool turn."""
    # 所有输入都从这里进入同一运行时，不再按 ASK/Agent 关键词分流。
    if args.show_tokens:
        print("查询分词结果:")
        print(CodeTokenizer().tokenize(args.query))
        print()

    try:
        llm_client = client or build_chat_client(args)
        config = build_runtime_config(args)
        run = run_unified_query(args.query, config, llm_client)
        while run.status == "approval_required":
            print(render_unified_run(run))
            try:
                approved = input("批准以上这一次写操作并真实执行？[y/N]> ").strip().casefold()
            except EOFError:
                approved = ""
            if approved not in {"y", "yes", "是"}:
                print("未批准，仓库未发生这次写操作。")
                return 4
            run = resume_unified_query(
                run,
                config,
                llm_client,
                approval_fingerprint=run.approval["fingerprint"],
            )
    except ValueError as error:
        print(f"配置或输入错误: {error}", file=sys.stderr)
        print_llm_setup_hint(args.llm_provider, file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 3

    if args.trace_out:
        write_trace_json(args.trace_out, run.trace)
    print(render_unified_run(run))
    if args.show_trace:
        print()
        print(render_trace_report(run.trace))
    return 3 if run.status == "failed" else 0


def run_interactive_cli(args: argparse.Namespace) -> int:
    """Run repeated unified turns with one shared LLM client."""
    repo_path = args.repo_path or prompt_with_default("仓库路径", ".")
    try:
        client = build_chat_client(args)
    except ValueError as error:
        print(f"LLM 配置错误: {error}", file=sys.stderr)
        print_llm_setup_hint(args.llm_provider, file=sys.stderr)
        return 2

    print("RepoPilot 交互模式")
    print(f"当前仓库: {repo_path}")
    print("写工具按调用逐次申请批准；Learning Mode 保持只读。")
    print("输入问题或修改需求，LLM 会自动决定直接回答或调用工具。")
    print("输入 :q、quit 或 exit 退出。")
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
        status = run_single_turn(turn_args, client=client)
        print()
        if status not in {0, 1}:
            return status


def build_chat_client(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    """根据 CLI 参数构建 OpenAI 兼容的聊天客户端。"""
    config = build_llm_config(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
    )
    return OpenAICompatibleChatClient(config)


def run_rollback(args: argparse.Namespace) -> int:
    """Restore one repository-bound snapshot without invoking an LLM."""
    if not args.repo_path:
        print("使用 --rollback 时必须提供仓库路径。", file=sys.stderr)
        return 2
    try:
        result = AgentExecutor(args.repo_path).rollback_snapshot(args.rollback)
    except Exception as error:
        print(f"恢复失败: {error}", file=sys.stderr)
        return 3
    print(f"已恢复快照: {result['snapshot_id']}")
    for path in result["restored_files"]:
        print(f"- 已还原 {path}")
    for path in result["removed_files"]:
        print(f"- 已移除本次操作新建的文件 {path}")
    return 0


def build_runtime_config(args: argparse.Namespace) -> UnifiedRunConfig:
    """将CLI参数转换为统一运行配置。"""
    return UnifiedRunConfig(
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
        agent_policy_path=args.agent_policy,
    )


def render_unified_run(run) -> str:
    """将运行记录渲染为统一格式的文本输出。"""
    summary = run.trace.get("summary") or {}
    lines = [
        "=" * 80,
        "RepoPilot 回答",
        "-" * 80,
        f"状态: {run.status}",
        f"写操作: {'已执行批准调用' if run.execution_enabled else '尚未执行写调用'}",
        (
            "调用统计: "
            f"LLM={summary.get('llm_call_count', 0)} "
            f"tools={summary.get('tool_call_count', 0)} "
            f"observations={summary.get('observation_count', 0)}"
        ),
        "",
        run.answer,
    ]
    approval = getattr(run, "approval", None)
    if approval:
        lines.extend(
            [
                "",
                "待批准写操作:",
                f"工具: {approval.get('tool') or '(unknown)'}",
                "目标文件:",
                *[f"- {path}" for path in approval.get("files") or []],
                f"一次性批准指纹: {approval.get('fingerprint') or '(missing)'}",
            ]
        )
        arguments = approval.get("arguments") or {}
        if approval.get("tool") == "run_command":
            lines.extend(["命令:", str(arguments.get("cmd") or "")])
        elif approval.get("tool") == "apply_patch":
            lines.extend(["待应用补丁:", str(arguments.get("diff") or "")])
    for event in run.trace.get("events") or []:
        artifacts = event.get("artifacts") if isinstance(event, dict) else None
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        result = artifacts.get("patch") or artifacts.get("command")
        if not isinstance(result, dict) or not result.get("snapshot_id"):
            continue
        lines.extend(["", f"回滚快照: {result['snapshot_id']}"])
        post_diff = result.get("post_change_diff") or {}
        if post_diff.get("text"):
            lines.extend(["修改后 diff:", str(post_diff["text"])])
    return "\n".join(lines)


def prompt_with_default(label: str, default: str) -> str:
    """交互式提示用户输入，若无输入或遇到EOF则返回默认值。"""
    try:
        value = input(f"{label} [{default}]> ").strip()
    except EOFError:
        return default
    return value or default


def configure_utf8_stdio() -> None:
    """Keep Chinese CLI text readable in Windows terminals and redirected logs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def print_llm_setup_hint(provider: str, file=None) -> None:
    """打印指定 LLM 提供商的 API key 配置提示。"""
    target = file or sys.stdout
    print(
        f"请在已有 .env 或环境变量中配置 {provider} API key；RepoPilot 不会自动创建 .env。",
        file=target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
