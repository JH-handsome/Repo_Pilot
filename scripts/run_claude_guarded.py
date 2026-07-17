"""Run Claude Code with per-task token and cache-miss guards.

The wrapper reads a prompt from stdin, forwards it to ``claude -p`` in
``stream-json`` mode, and stops the process when the observed usage exceeds the
configured limits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_TASK_TOKENS = 80_000
DEFAULT_MAX_CACHE_MISS_TOKENS = 20_000
DEFAULT_MAX_CACHE_MISS_RATIO = 0.35
DEFAULT_MAX_BUDGET_USD = 1.00
DEFAULT_TIMEOUT_SECONDS = 300


@dataclass
class ClaudeUsage:
    """汇总 Claude Code 子进程输出事件中观测到的令牌用量。

    Attributes:
        input_tokens: 输入令牌数。
        output_tokens: 输出令牌数。
        cache_read_input_tokens: 缓存读取的输入令牌数。
        cache_creation_input_tokens: 缓存创建的输入令牌数。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def task_tokens(self) -> int:
        """总任务令牌数 = input_tokens + output_tokens + cache_read_input_tokens + cache_creation_input_tokens。"""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def cache_miss_tokens(self) -> int:
        """缓存未命中令牌数 = input_tokens + cache_creation_input_tokens。"""
        return self.input_tokens + self.cache_creation_input_tokens

    @property
    def cache_input_tokens(self) -> int:
        """缓存输入令牌数 = input_tokens + cache_read_input_tokens + cache_creation_input_tokens。"""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def cache_miss_ratio(self) -> float:
        """缓存未命中比率 = cache_miss_tokens / cache_input_tokens，分母非正时返回 0.0。"""
        if self.cache_input_tokens <= 0:
            return 0.0
        return self.cache_miss_tokens / self.cache_input_tokens


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括守卫限额、权限、超时和日志参数。

    解析 --max-task-tokens、--max-cache-miss-tokens、--max-cache-miss-ratio、
    --max-budget-usd 等守卫限额，--permission-mode 权限模式，--timeout 超时秒数，
    以及 --output-log 日志路径。其余未知参数保留并转发给 Claude Code，
    同时去掉开头的 ``--`` 分隔符。
    """
    parser = argparse.ArgumentParser(description="Run Claude Code with token guards.")
    parser.add_argument(
        "--max-task-tokens",
        type=int,
        default=int(os.environ.get("CLAUDE_GUARD_MAX_TASK_TOKENS", DEFAULT_MAX_TASK_TOKENS)),
        help="Stop when total observed tokens exceed this value.",
    )
    parser.add_argument(
        "--max-cache-miss-tokens",
        type=int,
        default=int(
            os.environ.get(
                "CLAUDE_GUARD_MAX_CACHE_MISS_TOKENS",
                DEFAULT_MAX_CACHE_MISS_TOKENS,
            )
        ),
        help="Stop when non-cached input/cache-creation tokens exceed this value.",
    )
    parser.add_argument(
        "--max-cache-miss-ratio",
        type=float,
        default=float(
            os.environ.get(
                "CLAUDE_GUARD_MAX_CACHE_MISS_RATIO",
                DEFAULT_MAX_CACHE_MISS_RATIO,
            )
        ),
        help="Stop when cache miss ratio exceeds this value after enough tokens are seen.",
    )
    parser.add_argument(
        "--min-cache-input-tokens",
        type=int,
        default=int(os.environ.get("CLAUDE_GUARD_MIN_CACHE_INPUT_TOKENS", "5000")),
        help="Do not apply the cache miss ratio guard below this input-token count.",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=float(os.environ.get("CLAUDE_GUARD_MAX_BUDGET_USD", DEFAULT_MAX_BUDGET_USD)),
        help="Pass-through hard dollar cap for Claude Code.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CLAUDE_GUARD_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        help="Stop Claude Code after this many seconds.",
    )
    parser.add_argument(
        "--permission-mode",
        default=os.environ.get("CLAUDE_GUARD_PERMISSION_MODE", "acceptEdits"),
        choices=("acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"),
    )
    parser.add_argument(
        "--output-log",
        default=os.environ.get("CLAUDE_GUARD_OUTPUT_LOG", "artifacts/claude_guard_log.jsonl"),
        help="JSONL file for guard events and final usage.",
    )
    args, claude_args = parser.parse_known_args()
    if claude_args and claude_args[0] == "--":
        claude_args = claude_args[1:]
    args.claude_args = claude_args
    return args


def main() -> int:
    """从标准输入读取提示，启动 ``claude`` stream-json 子进程。

    逐行转发子进程 stdout 并聚合 ClaudeUsage；守卫在观察到 token 或缓存阈值超限时终止子进程并记录日志；美元预算仅传给 Claude Code，非守卫观测的预算上限；子进程等待超时时同样终止。
    守卫终止返回 124，空提示返回 2。
    """
    args = parse_args()
    prompt = sys.stdin.read().lstrip("\ufeff")
    if not prompt.strip():
        print("Claude guard: prompt is empty.", file=sys.stderr)
        return 2

    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "claude",
        "-p",
        "--permission-mode",
        args.permission_mode,
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-budget-usd",
        str(args.max_budget_usd),
    ]
    command.extend(args.claude_args or [])

    usage = ClaudeUsage()
    stopped_reason: str | None = None

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(prompt)
        process.stdin.close()

        try:
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                event = parse_json_line(line)
                if event is None:
                    continue
                update_usage_from_event(usage, event)
                stopped_reason = guard_reason(args, usage)
                if stopped_reason:
                    write_log(log_file, "guard_stop", stopped_reason, usage)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                stopped_reason = "timeout"
                write_log(log_file, "guard_stop", stopped_reason, usage)
                process.kill()
                return_code = process.wait()
        finally:
            stderr = process.stderr.read() if process.stderr is not None else ""

        if stderr:
            sys.stderr.write(stderr)

        write_log(log_file, "finished", stopped_reason or "completed", usage)

    if stopped_reason:
        print(f"Claude guard stopped task: {stopped_reason}", file=sys.stderr)
        return 124
    return return_code


def parse_json_line(line: str) -> dict[str, Any] | None:
    """仅当行是 JSON 对象时返回字典，否则返回 None。"""
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def update_usage_from_event(usage: ClaudeUsage, event: dict[str, Any]) -> None:
    """从事件的直接/message usage 与按模型汇总 usage 中取各字段较大值更新峰值观测。"""
    usage_payload = find_usage_payload(event)
    if usage_payload:
        usage.input_tokens = max(usage.input_tokens, int(usage_payload.get("input_tokens") or 0))
        usage.output_tokens = max(usage.output_tokens, int(usage_payload.get("output_tokens") or 0))
        usage.cache_read_input_tokens = max(
            usage.cache_read_input_tokens,
            int(usage_payload.get("cache_read_input_tokens") or 0),
        )
        usage.cache_creation_input_tokens = max(
            usage.cache_creation_input_tokens,
            int(usage_payload.get("cache_creation_input_tokens") or 0),
        )

    model_payloads = find_model_usage_payloads(event)
    if model_payloads:
        aggregate = ClaudeUsage()
        for model_usage in model_payloads:
            aggregate.input_tokens += int(model_usage.get("inputTokens") or 0)
            aggregate.output_tokens += int(model_usage.get("outputTokens") or 0)
            aggregate.cache_read_input_tokens += int(model_usage.get("cacheReadInputTokens") or 0)
            aggregate.cache_creation_input_tokens += int(
                model_usage.get("cacheCreationInputTokens") or 0
            )
        usage.input_tokens = max(usage.input_tokens, aggregate.input_tokens)
        usage.output_tokens = max(usage.output_tokens, aggregate.output_tokens)
        usage.cache_read_input_tokens = max(
            usage.cache_read_input_tokens,
            aggregate.cache_read_input_tokens,
        )
        usage.cache_creation_input_tokens = max(
            usage.cache_creation_input_tokens,
            aggregate.cache_creation_input_tokens,
        )


def find_usage_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    """查找事件顶层或 message 内的 usage 字典并返回。"""
    direct = event.get("usage")
    if isinstance(direct, dict):
        return direct
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        return message["usage"]
    return None


def find_model_usage_payloads(event: dict[str, Any]) -> list[dict[str, Any]]:
    """从 modelUsage 映射中过滤并返回字典值列表。"""
    model_usage = event.get("modelUsage")
    if not isinstance(model_usage, dict):
        return []
    return [value for value in model_usage.values() if isinstance(value, dict)]


def guard_reason(args: argparse.Namespace, usage: ClaudeUsage) -> str | None:
    """按总任务token、缓存未命中token、达到最小缓存输入后的未命中比例依次检查并返回首个停止原因。"""
    if usage.task_tokens > args.max_task_tokens:
        return f"task tokens {usage.task_tokens} exceeded limit {args.max_task_tokens}"
    if usage.cache_miss_tokens > args.max_cache_miss_tokens:
        return (
            f"cache miss tokens {usage.cache_miss_tokens} exceeded limit "
            f"{args.max_cache_miss_tokens}"
        )
    if (
        usage.cache_input_tokens >= args.min_cache_input_tokens
        and usage.cache_miss_ratio > args.max_cache_miss_ratio
    ):
        return (
            f"cache miss ratio {usage.cache_miss_ratio:.2%} exceeded limit "
            f"{args.max_cache_miss_ratio:.2%}"
        )
    return None


def write_log(log_file, event: str, reason: str, usage: ClaudeUsage) -> None:
    """把事件、原因和用量快照写为一条JSONL并刷新。"""
    payload = {
        "event": event,
        "reason": reason,
        "usage": {
            "task_tokens": usage.task_tokens,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "cache_miss_ratio": usage.cache_miss_ratio,
        },
    }
    log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log_file.flush()


if __name__ == "__main__":
    raise SystemExit(main())
