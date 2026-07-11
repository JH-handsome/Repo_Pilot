"""任务分类与 ASK/Agent 工作流入口。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.safety import AgentSafetyPolicy, load_agent_safety_policy
from coding_rag.rag.ask import (
    AskModeConfig,
    AskModeRun,
    build_ask_messages,
    retrieve_for_ask,
)
from coding_rag.tools.agent_readonly import ReadOnlyAgentTools, extract_task_identifiers, format_observations
from coding_rag.tools.bm25 import SearchResult
from coding_rag.rag.citation_validator import append_citation_validation_report, validate_answer_citations
from coding_rag.rag.prompt import GenerationMode, format_results_as_context
from coding_rag.rag.trace import build_tool_event, build_trace_event, build_trace_run


class WorkflowMode(str, Enum):
    """用户任务的执行模式。"""

    AUTO = "auto"
    ASK = "ask"
    AGENT = "agent"


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class TaskPlan:
    """一次用户输入的分类结果和执行计划。"""

    mode: WorkflowMode
    intent: str
    generation_mode: GenerationMode
    steps: list[str]
    reason: str


@dataclass(frozen=True)
class AgentToolSpec:
    """ReAct Agent 可调用工具的接口定义；当前只声明，不执行。"""

    name: str
    purpose: str
    input_schema: dict[str, str]


@dataclass(frozen=True)
class AgentPlanConfig:
    repo_path: str | Path = "."
    top_k: int = 5
    candidate_k: int | None = None
    chunk_size: int = 40
    overlap: int = 5
    recall_window: int = 2
    max_recall_results: int | None = None
    final_k: int | None = None
    min_final_score: float | None = None
    max_context_chars: int = 12000
    execute_readonly_tools: bool = False
    agent_policy_path: str | Path | None = None


@dataclass(frozen=True)
class AgentPlanRun:
    task: str
    plan: TaskPlan
    tools: list[AgentToolSpec]
    seed_results: list[SearchResult]
    recalled_results: list[SearchResult]
    final_results: list[SearchResult]
    trace: dict
    messages: list[dict[str, str]]
    plan_text: str | None
    observations: list[dict]


@dataclass(frozen=True)
class AgentToolCall:
    thought: str
    tool: str
    arguments: dict
    expected_observation: str
    raw_text: str


@dataclass(frozen=True)
class AgentExecutionRun:
    task: str
    plan: TaskPlan
    trace: dict
    messages: list[dict[str, str]]
    tool_calls: list[AgentToolCall]
    observations: list[dict]
    final_text: str | None


class ReActAgentInterface:
    """Agent 模式接口；当前只生成计划，不执行工具调用。"""

    def __init__(self, tools: list[AgentToolSpec] | None = None):
        self.tools = tools or default_agent_tools()

    def plan(self, task: str) -> TaskPlan:
        return classify_task(task, requested_mode=WorkflowMode.AGENT)

    def run(
        self,
        task: str,
        config: AgentPlanConfig | None = None,
        client: ChatClient | None = None,
        *,
        dry_run: bool = True,
        safe_mode: bool = False,
        max_steps: int = 3,
    ) -> AgentExecutionRun:
        return run_agent_execution_loop(
            task,
            config=config or AgentPlanConfig(),
            client=client,
            tools=self.tools,
            dry_run=dry_run,
            safe_mode=safe_mode,
            max_steps=max_steps,
        )

    def build_plan(
        self,
        task: str,
        config: AgentPlanConfig,
        client: ChatClient | None = None,
    ) -> AgentPlanRun:
        return run_agent_plan_mode(task, config=config, client=client, tools=self.tools)


def classify_task(
    user_input: str,
    requested_mode: WorkflowMode | str = WorkflowMode.AUTO,
    generation_mode: GenerationMode = GenerationMode.JUDGE,
) -> TaskPlan:
    """根据用户输入和显式模式生成执行计划。"""
    mode = WorkflowMode(requested_mode)
    if mode == WorkflowMode.AUTO:
        mode = infer_mode(user_input)

    if mode == WorkflowMode.AGENT:
        return TaskPlan(
            mode=WorkflowMode.AGENT,
            intent="plan_and_act",
            generation_mode=generation_mode,
            steps=[
                "理解项目要求并拆分目标",
                "通过工具检索、读取和验证代码证据",
                "按 ReAct 的 thought/action/observation 循环推进任务",
                "产出补丁计划、验证结果和运行记录",
            ],
            reason="任务包含修改、实现、修复或项目搭建意图，需要 Agent 执行链路。",
        )

    return TaskPlan(
        mode=WorkflowMode.ASK,
        intent="code_question_answering",
        generation_mode=generation_mode,
        steps=[
            "基于问题检索代码库",
            "召回相邻上下文并过滤结果",
            "把证据块整理为 RAG prompt",
            "调用大模型生成基于引用的回答",
        ],
        reason="任务更像代码库问答，适合 ASK 模式。",
    )


def infer_mode(user_input: str) -> WorkflowMode:
    q = user_input.casefold()
    agent_keywords = [
        "修改",
        "修复",
        "新增",
        "增加",
        "实现",
        "搭建",
        "重构",
        "优化",
        "改代码",
        "patch",
        "fix",
        "implement",
        "refactor",
    ]
    return WorkflowMode.AGENT if any(keyword in q for keyword in agent_keywords) else WorkflowMode.ASK


def run_ask_mode(
    query: str,
    config: AskModeConfig,
    client: ChatClient | None = None,
    requested_mode: WorkflowMode | str = WorkflowMode.ASK,
) -> AskModeRun:
    """执行 ASK 模式：RAG 检索、prompt 组装和可选 LLM 问答。"""
    plan = classify_task(query, requested_mode=requested_mode, generation_mode=config.generation_mode)
    if plan.mode != WorkflowMode.ASK:
        raise ValueError("run_ask_mode 只能执行 ASK 模式计划")

    seed_results, recalled_results, final_results, trace = retrieve_for_ask(query, config)
    context = format_results_as_context(final_results, max_context_chars=config.max_context_chars)
    messages = build_ask_messages(query, context, config.generation_mode)
    answer = None
    if client is not None:
        raw_answer = client.complete(messages).strip()
        validation = validate_answer_citations(raw_answer, final_results)
        answer = append_citation_validation_report(raw_answer, validation)

    return AskModeRun(
        query=query,
        plan=plan,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        trace=trace,
        messages=messages,
        answer=answer,
    )


def run_agent_plan_mode(
    task: str,
    config: AgentPlanConfig,
    client: ChatClient | None = None,
    tools: list[AgentToolSpec] | None = None,
) -> AgentPlanRun:
    """执行 Agent 计划模式：分析任务并生成 ReAct 工作流计划。

    Agent 计划阶段默认不检索仓库；只有开启 execute_readonly_tools 时才运行只读工具观察。
    """
    plan = classify_task(task, requested_mode=WorkflowMode.AGENT)
    tool_specs = tools or default_agent_tools()
    seed_results: list[SearchResult] = []
    recalled_results: list[SearchResult] = []
    final_results: list[SearchResult] = []
    trace = build_agent_plan_trace(task, config)
    observations = []
    if config.execute_readonly_tools:
        observations = collect_readonly_observations(task, config)
        trace["readonly_observations"] = observations
        trace["artifacts"]["tools"] = observations
        trace["events"].append(
            build_trace_event(
                step="readonly_observations",
                status="success",
                output_summary={"observation_count": len(observations)},
                artifacts={"observations": observations},
            )
        )
    context = "(计划阶段未检索仓库；请在分到具体功能后用只读工具检查是否已有对应实现代码。)"
    messages = build_agent_plan_messages(task, plan, context, tool_specs, observations=observations)
    plan_text = client.complete(messages).strip() if client is not None else None
    trace["summary"]["workflow_mode"] = WorkflowMode.AGENT.value
    return AgentPlanRun(
        task=task,
        plan=plan,
        tools=tool_specs,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        trace=trace,
        messages=messages,
        plan_text=plan_text,
        observations=observations,
    )


def build_agent_plan_trace(task: str, config: AgentPlanConfig) -> dict:
    summary = {
        "workflow_mode": WorkflowMode.AGENT.value,
        "seed_count": 0,
        "recalled_count": 0,
        "final_count": 0,
        "planning_retrieval": "disabled",
    }
    params = {
        "repo_path": str(config.repo_path),
        "top_k": config.top_k,
        "candidate_k": config.candidate_k,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "recall_window": config.recall_window,
        "max_recall_results": config.max_recall_results,
        "final_k": config.final_k,
        "min_final_score": config.min_final_score,
        "max_context_chars": config.max_context_chars,
        "execute_readonly_tools": config.execute_readonly_tools,
        "agent_policy_path": str(config.agent_policy_path) if config.agent_policy_path else None,
    }
    return {
        "trace_version": "1.0",
        "run": build_trace_run(
            mode="agent_plan",
            task=task,
            repo_path=config.repo_path,
            params=params,
            summary=summary,
        ),
        "events": [
            build_trace_event(
                step="agent_plan_setup",
                status="success",
                input={"task": task},
                output_summary={"planning_retrieval": "disabled"},
                artifacts={"tools_declared": True},
            )
        ],
        "artifacts": {"retrieval": {}, "agent": {"trajectory": []}, "tools": []},
        "query": task,
        "summary": summary,
        "params": params,
        "trajectory": [],
    }


def collect_readonly_observations(task: str, config: AgentPlanConfig) -> list[dict]:
    tools = ReadOnlyAgentTools(
        config.repo_path,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )
    observations: list[dict] = []

    calls: list[tuple[str, dict]] = [
        ("search_code", {"query": task, "top_k": config.top_k}),
        ("list_files", {"pattern": "**/*.py", "limit": 50}),
    ]
    for identifier in extract_task_identifiers(task, limit=3):
        calls.append(("inspect_symbol", {"symbol": identifier, "context_lines": 2}))

    for name, arguments in calls:
        try:
            output = tools.call(name, arguments)
            observations.append({"tool": name, "input": arguments, "output": output})
        except Exception as error:
            observations.append({"tool": name, "input": arguments, "error": str(error)})
    return observations


def run_agent_execution_loop(
    task: str,
    config: AgentPlanConfig,
    client: ChatClient | None,
    tools: list[AgentToolSpec] | None = None,
    *,
    dry_run: bool = True,
    safe_mode: bool = False,
    max_steps: int = 3,
    safety_policy: AgentSafetyPolicy | None = None,
) -> AgentExecutionRun:
    """Run a minimal ReAct tool loop with safe executor-backed tools."""
    if client is None:
        raise ValueError("Agent execution loop requires an LLM client")
    plan = classify_task(task, requested_mode=WorkflowMode.AGENT)
    tool_specs = tools or default_agent_tools()
    policy = safety_policy or load_agent_safety_policy(config.repo_path, config.agent_policy_path)
    executor = AgentExecutor(
        config.repo_path,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        dry_run=dry_run,
        safe_mode=safe_mode,
        safety_policy=policy,
    )
    trace = build_agent_execution_trace(task, config, dry_run=dry_run, safe_mode=safe_mode)
    messages = build_agent_execution_messages(task, plan, tool_specs)
    tool_calls: list[AgentToolCall] = []
    observations: list[dict] = []
    final_text: str | None = None

    allowed_tools = {"read_file", "search_code", "apply_patch", "run_command", "inspect_diff"}
    for step_index in range(1, max_steps + 1):
        raw = client.complete(messages).strip()
        call = parse_agent_tool_call(raw)
        tool_calls.append(call)
        if call.tool == "finish":
            final_text = call.expected_observation or call.thought
            trace["events"].append(
                build_trace_event(
                    step="finish",
                    status="success",
                    input={"thought": call.thought},
                    output_summary={"final": True},
                    artifacts={"message": final_text},
                )
            )
            break
        if call.tool not in allowed_tools:
            error = {"type": "UnknownTool", "message": f"unknown tool: {call.tool}", "recoverable": True}
            observation = {"tool": call.tool, "input": call.arguments, "error": error["message"]}
            observations.append(observation)
            trace["events"].append(
                build_trace_event(
                    step=call.tool or "unknown_tool",
                    status="failed",
                    input=call.arguments,
                    output_summary={"step": step_index},
                    artifacts={"thought": call.thought, "expected_observation": call.expected_observation},
                    error=error,
                )
            )
            break
        try:
            result = executor.call(call.tool, call.arguments)
            observation = {"tool": call.tool, "input": call.arguments, "output": result}
            event = build_tool_event(tool=call.tool, input=call.arguments, result=tool_result_payload(call.tool, result))
            event["artifacts"]["thought"] = call.thought
            event["artifacts"]["expected_observation"] = call.expected_observation
        except Exception as error:
            observation = {"tool": call.tool, "input": call.arguments, "error": str(error)}
            event = build_trace_event(
                step=call.tool,
                status="failed",
                input=call.arguments,
                output_summary={"step": step_index},
                artifacts={"thought": call.thought, "expected_observation": call.expected_observation},
                error={"type": error.__class__.__name__, "message": str(error), "recoverable": True},
            )
        observations.append(observation)
        trace["events"].append(event)
        trace["artifacts"]["tools"].append(observation)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "Observation:\n" + json.dumps(observation, ensure_ascii=False)})

    trace["run"]["summary"]["tool_call_count"] = len(tool_calls)
    trace["run"]["summary"]["observation_count"] = len(observations)
    trace["summary"] = trace["run"]["summary"]
    return AgentExecutionRun(
        task=task,
        plan=plan,
        trace=trace,
        messages=messages,
        tool_calls=tool_calls,
        observations=observations,
        final_text=final_text,
    )


def build_agent_execution_trace(
    task: str,
    config: AgentPlanConfig,
    *,
    dry_run: bool,
    safe_mode: bool,
) -> dict:
    params = {
        "repo_path": str(config.repo_path),
        "top_k": config.top_k,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "max_context_chars": config.max_context_chars,
        "dry_run": dry_run,
        "safe_mode": safe_mode,
        "agent_policy_path": str(config.agent_policy_path) if config.agent_policy_path else None,
    }
    summary = {"tool_call_count": 0, "observation_count": 0}
    return {
        "trace_version": "1.0",
        "run": build_trace_run(
            mode="agent_exec",
            task=task,
            status="dry_run" if dry_run or safe_mode else "success",
            repo_path=config.repo_path,
            params=params,
            flags={"llm": True, "dry_run": dry_run, "safe_mode": safe_mode},
            summary=summary,
        ),
        "events": [],
        "artifacts": {"retrieval": {}, "agent": {}, "tools": []},
        "query": task,
        "summary": summary,
        "params": params,
    }


def build_agent_execution_messages(task: str, plan: TaskPlan, tools: list[AgentToolSpec]) -> list[dict[str, str]]:
    system_prompt = """你是 RepoPilot 的 ReAct 执行 Agent。
你每次只能输出一个 JSON 对象，不要输出 markdown。
JSON 字段固定为 thought, tool, arguments, expected_observation。
tool 只能是 read_file, search_code, apply_patch, run_command, inspect_diff, finish。
默认处于 dry-run，apply_patch 不会落盘，run_command 不会真实执行。"""
    user_prompt = f"""任务: {task}

分类:
- mode: {plan.mode.value}
- intent: {plan.intent}
- reason: {plan.reason}

可用工具:
{format_tool_specs(tools)}

输出示例:
{{"thought":"先定位相关代码","tool":"search_code","arguments":{{"query":"{task}","top_k":5}},"expected_observation":"找到候选文件"}}
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def parse_agent_tool_call(raw_text: str) -> AgentToolCall:
    payload = parse_json_object(raw_text)
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return AgentToolCall(
        thought=str(payload.get("thought") or ""),
        tool=str(payload.get("tool") or ""),
        arguments=arguments,
        expected_observation=str(payload.get("expected_observation") or ""),
        raw_text=raw_text,
    )


def parse_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass
    inline = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if inline:
        try:
            payload = json.loads(inline.group(1))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def tool_result_payload(tool: str, result) -> dict:
    if isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"returncode": 0, "result": result}
    if "returncode" not in payload:
        payload["returncode"] = 0
    if tool in {"read_file", "search_code", "inspect_diff"} and "safety" not in payload:
        payload["safety"] = {"allowed": True, "matched_rule": "readonly"}
    return payload




def default_agent_tools() -> list[AgentToolSpec]:
    return [
        AgentToolSpec(
            name="search_code",
            purpose="按查询检索相关代码块，返回文件、行号、分数和文本片段。",
            input_schema={"query": "str", "top_k": "int"},
        ),
        AgentToolSpec(
            name="read_file",
            purpose="读取指定文件的完整内容或指定行号范围。",
            input_schema={"path": "str", "start_line": "int | None", "end_line": "int | None"},
        ),
        AgentToolSpec(
            name="list_files",
            purpose="列出仓库内匹配模式的文件，用于快速理解项目结构。",
            input_schema={"pattern": "str", "limit": "int"},
        ),
        AgentToolSpec(
            name="inspect_symbol",
            purpose="查看函数、类或模块符号的定义位置和邻近上下文。",
            input_schema={"symbol": "str"},
        ),
        AgentToolSpec(
            name="apply_patch",
            purpose="应用统一 diff 补丁；执行前统一安全检查，dry-run/safe-mode 不会落盘，批量删除文件会被拒绝。",
            input_schema={"diff": "str"},
        ),
        AgentToolSpec(
            name="run_command",
            purpose="在仓库根目录执行 allowlist 内的命令，统一安全检查后返回 stdout、stderr、退出码和截断信息。",
            input_schema={"cmd": "str", "timeout_seconds": "int | None"},
        ),
        AgentToolSpec(
            name="inspect_diff",
            purpose="查看当前工作区未提交 diff，用于补丁后自查。",
            input_schema={},
        ),
        AgentToolSpec(
            name="run_checks",
            purpose="运行测试、编译或评测命令并返回结果摘要。",
            input_schema={"command": "str"},
        ),
        AgentToolSpec(
            name="propose_patch",
            purpose="生成候选补丁草案；当前只允许计划，不自动应用。",
            input_schema={"files": "list[str]", "change_summary": "str"},
        ),
    ]


def build_agent_plan_messages(
    task: str,
    plan: TaskPlan,
    context: str,
    tools: list[AgentToolSpec],
    observations: list[dict] | None = None,
) -> list[dict[str, str]]:
    system_prompt = """你是 RepoPilot 的 ReAct 规划 Agent。
你的职责是先给出可审查的工作流计划，不执行工具、不编造结果、不直接修改文件。

规则：
1. 严格基于用户任务、已检索上下文和工具接口来规划。
2. 输出一个 ReAct 风格计划，包含 Thought / Action / Observation expectation。
3. Action 只能引用给定工具名；当前只是计划工具调用，不要声称已经执行。
4. 每一步必须说明目的、输入、预期观察结果和失败时的备选动作。
5. 必须给出最终交付物、验证命令和风险点。
"""
    user_prompt = f"""## 用户任务
{task}

## 初始分类计划
- mode: {plan.mode.value}
- intent: {plan.intent}
- reason: {plan.reason}
- initial_steps: {"; ".join(plan.steps)}

## 可用工具接口（只允许计划调用，不会实际执行）
{format_tool_specs(tools)}

## 初始检索上下文
{context}

## Read-only observations
{format_observations(observations or [])}

请输出一个可审查的 ReAct 工作流计划，格式如下：

## 任务分析
- 目标:
- 约束:
- 需要确认的未知点:

## ReAct 工作流计划
1. Thought:
   Action:
   Action Input:
   Expected Observation:
   Fallback:

## 候选文件
- path: reason

## 验证计划
- command: reason

## 风险与交付物
- risks:
- deliverables:
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def format_tool_specs(tools: list[AgentToolSpec]) -> str:
    lines: list[str] = []
    for tool in tools:
        schema = ", ".join(f"{key}: {value}" for key, value in tool.input_schema.items())
        lines.append(f"- {tool.name}: {tool.purpose} input={{ {schema} }}")
    return "\n".join(lines)


def render_agent_plan_run(run: AgentPlanRun) -> str:
    lines = [
        "## Agent 计划模式",
        f"- task: {run.task}",
        f"- seed_count: {len(run.seed_results)}",
        f"- recalled_count: {len(run.recalled_results)}",
        f"- final_count: {len(run.final_results)}",
        "",
        render_task_plan(run.plan),
        "",
        "## 工具接口（未执行）",
        format_tool_specs(run.tools),
        "",
        "## Read-only observations",
        format_observations(run.observations),
        "",
        "## LLM 计划",
        run.plan_text or "未启用 LLM，仅完成任务分析、上下文检索和计划 prompt 组装。",
    ]
    return "\n".join(lines)


def render_agent_plan_prompt(messages: list[dict[str, str]]) -> str:
    blocks = ["## Agent Plan Prompt"]
    for message in messages:
        blocks.append(f"### {message['role']}\n{message['content']}")
    return "\n\n".join(blocks)


def render_task_plan(plan: TaskPlan) -> str:
    lines = [
        "## 任务分类",
        f"- mode: {plan.mode.value}",
        f"- intent: {plan.intent}",
        f"- generation_mode: {plan.generation_mode.value}",
        f"- reason: {plan.reason}",
        "- steps:",
    ]
    lines.extend(f"  {index}. {step}" for index, step in enumerate(plan.steps, start=1))
    return "\n".join(lines)


def render_ask_prompt(messages: list[dict[str, str]]) -> str:
    blocks = ["## ASK Prompt"]
    for message in messages:
        blocks.append(f"### {message['role']}\n{message['content']}")
    return "\n\n".join(blocks)
