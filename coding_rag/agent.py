"""RepoPilot 代码 Agent 工作流。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coding_rag.agent_memory import AgentMemory, AgentMemoryStore, build_memory
from coding_rag.bm25_retriever import BM25Retriever, SearchResult, is_test_path
from coding_rag.code_splitter import split_python_files
from coding_rag.context_recaller import expand_with_neighbor_chunks
from coding_rag.file_loader import load_python_files
from coding_rag.result_filter import filter_recalled_results
from rag.prompt import format_results_as_context
from rag.trace import build_retrieval_trace, render_trace_report


AGENT_WORKFLOW: list[tuple[str, str]] = [
    ("receive_task", "接收用户任务，抽取目标、约束和可能的交付物。"),
    ("load_memory", "读取历史任务记忆，找出相关决策、踩坑和常用文件。"),
    ("retrieve_context", "用 Hybrid Search 检索相关代码，并召回邻近上下文。"),
    ("plan_changes", "基于上下文生成修改计划，明确候选文件、步骤和风险。"),
    ("implement", "按计划生成实现草案；启用 LLM 时产出更具体的改动方案。"),
    ("verify", "给出需要运行的测试、评测和人工检查点。"),
    ("remember", "把任务摘要、相关文件和关键决策写入长期记忆。"),
]


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class CodeAgentConfig:
    repo_path: str = "."
    top_k: int = 5
    candidate_k: int | None = None
    chunk_size: int = 40
    overlap: int = 5
    recall_window: int = 2
    max_recall_results: int | None = None
    final_k: int | None = None
    min_final_score: float | None = None
    memory_path: str | Path = "artifacts/agent_memory.jsonl"
    memory_limit: int = 5
    max_context_chars: int = 12000


@dataclass(frozen=True)
class TaskProfile:
    kind: str
    confidence: float
    reasons: list[str]
    recommended_checks: list[str]


@dataclass(frozen=True)
class AgentRun:
    task: str
    workflow: list[dict[str, str]]
    task_profile: TaskProfile
    agent_trace: list[dict]
    memories: list[AgentMemory]
    seed_results: list[SearchResult]
    recalled_results: list[SearchResult]
    final_results: list[SearchResult]
    trace: dict
    plan: str
    implementation: str
    memory: AgentMemory
    memory_written: bool


def run_code_agent(
    task: str,
    config: CodeAgentConfig,
    client: ChatClient | None = None,
) -> AgentRun:
    """执行一次轻量代码 Agent 工作流。"""
    memory_store = AgentMemoryStore(config.memory_path)
    task_profile = analyze_task(task)
    memories = memory_store.search(task, limit=config.memory_limit)
    seed_results, recalled_results, final_results, trace = retrieve_agent_context(task, config)
    task_profile = enrich_task_profile(task_profile, final_results)
    plan = build_agent_plan(task, final_results, memories, task_profile)
    implementation = (
        generate_agent_implementation(task, final_results, memories, task_profile, client, config.max_context_chars)
        if client
        else build_offline_implementation_note()
    )
    memory = build_memory(
        task=task,
        status="planned" if client is None else "drafted",
        summary=build_memory_summary(task, final_results),
        files=prioritized_result_files(final_results),
        decisions=build_memory_decisions(config, client is not None, task_profile),
    )
    memory_written = memory_store.append_if_new(memory)
    agent_trace = build_agent_step_trace(
        task=task,
        task_profile=task_profile,
        config=config,
        memories=memories,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        plan=plan,
        implementation=implementation,
        memory=memory,
        memory_written=memory_written,
        used_llm=client is not None,
    )

    return AgentRun(
        task=task,
        workflow=workflow_as_dicts(),
        task_profile=task_profile,
        agent_trace=agent_trace,
        memories=memories,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        trace=trace,
        plan=plan,
        implementation=implementation,
        memory=memory,
        memory_written=memory_written,
    )


def retrieve_agent_context(
    task: str,
    config: CodeAgentConfig,
) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], dict]:
    python_files = load_python_files(config.repo_path)
    chunks = split_python_files(python_files, chunk_size=config.chunk_size, overlap=config.overlap)
    if not chunks:
        raise ValueError("未找到 Python 代码块，请检查 repo_path 和文件内容")

    retriever = BM25Retriever(chunks)
    candidate_k = config.candidate_k or config.top_k
    seed_results = retriever.search(task, top_k=candidate_k)
    max_recall_results = config.max_recall_results
    if max_recall_results is None:
        max_recall_results = max(20, candidate_k * ((2 * max(config.recall_window, 0)) + 1))
    elif max_recall_results <= 0:
        max_recall_results = None

    recalled_results = expand_with_neighbor_chunks(
        chunks=chunks,
        seed_results=seed_results,
        window=config.recall_window,
        max_results=max_recall_results,
    )
    final_k = config.final_k if config.final_k is not None else config.top_k
    final_results = filter_recalled_results(
        query=task,
        recalled_results=recalled_results,
        retriever=retriever,
        final_k=None if final_k == 0 else final_k,
        min_score=config.min_final_score,
    )
    trace = build_retrieval_trace(
        query=task,
        seed_results=seed_results,
        recalled_results=recalled_results,
        final_results=final_results,
        params={
            "repo_path": config.repo_path,
            "top_k": config.top_k,
            "candidate_k": candidate_k,
            "chunk_size": config.chunk_size,
            "overlap": config.overlap,
            "recall_window": config.recall_window,
            "max_recall_results": max_recall_results,
            "final_k": final_k,
            "min_final_score": config.min_final_score,
        },
    )
    return seed_results, recalled_results, final_results, trace


def build_agent_plan(
    task: str,
    results: list[SearchResult],
    memories: list[AgentMemory],
    task_profile: TaskProfile,
) -> str:
    files = prioritized_result_files(results)
    lines = [
        "## Agent 工作流设计",
        *[f"{index}. {name}: {description}" for index, (name, description) in enumerate(AGENT_WORKFLOW, start=1)],
        "",
        "## 本次任务计划",
        f"- 任务: {task}",
        f"- 任务类型: {task_profile.kind} (confidence={task_profile.confidence:.2f})",
        f"- 判断依据: {', '.join(task_profile.reasons) if task_profile.reasons else '无明显关键词'}",
        "- 修改入口: 先从候选文件中定位最小改动面，再补测试和文档。",
        "- 候选文件:",
    ]
    lines.extend(f"  - {path}" for path in files[:8])
    if memories:
        lines.extend(["", "## 相关记忆"])
        lines.extend(f"- {memory.created_at}: {memory.summary}" for memory in memories)
    lines.extend(
        [
            "",
            "## 执行步骤",
            "1. 阅读候选文件，确认任务影响范围和已有模式。",
            "2. 在最小必要文件内实现功能，避免无关重构。",
            "3. 为新增行为补单元测试或评测样例。",
            "4. 运行相关测试、编译检查和必要的检索评测。",
            "5. 将结果、风险和后续优化点写入记忆。",
            "",
            "## 建议验证命令",
        ]
    )
    lines.extend(f"- `{command}`" for command in task_profile.recommended_checks)
    return "\n".join(lines)


def generate_agent_implementation(
    task: str,
    results: list[SearchResult],
    memories: list[AgentMemory],
    task_profile: TaskProfile,
    client: ChatClient,
    max_context_chars: int,
) -> str:
    messages = build_agent_messages(task, results, memories, task_profile, max_context_chars)
    return client.complete(messages).strip()


def build_agent_messages(
    task: str,
    results: list[SearchResult],
    memories: list[AgentMemory],
    task_profile: TaskProfile,
    max_context_chars: int,
) -> list[dict[str, str]]:
    memory_text = format_memories(memories)
    context = format_results_as_context(results, max_context_chars=max_context_chars)
    system_prompt = """你是 RepoPilot 的代码 Agent。
你的目标是从用户任务出发，自主完成代码功能设计：理解任务、定位代码、制定修改计划、给出实现草案和验证方案。
要求：
1. 严格基于提供的代码上下文和记忆，不要编造文件或 API。
2. 优先最小改动，遵循现有项目结构和风格。
3. 输出必须包含：任务理解、候选文件、实施计划、代码修改建议、测试计划、风险。
4. 如果证据不足，明确说明还需要读取哪些文件。"""
    user_prompt = f"""## 用户任务
{task}

## 任务画像
- kind: {task_profile.kind}
- confidence: {task_profile.confidence:.2f}
- reasons: {", ".join(task_profile.reasons)}
- recommended_checks: {", ".join(task_profile.recommended_checks)}

## 历史记忆
{memory_text}

## 检索上下文
{context}

请给出可执行的代码 Agent 实施草案。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_agent_run(run: AgentRun, *, show_trace: bool = True) -> str:
    lines = [
        "RepoPilot Agent Run",
        f"Task: {run.task}",
        f"Task Profile: {run.task_profile.kind} (confidence={run.task_profile.confidence:.2f})",
        "",
        render_agent_trace(run.agent_trace),
        "",
        run.plan,
        "",
        "## 实现草案",
        run.implementation,
        "",
        "## 本次记忆",
        f"- status: {run.memory.status}",
        f"- written: {run.memory_written}",
        f"- summary: {run.memory.summary}",
        f"- files: {', '.join(run.memory.files) if run.memory.files else '(none)'}",
    ]
    if show_trace:
        lines.extend(["", render_trace_report(run.trace, limit=8)])
    return "\n".join(lines)


def agent_run_to_dict(run: AgentRun) -> dict:
    return {
        "task": run.task,
        "workflow": run.workflow,
        "task_profile": run.task_profile.__dict__,
        "agent_trace": run.agent_trace,
        "memories": [memory.__dict__ for memory in run.memories],
        "trace": run.trace,
        "plan": run.plan,
        "implementation": run.implementation,
        "memory": run.memory.__dict__,
        "memory_written": run.memory_written,
    }


def workflow_as_dicts() -> list[dict[str, str]]:
    return [{"name": name, "description": description} for name, description in AGENT_WORKFLOW]


def build_agent_step_trace(
    task: str,
    task_profile: TaskProfile,
    config: CodeAgentConfig,
    memories: list[AgentMemory],
    seed_results: list[SearchResult],
    recalled_results: list[SearchResult],
    final_results: list[SearchResult],
    plan: str,
    implementation: str,
    memory: AgentMemory,
    memory_written: bool,
    used_llm: bool,
) -> list[dict]:
    files = prioritized_result_files(final_results)
    return [
        {
            "step": "receive_task",
            "status": "done",
            "detail": "已接收用户任务，并作为 Agent 执行目标。",
            "artifacts": {
                "task": task,
                "task_kind": task_profile.kind,
                "confidence": task_profile.confidence,
                "reasons": task_profile.reasons,
            },
        },
        {
            "step": "load_memory",
            "status": "done",
            "detail": f"读取相关历史记忆 {len(memories)} 条。",
            "artifacts": {
                "memory_path": str(config.memory_path),
                "memory_count": len(memories),
                "memory_ids": [memory.id for memory in memories],
            },
        },
        {
            "step": "retrieve_context",
            "status": "done",
            "detail": "已完成 Hybrid Search、邻近召回和最终过滤。",
            "artifacts": {
                "seed_count": len(seed_results),
                "recalled_count": len(recalled_results),
                "final_count": len(final_results),
                "candidate_files": files,
            },
        },
        {
            "step": "plan_changes",
            "status": "done",
            "detail": "已生成候选文件、执行步骤和验证建议。",
            "artifacts": {
                "plan_chars": len(plan),
                "candidate_file_count": len(files),
            },
        },
        {
            "step": "implement",
            "status": "drafted" if used_llm else "planned",
            "detail": "已调用 LLM 生成实现草案。" if used_llm else "未启用 LLM，输出离线实施说明。",
            "artifacts": {
                "used_llm": used_llm,
                "implementation_chars": len(implementation),
            },
        },
        {
            "step": "verify",
            "status": "planned",
            "detail": "已给出测试、编译和评测检查方向。",
            "artifacts": {
                "recommended_checks": task_profile.recommended_checks,
            },
        },
        {
            "step": "remember",
            "status": "done" if memory_written else "skipped",
            "detail": "已把本次任务摘要、相关文件和关键决策写入长期记忆。" if memory_written else "检测到相同任务和文件集合，跳过重复记忆写入。",
            "artifacts": {
                "memory_id": memory.id,
                "memory_path": str(config.memory_path),
                "memory_written": memory_written,
                "summary": memory.summary,
            },
        },
    ]


def render_agent_trace(agent_trace: list[dict]) -> str:
    lines = ["## Agent 运行轨迹"]
    for index, item in enumerate(agent_trace, start=1):
        lines.append(f"{index}. {item['step']} [{item['status']}]: {item['detail']}")
        artifact_summary = summarize_trace_artifacts(item.get("artifacts", {}))
        if artifact_summary:
            lines.append(f"   - {artifact_summary}")
    return "\n".join(lines)


def summarize_trace_artifacts(artifacts: dict) -> str:
    parts: list[str] = []
    for key in (
        "memory_count",
        "task_kind",
        "seed_count",
        "recalled_count",
        "final_count",
        "candidate_file_count",
        "used_llm",
        "memory_written",
        "memory_id",
    ):
        if key in artifacts:
            parts.append(f"{key}={artifacts[key]}")
    candidate_files = artifacts.get("candidate_files") or []
    if candidate_files:
        parts.append("files=" + ", ".join(candidate_files[:3]))
    return "; ".join(parts)


def build_memory_summary(task: str, results: list[SearchResult]) -> str:
    files = prioritized_result_files(results)
    file_text = ", ".join(files[:3]) if files else "no files"
    return f"任务“{task}”定位到 {file_text}"


def prioritized_result_files(results: list[SearchResult]) -> list[str]:
    files = dedupe([str(result.chunk.file_path) for result in results])
    return sorted(files, key=lambda path: (is_test_path(path), files.index(path)))


def build_memory_decisions(config: CodeAgentConfig, used_llm: bool, task_profile: TaskProfile) -> list[str]:
    return [
        f"任务类型: {task_profile.kind}, confidence={task_profile.confidence:.2f}",
        f"使用 Hybrid Search top_k={config.top_k}, recall_window={config.recall_window}",
        "使用上下文压缩后的证据块生成 Agent 计划",
        "启用 LLM 生成实现草案" if used_llm else "未启用 LLM，仅生成离线工作流和候选文件",
    ]


def build_offline_implementation_note() -> str:
    return (
        "未启用 LLM，因此本次 Agent 不生成具体代码补丁。"
        "请使用上方计划和候选文件继续实现，或追加 `--llm` 让 Agent 生成更详细的实现草案。"
    )


def format_memories(memories: list[AgentMemory]) -> str:
    if not memories:
        return "无相关历史记忆。"
    return "\n".join(
        f"- {memory.created_at} [{memory.status}] {memory.summary}; files={', '.join(memory.files)}"
        for memory in memories
    )


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def analyze_task(task: str) -> TaskProfile:
    q = task.casefold()
    rules = [
        ("bugfix", ["修复", "报错", "错误", "异常", "失败", "bug", "bad case", "badcase"]),
        ("test", ["测试", "单测", "unittest", "pytest", "断言"]),
        ("evaluation", ["评测", "trace", "轨迹", "指标", "mrr", "recall"]),
        ("docs", ["文档", "readme", "agent.md", "changelog", "说明"]),
        ("feature", ["增加", "新增", "实现", "支持", "功能", "添加"]),
        ("optimization", ["优化", "改进", "提升", "重构", "清理"]),
    ]
    matches: list[tuple[str, list[str]]] = []
    for kind, keywords in rules:
        hit = [keyword for keyword in keywords if keyword in q]
        if hit:
            matches.append((kind, hit))

    if not matches:
        return TaskProfile(
            kind="general",
            confidence=0.2,
            reasons=[],
            recommended_checks=build_recommended_checks("general", []),
        )

    kind, hit = matches[0]
    confidence = min(0.95, 0.45 + 0.15 * len(hit))
    return TaskProfile(
        kind=kind,
        confidence=confidence,
        reasons=hit,
        recommended_checks=build_recommended_checks(kind, []),
    )


def enrich_task_profile(task_profile: TaskProfile, results: list[SearchResult]) -> TaskProfile:
    files = prioritized_result_files(results)
    return TaskProfile(
        kind=task_profile.kind,
        confidence=task_profile.confidence,
        reasons=task_profile.reasons,
        recommended_checks=build_recommended_checks(task_profile.kind, files),
    )


def build_recommended_checks(kind: str, files: list[str]) -> list[str]:
    checks: list[str] = []
    test_modules = [test_module_name(path) for path in files if is_test_path(path)]
    for module in test_modules:
        checks.append(f"python -m unittest {module}")

    if kind in {"feature", "bugfix", "optimization", "general"}:
        checks.append("python -m unittest")
        checks.append("python -m compileall coding_rag rag scripts main.py web_ui.py tests")
    elif kind == "test":
        checks.append("python -m unittest")
    elif kind == "evaluation":
        checks.append("python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --top-k 5")
        checks.append("python -m unittest tests.test_retrieval_eval")
    elif kind == "docs":
        checks.append("python -m compileall coding_rag rag scripts main.py web_ui.py tests")

    return dedupe(checks)


def test_module_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [part for part in normalized.split("/") if part]
    if "tests" in parts:
        parts = parts[parts.index("tests") :]
    return ".".join(parts)
