# RepoPilot Function Map

本文件记录当前源码中关键文件、类和函数的职责，便于定位 ASK、Agent、检索、LLM、评测和前端流程。

## 顶层入口

### `main.py`

CLI 统一入口，所有问题和修改需求都交给 LLM 决定直接回答或调用工具。

- `parse_args()`: 定义检索、回答风格、LLM、policy、trace 和 `--rollback` 参数；不再包含 ASK/Agent 模式参数或总执行授权。
- `main()`: 加载已有 `.env`，决定进入交互模式还是单次统一运行。
- `run_single_turn(args, client)`: 构建统一配置并调用 `run_unified_query()`。
- `run_interactive_cli(args)`: 复用一个 LLM client 连续执行统一工作流。
- `build_chat_client(args)`: 根据 provider 和环境变量创建 OpenAI 兼容 client。
- `build_runtime_config(args)`: 将 CLI 参数转换为 `UnifiedRunConfig`。
- `render_unified_run(run)`: 渲染状态、执行授权、调用统计和最终回答。
- `prompt_with_default(label, default)`: 读取带默认值的交互输入。
- `configure_utf8_stdio()`: 在 Windows 终端和重定向日志中保持中文输出为 UTF-8。
- `print_llm_setup_hint(provider, file)`: 输出 LLM 环境变量配置提示。

### `frontend.py`

轻量浏览器前端，同时提供普通 Agent 的 `/api/run` 与只读教学会话的 `/api/learning/session`；两条路线保持独立输入、结果和权限边界。

- `parse_args()`: 解析监听地址、端口和 `--allow-tool-execution`。
- `RequestBodyTooLarge`: 表示请求正文在读取前已经超过 1 MiB HTTP 上限。
- `UnsupportedMediaType`: 表示请求未使用 `application/json`，用于阻止跨站 simple request 直接驱动本地 API。
- `PendingApproval`、`PendingApprovalStore`: 仅在服务端进程内保存可恢复运行状态；浏览器只拿到一次性不透明 ID，消费后立即移除。
- `RepoPilotServer`: 保存服务级写能力、一次性批准状态和进程内随机教学 session 签名密钥；密钥不进入公开响应。
- `RepoPilotHandler.do_GET()`: 返回首页 HTML 或 404。
- `RepoPilotHandler.do_POST()`: 分流 `/api/run`、`/api/rollback` 与 `/api/learning/session`；请求过大返回 413，教学业务失败仍按稳定结果返回 HTTP 200。
- `RepoPilotHandler.server_allows_execution()`: 读取服务级执行授权。
- `RepoPilotHandler.server_learning_signing_key()`: 读取至少 32 字节的进程内教学签名密钥。
- `RepoPilotHandler.read_json()`: 只接受 `application/json`，再在读取前限制 Content-Length，并解析 UTF-8 JSON object。
- `RepoPilotHandler.send_html()`: 返回 HTML 页面。
- `RepoPilotHandler.send_json()`: 返回 JSON API 响应。
- `RepoPilotHandler.log_message()`: 静默 HTTP server 默认日志。
- `run_frontend_query(payload, server_allows_execution, approval_store)`: 开始普通查询或消费一个服务端批准 ID；不接受总执行授权字段。
- `prepare_frontend_run_response(...)`: 序列化运行结果，并仅在服务允许写入时保存待批准状态。
- `run_frontend_rollback(payload, server_allows_execution)`: 按仓库和快照 ID 恢复一次已批准写操作。
- `run_learning_frontend_request(payload, signing_key)`: 把浏览器外层仓库/LLM 配置适配为现有严格 `LearningSessionRequest`，调用只读 `run_learning_session()` 并返回 JSON-safe 结果；明确拒绝 `execute_tools`。
- `frontend_integer(payload, name, default)`: 读取浏览器整数配置，拒绝 bool 和不可解析值。
- `empty_to_none(value)`: 将表单空值规范化为 `None`。
- `configure_utf8_stdio()`: 将前端服务帮助文本输出配置为 UTF-8。
- `build_index_html(allow_tool_execution)`: 将服务执行能力注入前端页面。
- `main()`: 启动本地 HTTP server。

### `web_ui.py`

Tkinter 桌面 UI，覆盖检索、LLM 回答、评测、bad case 查看和参数优化。

- `RepoPilotUI.__init__(root)`: 初始化状态、变量和布局。
- `RepoPilotUI._build_layout()`: 构建 Tkinter 控件。
- `RepoPilotUI._pick_repo_path()`: 选择仓库目录。
- `RepoPilotUI._run_async()`: 异步启动单次检索。
- `RepoPilotUI._run_pipeline()`: 执行检索、邻近召回、过滤、trace 和可选 LLM 回答。
- `RepoPilotUI._pick_evalset()`: 选择评测集文件。
- `RepoPilotUI._run_eval_async()`: 异步启动评测。
- `RepoPilotUI._run_eval()`: 执行评测集，生成 per-case 记录、stage 诊断和 trace 文件。
- `RepoPilotUI._export_last_trace()`: 导出最近一次评测 trace。
- `RepoPilotUI._show_bad_cases()`: 展示最终未命中的 bad case。
- `RepoPilotUI._optimize_async()`: 异步启动参数优化。
- `RepoPilotUI._run_optimize()`: 调用评测优化逻辑并展示报告。
- `RepoPilotUI._build_llm_args()`: 组装评测/回答用 LLM 参数。
- `RepoPilotUI._render_eval_output()`: 渲染评测指标、诊断和 LLM 输出。
- `RepoPilotUI._render_results()`: 渲染单次检索结果。
- `RepoPilotUI._set_output()`: 安全更新输出框。
- `RepoPilotUI._replace_output()`: 替换输出框内容。
- `RepoPilotUI._set_status()`: 更新状态栏。
- `main()`: 启动 Tkinter 应用。

## Agent 模块

### `coding_rag/agent/runtime.py`

CLI 和浏览器共用的统一结果模型、JSON 决策协议与辅助函数；公开入口委托给 LangGraph。

- `UnifiedRunConfig`: 统一运行配置，包括 Hybrid Search、回答风格、上下文限制、policy 和最大工具步数。
- `ModelDecision`: 模型结构化决策，字段为 `action/reason/answer/tool/arguments/expected_observation`。
- `UnifiedRun`: 最终回答、状态、决策、observation、trace 和执行授权。
- `run_unified_query(query, config, client, safety_policy)`: 普通 Agent 的稳定公开入口，延迟导入并委托 `run_unified_graph_query()`。
- `resume_unified_query(pending_run, config, client, approval_fingerprint, safety_policy)`: 校验待批准状态和精确指纹后恢复一次写调用。
- `build_hybrid_search_provider(config)`: 把 ASK 的候选检索、邻居召回和最终过滤包装为 executor 搜索 provider。
- `serialize_search_results(results, repo_path)`: 将最终检索结果转为带相对路径和行号的工具输出。
- `build_unified_messages(query, config)`: 生成 JSON 决策协议、工具 schema、逐调用批准规则和不可信仓库数据约束。
- `parse_model_decision(raw_text)`: 校验模型 JSON、action、工具名和参数类型。
- `build_unified_trace(...)`: 创建 `mode=unified` 的 run/events/artifacts 外壳。
- `compact_tool_result(tool, result, max_chars)`: 截断发送回 LLM 和写入 trace 的 observation。
- `collect_observed_ranges(tool, result, observed_ranges)`: 收集搜索和文件读取证据范围供引用校验。
- `finish_unified_run(...)`: 计算摘要并完成统一运行结果。
- `unified_run_to_dict(run)`: 生成 CLI/HTTP 共用的响应结构。

### `coding_rag/agent/langchain_tools.py`

同时提供 Learning Mode 只读工具表与普通 Agent 五工具表的 LangChain `StructuredTool` 适配层。

- `READONLY_LANGCHAIN_TOOL_NAMES`: 稳定工具顺序，仅包含 `search_code`、`read_file`。
- `AGENT_LANGCHAIN_TOOL_NAMES`: 普通 Agent 的稳定五工具顺序，增加 `apply_patch`、`run_command`、`inspect_diff`。
- `_ToolExecutor`: 约束两套适配器都只依赖 `call(name, arguments)` 窄入口的协议。
- `ReadOnlyToolInput`: 开启严格类型、去除字符串首尾空白，并拒绝额外字段的参数基类。
- `SearchCodeInput`: 要求非空查询，`top_k` 默认 5、范围 1–20。
- `ReadFileInput`: 要求非空仓库相对路径，并约束正行号和起止顺序。
- `ReadFileInput.validate_line_range()`: 在访问执行器前拒绝倒置行范围。
- `ApplyPatchInput`、`RunCommandInput`、`InspectDiffInput`: 写工具与 diff 查看工具的严格参数模型；补丁模型不裁剪末尾换行。
- `build_readonly_langchain_tool_map(executor)`: 构造两个严格 schema 的工具，并把校验后参数硬编码委托给 RepoPilot `AgentExecutor.call()`。
- `build_readonly_langchain_tools(executor)`: 按稳定顺序返回模型可见的两个只读工具。
- `build_agent_langchain_tool_map(executor)`、`build_agent_langchain_tools(executor)`: 构造并按稳定顺序返回普通 Agent 的五工具集合；真实执行权限仍由 executor 决定。

### `coding_rag/agent/graph_runtime.py`

同时承载普通 Agent 主图与 Learning Mode 可复用的只读图；状态只保存 JSON 可序列化数据，模型、工具和执行器由图工厂闭包注入。

- `READONLY_SYSTEM_PROMPT`: 限制模型只能直接回答或使用两个只读工具，并把仓库内容标为不可信数据。
- `READONLY_TOOL_NAMES`: 图路由允许的工具名集合。
- `BoundToolCallingModel`、`ToolCallingChatModel`: 描述工具绑定和同步调用能力的模型协议。
- `GraphState`: 保存序列化消息、回答/状态、调用计数、观察、引用范围、trace、图步骤和终止错误。
- `build_graph_input(query, config)`: 构造字段完整、可序列化的初始图状态。
- `UnifiedGraphState`、`build_unified_graph_input(...)`: 保存普通 Agent 的 JSON 决策、批准状态、写执行状态和五工具观察。
- `build_unified_graph(config, client, executor, resume_approved_write)`: 构建普通 Agent 的 `model/tool/request_approval/finalize/safe_fallback` 主图，以条件边替代原手写循环。
- `run_unified_graph_query(...)`: 加载安全 policy、构造执行器、运行或恢复主图，并生成兼容 `UnifiedRun` 的结果和 `runtime=langgraph` trace 标记。
- `build_readonly_graph(config, model, executor, safety_policy)`: 构建 `model/tool/finalize/safe_fallback` 四节点图；条件边拒绝未知、多重和超限调用，工具节点只回到模型节点。
- `run_graph_query(query, config, model, executor, safety_policy)`: 有界运行只读图，复用现有 `UnifiedRun` 与公开序列化结构。
- `build_readonly_executor(config, safety_policy)`: 复用 Hybrid Search，构造 `safe_mode` 只读 RepoPilot 执行器。
- `validate_ai_message(message, response_text)`: 把无效工具调用和空回答转为结构化模型错误。
- `ai_message_text(message)`: 提取字符串或文本块形式的模型回答。
- `ai_message_to_decision_payload(message, response_text)`: 将 AI 消息映射为兼容 `ModelDecision` 的可序列化字段。
- `last_ai_message(messages)`: 从序列化消息中找到最近的 AI 消息。
- `build_tool_message(...)`: 构造带调用 ID、成败状态和不可信数据提示的工具消息。
- `serialize_message(message)`、`json_safe(value)`: 在数据写入 GraphState 前统一转换为 JSON 安全值。
- `fallback_error_for_state(state)`、`terminal_error(...)`: 生成未知工具、多工具调用、缺少 ID、步数上限等错误结构。
- `fallback_status(state, error)`、`fallback_answer(error)`、`fallback_event_step(error)`: 将错误和既有观察收敛为稳定终态、回答与事件名。
- `build_graph_trace(query, config, state)`: 复用统一 trace schema，汇总模型决策、工具观察、检索子 trace 和图步骤。
- `messages_for_unified_run(messages)`、`message_role(message)`: 将 LangChain 消息转回现有结果结构的角色/文本列表。

### `coding_rag/agent/planner.py`

ASK/Agent 分类和 ReAct 计划模式核心。

- `WorkflowMode`: 用户任务模式枚举：`auto`、`ask`、`agent`。
- `TaskPlan`: 分类结果，包括模式、意图、生成模式、步骤和原因。
- `AgentToolSpec`: Agent 工具接口声明。
- `AgentPlanConfig`: Agent 计划/执行配置，包括检索参数、只读工具开关和可选 `agent_policy_path`。
- `AgentPlanRun`: Agent 计划运行产物。
- `ReActAgentInterface.plan(task)`: 强制把任务规划为 Agent 模式。
- `ReActAgentInterface.build_plan(task, config, client)`: 构建 Agent 计划运行结果。
- `classify_task(user_input, requested_mode, generation_mode)`: 按显式模式或关键词推断 ASK/Agent。
- `infer_mode(user_input)`: 使用 fix/implement/refactor 等关键词推断模式。
- `run_ask_mode(query, config, client, requested_mode)`: ASK 核心流程；检索、构建 prompt、可选调用 LLM、校验引用。
- `run_agent_plan_mode(task, config, client, tools)`: Agent 计划流程；默认不检索，开启只读工具时收集观察。
- `build_agent_plan_trace(task, config)`: 构造 Agent 计划 trace 骨架。
- `collect_readonly_observations(task, config)`: 调用 `search_code`、`list_files`、`inspect_symbol` 收集只读观察。
- `default_agent_tools()`: 声明 Agent 可用工具，包括 `read_file`、`search_code`、`apply_patch`、`run_command`、`inspect_diff` 等。
- `build_agent_plan_messages(task, plan, context, tools, observations)`: 组装发给 LLM 的 Agent 计划 prompt。
- `format_tool_specs(tools)`: 将工具声明渲染为 prompt 文本。
- `render_agent_plan_run(run)`: 渲染 Agent 计划运行结果。
- `render_agent_plan_prompt(messages)`: 渲染 Agent 计划 prompt。
- `render_task_plan(plan)`: 渲染任务分类结果。
- `render_ask_prompt(messages)`: 渲染 ASK prompt。

### `coding_rag/agent/workflow.py`

代码 Agent 工作流：记忆、检索、计划、实现草案、日志和推荐验证命令。

- `CodeAgentConfig`: 代码 Agent 配置，包括检索、记忆、日志和 `safe_mode`；开启后跳过记忆与运行日志持久化。
- `TaskProfile`: 任务画像，包括类型、置信度、原因和推荐检查。
- `AgentRun`: 一次代码 Agent 运行的完整产物。
- `run_code_agent(task, config, client)`: 代码 Agent 主流程。
- `retrieve_agent_context(task, config)`: 加载文件、切 chunk、BM25 检索、邻近召回、最终过滤和 trace。
- `build_agent_plan(task, results, memories, task_profile)`: 生成离线执行计划。
- `generate_agent_implementation(...)`: 调用 LLM 生成交给执行者的实施草案。
- `build_agent_messages(...)`: 组装代码 Agent LLM prompt。
- `render_agent_run(run, show_trace)`: 渲染 Agent 运行结果。
- `agent_run_to_dict(run)`: 序列化 Agent 运行结果。
- `append_agent_run_log(path, run)`: 追加独立运行日志 JSONL。
- `workflow_as_dicts()`: 将固定 Agent 步骤转换为结构化列表。
- `build_agent_step_trace(...)`: 生成 receive/load/retrieve/plan/implement/verify/remember 轨迹。
- `render_agent_trace(agent_trace)`: 渲染 Agent 轨迹。
- `summarize_trace_artifacts(artifacts)`: 摘要化每步 artifacts。
- `build_memory_summary(task, results)`: 生成长期记忆摘要。
- `prioritized_result_files(results)`: 对候选文件去重并优先展示非测试文件。
- `build_memory_decisions(config, used_llm, task_profile)`: 生成记忆决策列表。
- `build_offline_implementation_note()`: 未启用 LLM 时的实施说明。
- `format_memories(memories)`: 渲染历史记忆。
- `dedupe(values)`: 保序去重。
- `analyze_task(task)`: 按关键词生成任务画像。
- `enrich_task_profile(task_profile, results)`: 用候选文件补充推荐检查命令。
- `build_recommended_checks(kind, files)`: 根据任务类型和测试文件生成检查命令。
- `test_module_name(path)`: 将测试文件路径转换为 unittest 模块名。

### `coding_rag/agent/executor.py`

Agent 工具执行器，封装可执行工具表面。

- `CommandResult`: 命令执行结果数据结构。
- `WriteApprovalRequired`: 缺少或不匹配精确一次性批准时返回完整批准申请。
- `AgentExecutor.__init__(repo_path, ..., safe_mode, safety_policy, search_provider, write_approval, snapshot_root)`: 绑定仓库根目录、只读工具、安全策略、一次性批准和外部快照库。
- `AgentExecutor.call(name, arguments)`: 工具分发入口。
- `AgentExecutor.read_file(path, start_line, end_line)`: 读取仓库内文件。
- `AgentExecutor.search_code(query, top_k)`: 优先调用注入的 Hybrid Search provider，否则复用基础只读检索。
- `AgentExecutor.preview_write(name, arguments)`: 校验写调用，列出目标文件并生成绑定完整参数的批准指纹。
- `AgentExecutor.apply_patch(diff)`: 安全检查和 `git apply --check` 后消费批准、创建快照、应用补丁并返回修改后 diff。
- `AgentExecutor.run_command(cmd, affected_files, timeout_seconds)`: 校验命令和声明文件，消费批准、创建快照后以 `shell=False` 执行。
- `AgentExecutor.rollback_snapshot(snapshot_id)`: 恢复声明文件的修改前版本。
- `AgentExecutor.inspect_diff()`: 查看当前 `git diff --no-ext-diff --`。
- `AgentExecutor._run_git(command, stdin)`: 执行 git 子命令。
- `run_subprocess(command, cwd, timeout, shell, stdin)`: 子进程执行封装。
- `command_result_to_dict(result, safety)`: 将命令结果转 JSON dict，并按安全策略截断 stdout/stderr。
- `ensure_allowed(decision)`: 将安全检查拒绝结果转换为工具错误。
- `decision_to_dict(decision)`: 序列化安全检查结果。
- `optional_int(value)`: 参数转可选整数。

### `coding_rag/agent/snapshots.py`

仓库外的一次写调用快照库，用于生成操作专属 diff 和恢复修改前版本。

- `SnapshotStore.create(paths, tool)`: 在执行前缓存每个声明文件，记录原本不存在的文件。
- `SnapshotStore.diff(snapshot_id)`: 将当前文件与该次快照比较，生成 unified diff 或二进制变化标记。
- `SnapshotStore.rollback(snapshot_id)`: 还原原文件并逐个移除本次新建文件。
- `normalize_snapshot_paths(paths)`: 规范化、去重并拒绝绝对路径、盘符和父目录穿越。
- `resolve_repo_file(repo_path, relative_path)`: 确保声明路径解析后仍位于仓库内。

### `coding_rag/agent/safety.py`

Agent 可执行工具安全策略层，统一约束命令执行和 patch 应用。

- `AgentSafetyPolicy`: 配置 safe-mode、命令超时、输出截断、patch 大小、allowlist 和 denylist。
- `SafetyDecision`: 表示安全检查结果，包括是否允许、命中规则、原因和元数据。
- `AgentSafetyGuard.check_command(cmd)`: 解析命令，拒绝 shell 拼接/重定向、denylist 和不在 allowlist 的命令。
- `AgentSafetyGuard.check_patch(diff)`: 检查 diff 是否为空、过大、二进制、路径越界、修改 `.env` 或批量删除文件。
- `AgentSafetyGuard.resolve_timeout(timeout_seconds)`: 解析命令超时并限制上限。
- `AgentSafetyGuard.truncate_output(text)`: 对 stdout/stderr 做字符数截断并返回截断标记。
- `parse_command(cmd)`: 用 `shlex` 解析命令字符串。
- `normalize_args(args)`: 规范化命令参数用于规则匹配。
- `matching_prefix(args, prefixes)`: 匹配 allowlist/denylist 前缀。
- `inspect_patch_paths(diff)`: 提取 patch 修改和删除的文件路径。
- `is_unsafe_patch_path(path)`: 判断 patch 路径是否绝对路径或包含 `..`。
- `is_env_path(path)`: 判断 patch 是否修改 `.env` 类文件。
- `load_agent_safety_policy(repo_path, policy_path)`: 读取 `.repopilot/policy.json` 或显式策略文件，默认 denylist 保持优先。
- `parse_rule_list(value, default)`: 把策略配置中的字符串/列表规则解析为命令前缀元组。
- `merge_rule_lists(...)`: 合并规则并去重。

### `coding_rag/agent/memory.py`

Agent 长期记忆 JSONL 存储。

- `AgentMemory`: 单条 Agent 记忆记录。
- `AgentMemoryStore.__init__(path, tokenizer)`: 初始化记忆库路径和 tokenizer。
- `AgentMemoryStore.load_all()`: 读取并解析 JSONL 记忆。
- `AgentMemoryStore.append(memory)`: 追加记忆。
- `AgentMemoryStore.append_if_new(memory)`: 检测重复后追加。
- `AgentMemoryStore.has_duplicate(memory)`: 判断任务和文件集合是否等价。
- `AgentMemoryStore.search(task, limit)`: 用 token overlap 检索相关记忆。
- `build_memory(task, status, summary, files, decisions)`: 构造记忆对象。
- `token_overlap(query_tokens, memory_tokens)`: 计算记忆相关性。
- `dedupe(values)`: 保序去重。
- `memory_identity(memory)`: 生成去重身份。
- `normalize_memory_text(text)`: 规范化记忆文本。

### `coding_rag/agent/plan_reviewer.py`

LLM 审查 Agent 计划合理性的模块。

- `AgentPlanReview`: 审查结果，包括 verdict、score、issues、suggestions。
- `AgentPlanReview.passed`: 是否通过审查。
- `review_agent_plan(run, client)`: 调用 LLM 审查计划。
- `build_agent_plan_review_messages(run)`: 组装审查 prompt。
- `parse_agent_plan_review(text)`: 解析 LLM 返回 JSON。
- `parse_json_object(text)`: 从文本中提取 JSON 对象。
- `clamp_score(value)`: 将分数限制到 0-1。
- `string_list(value)`: 将任意值规范为字符串列表。
- `candidate_files(run)`: 从 run 中提取候选文件。
- `render_agent_plan_review(review)`: 渲染审查结果。

## Learning Mode

### `coding_rag/learning/models.py`

Learning Mode 的结构化输出契约；所有模型拒绝额外字段、隐式类型转换和空白教学列表项。

- `EvidenceRef`: 将模型声明的 claim 绑定到仓库相对路径和正向行范围；拒绝空 claim、冒号、绝对路径、父目录穿越和倒置行号。claim 的自然语言蕴含关系本次不作自动判定，位置边界由运行时验证。
- `ProjectComponent`: 组件名称、职责和至少一条源码证据。
- `ProjectProfile`: 项目简介、技术栈、学习前置、入口和组件画像。
- `ReproductionStep`: 单步学习目标、前序依赖、待创建文件、任务、原因、收益、验证、证据、常见问题和优化问题；单步最多 20 条验收项，保证逐项审查能落入模型输出上限。
- `LearningPlan`: 项目画像和 4 到 6 个步骤；校验步骤 ID 唯一且依赖只指向前序。
- `StepSubmission`: 学习者提交的实现说明与测试输出；分别限制为 12,000 和 20,000 字符，原文只在当前回合状态中使用。
- `StepReviewFinding`、`StepReview`: 对当前步骤验收项的逐项判断与总评；`verification_scope` 固定为 `learner_reported_evidence`，结论只表示学习者报告证据是否充分。
- `RevisionFeedback`: 从失败审查派生的缺口与最小修改提示。
- `StepReviewRecord`、`StepReflectionRecord`、`LearningSessionEvent`: 保存连续尝试、反思和状态转换记录；审查记录固定 `execution_performed=false`，不声称执行过代码或测试。
- `LearningSession`: 由调用方保存并逐轮回传的完整教学会话；只允许 `awaiting_submission`、`needs_revision`、`awaiting_reflection`、`completed` 四种持久阶段，并校验步骤、审查、反思与历史转换的一致性。
- `StartLearningSessionRequest`、`SubmitLearningSessionRequest`、`ReflectLearningSessionRequest`、`LearningSessionRequest`: `start/submit/reflect` 三类严格请求及其判别联合。
- `LearningSessionError`、`LearningSessionResult`: 单回合稳定结果；`failed` 只属于本次结果，不写入持久会话阶段，携带有效会话的失败会保留上一个等待点。
- `LearningError`: 对调用方公开的稳定错误码与脱敏消息。
- `LearningWorkflowResult`: 成功时返回完整路线与第一步；失败时禁止暴露未校验计划。
- `normalize_repo_relative_path(value)`: 把路径规范为 POSIX 相对路径，并拒绝冒号、绝对路径和 `..`。

### `coding_rag/learning/workflow.py`

独立于 `UnifiedRun` 的项目拆解工作流；client、prompt、parser 留在图工厂闭包，executor 在 `collect_evidence` 节点内延迟构造，均不进入只保存 JSON 数据的 `LearningState`。

- `LearningChatClient`: 声明现有 `complete(messages) -> str` 客户端协议。
- `LearningSearchProvider`: 声明可调用的 `(query, top_k) -> list | dict` 搜索 provider；dict 使用 `results` 和可选 `retrieval_trace`。
- `LearningWorkflowConfig`: 配置仓库路径、Hybrid Search、切片和上下文上限。
- `LearningState`: 保存学习目标、证据目录、结构化计划、第一步、检索子 trace、安全 flags、路线和错误。
- `build_learning_input(learning_goal, learner_level)`: 构造字段完整的初始纯数据状态。
- `build_learning_graph(config, client, search_provider)`: 构建 `collect_evidence -> analyze_and_plan -> present_step` 三节点图；任一失败立即结束。
- `run_learning_workflow(...)`: 运行一次拆解并返回 `LearningWorkflowResult`。
- `build_learning_prompt()`: 用 LangChain `ChatPromptTemplate` 组合教学规则、不可信源码边界和结构化格式说明。
- `build_learning_executor(config, search_provider)`: 复用 RepoPilot Hybrid Search 和只读文件能力，执行器不暴露给模型。
- `collect_repository_evidence(...)`: 搜索后再次通过安全 `read_file` 固化精确路径、范围、总行数和文本。
- `normalize_search_payload(payload)`: 兼容完整 Hybrid Search 和基础只读搜索的结果形状。
- `compact_evidence(rows, max_context_chars)`: 只按完整代码行压缩上下文，并同步收窄可引用范围。
- `build_evidence_catalog(evidence)`: 建立精确路径到真实总行数和已观察范围的可信目录。
- `validate_plan_evidence(plan, catalog)`: 拒绝虚构路径、真实文件越界和本次未观察的引用范围。
- `build_learning_trace(config, state)`: 复用统一 trace 外壳记录检索、规划、展示和失败路线。
- `default_learning_safety_flags()` / `learning_executor_flags(executor)`: 记录实际 safe-mode 策略和固定只读边界。
- `coding_rag/learning/__init__.py`: 导出计划工作流、教学会话的严格数据模型，以及 `LearningSessionDependencies`、`run_learning_session()` 和三个 action 便捷入口；会话图构建器与签名辅助函数不属于公开导出面。

### `coding_rag/learning/session.py`

无服务端持久化的可恢复教学会话。每次调用统一进入一张 LangGraph，只处理一个 `start`、`submit` 或 `reflect` action，并在下一个需要学习者输入的等待点结束；调用方负责保存结果中返回的完整 session，并在下一轮原样回传。

- `LearningSessionDependencies`: 注入第 30 次任务的计划工作流配置、LLM client、至少 32 字节的服务端签名密钥、可选搜索 provider 和会话 ID 工厂；这些运行依赖不会进入图状态。
- `run_learning_session(request, dependencies)`: 统一公开入口；先限制请求体积、解析严格判别联合并验证 session 签名，再调用单回合图。在总请求未超限且 action 可识别时，即使其他字段无效，只要嵌套 session 结构和签名可信也返回原等待点；结构或签名无效的 session 不回显。
- `start_learning_session(...)`: 复用项目拆解工作流生成路线，签发 session，并停在第一步 `awaiting_submission`。
- `submit_learning_session(...)`: 对当前步骤的实现说明与测试输出做非空预检和结构化审查；未通过时停在 `needs_revision`，通过时停在 `awaiting_reflection`。
- `reflect_learning_session(...)`: 只在 `awaiting_reflection` 接收非空优化回答；推进一个步骤至 `awaiting_submission`，或在最后一步进入 `completed`。
- LangGraph 路线：`START -> dispatch` 后按 action 进入 `start_session`、`validate_submission -> review_submission -> apply_review` 或 `apply_reflection`，最后统一经过 `finalize -> END`；最终图状态会清除本回合临时 `submission/reflection` 输入槽，合法反思仍写入签名 session 的反思记录。
- 完整性边界：session 的 `integrity_token` 是 HMAC-SHA256；签名覆盖除 token 自身之外的完整 canonical JSON。审查记录只保存域分离的 keyed HMAC 提交摘要、字符数和按当前验收项生成的定长具体反馈，不持久化模型自由文本或原始提交。
- 资源边界：完整请求 canonical JSON 上限为 500,000 字符，签名 session 上限为 450,000 canonical JSON 字符，单次模型审查输出上限为 50,000 字符；每次签发还会模拟“剩余每步一次通过 + 最短反思”，拒绝没有最小完成空间的状态。字段级长度限制仍由严格 Pydantic 模型执行。
- 安全边界：步骤审查只依据学习者报告，`execution_performed` 恒为 `false`；公开 trace 只展示仓库名，不暴露服务端绝对路径。会话核心不接 checkpoint、数据库、文件写入或代码执行，现由 `frontend.py` 的独立只读 API 适配。无状态签名只检测篡改，不提供重放或分叉防护；服务重启若更换签名密钥，旧 session 会失效。

## RAG 与 LLM 模块

### `coding_rag/rag/ask.py`

ASK 模式的数据结构和检索/prompt 流程。

- `AskModeConfig`: ASK 检索和 prompt 配置。
- `AskModeRun`: ASK 运行产物。
- `retrieve_for_ask(query, config)`: 加载文件、切 chunk、检索、邻近召回、过滤并生成 trace。
- `resolve_max_recall_results(candidate_k, recall_window, max_recall_results)`: 解析召回结果数量上限。
- `build_ask_messages(query, context, generation_mode)`: 构造 ASK LLM 消息。

### `coding_rag/rag/prompt.py`

LLM prompt 生成和上下文压缩。

- `GenerationMode`: 生成模式：judge、code-understand、code-generate、leetcode、api。
- `build_judge_user_prompt(query, context)`: 构造判定/回答 prompt。
- `build_code_understand_user_prompt(query, context)`: 构造代码理解 prompt。
- `build_code_generate_user_prompt(query, context)`: 构造代码生成 prompt。
- `build_leetcode_user_prompt(query, context)`: 构造 LeetCode 解题 prompt。
- `build_api_user_prompt(query, context)`: 构造 API 示例 prompt。
- `get_system_prompt(mode)`: 获取对应模式的 system prompt。
- `build_user_prompt(mode, query, context)`: 按模式分发到具体 user prompt。
- `ContextBlock`: 合并后的上下文块。
- `format_results_as_context(results, max_context_chars)`: 将检索结果格式化为 LLM 上下文。
- `compact_results_for_context(results, max_context_chars)`: 合并同文件相邻结果，减少重复上下文。
- `merge_result_into_block(block, result)`: 将结果合入待输出上下文块。

### `coding_rag/rag/llm_client.py`

OpenAI-compatible LLM client 和 provider 配置。

- `ProviderPreset`: provider 预设。
- `LLMConfig`: LLM 客户端配置。
- `OpenAICompatibleChatClient.__init__(config)`: 初始化客户端。
- `OpenAICompatibleChatClient.complete(messages)`: 调用 `/chat/completions` 并返回文本。
- `build_llm_config(...)`: 合并 provider preset、CLI 参数和环境变量构造配置。
- `available_provider_names()`: 返回可用 provider 名称。
- `chat_completions_url(base_url)`: 构造 chat completions URL。
- `extract_message_content(payload)`: 从响应 JSON 中提取文本。
- `first_value(*values)`: 返回第一个非空配置值。

### `coding_rag/rag/answer_generator.py`

较新的 LLM 回答生成封装。

- `AnswerGenerator.__init__(client, mode, max_context_chars)`: 初始化生成器。
- `AnswerGenerator.generate(query, results)`: 将检索结果组织进 prompt 并调用 LLM。
- `AnswerGenerator.set_mode(mode)`: 切换生成模式。
- `AnswerGenerator.set_max_context_chars(max_chars)`: 调整上下文长度限制。
- `build_generator(...)`: 从 provider 参数创建 `AnswerGenerator`。

### `coding_rag/rag/judge.py`

旧版 LLM judge 兼容流程。

- `LLMJudge.judge(query, results)`: 调用 legacy judge prompt。
- `build_judge_messages(query, results, max_context_chars)`: 构造 legacy judge 消息。
- `format_results_for_prompt(results, max_context_chars)`: 格式化检索结果给 legacy prompt。

### `coding_rag/rag/citation_validator.py`

回答引用校验。

- `Citation`: 单个 `path:start-end` 引用。
- `CitationValidationResult`: 引用校验结果。
- `CitationValidationResult.has_issues`: 是否存在缺失或无效引用。
- `extract_citations(text)`: 提取回答中的引用。
- `validate_answer_citations(answer, results)`: 校验引用是否由检索上下文支持。
- `validate_answer_citations_against_ranges(answer, available_ranges)`: 校验统一工具循环中已观察文件范围的引用。
- `citation_is_supported(citation, results)`: 判断单个引用是否覆盖检索结果行号。
- `paths_match(left, right)`: 判断引用路径和结果路径是否匹配。
- `normalize_path(path)`: 规范化路径分隔符。
- `is_refusal_answer(answer)`: 判断回答是否是证据不足的拒答。
- `append_citation_validation_report(answer, validation)`: 在回答后追加引用校验报告。

### `coding_rag/rag/trace.py`

RAG trace 构造、序列化和渲染。

- `TRACE_VERSION`: 当前统一 trace 格式版本。
- `build_retrieval_trace(query, seed_results, recalled_results, final_results, params, include_text)`: 构造完整检索轨迹。
- `build_trace_run(...)`: 构造统一 trace 的 run 级摘要。
- `build_trace_event(...)`: 构造统一 trace 的 step/tool 事件。
- `build_unified_retrieval_events(stages)`: 将旧检索 stages 转换为统一 events。
- `build_unified_retrieval_trace(...)`: 用已序列化 stages 构造统一检索 trace。
- `build_agent_events(agent_trace)`: 将旧 Agent step trace 转换为统一 events。
- `build_tool_event(tool, result, input)`: 将 executor 工具结果转换为统一 tool event。
- `normalize_event_status(status)`: 将旧 Agent 状态规范为统一事件状态。
- `summarize_artifact_counts(artifacts)`: 从事件 artifacts 中提取计数摘要。
- `serialize_results(results, include_text)`: 序列化结果列表。
- `serialize_result(result, include_text)`: 序列化单个结果。
- `serialize_context_blocks(results)`: 序列化上下文压缩块。
- `render_trace_report(trace, limit)`: 将 trace 渲染为文本。
- `get_retrieval_stages(trace)`: 从统一 `artifacts.retrieval` 或旧 `stages` 中读取检索阶段。
- `render_trace_row(row)`: 渲染 trace 表格行。
- `write_trace_json(path, trace)`: 写入 trace JSON。
- `preview_text(text, limit)`: 生成代码预览文本。

## 检索与仓库结构

### `coding_rag/tools/bm25.py`

Hybrid Search 检索器，融合 BM25、结构化 metadata、测试文件降权和文件多样性。

- `SearchResult`: 检索结果。
- `BM25Retriever.__init__(chunks)`: 构建 BM25 corpus 和 repo index。
- `BM25Retriever.search(query, top_k)`: 返回排序后的检索结果。
- `BM25Retriever.hybrid_scores(query_tokens, bm25_scores)`: 融合 BM25 和结构化分数。
- `BM25Retriever.rescore_results(query, results)`: 对召回结果重新评分。
- `chunk_to_document(chunk)`: 将 chunk 文本转检索文档。
- `chunk_to_metadata_document(chunk, repo_index)`: 将路径、符号、import、调用关系转 metadata 文档。
- `normalize_scores(scores)`: 分数归一化。
- `token_overlap_score(query_tokens, document_tokens)`: token overlap 分数。
- `adjust_score_for_document_role(path, query_tokens, score)`: 根据测试意图调整测试文件排序。
- `has_test_intent(query_tokens)`: 判断查询是否明确找测试。
- `is_test_path(path)`: 判断路径是否是测试文件。
- `normalize_path(path)`: 规范化路径。
- `diversify_by_file(results, top_k)`: 提升文件多样性。
- `chunk_key(result)`: 生成结果去重键。

### `coding_rag/tools/tokenizer.py`

中英混合 tokenizer 和查询词扩展。

- `CodeTokenizer.tokenize(text)`: 将文本切成检索 token，并注入同义扩展。
- `CodeTokenizer.tokenize_identifier(identifier)`: 切分 snake_case / camelCase 标识符。
- `CodeTokenizer.tokenize_chinese(text)`: 为中文生成 n-gram token。
- `tokenize(text)`: 默认 tokenizer 便捷函数。
- `split_identifier(identifier)`: 标识符拆分工具。
- `is_chinese(text)`: 判断文本是否全中文。

### `coding_rag/tools/recall.py`

邻近 chunk 召回。

- `expand_with_neighbor_chunks(chunks, seed_results, window, max_results)`: 围绕种子结果召回同文件相邻 chunk。
- `group_chunks_by_file(chunks)`: 按文件分组 chunk。
- `build_chunk_index(chunks_by_file)`: 构造 chunk 到文件内位置索引。
- `add_or_upgrade_result(recalled, result)`: 合并或升级召回结果来源/分数。
- `is_seed_source(source)`: 判断是否是种子来源。
- `chunk_key(chunk)`: 生成 chunk 唯一键。

### `coding_rag/tools/filter.py`

召回后结果重排和过滤。

- `filter_recalled_results(query, recalled_results, retriever, final_k, min_score)`: 对召回结果重评分并过滤。
- `sort_key(result)`: 最终排序键。
- `source_priority(source)`: 设置不同来源优先级。

### `coding_rag/tools/env.py`

本地 `.env` 配置。

- `ensure_dotenv(path)`: 不存在时创建 `.env` 模板。
- `load_dotenv(path, override)`: 加载 `.env` 到环境变量。
- `parse_env_line(line)`: 解析 `.env` 单行。
- `unquote_env_value(value)`: 去掉引号并处理转义。

### `coding_rag/tools/agent_readonly.py`

Agent 只读工具。

- `AgentToolError`: 工具调用错误。
- `ReadOnlyAgentTools.__init__(repo_path, chunk_size, overlap)`: 绑定仓库和索引参数。
- `ReadOnlyAgentTools.call(name, arguments)`: 只读工具分发入口。
- `ReadOnlyAgentTools.search_code(query, top_k)`: 搜索代码并返回结构化结果。
- `ReadOnlyAgentTools.read_file(path, start_line, end_line)`: 读取仓库内文件范围。
- `ReadOnlyAgentTools.list_files(pattern, limit)`: 列出匹配文件。
- `ReadOnlyAgentTools.inspect_symbol(symbol, context_lines)`: 查找符号定义及上下文。
- `ReadOnlyAgentTools.resolve_repo_path(path)`: 防止访问仓库外路径。
- `ReadOnlyAgentTools.display_path(path)`: 转相对展示路径。
- `ReadOnlyAgentTools.chunks`: 懒加载 chunk。
- `ReadOnlyAgentTools.retriever`: 懒加载 BM25 检索器。
- `ReadOnlyAgentTools.repo_index`: 懒加载结构化索引。
- `extract_task_identifiers(task, limit)`: 从任务中提取候选标识符。
- `format_observations(observations)`: 将只读工具观察渲染成 prompt 文本。
- `format_location(item)`: 格式化文件行号位置。
- `optional_int(value)`: 参数转可选整数。

### `coding_rag/repository/files.py`

仓库文件加载。

- `PythonFile`: Python 文件数据结构。
- `load_python_files(repo_path)`: 遍历仓库并加载 `.py` 文件。
- `should_skip(path, repo_path)`: 忽略 `.git`、虚拟环境、缓存等目录。

### `coding_rag/repository/chunks.py`

代码切片。

- `CodeChunk`: 代码块数据结构。
- `split_python_files(python_files, chunk_size, overlap)`: 批量切分 Python 文件。
- `split_code_by_lines(file_path, text, chunk_size, overlap)`: 滑动窗口按行切分单个文件。

### `coding_rag/repository/index.py`

基于 AST 的结构化仓库索引。

- `SymbolRecord`: 函数/类符号记录。
- `FileIndex`: 单文件结构化索引。
- `FileIndex.symbols_in_range(start, end)`: 查找范围内符号。
- `RepoIndex.for_chunk(chunk)`: 找到 chunk 对应文件索引。
- `RepoIndex.metadata_for_chunk(chunk)`: 获取 chunk 的结构化 metadata。
- `RepoIndex.structural_score(query_tokens, chunk)`: 计算结构化匹配分。
- `build_repo_index(chunks)`: 从 chunk 构建仓库索引。
- `build_file_index(path, chunks)`: 构建单文件索引。
- `rebuild_source(chunks)`: 拼回文件源码文本。
- `extract_symbols(tree)`: 提取类/函数符号。
- `extract_imports(tree)`: 提取 import/from import。
- `extract_calls(tree)`: 提取函数调用。
- `dotted_name(node)`: 获取 dotted name。
- `function_signature(node)`: 生成函数签名。
- `class_signature(node)`: 生成类签名。
- `symbol_metadata(symbols, start, end)`: 生成符号 metadata。
- `import_terms(imports)`: 生成 import 检索词。
- `module_name(path)`: 从路径推断模块名。
- `ranges_overlap(a_start, a_end, b_start, b_end)`: 判断行号范围重叠。
- `normalize_path(path)`: 规范化路径。
- `unique_preserve_order(values)`: 保序去重。
- `terms_match(query_terms, doc_terms)`: 判断 token 是否匹配。
- `identifier_terms(text)`: 从标识符文本提取 token。

## 脚本与辅助文件

### `scripts/retrieval_eval.py`

检索评测、bad case 诊断、trace 导出和参数优化。

- `EvalCase`: 单条评测用例。
- `load_evalset(path)`: 加载 JSON/JSONL 评测集。
- `normalize_eval_case(payload, index)`: 规范化评测样例。
- `build_eval_generator(args)`: 构建 LLM 评测回答生成器。
- `generate_eval_answer(generator, setup_error, query, results)`: 为单条样例生成 LLM 回答。
- `pipeline(case, args)`: 执行单条评测检索流水线。
- `default_recall_max_results(candidate_k, recall_window)`: 计算默认召回上限。
- `mrr_and_recall_at_k(rows, k)`: 计算 MRR 和 Recall@k。
- `analyze_bad_case(case, seed, recalled, final)`: 诊断 bad case 阶段。
- `build_stage_file_record(case, seed, recalled, final)`: 构造阶段文件记录。
- `build_stage_snapshot(results)`: 构造阶段快照。
- `matched_relevant_files(results, relevant)`: 计算命中的 gold 文件。
- `unique_result_files(results)`: 提取唯一结果文件。
- `readable_path(path)`: 路径显示规范化。
- `diagnose_stage_drop(record)`: 判断正确文件在哪个阶段丢失。
- `aggregate_stage_diagnostics(rows)`: 汇总 bad case 原因。
- `context_char_totals(rows)`: 统计上下文字符数。
- `render_stage_diagnostics_summary(diagnostics)`: 渲染诊断摘要。
- `ordered_diagnoses(diagnostics)`: 按固定顺序输出诊断项。
- `is_relevant_result(result, relevant)`: 判断结果是否命中 gold。
- `matched_relevant_index(results, relevant)`: 返回首个命中位置。
- `comparable_path_forms(path)`: 生成可比较路径形式。
- `normalize_path_text(path)`: 路径文本规范化。
- `stage_files_path_for_trace(path)`: trace 文件名派生 stage files 路径。
- `stage_summary_path_for_trace(path)`: trace 文件名派生 stage summary 路径。
- `write_json(path, data)`: 写 JSON。
- `write_jsonl(path, rows)`: 写 JSONL。
- `run_eval(args)`: 执行完整评测。
- `optimize(args, cases)`: 参数优化入口。
- `collect_optimization_results(args, cases)`: 收集不同参数组合结果。
- `render_optimization_report(rows, best)`: 渲染优化报告。
- `print_llm_outputs(rows)`: 打印每条 LLM 输出。
- `parse_args()`: 解析评测脚本参数。

### `scripts/run_claude_guarded.py`

带 token 和 cache miss 守卫的 Claude Code 调用脚本。

- `ClaudeUsage.task_tokens`: 当前任务 token 总量。
- `ClaudeUsage.cache_miss_tokens`: cache miss token 数。
- `ClaudeUsage.cache_input_tokens`: cache input token 数。
- `ClaudeUsage.cache_miss_ratio`: cache miss 比例。
- `parse_args()`: 解析 guard 参数。
- `main()`: 启动 Claude 子进程、读取事件流、执行守卫。
- `parse_json_line(line)`: 解析 JSON 事件行。
- `update_usage_from_event(usage, event)`: 从事件更新 token 使用量。
- `find_usage_payload(event)`: 查找 usage payload。
- `find_model_usage_payloads(event)`: 查找模型 usage payload。
- `guard_reason(args, usage)`: 判断是否需要终止。
- `write_log(path, record)`: 写 guard 日志。

### `scripts/import_agent_plan_cases.py`

从 SWE-bench Verified 导入 Agent 计划测试集。

- `parse_args()`: 解析导入参数。
- `main()`: 拉取数据、转换并写入 JSON。
- `fetch_rows(limit)`: 从数据源取样例。
- `convert_row(row)`: 转换单条样例。
- `diff_paths(patch_text)`: 从 patch 中提取文件路径。
- `build_repo_files(row, paths)`: 构造最小 repo 文件集合。
- `normalize_task(text)`: 清理任务文本。
- `parse_json_list(value)`: 解析 JSON 列表字段。

### `scripts/generate_function_map.py`

基于 AST 生成函数地图 Markdown。

- `parse_args()`: 解析扫描根目录和输出路径。
- `discover_python_files(paths, base)`: 展开文件/目录并过滤缓存、虚拟环境和 Git 目录。
- `inspect_file(path)`: 提取顶层类、函数和类方法。
- `first_sentence(text)`: 提取 docstring 首句。
- `generate_function_map(paths, base)`: 生成函数地图 Markdown 文本。
- `main()`: 写出或打印函数地图。

### `scripts/agent_eval.py`

Agent 计划离线评测入口。

- `OfflinePlanClient.complete(messages)`: 返回稳定的离线计划模板。
- `evaluate_cases(evalset, limit)`: 运行测试集并统计候选文件、工具覆盖、章节覆盖、trace 完整性和安全工具接口。
- `coverage(expected, actual)`: 计算集合覆盖率。
- `average(values)`: 计算平均值。
- `write_jsonl(path, rows)`: 写入 JSONL 明细。
- `main()`: 生成 JSONL 明细和 summary JSON。

### `scripts/patch_leetcode_imports.py`

批量为 LeetCode 解答补类型 import 的工具脚本。

- `parse_args()`: 解析脚本参数。
- `main()`: 遍历文件并插入 import。
- `should_skip(path)`: 判断文件是否跳过。
- `find_insert_index(lines)`: 找到 import 插入位置。
- `skip_blank_and_comment_lines(lines, index)`: 跳过空行和注释。
- `skip_module_docstring(lines, index)`: 跳过模块 docstring。

### `leetcode_types.py`

LeetCode 本地运行用数据结构和转换函数。

- `ListNode`: 链表节点。
- `TreeNode`: 二叉树节点。
- `Node`: 通用节点，兼容多种 LeetCode 结构。
- `build_linked_list(values)`: 从列表构造链表。
- `linked_list_to_list(head)`: 链表转列表。
- `build_binary_tree(values)`: 层序列表构造二叉树。
- `binary_tree_to_list(root)`: 二叉树转层序列表。
