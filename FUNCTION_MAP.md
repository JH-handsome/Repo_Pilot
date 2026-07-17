# RepoPilot Function Map

本文件记录当前源码中关键文件、类和函数的职责，便于定位 ASK、Agent、检索、LLM、评测和前端流程。

## 顶层入口

### `main.py`

CLI 统一入口，所有问题和修改需求都交给 LLM 决定直接回答或调用工具。

- `parse_args()`: 定义检索、回答风格、LLM、policy、trace 和 `--execute-tools` 参数；不再包含 ASK/Agent 模式参数。
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

轻量浏览器前端，只提供统一 `/api/run`，并使用服务端与页面双重执行授权。

- `parse_args()`: 解析监听地址、端口和 `--allow-tool-execution`。
- `RepoPilotServer`: 保存服务级工具执行授权。
- `RepoPilotHandler.do_GET()`: 返回首页 HTML 或 404。
- `RepoPilotHandler.do_POST()`: 只处理 `/api/run`，按错误类型返回 400/403/502/500。
- `RepoPilotHandler.server_allows_execution()`: 读取服务级执行授权。
- `RepoPilotHandler.read_json()`: 读取并解析请求体 JSON。
- `RepoPilotHandler.send_html()`: 返回 HTML 页面。
- `RepoPilotHandler.send_json()`: 返回 JSON API 响应。
- `RepoPilotHandler.log_message()`: 静默 HTTP server 默认日志。
- `run_frontend_query(payload, server_allows_execution)`: 校验双重授权并运行统一 LLM 工具循环。
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

CLI 和浏览器共用的统一 LLM 决策循环。

- `UnifiedRunConfig`: 统一运行配置，包括 Hybrid Search、回答风格、上下文限制、policy 和最大工具步数。
- `ModelDecision`: 模型结构化决策，字段为 `action/reason/answer/tool/arguments/expected_observation`。
- `UnifiedRun`: 最终回答、状态、决策、observation、trace 和执行授权。
- `run_unified_query(query, config, client, execute_tools, safety_policy)`: 在 `answer` 与五个工具之间循环，处理修复、步数限制、动态状态和引用校验。
- `build_hybrid_search_provider(config)`: 把 ASK 的候选检索、邻居召回和最终过滤包装为 executor 搜索 provider。
- `serialize_search_results(results, repo_path)`: 将最终检索结果转为带相对路径和行号的工具输出。
- `build_unified_messages(query, config, execution_enabled)`: 生成 JSON 决策协议、工具 schema 和不可信仓库数据约束。
- `parse_model_decision(raw_text)`: 校验模型 JSON、action、工具名和参数类型。
- `build_unified_trace(...)`: 创建 `mode=unified` 的 run/events/artifacts 外壳。
- `compact_tool_result(tool, result, max_chars)`: 截断发送回 LLM 和写入 trace 的 observation。
- `collect_observed_ranges(tool, result, observed_ranges)`: 收集搜索和文件读取证据范围供引用校验。
- `finish_unified_run(...)`: 计算摘要并完成统一运行结果。
- `unified_run_to_dict(run)`: 生成 CLI/HTTP 共用的响应结构。

### `coding_rag/agent/planner.py`

ASK/Agent 分类和 ReAct 计划模式核心。

- `WorkflowMode`: 用户任务模式枚举：`auto`、`ask`、`agent`。
- `TaskPlan`: 分类结果，包括模式、意图、生成模式、步骤和原因。
- `AgentToolSpec`: Agent 工具接口声明。
- `AgentPlanConfig`: Agent 计划/执行配置，包括检索参数、只读工具开关和可选 `agent_policy_path`。
- `AgentPlanRun`: Agent 计划运行产物。
- `ReActAgentInterface.plan(task)`: 强制把任务规划为 Agent 模式。
- `ReActAgentInterface.run(task)`: 执行最小 ReAct 工具循环；解析 LLM JSON tool call，默认 dry-run，并把工具调用写入统一 trace。
- `ReActAgentInterface.build_plan(task, config, client)`: 构建 Agent 计划运行结果。
- `classify_task(user_input, requested_mode, generation_mode)`: 按显式模式或关键词推断 ASK/Agent。
- `infer_mode(user_input)`: 使用 fix/implement/refactor 等关键词推断模式。
- `run_ask_mode(query, config, client, requested_mode)`: ASK 核心流程；检索、构建 prompt、可选调用 LLM、校验引用。
- `run_agent_plan_mode(task, config, client, tools)`: Agent 计划流程；默认不检索，开启只读工具时收集观察。
- `build_agent_plan_trace(task, config)`: 构造 Agent 计划 trace 骨架。
- `collect_readonly_observations(task, config)`: 调用 `search_code`、`list_files`、`inspect_symbol` 收集只读观察。
- `run_agent_execution_loop(task, config, client, tools, dry_run, safe_mode, max_steps, safety_policy)`: 调度 `read_file`、`search_code`、`apply_patch`、`run_command`、`inspect_diff`，处理未知工具和安全拒绝。
- `build_agent_execution_trace(task, config, dry_run, safe_mode)`: 生成 Agent 执行统一 trace 外壳。
- `parse_agent_tool_call(raw_text)`: 从 LLM 输出中解析 `thought/tool/arguments/expected_observation`。
- `default_agent_tools()`: 声明 Agent 可用工具，包括 `read_file`、`search_code`、`apply_patch`、`run_command`、`inspect_diff` 等。
- `build_agent_plan_messages(task, plan, context, tools, observations)`: 组装发给 LLM 的 Agent 计划 prompt。
- `format_tool_specs(tools)`: 将工具声明渲染为 prompt 文本。
- `render_agent_plan_run(run)`: 渲染 Agent 计划运行结果。
- `render_agent_plan_prompt(messages)`: 渲染 Agent 计划 prompt。
- `render_task_plan(plan)`: 渲染任务分类结果。
- `render_ask_prompt(messages)`: 渲染 ASK prompt。

### `coding_rag/agent/workflow.py`

代码 Agent 工作流：记忆、检索、计划、实现草案、日志和推荐验证命令。

- `CodeAgentConfig`: 代码 Agent 配置，包括检索、记忆、日志、dry-run/safe-mode。
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
- `AgentExecutor.__init__(repo_path, chunk_size, overlap, command_timeout, dry_run, safe_mode, safety_policy, search_provider)`: 绑定仓库根目录、只读工具、安全策略和可选 Hybrid Search provider。
- `AgentExecutor.call(name, arguments)`: 工具分发入口。
- `AgentExecutor.read_file(path, start_line, end_line)`: 读取仓库内文件。
- `AgentExecutor.search_code(query, top_k)`: 优先调用注入的 Hybrid Search provider，否则复用基础只读检索。
- `AgentExecutor.apply_patch(diff)`: 经过安全检查后校验并应用 unified diff；dry-run/safe-mode 只做 `git apply --check`。
- `AgentExecutor.run_command(cmd, timeout_seconds)`: 经过 allowlist/denylist 检查后以 `shell=False` 执行命令，并返回截断信息。
- `AgentExecutor.inspect_diff()`: 查看当前 `git diff --no-ext-diff --`。
- `AgentExecutor._run_git(command, stdin)`: 执行 git 子命令。
- `run_subprocess(command, cwd, timeout, shell, stdin)`: 子进程执行封装。
- `command_result_to_dict(result, safety)`: 将命令结果转 JSON dict，并按安全策略截断 stdout/stderr。
- `ensure_allowed(decision)`: 将安全检查拒绝结果转换为工具错误。
- `decision_to_dict(decision)`: 序列化安全检查结果。
- `optional_int(value)`: 参数转可选整数。

### `coding_rag/agent/safety.py`

Agent 可执行工具安全策略层，统一约束命令执行和 patch 应用。

- `AgentSafetyPolicy`: 配置 dry-run/safe-mode、命令超时、输出截断、patch 大小、allowlist 和 denylist。
- `SafetyDecision`: 表示安全检查结果，包括是否允许、命中规则、原因、dry-run 标记和元数据。
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
