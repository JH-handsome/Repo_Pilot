# RepoPilot Function Map

本文件由 `scripts/generate_function_map.py` 基于 AST 生成，列出主要 Python 文件中的类和函数。

## `coding_rag/agent/conversation.py`

- `function` `conversation_path` (line 16) - Resolve the chat database independently of optional trace persistence.
- `function` `session_key` (line 24) - Validate a UUID capability and scope it to the canonical repository.
- `class` `ConversationStore` (line 33) - Small SQLite turn log; never stores graph checkpoints or write approvals.
- `method` `ConversationStore.__init__` (line 36) - Bind a validated session to a local database and repository.
- `method` `ConversationStore.connect` (line 42) - Create a short-lived connection with an independent chat schema.
- `method` `ConversationStore.load` (line 52) - Read complete turns in order, returning their optimistic revision.
- `method` `ConversationStore.append` (line 61) - Append once; reject stale state instead of overwriting concurrent history.
- `function` `history_messages` (line 79) - Keep the initial context and latest complete turns; report every omission.

## `coding_rag/agent/executor.py`

- `class` `CommandResult` (line 20) - 子进程命令执行的结果，包含命令、返回码、标准输出和标准错误。
- `class` `WriteApprovalRequired` (line 32) - Raised when a real write lacks an exact, unused approval fingerprint.
- `method` `WriteApprovalRequired.__init__` (line 35) - Preserve the approval envelope for CLI/HTTP presentation.
- `class` `AgentExecutor` (line 41) - Small execution surface for Agent tool calls.
- `method` `AgentExecutor.__init__` (line 48) - 初始化 Agent 执行器，绑定仓库路径并配置安全策略与工具集。
- `method` `AgentExecutor.call` (line 86) - 根据工具名称分派调用。
- `method` `AgentExecutor.read_file` (line 118) - 委托仓库文件读取。
- `method` `AgentExecutor.search_code` (line 127) - 优先使用注入的搜索提供者，否则回退到只读后端。
- `method` `AgentExecutor.trace_call_path` (line 133) - Trace static Python calls using the same repository read boundary.
- `method` `AgentExecutor.preview_write` (line 137) - Validate a write request and return its exact approval envelope.
- `method` `AgentExecutor.apply_patch` (line 165) - Check, approve, snapshot and apply one unified diff.
- `method` `AgentExecutor.run_command` (line 196) - Check, approve, snapshot and execute one allowlisted command.
- `method` `AgentExecutor.inspect_diff` (line 239) - 仅返回仓库读取策略允许的未暂存文件差异。
- `method` `AgentExecutor.rollback_snapshot` (line 269) - Restore a previously cached approved write.
- `method` `AgentExecutor._consume_write_approval` (line 273) - Consume one approval only when the exact tool arguments match.
- `method` `AgentExecutor._rollback_failed_write` (line 279) - Restore declared files after an execution exception, surfacing rollback failure.
- `method` `AgentExecutor._run_git` (line 289) - 使用策略超时在仓库中运行 git。
- `function` `run_subprocess` (line 300) - 运行子进程并将超时转换为 AgentToolError。
- `function` `command_result_to_dict` (line 329) - 截断命令输出以构建 payload。
- `function` `ensure_allowed` (line 345) - 拒绝被禁止的安全决策。
- `function` `decision_to_dict` (line 351) - 序列化安全决策。
- `function` `canonical_write_arguments` (line 361) - Normalize fields covered by one approval fingerprint.
- `function` `write_call_fingerprint` (line 379) - Hash one exact write tool call for single-use approval.
- `function` `optional_int` (line 390) - 将可选值转换为 int。

## `coding_rag/agent/graph_runtime.py`

- `function` `consume_llm_call_observation` (line 84) - Safely consume one optional observation without weakening ChatClient compatibility.
- `class` `BoundToolCallingModel` (line 96) - 描述已经绑定工具、可以同步调用的聊天模型。
- `method` `BoundToolCallingModel.invoke` (line 99) - 根据消息列表返回一条模型消息。
- `class` `ToolCallingChatModel` (line 104) - 描述支持 ``bind_tools`` 的可注入聊天模型。
- `method` `ToolCallingChatModel.bind_tools` (line 107) - 绑定模型可见工具并返回可调用模型。
- `class` `GraphState` (line 116) - LangGraph 节点之间共享的纯数据状态。
- `class` `UnifiedGraphState` (line 136) - 普通 Agent 的 JSON 决策图状态。
- `function` `build_graph_input` (line 161) - 为 LangGraph 构造字段完整、可序列化的初始状态。
- `function` `build_unified_graph_input` (line 186) - 为普通 Agent 构造字段完整、可序列化的初始图状态。
- `function` `build_readonly_graph` (line 217) - 构建并编译只暴露只读仓库工具的 LangGraph。
- `function` `build_unified_graph` (line 469) - 构建普通 Agent 的 JSON 决策 LangGraph。
- `function` `run_unified_graph_query` (line 859) - 运行普通 Agent 的有界 LangGraph，并返回现有 UnifiedRun 结构。
- `function` `run_graph_query` (line 960) - 运行一次有界的 LangGraph 只读查询并返回现有结果结构。
- `function` `build_readonly_executor` (line 1006) - 构造带 Hybrid Search、且由 safe-mode 拒绝所有写操作的执行器。
- `function` `validate_ai_message` (line 1027) - 把无效工具调用或空回答转换为不可恢复的模型错误。
- `function` `ai_message_text` (line 1049) - 从字符串或文本内容块中提取可展示的模型文本。
- `function` `ai_message_to_decision_payload` (line 1063) - 把 AIMessage 转换为可序列化、且兼容 ModelDecision 的字段。
- `function` `last_ai_message` (line 1085) - 从序列化消息列表末尾向前寻找最近的 AIMessage。
- `function` `build_tool_message` (line 1093) - 构造带不可信数据提示、调用 ID 和成败状态的 ToolMessage。
- `function` `serialize_message` (line 1114) - 把 LangChain 消息转换为只含 JSON 安全值的状态字典。
- `function` `json_safe` (line 1119) - 递归规范化任意值，保证其可以直接写入 JSON 状态。
- `function` `fallback_error_for_state` (line 1124) - 根据当前状态确定 safe_fallback 的结构化错误。
- `function` `terminal_error` (line 1182) - 构造统一的不可恢复错误结构。
- `function` `fallback_status` (line 1197) - 根据已有观察和错误类型决定 failed 或 partial。
- `function` `fallback_answer` (line 1209) - 把 fallback 错误转换为不会伪装成功的最终回答。
- `function` `fallback_event_step` (line 1216) - 为关键 fallback 分支选择稳定的 trace 事件名。
- `function` `build_graph_trace` (line 1225) - 使用现有 trace schema 汇总 LangGraph 运行状态。
- `function` `messages_for_unified_run` (line 1284) - 把 LangChain 消息转换为现有 UnifiedRun 的角色/文本列表。
- `function` `message_role` (line 1300) - 把 LangChain 消息类型映射为现有运行时角色名。

## `coding_rag/agent/langchain_tools.py`

- `class` `_ToolExecutor` (line 28) - 描述 RepoPilot 窄执行入口的结构化类型。
- `method` `_ToolExecutor.call` (line 31) - 分派一次已经完成参数校验的仓库工具调用。
- `class` `ReadOnlyToolInput` (line 36) - 只读工具参数基类，拒绝模型提供的未声明参数。
- `class` `SearchCodeInput` (line 42) - 仓库代码搜索工具的参数约束。
- `class` `ReadFileInput` (line 54) - 仓库文件读取工具的参数约束。
- `method` `ReadFileInput.validate_line_range` (line 70) - 在访问仓库前拒绝起止行倒置的显式范围。
- `class` `TraceCallPathInput` (line 81) - Exact Python symbol to trace after a file has been located.
- `class` `ApplyPatchInput` (line 88) - 代码补丁工具的参数约束。
- `class` `RunCommandInput` (line 96) - 仓库命令工具的参数约束。
- `class` `InspectDiffInput` (line 112) - 查看当前仓库 diff；该工具不接受参数。
- `function` `build_readonly_langchain_tool_map` (line 116) - 构建只读运行时允许使用的 LangChain 工具。
- `function` `build_readonly_langchain_tools` (line 182) - 按稳定顺序返回可绑定到模型的只读工具。
- `function` `build_agent_langchain_tool_map` (line 188) - 构建普通 Agent 的完整工具集合，执行控制仍由 executor 负责。
- `function` `build_agent_langchain_tools` (line 249) - 按稳定顺序返回普通 Agent 的完整 LangChain 工具集合。

## `coding_rag/agent/memory.py`

- `class` `AgentMemory` (line 14) - 一次 Agent 任务的长期记忆记录。
- `class` `AgentMemoryStore` (line 26) - JSONL 本地记忆库。
- `method` `AgentMemoryStore.__init__` (line 29) - 初始化文件持久化的Agent记忆存储，绑定路径和可选的分词器。
- `method` `AgentMemoryStore.load_all` (line 34) - 从持久化JSONL文件中加载全部Agent记忆记录。
- `method` `AgentMemoryStore.append` (line 50) - 将单条Agent记忆序列化为JSON行并追加写入持久化文件。
- `method` `AgentMemoryStore.append_if_new` (line 56) - 写入非重复记忆；如果已有等价任务和文件集合，则跳过。
- `method` `AgentMemoryStore.has_duplicate` (line 63) - 通过比较身份键检测重复记忆。
- `method` `AgentMemoryStore.search` (line 68) - 基于查询与记忆文本之间的token重叠度进行搜索排序。
- `function` `build_memory` (line 88) - 根据任务元数据构造AgentMemory实例，并附加UTC创建时间戳。
- `function` `token_overlap` (line 108) - 计算查询token集合与记忆token集合的交集占比。
- `function` `dedupe` (line 115) - 按首次出现顺序进行稳定去重。
- `function` `memory_identity` (line 124) - 基于规范化任务文本和排序文件列表计算记忆的身份标识。
- `function` `normalize_memory_text` (line 129) - 通过大小写折叠与空白压缩对文本进行规范化处理。

## `coding_rag/agent/plan_reviewer.py`

- `class` `ChatClient` (line 13) - 聊天客户端协议。
- `method` `ChatClient.complete` (line 15) - 完成消息对话。
- `class` `AgentPlanReview` (line 21) - 智能体计划审查结果。
- `method` `AgentPlanReview.passed` (line 30) - 审查是否通过。
- `function` `review_agent_plan` (line 35) - Call an LLM to review whether a generated Agent plan is reasonable.
- `function` `build_agent_plan_review_messages` (line 42) - 构造审查提示消息。
- `function` `parse_agent_plan_review` (line 82) - 解析审查器响应。
- `function` `parse_json_object` (line 97) - 从响应文本中提取 JSON 对象。
- `function` `clamp_score` (line 124) - 将评分限制在 [0.0, 1.0] 范围内。
- `function` `string_list` (line 133) - 将列表中每个元素转为 str，非列表时返回 []。
- `function` `candidate_files` (line 140) - 从 run.final_results 的检索块中收集唯一文件路径，而非从计划步骤中收集。
- `function` `render_agent_plan_review` (line 150) - 将审查结论、评分、通过状态、问题与建议格式化为 Markdown 风格的审查文本。

## `coding_rag/agent/planner.py`

- `class` `WorkflowMode` (line 16) - 兼容规划接口的模式；普通请求始终交给统一 Agent。
- `class` `ChatClient` (line 23) - 聊天补全协议客户端。
- `method` `ChatClient.complete` (line 26) - 执行聊天补全请求。
- `class` `TaskPlan` (line 32) - 一次用户输入的分类结果和执行计划。
- `class` `AgentToolSpec` (line 43) - ReAct Agent 可调用工具的接口定义；当前只声明，不执行。
- `class` `AgentPlanConfig` (line 52) - 智能体规划配置数据类。
- `class` `AgentPlanRun` (line 70) - 智能体规划运行结果数据类。
- `class` `ReActAgentInterface` (line 85) - 向后兼容的 Agent 规划接口；执行统一由 LangGraph runtime 负责。
- `method` `ReActAgentInterface.__init__` (line 88) - 初始化接口，注入可用工具列表。
- `method` `ReActAgentInterface.plan` (line 92) - 委托任务分类生成执行计划。
- `method` `ReActAgentInterface.build_plan` (line 96) - 构建向后兼容的规划运行结果。
- `function` `classify_task` (line 106) - 只规划统一 Agent；具体回答、检索或修改由运行时模型决定。
- `function` `run_agent_plan_mode` (line 127) - 执行 Agent 计划模式：分析任务并生成 ReAct 工作流计划。
- `function` `build_agent_plan_trace` (line 175) - 构建规划追踪事件。
- `function` `collect_readonly_observations` (line 224) - 执行配置的只读工具并收集观察结果。
- `function` `default_agent_tools` (line 249) - 返回默认的Agent工具规格列表。
- `function` `build_agent_plan_messages` (line 300) - 构建Agent规划消息列表。
- `function` `format_tool_specs` (line 366) - 格式化工具规格为字符串。
- `function` `render_agent_plan_run` (line 375) - 渲染Agent计划运行结果。
- `function` `render_agent_plan_prompt` (line 398) - 渲染已存储的Agent规划提示消息。
- `function` `render_task_plan` (line 406) - 渲染任务计划。
- `function` `render_ask_prompt` (line 420) - 渲染ASK消息。

## `coding_rag/agent/runtime.py`

- `class` `ChatClient` (line 38) - 聊天补全协议。
- `method` `ChatClient.complete` (line 41) - 执行聊天补全请求。
- `class` `UnifiedRunConfig` (line 51) - 统一运行配置记录。
- `class` `ModelDecision` (line 81) - 模型决策记录。
- `class` `UnifiedRun` (line 94) - 统一执行结果记录。
- `function` `run_unified_query` (line 111) - 通过 LangGraph 运行普通 Agent 的统一公开入口。
- `function` `resume_unified_query` (line 147) - Resume one exact pending write after the user approves its fingerprint.
- `function` `persist_unified_run` (line 197) - Attach stable lifecycle metadata and persist a diagnostic copy after each return.
- `function` `build_hybrid_search_provider` (line 221) - 构建 BM25、元数据与结构评分融合的搜索提供者。
- `function` `build_repository_index_provider` (line 260) - Share the hybrid search snapshot with static read-only navigation.
- `function` `serialize_search_results` (line 278) - 将搜索结果序列化为字典列表。
- `function` `build_unified_messages` (line 302) - 构建统一模型消息列表。
- `function` `parse_model_decision` (line 359) - 解析模型响应的决策（answer/tool）。
- `function` `parse_json_object` (line 385) - 从原始文本提取 JSON 对象。
- `function` `build_unified_trace` (line 402) - 构建统一运行追踪记录，整合查询、配置和安全策略等元信息。
- `function` `build_runtime_tool_event` (line 449) - 创建运行时工具调用的追踪事件，附带耗时与压缩结果。
- `function` `compact_tool_result` (line 479) - 按最大字符数截断工具输出，用于减少模型上下文长度。
- `function` `bounded_tool_preview` (line 546) - Keep a JSON-serializable fallback within the exact model-context budget.
- `function` `collect_observed_ranges` (line 568) - Collect only complete source lines actually visible in the compacted tool output.
- `function` `failed_safety_details` (line 613) - 重新计算并序列化 run_command/apply_patch 的安全决策；其他工具返回 None。
- `function` `finish_unified_run` (line 631) - 最终完成统一运行，汇总追踪、状态、答案及统计信息。
- `function` `unified_run_to_dict` (line 697) - 序列化 UnifiedRun 为字典。
- `function` `decision_to_payload` (line 715) - 序列化 ModelDecision 为字典荷载。
- `function` `resolve_success_status` (line 727) - Normalize successful completion, preserving partial after tool failure.
- `function` `error_payload` (line 734) - 构建错误荷载字典。
- `function` `elapsed_ms` (line 743) - 计算自 started 以来的经过毫秒数。
- `function` `truncate_text` (line 748) - 截断文本并附加省略信息。
- `function` `optional_text` (line 755) - 将非空值转换为字符串，空值返回 None。

## `coding_rag/agent/safety.py`

- `class` `SafetyDecision` (line 67) - 不可变的安全决策记录，包含允许/拒绝结果及匹配的规则元数据。
- `class` `AgentSafetyPolicy` (line 77) - 不可变的安全限制与允许/拒绝策略配置。
- `class` `AgentSafetyGuard` (line 88) - Central safety policy for patch and command execution.
- `method` `AgentSafetyGuard.__init__` (line 91) - 使用提供的策略或默认策略初始化安全守卫。
- `method` `AgentSafetyGuard.check_command` (line 95) - 解析命令，并限制显式路径参数只能指向仓库内部。
- `method` `AgentSafetyGuard.check_patch` (line 145) - 检查补丁路径、safe-mode 及删除数量是否符合安全策略。
- `method` `AgentSafetyGuard.resolve_timeout` (line 188) - 将请求/默认超时值限制在策略和硬上限范围内。
- `method` `AgentSafetyGuard.truncate_output` (line 194) - 截断文本至策略上限并返回文本、截断标志和原始长度。
- `function` `parse_command` (line 204) - 使用 POSIX shlex 规则将命令字符串拆分为参数列表，shlex 语法错误转换为 AgentToolError。
- `function` `normalize_args` (line 212) - 规范化可执行文件别名并返回调整后的参数列表。
- `function` `matching_prefix` (line 225) - 返回第一个与 args 开头匹配的前缀（按给定顺序），无匹配则返回 None。
- `function` `command_rule_name` (line 236) - 将命令-规则令牌元组合并为其显示名称。
- `function` `find_executing_command_option` (line 241) - Reject allowlisted commands' flags that can execute tools or write undeclared files.
- `function` `find_outside_repo_command_path` (line 252) - Return the first explicit command path that can escape the repository.
- `function` `command_path_candidate` (line 280) - Extract path-like command operands, including ``--option=value`` forms.
- `function` `inspect_patch_paths` (line 299) - 解析 diff 头信息，返回变更路径列表和删除路径列表。
- `function` `strip_git_prefix` (line 333) - 去除diff路径的前缀 "a/" 或 "b/"。
- `function` `is_unsafe_patch_path` (line 340) - 检测绝对路径、路径遍历或其它不允许的补丁路径。
- `function` `is_env_path` (line 349) - 检测路径的 basename 是否为 .env 或以 .env. 开头。
- `function` `load_agent_safety_policy` (line 355) - Load an optional project safety policy while keeping denylist defaults.
- `function` `parse_rule_list` (line 381) - 验证并解析策略规则条目为命令令牌元组。
- `function` `merge_rule_lists` (line 400) - 按顺序合并各规则可迭代对象，丢弃空规则和重复项。

## `coding_rag/agent/snapshots.py`

- `class` `SnapshotStore` (line 20) - Cache declared file versions before a write and restore them on demand.
- `method` `SnapshotStore.__init__` (line 23) - Bind snapshots to one repository and an external cache root.
- `method` `SnapshotStore.create` (line 34) - Snapshot each declared repository-relative file before an approved write.
- `method` `SnapshotStore.diff` (line 77) - Build an operation-specific unified diff against one snapshot.
- `method` `SnapshotStore.rollback` (line 101) - Restore all cached versions and remove files created by the approved write.
- `method` `SnapshotStore._load` (line 125) - Load and validate a snapshot manifest bound to this repository.
- `function` `normalize_snapshot_paths` (line 142) - Normalize and deduplicate repository-relative file paths.
- `function` `resolve_repo_file` (line 162) - Resolve one declared file and keep it inside the bound repository.
- `function` `render_file_diff` (line 172) - Render a text diff, or a stable binary-change marker when decoding fails.

## `coding_rag/agent/workflow.py`

- `class` `ChatClient` (line 31) - 聊天补全协议接口。
- `method` `ChatClient.complete` (line 34) - 执行聊天补全并返回响应文本。
- `class` `CodeAgentConfig` (line 40) - 代码智能体工作流配置。
- `class` `TaskProfile` (line 59) - 分析后的任务画像。
- `class` `AgentRun` (line 68) - 工作流运行结果。
- `function` `run_code_agent` (line 87) - 执行一次轻量代码 Agent 工作流。
- `function` `retrieve_agent_context` (line 151) - 检索并召回任务相关的代码上下文。
- `function` `build_agent_plan` (line 205) - 根据任务画像和检索上下文构建修改计划。
- `function` `generate_agent_implementation` (line 246) - 通过 LLM 调用生成实现响应并返回。
- `function` `build_agent_messages` (line 259) - 构建发送给 LLM 的消息列表。
- `function` `render_agent_run` (line 298) - 渲染 Agent 运行摘要。
- `function` `agent_run_to_dict` (line 324) - 将 AgentRun 序列化为字典。
- `function` `build_agent_run_trace` (line 344) - 构建 Agent 运行追踪结构。
- `function` `append_agent_run_log` (line 383) - 追加 Agent 单次运行日志；不同于长期记忆，日志保留完整运行环节。
- `function` `workflow_as_dicts` (line 394) - 将 Agent 工作流步骤序列化为字典列表。
- `function` `build_agent_step_trace` (line 399) - 构建 Agent 各步骤的追踪事件列表。
- `function` `render_agent_trace` (line 499) - 渲染 Agent 追踪为 Markdown 文本。
- `function` `summarize_trace_artifacts` (line 510) - 汇总追踪产物为键值摘要字符串。
- `function` `build_memory_summary` (line 533) - 根据任务和搜索结果构建摘要文本。
- `function` `prioritized_result_files` (line 540) - 返回去重并按优先级排序（非测试文件优先）的结果文件路径列表。
- `function` `build_memory_decisions` (line 546) - 根据配置、LLM 使用标志和任务画像构建决策描述字符串列表。
- `function` `build_offline_implementation_note` (line 556) - 构建离线模式下的实现说明，提示用户未启用 LLM 且提供后续操作建议。
- `function` `format_memories` (line 564) - 将 Agent 记忆列表格式化为可用于提示上下文的文本。
- `function` `dedupe` (line 574) - 对字符串列表进行稳定去重，保留首次出现的顺序。
- `function` `analyze_task` (line 583) - 根据关键词对任务进行分类分析，返回任务画像。
- `function` `enrich_task_profile` (line 618) - 利用仓库搜索结果丰富任务画像信息。
- `function` `build_recommended_checks` (line 629) - 根据任务类型和文件列表构建推荐的验证命令。
- `function` `test_module_name` (line 650) - 将测试文件路径转换为 Python 模块名。

## `coding_rag/evaluation/agent_patch_provider.py`

- `class` `AgentCandidateRejected` (line 34) - Represent a private-policy rejection without exposing candidate content.
- `class` `AgentCandidateProvider` (line 38) - Run one ChatClient in a disposable copy and return only its cumulative diff.
- `method` `AgentCandidateProvider.__init__` (line 47) - Bind a generic client and strict per-task execution limits.
- `method` `AgentCandidateProvider.scorecard_metadata` (line 95) - Return bounded aggregate metadata without prompts, source, patches, or answers.
- `method` `AgentCandidateProvider.provide` (line 118) - Run the real graph in a second workspace and return its validated cumulative diff.
- `method` `AgentCandidateProvider._validate_private_patch` (line 194) - Apply evaluator-private scope and deletion rules before granting approval.
- `method` `AgentCandidateProvider._record_run` (line 210) - Aggregate only sanitized counters from a completed or stopped graph run.
- `method` `AgentCandidateProvider._reject_links` (line 220) - Reject links before copying so the candidate cannot reach outside its source tree.
- `method` `AgentCandidateProvider._prepare_git_index` (line 227) - Stage the faulted allowed files as the private baseline for cumulative diffing.
- `method` `AgentCandidateProvider._cumulative_diff` (line 237) - Read a bounded diff while keeping Git diagnostics out of patch bytes.
- `method` `AgentCandidateProvider._workspace_metadata` (line 262) - Capture best-effort metadata outside the evaluator-owned Git directory.
- `method` `AgentCandidateProvider._changed_workspace_paths` (line 277) - Return added, removed, or metadata-changed paths after Agent execution.
- `function` `_bounded_counter` (line 285) - Accept only small nonnegative integer counters from trace summaries.

## `coding_rag/evaluation/agent_runtime.py`

- `function` `_decision` (line 31) - Encode one unified-runtime decision without hidden dataset fields.
- `function` `_last_observation` (line 43) - Parse only the runtime-provided Observation appended after a tool call.
- `class` `ScriptedRuntimePolicy` (line 61) - A deterministic policy that sees only the query and runtime observations.
- `method` `ScriptedRuntimePolicy.__init__` (line 66) - Bind the original query; no case metadata or gold paths are accepted.
- `method` `ScriptedRuntimePolicy.complete` (line 73) - Search, choose from returned rows, read, then cite the observed range.
- `function` `_normalize` (line 108) - Normalize one repository-relative path for metric comparison.
- `function` `_ranked_files` (line 113) - Extract ordered unique files from the real search observation.
- `function` `_selected_file` (line 129) - Return the file chosen by the policy's read_file decision.
- `function` `_read_file` (line 137) - Return the path successfully observed from read_file.
- `function` `evaluate_case` (line 146) - Run one real unified graph case; gold is consulted only after the run returns.
- `function` `_case_index_lifecycle` (line 218) - Read the repository-index lifecycle recorded for a measured search.
- `function` `_percentile` (line 226) - Compute one linearly interpolated percentile.
- `function` `_aggregate` (line 243) - Aggregate contract quality, index reuse, steps, and latency.
- `function` `_group` (line 256) - Aggregate case rows by one categorical dimension.
- `function` `evaluate_agent_runtime_suite` (line 264) - Warm each repo without gold, then run every case against memory-resident indexes.
- `function` `_raw` (line 326) - Read one dotted scorecard field, returning None if absent.
- `function` `compare_agent_runtime_scorecards` (line 336) - Enforce identity/config compatibility, quality floors, and stable case latency.
- `function` `gate_exit_code` (line 361) - Map pass, regression, and incompatibility to 0, 1, and 2.

## `coding_rag/evaluation/call_path_runtime.py`

- `class` `CallPathScriptedPolicy` (line 19) - Exercise search → static trace → two reads with no gold or model access.
- `method` `CallPathScriptedPolicy.__init__` (line 26) - Bind only the user's question and its first explicit callable name.
- `method` `CallPathScriptedPolicy._target_edge` (line 36) - Choose one query-mentioned call; ties and unlocated edges fail closed.
- `method` `CallPathScriptedPolicy._read_arguments` (line 81) - Limit evidence reads to forty source lines per file.
- `method` `CallPathScriptedPolicy.complete` (line 85) - Make the next tool decision from the original query and last observation.
- `function` `observed_line` (line 140) - Require non-empty source text at the expected path and numbered line.
- `function` `evaluate_case` (line 151) - Score real graph observations and citations only after the policy finishes.

## `coding_rag/evaluation/gate.py`

- `class` `GateResult` (line 9) - Result of comparing one scorecard with its compatible baseline.
- `function` `_value` (line 30) - Read a numeric scorecard value using a dotted field path.
- `function` `_raw_value` (line 38) - Read any nested scorecard value using a dotted field path.
- `function` `compare_scorecards` (line 48) - Check dataset compatibility and all configured regression thresholds.

## `coding_rag/evaluation/multirepo.py`

- `class` `RepositorySpec` (line 29) - One validated pinned repository from the portable manifest.
- `class` `MultiRepoCase` (line 43) - One source-grounded retrieval case assigned to a manifest repository.
- `class` `MultiRepoSuite` (line 55) - Validated repositories, cases, and portable input identities.
- `class` `MultiRepoValidationError` (line 66) - Raised with every actionable manifest or evalset validation finding.
- `method` `MultiRepoValidationError.__init__` (line 69) - Preserve individual validation findings and a concise combined message.
- `function` `normalize_remote` (line 75) - Normalize common HTTPS, SSH, and scp-style Git remotes for comparison.
- `function` `_git` (line 90) - Return stripped Git output or None without mutating the checkout.
- `function` `_safe_relative` (line 98) - Return a normalized safe relative POSIX path, otherwise None.
- `function` `_read_jsonl` (line 107) - Read a JSONL file and require every non-empty line to be an object.
- `function` `load_and_validate_suite` (line 120) - Strictly validate inputs, checkout identities, confinement, and gold files.
- `function` `_percentile` (line 243) - Return a linearly interpolated percentile.
- `function` `_aggregate` (line 255) - Aggregate retrieval rows into standard metrics and bad-case identities.
- `function` `_group` (line 273) - Aggregate rows by one portable case dimension.
- `function` `evaluate_suite` (line 281) - Build one index per repository and aggregate source-grounded metrics.
- `function` `_raw` (line 347) - Read a nested scorecard value, returning None when absent.
- `function` `compare_multirepo_scorecards` (line 357) - Check strict experiment compatibility and configured regression thresholds.
- `function` `gate_exit_code` (line 386) - Map compatible pass/regression/incompatibility to process exit codes 0/1/2.

## `coding_rag/evaluation/patch_outcome.py`

- `class` `PatchOutcomeValidationError` (line 47) - Report all actionable task-manifest validation errors.
- `method` `PatchOutcomeValidationError.__init__` (line 50) - Preserve individual findings as well as a concise combined message.
- `class` `PatchTask` (line 57) - A real pinned-source regression whose oracle is excluded from candidate input.
- `class` `PatchRepository` (line 77) - One locally prepared, immutable upstream checkout identity.
- `class` `PatchSuite` (line 91) - Portable task suite plus stable manifest and fixture identities.
- `class` `CandidateRequest` (line 103) - The complete provider-facing request; deliberately excludes gold patch data.
- `class` `CandidateProvider` (line 113) - Return one unified diff using only public task input and the isolated checkout.
- `method` `CandidateProvider.provide` (line 118) - Produce a candidate unified diff without receiving oracle fields.
- `class` `StaticPatchProvider` (line 123) - Offline provider backed by caller-supplied task-id to patch mappings.
- `method` `StaticPatchProvider.__init__` (line 128) - Copy caller-owned patch mappings so later mutation cannot affect a run.
- `method` `StaticPatchProvider.provide` (line 132) - Return the mapped patch or an explicit no-op candidate.
- `class` `OraclePatchProvider` (line 137) - Self-test-only provider; gold paths stay private to this explicit adapter.
- `method` `OraclePatchProvider.__init__` (line 142) - Bind private oracle paths for explicit evaluator self-checks only.
- `method` `OraclePatchProvider.provide` (line 146) - Read the oracle patch without exposing its path through CandidateRequest.
- `function` `safe_relative_path` (line 151) - Accept only actual, nonempty, confined relative POSIX paths.
- `function` `file_sha256` (line 165) - Return a lowercase SHA-256 digest for one file.
- `function` `tree_identity` (line 170) - Hash relative paths and bytes, rejecting symlinks and special files.
- `function` `_confined_file` (line 187) - Resolve a validated relative path and retain workspace confinement.
- `function` `_git_text` (line 197) - Run a bounded read-only Git query and return stripped text on success.
- `function` `_git_blob` (line 206) - Read one pinned Git blob without consulting working-tree contents.
- `function` `_normalized_remote` (line 215) - Normalize common HTTPS and SSH Git remote spellings for comparison.
- `function` `load_and_validate_patch_suite` (line 230) - Strictly validate portable schema, identities, paths, tests, and oracle files.
- `function` `_path_list` (line 376) - Validate, normalize, and deduplicate one required relative-path list.
- `function` `_valid_test_command` (line 391) - Accept only fixed unittest argv with safe dotted test targets.
- `function` `_tree_files` (line 400) - Snapshot regular-file content hashes outside evaluator-owned Git metadata.
- `function` `_tree_metadata` (line 413) - Best-effort trusted-provider side-effect detection without full-tree rehashing.
- `function` `_run` (line 425) - Run argv without a shell while continuously draining output into a hard cap.
- `function` `_test_environment` (line 489) - Force local-only proxy settings and load an evaluator-owned socket guard.
- `function` `_ensure_patch_targets_not_symlinks` (line 529) - Reject a patch target if any existing component is a symbolic link.
- `function` `validate_candidate_patch_paths` (line 539) - Reject every path/header form except consistent in-place edits to allowed files.
- `function` `_invalid_harness_row` (line 580) - Return a score-zero row without invoking a candidate on an invalid harness state.
- `function` `materialize_pinned_archive` (line 596) - Safely materialize only trusted roots from the exact pinned Git commit.
- `function` `_extract_validated_archive` (line 624) - Validate all members before manually extracting regular files.
- `function` `_apply_patch` (line 667) - Check and apply a validated patch through bounded, shell-free Git argv.
- `function` `evaluate_patch_task` (line 684) - Evaluate one candidate only inside an evaluator-owned temporary checkout.
- `function` `_percentile` (line 809) - Return a linearly interpolated percentile or zero for no observations.
- `function` `evaluate_patch_suite` (line 821) - Evaluate every task and return a portable quality scorecard.
- `function` `patch_scorecard_identity` (line 854) - Build fields that must match before any candidate or paid evaluation starts.
- `function` `validate_patch_gate_preflight` (line 892) - Reject malformed or incompatible gate inputs before candidate evaluation.
- `function` `json_safe_candidate_metadata` (line 907) - Keep only the fixed, non-sensitive Agent candidate metadata contract.
- `function` `_raw` (line 941) - Read one dotted scorecard field, returning None when absent.
- `function` `compare_patch_scorecards` (line 951) - Return incompatibility, regression, or pass without gating wall-clock latency.
- `function` `validate_patch_gate_policy` (line 982) - Require a closed, finite, nonnegative regression policy schema.
- `function` `gate_exit_code` (line 1006) - Map pass, regression, and incompatibility to exit codes 0, 1, and 2.

## `coding_rag/evaluation/retrieval.py`

- `class` `StructuralRanker` (line 26) - Extension point for a future graph-ranked A/B implementation.
- `method` `StructuralRanker.score` (line 31) - Return one structural score per chunk for an experimental ranker.
- `class` `EvalCase` (line 37) - Normalized retrieval evaluation case.
- `class` `EvalsetValidation` (line 47) - All structural and repository-path validation findings for an evalset.
- `method` `EvalsetValidation.valid` (line 57) - Return whether the evalset is non-empty and has no findings.
- `method` `EvalsetValidation.to_dict` (line 61) - Serialize findings together with the computed validity flag.
- `class` `RetrievalEvalConfig` (line 67) - Stable parameters that define one retrieval experiment.
- `method` `RetrievalEvalConfig.public_dict` (line 80) - Return portable config metadata without an absolute repository path.
- `class` `PreparedRetrievalPipeline` (line 96) - Repository files, chunks, and index shared by every benchmark query.
- `method` `PreparedRetrievalPipeline.build` (line 105) - Scan, chunk, and index the configured repository exactly once.
- `method` `PreparedRetrievalPipeline.query` (line 120) - Run search, neighbor recall, filtering, and context compaction.
- `function` `_raw_eval_items` (line 148) - Load raw cases from a JSON array or JSONL file.
- `function` `load_evalset` (line 160) - Load JSON/JSONL aliases into normalized evaluation cases.
- `function` `validate_evalset` (line 177) - Validate identities, questions, gold lists, and safe existing gold paths.
- `function` `_normalized` (line 210) - Normalize a result or gold path into a portable comparison form.
- `function` `_ranked_gold` (line 221) - Return binary relevance by rank while counting each gold file once.
- `function` `_percentile` (line 235) - Compute a linearly interpolated percentile for deterministic summaries.
- `function` `evaluate_retrieval` (line 247) - Evaluate all cases and aggregate quality, cost, and bad-case metrics.
- `function` `dataset_sha256` (line 311) - Return the byte-level SHA-256 identity of an evaluation set.
- `function` `repo_fingerprint` (line 316) - Hash indexed relative paths and contents into a portable repository identity.

## `coding_rag/evaluation/scorecard.py`

- `function` `_git` (line 14) - Run a read-only Git query and return unknown when unavailable.
- `function` `build_capability_scorecard` (line 22) - Combine retrieval, efficiency, agent, and experiment metadata.

## `coding_rag/learning/models.py`

- `class` `StrictLearningModel` (line 45) - 拒绝额外字段和隐式类型转换的 Learning Mode 模型基类。
- `class` `EvidenceRef` (line 55) - 把一个明确事实绑定到本次只读检索已经观察到的源码范围。
- `method` `EvidenceRef.normalize_path` (line 65) - 规范仓库相对路径，并拒绝绝对路径与父目录穿越。
- `method` `EvidenceRef.validate_line_range` (line 70) - 保证证据范围的结束行不早于开始行。
- `class` `ProjectComponent` (line 77) - 项目中的一个可解释组件及其源码依据。
- `class` `ProjectProfile` (line 85) - 面向初学者的项目目标、技术栈、入口和组件画像。
- `method` `ProjectProfile.validate_unique_components` (line 96) - 拒绝大小写不同但含义相同的重复组件名称。
- `class` `ReproductionStep` (line 104) - 从空目录复现项目时的一步教学任务。
- `method` `ReproductionStep.validate_dependency_shape` (line 125) - 拒绝空依赖、重复依赖以及不稳定的步骤标识。
- `method` `ReproductionStep.normalize_created_files` (line 139) - 把待创建文件限制为新项目内的相对路径。
- `method` `ReproductionStep.validate_optimization_question` (line 148) - 确保优化思考字段确实以问题形式呈现。
- `method` `ReproductionStep.reject_self_dependency` (line 155) - 单步层面拒绝依赖自身，完整顺序由 LearningPlan 校验。
- `class` `LearningPlan` (line 162) - 一次项目拆解生成的项目画像和四到六步复现路线。
- `method` `LearningPlan.validate_step_order` (line 169) - 保证步骤标识唯一，并且依赖只指向已经出现的前序步骤。
- `class` `StepSubmission` (line 182) - 学习者为当前步骤提交的实现说明和测试输出。
- `class` `StepReviewFinding` (line 195) - 一次审查对单条验收要求作出的判断。
- `class` `StepReview` (line 203) - 只基于学习者报告证据形成的结构化步骤审查。
- `method` `StepReview.validate_review_conclusion` (line 217) - 保证逐项判断、证据充分性和最终结论彼此一致。
- `class` `RevisionFeedback` (line 243) - 向学习者公开的最小修改反馈。
- `class` `StepReviewRecord` (line 253) - 保存一次步骤审查及其不含原始提交文本的审计元数据。
- `class` `StepReflectionRecord` (line 266) - 保存学习者完成一个步骤后的反思。
- `class` `LearningSessionEvent` (line 283) - 会话的一次公开状态转换记录。
- `class` `LearningSession` (line 294) - 由调用方保存并在每轮请求中回传的完整学习会话。
- `method` `LearningSession.validate_session_id` (line 311) - 会话标识保持字符串形态，同时必须是合法 UUID。
- `method` `LearningSession.validate_session_state` (line 320) - 拒绝无法由公开 action 流程产生的会话状态组合。
- `class` `StartLearningSessionRequest` (line 473) - 启动一条新的学习路线。
- `class` `SubmitLearningSessionRequest` (line 481) - 提交当前步骤的学习者报告证据。
- `class` `ReflectLearningSessionRequest` (line 489) - 记录反思，并在安全时推进至下一步骤。
- `class` `LearningSessionError` (line 505) - 学习会话入口对调用方公开的稳定错误结构。
- `class` `LearningSessionResult` (line 512) - 一次 start、submit 或 reflect 调用的公开结果。
- `method` `LearningSessionResult.validate_result_state` (line 527) - 保证结果字段和会话阶段一致，失败时仍可安全重试。
- `class` `LearningError` (line 604) - Learning Mode 对调用方公开的稳定错误结构。
- `class` `LearningWorkflowResult` (line 611) - Learning Mode 一次启动请求的公开、可 JSON 序列化结果。
- `method` `LearningWorkflowResult.validate_result_state` (line 625) - 避免失败结果泄露未校验计划，并保证成功结果字段完整。
- `function` `normalize_repo_relative_path` (line 644) - 返回稳定 POSIX 相对路径，并拒绝冒号、绝对路径和 ``..``。

## `coding_rag/learning/session.py`

- `class` `LearningSessionDependencies` (line 67) - 保存会话图需要、但绝不能进入 JSON state 的运行依赖。
- `method` `LearningSessionDependencies.__post_init__` (line 80) - 拒绝无法提供真实完整性保护的短密钥或缺失依赖。
- `class` `LearningSessionState` (line 96) - 一次 action 调用使用的纯 JSON LangGraph 状态。
- `function` `build_learning_session_graph` (line 114) - 构建统一分发 start、submit、reflect 的单回合 LangGraph。
- `function` `run_learning_session` (line 576) - 验证请求和 session 完整性，再运行一次 action 到下一个等待点。
- `function` `start_learning_session` (line 651) - 创建项目拆解路线并停在第一步提交等待点。
- `function` `submit_learning_session` (line 668) - 提交本步实现证据，并停在修改或优化思考等待点。
- `function` `reflect_learning_session` (line 688) - 记录当前步骤的优化回答，并推进到下一步或完成。
- `function` `build_learning_session_input` (line 704) - 把已经结构校验、签名校验的请求转换为纯 JSON 图状态。
- `function` `build_step_review_prompt` (line 732) - 构造只评估学习者报告证据、从不执行文本的 LangChain prompt。
- `function` `issue_signed_session` (line 755) - 规范化 session 后用服务端 HMAC 密钥签发完整性 token。
- `function` `session_has_completion_headroom` (line 773) - 模拟每个剩余步骤一次通过和最短反思，拒绝会被体积上限永久卡住的状态。
- `function` `update_signed_session` (line 868) - 在每次合法状态转换后重新校验全部不变量并重新签名。
- `function` `compute_learning_session_token` (line 879) - 对排除 token 的 canonical JSON 计算 HMAC-SHA256。
- `function` `verify_learning_session` (line 894) - 使用常量时间比较拒绝结构合法但内容被修改的 session。
- `function` `apply_review_to_session` (line 903) - 记录一次审查摘要，保留原始提交隐私，并转换到下一个等待点。
- `function` `build_precheck_review` (line 961) - 为信息不足的提交生成确定性失败记录，不消耗模型调用。
- `function` `validate_review_against_step` (line 985) - 要求模型 findings 恰好覆盖当前步骤的每一个验收项。
- `function` `normalize_review_for_storage` (line 992) - 只保留模型判断，并按可信验收项重建具体文案以防敏感文本回显。
- `function` `bounded_review_text` (line 1051) - 在保留验收项辨识度的同时，将确定性反馈限制到 ReviewText 上限。
- `function` `submission_digest` (line 1061) - 用域分离 HMAC 保存提交指纹，避免低熵文本被离线字典恢复。
- `function` `parse_learning_session_request` (line 1086) - 用判别联合解析三类请求，拒绝额外字段和隐式类型转换。
- `function` `classify_request_validation_error` (line 1099) - 区分普通请求错误和无法建立可信边界的 session 结构错误。
- `function` `recover_verified_request_session` (line 1118) - 仅在外层请求无效时恢复结构和 HMAC 都可信的嵌套 session。
- `function` `request_session` (line 1135) - 从 submit/reflect 请求取出 session，start 返回 None。
- `function` `require_state_session` (line 1142) - 读取进入图前已完成结构和签名校验的 session。
- `function` `session_payload` (line 1149) - 把公开 session 转换为请求可携带的 JSON 字典。
- `function` `session_failure_update` (line 1159) - 返回脱敏的回合失败增量，保留 state 中原来的合法 session。
- `function` `trace_error` (line 1186) - 将公开错误转换为统一 trace 的稳定错误结构。
- `function` `session_result_from_state` (line 1198) - 将纯图状态映射为前端可直接消费的稳定结果。
- `function` `build_immediate_failure` (line 1223) - 为图外请求/签名校验失败生成不回显不可信 session 的结果。
- `function` `build_verified_session_failure` (line 1267) - 将意外图错误收敛为失败结果，并仅回显已通过 HMAC 的旧 session。
- `function` `session_view` (line 1302) - 从可信 session 唯一派生当前步骤和下一 action。
- `function` `build_learning_session_trace` (line 1312) - 记录路线和权限摘要，不保存密钥、原始提交或模型原文。
- `function` `raw_action` (line 1356) - 只在错误报告中保留已知 action，不信任其他输入值。
- `function` `request_within_size_limit` (line 1372) - 在深层 Pydantic 校验前限制无持久化 session 的传输体积。
- `function` `json_payload_within_limit` (line 1385) - 用与 HMAC 相同的严格 JSON 规则限制传输数据体积。

## `coding_rag/learning/workflow.py`

- `class` `LearningChatClient` (line 41) - 描述 RepoPilot 现有 ``complete(messages)`` 聊天客户端协议。
- `method` `LearningChatClient.complete` (line 44) - 根据 system/user 消息返回一次模型文本。
- `class` `LearningWorkflowConfig` (line 53) - Learning Mode 的仓库、检索和上下文上限配置。
- `method` `LearningWorkflowConfig.__post_init__` (line 63) - 在访问仓库前拒绝无效检索和切片参数。
- `class` `LearningState` (line 77) - 三个教学节点共享的纯 JSON 数据状态。
- `class` `LearningWorkflowFailure` (line 94) - 携带稳定错误码的预期 Learning Mode 失败。
- `method` `LearningWorkflowFailure.__init__` (line 97) - 保存可公开的错误码和已脱敏消息。
- `function` `build_learning_input` (line 104) - 构造字段完整且可以直接 JSON 序列化的初始图状态。
- `function` `build_learning_graph` (line 130) - 构建 ``collect -> analyze -> present`` 的独立只读 Learning Graph。
- `function` `run_learning_workflow` (line 309) - 运行一次项目画像与复现路线生成，并返回稳定公开结果。
- `function` `build_learning_prompt` (line 344) - 构建包含不可信源码边界和 Pydantic 格式说明的 LangChain Prompt。
- `function` `build_learning_executor` (line 370) - 构造只在证据节点闭包内使用的 RepoPilot 只读执行器。
- `function` `collect_repository_evidence` (line 397) - 搜索项目结构并用安全 read_file 固化实际路径、行号和文本。
- `function` `normalize_search_payload` (line 467) - 把完整 Hybrid Search 或只读回退结果规范为统一列表。
- `function` `compact_evidence` (line 489) - 按完整代码行压缩上下文，并同步缩小允许引用的结束行。
- `function` `build_evidence_catalog` (line 531) - 建立精确相对路径到文件总行数和已观察范围的可信目录。
- `function` `merge_ranges` (line 545) - 合并重叠或相邻的已观察范围，保留 JSON 友好的列表形式。
- `function` `validate_plan_evidence` (line 556) - 要求每条模型引用精确落在本次实际发送给模型的源码范围内。
- `function` `iter_plan_evidence` (line 585) - 依次产出入口、组件和复现步骤中的全部源码引用。
- `function` `evidence_ranges` (line 594) - 从证据中移除源码文本，只保留可审计的路径和范围元数据。
- `function` `messages_to_payload` (line 607) - 把 LangChain Prompt 消息转换为现有聊天客户端接受的字典。
- `function` `message_role` (line 617) - 将 LangChain 消息类型映射为 OpenAI 兼容角色。
- `function` `failure_update` (line 628) - 生成不会携带原始模型文本或本地异常详情的失败状态增量。
- `function` `default_learning_safety_flags` (line 661) - 返回即使执行器初始化失败也成立的 Learning Mode 权限边界。
- `function` `learning_executor_flags` (line 671) - 把实际执行器策略转换为可序列化、可审计的只读标志。
- `function` `learning_error_type` (line 678) - 把稳定错误码映射为适合 trace 阅读的错误类型。
- `function` `build_learning_trace` (line 689) - 复用 RepoPilot 统一 trace 外壳记录检索、规划与展示路线。
- `function` `required_string` (line 729) - 从工具结果读取必需非空字符串，否则转为稳定仓库错误。
- `function` `required_positive_int` (line 737) - 从工具结果读取严格正整数，拒绝 bool 和字符串数字。
- `function` `optional_float` (line 745) - 仅保留真实数值搜索分数，避免不可信 provider 注入其他类型。
- `function` `elapsed_ms` (line 754) - 把节点单调时钟耗时转换为非负毫秒整数。

## `coding_rag/rag/answer_generator.py`

- `class` `AnswerGenerator` (line 29) - 基于RAG检索结果的答案生成器
- `method` `AnswerGenerator.__init__` (line 32) - 初始化答案生成器 Args: config: LLM客户端配置 mode: 生成模式 max_context_chars: 发送给LLM的最大上下文字符数
- `method` `AnswerGenerator.generate` (line 49) - 基于检索结果生成答案 Args: query: 用户查询 results: 检索结果列表 Returns: 生成的答案 Raises: RuntimeError: LLM请求失败
- `method` `AnswerGenerator.set_mode` (line 89) - 设置生成模式 Args: mode: 新的生成模式
- `method` `AnswerGenerator.set_max_context_chars` (line 97) - 设置最大上下文字符数 Args: max_chars: 最大上下文字符数
- `function` `build_generator` (line 106) - 构建答案生成器 Args: provider: LLM提供者名称 model: 覆盖预设的模型名称 base_url: 覆盖预设的API基础URL api_key_env: 覆盖预设的API密钥环境变量名 timeout: LLM请求超时时间（秒） max_tokens: 最大输出token数 temperature: 温度参数 mode: 生成模式 max_context_chars: 最大上下文字符数 Returns: 配置好的AnswerGenerator实例 Raises: ValueError: 缺少必要的配置

## `coding_rag/rag/ask.py`

- `class` `RetrievalConfig` (line 22) - 统一 Agent 的代码检索配置。
- `function` `retrieve_code` (line 47) - 执行代码检索流水线。
- `function` `resolve_max_recall_results` (line 157) - 解析召回阶段的最大结果数。

## `coding_rag/rag/citation_validator.py`

- `class` `Citation` (line 21) - 表示解析后的代码引用，包含文件路径与行范围。
- `class` `CitationValidationResult` (line 31) - 保存引用校验结果，包含有效引用、无效引用和缺失标记。
- `method` `CitationValidationResult.has_issues` (line 39) - 当存在无效引用或缺失引用时返回 True。
- `function` `extract_citations` (line 44) - 从回答文本中通过正则提取 path:start-end 格式的引用。
- `function` `validate_answer_citations` (line 61) - 根据可用检索结果范围校验回答中的引用。
- `function` `validate_answer_citations_against_ranges` (line 70) - Validate citations against code ranges observed through Agent tools.
- `function` `citation_is_supported` (line 90) - 检查引用行范围是否被任一可用范围包含。
- `function` `paths_match` (line 111) - Accept an exact path or a relative citation suffix of an observed absolute path.
- `function` `normalize_path` (line 120) - 归一化文件路径以支持跨平台大小写不敏感比较。
- `function` `is_refusal_answer` (line 125) - 检测回答是否表明无法找到相关代码。
- `function` `append_citation_validation_report` (line 135) - 当存在校验问题时追加中文引用校验报告段落。

## `coding_rag/rag/judge.py`

- `class` `LLMJudge` (line 32) - 基于检索结果调用 LLM 做相关性判断和回答。
- `method` `LLMJudge.judge` (line 38) - 基于检索结果调用 LLM 判断相关性并生成回答。
- `function` `build_judge_messages` (line 44) - 把用户问题和检索结果拼成 legacy judge 模式使用的消息列表。
- `function` `format_results_for_prompt` (line 70) - 把检索结果格式化为带路径、行号和代码块的 prompt 上下文。

## `coding_rag/rag/llm_client.py`

- `class` `LLMRetryPolicy` (line 35) - Strict, testable retry limits for explicitly transient transport failures.
- `method` `LLMRetryPolicy.__post_init__` (line 43) - Reject unbounded or nonsensical retry settings at configuration time.
- `class` `LLMCallObservation` (line 58) - Sanitized metadata for one logical LLM call across all retry attempts.
- `function` `normalize_llm_call_observation` (line 71) - Return a bounded fixed-schema observation, rejecting unsafe or malformed data.
- `class` `LLMRequestError` (line 111) - Safe public error that carries retry classification without response details.
- `method` `LLMRequestError.__init__` (line 114) - Initialize a sanitized failure with stable classification metadata.
- `class` `ProviderPreset` (line 122) - LLM 服务提供商预设配置 Attributes: name: 提供商名称 base_url: API 基础 URL model: 默认模型名称 api_key_env: API 密钥环境变量名
- `class` `LLMConfig` (line 168) - LLM 客户端配置 Attributes: provider: 提供商名称 base_url: API 基础 URL model: 模型名称 api_key: API 密钥 api_key_env: API 密钥环境变量名 timeout: 请求超时时间（秒） max_tokens: 最大输出 token 数 temperature: 温度参数，控制生成随机性 retry_policy: 有界暂时性故障重试策略
- `class` `OpenAICompatibleChatClient` (line 194) - OpenAI 兼容的聊天补全 API 客户端 使用轻量级 HTTP 客户端实现，无需额外依赖
- `method` `OpenAICompatibleChatClient.__init__` (line 201) - 初始化客户端 Args: config: LLM 配置对象
- `method` `OpenAICompatibleChatClient.observations` (line 228) - Return a stable snapshot of bounded, sanitized call observations.
- `method` `OpenAICompatibleChatClient.last_observation` (line 234) - Return the latest sanitized logical-call observation, if any.
- `method` `OpenAICompatibleChatClient.take_call_observation` (line 239) - Consume this thread's latest logical-call observation exactly once.
- `method` `OpenAICompatibleChatClient.complete` (line 246) - 发送聊天补全请求 Args: messages: 消息列表，每个消息包含 role 和 content Returns: LLM 返回的文本内容 Raises: RuntimeError: 请求失败时抛出
- `method` `OpenAICompatibleChatClient._request_once` (line 304) - Perform one HTTP attempt and translate failures into safe classifications.
- `method` `OpenAICompatibleChatClient._retry_delay` (line 357) - Calculate bounded exponential backoff with injectable symmetric jitter.
- `method` `OpenAICompatibleChatClient._record_observation` (line 364) - Store, log, and optionally emit only the sanitized observation payload.
- `function` `classify_url_error` (line 378) - Classify a URL failure without copying its possibly sensitive reason text.
- `function` `classify_transport_reason` (line 383) - Retry only well-known timeout and interrupted-network exception classes.
- `function` `build_llm_config` (line 396) - 构建 LLM 配置 支持通过参数、环境变量或预设来构建配置 Args: provider: 提供商名称 model: 覆盖预设的模型名称 base_url: 覆盖预设的 API 基础 URL api_key_env: 覆盖预设的 API 密钥环境变量名 timeout: 请求超时时间（秒） max_tokens: 最大输出 token 数 temperature: 温度参数 retry_max_attempts: 单次逻辑调用允许的最大 HTTP 尝试次数 retry_base_delay_seconds: 指数退避初始秒数 retry_max_delay_seconds: 单次退避严格上限秒数 retry_jitter_ratio: 对称抖动比例 Returns: 构建好的 LLMConfig 对象 Raises: ValueError: 缺少必要配置时抛出
- `function` `available_provider_names` (line 491) - 获取所有可用的提供商名称列表
- `function` `chat_completions_url` (line 496) - 构建聊天补全 API 的完整 URL Args: base_url: API 基础 URL Returns: 完整的聊天补全 API URL
- `function` `extract_message_content` (line 512) - 从 LLM 响应中提取消息内容 Args: payload: LLM 返回的 JSON 响应 Returns: 提取的消息内容 Raises: RuntimeError: 响应格式错误时抛出
- `function` `first_value` (line 549) - 返回第一个非 None 的值 Args: *values: 可选值列表 Returns: 第一个非 None 的值，如果没有则返回 None

## `coding_rag/rag/prompt.py`

- `class` `GenerationMode` (line 16) - 生成模式枚举
- `function` `build_judge_user_prompt` (line 49) - 构建搜索结果判断的用户提示词
- `function` `build_code_understand_user_prompt` (line 87) - 构建代码理解的用户提示词
- `function` `build_code_generate_user_prompt` (line 123) - 构建代码生成的用户提示词
- `function` `build_leetcode_user_prompt` (line 160) - 构建 LeetCode 问题解答的用户提示词
- `function` `build_api_user_prompt` (line 198) - 构建 API 使用的用户提示词
- `function` `get_system_prompt` (line 212) - 根据生成模式获取系统提示词 Args: mode: 生成模式 Returns: 对应模式的系统提示词字符串
- `function` `build_user_prompt` (line 232) - 根据生成模式构建用户提示词
- `class` `ContextBlock` (line 251) - 发送给 LLM 的合并后证据块
- `class` `_PendingContextBlock` (line 265) - 构建中的上下文块，用于合并同一文件中相邻或重叠的检索结果片段。
- `function` `format_results_as_context` (line 277) - 将检索结果格式化为上下文字符串 Args: results: 检索结果列表，每个结果应包含 chunk、score、source 属性 max_context_chars: 最大上下文字符数 Returns: 格式化的上下文字符串
- `function` `format_results_as_context_with_ranges` (line 295) - Render the existing prompt format and record complete code lines actually shown.
- `function` `compact_results_for_context` (line 348) - 合并同文件中连续或重叠的检索块，减少上下文重复
- `function` `merge_result_into_block` (line 398) - 将单个检索结果合并到已有的上下文块中，去重重叠行并更新分数、来源等信息。

## `coding_rag/rag/trace.py`

- `function` `build_retrieval_trace` (line 18) - 构建单次 RAG 检索轨迹。
- `function` `build_trace_run` (line 69) - Build the run-level envelope shared by all trace producers.
- `function` `build_trace_event` (line 96) - Build one normalized step/tool event.
- `function` `build_unified_retrieval_events` (line 120) - Convert legacy retrieval stages into normalized events.
- `function` `build_unified_retrieval_trace` (line 136) - Build a normalized trace around already-serialized retrieval stages.
- `function` `build_agent_events` (line 164) - Convert legacy Agent step trace rows into normalized events.
- `function` `build_tool_event` (line 180) - Convert executor tool results into a normalized event.
- `function` `normalize_event_status` (line 234) - 将 done/drafted/planned 映射为 success，保留 skipped，否则返回原状态。
- `function` `summarize_artifact_counts` (line 243) - 从工件中提取固定的计数字段形成摘要。
- `function` `serialize_results` (line 252) - 序列化搜索结果，包含排名和来源元数据。
- `function` `serialize_result` (line 260) - 将单个搜索结果序列化为追踪字典。
- `function` `serialize_context_blocks` (line 282) - 序列化压缩后的上下文块及其来源信息。
- `function` `render_trace_report` (line 308) - 把轨迹渲染成适合直接阅读的文本报告。
- `function` `render_event_trace_report` (line 347) - 将事件跟踪数据渲染为 Markdown 报告。
- `function` `get_retrieval_stages` (line 372) - Return retrieval stages from either new artifacts or legacy fields.
- `function` `render_trace_row` (line 385) - 渲染单行检索追踪信息，返回格式化字符串。
- `function` `write_trace_json` (line 401) - 将 trace 写入 UTF-8 JSON，并向配置的 SQLite 归档脱敏副本。
- `function` `preview_text` (line 409) - 将多行文本压缩为单行预览，超出指定长度时截断并追加省略号。

## `coding_rag/repository/call_edge_candidates.py`

- `function` `first_non_test_source_candidate` (line 24) - Choose an unverified query-only source hint from at most five retrieved files.
- `function` `query_only_call_edges` (line 46) - Trace one unverified source hint, abstaining on absent or ambiguous symbols.
- `function` `score_call_edge` (line 75) - Rank an actual call edge using query words, names, and source line.
- `function` `candidate_call_edges` (line 107) - Return ranked resolvable import-bound edges from retrieved source files.

## `coding_rag/repository/call_path.py`

- `class` `ImportedReference` (line 16) - One import-bound module or symbol that can be followed statically.
- `class` `ModuleCalls` (line 24) - Parsed top-level bindings and directly addressable function scopes.
- `function` `dotted_name` (line 34) - Return a syntactic call name; dynamic expressions remain unresolved.
- `function` `matching_scope` (line 44) - Match a direct function/class scope by Python's symbol-table identity.
- `function` `local_names` (line 53) - Names that Python will resolve in a function scope before module imports.
- `function` `direct_calls` (line 60) - Yield calls in a body, excluding nested definitions and deferred lambdas.
- `function` `direct_nodes` (line 72) - Yield nodes in one function without entering deferred or nested scopes.
- `class` `PythonCallPathIndex` (line 83) - Resolve import-bound calls without importing or executing repository code.
- `method` `PythonCallPathIndex.__init__` (line 86) - Map indexed modules and chunks without executing repository code.
- `method` `PythonCallPathIndex._import_base` (line 101) - Resolve the syntactic base of an absolute or relative import.
- `method` `PythonCallPathIndex._module_calls` (line 115) - Parse one indexed module on demand and retain a bounded LRU entry.
- `method` `PythonCallPathIndex._resolve_export` (line 210) - Follow a short chain of package-level reexports, stopping on cycles.
- `method` `PythonCallPathIndex._symbol_range` (line 223) - Find the indexed line range of a target symbol and optional class.
- `method` `PythonCallPathIndex._lexical_function` (line 234) - Resolve a bare nested function in its lexical scope before module functions.
- `method` `PythonCallPathIndex._assigned_instance_candidates` (line 246) - Hint at methods of imported instances assigned once to self in this function.
- `method` `PythonCallPathIndex.trace` (line 331) - Return source-grounded static edges from one exact Python function or method.

## `coding_rag/repository/chunks.py`

- `class` `CodeChunk` (line 13) - 表示一个代码块的不可变数据类 Attributes: file_path: 所属文件路径 start_line: 起始行号（从1开始） end_line: 结束行号 text: 代码块文本内容
- `function` `split_python_files` (line 29) - 将多个 Python 文件分割成基于行的代码块 Args: python_files: PythonFile 对象列表 chunk_size: 每个代码块的行数，默认为40 overlap: 相邻代码块之间的重叠行数，默认为5 Returns: CodeChunk 对象列表
- `function` `split_code_by_lines` (line 51) - 将单个 Python 文件分割成基于行的代码块 使用滑动窗口方式分割，支持代码块之间的重叠 Args: file_path: 文件路径 text: 文件内容 chunk_size: 每个代码块的行数 overlap: 相邻代码块之间的重叠行数 Returns: CodeChunk 对象列表 Raises: ValueError: 参数值不合法时抛出

## `coding_rag/repository/files.py`

- `function` `is_env_name` (line 24) - Match local environment config files and directories.
- `function` `is_sensitive_name` (line 29) - Conservatively exclude common credential files from model-visible source.
- `class` `PythonFile` (line 37) - 表示一个 Python 文件的不可变数据类
- `function` `load_python_files` (line 43) - 读取指定仓库路径下的所有 .py 文件 Args: repo_path: 仓库路径，可以是字符串或 Path 对象 Returns: PythonFile 对象列表 Raises: FileNotFoundError: 路径不存在 NotADirectoryError: 路径不是目录
- `function` `iter_python_paths` (line 69) - Yield Python files deterministically while pruning ignored directory trees.
- `function` `should_skip` (line 100) - 检查是否应该跳过某个路径 跳过常见的生成目录和环境目录 Args: path: 要检查的文件路径 Returns: 如果路径包含被忽略的目录，则返回 True

## `coding_rag/repository/graph.py`

- `function` `build_dependency_graph` (line 10) - Build an undirected file graph from imports resolved to indexed modules.
- `function` `personalized_dependency_scores` (line 31) - Diffuse lexical relevance across imports and return one score per chunk.

## `coding_rag/repository/index.py`

- `class` `SymbolRecord` (line 19) - 不可变符号记录，描述一个AST符号的基本信息。
- `class` `FileIndex` (line 30) - 不可变文件索引记录，存储模块路径、目录层级、符号、导入和调用信息。
- `method` `FileIndex.symbols_in_range` (line 39) - 选择与指定行范围重叠的符号。
- `class` `RepoIndex` (line 49) - 仓库级文件索引。
- `method` `RepoIndex.for_chunk` (line 54) - 获取代码块对应的文件索引。
- `method` `RepoIndex.metadata_for_chunk` (line 58) - 为代码块组合文件、符号、导入及调用元数据。
- `method` `RepoIndex.structural_score` (line 87) - 根据代码块的结构化元数据与查询词元的匹配程度进行评分。
- `function` `build_repo_index` (line 113) - 将代码块按文件分组并构建仓库索引。
- `function` `build_file_index` (line 130) - 从 chunks 重建源码并提取文件结构索引。
- `function` `display_path` (line 154) - Return a repository-relative metadata path while preserving the real lookup path.
- `function` `rebuild_source` (line 167) - 从有序代码块重建文件源文本。
- `function` `extract_symbols` (line 180) - 从 AST 树中提取所有符号记录（类、函数、异步函数）。
- `function` `extract_imports` (line 236) - 遍历AST提取所有import语句中的模块名称，返回字符串列表。
- `function` `extract_calls` (line 254) - 提取AST中所有函数调用的点分名称及简短名称列表。
- `function` `dotted_name` (line 266) - 从 AST 节点中提取点分名称字符串，若为 Name 或 Attribute 链则返回，否则返回 None。
- `function` `function_signature` (line 278) - 从AST函数节点提取形如 ``name(arg1, arg2)`` 的签名字符串。
- `function` `class_signature` (line 288) - 返回类的签名，包含类名和基类名称。
- `function` `symbol_metadata` (line 297) - 将符号记录格式化为元数据字符串。
- `function` `import_terms` (line 303) - 将点分隔的名称拆分为完整名称及其各组成部分的列表。
- `function` `module_name` (line 309) - 将路径转换为 Python 模块名（去除 .py 后缀并将路径分隔符替换为点号）。
- `function` `ranges_overlap` (line 317) - 判断两个整数区间是否有交集。
- `function` `normalize_path` (line 322) - 将路径统一转为小写正斜杠分隔的字符串形式。
- `function` `unique_preserve_order` (line 327) - 去重并保持原始顺序，忽略空白项及空字符串。
- `function` `terms_match` (line 340) - 检查查询词元与值列表中任一标识符词元是否存在交集。
- `function` `identifier_terms` (line 348) - 将标识符字符串归一化并拆分为词条集合。

## `coding_rag/repository/search_index.py`

- `function` `default_cache_dir` (line 39) - Return an OS user-cache location outside source checkouts by default.
- `class` `ManifestEntry` (line 54) - Portable file identity used to invalidate an in-memory index.
- `class` `RepositoryIndexKey` (line 63) - All repository and retrieval settings that affect index contents or scores.
- `method` `RepositoryIndexKey.create` (line 75) - Normalize an index configuration into a stable registry key.
- `class` `PreparedRepositoryIndex` (line 99) - An immutable published repository snapshot and its search structures.
- `class` `RepositoryIndexAccess` (line 111) - One registry lookup paired with cache and build diagnostics.
- `method` `RepositoryIndexAccess.trace_metadata` (line 130) - Return JSON-safe index lifecycle evidence for retrieval traces.
- `class` `_BuildStats` (line 157) - Internal file and chunk reuse evidence for one candidate build.
- `class` `RepositoryIndexRegistry` (line 168) - Bounded LRU registry with per-key single-flight index construction.
- `method` `RepositoryIndexRegistry.__init__` (line 171) - Create a bounded registry; builder injection is reserved for deterministic tests.
- `method` `RepositoryIndexRegistry.get` (line 190) - Return a current index, rebuilding once when its Python manifest changed.
- `method` `RepositoryIndexRegistry._publish` (line 270) - Atomically publish one complete snapshot to the bounded in-memory LRU.
- `method` `RepositoryIndexRegistry._cache_path` (line 278) - Return a filename derived only from a one-way hash of repo identity and config.
- `method` `RepositoryIndexRegistry._validate_cache_location` (line 283) - Reject disk cache paths that could pollute an indexed repository.
- `method` `RepositoryIndexRegistry._read_disk_snapshot` (line 297) - Load and strictly validate safe JSON cache data, falling back on any defect.
- `method` `RepositoryIndexRegistry._write_disk_snapshot` (line 314) - Atomically replace safe JSON cache data; preserve the old cache on failure.
- `method` `RepositoryIndexRegistry.clear` (line 347) - Clear published entries and per-key locks for tests or explicit lifecycle shutdown.
- `method` `RepositoryIndexRegistry.entry_count` (line 353) - Return the number of currently published snapshots.
- `method` `RepositoryIndexRegistry._key_lock` (line 358) - Return the single-flight lock assigned to one exact index key.
- `function` `build_manifest` (line 364) - Scan indexable Python files without reading their contents.
- `function` `build_repository_index` (line 377) - Build a complete candidate snapshot without publishing partial state.
- `function` `build_incremental_repository_index` (line 395) - Reuse unchanged file chunks and derived data, rebuilding only changed files.
- `function` `cache_identity` (line 479) - Hash repository identity and every index-affecting setting into a cache filename.
- `function` `repository_identity` (line 489) - Return a one-way local repository identity without persisting an absolute path.
- `function` `cache_config` (line 495) - Return all portable settings required for strict cache compatibility.
- `function` `canonical_json` (line 508) - Encode deterministic JSON bytes for checksums without executable serialization.
- `function` `snapshot_envelope` (line 515) - Serialize a repository snapshot and reusable derived data into checksummed JSON.
- `function` `serialize_file_index` (line 568) - Serialize one AST-derived file index using only relative paths and JSON values.
- `function` `serialize_term_frequencies` (line 580) - Encode each BM25 document once per distinct token instead of per occurrence.
- `function` `snapshot_from_envelope` (line 590) - Strictly restore safe JSON data and rebuild only BM25 corpus statistics.
- `function` `safe_relative_path` (line 620) - Validate and normalize one non-empty repository-relative cache path.
- `function` `confined_path` (line 635) - Resolve a cached relative path and prove it remains beneath the repository root.
- `function` `parse_manifest` (line 647) - Validate cached manifest structure, ordering, and unique relative paths.
- `function` `parse_sources` (line 665) - Validate deduplicated cached sources and deterministically recreate chunks.
- `function` `parse_prepared` (line 703) - Validate AST and token-derived JSON structures aligned to cached chunks.
- `function` `parse_term_frequency_matrix` (line 744) - Validate sparse token/count rows without expanding token occurrences.
- `function` `parse_token_id_matrix` (line 776) - Validate compact token-id arrays and expand them through one shared vocabulary.
- `function` `parse_file_index` (line 794) - Validate and reconstruct one cached AST-derived file index.
- `function` `changed_file_count` (line 827) - Count added, removed, or metadata-changed manifest paths.
- `function` `clear_repository_index_registry` (line 840) - Clear the process-wide registry through an explicit test/lifecycle hook.

## `coding_rag/repository/symbol_locator.py`

- `function` `missing_qualified_definitions` (line 26) - Find named Class.method references lacking a direct indexed definition.
- `function` `exact_definition_lookup_should_abstain` (line 43) - Abstain only on one explicit missing definition request, not general behavior queries.
- `function` `locate_file_symbols` (line 49) - Rank real definitions by query overlap; never infer a symbol from a chunk alone.
- `function` `locate_retrieved_files` (line 113) - Shortlist definitions from retrieved files without changing their file order.

## `coding_rag/storage/artifacts.py`

- `function` `archive_json_file` (line 17) - Best-effort archive; a database failure leaves the generated file usable.
- `function` `write_json_artifact` (line 29) - Keep normal UTF-8 file output and archive a filtered copy in SQLite.

## `coding_rag/storage/persistence.py`

- `function` `configured_trace_db` (line 26) - Resolve a server/CLI setting; 'off' disables storage without changing tools.
- `function` `sanitize_trace` (line 38) - Copy a trace, omitting execution state and masking recognizable credentials.
- `function` `persist_trace` (line 69) - Save once, or spool a sanitized snapshot; never retry model or tool execution.
- `function` `replay_pending` (line 95) - Replay one pending snapshot without executing code or removing its source file.

## `coding_rag/storage/sqlite.py`

- `class` `TraceSchemaError` (line 68) - The configured database is not compatible with this trace store.
- `function` `utc_now` (line 72) - Return an unambiguous UTC timestamp for new records.
- `class` `TraceStore` (line 77) - Backend-independent contract for sanitized diagnostic snapshots.
- `method` `TraceStore.save_trace` (line 80) - Atomically save a newer snapshot; return false for an old revision.
- `method` `TraceStore.get_trace` (line 84) - Return the latest trace or none when it does not exist.
- `method` `TraceStore.list_traces` (line 88) - Return a bounded page of run summaries and an optional next cursor.
- `class` `SQLiteTraceStore` (line 94) - Use one short-lived connection per operation, safe across HTTP threads.
- `method` `SQLiteTraceStore.__init__` (line 97) - Bind a local database file without creating it until the first save.
- `method` `SQLiteTraceStore._connection` (line 103) - Open and close a connection, checking schema before accessing tables.
- `method` `SQLiteTraceStore._enable_wal` (line 152) - Bound retries for the first WAL transition, which can fail without waiting.
- `method` `SQLiteTraceStore.initialize` (line 167) - Create or migrate a database without inserting synthetic trace records.
- `method` `SQLiteTraceStore.save_json_file` (line 172) - Archive sanitized JSON/JSONL content, retaining distinct file versions.
- `method` `SQLiteTraceStore.get_json_file` (line 202) - Return file metadata and archived UTF-8 content for local export.
- `method` `SQLiteTraceStore.list_json_files` (line 212) - List bounded file metadata, optionally restricting to one source path.
- `method` `SQLiteTraceStore.save_trace` (line 235) - Save a cumulative snapshot and only append events absent from the store.
- `method` `SQLiteTraceStore.get_llm_calls` (line 315) - Return bounded sanitized logical-call records for one run.
- `method` `SQLiteTraceStore.get_llm_summary` (line 331) - Aggregate attempts, retries, latency, outcomes and errors for one run.
- `method` `SQLiteTraceStore.get_trace` (line 352) - Read a complete sanitized trace without returning any execution state.
- `method` `SQLiteTraceStore.list_traces` (line 362) - Filter and paginate summaries using an opaque timestamp/run ID cursor.
- `function` `_validate_run_id` (line 412) - Reject malformed identifiers before using them in storage queries.
- `function` `_llm_call_records` (line 418) - Extract and strictly validate fixed-schema call evidence from trace events.

## `coding_rag/tools/agent_readonly.py`

- `class` `AgentToolError` (line 18) - Raised when a read-only Agent tool call is invalid or rejected.
- `class` `ReadOnlyAgentTools` (line 22) - Repo-bound read-only tools used before full Agent execution exists.
- `method` `ReadOnlyAgentTools.__init__` (line 25) - 绑定仓库路径并初始化工具缓存。
- `method` `ReadOnlyAgentTools.call` (line 43) - 调度支持的只读工具并拒绝未知工具。
- `method` `ReadOnlyAgentTools.search_code` (line 76) - 对仓库进行排序检索。
- `method` `ReadOnlyAgentTools.read_file` (line 96) - 验证仓库路径并返回所请求行范围及元数据。
- `method` `ReadOnlyAgentTools.list_files` (line 125) - 列出仓库文件，排除忽略目录，遵守数量限制。
- `method` `ReadOnlyAgentTools.inspect_symbol` (line 141) - 定位指定符号并返回其源代码上下文与元数据。
- `method` `ReadOnlyAgentTools.outline_file` (line 189) - List Python symbols in one authorized file without scanning the repository.
- `method` `ReadOnlyAgentTools.trace_call_path` (line 230) - Trace bounded static calls from an exact indexed Python symbol.
- `method` `ReadOnlyAgentTools.resolve_repo_path` (line 259) - 解析仓库内路径并拒绝越权访问。
- `method` `ReadOnlyAgentTools.display_path` (line 274) - 返回相对于仓库根目录的路径，无法相对时返回绝对路径。
- `method` `ReadOnlyAgentTools.chunks` (line 283) - 懒加载并缓存仓库代码片段。
- `method` `ReadOnlyAgentTools.retriever` (line 305) - 惰性构建并缓存基于仓库代码块的 BM25 检索器。
- `method` `ReadOnlyAgentTools.repo_index` (line 316) - 惰性构建并缓存仓库符号索引。
- `function` `extract_task_identifiers` (line 343) - 从任务中提取有意义的规范化标识符，同时过滤常见词。
- `function` `format_observations` (line 359) - 将只读工具观察结果字典渲染为用于提示的文本。
- `function` `format_location` (line 385) - 将路径和可选的起止行号格式化为源码位置字符串。
- `function` `optional_int` (line 395) - 将非空值转换为 int，并将 None/空字符串映射为 None。

## `coding_rag/tools/bm25.py`

- `class` `SearchResult` (line 32) - Search result.
- `class` `PreparedBM25Data` (line 40) - Safe, reusable derived data that avoids repeated AST parsing and tokenization.
- `class` `BM25Retriever` (line 49) - Hybrid Search retriever.
- `method` `BM25Retriever.__init__` (line 51) - 构建 BM25 索引，初始化分词器、代码块和元数据倒排索引。
- `method` `BM25Retriever.prepared_data` (line 111) - Return immutable safe-to-serialize derived structures aligned with chunks.
- `method` `BM25Retriever.search` (line 120) - 对查询进行混合评分并返回 Top-K 多样化搜索结果。
- `method` `BM25Retriever.search_with_scores` (line 125) - Return search results and the same full score vector for downstream reranking.
- `method` `BM25Retriever._promote_explicit_definitions` (line 141) - Surface AST definitions for explicit symbol-location requests, not call-site requests.
- `method` `BM25Retriever._align_qualified_method_chunks` (line 165) - Use an exact Class.method() hint inside selected files without changing file scores.
- `method` `BM25Retriever.hybrid_scores` (line 209) - 计算融合后的检索分数
- `method` `BM25Retriever.rescore_results` (line 263) - 基于 Hybrid Search 对现有结果重新评分
- `function` `chunk_to_document` (line 292) - 将代码块转换为包含文件路径和文本内容的文档字符串。
- `function` `prepare_bm25_data` (line 297) - Build reusable AST metadata and tokenized corpora without creating BM25 statistics.
- `function` `term_frequencies` (line 333) - Return sparse frequencies in deterministic first-occurrence order.
- `function` `bm25_from_term_frequencies` (line 338) - Build rank-bm25 numeric state directly from validated sparse rows.
- `function` `chunk_to_metadata_document` (line 364) - Extract path, symbol, import, and call metadata.
- `function` `display_path` (line 376) - Render a repository-relative metadata path without changing real chunk paths.
- `function` `normalize_scores` (line 390) - 将分数列表归一化到 [0,1] 区间。
- `function` `token_overlap_score` (line 401) - 计算查询词元在文档词元中的覆盖率（交集 / 查询词元数）。
- `function` `path_identifier_score` (line 408) - Score exact identifier matches against path segments and filenames.
- `function` `identifier_variants` (line 437) - Add a conservative singular form for ASCII identifier tokens.
- `function` `adjust_score_for_document_role` (line 450) - 根据查询意图轻量调整测试文件和实现文件的相对顺序
- `function` `has_test_intent` (line 460) - 检测查询词元是否包含测试意图关键词。
- `function` `is_test_path` (line 465) - 检测路径是否符合测试文件命名惯例。
- `function` `normalize_path` (line 477) - 统一路径分隔符并转为小写。
- `function` `diversify_by_file` (line 482) - 优先选择不同文件，不足时用文件内候补补齐。
- `function` `chunk_key` (line 507) - 构建由文件路径和起止行号组成的代码块身份元组。

## `coding_rag/tools/env.py`

- `function` `ensure_dotenv` (line 23) - 当 `.env` 不存在时创建默认模板。
- `function` `load_dotenv` (line 42) - 把 `.env` 中的键值对加载到 `os.environ`。
- `function` `parse_env_line` (line 63) - 解析一行 `.env` 内容，忽略空行和注释。
- `function` `unquote_env_value` (line 80) - 去掉单引号或双引号包裹，并处理双引号内的转义字符。

## `coding_rag/tools/filter.py`

- `function` `filter_recalled_results` (line 11) - 对召回的代码块重新评分并保留最相关的最终上下文 召回阶段有意引入相邻的代码块。
- `function` `diversify_final_results` (line 64) - Prefer distinct files before filling remaining slots with extra chunks.
- `function` `sort_key` (line 82) - 生成排序键 排序优先级： 1.
- `function` `source_priority` (line 101) - 获取来源优先级 Args: source: 来源标识字符串 Returns: 优先级值，hybrid/bm25 返回 1，其他返回 0

## `coding_rag/tools/recall.py`

- `function` `expand_with_neighbor_chunks` (line 15) - 扩展检索种子结果，召回相邻的代码块 检索经常精确命中某一行范围，但会遗漏附近的定义、装饰器、 辅助函数或返回处理。
- `function` `group_chunks_by_file` (line 74) - 按文件路径对代码块进行分组 Args: chunks: 代码块列表 Returns: 以文件路径为键的字典，值为该文件的代码块列表
- `function` `build_chunk_index` (line 90) - 构建代码块索引 为每个文件中的代码块分配索引位置 Args: chunks_by_file: 按文件分组的代码块字典 Returns: 以代码块键为值的字典，值为该代码块在所属文件中的索引
- `function` `add_or_upgrade_result` (line 111) - 添加或升级召回结果 如果结果已存在且当前来源是种子检索结果，则升级为该种子来源。
- `function` `is_seed_source` (line 137) - 判断来源是否为直接检索结果（而非相邻召回）。
- `function` `chunk_key` (line 142) - 生成代码块的唯一键

## `coding_rag/tools/tokenizer.py`

- `class` `CodeTokenizer` (line 66) - Lightweight tokenizer.
- `method` `CodeTokenizer.tokenize` (line 73) - 对文本进行分词 Args: text: 要分词的文本 Returns: 分词后的 token 列表
- `method` `CodeTokenizer.tokenize_query` (line 98) - Expand only query-side Chinese concepts against English code identifiers.
- `method` `CodeTokenizer.tokenize_identifier` (line 109) - 对标识符进行分词 Args: identifier: 标识符字符串 Returns: 分词后的 token 列表
- `method` `CodeTokenizer.tokenize_chinese` (line 124) - 对中文文本进行 n-gram 分词 Args: text: 中文文本 Returns: n-gram 分词列表
- `function` `_identifier_tokens` (line 148) - Compute immutable identifier tokens using the existing normalization rules.
- `function` `_cached_identifier_tokens` (line 164) - Bound repeated source-token splitting without retaining unbounded strings.
- `function` `tokenize` (line 169) - 便捷函数：使用默认分词器对文本进行分词
- `function` `split_identifier` (line 174) - 将标识符分割为更小的部分 处理 snake_case 和 camelCase/PascalCase Args: identifier: 要分割的标识符 Returns: 分割后的部分列表（小写）
- `function` `is_chinese` (line 201) - 判断文本是否全为中文 Args: text: 要判断的文本 Returns: 如果文本全部由中文字符组成则返回 True

## `frontend.py`

- `class` `RequestBodyTooLarge` (line 44) - 表示 HTTP 请求体在读取前已经超过公开上限。
- `class` `UnsupportedMediaType` (line 48) - 表示 API 请求没有使用 application/json。
- `class` `PendingApproval` (line 53) - Server-only state required to resume one exact write call.
- `class` `PendingApprovalStore` (line 61) - Thread-safe, one-use browser approval state store.
- `method` `PendingApprovalStore.__init__` (line 64) - Initialize an empty process-local store.
- `method` `PendingApprovalStore.put` (line 69) - Store one pending run and return an opaque browser handle.
- `method` `PendingApprovalStore.pop` (line 76) - Consume one pending run exactly once.
- `function` `parse_args` (line 85) - 解析命令行参数并返回 argparse.Namespace。
- `class` `RepoPilotServer` (line 100) - RepoPilot 多线程 HTTP 服务器。
- `method` `RepoPilotServer.__init__` (line 106) - Initialize HTTP service state, including one-time approvals.
- `class` `RepoPilotHandler` (line 118) - RepoPilot HTTP 请求处理器。
- `method` `RepoPilotHandler.do_GET` (line 120) - 处理 GET 请求，返回首页 HTML。
- `method` `RepoPilotHandler.get_trace_history` (line 131) - Authenticate history reads before touching the server-bound database.
- `method` `RepoPilotHandler.do_POST` (line 181) - 分发普通查询与只读教学会话请求。
- `method` `RepoPilotHandler.server_allows_execution` (line 240) - 读取服务器是否允许工具执行的开关。
- `method` `RepoPilotHandler.server_learning_signing_key` (line 244) - 读取仅保存在服务端进程内的教学会话签名密钥。
- `method` `RepoPilotHandler.read_json` (line 251) - 只接受 application/json，并在读取前限制请求体大小。
- `method` `RepoPilotHandler.send_html` (line 274) - 以 UTF-8 HTML 响应发送文本。
- `method` `RepoPilotHandler.send_json` (line 283) - 以指定状态码发送 UTF-8 JSON 响应。
- `method` `RepoPilotHandler.log_message` (line 292) - 覆盖基类日志钩子以静默 HTTP 请求日志。
- `function` `run_frontend_query` (line 297) - Start or approve one browser run without accepting blanket execution permission.
- `function` `prepare_frontend_run_response` (line 366) - Serialize a run and retain resumable state only on the server.
- `function` `run_frontend_rollback` (line 383) - Restore one snapshot only when the server write capability is enabled.
- `function` `run_learning_frontend_request` (line 394) - 把浏览器外层配置适配成一次严格、只读的 Learning Session 调用。
- `function` `frontend_integer` (line 439) - 读取浏览器整数配置，拒绝 bool 与不可解析值。
- `function` `empty_to_none` (line 452) - 把空白值归一化为 None。
- `function` `configure_utf8_stdio` (line 458) - Keep Chinese help text readable in Windows terminals.
- `function` `build_index_html` (line 466) - 把工具执行许可开关注入首页 HTML 模板。
- `function` `main` (line 474) - 按命令行配置启动前端 HTTP 服务，并在退出时关闭服务器。

## `main.py`

- `function` `parse_args` (line 23) - Parse CLI arguments for the unified LLM-directed workflow.
- `function` `main` (line 66) - Run the CLI entrypoint.
- `function` `run_single_turn` (line 78) - Run one unified answer/tool turn.
- `function` `run_interactive_cli` (line 130) - Run repeated unified turns with one shared LLM client.
- `function` `build_chat_client` (line 171) - 根据 CLI 参数构建 OpenAI 兼容的聊天客户端。
- `function` `run_rollback` (line 185) - Restore one repository-bound snapshot without invoking an LLM.
- `function` `build_runtime_config` (line 203) - 将CLI参数转换为统一运行配置。
- `function` `render_unified_run` (line 227) - 将运行记录渲染为统一格式的文本输出。
- `function` `prompt_with_default` (line 280) - 交互式提示用户输入，若无输入或遇到EOF则返回默认值。
- `function` `configure_utf8_stdio` (line 289) - Keep Chinese CLI text readable in Windows terminals and redirected logs.
- `function` `print_llm_setup_hint` (line 297) - 打印指定 LLM 提供商的 API key 配置提示。

## `scripts/agent_eval.py`

- `class` `OfflinePlanClient` (line 20) - 离线计划客户端，返回固定的 ReAct 计划文本以支持离线评测。
- `method` `OfflinePlanClient.complete` (line 23) - 返回预定义的 ReAct 计划字符串，模拟 LLM 生成计划的过程。
- `function` `parse_args` (line 47) - 解析命令行参数：评测数据集路径、trace 输出路径、汇总输出路径及用例数量限制。
- `function` `evaluate_cases` (line 57) - 运行固定离线规划 contract，并真实评估只读检索候选文件。
- `function` `normalize_file` (line 153) - Normalize a repository-relative result or gold file path.
- `function` `safe_fixture_path` (line 158) - Resolve one fixture path and reject absolute or parent-traversal writes.
- `function` `normalized_unique_files` (line 173) - Normalize file paths and preserve their first-seen order.
- `function` `search_result_files` (line 183) - Extract ordered unique paths from the real search_code observation.
- `function` `candidate_file_metrics` (line 197) - Compute metrics over available gold; no available gold scores zero explicitly.
- `function` `summarize_by_repo` (line 220) - Aggregate real candidate metrics by upstream repository.
- `function` `coverage` (line 246) - 计算预期项在实际项中的覆盖率；预期为空时返回 1.0。
- `function` `average` (line 253) - 计算数值序列的算术平均值；序列为空时返回 0.0。
- `function` `write_jsonl` (line 259) - 将字典列表写入 JSONL 文件，自动创建父目录。
- `function` `main` (line 266) - 主入口：解析参数、执行离线计划评测、写入 trace 和汇总文件并打印结果。

## `scripts/agent_runtime_eval.py`

- `function` `parse_args` (line 24) - Parse portable suite, retrieval, output, and gate controls.
- `function` `_write` (line 45) - Write a formatted JSON artifact after creating its directory.
- `function` `main` (line 51) - Validate, run, optionally update, and gate the runtime contract.

## `scripts/call_edge_expansion_ab.py`

- `function` `parse_args` (line 28) - Parse pinned evaluation inputs and optional call-line ranking weight.
- `function` `main` (line 46) - Score candidate edges separately from the unchanged retrieval baseline.

## `scripts/call_path_challenge_eval.py`

- `function` `load_cases` (line 27) - Validate frozen labels and confined source paths before any graph run.
- `function` `evaluate_row` (line 66) - Run the unchanged scripted policy; inspect labels only after it finishes.
- `function` `evaluate` (line 119) - Validate pinned repositories, then score each fixed label after execution.
- `function` `compare` (line 150) - Reject benchmark identity drift, lost links, or increased false links.
- `function` `main` (line 168) - Write a challenge scorecard and optionally enforce its fixed v3 baseline.

## `scripts/call_path_eval.py`

- `function` `confined_path` (line 21) - Accept only normalized, relative POSIX paths inside a pinned checkout.
- `function` `load_cases` (line 34) - Validate component fixtures independently of runtime tool observations.
- `function` `valid_edge` (line 66) - Check emitted locators and repository confinement without trusting the tool.
- `function` `score_case` (line 87) - Score a declared target subset; other returned edges are not labeled false.
- `function` `evaluate` (line 121) - Run one shared read-only tool instance per pinned repository.
- `function` `gate` (line 151) - Reject identity drift and metric regressions against an explicit policy.
- `function` `main` (line 172) - Run the pinned component suite and optionally enforce its gate.

## `scripts/call_path_negative_eval.py`

- `function` `load_cases` (line 22) - Validate unique cases and confined source paths before any index build.
- `function` `score_case` (line 48) - Reject one exact forbidden target while reporting incomplete trace evidence.
- `function` `evaluate` (line 66) - Run every case against a pinned source checkout, reusing one tool index per repo.
- `function` `main` (line 92) - Emit an offline JSON report; gate only exact negative-link safety.

## `scripts/call_path_runtime_eval.py`

- `function` `evaluate` (line 34) - Run fixed cases; the policy sees only query and graph observations.
- `function` `compare` (line 82) - Reject identity drift and any regression in evidence-backed contract metrics.
- `function` `main` (line 98) - Run the portable offline contract and optionally enforce a fixed baseline.

## `scripts/capability_eval.py`

- `function` `_write_json` (line 22) - Write a formatted JSON artifact after creating its parent directory.
- `function` `parse_args` (line 28) - Parse scorecard, baseline, policy, and retrieval configuration options.
- `function` `main` (line 49) - Build a scorecard and optionally update or enforce a compatible baseline.

## `scripts/discovery_absence_eval.py`

- `function` `validate_negative_case` (line 33) - Reject invented labels unless a class exists and its exact method does not.
- `function` `evaluate` (line 60) - Validate source labels, then score the query-only guard on both label types.
- `function` `compare` (line 122) - Reject identity drift and regression without silently rewriting the baseline.
- `function` `main` (line 142) - Write one reproducible scorecard without invoking a model or mutating source.

## `scripts/edge_verification_eval.py`

- `function` `evaluate` (line 23) - Validate checkout identity and labels, then score one static check per case.
- `function` `main` (line 92) - Save a reproducible scorecard and optionally require all audited labels.

## `scripts/generate_function_map.py`

- `class` `SymbolInfo` (line 15) - 符号信息数据类，存储代码符号的基本元数据（种类、名称、行号、文档字符串首句）。
- `function` `parse_args` (line 24) - 解析命令行参数，返回包含扫描根路径和输出路径的命名空间。
- `function` `discover_python_files` (line 32) - 发现指定路径下的 Python 文件，自动排除缓存、版本控制和虚拟环境目录（__pycache__、.git、.venv）。
- `function` `inspect_file` (line 46) - 检查单个 Python 文件，提取模块顶层类、类直接方法及模块顶层函数的符号信息；语法错误时返回空列表。
- `function` `first_sentence` (line 71) - 提取文本的首句，以中文句号（。
- `function` `generate_function_map` (line 83) - 生成 RepoPilot 函数映射的 Markdown 文本，默认扫描 DEFAULT_ROOTS 中的路径。
- `function` `main` (line 106) - 程序入口：解析命令行参数并生成函数映射文档，输出到指定文件或标准输出。

## `scripts/import_agent_plan_cases.py`

- `function` `parse_args` (line 18) - 解析命令行参数。
- `function` `main` (line 34) - 主入口函数。
- `function` `fetch_rows` (line 57) - 通过 Hugging Face datasets-server HTTP API 获取数据集行。
- `function` `convert_row` (line 90) - 将原始数据集行转换为 Agent 计划评测用例格式。
- `function` `diff_paths` (line 122) - 从 Git diff 文本中提取唯一的 Python 目标文件路径。
- `function` `build_repo_files` (line 141) - 为离线计划评测生成最多若干个上下文文件。
- `function` `normalize_task` (line 177) - 规范化任务文本，去除每行尾部空白并压缩首尾空行。
- `function` `parse_json_list` (line 189) - 解析 JSON 字符串为列表；解析失败或结果非列表时返回空列表。

## `scripts/import_json_artifacts.py`

- `function` `main` (line 19) - Initialize the DB and import each selected file, reporting partial failures.

## `scripts/multirepo_eval.py`

- `function` `parse_args` (line 25) - Parse portable inputs, retrieval config, and baseline controls.
- `function` `_write_json` (line 46) - Write a formatted JSON artifact, creating its parent directory.
- `function` `main` (line 52) - Validate, evaluate, and optionally update or enforce the independent gate.

## `scripts/patch_leetcode_imports.py`

- `function` `parse_args` (line 21) - 解析命令行参数。
- `function` `main` (line 30) - 主入口：扫描仓库中所有 Python 文件，为需要类型导入的文件添加补丁。
- `function` `should_skip` (line 73) - 判断是否应跳过该文件。
- `function` `find_insert_index` (line 81) - 找到应插入导入语句的行索引。
- `function` `skip_blank_and_comment_lines` (line 100) - 跳过空白行和注释行。
- `function` `skip_module_docstring` (line 111) - 跳过模块文档字符串。

## `scripts/patch_outcome_eval.py`

- `function` `parse_args` (line 35) - Parse real-source manifest, trusted candidate, artifact, and gate controls.
- `function` `_write_json` (line 73) - Write a deterministic human-readable JSON artifact.
- `function` `_provider` (line 79) - Build an oracle, trusted static, or explicitly enabled live Agent provider.
- `function` `main` (line 135) - Validate, evaluate, optionally update oracle baseline, and enforce the gate.

## `scripts/quality_check.py`

- `function` `gate_specs` (line 18) - Keep model/network flags out of this intentionally fixed offline suite.
- `function` `summarize` (line 46) - Expose comparable facts while preserving each evaluation's distinct meaning.
- `function` `run_quality_check` (line 91) - Run gates serially so latency checks do not contend with one another.
- `function` `main` (line 143) - Print one aggregated scorecard and fail if any required offline gate fails.

## `scripts/replay_trace_pending.py`

- `function` `main` (line 16) - Save each explicitly supplied pending file; retain all source files.

## `scripts/repo_inspect.py`

- `function` `verify_edge` (line 46) - Check one direct static call, returning bounded evidence or an explicit unknown.
- `function` `bound_discovery_groups` (line 85) - Preserve file coverage before extra symbols when the output budget is tight.
- `function` `parse_args` (line 103) - Accept one repository and one bounded read-only tool request.
- `function` `inspect` (line 157) - Dispatch bounded read-only repository tools without a model.
- `function` `serve` (line 394) - Reuse one index registry across bounded JSONL requests without changing tool semantics.
- `function` `main` (line 439) - Print JSON evidence with distinct success and invalid-request exit codes.

## `scripts/retrieval_eval.py`

- `class` `EvalCase` (line 32) - 单个检索评测用例。
- `function` `load_evalset` (line 61) - 从文件路径加载评测用例集。
- `function` `normalize_eval_case` (line 81) - 将原始字典规范化为 EvalCase。
- `function` `build_eval_generator` (line 102) - 按命令行参数构建回答生成器。
- `function` `generate_eval_answer` (line 117) - 返回 answer/error 字典，优先返回配置错误，无生成器时返回空状态，并把生成时 RuntimeError 转为错误文本。
- `function` `pipeline` (line 135) - 检索流水线，依次执行初搜、邻居召回与最终过滤三个阶段的检索。
- `function` `default_recall_max_results` (line 159) - 邻居召回的默认上限条数。
- `function` `mrr_and_recall_at_k` (line 168) - 在前 k 项中按等价路径去重 gold 项，计算 MRR 与召回率。
- `function` `analyze_bad_case` (line 195) - 比较初搜、召回和最终前 k 命中，生成启发式原因与提示。
- `function` `build_stage_file_record` (line 239) - 为各检索阶段构建文件命中快照并按阶段丢失位置生成诊断记录。
- `function` `build_stage_snapshot` (line 275) - 构建阶段快照，包含前 k 个唯一结果文件、命中的 gold 文件列表以及是否命中。
- `function` `matched_relevant_files` (line 290) - 从检索结果的前 k 个中匹配相关的 gold 文件，按检索结果中首次命中的顺序去重，并返回对应的 gold 路径。
- `function` `unique_result_files` (line 307) - 提取检索结果前 k 个中按首次出现去重后的唯一文件路径。
- `function` `readable_path` (line 321) - 将路径转为可读字符串：对绝对路径优先尝试相对于给定 roots 或当前目录展示，失败则保留规范化绝对路径，统一使用正斜杠。
- `function` `diagnose_stage_drop` (line 338) - 按优先级返回阶段诊断结果。
- `function` `aggregate_stage_diagnostics` (line 359) - 聚合阶段诊断记录，统计诊断计数与用例、bad case，以及可选的上下文字符统计。
- `function` `context_char_totals` (line 410) - 计算每行上下文字符总数。
- `function` `render_stage_diagnostics_summary` (line 429) - 渲染阶段诊断摘要字符串。
- `function` `ordered_diagnoses` (line 458) - 按固定顺序返回诊断名列表。
- `function` `is_relevant_result` (line 468) - 判断检索结果是否相关，相关性由等价路径形式集合的交集判定.
- `function` `matched_relevant_index` (line 477) - 返回第一个与结果路径形式集合有交集的 gold 索引，无匹配则返回 None.
- `function` `comparable_path_forms` (line 490) - 构建路径的等价形式集合，包含规范化原始路径、基于当前目录及给定 roots 解析的绝对形式和可相对化形式.
- `function` `normalize_path_text` (line 513) - 规范化路径文本：统一正斜杠、去除首尾空白并 casefold.
- `function` `stage_files_path_for_trace` (line 518) - 在 trace 文件同目录下，基于 stem 生成 ``{stem}_stage_files`` 路径。
- `function` `stage_summary_path_for_trace` (line 528) - 在 trace 文件同目录下，基于 stem 生成 ``{stem}_stage_summary.json`` 路径。
- `function` `write_json` (line 534) - 创建父目录并以 UTF-8 写入格式化 JSON。
- `function` `write_jsonl` (line 544) - 创建父目录并以 UTF-8 逐行写入 JSON。
- `function` `run_eval` (line 557) - 可选初始化 LLM（配置错误记入逐例结果而非立即终止）、运行所有检索评测、写 trace/阶段文件/阶段汇总、打印聚合指标与可选 LLM 输出，并可继续参数搜索；成功返回 0。
- `function` `optimize` (line 652) - 收集网格搜索结果，打印并返回报告。
- `function` `collect_optimization_results` (line 660) - 在固定 OPTIMIZATION_SEARCH_SPACE 中按各用例 MRR 求均值，严格更高时更新最佳项。
- `function` `render_optimization_report` (line 685) - 格式化各组合及可选推荐，无行时说明无用例。
- `function` `print_llm_outputs` (line 711) - 按用例打印 LLM 答案、错误或空输出提示。
- `function` `parse_args` (line 727) - 解析检索、追踪、优化、环境和可选 LLM 生成参数。

## `scripts/run_claude_guarded.py`

- `class` `ClaudeUsage` (line 28) - 汇总 Claude Code 子进程输出事件中观测到的令牌用量。
- `method` `ClaudeUsage.task_tokens` (line 44) - 总任务令牌数 = input_tokens + output_tokens + cache_read_input_tokens + cache_creation_input_tokens。
- `method` `ClaudeUsage.cache_miss_tokens` (line 54) - 缓存未命中令牌数 = input_tokens + cache_creation_input_tokens。
- `method` `ClaudeUsage.cache_input_tokens` (line 59) - 缓存输入令牌数 = input_tokens + cache_read_input_tokens + cache_creation_input_tokens。
- `method` `ClaudeUsage.cache_miss_ratio` (line 68) - 缓存未命中比率 = cache_miss_tokens / cache_input_tokens，分母非正时返回 0.0。
- `function` `parse_args` (line 75) - 解析命令行参数，包括守卫限额、权限、超时和日志参数。
- `function` `main` (line 147) - 从标准输入读取提示，启动 ``claude`` stream-json 子进程。
- `function` `parse_json_line` (line 231) - 仅当行是 JSON 对象时返回字典，否则返回 None。
- `function` `update_usage_from_event` (line 240) - 从事件的直接/message usage 与按模型汇总 usage 中取各字段较大值更新峰值观测。
- `function` `find_usage_payload` (line 277) - 查找事件顶层或 message 内的 usage 字典并返回。
- `function` `find_model_usage_payloads` (line 288) - 从 modelUsage 映射中过滤并返回字典值列表。
- `function` `guard_reason` (line 296) - 按总任务token、缓存未命中token、达到最小缓存输入后的未命中比例依次检查并返回首个停止原因。
- `function` `write_log` (line 316) - 把事件、原因和用量快照写为一条JSONL并刷新。

## `scripts/source_discovery_eval.py`

- `function` `source_span` (line 54) - Require an unambiguous indexed function or method matching the audit anchor.
- `function` `rank_source` (line 69) - Separate file, function-body overlap, and actual definition-line evidence.
- `function` `joint_pair_rank` (line 86) - Count candidate definitions in the order Codex would review them.
- `function` `summarize` (line 97) - Keep file discovery separate from evidence inside the source function.
- `function` `evaluate` (line 130) - Run the same retrieval pipeline on pinned positives and local/same-file negatives.
- `function` `main` (line 210) - Write a portable scorecard and return a diagnostic exit status.

## `scripts/validate_evalset.py`

- `function` `main` (line 17) - Print validation findings and return zero for valid data or two otherwise.

## `web_ui.py`

- `class` `RepoPilotUI` (line 41) - Simple desktop UI for interactive RAG search and evals.
- `method` `RepoPilotUI.__init__` (line 44) - 初始化 RepoPilot 用户界面及所有控件变量。
- `method` `RepoPilotUI._build_layout` (line 67) - 构建主界面布局，包含仓库路径选择、查询输入和参数配置控件。
- `method` `RepoPilotUI._new_conversation` (line 129) - Start an isolated chat while retaining the selected repository.
- `method` `RepoPilotUI._pick_repo_path` (line 135) - 弹出目录选择对话框，让用户选取仓库路径。
- `method` `RepoPilotUI._run_async` (line 141) - 在新线程中异步执行查询流水线。
- `method` `RepoPilotUI._run_pipeline` (line 150) - 执行仓库代码检索、可选的 LLM 回答生成，并更新界面状态和输出。
- `method` `RepoPilotUI._pick_evalset` (line 231) - 选择评测集文件并更新路径。
- `method` `RepoPilotUI._run_eval_async` (line 240) - 在守护线程中启动评测。
- `method` `RepoPilotUI._run_eval` (line 244) - 执行检索评测，可选生成 LLM 回答，写出追踪与阶段诊断，并更新界面。
- `method` `RepoPilotUI._export_last_trace` (line 316) - 将最近评测 trace 导出到用户选择的位置。
- `method` `RepoPilotUI._show_bad_cases` (line 327) - 读取最近 trace 并在界面展示未命中的 bad case。
- `method` `RepoPilotUI._optimize_async` (line 343) - 在守护线程启动调参。
- `method` `RepoPilotUI._run_optimize` (line 347) - 运行评测参数优化并更新界面。
- `method` `RepoPilotUI._build_llm_args` (line 365) - 根据界面选项构造评测生成器所需的临时 LLM 参数对象。
- `method` `RepoPilotUI._render_eval_output` (line 380) - 将评测指标、产物路径、阶段摘要及可选 LLM 输出渲染为文本。
- `method` `RepoPilotUI._render_results` (line 422) - 把检索结果格式化为包含分数、来源和代码内容的文本块。
- `method` `RepoPilotUI._set_output` (line 439) - 把输出替换操作调度到 Tk 事件循环。
- `method` `RepoPilotUI._replace_output` (line 443) - 立即替换文本控件内容。
- `method` `RepoPilotUI._set_status` (line 448) - 更新界面状态文本。
- `function` `main` (line 453) - 创建 Tk 根窗口、初始化 RepoPilotUI 并进入事件循环。
