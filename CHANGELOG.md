# Change Log

本文件用于记录 RepoPilot 每次修改的主要内容。后续修改请按时间倒序追加，重点写清楚改了什么、影响哪些文件、是否需要注意测试或配置。

## 2026-07-31

- `requirements.txt` 增加相互兼容的 LangChain、LangGraph 与 Pydantic 版本区间。
- 新增 `coding_rag/agent/langchain_tools.py`：仅暴露严格 Pydantic schema 的 `search_code`、`read_file` 两个 `StructuredTool`，校验后统一委托 RepoPilot 的 `AgentExecutor.call()`，不提供写工具或通用工具名透传。
- 新增 `coding_rag/agent/graph_runtime.py`：使用可序列化 `GraphState`、四个显式节点和条件边实现有界只读循环；复用 Hybrid Search、结果压缩、引用范围/校验、统一 trace 与 `UnifiedRun`，并将模型、工具、路由和步数错误收敛为 `failed` 或 `partial`。
- 保留现有 `coding_rag/agent/runtime.py`、CLI 和前端不变；未使用 LangChain 的 `create_agent`、`AgentExecutor` 或 `ToolNode`，未接入写工具、流式执行、检查点或记忆。
- 新增 `tests/test_graph_runtime.py`，覆盖严格参数 schema、执行入口、直答/工具循环、路径边界、工具与模型错误、非法/多重调用、步数上限、上下文压缩、引用校验、状态序列化和公开结果兼容性。

## 2026-07-14

- 新增 `coding_rag/agent/runtime.py`，将 ASK 与 Agent 用户入口合并为统一 LLM 决策循环；模型可以直接回答，或调用 `read_file`、`search_code`、`apply_patch`、`run_command`、`inspect_diff`。
- 统一入口默认强制 dry-run；CLI 新增 `--execute-tools`，浏览器新增 `--allow-tool-execution` 与页面复选框双重授权，项目安全 policy 仍可强制只读。
- CLI 移除 `--workflow-mode`、`--agent`、`--agent-exec`、`--llm`、`--dry-run`、`--safe-mode` 等旧工作流参数，不再自动创建 `.env`。
- 浏览器前端改为单一 `POST /api/run`，移除 ASK/Agent 模式控件和旧 API，并按事件流展示模型决策、工具、安全检查和最终回答。
- 统一 trace 新增真实 LLM/工具耗时、observation 截断、Hybrid Search 子 trace、引用范围、安全拒绝和动态 `success/dry_run/partial/failed` 状态。
- 增加统一运行时、CLI 清理、网页授权和旧端点 404 测试；更新 README 与函数地图。

## 2026-07-11

- 打通 `ReActAgentInterface.run()` 最小工具调用循环，支持结构化 JSON tool call，接入 `read_file`、`search_code`、`apply_patch`、`run_command`、`inspect_diff`，默认 dry-run，并将工具结果写入统一 trace event。
- 新增 Agent 安全策略配置读取：支持 `.repopilot/policy.json` 和 `--agent-policy`，默认 denylist 永远保留，配置 allowlist 只扩展允许命令。
- 统一 trace 消费侧补强：`render_trace_report()` 可渲染 Agent/tool events，浏览器前端 API 返回统一 `trace`，页面增加 Trace Events 调试入口。
- 新增 `scripts/generate_function_map.py` 和 `scripts/agent_eval.py`，分别用于 AST 函数地图生成和 Agent 计划离线评测；补充 CLI smoke、frontend trace、Agent loop、安全策略和评测测试。
- 删除冗余临时测试入口 `_run_tests.py`，统一使用 `python -m unittest` 运行测试。
- 更新 `README.md` 项目结构和评测示例路径，移除已清理的顶层 `rag/`、`legacy/` 说明，改为当前 `coding_rag/agent`、`coding_rag/rag`、`coding_rag/repository`、`coding_rag/tools` 包结构。
- 更新 `FUNCTION_MAP.md` 中 `AgentExecutor`、`AgentSafetyPolicy` 和统一 trace 相关函数说明，避免函数地图落后于当前实现。

## 2026-06-04

- 新增 `FUNCTION_MAP.md`：按入口、Agent、RAG、检索/仓库、脚本分组整理关键文件、类和函数职责，方便后续定位工作流。
- 清理 `coding_rag` 根目录下仅做转发的兼容文件，调用方统一改为 `coding_rag.agent`、`coding_rag.rag`、`coding_rag.repository`、`coding_rag.tools` 下的真实模块路径；同步修复顶层 `rag/` 删除后的 import 断链。
- 新增 `coding_rag/agent/executor.py`：提供 Agent 执行器 `read_file(path)`、`search_code(query)`、`apply_patch(diff)`、`run_command(cmd)`、`inspect_diff()`，并在 Agent 工具声明中加入可执行工具。
- 更新 `main.py`：新增 `--agent-exec` 作为代码 Agent 执行入口，支持 `--agent-exec --llm` 生成 LLM 实施草案，新增 `--dry-run` 和 `--safe-mode` 在 Agent 执行时跳过记忆与运行日志写入。
- 更新 `coding_rag/agent/workflow.py` 和测试：Agent 配置增加 dry-run/safe mode，无写入模式下仍完成检索、计划和可选 LLM 草案，但不产生持久化副作用。
- 扩展 `frontend.py` 浏览器前端：新增 `POST /api/agent-plan` 端点和 `run_frontend_agent_plan` 函数；UI 新增 ASK/Agent 模式切换、Agent 计划审查开关，Agent 模式下展示 LLM 计划、工具接口、审查结果和检索统计。
- 更新 `tests/test_frontend.py`：新增 Agent 计划端点测试，覆盖无 LLM 调用、LLM 调用、review_agent_plan 校验错误和 mock 审查链。
- 新增 `coding_rag/agent_plan_reviewer.py`：调用 LLM 审查 Agent 计划是否合理，解析 verdict/score/issues/suggestions；CLI 新增 `--review-agent-plan`。
- 新增 `tests/test_agent_plan_reviewer.py`，覆盖审查 prompt、JSON 解析、审查渲染和 50 条 Agent 计划测试集的审查链路。
- 扩展 Agent 计划测试集：新增 `scripts/import_agent_plan_cases.py`，从 SWE-bench Verified 导入前 50 条任务到 `datasets/eval/agent_plan_cases.json`，并让测试校验测试集规模和计划结构。
- 搭建 ReAct Agent 计划模式：`coding_rag/task_planner.py` 新增 `AgentPlanConfig`、`AgentToolSpec`、`AgentPlanRun` 和 `run_agent_plan_mode`，先完成任务分析、RAG 上下文检索、工具接口声明和 LLM 计划 prompt 组装，不执行工具调用。
- 更新 `main.py`：`--workflow-mode agent` 进入新的 Agent 计划模式，`--show-agent-prompt` 可查看发给 LLM 的规划 prompt；旧版 Agent 仍通过 `--agent` 进入。
- 新增 `datasets/eval/agent_plan_cases.json`，用于观察 Agent 计划是否覆盖关键章节和工具接口；更新 `tests/test_task_planner.py` 覆盖计划模式。

## 2026-06-03

- 新增 `frontend.py` 浏览器前端：访问 `http://127.0.0.1:8765` 后可输入仓库路径和问题，运行 ASK/RAG，并可勾选调用大模型回答。
- 修复 `web_ui.py` 中对已删除 `main.run_answer_generator` 的引用，改为直接使用 `rag.answer_generator.build_generator`。
- 新增 `tests/test_frontend.py`，覆盖浏览器前端页面控件和无 LLM 的 ASK API 返回。
- 新增交互式入口：`python main.py` 或 `python main.py <repo_path>` 会进入连续问答模式，输入问题自动走 ASK，修改/修复类需求进入 ReAct Agent 占位。
- 更新 `tests/test_interactive_cli.py`，覆盖无位置参数解析、立即退出和一次 ASK 交互查询。
- 新增 `coding_rag/task_planner.py`：建立 ASK/Agent 分类模块，ASK 模式封装 RAG 检索、prompt 组装和可选 LLM 问答；Agent 模式先提供 ReAct 接口占位。
- 更新 `main.py`：新增 `--workflow-mode auto|ask|agent`、`--show-plan` 和 `--show-ask-prompt`，普通查询默认走 ASK 工作流，旧 `--agent` 参数保留兼容。
- 修正工作流路由：`--workflow-mode` 默认改为 `auto`，自动区分 ASK 和 ReAct Agent 占位；旧版代码 Agent 仅通过 `--agent` 显式进入。
- 新增 `tests/test_task_planner.py`，覆盖任务分类、ASK prompt 构建、ASK 模式检索调用和 ReAct 接口占位。
- 拆分 Agent 记忆和运行日志：`coding_rag/agent.py` 新增独立运行日志 JSONL 写入，CLI 新增 `--agent-log`，默认写入 `artifacts/agent_run_log.jsonl`。
- 调整 Agent 分工：RepoPilot Agent 负责检索、计划、记录和验证建议，实际代码修改交给 Claude Code。
- 精简 `AGENT.md`，仅保留基本运行规则、运行环节和记录文件；更新 `CLAUDE.md` 为 Claude Code 执行规则。
- 更新 `tests/test_agent.py`，覆盖 Agent 运行日志单独落盘行为。

## 2026-05-30

- 增强 RAG 检索：新增 `coding_rag/repo_index.py`，基于 AST 提取文件树、函数/类签名、import 和调用关系，并接入 `BM25Retriever` 的结构化 metadata 检索。
- 更新 `coding_rag/tokenizer.py`：补充函数、类、签名、导入、调用、文件树、索引等中文查询词到代码 token 的扩展。
- 新增 `tests/test_repo_index.py`、`tests/test_bm25_repo_index.py` 和 `tests/test_tokenizer_repo_terms.py`，覆盖 repo 索引和结构化检索行为。
- 优化评测诊断：`scripts/retrieval_eval.py` 新增阶段掉点聚合统计、上下文长度统计和 `_stage_summary.json` 输出，便于从 trace 中定位初始检索缺失、召回阶段丢失、过滤后掉点等问题。
- 优化自动调参：参数搜索结果改为可复用报告，CLI 继续打印，`web_ui.py` 会直接在窗口中展示推荐参数和各组 MRR。
- 更新 `rag.trace`：context compaction 阶段新增 `char_count`，便于统计上下文压缩后的长度。
- 新增 `rag/citation_validator.py`：校验 LLM 输出中的 `path:start-end` 引用是否来自本次检索上下文，缺失或越界时追加引用校验提示。
- 更新 `rag/answer_generator.py`：生成答案后自动执行引用校验。
- 更新 `AGENT.md`：同步当前进度，并清理已经落地或过时的继续推进建议。
- 优化 Agent：新增任务画像 `task_profile`，按任务类型生成动态验证命令，并写入 Agent 输出和 JSON 记录。
- 优化 Agent 记忆：新增 `append_if_new` 去重写入逻辑，避免相同任务和文件集合重复污染记忆库。
- 增强 Agent 运行轨迹显示：`coding_rag.agent.AgentRun` 新增 `agent_trace`，CLI 默认展示 Agent 每一步状态和关键产物，JSON 输出同步包含该字段。
- 新增 `coding_rag/agent.py`，实现代码 Agent 工作流：接收任务、读取记忆、RAG 检索、生成计划/实现草案、写入记忆。
- 新增 `coding_rag/agent_memory.py`，使用 JSONL 保存 Agent 长期记忆，并支持按任务相关性检索历史记忆。
- 更新 `main.py`，新增 `--agent`、`--agent-memory`、`--agent-memory-limit` 参数；`--agent --llm` 可生成 LLM 实施草案。
- 新增 `tests/test_agent.py` 和 `tests/test_agent_memory.py`，覆盖 Agent 离线运行、LLM 草案和记忆检索。
- 新增 `rag/trace.py`，记录并渲染 RAG 检索轨迹，覆盖 initial search、neighbor recall、final filter 和 context compaction 四个阶段。
- 更新 `main.py`，支持 `--show-trace`、`--trace-out` 和 `--trace-include-text`，便于单次查询排查。
- 更新评测和 UI trace 输出，新增 `trajectory` 与 `context` 字段，方便后续定位优化点。
- 更新 `.gitignore`，忽略 `artifacts/*.json`，避免单次查询 trace 污染 git 状态。
- 优化 RAG 上下文组织：`rag.prompt` 在发送给 LLM 前合并同文件连续/重叠代码块，减少重复 token，并新增 `tests/test_prompt_context.py`。
- 更新 `coding_rag/context_recaller.py`，保留 Hybrid Search 种子结果的 `hybrid` 来源标签，相邻块继续使用 `recall:*`。
- 优化 Hybrid Search 文档角色排序：非测试意图查询轻微降权测试文件，测试意图查询保留测试文件优先能力。
- 删除 `coding_rag/bm25_retriever.py` 中针对具体 bad case 的 `query_path_boost` 路径补丁，改为 Hybrid Search：融合正文 BM25、路径/符号 BM25 和 token 覆盖度。
- 更新 `coding_rag/tokenizer.py`，增加通用中英代码检索词汇扩展，改善中文问题和英文代码路径/符号之间的匹配。
- 更新 `tests/test_bm25_retriever.py`，移除路径补丁测试，新增 Hybrid Search 行为测试。
- 更新 `coding_rag/result_filter.py`，让 `hybrid` 种子结果和原 `bm25` 结果拥有同等来源优先级。
- 新增 `AGENT.md`，整理项目当前进度、主要目录、常用命令、开发约定和后续方向。
- 新增 `CHANGELOG.md`，作为后续主要修改记录入口。
