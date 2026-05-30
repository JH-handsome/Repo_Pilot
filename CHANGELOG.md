# Change Log

本文件用于记录 RepoPilot 每次修改的主要内容。后续修改请按时间倒序追加，重点写清楚改了什么、影响哪些文件、是否需要注意测试或配置。

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
