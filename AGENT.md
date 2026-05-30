# RepoPilot Agent Notes

本文档记录 RepoPilot 当前开发状态、项目约定和后续修改时需要注意的事项。目标是在不改变项目主线的前提下，让后续维护者能快速理解代码结构、运行流程和当前进度。

## 项目定位

RepoPilot 是一个轻量级代码 RAG 检索工具，主要用于学习、实验和调试代码仓库问答流程。当前重点是读取 Python 仓库，将代码切成带行号的片段，用 Hybrid Search 检索相关代码块，并可选接入大模型生成回答。

## 当前进度

- 已完成命令行入口 `main.py`，支持仓库路径、查询文本、top-k、切片大小、重叠行数、召回窗口、最终过滤和 LLM 参数。
- 已完成核心检索链路：`file_loader` 读取 Python 文件，`code_splitter` 生成 `CodeChunk`，`tokenizer` 处理代码标识符、中文 n-gram 和通用中英代码检索词汇扩展，`BM25Retriever` 以 Hybrid Search 方式融合正文 BM25、路径/符号/AST metadata BM25 和 token 覆盖度。
- 已加入轻量 repo 索引：`coding_rag.repo_index` 会基于 AST 提取文件树、函数/类签名、import 和调用关系，并接入 Hybrid Search 的结构化 metadata。
- 已加入邻近上下文召回：`context_recaller` 会围绕 Hybrid Search 命中的种子块补充同文件相邻代码块，并保留种子来源标签。
- 已加入结果过滤与重排：`result_filter` 会对召回结果重新评分，并按最终数量和最低分数筛选。
- 已加入 RAG 上下文压缩：`rag.prompt` 会在发送给 LLM 前合并同文件连续/重叠代码块，减少 chunk overlap 带来的重复内容。
- 已加入 RAG 检索轨迹：`rag.trace` 可以记录 `initial_search`、`neighbor_recall`、`final_filter` 和 `context_compaction` 四个阶段，CLI 支持 `--show-trace` 与 `--trace-out`。
- 已加入代码 Agent 工作流：`coding_rag.agent` 可以从用户任务出发，读取记忆、检索上下文、生成计划/实现草案，并写入长期记忆。
- 已加入 Agent 运行轨迹显示：Agent 输出会展示 `receive_task`、`load_memory`、`retrieve_context`、`plan_changes`、`implement`、`verify`、`remember` 每一步的状态和关键产物。
- 已加入 Agent 记忆：`coding_rag.agent_memory` 使用 JSONL 保存任务摘要、相关文件和关键决策，默认路径为 `artifacts/agent_memory.jsonl`。
- 已加入 Agent 任务画像和动态验证建议：Agent 会识别 `feature`、`bugfix`、`optimization`、`evaluation`、`docs`、`test` 等任务类型，并给出对应检查命令。
- 已加入 Agent 记忆去重：相同任务和相同文件集合不会重复写入记忆库，降低长期记忆噪声。
- 已加入文件多样化逻辑，改善单文件结果过多的问题；原先针对具体 bad case 的路径补丁已移除。
- 已加入通用文档角色感知：非测试查询会轻微降权测试文件，测试意图查询仍可优先返回测试文件。
- 已支持 OpenAI 兼容的大模型调用，包含 DeepSeek、Qwen、Kimi、Zhipu 和 custom 预设。
- 已拆分 `rag/` 中的 prompt 与答案生成器，支持 `judge`、`code-understand`、`code-generate`、`leetcode`、`api` 等生成模式。
- 已完成 Tkinter 桌面界面 `web_ui.py`，支持检索、LLM、评测、trace 导出、bad case 面板和自动调参入口。
- 已有检索评测脚本 `scripts/retrieval_eval.py`，支持 JSON/JSONL 评测集、MRR/Recall、trace 保存、阶段文件诊断和参数搜索。
- 已提供 LeetCode 本地类型补丁和导入修复脚本，便于处理 LeetCode 风格代码数据。
- 已有单元测试覆盖分词、加载、Hybrid Search、召回、过滤、上下文压缩、环境变量、LLM 配置和评测流程。

## 主要目录

```text
coding_rag/   核心检索模块：文件加载、切片、分词、Hybrid Search、召回、过滤、LLM 客户端
rag/          Prompt 模板和答案生成逻辑
scripts/      离线脚本：检索评测、LeetCode 补丁、数据下载
datasets/     示例评测集和 LeetCode 参考数据
artifacts/    运行产生的 trace、评测结果等临时产物
tests/        单元测试
legacy/       早期原型，仅作参考，不参与当前主流程
```

阅读代码时优先从 `main.py`、`coding_rag/` 和 `scripts/retrieval_eval.py` 开始；`legacy/` 不作为当前功能修改依据。

## 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

运行一次检索：

```bash
python main.py . "BM25 检索器在哪里建立索引？" --top-k 5
```

查看分词结果：

```bash
python main.py . "代码在哪里切成 chunk？" --show-tokens
```

启用大模型回答：

```bash
python main.py . "解释代码切片逻辑" --llm --mode code-understand
```

启动桌面界面：

```bash
python web_ui.py
```

运行检索评测：

```bash
python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --top-k 5
```

运行 Agent：

```bash
python main.py . "给 RepoPilot 增加一个新的功能" --agent
python main.py . "给 RepoPilot 增加一个新的功能" --agent --llm --llm-provider deepseek
python main.py . "给 RepoPilot 增加一个新的功能" --agent --show-trace --trace-out artifacts/agent_run.json
```

运行测试：

```bash
python -m unittest
python -m compileall coding_rag rag scripts main.py web_ui.py tests
```

## 开发约定

- 优先保持现有轻量实现，不引入不必要的大型框架。
- 新增检索能力时，优先放在 `coding_rag/`，并在 CLI、UI 和评测脚本之间保持参数含义一致。
- 修改检索排序、召回、过滤或 prompt 时，应同步补充或更新测试。
- 评测相关输出默认写入 `artifacts/`，避免污染源代码目录。
- `.env` 只保存本地密钥配置，不要提交真实 API key。
- 修改项目主要功能后，请在 `CHANGELOG.md` 记录日期、文件和主要内容。

## Agent 工作流设计

目标：让 RepoPilot 从收到用户任务开始，尽可能自主地完成“定位代码、制定计划、生成实现草案、给出验证方案、沉淀记忆”的闭环。当前实现先提供安全的计划与草案能力，不直接修改工作区文件。

1. `receive_task`：接收用户任务，识别目标、约束、交付物和可能涉及的模块。
2. `load_memory`：从 `artifacts/agent_memory.jsonl` 读取历史任务，按 token overlap 找相关记忆。
3. `retrieve_context`：复用 Hybrid Search、邻近召回、最终过滤和 trace，找到候选代码块。
4. `plan_changes`：结合记忆和检索结果，生成候选文件、改动步骤、风险和验证点。
5. `implement`：未启用 LLM 时输出离线实施说明；启用 LLM 时基于上下文生成实现草案。
6. `verify`：根据任务画像输出应运行的单测、编译检查、检索评测或人工检查项。
7. `remember`：把任务、状态、摘要、相关文件和关键决策追加写入 Agent 记忆；若检测到重复任务和文件集合则跳过写入。

Agent 运行时会生成 `agent_trace`，包含每个步骤的 `step`、`status`、`detail` 和 `artifacts`。CLI 默认显示这条 Agent 轨迹；`--show-trace` 额外显示底层 RAG 检索轨迹。

Agent 任务画像会写入 `task_profile`，包含：

```json
{"kind":"optimization","confidence":0.60,"reasons":["优化"],"recommended_checks":["python -m unittest"]}
```

Agent 记忆格式为 JSONL，每行包含：

```json
{"id":"agent-...","created_at":"...","task":"...","status":"planned","summary":"...","files":["..."],"decisions":["..."]}
```

后续可以继续把 Agent 从“计划/草案模式”推进到“受控补丁执行模式”：要求 LLM 输出统一 diff，经过解析、测试和用户确认后再应用。

## 近期可继续推进

- 修正或清理局部重复逻辑，例如 `main.py` 中候选数量赋值的重复语句。
- 继续扩大评测集，覆盖更多中文问题、路径问题、函数名问题和 LeetCode 查询。
- 让 UI 的自动调参结果直接显示在窗口里，而不是只输出到终端。
- 继续优化 Hybrid Search 的权重、字段提取和评测集表现。
- 为 LLM 生成结果增加更稳定的引用格式校验。
- 基于 trace 聚合统计常见掉点阶段，例如初始检索缺失、召回后被过滤、上下文压缩过长等。
- 为 Agent 增加受控 patch apply 能力：只接受标准 diff，先 dry-run，再运行测试，最后由用户确认落盘。
- 为agent提供更多的api接口，对接入的工具进行测试，精简返回结果，保持每一步编写state
- 进行实现之前先制定计划流，生成给用户，方便用户对其进行审查，对计划流中的结果进行摘要总结后显示
