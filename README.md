# RepoPilot：轻量级代码 RAG 检索工具

RepoPilot 是一个用于学习和实验的代码检索项目。它会读取一个 Python 仓库，把代码切成带行号的片段，用 Hybrid Search 找出和问题最相关的代码块，并可选调用大模型生成回答。

## 项目结构

```text
.
├── coding_rag/
│   ├── agent/               # Agent 计划、执行器、安全策略、记忆和审查
│   ├── rag/                 # ASK、LLM 客户端、prompt、回答生成、引用校验和 trace
│   ├── repository/          # 文件加载、代码切片和仓库结构索引
│   └── tools/               # Hybrid Search、分词、召回、过滤和只读工具
├── scripts/                 # 离线脚本：检索评测、LeetCode 兼容补丁、数据下载
├── datasets/
│   ├── eval/                # 评测集，支持 JSON 数组和 JSONL trace 两种格式
│   └── leetcode_reference/  # LeetCode 本地类型参考
├── artifacts/               # 运行后生成的 trace、评测结果等临时产物
├── tests/                   # 单元测试
├── main.py                  # 命令行入口
├── frontend.py              # 浏览器前端
├── web_ui.py                # Tkinter 桌面界面
└── leetcode_types.py        # 本地运行 LeetCode 代码时的类型补丁
```

阅读代码时，优先从 `main.py`、`coding_rag/agent/`、`coding_rag/rag/`、`coding_rag/tools/` 和 `coding_rag/repository/` 开始。

## 安装

```bash
pip install -r requirements.txt
```

## 命令行检索

浏览器前端：

```bash
python frontend.py
```

打开 `http://127.0.0.1:8765`，输入仓库路径和问题即可运行 RAG；勾选 `调用大模型回答` 后会基于检索结果生成答案。

在界面顶部选择 **ASK** 或 **Agent** 工作流模式：

- **ASK 模式**：调用 `/api/ask`，检索代码上下文并可选由 LLM 生成回答，适合代码库问答。
- **Agent 模式**：调用 `/api/agent-plan`，生成 ReAct Agent 工作流计划（包含工具接口、候选文件和验证步骤），可选启用 LLM 生成详细计划和审查。

Agent 模式请求示例：

```json
{
  "repo_path": ".",
  "query": "实现 load_data 的缓存功能",
  "top_k": 5,
  "recall_window": 2,
  "use_llm": true,
  "provider": "deepseek",
  "review_agent_plan": false
}
```

响应包含 `plan`、`plan_text`（LLM 计划）、`prompt`（LLM prompt）、`tools`（工具接口列表）、`stats`（检索统计）和可选的 `review`（计划审查结果）。

最简单的交互式入口：

```bash
python main.py
```

按提示输入仓库路径，之后连续输入问题或需求。只指定仓库也可以直接进入交互模式：

```bash
python main.py .
python main.py data\repos\django
```

交互模式会先询问是否调用大模型回答。选择 `y` 后会读取 `.env` 并用 RAG 检索结果生成答案；选择 `n` 则只展示检索到的代码块。

基本用法：

```bash
python main.py /path/to/repo "你的问题" --top-k 5
```

示例：

```bash
python main.py . "BM25 检索器在哪里建立索引？" --top-k 5
```

默认检索流程：

1. `coding_rag.repository.files` 递归读取 Python 文件，并跳过 `.git`、虚拟环境、缓存目录。
2. `coding_rag.repository.chunks` 按行号把文件切成 `CodeChunk`。
3. `coding_rag.tools.bm25.BM25Retriever` 建立正文 BM25、路径/符号 BM25 和 token 覆盖度信号，并融合检索种子代码块。
4. `coding_rag.tools.recall` 召回同文件相邻代码块，补足上下文。
5. `coding_rag.tools.filter` 重新评分并保留最终上下文。
6. `coding_rag.rag.prompt` 在发送给 LLM 前合并同文件连续/重叠代码块，减少重复上下文。
7. CLI 打印代码块；如果开启 `--llm`，再把上下文交给大模型回答。

常用参数：

```bash
python main.py . "代码在哪里切成 chunk？" --top-k 5 --recall-window 2
python main.py . "代码在哪里切成 chunk？" --candidate-k 20 --max-recall-results 30
python main.py . "代码在哪里切成 chunk？" --final-k 5 --min-final-score 0.1
python main.py . "代码在哪里切成 chunk？" --show-tokens
python main.py . "代码在哪里切成 chunk？" --show-trace
python main.py . "代码在哪里切成 chunk？" --trace-out artifacts/query_trace.json
```

分词器会同时照顾代码标识符、中文查询和常见中英代码检索词：

- `build_binary_tree` 会保留完整词，并拆成 `build`、`binary`、`tree`
- `twoSum` 会拆成 `twosum`、`two`、`sum`
- 中文查询会展开成短 n-gram，方便匹配中文注释和问题
- `检索`、`评测`、`文件` 等中文词会扩展出 `retrieval`、`eval`、`file` 等通用英文检索词，方便匹配英文路径和符号

## 大模型回答

开启 `--llm` 后，系统会把检索结果整理成 prompt，并调用 OpenAI 兼容的 `/chat/completions` 接口。第一次运行时会自动创建 `.env` 模板。

DeepSeek 示例配置：

```dotenv
DEEPSEEK_API_KEY=your_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY_ENV=DEEPSEEK_API_KEY
```

支持的预设：

| 服务商 | 默认模型 | API Key 环境变量 |
| --- | --- | --- |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `qwen` | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `kimi` | `kimi-k2.6` | `MOONSHOT_API_KEY` |
| `zhipu` | `glm-4.7` | `ZHIPU_API_KEY` |

生成模式：

| 模式 | 作用 |
| --- | --- |
| `judge` | 判断检索结果相关性并回答问题 |
| `code-understand` | 深入解释代码实现 |
| `code-generate` | 基于参考代码生成实现 |
| `leetcode` | 生成 LeetCode 风格解法 |
| `api` | 生成 API 使用示例 |

示例：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py . "BM25 检索模块整体做了什么？" --top-k 5 --llm --llm-provider deepseek
python main.py . "解释代码切片逻辑" --llm --mode code-understand
python main.py . "生成一个使用 BM25Retriever 的示例" --llm --mode api
```

自定义 OpenAI 兼容接口：

```powershell
$env:LLM_API_KEY="your_key"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="your-model"
python main.py . "linked list cycle" --llm --llm-provider custom
```

## Agent 工作流

新的 ReAct Agent 计划模式只做计划，不执行工具调用：

```bash
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --workflow-mode agent
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --workflow-mode agent --show-agent-prompt
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --workflow-mode agent --llm --llm-provider deepseek
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --workflow-mode agent --llm --review-agent-plan
```

当前计划模式会完成任务分析、RAG 上下文检索、工具接口声明，并把这些内容整理成 LLM 计划 prompt。工具接口已经声明，但还不会实际调用工具或修改代码。
`--review-agent-plan` 会在生成计划后再次调用 LLM 审查计划是否合理，并输出评分、问题和改进建议。

Agent 计划测试集可从 SWE-bench Verified 导入：

```bash
python scripts/import_agent_plan_cases.py --limit 50 --output datasets/eval/agent_plan_cases.json
python -m unittest tests.test_task_planner
```

如果希望从 `用户任务` 出发，让 RepoPilot 自动读取记忆、检索上下文、生成实施计划，并记录本次经验：

```bash
python main.py . "给 RepoPilot 增加一个新的检索优化点" --agent
```

启用 LLM 后，Agent 会基于检索上下文和历史记忆生成更具体的实现草案：

```bash
python main.py . "给 RepoPilot 增加一个新的检索优化点" --agent --llm --llm-provider deepseek
```

常用参数：

```bash
python main.py . "优化 trace 分析" --agent --show-trace
python main.py . "优化 trace 分析" --agent --trace-out artifacts/agent_run.json
python main.py . "优化 trace 分析" --agent --agent-memory artifacts/agent_memory.jsonl
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --agent-exec --llm
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --agent-exec --dry-run
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --agent-exec --safe-mode
python main.py . "修复 ASK 模式没有调用大模型回答的问题" --agent-exec --agent-policy .repopilot/policy.json
```

Agent executor 安全策略可通过 `.repopilot/policy.json` 或 `--agent-policy PATH` 配置。默认 denylist 始终优先，dry-run/safe-mode 会强制 patch 不落盘、命令不执行。

Agent 默认记忆路径是 `artifacts/agent_memory.jsonl`。每条记忆会记录任务、状态、摘要、相关文件和关键决策；下一次 Agent 运行会按任务相关性读取近期记忆。相同任务和相同文件集合的记忆会自动跳过，避免重复污染记忆库。

Agent 输出默认包含 `Agent 运行轨迹`，展示每一步的状态和关键产物；加上 `--show-trace` 后，还会显示更详细的 RAG 检索轨迹。Agent 还会生成任务画像，例如 `feature`、`bugfix`、`optimization`、`evaluation`、`docs`、`test`，并据此给出更贴近任务类型的验证命令。

当前 Agent 工作流：

1. 接收任务：抽取目标、约束和交付物。
2. 读取记忆：检索历史任务中的相关文件、决策和踩坑。
3. 检索上下文：用 Hybrid Search 找代码块，并召回相邻上下文。
4. 规划修改：生成候选文件、执行步骤和风险点。
5. 生成草案：未启用 LLM 时输出离线计划；启用 LLM 时生成实施草案。
6. 验证建议：给出应运行的测试、编译和评测。
7. 写入记忆：把本次任务摘要和决策保存为长期记忆。

## 桌面界面

如果想用图形界面调整参数和查看结果：

```bash
python web_ui.py
```

界面支持选择仓库、输入问题、调整检索参数、开启 LLM、运行评测、导出 trace、查看 bad case 和自动调参。

## 检索评测

运行示例评测集：

```bash
python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --top-k 5
```

运行 20 条 trace 测试集：

```bash
python scripts/retrieval_eval.py . datasets/eval/trace.jsonl --top-k 5 --trace-out artifacts/trace_eval_trace.jsonl
```

保存完整检索 trace：

```bash
python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --trace-out artifacts/retrieval_trace.jsonl
```

Trace 中会包含 `trajectory` 字段，用来观察 `initial_search`、`neighbor_recall`、`final_filter` 和 `context_compaction` 四个阶段的文件、行号、分数、来源和预览文本。单次查询也可以用 `main.py --show-trace` 直接打印可读轨迹。

执行 bad case 归因和参数搜索：

```bash
python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --optimize
```

Agent 计划离线评测：

```bash
python scripts/agent_eval.py --evalset datasets/eval/agent_plan_cases.json --trace-out artifacts/agent_eval_trace.jsonl --summary-out artifacts/agent_eval_summary.json
```

生成函数地图：

```bash
python scripts/generate_function_map.py --output FUNCTION_MAP.md
```

评测集可以写成 JSON 数组：

```json
[
  {"id": "case-1", "query": "BM25 检索器在哪里建立索引？", "relevant": ["coding_rag/tools/bm25.py"]}
]
```

也可以写成 JSONL，每行一个问题：

```json
{"id": "q001", "question": "代码文件是在哪里被读取的？", "gold_files": ["coding_rag/repository/files.py"]}
```

## LeetCode 数据

LeetCode 的判题环境内置 `ListNode`、`TreeNode` 和多种 `Node` 类型。本项目提供 `leetcode_types.py` 作为本地补丁。下载 LeetCode 代码后可运行：

```bash
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

## 测试

```bash
python -m unittest
python -m compileall coding_rag main.py frontend.py web_ui.py scripts tests
```
