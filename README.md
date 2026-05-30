# RepoPilot：轻量级代码 RAG 检索工具

RepoPilot 是一个用于学习和实验的代码检索项目。它会读取一个 Python 仓库，把代码切成带行号的片段，用 Hybrid Search 找出和问题最相关的代码块，并可选调用大模型生成回答。

## 项目结构

```text
.
├── coding_rag/              # 核心检索流程：加载、切片、分词、Hybrid Search、召回、过滤、LLM 客户端
├── rag/                     # 面向大模型的提示词模板和答案生成器
├── scripts/                 # 离线脚本：检索评测、LeetCode 兼容补丁、数据下载
├── datasets/
│   ├── eval/                # 评测集，支持 JSON 数组和 JSONL trace 两种格式
│   └── leetcode_reference/  # LeetCode 本地类型参考
├── artifacts/               # 运行后生成的 trace、评测结果等临时产物
├── legacy/                  # 早期原型代码，仅保留作参考
├── tests/                   # 单元测试
├── main.py                  # 命令行入口
├── web_ui.py                # Tkinter 桌面界面
└── leetcode_types.py        # 本地运行 LeetCode 代码时的类型补丁
```

阅读代码时，优先从 `main.py` 和 `coding_rag/` 开始。`legacy/` 只是旧版本原型，不参与当前主流程。

## 安装

```bash
pip install -r requirements.txt
```

## 命令行检索

基本用法：

```bash
python main.py /path/to/repo "你的问题" --top-k 5
```

示例：

```bash
python main.py . "BM25 检索器在哪里建立索引？" --top-k 5
```

默认检索流程：

1. `file_loader` 递归读取 Python 文件，并跳过 `.git`、虚拟环境、缓存目录。
2. `code_splitter` 按行号把文件切成 `CodeChunk`。
3. `BM25Retriever` 建立正文 BM25、路径/符号 BM25 和 token 覆盖度信号，并融合检索种子代码块。
4. `context_recaller` 召回同文件相邻代码块，补足上下文。
5. `result_filter` 重新评分并保留最终上下文。
6. `rag.prompt` 在发送给 LLM 前合并同文件连续/重叠代码块，减少重复上下文。
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

如果希望从“用户任务”出发，让 RepoPilot 自动读取记忆、检索上下文、生成实施计划，并记录本次经验：

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
```

Agent 默认记忆路径是 `artifacts/agent_memory.jsonl`。每条记忆会记录任务、状态、摘要、相关文件和关键决策；下一次 Agent 运行会按任务相关性读取近期记忆。相同任务和相同文件集合的记忆会自动跳过，避免重复污染记忆库。

Agent 输出默认包含“Agent 运行轨迹”，展示每一步的状态和关键产物；加上 `--show-trace` 后，还会显示更详细的 RAG 检索轨迹。Agent 还会生成任务画像，例如 `feature`、`bugfix`、`optimization`、`evaluation`、`docs`、`test`，并据此给出更贴近任务类型的验证命令。

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

评测集可以写成 JSON 数组：

```json
[
  {"id": "case-1", "query": "BM25 检索器在哪里建立索引？", "relevant": ["coding_rag/bm25_retriever.py"]}
]
```

也可以写成 JSONL，每行一个问题：

```json
{"id": "q001", "question": "代码文件是在哪里被读取的？", "gold_files": ["coding_rag/file_loader.py"]}
```

## LeetCode 数据

LeetCode 的判题环境内置 `ListNode`、`TreeNode` 和多种 `Node` 类型。本项目提供 `leetcode_types.py` 作为本地补丁。下载 LeetCode 代码后可运行：

```bash
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

## 测试

```bash
python -m unittest
python -m compileall coding_rag rag scripts main.py web_ui.py tests
```
