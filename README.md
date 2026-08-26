# RepoPilot：轻量级代码 RAG 检索工具

RepoPilot 是一个用于学习和实验的代码检索项目。它会读取 Python 仓库，由 LLM 自动决定直接回答或调用 Hybrid Search、文件读取、补丁和验证命令等工具。

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

## 统一入口

浏览器前端：

```bash
python frontend.py
```

打开 `http://127.0.0.1:8765` 后直接输入问题或修改需求。普通 Agent 的主运行时由 LangGraph 编排，LLM 会自动决定直接回答，还是调用 `search_code`、`read_file`、`apply_patch`、`run_command`、`inspect_diff`。

浏览器默认不允许执行写工具。需要真实应用补丁或执行命令时，先以写能力启动服务；模型提出写调用后，页面会列出目标文件，并要求用户只批准当前这一次调用：

```bash
python frontend.py --allow-tool-execution
```

统一 API 为 `POST /api/run`：

```json
{
  "repo_path": ".",
  "query": "实现 load_data 的缓存功能",
  "top_k": 5,
  "recall_window": 2,
  "provider": "deepseek",
  "mode": "judge"
}
```

响应包含 `status`、`answer`、`execution`、`approval`、`approval_id`、`summary` 和统一 `trace`。`approval_id` 只在服务允许写入且当前状态为 `approval_required` 时返回；把它单独提交到 `/api/run` 只会批准对应调用一次。旧 `/api/ask`、`/api/agent-plan` 端点不再提供。

### Learning Mode 浏览器教学

首页的“项目学习”模式会基于仓库源码生成 4 到 6 步复现路线，并按 `start -> submit -> reflect` 一次推进一个人机等待点。当前步骤会展示学习目标、任务、原因、收益、验收项、源码证据、常见问题和优化问题。

Learning Mode 使用独立的 `POST /api/learning/session`，不会改变 `/api/run` 的普通问答与工具授权契约：

```json
{
  "repo_path": ".",
  "provider": "deepseek",
  "top_k": 8,
  "recall_window": 2,
  "request": {
    "action": "start",
    "learning_goal": "从零复现 RepoPilot 的只读检索链路",
    "learner_level": "beginner"
  }
}
```

后续 `submit` 或 `reflect` 请求要在 `request` 中原样带回上一轮返回的完整 `session`。接口返回 `status`、`session`、`current_step`、`next_action`、`review`、`feedback` 和 `trace`；业务失败也使用这个稳定结构，因此携带有效 session 时可以停在原等待点继续修改。

教学模式始终只读：学习者粘贴的实现说明、命令和测试输出只作为报告证据，不会被执行，也不受 `--allow-tool-execution` 影响。session 只保存在当前页面内存，页面刷新后重新开始；服务进程重启会更换临时 HMAC 密钥，使旧 session 失效。当前本地 HTTP 服务没有 TLS、鉴权或多用户隔离，不应直接暴露到公网。

最简单的交互式入口：

```bash
python main.py
```

按提示输入仓库路径，之后连续输入问题或需求。只指定仓库也可以直接进入交互模式：

```bash
python main.py .
python main.py data\repos\django
```

交互模式始终调用 LLM。系统只读取已有 `.env` 或环境变量，不会自动创建配置文件。

基本用法：

```bash
python main.py /path/to/repo "你的问题" --top-k 5
```

示例：

```bash
python main.py . "BM25 检索器在哪里建立索引？" --top-k 5
python main.py . "修复统一入口的 trace 状态"
```

默认检索流程：

1. `coding_rag.repository.files` 递归读取 Python 文件，并跳过 `.git`、虚拟环境、缓存目录。
2. `coding_rag.repository.chunks` 按行号把文件切成 `CodeChunk`。
3. `coding_rag.tools.bm25.BM25Retriever` 建立正文 BM25、路径/符号 BM25 和 token 覆盖度信号，并融合检索种子代码块。
4. `coding_rag.tools.recall` 召回同文件相邻代码块，补足上下文。
5. `coding_rag.tools.filter` 重新评分并保留最终上下文。
6. `coding_rag.rag.prompt` 在发送给 LLM 前合并同文件连续/重叠代码块，减少重复上下文。
7. LLM 根据已有观察决定继续调用工具或生成最终回答。

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

统一入口必须配置 LLM，并调用 OpenAI 兼容的 `/chat/completions` 接口。请手动准备 `.env` 或环境变量；程序不会自动创建或改写 `.env`。

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
python main.py . "BM25 检索模块整体做了什么？" --top-k 5 --llm-provider deepseek
python main.py . "解释代码切片逻辑" --mode code-understand
python main.py . "生成一个使用 BM25Retriever 的示例" --mode api
```

自定义 OpenAI 兼容接口：

```powershell
$env:LLM_API_KEY="your_key"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="your-model"
python main.py . "linked list cycle" --llm-provider custom
```

## Agent 工作流

用户不再选择 ASK 或 Agent。CLI 与浏览器都通过 `run_unified_query()` 进入普通 Agent 的 LangGraph；图中的 `model -> tool -> model` 条件循环处理每轮 `answer` 或 `tool` JSON 决策，最终进入 `finalize` 或安全失败节点：

```bash
python main.py . "解释统一 trace 的事件顺序" --show-trace
python main.py . "修复统一 trace 的状态计算" --trace-out artifacts/unified_run.json
python main.py . "修复统一 trace 的状态计算"
python main.py . "修复统一 trace 的状态计算" --agent-policy .repopilot/policy.json
```

`apply_patch` 和 `run_command` 没有 dry-run 分支。每个写调用先经过路径、补丁、allowlist/denylist 等安全检查，再进入 `approval_required`；CLI/浏览器会同时展示目标文件、具体命令或补丁正文。授权绑定工具名和完整参数的 SHA-256 指纹，只能消费一次。批准后，执行器先在仓库外缓存声明文件的修改前版本，再真实执行并返回操作专属 diff 与快照 ID。`safe_mode` 仍可强制拒绝写调用，denylist 始终优先。

CLI 会在每次写调用前提示确认。需要恢复时使用执行结果中的快照 ID：

```bash
python main.py /path/to/repo --rollback SNAPSHOT_ID
```

浏览器会在修改后直接展示 diff，并提供“恢复到修改前”按钮。`run_command` 必须在 `affected_files` 中声明所有可能修改的仓库相对路径；快照与回滚保证只覆盖这些已声明文件。

普通 Agent 的 LangChain 工具适配层公开五个严格命名工具；实际仓库访问仍统一进入 `AgentExecutor.call()`。Learning Mode 使用另一套只读图和工具表，只绑定 `search_code`、`read_file`，不会因普通 Agent 获得写工具而改变权限。

统一 trace 依次记录 `model_decision`、`approval_required`、工具事件和 `final_answer`，包含耗时、检索阶段、输出截断、安全决策、快照、修改后 diff 和命令结果。状态可能是 `success`、`approval_required`、`partial` 或 `failed`。

旧 Agent 计划与审查函数继续供离线评测使用，不再作为 CLI 或浏览器工作流。计划测试集仍可通过以下命令维护：

```bash
python scripts/import_agent_plan_cases.py --limit 50 --output datasets/eval/agent_plan_cases.json
python -m unittest tests.test_task_planner
```

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
