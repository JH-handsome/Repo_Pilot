# RepoPilot：基于 LangGraph 的本地代码助手

RepoPilot 帮助你在本地多语言仓库中定位代码、理解实现，并在逐次确认后执行修改。
你可以直接描述问题，模型通过 LangGraph 调用代码搜索、文件读取等工具，依据源码给出回答；
需要应用补丁或执行命令时，先展示操作内容，等待你批准。

项目同时提供多轮代码问答、项目复现教学和检索评测，适合阅读陌生仓库，
也适合学习代码 RAG、工具调用和 Agent 工作流的实现。

## 能做什么

| 能力 | 当前实现 |
| --- | --- |
| 代码定位与解释 | 融合正文 BM25、路径与符号信息及结构评分，召回邻近代码，并提供文件与行号引用。 |
| 多轮对话 | 按仓库和会话 ID 隔离历史，使用本地 SQLite 保存完成轮次；浏览器刷新可继续，CLI 和桌面入口可通过原会话 ID 恢复。 |
| 确认后修改 | 每次补丁或命令分别申请批准；执行后展示 diff，支持对声明文件创建快照并回滚。 |
| 项目复现教学 | 根据源码生成分步学习路线，审查你提交的实现说明和测试结果，再通过反思推进；教学模式始终只读。 |
| 运行记录与评测 | 查看模型决策、工具结果和检索阶段；普通 CLI／浏览器运行支持 SQLite trace 存储，另提供离线检索与规划评测脚本。 |

## 选择使用方式

- **Codex 内直接操作**：使用下文的 `scripts/repo_inspect.py` 只读检索／源码核验命令，以及 `scripts/quality_check.py` 查看量化离线门槛；不必启动额外页面，也不会因此调用模型。
- **浏览器**：运行 `python frontend.py`，访问 [本地页面](http://127.0.0.1:8765)，使用代码问答或项目学习。
- **命令行**：运行 `python main.py <仓库路径>`，连续输入问题或需求；也可以直接传入一个问题。
- **桌面界面**：运行 `python web_ui.py`，选择仓库、调整检索参数、查询代码或运行评测。

开始前需安装依赖，并配置兼容 OpenAI Chat Completions 的模型服务，具体见下文。
代码索引支持各类文本源码；检索本身不要求部署向量数据库。
本地保存的是聊天记录和运行数据，模型调用仍会把所需问题及代码上下文发送到你配置的服务。
普通聊天与项目学习采用不同的会话机制：聊天支持持久化恢复，教学页面刷新后重新开始。

### 语言支持

浏览器、命令行、桌面界面和项目学习均可读取多语言仓库，包括 JavaScript／TypeScript、
Vue／Svelte、Java、Go、Rust、C／C++、C#、SQL、Shell、PowerShell 等。
文件发现不使用语言扩展名白名单：未知扩展名及 Dockerfile、Makefile 等无扩展名源码也能按文本检索，
JSON／YAML／TOML 等配置同样参与检索。代码上下文按实际语言标注，未知语言标为普通文本。

支持 UTF-8（含 BOM）和带 BOM 的 UTF-16；单文件上限 2 MiB。
二进制、不可解码文件、常见文档／日志／锁文件、依赖及构建目录、敏感路径不会进入源码索引；
`CMakeLists.txt` 和 `requirements.txt` 作为构建配置保留。默认检索还排除仓库根目录下的
`data`、`datasets`、`artifacts`，避免把数据与运行产物当成项目实现。
文件增删或修改会刷新索引；旧版索引缓存自动失效并重建。

Python 保留 AST 符号定位与静态调用链分析。其他语言目前使用文本检索、文件读取与模型分析，
尚未提供语法级符号／调用链导航；`outline`、`locate`、`symbol`、`trace` 等结构工具仍面向 Python。
各语言代码补丁均走原有逐次审批流程；常见 npm／pnpm／yarn、Go、Cargo、.NET、Maven／Gradle
测试或构建命令也必须逐次批准，实际执行需要本机安装对应工具链。

## 项目结构

```text
.
├── coding_rag/
│   ├── agent/               # Agent 计划、执行器、安全策略、记忆和审查
│   ├── learning/            # 只读项目教学、分步路线与学习会话
│   ├── rag/                 # 检索流水线、LLM 客户端、提示、引用校验和 trace
│   ├── repository/          # 文件加载、代码切片和仓库结构索引
│   ├── storage/             # SQLite trace 存储、过滤和失败补录
│   └── tools/               # Hybrid Search、分词、召回、过滤和只读工具
├── scripts/                 # 离线脚本：检索评测、LeetCode 兼容补丁、数据下载
├── datasets/
│   ├── eval/                # 评测集，支持 JSON 数组和 JSONL trace 两种格式
│   └── leetcode_reference/  # LeetCode 本地类型参考
├── artifacts/               # 本地聊天历史、运行 trace 与评测结果
├── tests/                   # 单元测试
├── main.py                  # 命令行入口
├── frontend.py              # 浏览器前端
├── web_ui.py                # Tkinter 桌面界面
└── leetcode_types.py        # 本地运行 LeetCode 代码时的类型补丁
```

阅读代码时，优先从 `main.py`、`coding_rag/agent/`、`coding_rag/rag/`、`coding_rag/tools/` 和 `coding_rag/repository/` 开始。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 统一入口

浏览器前端：

```bash
python frontend.py
```

打开 `http://127.0.0.1:8765` 后直接输入问题或修改需求。普通 Agent 的主运行时由 LangGraph 编排，LLM 会自动决定直接回答，还是调用 `search_code`、`read_file`、`trace_call_path`、`apply_patch`、`run_command`、`inspect_diff`。

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
  "provider": "deepseek"
}
```

响应包含 `status`、`answer`、`execution`、`approval`、`approval_id`、`summary`、`conversation` 和统一 `trace`。普通多轮聊天需复用 `conversation.session_id`，详见文末的历史与恢复说明。`approval_id` 只在服务允许写入且当前状态为 `approval_required` 时返回；把它单独提交到 `/api/run` 只会批准对应调用一次。旧 `/api/ask`、`/api/agent-plan` 端点不再提供。

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

如果直接在 Codex 里只需代码证据、不调用外部模型或页面，可运行同一套混合检索／仓库读取策略的只读命令：

```bash
python scripts/repo_inspect.py . search "调用链入口在哪里" --top-k 5
python scripts/repo_inspect.py data/repos/aider discover "How are lint error line numbers converted?" --top-k 5 --per-file 3
python scripts/repo_inspect.py . related "哪个调用把入口连接到跨文件 helper" --limit 3
python scripts/repo_inspect.py data/repos/aider related "哪个导入函数解析启动参数" --source-path aider/main.py --source-symbol main --limit 3
python scripts/repo_inspect.py . outline coding_rag/tools/agent_readonly.py --limit 50
python scripts/repo_inspect.py data/repos/aider locate aider/linter.py "How are error line numbers converted to zero-based positions?" --limit 3
python scripts/repo_inspect.py . symbol ReadOnlyAgentTools.trace_call_path
python scripts/repo_inspect.py . trace coding_rag/tools/agent_readonly.py ReadOnlyAgentTools.trace_call_path
python scripts/repo_inspect.py data/repos/aider verify-edge aider/linter.py Linter.run_cmd aider/run_cmd.py run_cmd_subprocess
python scripts/repo_inspect.py . read coding_rag/tools/agent_readonly.py --start-line 180 --end-line 220
```

命令只开放 `search`、候选调用边 `related`、跨文件候选 `discover`、单文件 `outline`／`locate`、精确 `symbol`、`trace`、`verify-edge` 和最多 80 行的 `read`，输出有界 JSON；不加载 API key、不请求模型、不执行仓库写工具。搜索／精确符号定位的索引仍可能写入仓库外的用户缓存；当默认缓存位置不可写时，可在仓库路径后加 `--cache-dir <仓库外可写目录>`，让多次 Codex 命令复用同一份安全校验的索引快照。`outline` 只解析指定 Python 文件，可在搜索找到入口文件后快速取得完整嵌套符号名，再交给 `trace`。`locate` 在已选中的一个 Python 文件内按问题排列真实函数定义；`discover` 则将检索前列文件与各文件的函数候选分组显示，保留文件检索顺序，不跨文件比较局部分数。两者均不保证首位候选就是答案，需用 `read` 核对定义行。`related` 从已命中的源码文件里列出可静态解析、目标定义存在的跨文件调用边，附入口调用行与目标定义行；排序只是候选线索，必须再用 `read` 核对两处源码，不能直接当成回答或替代模型推理。固定 Django 和 Aider 的单文件负例各有 4/4 收到无关跨文件建议；Aider 另 4 条真实同文件调用题也误导 4/4，说明只靠调用动词无法可靠拒答。自仓库跨文件烟测的精确边首位仅 1/4，因此该命令不用于单文件问题，也不自动进入 Agent 的默认回答链路。重复定义的符号仍按静态歧义策略拒绝选择。`symbol` 从同一索引精确定位定义，适合已知标识符但普通 Top-K 被测试／评测文件占据的情况；它不改变默认搜索排序。

JSON stdout 使用 ASCII 转义保存 Unicode，Windows GBK 终端与 UTF-8 消费端都能无损解析中文源码；读取时应解析 JSON，而不是把转义序列当作源码文本。

连续排查同一仓库时，可启动本地只读会话，避免每条命令都重新构建索引：

```text
python scripts/repo_inspect.py data/repos/django serve
["discover", "Where does BaseHandler resolve an incoming URL?"]
["trace", "django/core/handlers/base.py", "BaseHandler.resolve_request"]
```

第一行启动进程，后两行分别作为一行 JSON argv 发到其标准输入；每行返回 `{"ok": true, "result": ...}` 或有界错误。会话只接受上列只读工具，不开放写入或执行命令，也不启动网络服务。它复用进程内索引，但每次请求仍扫描仓库变更；输入结束即退出。即使使用 `--cache-dir`，也应选择仓库外或已排除的 `artifacts` 目录，因为索引快照包含源码。

在 Codex 中可用一条命令运行当前离线质量门槛，无需打开额外页面：

```bash
python scripts/quality_check.py
```

命令串行执行单元测试、四项核心门槛，以及固定 Aider／Django 双文件检索和显式缺失定义拒给候选门槛；最终输出逐项 JSON 指标，任一项失败即返回非零退出码。扩展门槛需要按 `datasets/README.md` 准备固定的 Aider、Django 源码；只检查核心项可用 `--core-only`。该命令不请求模型服务：Agent 指标来自离线脚本策略，补丁 100 分仅是 oracle 评测框架自检，已揭盲题集仅可作回归检查，不是模型编程能力或新题泛化证明。

若已通过 `search`／`symbol`／`outline` 确认入口，可给 `related` 同时传入 `--source-path` 和 `--source-symbol`：只枚举该函数的跨文件调用，不再从 Top5 其他文件的所有函数泛化。固定 Aider 组件诊断中，4 条跨文件正例精确边 Top1 为 4/4，8 条单文件负例中 7 条返回空候选；但这使用人工审定的正确入口，且 `main()` 同时有许多真实跨文件调用，剩余同文件问题仍会收到误导建议。空候选只表示“该入口没有可静态解析的跨文件边”，不能证明运行时绝无跨文件调用。

如果还没有确认入口，可显式试 `python scripts/repo_inspect.py data/repos/django related "When BaseHandler resolves a request URL, where is the resolver implemented?" --auto-anchor`。此模式只取检索 Top-5 中首个非测试源码文件的首位函数，列出该函数可静态解析的跨文件边；输出的 `auto_source_candidate` 是未核验的入口线索，歧义或无候选时弃权，返回空边不代表目标不存在。它不改变普通 `related`、默认 Agent 或检索排序。固定 Aider／Django 已揭盲题的首次诊断为精确边 Top1 5/8、单文件／同文件负例无跨文件建议 12/12；普通未锚定模式在同 12 题中均给出无关建议。部分正例的入口文件不在 Top-5，自动模式仍会漏检；身份、逐题结果和限制见 `benchmarks/query-only-auto-source-first-look-v1.json`。使用边前仍须读取两处源码。

自然语言入口定位另用 `python scripts/source_discovery_eval.py --out <scorecard.json>` 离线诊断。固定 Aider／Django 共 20 条已揭盲题，正确入口文件进入 Top5 为 19/20；正确函数定义行进入 Top5 仅 12/20，Top20 仍仅 12/20。较宽松的函数范围相交 Top5 为 14/20，可能被长函数高估。这将“找到文件”与“找到函数定义”分开计分；定义行可见也不等于自动选中了正确函数，更不是模型修复能力。当前优先改善文件内入口定位，不将人工入口组件率并入端到端分数。

对已经选定的文件使用 `locate`，同一组 20 题在人工给定正确文件时，正确符号 Top1 为 15/20、Top3 为 19/20；入口文件 Top5 与文件内符号 Top3 同时满足为 18/20。这是候选可达率上界，仍未测出系统能否自动从 Top5 文件中选对文件。结果见 `benchmarks/file-local-symbol-first-look-v1.json`；题目已揭盲，暂不作为默认 Agent 排序信号。

`discover` 的无人工入口联合诊断在相同 20 题中，正确“文件＋函数”实际出现在 5 组文件、每组至多 3 个函数的候选里为 18/20，但第一组第一个候选只对 10/20，Django 子集仅 2/8。按输出顺序逐项核验时，前 3、5、10 个候选分别覆盖 13、17、18 题；这是已揭盲题上的人工核验成本，不是自动选对率。输出预算不足时，`discover` 优先保留不同文件的首位函数，再裁剪低优先级函数和整组文件，并标记 `truncated`。报告见 `benchmarks/joint-source-discovery-first-look-v1.json`，当前逐题诊断可由 `scripts/source_discovery_eval.py` 重算；默认 Agent 回答链路未改。

如果问题问的是整个类／组件而非某个方法，可显式运行 `python scripts/repo_inspect.py data/repos/django discover "Where is URLValidator implemented?" --kind class`，或对已选文件使用 `locate <path> <query> --kind class`；默认仍只列函数，`--kind all` 才混合列出类与函数。类候选按真实 AST 定义定位；已揭盲的 Django URLValidator 问题可把定义列在首位，复合认证后端问题可列出 ModelBackend。但有些问题同时涉及多个类／方法，薄封装也可能排在实际实现前面，因此必须继续读取相关方法后才能回答行为问题。这是人工导航能力，不是自动选类或最终答题能力门禁。

当问题明确要求定位一个 `Class.method` 定义，且该类在索引中存在、该精确方法却没有直接 AST 定义时，`discover` 返回 `abstained: true`、空候选和缺失名称，而不是拿近邻方法冒充结果；提示会说明它仍可能由继承或动态机制提供。例如 `python scripts/repo_inspect.py data/repos/aider discover "Locate RepoMap.get_incremental_tags_map"`。`python scripts/discovery_absence_eval.py --gate --out <scorecard.json>` 在固定的 8 条此类缺失题及 20 条原有正例上检查回归：缺失题拒给候选 8/8、正例误拒 0/20。身份和限制见 `benchmarks/discovery-absence-baseline-v1.json`。这不解决普通自然语言问题的不可回答判断，也不代表模型会正确拒答。

已知类方法入口时，可显式加 `--same-class` 让 `trace` 列出 `self.method()` 且目标在同一个类中有定义的 `same_class_method_candidate`，例如 `python scripts/repo_inspect.py data/repos/aider trace aider/repomap.py RepoMap.get_repo_map --same-class` 可定位本文件第 135 行调用与 `RepoMap.get_ranked_tags_map` 的定义行。该边仅是语法级候选：子类覆写等动态分派仍可能改变运行时目标；默认 `trace_call_path` 与离线跨文件自动评测保持原有行为。

当 `trace` 的静态边为空或不完整时，可显式加 `--unresolved` 查看入口函数内尚未解析的点号调用及调用行，例如 `python scripts/repo_inspect.py data/repos/aider trace aider/commands.py Commands.cmd_lint --unresolved --call-contains lint` 仅列出匹配 `lint` 的未解析调用，其中第 379 行是 `self.coder.linter.lint`。`--call-contains` 是大小写不敏感的文字筛选，不推断被调文件、类型或运行时目标；应继续读取调用点，再显式用 `symbol Linter.lint` 定位候选定义并核对对象来源。可与 `--same-class` 组合使用。曾尝试按属性名唯一匹配类方法，但在 8 条已揭盲同文件负例中有 1 条收到无关跨文件建议，故未上线；实验见 `benchmarks/unresolved-name-hint-rejection-v1.json`。默认 Agent 工具输出不增加这些线索。

已确认类方法入口后，还可显式运行 `python scripts/repo_inspect.py data/repos/aider trace aider/commands.py Commands.cmd_web --assigned-instance`。仅当同一方法里把已导入的类构造结果赋给 `self.属性`、该属性没有再次写入，且后续出现 `self.属性.方法()` 时，输出独立的 `assigned_instance_candidates`，包含赋值行、调用行与目标定义行。固定 Aider 的 `/web`、`/help` 及 GUI 网页抓取入口分别能提示 `Scraper.scrape`、`Help.ask`、`Scraper.scrape`。它不是已证实的调用边：条件分支、预存属性与运行时分派仍可能改变对象，必须读取三处源码后核验；默认搜索、`trace_call_path` 和 Agent 回答均不自动使用这些提示。

统一 Agent 的 `path:start-end` 引用只按实际送入模型的工具观察内容校验：若上下文预算删掉了 `read_file` 源码或搜索结果的代码正文，保留的路径／行号元数据不能使引用有效；被截在行中间的片段也不计为可引用源码。此校验只能证明代码行曾展示给模型，不自动证明自然语言结论正确。

工具结果的 `max_context_chars` 是返回内容的硬字符预算：即使通用兜底预览包含需 JSON 转义的引号、反斜杠或中文，也会按序列化后的实际长度裁剪；预算极小时只返回可序列化占位值，且观察元数据标明已截断，不把半段内容当作完整观察。

引用路径须与已观察路径一致，或是其无歧义的相对后缀；凭空添加目录前缀、以及同时匹配多个文件的同名短路径都会标为无效。

`verify-edge` 在已确认入口与目标文件／符号后核验**直接**静态调用。`supported_static_call` 附调用点和目标定义行的只读源码片段；`not_established` 只表示静态索引未建立该直连，不表示运行时绝对没有调用，也不能据此否认跨函数的间接调用。它不会把 `self.method()` 的语法候选当成已证实的运行时目标。`python scripts/edge_verification_eval.py --gate --out <scorecard.json>` 在固定 Django／Aider 的 7 条真实边、7 条近邻负例上分别得到 7/7 与 7/7，误支持 0，输出源码证据共 70 行；这是已揭盲的静态组件回归，不是自然语言入口发现、默认 Agent 或真实模型能力。CLI 保持人工显式调用，默认运行逻辑不变。

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
6. 普通 Agent 将工具结果整理成 observation，并在保留完整路径的前提下限制发送给模型的内容长度。
7. LLM 根据已有观察决定继续调用工具或生成最终回答。

普通 Agent 的仓库索引由进程级 `RepositoryIndexRegistry` 复用。索引键包含仓库、切片参数、retriever 版本和排序权重；每次搜索只扫描 Python 文件的相对路径、大小和 `mtime_ns` 清单，并在遍历阶段直接剪枝排除目录。清单未变化时优先命中内存，进程重启后可从操作系统用户缓存目录恢复 gzip 压缩的安全 JSON 快照。快照只保存相对路径、源码、AST 元数据和稀疏 BM25 词频状态，不使用 pickle，也不把盘符、用户名或绝对 checkout 路径写入缓存。新增、修改或删除 Python 文件时，仅重新读取和切片变化文件，未变文件的 chunk、AST 和词频状态直接复用；非 Python 文件变化不会使索引失效。写盘采用临时文件、`fsync` 和原子替换，损坏、不兼容或写入失败会安全回退且不会覆盖最后一次成功快照。同一进程仍由 single-flight 和最多 8 项的 LRU 控制并发与内存占用。

同一轮 Agent 在混合搜索后调用 `trace_call_path` 时，也复用这份索引快照及其符号元数据，不再为静态导航单独加载、切片和解析整个仓库；路径边界和原有追踪结果不变。

检索 trace 的 `artifacts.index` 会报告兼容的 `status`（`hit`、`miss` 或 `stale`），以及更明确的 `lifecycle`（`memory_hit`、`disk_hit`、`incremental` 或 `full_rebuild`）、manifest 扫描／构建／磁盘读写耗时、文件与 chunk 的复用／重建／移除数量，便于区分模型延迟与本地索引成本。默认排除 `data`、`datasets`、`artifacts` 根目录；如果这些目录本身被指定为目标仓库，其内部文件仍会正常索引。默认缓存位于 `%LOCALAPPDATA%/RepoPilot/repository_indexes`（Windows）或用户 XDG/cache 目录；测试和嵌入方可显式注入其他仓库外目录。

冷建索引的标识符分词另有最多 32,768 项、仅缓存长度不超过 128 字符标识符的进程内缓存；返回值仍是独立列表，磁盘快照格式与检索排序不变。在本机固定 Django checkout（2,918 个 Python 文件、15,846 个 chunk）的直接冷建测量中，改前 37.4 秒、改后 19.3 秒；这是单机单次观测，不是跨机器延迟承诺，也不改善仍漏检的跨文件目标。

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
- 查询阶段还会把 `审批`、`指纹`、`行号`、`目录` 等开发术语映射到 `approval`、`fingerprint`、`start_line`、`dirnames` 等英文标识符；这组扩展不进入文档索引

## 大模型回答

统一入口必须配置 LLM，并调用 OpenAI 兼容的 `/chat/completions` 接口。请手动准备 `.env` 或环境变量；程序不会自动创建或改写 `.env`。

OpenAI 兼容客户端对每次逻辑调用默认最多尝试 3 次，只重试 HTTP 408／425／429／500／502／503／504 和明确的 timeout、connection reset／abort、broken pipe。退避使用有严格上限的指数策略与抖动；HTTP 4xx 配置／鉴权错误、无效 JSON 和响应结构错误会立即失败。`LLMRetryPolicy` 可在代码配置中覆盖次数与退避参数，但次数硬上限为 5。

每次逻辑调用只生成一条 `llm_call_observation`，包含 provider、model、总 attempts、总 latency、outcome 和稳定 error class。客户端内存中最多保留最近 100 条，也支持注入 observer；观测和日志不包含 prompt、响应正文、API key、token 或 cookie。暂时性失败的 provider response body 同样不会进入公开异常或日志。

普通 Agent 的统一 LangGraph 会把可用观测绑定到对应的 `model_decision.artifacts.llm_call`；一次 HTTP 暂时性错误重试仍只算一个逻辑调用，决策 JSON 修复则是新的逻辑调用。运行 summary 汇总观测调用数、attempts、retries、outcome、总／平均 latency 和 error class。该能力是可选的：只实现 `complete()` 的通用 ChatClient 继续工作，只是没有调用级观测。观测通过当前线程的一次性消费接口关联，调用前会清除未消费值，避免复用客户端时读取上一任务记录。

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

普通请求不选择问答／Agent 或回答风格。统一 Agent 根据用户目标和工具观察自行判断：解释与定位先读取证据；实现与修复先检查代码，再申请每一次写操作的批准；意图或范围不明确时先澄清。项目学习仍是单独的结构化教学流程。

示例：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
python main.py . "BM25 检索模块整体做了什么？" --top-k 5 --llm-provider deepseek
python main.py . "解释代码切片逻辑"
python main.py . "生成一个使用 BM25Retriever 的示例"
```

自定义 OpenAI 兼容接口：

```powershell
$env:LLM_API_KEY="your_key"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="your-model"
python main.py . "linked list cycle" --llm-provider custom
```

## Agent 工作流

CLI、浏览器和启用 LLM 的 Tk 桌面查询都通过 `run_unified_query()` 进入普通 Agent 的 LangGraph；旧的 ASK 独立规划入口已移除。图中的 `model -> tool -> model` 条件循环处理每轮 `answer` 或 `tool` JSON 决策，最终进入 `finalize` 或安全失败节点：

```bash
python main.py . "解释统一 trace 的事件顺序" --show-trace
python main.py . "修复统一 trace 的状态计算" --trace-out artifacts/unified_run.json
python main.py . "修复统一 trace 的状态计算"
python main.py . "修复统一 trace 的状态计算" --agent-policy .repopilot/policy.json
```

`apply_patch` 和 `run_command` 没有 dry-run 分支。每个写调用先经过路径、补丁、allowlist/denylist 等安全检查，再进入 `approval_required`；CLI/浏览器会同时展示目标文件、具体命令或补丁正文。授权绑定工具名和完整参数的 SHA-256 指纹，只能消费一次。批准后，执行器先在仓库外缓存声明文件的修改前版本，再真实执行并返回操作专属 diff 与快照 ID。`safe_mode` 仍可强制拒绝写调用，denylist 始终优先。

只读工具也有仓库边界：`read_file`、`list_files` 和 `trace_call_path` 拒绝 `.env*`、常见凭据名称（如 `secrets.py`、`credentials.json`、私钥后缀及 `.ssh` 目录）、忽略目录及仓库外路径；Python 索引和磁盘快照也跳过这些路径及符号链接。此名称拒绝表不是秘密扫描器：普通源码文件中仍可能写有凭据，使用在线模型前必须自行检查仓库；历史缓存快照若曾包含敏感源码，也不会被自动批量删除。`trace_call_path(path, symbol)` 接受已定位的 Python 文件及精确函数名、`Class.method` 或嵌套函数限定名，在索引源码上静态解析本地函数调用、作用域内直接导入和有限的包级再导出，返回有界的目标路径与行号；它不执行仓库代码，也不推断动态分派。返回闭包的外层函数不会被误报成直接调用闭包；要看闭包内部，需明确给出其完整限定名。同名条件定义存在歧义时拒绝静默选择。追踪结果只用于导航，不自动算作已读取的引用证据，回答前仍需用 `read_file` 核对。`inspect_diff` 只返回同一读取策略允许的未暂存文件差异，连已删除的受跟踪符号链接也会排除，避免本地配置或仓库外路径进入模型上下文。这是应用层防误读策略，不是针对恶意本地进程的操作系统沙箱。

CLI 会在每次写调用前提示确认。需要恢复时使用执行结果中的快照 ID：

```bash
python main.py /path/to/repo --rollback SNAPSHOT_ID
```

浏览器会在修改后直接展示 diff，并提供“恢复到修改前”按钮。`run_command` 必须在 `affected_files` 中声明所有可能修改的仓库相对路径；命令里的显式路径参数也必须使用仓库相对路径，绝对路径、`..`、`~` 和符号链接逃逸会在审批前被拒绝。快照与回滚保证只覆盖这些已声明文件。

普通 Agent 的 LangChain 工具适配层公开六个严格命名工具；实际仓库访问仍统一进入 `AgentExecutor.call()`。只读图绑定 `search_code`、`read_file`、`trace_call_path`；Learning Mode 的写权限不因此改变。

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

## Trace 与 JSON 文件数据库存储

CLI 与浏览器普通 Agent 默认在项目的 `artifacts/traces.sqlite3` 保存运行 trace。
使用 Python 标准库 SQLite，无需安装数据库服务；直接调用 Python API 时，需显式设置
`UnifiedRunConfig(trace_db_path=...)` 才会保存运行索引。trace JSON 导出、检索评测和各项能力评测的
JSON/JSONL 输出另会自动归档到同一个数据库。Learning Mode 和旧离线流程尚不自动建立运行索引。

```bash
python frontend.py --trace-db artifacts/traces.sqlite3
python main.py . "解释检索流程" --trace-db artifacts/traces.sqlite3
python main.py . "解释检索流程" --trace-db off
```

也可以设置 `REPOPILOT_TRACE_DB`；命令行参数优先，值为 `off` 时禁用数据库保存。
自定义相对路径按启动工作目录解析。数据库位于本机磁盘；默认数据库及 WAL 附属文件已被 Git 忽略。

数据分成 `trace_runs`、`trace_events`、`llm_calls` 和 `json_files` 四张表。前者保存运行摘要与经过过滤的完整 trace JSON，
事件表保存可查询的事件序号、状态、时间、耗时和错误摘要；调用表只保存固定且有界的脱敏字段，并以 `run_id` 关联逻辑调用和对应事件。`SQLiteTraceStore.get_llm_calls(run_id)` 返回调用明细，`get_llm_summary(run_id)` 返回 attempts／retries／outcome／latency／error class 聚合。数据库结构版本使用 `PRAGMA user_version`，
独立于 JSON 的 `trace_version`。未知结构版本及已有其他表的未版本化数据库会被拒绝，不自动覆盖。
结构 v1/v2 会在下一次写入时于同一事务迁移到 v3；迁移保留旧 trace，旧记录没有事后推断的调用观测。

### JSON/JSONL 文件归档与历史导入

`json_files` 保存来源绝对路径、格式、脱敏内容的 SHA-256、UTF-8 字节数、记录数、保存时间及完整脱敏内容。
同一路径、同一脱敏内容重复保存不会增加记录；内容变化时新增版本，保留旧版本。JSONL 的每行分别解析和过滤，
任意一行无效时整份文件不入库。归档内容是重新序列化后的 JSON，不保留原文件的空白排版。

导入已经生成的文件（默认递归导入项目 `artifacts/`，也可显式指定目录或文件）：

```bash
python scripts/import_json_artifacts.py
python scripts/import_json_artifacts.py artifacts trace_pending --trace-db artifacts/traces.sqlite3
python scripts/import_json_artifacts.py artifacts/agent_trace_demo.json
```

命令创建或升级数据库，输出 `saved`、`unchanged`、`failed` 数量。单个文件失败不阻止其他文件导入，
存在失败时退出码为 1；重复目录和文件参数会去重。源文件始终保留。
历史文件归档只进入 `json_files`，不推断旧文件的 `run_id`，不自动生成 `trace_runs` 运行记录。

可显式把旧 trace 转换为 `trace_runs` 与 `trace_events`，从现有历史查询接口统一读取：

```bash
# 直接转换数据库里已归档的所有版本，原文件已移动也可使用
python scripts/import_json_artifacts.py --convert-traces --archives-only
# 先归档指定目录，再转换数据库里全部旧 trace
python scripts/import_json_artifacts.py artifacts trace_pending --convert-traces
```

支持完整运行 trace、待补录 `trace/context` 包装、早期检索 `query/params/stages`、
Agent 工作流包装与逐条评测 JSONL，也支持 JSON trace 数组。汇总、阶段诊断副本、应用记忆及其他非 trace 数据
仍留在 `json_files`，不伪造为独立执行。转换结果在 `traces` 中报告 `saved/unchanged/skipped/failed`，
错误包含归档文件 ID 和记录序号；一条记录失败不阻止后续记录，存在失败时退出码为 1。

- 已有 `run_id` 原样保留；相同内容或更旧版本跳过，新版本仅在元数据和已有事件历史兼容时保存。
  同版本不同内容报告冲突，保留数据库原记录；导入副本不会重写已有运行的来源身份。
- 没有 `run_id` 时，使用脱敏原始记录的规范 JSON 内容哈希生成 `legacy-...` ID，不包含文件路径、
  行号或导入时间。复制文件、JSON 键顺序变化和重复执行不会新增运行；内容不同则保留为独立记录。
  原始记录没有执行 ID 时，无法区分“内容相同的不同执行”，按相同记录去重。
- 旧记录原文保存在 `artifacts.legacy_import.original_record`，来源标为 `legacy_import`；
  完整待补录包装优先采用原有 context。缺失运行状态为 `unknown`，缺失开始、结束及事件时间保留空值；
  缺失更新时间仅用首份归档的保存时间作排序兜底，并标注此来源，不当作历史执行时间。
  仓库信息仅使用记录中已有值，缺失为 `unknown`，不根据当前工作目录推测历史仓库。

本地读取或导出可使用 Python API（文件归档暂不通过 HTTP 历史接口提供）：

```python
from pathlib import Path
from coding_rag.storage.sqlite import SQLiteTraceStore

store = SQLiteTraceStore("artifacts/traces.sqlite3")
files = store.list_json_files(limit=20)  # 元数据；支持 source_path 和 offset
if files:
    archived = store.get_json_file(files[0]["file_id"])
    Path("restored." + archived["format"]).write_text(archived["content"], encoding="utf-8")
```

自动归档遵循 `REPOPILOT_TRACE_DB`；普通 CLI 的 trace 导出同时遵循 `--trace-db`，`off` 可禁用。
数据库归档失败会记录警告，已生成的原文件保留，可再次运行导入命令补录。
原文件格式与内容不变；数据库副本采用现有 trace 脱敏规则，不能保证识别自由文本中的所有秘密。

每次普通 Agent 返回结果或等待授权时，用一个事务保存累计 trace。等待授权与批准后的运行沿用
同一 `run_id`，递增 `revision`，仅追加新事件；重复保存无副作用，旧版本不能覆盖新版本，
同版本不同内容会报告冲突。等待授权时 `finished_at` 为空。时间使用 UTC；这不是逐节点实时落库，
进程中断前尚未返回的事件不会被保存，输入校验/初始化失败也不保证生成 trace。

### 历史查询接口

历史查询默认关闭，需要在服务启动前配置至少 32 字符的独立随机令牌
`REPOPILOT_TRACE_READ_TOKEN`。每次历史请求必须携带 `Authorization: Bearer <令牌>`。
令牌不进入 HTML、trace 或应用输出；未配置返回 403，认证失败返回 401。
该令牌可读取本服务数据库中全部运行，不提供按用户隔离。服务继续面向本机使用。

```powershell
$env:REPOPILOT_TRACE_READ_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
python frontend.py
```

在配置相同令牌的调用环境中查询：

```powershell
$headers = @{ Authorization = "Bearer $env:REPOPILOT_TRACE_READ_TOKEN" }
Invoke-RestMethod 'http://127.0.0.1:8765/api/traces?status=failed&limit=20' -Headers $headers
```

- `GET /api/traces`：返回 `items` 和 `next_cursor`；列表不包含完整 trace。
- 支持 `repo_key`、`mode`、`status`、`source`、`session_id`、`since`、`until` 筛选。
  时间筛选针对更新时间，必须包含时区；`limit` 为 1–100，默认 50。
  后续页将 `next_cursor` URL 编码后作为 `cursor` 传入，并保持相同筛选条件。
- `GET /api/traces/{run_id}`：返回经过过滤的完整 `trace`，可保存为 JSON；不存在返回 404。
- 接口只读取服务启动时配置的数据库，不接受客户端指定数据库路径。
- 数据库关闭或不可用返回 503。已有 `/api/run` 响应额外包含 `storage` 保存状态。

分页按更新时间和运行 ID 排序。正在更新的运行可能改变列表位置，跨页查询不承诺数据库快照一致性。
历史列表页面、教学与评测的结构化运行索引和统计面板属于后续阶段。

### 保存失败与补录

数据库写入失败不重跑模型或工具，也不改变本次 Agent 的业务结果。CLI/浏览器会提示保存失败；
系统尝试在数据库同目录的 `trace_pending/` 写入独立 JSON 副本，`storage.pending` 表示副本是否成功保存。
两处都无法写入时会明确报告失败，不承诺数据已保存。

```bash
python scripts/replay_trace_pending.py artifacts/trace_pending/某个文件.json --trace-db artifacts/traces.sqlite3
```

补录脚本只保存 trace，重复执行不会重复插入；源文件保留，不自动删除。
数据库与待补录副本都会移除授权对象、恢复状态、消息列表和原始模型输出，并遮盖已配置凭据及常见凭据格式。
自由文本过滤是尽力识别，不能保证识别所有任意格式的秘密；查询、代码片段、diff 和工具结果仍属于本地诊断数据。
原有 `--trace-out` 导出行为保持不变，不受数据库过滤逻辑影响。
数据库只用于诊断历史，不能据此在服务重启后恢复工具执行或授权。

验证数据库接入：

```bash
python -m unittest tests.test_legacy_trace_import tests.test_json_artifacts tests.test_trace_storage tests.test_unified_runtime tests.test_trace tests.test_frontend tests.test_interactive_cli
```

## 检索评测

RepoPilot 现在提供可复现的统一能力记分卡。评测会先校验 gold 文件，再对仓库只扫描、切片和建索引一次，随后复用索引执行全部查询。默认不把 `data`、`datasets`、`artifacts` 中的下载语料和运行产物误计入项目自身索引。

```bash
python scripts/validate_evalset.py . datasets/eval/sample_evalset.json
python scripts/validate_evalset.py . datasets/eval/trace_v2.jsonl
python scripts/capability_eval.py --repo . --out artifacts/eval/capability-scorecard.json
python scripts/capability_eval.py --repo . --baseline benchmarks/retrieval-baseline-v5.json --policy benchmarks/capability-gate-v2.json --gate
```

Scorecard 包含 Hit@1、Hit@5、MRR@5、Recall@5、nDCG@5、全部 gold 命中率 `all_gold_at_5`、未完整命中的 case、索引构建耗时、查询延迟和上下文字符数。原有 `bad_case_ids` 仅代表零命中，不能代替跨文件调用链完成率；新增诊断指标暂不改变质量分公式。默认自仓库检索集使用 `trace_v2.jsonl`，将退出主路径的旧 `AnswerGenerator` gold 更新为统一 Agent 的真实运行模块；旧 `trace.jsonl` 和 v1 基线保留供历史结果复现。两版数据集分数不可直接比较。query-only 术语扩展对应历史 v5 身份及自仓库 v3 基线；精确 `Class.method()` 文件内片段对齐对应 v6；当前显式符号定义定位对应 v7，默认门禁使用自仓库 v5、多仓 v4、Agent runtime v4 基线。旧版本保留作同数据集 A/B 证据，不覆盖或混用。Agent 离线 fixture 会执行只读 `search_code`，将有序 Top-5 结果与 SWE-bench 的 `changed_files + test_files` 比较，并在 `fixture_diagnostic` 中输出候选指标与数据覆盖缺口。当前 50 个 fixture 的 `repo_files` 没有任何负样本，且文件内容只复用了题目描述，因此明确标记 `discriminative=false`；这些必满候选指标只用于暴露 fixture 结构，不代表真实 Agent 文件定位能力，也不参与综合分或质量门禁。固定 `OfflinePlanClient` 的 trace 与安全工具接口检查记录在 `offline_contract`。

Schema v2 同时记录 retrieval 与 agent evalset 的文件名、SHA-256 和 case 数。门禁会检查 schema、两套数据集身份、完整检索配置和 retriever 版本，约束 retrieval 质量、Agent fixture 数据覆盖率及 trace／安全契约，但不约束无判别力的候选指标；通过返回 0，质量或延迟退化返回 1，数据集或实验配置不兼容返回 2。Agent fixture 写入临时仓库前还会拒绝绝对路径和 `..` 越界路径。实验元数据记录 Git 状态、Python/平台、仓库指纹、检索参数与权重，不保存机器绝对路径。只有显式 `--update-baseline` 才会覆盖基线。

### 真实多仓库源码基准

独立的 `multirepo_eval.py` 使用固定提交的 Django 和 LeetCode Python 真实 checkout。20 条问题直接检索完整源码，按仓库各建一次索引，并输出总体、仓库、任务类型和难度分层；它不读取或混入上面的 Agent fixture 分数。

```bash
python scripts/multirepo_eval.py
python scripts/multirepo_eval.py --update-baseline
python scripts/multirepo_eval.py --gate
```

仓库身份来自 `datasets/eval/multirepo_manifest.json`，包含相对路径、规范化 remote、固定 commit、许可证和用途。运行前会严格检查 manifest schema、工作区路径边界、Git HEAD／origin，以及全部 case ID、repo ID 和 gold 文件。缺少 checkout 时只打印带固定 commit 的准备提示，不会联网下载。独立门禁将 schema、数据集、manifest、完整配置和固定仓库身份变化视为不兼容并返回 2；质量退化返回 1，通过返回 0。普通运行不会覆盖 `benchmarks/multirepo-retrieval-baseline-v4.json`。

另有固定 Aider／Django 的 8 条跨文件“入口 → 提供者”检索题。默认离线检索的入口文件 Top-1 为 8/8，但 Top-5 同时找齐两个正确文件仅 6/8；Aider 的 lint 实现与 Django 的 URL resolver 实现分别落在第 9、12 位。路径权重只读 A/B 表明 0.5 使完整链路降至 5/8，0 则使入口 Top-1 降至 5/8，所以保留默认 0.25。这组已揭盲题由独立基线约束，不能当作模型修复成功率或盲测泛化结果；运行命令见 `datasets/README.md`。

v6 检索器只在问题明确写出 `Class.method()` 且已选中文件确有该符号时，把该文件的代表代码块定位到方法定义处；文件选择和文件分数保持不变。固定 Django 登录题的首个片段因此从同文件其他方法移到 `LoginView.form_valid`，但这不补齐仍缺失的跨文件目标，也不证明模型最终回答质量。

v7 进一步处理明确写出 `snake_case` 或 `Class.method` 并询问“定义／实现／入口在哪里”的查询：根据 AST 符号记录，将定义所在片段优先于调用示例；问调用处或测试位置不使用该信号。自仓库 `trace_call_path` 的定义从 Top-5 外进入前三；Django `LoginView.form_valid` 同文件首片段从第 351 行改为定义附近的第 106 行。它不推断未写出的符号，也不补齐跨文件调用链。当前模型修复评测按用户要求暂不运行。

### 真实 Agent 运行时契约基准

调用路径工具另有固定源码的组件门禁：

```bash
python scripts/call_path_eval.py --gate
```

它在固定 Django 提交上从 7 个已给定的入口函数／方法出发，量化声明目标边召回、目标行号可定位率、完整用例率和无效定位数。输入入口由评测集提供，因此该门禁仅检验**已知入口后的静态导航**，不检验搜索能否发现入口、未标注边的语义精度、模型是否正确选用工具或最终修复能力。数据集、manifest SHA-256 与基线绑定；改动评测集须建立新身份，不能把旧分数直接比较。检索留出集仍单独报告，不因这个组件门禁通过就视为调用链召回已解决。

显式调用链问题另有统一 Agent 图上的离线运行时契约：

```bash
python scripts/call_path_runtime_eval.py --gate
```

它对固定 Django 留出题与转移题执行 `search_code → trace_call_path → read_file(入口调用行) → read_file(目标定义行) → 引用回答`，脚本策略只读取问题和真实工具 observation，不接收 gold。记分卡分开报告两份源码是否实际读取、静态调用行与目标定义行是否确实出现在返回文本中、双文件是否被有效引用，以及完整工具契约成功率；不能用 LangGraph 的普通 `terminal_success` 替代证据成功率。当前规则仅支持题目明确写出的首个 `name()`／`Class.method()` 入口和唯一可定位的查询目标；歧义目标会拒绝继续。它是工具接线与证据链门禁，**不是**自然语言 Agent 或真实模型修复能力评测；跨文件搜索 Top-5 缺口仍独立存在。

额外的未点名入口检索探针：`python scripts/multirepo_eval.py --evalset datasets/eval/multirepo_call_path_generalization_v1.jsonl --out artifacts/eval/call-path-generalization-scorecard.json`。首次冻结后运行的 4 条 Django 题中，Top-5 全文件命中率为 2/4（另 2 条只找到入口），不能用“至少一文件命中”4/4 代替。`python scripts/call_path_negative_eval.py --gate` 检查 3 条已知入口的错误目标，目前误连 0；它只证明静态组件的近邻拒连，不证明模型能从自然语言发现入口。装饰器题还暴露出嵌套闭包导航需求：现在可用 `python scripts/repo_inspect.py data/repos/django trace django/contrib/auth/decorators.py user_passes_test.decorator._redirect_to_login` 查看闭包中的真实调用边。

另有固定 Django 源码审查过的正负挑战集，专测误连和拒答：

```bash
python scripts/call_path_challenge_eval.py --gate
```

题目包含 4 条真实跨文件边和 5 条近邻／多目标拒答题。首次运行旧 v2 策略时，正例 4/4、负例 3/5，出现 2 次误连；原因是题面普通词覆盖了显式目标，以及单目标策略对双目标问题给出部分答案。v3 优先精确匹配显式 `target()`，明确出现多个目标时拒答；在**同一组已参与修复的题**上回归为正例 4/4、负例 5/5、误连 0。`benchmarks/call-path-challenge-first-look-v1.json` 保留首次诊断，当前门禁仅防止既有行为退化，不构成新的盲测泛化证明。

`agent_runtime_eval.py` 在同一组固定 Django／LeetCode 真实源码和 20 条问题上执行完整统一 LangGraph 流程。默认 policy 是离线、确定性的 `search → read → cited answer`：它只接收原始 query 和真实工具 observation，先用原问题调用 `search_code`，再只从返回候选中选择首个文件调用 `read_file`，最后只引用实际读取范围；policy 的构造和决策接口不接收 dataset case 或 gold。该分数明确表示 runtime／retrieval contract，不代表通用 LLM 推理能力，也不依赖网络、API key 或在线 judge。

```bash
python scripts/agent_runtime_eval.py
python scripts/agent_runtime_eval.py --update-baseline
python scripts/agent_runtime_eval.py --gate
```

基准在每个 repo 开始前只按 repo/config 显式预热索引，并单独记录 warmup lifecycle 与耗时；20 个 case 的搜索都必须命中进程内索引，因此 case latency 不混入冷建或磁盘恢复成本，`memory_index_hit_rate` 必须保持 1。Top-K 语义固定为 5，传入其他值会作为不兼容配置返回 2。指标包含 Search Hit@5／MRR@5、selected/read file Hit@1、工具序列成功率、引用存在、已读范围引用有效性、gold 文件引用 grounding、终态成功率、步数和 p50/p95 延迟，并按 repo、任务类型、难度分层。Schema v2 还报告每个 repo 及总体的 indexed file、unique gold file 和 negative file 数；任何 repo 没有真实负样本都会拒绝评测。独立门禁固定 schema、dataset/manifest、repo commits/remotes、完整检索配置和 scripted policy version；不兼容返回 2、退化返回 1、通过返回 0。错误候选仍可产生结构有效的引用，但会降低 selected/read/gold-grounding，避免把“引用了某段代码”误算成“找到了正确代码”。

### 真实补丁／测试结果基准

`patch_outcome_eval.py` 衡量 coding agent 的最终补丁是否真正修复了回归，不把结果包装成 LLM 智能分数。默认 v2 suite 包含三个不同 Django 源码文件的回归任务，固定到提交 `09fdc06346b167ff6231a4cb80b85ae67b53c715`；原有两例 v1 suite 和基线保留，避免改变历史分数含义。评测仅通过 `git archive` 导出无符号链接的 `django`／`tests` 根目录。每个 case 先证明上游原始测试通过，再注入已校验的真实源码 fault 并证明测试失败，最后才调用候选 provider、应用候选 unified diff 并复跑同一条上游测试。候选接口只获得需求、允许路径和测试命令，不获得 fault 或 oracle patch。

```bash
# 默认 oracle 仅用于验证 harness 自身，不代表 Agent 能力
python scripts/patch_outcome_eval.py
python scripts/patch_outcome_eval.py --update-baseline
python scripts/patch_outcome_eval.py --gate

# 可复现的 no-op 负向诊断：相对 oracle baseline 应返回 1
python scripts/patch_outcome_eval.py --candidate-mode static \
  --candidate-patches datasets/eval/patch_outcome_noop_candidates_v2.json \
  --allow-local-code-execution --gate
```

Scorecard 记录 pristine／fault 先决条件、patch applied、目标测试、允许路径精度、forbidden 未触碰、最小改动、超时和有界输出。只有显式 oracle self-check 且全部 harness／测试通过时才允许更新基线；普通运行不会覆盖它。门禁对 evaluator、数据集、固定源码身份和安全／测试配置不兼容返回 2，对真实候选退化返回 1，通过返回 0；provider 身份只是运行元数据，不会让 static 候选与 oracle baseline 被误判为不兼容。墙钟耗时用于诊断，不作为跨机器硬门禁。

外部 static candidate 必须显式传入 `--allow-local-code-execution`。该 harness 只适用于 `trusted_local_candidate_only`：临时工作区、最小环境、Python socket guard、严格 diff 路径、大小／输出／超时限制是防误用和可复现性边界，`workspace_isolation_not_os_sandbox`，不能安全执行恶意候选代码。子进程始终使用固定 argv 且 `shell=False`；候选 patch 在应用前拒绝删除、新建、rename/copy、绝对路径、父目录穿越、符号链接和未允许路径。

可选的 `agent-live` provider 会让现有普通 Agent LangGraph 在第二个 evaluator-owned 候选副本中执行 `search_code`／`read_file`，并沿用 `approval_required → exact fingerprint → resume_unified_query` 路径。只有通过私有任务范围复核的 `apply_patch` 会在候选副本内自动批准；`run_command` 永不自动批准，fault 外层工作区仍由 harness 独立应用最终累计 diff。模型消息只包含公开 requirement、通用系统规则以及它主动获得的仓库 observation，不包含 allowed／forbidden path、固定测试命令、fault/oracle 路径、patch 或哈希。索引使用 `disk_cache=False` 的临时 registry，trace SQLite 关闭，rollback snapshot 也仅写 evaluator 临时目录。

```bash
# 会发送任务需求和模型主动读取的公开 Django 源码，并可能产生第三方费用
python scripts/patch_outcome_eval.py --candidate-mode agent-live \
  --allow-local-code-execution --allow-network-llm --accept-provider-charges \
  --llm-provider deepseek
```

默认仍是离线 oracle self-check，上述 live 模式绝不会隐式运行。v1 将每次输出 token 上限限制为 8192，并报告逻辑调用数、HTTP attempts/retries 及理论输出 token 上限；provider 未返回可靠 usage，因而明确记录 `billing_measurement=unavailable`，不声称准确费用。live scorecard 不保存 prompt、源码、patch、完整回答、base URL 或 key。`agent-live --gate` 当前会在任何模型调用前返回 2：默认 100 分 oracle baseline 只用于 harness 自检和确定性负向诊断，不能冒充同一 provider/model 的模型回归基线。

设计参考 Aider 的 repo map 与 benchmark 思路，固定参考上游提交 [`5dc9490`](https://github.com/Aider-AI/aider/tree/5dc9490bb35f9729ef2c95d00a19ccd30c26339c) 及其 [Apache-2.0 许可证](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/LICENSE.txt)。本项目没有复制 Aider 源码；实现了可选的依赖图扩散实验，但只有通过固定基准的排序信号才会进入默认链路。

检索器默认融合正文、路径／符号 metadata、词元覆盖、AST 结构与路径标识符精确匹配。依赖图扩散可用 `--graph-weight` 做实验，但固定评测未显示收益，因此默认权重为 0；这避免把没有量化收益的复杂度带入普通运行链路。

最终过滤会先按相关性排序，再优先保留不同文件；只有不同文件不足 `final_k` 时才补入同文件的额外 chunk。这防止一个高分文件的相邻片段占满全部上下文，同时仍保留需要深读单文件时的候补片段。

运行示例评测集：

```bash
python scripts/retrieval_eval.py . datasets/eval/sample_evalset.json --top-k 5
```

运行 20 条 trace 测试集：

```bash
python scripts/retrieval_eval.py . datasets/eval/trace_v2.jsonl --top-k 5 --trace-out artifacts/trace_eval_trace.jsonl
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


### 普通聊天的多轮历史与恢复

普通聊天使用独立的 `artifacts/conversations.sqlite3`，可用环境变量
`REPOPILOT_CHAT_DB`、CLI 的 `--chat-db` 或 Python 配置的 `conversation_db_path` 指定。
仅依赖 Python 标准库 SQLite，不增加安装依赖。数据库保存用户问题和最终回答，
按规范化的仓库绝对路径与 UUID `session_id` 隔离；关闭 trace 存储不关闭聊天历史。

- 浏览器在当前标签页的 `sessionStorage` 保存会话 ID 与仓库选择，刷新后继续同一对话。
  点击“新对话”生成新 ID；旧记录仍在本地，但不会传给新对话。关闭标签页后不自动找回 ID。
- CLI 交互模式自动生成并显示会话 ID。重启后使用下面的命令继续原对话：
  `python main.py D:/develop/repo_pilot --interactive --session-id <原会话UUID>`。
  单次 CLI/Python 调用未指定 `session_id` 时仍为独立请求。
- Tk 桌面的 LLM 查询使用相同运行时；可复制“会话 ID”并在重启后粘贴恢复，或点“新对话”。
  未启用 LLM 的检索和评测仍使用各自的离线／评测组件。
- `POST /api/run` 接受 `session_id`；省略时生成新 ID，并在响应 `conversation.session_id`
  返回。客户端应重复提交这个 ID。历史由服务端读取，不接受客户端提供的 messages。

这里采用“完成轮次持久化 + 每轮显式装载历史”，没有配置 LangGraph checkpointer，
也不依赖 `thread_id` 隐式恢复。Graph 每轮仍重建工具与批准状态，但初始 messages
包含系统规则、历史用户／助手消息和当前问题；图内 reducer 累积本轮工具与最终助手消息。
只保存成功／部分完成的轮次；失败与等待批准不会作为完成轮次写入。批准完成后只追加一次。
同一服务内的相同会话串行处理，SQLite 修订检查防止跨进程并发覆盖；冲突会报告保存失败。

模型默认使用最多 24,000 字符的历史正文（不含消息封装），可通过 Python 配置
`max_history_chars` 调整。优先保留首轮和最近的完整轮次，不切断用户／助手消息对；
无法放入预算的轮次会省略，并在系统提示与响应 `conversation.omitted_turns` 中明确标记。
数据库中的原始轮次不受此裁剪影响。工具观察另受 `max_context_chars` 限制；
搜索结果过长时优先省略正文和完整行，保留可直接传入 `read_file` 的完整路径。

历史回答仅帮助理解对话，涉及当前仓库事实时仍须重新读取证据。服务重启可恢复已保存的聊天，
不会恢复尚未批准的工具调用，也不会从历史重放写操作。聊天保存失败会在页面／CLI 提示。
Learning Mode 保持原有的页面内存签名会话，刷新重新开始，服务重启后旧签名失效。
这些机制均与开发助手的 `.engramory-memory` 不同。

兼容边界：`AgentExecutor` 和 `langchain_tools` 是当前 Graph 的共用工具层；
`planner.py`、`workflow.py` 和旧问答生成器仍服务于兼容 API、离线规划或评测，
不会作为普通聊天的失败回退。Graph 的 `safe_fallback` 只结束本次运行并报告错误。

复测：同一对话先说“记住我的项目代号 cobalt-731”，再问“代号是什么”；刷新页面、
重启服务后再问。点击“新对话”后再次询问，模型不应从旧会话获得代号。
再要求搜索一个明确函数，并使用搜索返回的 `path` 读取文件，检查 Trace Events 中
`search_code -> read_file` 均成功。JSON 中 `\\` 表示转义，不等于路径中有两个分隔符；
仓库外绝对路径会被读取权限拒绝，这是预期边界。

离线回归：`python -m unittest tests.test_conversation_runtime -v`；完整回归：
`python -m unittest discover -s tests -q`。测试使用可检查输入的模型替身，包含新 Python
进程恢复、真实 HTTP 服务重建与浏览器脚本刷新测试，不需要真实模型 API 密钥。
