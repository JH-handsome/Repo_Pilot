# 数据集目录

这里放检索语料和评测集。

## 推荐语料

建议从这个仓库开始：

- 仓库：`https://github.com/cnkyrpsgl/leetcode`
- 原因：MIT 协议、Python-only、文件结构简单，适合 `file_loader` 读取
- 本地目录：`datasets/leetcode-python`

下载：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_leetcode_corpus.ps1
```

下载后检索：

```powershell
python main.py datasets "two sum hash map" --top-k 5
python main.py datasets "binary tree level order null TreeNode" --top-k 5
python main.py datasets "linked list ListNode head next" --top-k 5
```

## LeetCode 类型参考

`leetcode_reference/leetcode_formats.py` 定义了常见 LeetCode 结构，例如：

- `ListNode`
- `TreeNode`
- `RandomListNode`
- `GraphNode`
- `NaryNode`

如果要把下载下来的 LeetCode 代码单独本地运行，可以用项目根目录的 `leetcode_types.py` 做类型补丁：

```powershell
python scripts/patch_leetcode_imports.py datasets/leetcode-python
```

## 评测集

`eval/` 下保存检索评测数据：

- `sample_evalset.json`: 小型示例评测集
- `trace.jsonl`／`trace_v2.jsonl`: 20 条中文自仓库检索问题；v2 按统一 Agent 当前运行路径重审 gold，v1 保留供历史结果复现
- `multirepo_manifest.json`: 固定 Django 与 LeetCode checkout 的 portable 身份、许可证和用途
- `multirepo_source.jsonl`: 20 条真实源码检索问题（Django 12 条、LeetCode 8 条）
- `aider_django_multistep_manifest_v1.json` 与 `aider_django_multistep_v1.jsonl`：固定 Aider、Django 提交的 8 条“入口文件 → 实际提供者文件”双 gold 题。首次检索前已冻结题目；默认 Top-5 同时找齐 6/8，入口文件 Top-1 为 8/8，漏掉的实现文件分别是 `aider/linter.py` 与 `django/urls/resolvers.py`。题目已经揭盲，后续只作开发回归约束，不作独立泛化证明。运行 `python scripts/multirepo_eval.py --manifest datasets/eval/aider_django_multistep_manifest_v1.json --evalset datasets/eval/aider_django_multistep_v1.jsonl --baseline benchmarks/aider-django-multistep-baseline-v1.json --policy benchmarks/aider-django-multistep-gate-v1.json --out artifacts/eval/aider-django-multistep-scorecard.json --gate`。
- `aider_django_multistep_v2.jsonl`：同一固定仓库上的 6 条新双文件题，覆盖 Aider 提交／网页抓取／帮助与 Django 模板渲染／邮件／上传；先按源码冻结，再首次运行默认检索，Top-5 同时找齐 3/6。完整首轮记分卡见 `benchmarks/aider-django-multistep-v2-first-look.json`。随后用该集检查旧题产生的查询拆分候选，融合后为 4/6，但旧 8 题出现有效实现文件退化，故没有改动默认检索；实验与拒绝理由见 `benchmarks/query-decomposition-rejection-v1.json`。这 6 题现已揭盲，只作回归，不再称新的留出集。
- `multirepo_call_path_probe.jsonl`: 8 条额外的固定 Django 跨文件调用链诊断题；gold 对应源码中的实际调用边，只作独立诊断，不与默认 20 题门禁混为同一基线
- `multirepo_call_path_holdout.jsonl`: 4 条另行固定的 Django 调用链留出题，用来检查调用点导航或排序实验是否能泛化；不得在同一轮调参后把留出集分数冒充独立复验
- `multirepo_call_path_transfer.jsonl`: 4 条在本轮排序 A/B 前固定、跨 Django 子系统的调用链转移题；与旧留出题分开报告，不用于调整同一轮候选算法
- `call_path_tool_cases.json`: 7 条固定 Django 源码的静态调用路径组件用例，提供已知入口与声明目标边；只衡量入口之后的导航，不是独立检索留出集或模型评测
- `call_path_challenge_v1.json`: 4 条真实跨文件边、4 条“导入或兄弟函数相关但当前入口未调用”的近邻负例，以及 1 条双目标超出单目标契约的拒答题。固定 Django 提交和题目 SHA 由 `call_path_challenge_eval.py` 校验；首次诊断已用于修复 v3 策略，后续只作为回归集，不再当盲测。
- `multirepo_call_path_generalization_v1.jsonl`: 4 条在首次检索前固定、未直接点名入口函数的 Django 跨文件题。首次 Top-5 至少一文件命中 4/4，全部 gold 同时命中 2/4；此后作为开发／回归集，不再称盲测。
- `multirepo_call_path_generalization_transfer_v1.jsonl`: 另 4 条先冻结再运行的 Django 跨文件转移题，覆盖斜杠重定向、PBKDF2、消息 Cookie 签名与缓存 Expires。首次 Top-5 入口 4/4、全部 gold 1/4；首轮结果已查看，后续不再称盲测，也不与旧数据集合并成单一分数。
- `multirepo_call_path_generalization_holdout_v2.jsonl`: 4 条在调用边源码行权重实验前固定的 Django 跨文件题；已参与该权重选择，现只作开发／回归题。
- `multirepo_call_path_generalization_holdout_v3.jsonl`: 4 条在源码行权重固定后、首次实验前另行冻结的 Django 转移题；首次默认 Top-5 全部 gold 2/4。首次调用边候选实验后也已揭盲，后续只作回归。
- `call_edge_grounding_v1.json`／`call_edge_grounding_v2.json`: 上述 12 条开发题和 4 条新转移题的源码审定入口／目标符号契约，供 `call_edge_expansion_ab.py` 单独检查具体调用边，不把“目标文件相同、调用边不同”计为成功。
- `multirepo_call_edge_negative_v1.jsonl`: 4 条固定 Django 的单文件问题，明确不需要跨文件建议；首次运行误导建议首位为 4/4，之后只作为回归集。
- `self_call_edge_probe_v1.jsonl`／`self_call_edge_grounding_v1.json`: 4 条 RepoPilot 当前源码的跨文件烟测；仓库持续改动且未固定提交，只用于迁移诊断，不构成独立留出评测。
- `aider_manifest_v1.json`: 独立固定 Aider `5dc9490` 与 Apache-2.0 许可证，不改变既有 Django／LeetCode 门禁清单；本地 checkout 位于 Git 忽略的 `data/repos/aider`。
- `aider_call_edge_positive_v1.jsonl`／`aider_call_edge_grounding_v1.json`: 4 条 Aider 真实源码跨文件调用题，覆盖 repo map、lint 命令、启动参数与 reasoning tag；首次精确边 Top1／Top3 均 3/4，漏题入口文件未进入 Top5。
- `aider_call_edge_negative_v1.jsonl`／`aider_call_edge_near_negative_v1.jsonl`: 各 4 条 Aider 单文件问题；后者故意询问真实的同文件调用，避免把“题面有 call 一词”误当成跨文件意图。原候选逻辑对两组均误导 4/4。
- `aider_call_edge_source_anchors_v1.json`: 上述 12 题经源码审定的源文件／函数，仅用于“入口已经确认”的组件诊断；不能让自动搜索偷看此文件。
- `django_call_edge_source_anchors_v1.json`: 上述 Django v3 跨文件题与 4 条单文件负例的独立入口锚点，供相同组件诊断使用；结果不回填默认检索基线。
- `call_path_generalization_negative_v1.json`: 3 条固定 Django 的静态错误目标近邻题，分别检查同文件错误格式化函数、base-36 编码／解码混淆和无关 URL helper；只评估已知入口后的静态边，不代表模型拒答能力。

`multirepo_call_path_holdout.jsonl` 与 `multirepo_call_path_transfer.jsonl` 还共同驱动 `python scripts/call_path_runtime_eval.py --gate`：固定离线策略从题面首个显式 `name()`／`Class.method()` 出发，经真实统一 Agent 图检索、追踪、读取入口调用行和目标定义行，再检查两份源码的有效引用。基线固定数据 SHA、仓库提交、配置和策略身份；v2 约束实际返回文本中必须包含两处关键行，v3 增加显式目标精确匹配与多目标拒答，旧基线保留复现。`contract_success_rate` 是证据链指标，不是模型推理准确率。这两组题已用于开发该契约，不能再视为对该策略的全新盲测。
- `django_patch_outcome_v1.json`／`django_patch_outcome_v2.json`: 固定 Django 提交、真实上游测试、fault/oracle 哈希和允许路径的补丁结果任务清单；v2 新增独立的配置解析故障，v1 保留供历史结果复现
- `patch_outcome_noop_candidates.json`／`patch_outcome_noop_candidates_v2.json`: 对应 suite 的可复现 no-op 负向候选，仅用于证明 gate 会返回质量退化，不是基线或 oracle

每条用例必须有唯一 `id`、非空 `query`／`question`、至少一个仓库相对 gold 路径，以及 `task_type` 和 `difficulty`。提交前运行：

```powershell
python scripts/validate_evalset.py . datasets/eval/sample_evalset.json
python scripts/validate_evalset.py . datasets/eval/trace_v2.jsonl
```

校验器会拒绝重复 ID、空问题、空 gold、绝对路径、包含 `..` 的路径和不存在的 gold 文件。

多仓库评测使用独立入口：

```powershell
python scripts/multirepo_eval.py
python scripts/multirepo_eval.py --gate
```

调用链探针可单独运行 `python scripts/multirepo_eval.py --evalset datasets/eval/multirepo_call_path_probe.jsonl --out artifacts/eval/multirepo-call-path-probe-scorecard.json`。该探针没有可冒充为主门禁的独立 baseline；目前用于检查入口与被调用实现是否同时进入 Top-5。Scorecard 的 `all_gold_at_5` 要求一题的全部 gold 文件同时命中，`incomplete_case_ids` 列出仍缺文件的题；原有 `bad_case_ids` 只表示零命中。2026-09-24 的诊断中 Hit@5 为 1.0，但 `all_gold_at_5` 仅 0.625（5/8 条，平均 Recall@5 为 0.8333），所以不能把“至少命中一个文件”当成调用链完成。缺失分别涉及 `render→loader`、`redirect→reverse` 和模板引擎的 `EngineHandler`。题面中的模板渲染用例明确限定为立即返回 `HttpResponse` 的 `render()` helper，以排除延迟渲染的 `TemplateResponse` 另一条有效路径。

转移题可用 `python scripts/multirepo_eval.py --evalset datasets/eval/multirepo_call_path_transfer.jsonl --out artifacts/eval/multirepo-call-path-transfer-scorecard.json` 单独运行。2026-09-24 原排序的 `all_gold_at_5` 为 4/4，说明这批题没有提供进一步改善完整率的空间；不能拿它们为自动调用点加分背书，也不能并入旧基线后比较总分。

未点名入口探针可运行 `python scripts/multirepo_eval.py --evalset datasets/eval/multirepo_call_path_generalization_v1.jsonl --out artifacts/eval/call-path-generalization-scorecard.json`；已知入口的错误目标门禁运行 `python scripts/call_path_negative_eval.py --gate`。两者分别检查文件召回与静态误连，不能相加或解释为真实模型调用链正确率。

转移题可单独运行 `python scripts/multirepo_eval.py --evalset datasets/eval/multirepo_call_path_generalization_transfer_v1.jsonl --out artifacts/eval/call-path-generalization-transfer-scorecard.json`。首次默认路径权重 0.25 的完整命中为 1/4，说明跨文件目标召回尚不稳定；在旧开发题上试降路径权重到 0.1 反而使完整命中从 2/4 降到 1/4，因此未改默认配置。

调用边候选可用 `python scripts/call_edge_expansion_ab.py --repo data/repos/django --manifest datasets/eval/multirepo_manifest.json --evalset datasets/eval/multirepo_call_path_generalization_holdout_v3.jsonl --grounding datasets/eval/call_edge_grounding_v2.json --line-weight 1 --out <scorecard.json>` 单独诊断。`--manifest` 校验固定 Git 身份与 gold；`--out` 保存仓库指纹、数据集 SHA、候选版本、逐题及汇总指标。不传 manifest 的报告明确标为 `unverified`。源码行词重叠权重 1 在旧 12 题上使精确调用边首位从 9/12 升到 11/12；新固定 4 题首位 2/4、前三 4/4，两条默认漏目标题的首位候选都找回目标文件。另运行单文件负例得到 `negative_suggestion_at_1=4/4`；自仓库迁移烟测精确边首位 1/4、前三 2/4。后两组均已揭盲且不应混入 Django 正例分数。故新题只支持“候选导航值得保留”，不支持自动改写最终答案：签名 Cookie 的 `dumps`／`loads`、日期字段的 `make_naive`／`make_aware` 仍有排序歧义。该脚本不请求模型，`repo_inspect.py related` 复用候选逻辑并要求人工／Codex 继续读取源码。

曾临时试验把候选限制在检索命中代码块所交叠的函数：Django v3 正例精确边 Top1 从 2/4 到 3/4，但单文件负例仍误导 4/4，自仓库精确边 Top3 从 2/4 降到 1/4。因此未保留该规则，也未把这组已揭盲数据上的局部收益纳入默认策略。

Aider 固定源码可运行 `python scripts/call_edge_expansion_ab.py --repo data/repos/aider --manifest datasets/eval/aider_manifest_v1.json --evalset datasets/eval/aider_call_edge_positive_v1.jsonl --grounding datasets/eval/aider_call_edge_grounding_v1.json --line-weight 1 --out <scorecard.json>`。两个单文件负例集分别用同一命令替换 `--evalset`，去掉 `--grounding`；`negative_suggestion_at_1` 量化错给候选的题数。曾试用“题面有调用动词、无 local 字样”做保守意图门槛，普通负例会拒给候选，但同文件调用近邻负例仍误导 4/4，故未上线。以上小样本首轮均已揭盲；不能作为统计意义上的通用模型能力结论。

若本地没有 `data/repos/aider`，先从 [Aider 官方仓库](https://github.com/Aider-AI/aider) 克隆到该目录，再检出清单指定的完整提交 `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`；不要用不断变化的默认分支直接比较分数。首次诊断的紧凑身份与指标记录在 `benchmarks/aider-call-edge-first-look-v1.json`。

已知入口后的边排序可另行用 `--anchors datasets/eval/aider_call_edge_source_anchors_v1.json` 诊断；同时传入 Aider 三份 evalset（正例逐题精确边还需传 grounding 并用 `--case-id` 只选 4 条正例）。与原始检索种子模式分开报告：首轮正例精确边 Top1 4/4，8 条负例中 7 条不给跨文件候选，剩余 `main()` 的同文件 helper 问题仍有无关跨文件建议。身份与结论在 `benchmarks/aider-source-anchored-first-look-v1.json`。源锚点由标签提供，这不是自然语言入口发现、Agent 决策或最终回答的分数。

`call_edge_expansion_ab.py --auto-anchor --line-weight 1` 不使用人工入口：从检索前 5 文件选首个非测试文件的首位函数，沿其静态直接调用出候选；此模式与 `repo_inspect related --auto-anchor` 共用选择逻辑。固定 Aider 4 正例／8 负例及 Django 4 正例／4 负例的已揭盲诊断，精确边首位分别为 3/4、2/4，负例全部返回空候选；原未锚定模式在这 12 条负例中全部给出跨文件建议。报告 `benchmarks/query-only-auto-source-first-look-v1.json` 保存仓库身份、题集 SHA、配置和逐题结果；它是可选导航的开发证据，不是盲测泛化、最终答案或真实模型能力。

同一显式入口模式在已揭盲的 Django v3 4 正例与 4 单文件负例上，精确边 Top1 4/4、负例不给跨文件候选 4/4，记录在 `benchmarks/django-source-anchored-first-look-v1.json`。跨仓库分数说明该组件值得作为“先确认入口、再查调用”的 Codex 工作流保留，但这些入口来自人工标签，不允许把组件率当成端到端生产能力。

`python scripts/source_discovery_eval.py --out <scorecard.json>` 将上述 Aider 12 题、Django 8 题的人工入口只用于事后评分；检索阶段只接收自然语言 query。每题保存正确源文件首次出现名次、正确函数范围首次与结果片段相交名次、正确定义行首次出现名次，以及实际返回路径。首轮 Source file@5 为 19/20、function-span@5 为 14/20、definition@5 为 12/20；@20 分别为 20/20、15/20、12/20，紧凑记录见 `benchmarks/source-discovery-first-look-v1.json`。Django 用固定 v3 正例与单文件负例，Aider 用固定正例、普通负例、同文件调用负例。该脚本核对固定 Git 身份和题集／锚点身份；所有题已揭盲，没有独立盲测结论。长函数会让范围相交指标偏乐观，定义行可见也不能代替语义判断；文件命中不可算作函数命中。

同一脚本还在人工审定的正确文件内评测可选 `repo_inspect locate`：符号 Top1 15/20、Top3 19/20、Top5 20/20；入口文件进入检索 Top5 且正确符号进入文件内 Top3 的上界为 18/20。紧凑记录见 `benchmarks/file-local-symbol-first-look-v1.json`。这不是自动选文件成功率，更不是答案正确率；该数据集已揭盲，不能据此调高默认 Agent 的排序权重。

不传人工路径的 `repo_inspect discover` 对同 20 题输出检索前 5 个文件内各前 3 个函数候选。实际正确文件＋符号入选 18/20，首组首符号正确仅 10/20（Django 2/8），详见 `benchmarks/joint-source-discovery-first-look-v1.json`。因此只能作为带行号的核验清单，不能自动选首位作答；后续需要独立题和错误候选拒答评测。

`python scripts/source_discovery_eval.py --suite transfer --out <scorecard.json>` 对固定 Django／LeetCode 的 8 条单文件任务做函数定位迁移诊断，题目与锚点见 `source_discovery_transfer_v1.jsonl` 及同名前缀的两个锚点文件。首次运行前冻结输入，结果记录在 `benchmarks/source-discovery-transfer-first-look-v1.json`：源文件 Top5 为 8/8、定义行 Top5 为 6/8、正确文件＋符号进入前 3 个候选为 7/8。题目取自已揭盲的文件检索集，故仅是符号定位组件的首次迁移检查，不是端到端盲测；其中 ModelBackend 问题同时询问认证和权限，单独标 `authenticate` 只能覆盖部分行为，不能据此调整算法并宣称泛化。该组自首次运行后也已揭盲，只作回归诊断。

`discovery_explicit_missing_v1.json` 在运行拒答规则前冻结了 Aider／Django 各 4 条“类确实存在、点名的直接方法定义不存在”的近邻问题；审计路径和具体缺失名称逐题列明。`python scripts/discovery_absence_eval.py --gate --out <scorecard.json>` 校验固定 Git 清单、类定义存在和该直接方法确实未被索引，再对 8 负例与上述 20 正例运行 query-only 检查。首次结果为精确缺失题拒给候选 8/8、正例误拒 0/20，基线见 `benchmarks/discovery-absence-baseline-v1.json`。原命令对 8 题每题都返回 5 组共 15 个近邻候选；新状态只意味着“无直接索引定义”，不能证明继承、动态生成或其他运行时行为不存在。所有题已揭盲，不将其外推到一般不可回答问题。

`direct_edge_verification_v1.json` 冻结已知入口与目标的 14 条直接调用核验题（Django／Aider 各 7 条，正负各半）。`python scripts/edge_verification_eval.py --gate --out <scorecard.json>` 先校验两个固定 checkout 的清单与安全路径，再复用只读索引逐题检查是否有调用点及目标定义行；首次正例 7/7、负例 7/7、误支持 0，正例共返回 70 行源码片段。负例的 `not_established` 是静态证据不足，不是运行时不存在证明；题目已揭盲，分数只用于组件回归，不可当作自然语言 Agent 准确率。

每条多仓库用例必须包含唯一 `id`、manifest 中的 `repo_id`、非空 `query`、安全且存在的 `gold_files`、`task_type` 和 `difficulty`。入口还会核对每个本地 Git checkout 的 HEAD 与 origin；缺失或不匹配返回 2，且不会自动 clone 或访问网络。

`patch_outcome/django/` 保存真实 Django regression 的 fault 与 oracle unified diff。清单同时记录上游源码／测试文件内容哈希、patch 哈希、remote、commit、许可证和只允许导出的 archive roots。运行时从固定 commit 导出临时副本；fault/oracle 只属于 evaluator 自检材料，不会进入 `CandidateRequest`。运行 `python scripts/patch_outcome_eval.py` 可执行 oracle harness self-check，`--gate` 执行独立 0／1／2 门禁；外部 static patch 仅限可信本地候选，必须显式确认 `--allow-local-code-execution`，临时目录不构成 OS sandbox。
