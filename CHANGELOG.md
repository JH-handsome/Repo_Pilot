# Change Log

本文件用于记录 RepoPilot 每次修改的主要内容。后续修改请按时间倒序追加，重点写清楚改了什么、影响哪些文件、是否需要注意测试或配置。

## 2026-10-18 — 多语言源码检索与项目学习

- 将浏览器／CLI 共享索引、只读搜索、规划、桌面入口、教学和离线评测切换为语言无关的文本源码加载；保留旧 Python 加载与切片 API。未知扩展名及无扩展名源码也可参与检索。
- 支持 UTF-8／BOM 和带 BOM 的 UTF-16，统一拒绝二进制、不可解码及超过 2 MiB 的文件；扩展依赖／构建目录与敏感文件过滤。索引缓存格式升级，非 Python 文件也触发增量刷新。
- 上下文代码块使用实际语言标签，教学及代码生成提示不再限定 Python；非 Python 文件不再被 Python AST 误解析。Python 静态结构导航仍保留，其他语言暂用文本证据与模型分析。
- 列文件默认覆盖所有类型；非 Python 规划不再推荐 Python 验证，新增常见多语言测试／构建命令前缀，仍要求原有逐次审批。
- 增加跨语言检索、纯 TypeScript 教学证据、缓存增删改与恢复、编码、过滤、语言标签及命令审批回归测试。
- 旧检索门禁未全通过：自仓库 Top-5 召回从基线 95% 降到 90%（额外漏掉旧 Python 空仓库题）；多仓库及 Agent runtime 门禁因索引语料扩大而身份不兼容。原题目、基线与阈值保持原样，尚不能宣称检索质量全面通过回归。

## 2026-10-18 — Trace 与 JSON/JSONL 产物归档

- 增加 `--convert-traces --archives-only`，将已经归档的旧检索、Agent、逐条评测和完整 trace 转入统一运行与事件表。已有 ID 沿用并遵守版本及事件历史约束；无 ID 按规范内容哈希去重，重复导入或复制文件不新增运行。同版本冲突保留原记录，缺失历史时间不伪造；阶段副本和汇总仍仅归档。
- 复用 `artifacts/traces.sqlite3`，新增 `json_files` 保存脱敏文件内容、来源路径、哈希和版本；结构 v1/v2 在写入时事务升级至 v3，保留已有 trace 和调用观测。
- trace JSON 导出、检索评测与能力评测的 JSON/JSONL 输出增加自动归档，原文件继续保存；数据库失败记录警告，不影响已有文件。
- 新增 `scripts/import_json_artifacts.py` 递归导入历史产物，重复导入去重，修改后保留新版本，无效文件报错且不部分写入，不删除源文件。
- 新增归档、并发、迁移、关闭配置、失败保留和导入命令回归测试；使用与查询说明见 README 的数据库存储章节。

## 2026-09-29 — 工具上下文预算硬约束

- 修正工具结果兜底预览可能突破 `max_context_chars` 的问题；按实际 JSON 字符数限制返回内容，极小预算退化为可序列化占位值。300 字符预算、需转义的长输出及 1 字符边界均有回归测试；默认检索与模型调用逻辑不变。

## 2026-09-28 — 调用点级候选导航与精确边评测

- ASK 与兼容答案生成器的引用校验改用实际送入模型的上下文范围，截断后的未展示行和未放入预算的文件不再算证据；引用路径匹配也拒绝伪造目录前缀及多文件共用的歧义短路径。检索结果与提示正文格式保持不变。
- 修正统一 Agent 与 LangGraph 只读链路的引用证据边界：引用范围现在从压缩后真正送入模型的完整源码行收集，搜索结果仅剩元数据、`read_file` 正文被删空或源码行被截断时，不再允许用原始工具返回的行范围验证引用。保持原有工具与回答协议不变。
- Codex 只读命令 `repo_inspect trace --unresolved` 现在可显式列出未被静态导入／同文件边解析的点号调用及行号，不推断目标路径。固定 Aider 的 `Commands.cmd_lint` 第 379 行 `self.coder.linter.lint` 可见，解决空 `edges` 隐藏动态调用点的问题；默认 Agent 与排序路径不变。
- `repo_inspect related` 增加显式 `--auto-anchor`：只从检索前五文件的首个非测试源码文件选一个未核验函数，再给其可静态解析的跨文件边；歧义时弃权。固定 Aider／Django 已揭盲诊断精确边 Top1 5/8、负例无跨文件建议 12/12，优于未锚定模式在同负例上的 12/12 无关建议，但仍漏掉入口缺失／选错的正例。评测与命令共用规则，默认 Agent 和排序不变。
- 对冷建索引时反复出现的短标识符分词增加有界进程内缓存，长标识符不缓存且结果列表独立；固定 Django 直接冷建单次观测 37.4 秒降至 19.3 秒，文件／chunk 数相同。全部检索和 Agent 门禁保持原分数，索引格式与默认排序不变；不将此称为召回提升。
- `repo_inspect.py` 新增本地 JSONL `serve` 模式，供 Codex 连续调用现有只读工具时复用同一进程内索引；每次请求仍检测源码变更，非法／写入请求返回错误而不终止会话。无页面、网络服务或模型调用，默认检索排序不变。
- 新增固定 Aider／Django 的 8 条双文件多步源码检索基准及独立回归门禁。首轮默认检索同时找齐入口和提供者 6/8；路径权重 A/B 未带来稳健提升，因此未修改默认排序。该题集现已揭盲，仅用于防回归。
- 修复 `repo_inspect.py` 在 Windows GBK stdout 与 Codex UTF-8 消费端之间的中文源码乱码：只读 JSON 用 ASCII Unicode 转义，解析后仍还原原文；新增跨编码子进程回归测试。不改变检索或 Agent 决策。
- 统一 Python 文件发现与只读工具的敏感路径拒绝策略，排除常见凭据文件／目录和私钥后缀；`search_code`、`read_file`、`list_files`、静态追踪及后续索引快照不再纳入这些路径。合成凭据测试覆盖文件发现、manifest、搜索及读取。名称拒绝表不是内容级秘密检测；历史缓存不自动批量删除。
- 新增显式 `repo_inspect verify-edge <源文件> <源符号> <目标文件> <目标符号>`，只在静态直接调用及目标定义行均可定位时返回两侧源码证据；否则标记 `not_established`，不把缺失静态证据当成运行时否定。固定 Django／Aider 共 14 条正负题的离线门禁为正例 7/7、近邻负例 7/7、误支持 0，正例证据 70 行；默认 Agent 不变、未运行在线模型。
- `discover` 对明确定位一个不存在的 `Class.method` 直接定义增加保守拒给候选状态，并保留继承／动态定义的不确定性提示。先冻结固定 Aider／Django 各 4 条精确缺失近邻题，再实现与评测：8/8 返回无直接定义、原 20 条正例误拒 0；默认 Agent 不变。新增身份锁定的 `discovery_absence_eval.py --gate`，不将此窄项描述为通用拒答能力。
- 增加可选 `repo_inspect discover <query>`，一次列出检索文件及各文件的函数定义候选，仍不改变 Agent 默认回答。固定 Aider／Django 20 道已揭盲题无人工入口时正确文件＋符号候选覆盖 18/20，但第一组首位仅 10/20，故不自动采纳，记录见 `benchmarks/joint-source-discovery-first-look-v1.json`。
- 新增可选 `repo_inspect locate <path> <query>`：在已选定 Python 文件内按问题列出真实函数定义与行号；无词汇证据时返回空，不改变默认 Agent。固定 Aider／Django 20 条已揭盲题，人工给正确文件后的符号 Top3 为 19/20，两阶段可达率上界为 18/20。这不证明自动选文件或回答能力，记录见 `benchmarks/file-local-symbol-first-look-v1.json`。
- 可选 `repo_inspect related --source-path ... --source-symbol ...` 只从已确认的源函数枚举导入绑定跨文件边；不改默认 Agent 检索与无锚点命令。固定 Aider 的“人工给正确入口”组件诊断从跨文件精确 Top1 3/4 到 4/4，单文件误导从 8/8 到 1/8。剩余 `main()` 近邻题证明源锚点不能替代目标判断；这组分数不代表自动发现入口或最终回答。
- 同一显式入口模式在已揭盲的固定 Django 4 正例／4 单文件负例上得到精确边 Top1 4/4、负例无建议 4/4；对应身份和数值另存独立 scorecard。该结果只验证入口已知后的组件能力，不升级默认 Agent 或无锚点 `related`。
- 将原本仅用于设计参考的 Aider 固定版本加入独立、离线的迁移评测清单，本地 checkout 置于 Git 忽略目录，未复制进产品源码或改写既有多仓门禁。4 条真实跨文件题首次精确边 Top1／Top3 均为 3/4；缺口是入口文件未进入 Top5。4 条普通单文件题与 4 条真实同文件调用近邻题均出现 4/4 的无关跨文件候选。基于检索片段限制候选函数、基于调用动词判断意图两种试验各在迁移或近邻负例上失败，均未上线。
- 精确 `trace` 原本对 `self.get_ranked_tags_map()` 返回空边；现在可显式加 `--same-class`，仅当被调方法在当前类中有明确且无歧义的定义且 `self` 未被重绑时，输出单独的 `same_class_method_candidate` 与两侧行号，不假称动态分派已确定。默认 Agent 的 `trace_call_path` 不变；固定 Aider 的 `RepoMap.get_repo_map` 第 135 行已实际定位到同文件目标定义。
- 调用边诊断脚本现在可验证固定 manifest 并导出带仓库指纹、数据集 SHA、候选版本、逐题结果的 JSON scorecard；新增单文件负例建议率和拒绝建议数。固定 Django 单文件问题首次误导建议 4/4；当前仓库跨文件迁移烟测精确边 Top1 1/4、Top3 2/4（仓库未固定，报告标为未验证）。这些失败阻止 `related` 候选进入默认 Agent 答案链路；没有发起在线模型评测。

- 将未点名跨文件题的验收从“目标文件对了”细化到真实入口符号、调用点与目标函数。旧 12 题的候选调用边首位原为 9/12；加入源码调用行词重叠后为 11/12。随后才冻结并首测 4 条新 Django 转移题：默认 Top-5 全文件 2/4，调用边候选精确首位 2/4、前三 4/4，两个漏目标题的候选首位都找回目标文件。`dumps`／`loads` 等歧义证明仍不能把候选首位直接当答案或自动改默认排序。
- 新增无需模型和页面的 `repo_inspect.py related`：从真实混合检索命中的源码出发，只列静态导入绑定且目标定义存在的跨文件边，给出入口调用行和目标定义行，供 Codex 继续 `read` 核验。拒绝未解析属性与有歧义的重复定义，输出按完整候选项裁剪。固定 Django 消息 Cookie 签名路径试跑返回正确源码位置；该命令是可选导航，不改变普通 Agent 的默认搜索和回答。
- 一次性只读命令支持显式 `--cache-dir` 指向仓库外可写目录，搜索与调用边导航复用同一索引注册表；目录在仓库内会被拒绝。固定 Django 本机一次冷启动约 65 秒、再次独立启动约 8.6 秒；这是该机器的单次观察，不是跨机器延迟保证。缓存快照由既有校验、大小上限与格式版本约束保护，未改默认缓存位置。

## 2026-09-28 — 未点名跨文件转移诊断与单文件导航

- 在首次运行前另冻结 4 条不直接给出入口符号的 Django 跨文件转移题。默认 Top-5 入口命中 4/4，但双文件完整命中 1/4，独立暴露 URL 有效性、消息 Cookie 签名与缓存日期格式化目标漏召回。旧开发题的路径权重 0.1 A/B 由完整 2/4 降至 1/4，故保留默认 0.25；这不是排序能力改善。
- 只读 `repo_inspect.py outline` 对指定 Python 文件局部 AST 解析，输出完整嵌套函数名和行号，支持从搜索入口继续精确 `trace`，不新建页面或调用模型。真实 Django 装饰器文件冷启动约 0.6 秒，较复用全仓索引的首次约 31 秒明显缩短；该单次时延不是性能门禁。路径策略沿用只读工具，输出按完整符号项裁剪。

## 2026-09-28 — 未点名调用链探针与嵌套闭包导航

- 在固定 Django 上游源码中先冻结 4 条不直接点名入口函数的跨文件查询及 3 条源码审定的错误目标，再运行首轮检索。Top-5 至少命中一个文件为 4/4，但全部目标同时命中仅 2/4；密码验证器动态导入与重置令牌 base-36 解码的目标文件均漏出 Top-5。它们现已用于诊断，后续不是全新盲测；不因单题缺口直接调整全局排序。
- `trace_call_path` 允许显式追踪嵌套函数的完整限定名，解析该作用域内的直接导入及可静态定位的兄弟闭包调用；返回闭包的外层工厂仍不被误称为立即调用闭包。固定 Django 的 `user_passes_test.decorator._redirect_to_login` 可定位 `django/contrib/auth/views.py` 中的 `redirect_to_login`。同一限定名若在条件分支重复定义则拒绝静默选择。新增 3 条错误目标的离线静态组件门禁；该门禁不证明自然语言模型会选择正确入口。

## 2026-09-26 — 显式符号定义定位

- 对明确点名 Python 标识符且询问定义／实现位置的查询，利用已有 AST 符号索引将定义所在代码块排到调用示例前；`Class.method` 限定到指定类。问调用处或测试位置时不触发该加分。自仓库 `trace_call_path` 的定义此前不在 Top-5，现在进入前三；固定 Django `LoginView.form_valid` 在相同首名文件内从错误片段对齐到定义片段。该信号不解决未点名符号的自然语言定位或跨文件调用链召回。
- 排序身份升为 v7，旧 v6 基线保留；自仓库、多仓、Agent runtime 与调用链契约另立对应基线。固定 20 条真实多仓题质量分保持 83.93，离线 Agent 契约分保持 84.58，调用链正负挑战 4/4 与 5/5。自仓库整体分数随新增源码／测试有所变化；同一当前工作区的 A/B 显示此定义信号未改变原 20 题的文件级 MRR 或 nDCG。真实模型能力仍未验证。
- v7 多仓首次延迟基线采样过快，后续两次门禁虽质量一致但因 Django 查询整体耗时波动触发 P95 阈值；同进程开关定义信号的 A/B 未显示该信号导致变慢。性能基线改用三次 P95 观测的中位数对应完整 scorecard（约 644 ms），再次运行门禁通过。延迟门禁仍保留 1.5 倍阈值，单机计时不等于跨机器性能保证。

## 2026-09-24 — 检索评分复用

- 普通 Agent／只读图的 `trace_call_path` 改为复用混合搜索已有的仓库快照与符号索引，避免同轮导航再次全量加载、切片和解析；固定 Django 单次诊断中旧路径约 9.1 秒、复用路径约 24 毫秒，输出调用边一致。新增无需页面或外部模型的 `repo_inspect.py` 只读 JSON 命令，复用现有搜索、精确符号索引、静态追踪及文件路径策略；自仓库已知符号查询可绕开 Top-5 被评测／测试文件占据的检索噪声，但不改变默认排序。
- 增加固定 Django 的调用链正负挑战集与独立误连门禁：首次旧 v2 策略在 4 条正例全部通过时仍对 5 条负例误连 2 条。通用修复使题面显式 `target()` 精确优先，多个显式目标超出单目标契约时拒答；同集回归正例 4/4、负例 5/5、误连 0，旧首次结果保留，修复后的同集分数不冒充新盲测。运行时策略身份升为 v3，旧 v1/v2 基线保留。
- 新增显式调用链的离线统一 Agent 运行时门禁：题面首个函数／方法名驱动真实检索、静态追踪和入口／目标两次源码读取，并分别核对 gold 读取与有效引用。固定 Django 留出集和转移集共 8 题的契约成功率为 8/8；这些题已参与策略开发，不作为全新盲测，也不代表在线模型推理或补丁能力。最高分片段可能是被调实现而非入口，故入口选择需核对定义；同一目标的多个调用点按目标去重，歧义的不同目标拒绝自动选择。
- 调用链运行时契约升为 v2：入口读取窗口围绕实际调用行，目标读取从定义行起；评分要求两处行号都确实出现在工具返回的文本中，避免仅因读取了正确文件就给出完整证据分。旧 v1 基线保留，新门禁默认使用 v2 身份。
- 对问题中精确的 `Class.method()` 增加文件内代码块定位：仅替换已选中文件的代表片段，保留文件分数与 Top-K 文件集合；固定 Django 登录题从同文件错误方法改为 `LoginView.form_valid` 片段。检索器身份升为 v6，三套相关基线另立版本，v5 保留作同数据集对照；不把片段改善冒充跨文件召回或模型能力提升。
- 修复受限 Windows 环境中磁盘索引缓存目录不可写时，临时文件创建反复重试导致的长时间卡顿：快照现在先单次独占创建临时文件，再序列化；权限失败立即保留进程内索引并跳过磁盘写入。增加权限拒绝回归测试，全量单测由此能够完成。
- 新增 4 条跨 Django 子系统的调用链转移诊断题。原检索已完整命中 4/4；精确 `Class.method` 调用点加分虽改善旧留出题，但在转移题无完整率收益且一题提升无关目标，因此未进入默认排序。另一个只在命中文件内对齐精确方法片段的只读 A/B 不改变 28 题的文件列表，尚未进入产品链路。
- 对 `trace_call_path` 增加局部变量／参数遮蔽导入的静态作用域检查，避免把局部调用误报成跨文件边；新增 7 条固定 Django 目标边的组件门禁，量化目标召回和行号定位，不冒充端到端模型能力。
- 新增只读 `trace_call_path(path, symbol)`：在已索引 Python 源码上解析函数内调用、本地跳转、导入别名和有限包级再导出，输出带行号的静态候选边，接入普通 Agent 与只读图，不执行仓库代码、不越过既有路径策略，且导航结果不自动充当引用证据。固定 Django 源码验证 `render→loader`、`redirect→reverse`、模板引擎与登录视图路径；自动排序仍不变。
- 将 4 条 Django 调用链题单独冻结为留出集。调用点限定的只读排序 A/B 在原诊断集改善了完整路径率，但留出集未提升，因此没有启用默认排序；静态调用导航作为显式工具提供。
- Django 跨文件调用链诊断集从 4 条扩为 8 条，新增重定向／URL 反向解析、multipart 上传、签名 HMAC、模板引擎注册表路径。Scorecard 新增 `all_gold_at_K` 和未完整命中的 case 列表，保留旧质量分公式；Hit@5 虽为 1.0，`all_gold_at_5` 只有 0.625，平均 Recall@5 为 0.8333。只读 A/B 发现泛化的文件级依赖加分会引入无关模块并挤掉正确入口，因此拒绝作为默认排序规则，继续研究调用点级证据。
- ASK、Agent 和离线检索流水线将初检的全量混合分数显式传给最终过滤，避免同一查询对全索引重复评分；保留独立过滤 API 的重算路径，并用一致性测试与固定门禁验证排序不变。此改动不解决跨文件调用链的 Top-5 缺口。

## 2026-09-23 — 真实补丁／测试结果基准

- 新增 4 条基于固定 Django 源码调用边的跨文件诊断题，避免只用单文件符号题评估 Coding RAG；单独运行时暴露 `render()` 入口在第 3 位、被调用的模板 loader 在第 7 位。题面消除与延迟渲染路径的歧义；路径权重降为 0 降低主基准与探针质量，依赖图权重 0.1 无变化、0.4 明显退化，因此不改默认排序或既有门禁基线。
- 修复普通 Agent 的 `inspect_diff` 绕过只读策略：先用 NUL 分隔的 Git 变更路径和索引模式筛选，再只取允许文件的差异；排除 `.env*`、忽略目录、符号链接（包括已删除链接），防止未审批的 diff 工具输出泄露本地配置或外部目标路径。
- 混合检索查询的路径评分改为复用每个文件预先计算的 filename stem 词元，查询标识符变体也只计算一次；不改变评分公式或持久化缓存格式，固定多仓门禁验证排名不变并降低查询延迟。
- 只读 Agent 工具增加仓库读取边界：拒绝直接或通过符号链接读取 `.env*`、`.git`、虚拟环境等忽略路径；文件列表和 Python 索引排除符号链接，避免仓库外源码进入检索与模型上下文，并补充复现性回归测试。
- 检索查询阶段新增通用中文开发术语到英文标识符的 query-only 扩展，不改变正文词频索引；自仓库 v2 评测的 Hit@5 由 0.80 提升到 0.95，四条坏例降为一条，固定 Django／LeetCode 多仓质量分数未变。检索器升为 v5，三套相关基线另立新版本，旧基线保留用于 A/B，不混用门禁身份。
- 自仓库检索评测另立 `trace_v2.jsonl` 和独立基线／门禁：保留旧 20 例历史身份，将已退出普通用户路径的 `AnswerGenerator` 标注改为统一 Agent 的实际编排、observation、审批、只读工具和空索引代码。新分数与 v1 不直接比较，真实坏例继续作为排序改进输入。
- 将默认任务集升级为独立 v2：在保留两例 v1 清单和基线的同时，加入 Django `termcolors` 的前景色／背景色／选项解析故障；新任务通过原始测试、故障失败和 oracle 恢复三段校验。v2 使用单独基线、门禁和 no-op 负例，避免跨数据集分数误比。
- 新增固定 Django 提交的真实 patch outcome harness：从受限 archive roots 物化源码，逐 case 验证 pristine 通过、fault 注入后失败、候选补丁应用和相同上游测试结果；该分数明确标记为补丁结果验证，不冒充 LLM 智能。
- 新增两条真实回归任务、源码／测试／patch provenance 哈希、oracle self-check、可复现 no-op 负向候选，以及与现有能力门禁隔离的 portable baseline 和 0／1／2 gate。只有全部通过的显式 oracle self-check 可更新基线，static 候选与 oracle baseline 可比较为退化而非不兼容。
- 候选接口不接收 fault/oracle；diff 在应用前严格拒绝未允许路径、删除、新建、rename/copy、路径穿越、绝对路径和 symlink。测试命令使用固定 argv、`shell=False`、最小脱敏环境、超时及执行期有界输出。
- 执行边界明确为 `trusted_local_candidate_only`／`workspace_isolation_not_os_sandbox`；临时工作区和 Python 网络 guard 只提供可复现性与防误用保护，不用于执行恶意代码。
- 新增 `AgentCandidateProvider`：复用普通 Agent LangGraph 和逐次 fingerprint 审批，只在第二个 evaluator 临时副本内自动批准私有范围复核后的 `apply_patch`，永不批准 `run_command`；最终累计 diff 返回外层真实测试，fault 工作区保持不变。
- 普通运行入口新增可选的进程内索引 registry 注入，产品默认全局缓存路径不变；Agent candidate 使用 `disk_cache=False` 的临时 registry、临时 rollback snapshot 且关闭 SQLite trace，避免短命仓库留下缓存或敏感证据。
- CLI 新增显式 opt-in 的 `agent-live`：必须同时确认本地候选代码执行、网络调用和潜在费用；限制 LLM timeout/output tokens，调用前提示公开源码披露边界。live 只输出真实修复 scorecard，v1 禁止用 oracle baseline 执行伪回归 gate。
- 新增无网络 generic fake ChatClient 测试，覆盖真实 graph 的 search/read/apply/approval/resume、越权和命令拒绝、外层隔离及脱敏；另以真实 pinned Django 单 case 验证 scripted offline contract，但不计作 live 模型修复率。

## 2026-09-15 — Capability 基线与回归门禁

- 新增统一 Capability Scorecard，量化检索质量、索引与查询效率、上下文成本及 Agent 离线能力，并记录可复现实验元数据。
- 新增评测集严格校验、显式基线更新和回归门禁；修复模块迁移后失效的 gold 路径。
- 检索评测改为一次建索引、多 case 复用，并默认排除下载语料与运行产物目录。
- 参考 Aider 固定提交的 repo map/benchmark 思路预留结构 ranker 接口，本阶段未实现 graph ranking，也未复制上游源码。
- 在同一基准上 A/B 验证依赖图扩散：该信号未提升质量且高权重会退化，因此默认关闭；采用可解释的路径标识符精确匹配补强 Hybrid Search。
- 新增进程级有界 `RepositoryIndexRegistry`：通过 Python 文件 manifest 感知变更，跨 Agent 轮次复用索引，以每个配置键的 single-flight 避免并发重复构建；失败候选不会覆盖最后一次成功快照。
- 文件发现改为在目录遍历阶段剪枝 `data`、`datasets`、`artifacts` 等排除根目录，避免缓存命中时仍扫描大型下载语料目录。
- 统一 ASK／Agent 检索 trace 新增索引命中状态、扫描／构建耗时、文件／chunk 数和变化文件数；索引键隔离切片参数、排序权重、排除目录和 retriever 版本。
- `RepositoryIndexRegistry` 新增跨进程安全 JSON 磁盘快照与按文件增量更新：默认写入操作系统用户缓存目录，使用 gzip、校验和、严格格式／配置／仓库身份校验及原子替换；缓存损坏或不兼容时全量重建，写入失败不影响已发布的内存索引或最后成功快照。
- BM25 缓存改为稀疏 token-frequency 数值状态，恢复时不再展开完整 token occurrence 或重新执行 AST／tokenization；路径派生元数据统一使用 repo-relative display path，并将 retriever 版本升级为 `hybrid-bm25-v4`。trace 可区分 `memory_hit`、`disk_hit`、`incremental`、`full_rebuild` 并报告复用／重建／移除及磁盘耗时。
- 修复 Agent 离线评测统计语义：开启只读检索，以有序候选对比 changed/test gold，输出 result files、缺失 gold 和 repo 分层；同时识别现有 50-case fixture 没有负样本，将候选指标标记为非判别性的 `fixture_diagnostic`，不再误称真实 Agent 定位能力。
- Capability schema 升至 v2，同时校验两套数据集身份；固定规划指标归入 `offline_contract`，无判别力的候选指标不参与综合分或硬门禁，门禁仅保留 fixture 数据覆盖率与 trace／安全契约约束。
- 新增独立真实多仓库源码基准：固定 Django 与 LeetCode checkout，提供 20 条 source-grounded 用例、严格仓库／gold 校验、按 repo／task／difficulty 分层、portable scorecard，以及与原 capability baseline 隔离的 0／1／2 回归门禁。
- 回归门禁额外校验完整检索配置和 retriever 版本，Agent fixture 临时文件写入拒绝绝对路径与父目录穿越，避免用不可比实验或不安全数据产生误导分数。
- 新增固定 Django／LeetCode 真实源码多仓库基准与独立门禁；真实 bad case 暴露最终 Top-K 被同文件相邻 chunk 占满后，最终过滤改为优先保留不同文件，retriever 版本升至 v3。
- OpenAI 兼容 LLM 客户端新增严格有界的暂时性错误重试、可测试指数退避／抖动，以及每个逻辑调用一次的脱敏观测；非暂时性错误立即失败，日志不记录 prompt、响应正文或任何凭据。
- 新增真实多仓库 Agent 运行时契约基准：在固定 Django／LeetCode 20 case 上执行统一 LangGraph 的 `search_code → read_file → cited answer`，scripted policy 只读取 query 与真实 observation，不读取 gold，也不依赖网络或密钥。
- Agent runtime scorecard 分离检索命中、实际选择／读取命中、工具序列、引用存在／已读范围有效性／gold grounding、终态、步数和 case 延迟，并按 repo／task／difficulty 分层；每个 repo 先单独预热，case latency 不包含索引 cold/disk 成本。新增独立 schema、baseline 与 0／1／2 gate。
- Agent runtime schema v2 固定 Top-5 口径，强制 measured cases 全部使用内存索引，并把 memory-hit rate 纳入零容忍门禁；同时报告 indexed／unique-gold／negative file 数，没有真实负样本的仓库会拒绝评测。
- 脱敏 LLM 逻辑调用观测接入统一 LangGraph：每个 `model_decision` 精确关联当前线程的一次性 observation，summary 聚合 attempts／retries／outcome／latency／error class；通用 `ChatClient` 无需实现观测协议。
- Trace SQLite schema 升至 v2，新增固定字段、有界的 `llm_calls` 明细表和按 `run_id` 查询／聚合 API；v1 数据库事务迁移，写库或补录失败仍不改变主运行结果，调用证据不保存 prompt、response、body 或凭据。

## 2026-09-05 — 查询路径、普通聊天入口和多轮历史

- 普通搜索结果统一用 `Path.as_posix()` 序列化；仓库外路径使用绝对形式。过长工具结果不再把结构化路径截成 JSON preview，保留完整路径供后续读取。
- Tk 的 LLM 查询迁移至普通 LangGraph 入口；保留仍用于离线规划、检索、评测及 Graph 工具执行的共用组件。
- 新增标准库 SQLite 聊天轮次存储，按仓库与会话 UUID 隔离；浏览器刷新保留 ID，CLI/Tk 可复用 ID 在重启后恢复已完成对话。完成消息显式传入下一轮 Graph。
- 历史裁剪保留完整轮次并报告省略数量；工具批准仍只保存在待执行状态中，历史不重放写操作，旧批准遇到已推进的聊天会被拒绝。
- 新增搜索到读取的端到端测试、多轮与跨会话隔离、进程／HTTP 服务重启、浏览器脚本、并发追加和批准恢复测试；教学会话的内存约束保持独立。

## 2026-09-04 — Trace SQLite 接入（第一阶段）

- 新增独立 SQLite trace 存储层，使用两张表、WAL、事务、结构版本和事件唯一键；支持累计快照、重复保存、旧版本保护及分页筛选。
- CLI/浏览器普通 Agent 默认保存 trace，可使用 `--trace-db` 或 `REPOPILOT_TRACE_DB` 配置/禁用；授权恢复沿用运行 ID 和开始时间，递增 revision。
- 新增独立令牌认证的历史列表和详情接口；未配置读取令牌时关闭历史访问，数据库由服务端指定。
- 保存诊断副本前移除授权和恢复数据并过滤常见凭据；保存失败尝试本地待补录文件，补录脚本不重跑工具、不删除源文件。
- 增加事务回滚、并发建库、幂等、授权恢复、补录、鉴权和 HTTP 接入测试；教学/评测接入与历史页面留待后续阶段。
## 2026-08-26

- 收紧写工具的仓库边界：`apply_patch` 与 `run_command` 的声明文件在审批前解析到仓库根目录，命令中的显式路径参数拒绝绝对路径、父目录穿越、用户目录缩写和符号链接逃逸；新增越界命令回归测试。
- 将 CLI 和浏览器共用的 `run_unified_query()` 切换为 LangGraph 主运行时；现有 OpenAI 兼容客户端继续使用可修复的 JSON 决策协议，图节点负责模型调用、工具执行、最终引用校验和安全失败收敛，原手写统一循环和旧 ReAct 执行循环均已移除。
- 扩展 LangChain 工具适配层：普通 Agent 按稳定顺序获得 `search_code`、`read_file`、`apply_patch`、`run_command`、`inspect_diff`，所有调用仍经过 `AgentExecutor.call()`；补丁文本保留末尾换行，避免 strict schema 损坏 unified diff。
- 移除普通 Agent 与旧离线 CodeAgent 的 `dry_run`，并移除总执行授权：`apply_patch`、`run_command` 每次都先返回 `approval_required`，同时展示目标文件、具体命令或补丁正文，并生成绑定完整参数的一次性指纹；CLI 逐次询问，浏览器只保存服务端待批准状态并消费一次性 ID。旧离线记忆/日志只读行为统一由 `safe_mode` 控制。
- 新增仓库外快照库：批准后先缓存声明文件的修改前版本，再真实执行，返回操作专属 diff 和快照 ID；CLI/浏览器支持按 ID 回滚，命令必须声明全部 `affected_files`。
- Learning Mode 继续使用独立只读 LangGraph，只绑定 `search_code`、`read_file`，并改用 `safe_mode` 审计只读边界；新增回归测试覆盖精确授权、防重放、快照、diff、回滚和浏览器批准链。

## 2026-08-20

- 浏览器新增独立 `POST /api/learning/session`，把仓库/LLM 配置适配到第 31 次的严格 `start/submit/reflect` 会话入口；原 `/api/run`、双重工具授权和旧 404 边界保持不变。
- 首页新增“项目学习”模式：在页面内存中保存签名 session，按 `next_action` 展示实现提交、修改反馈或优化反思，并渲染项目画像、步骤原因、收益、验收项、证据和常见问题。刷新页面后重新开始，不使用浏览器存储、数据库或 checkpoint。
- `RepoPilotServer` 在进程内生成临时 HMAC 密钥，Learning Mode 不接受 `execute_tools`，也不执行学习者粘贴的代码、命令或测试输出；服务重启后旧 session 会稳定失效。
- HTTP JSON 入口只接受 `application/json`，并增加读取前 1 MiB 限制、UTF-8/JSON object 校验，避免跨站 simple request 直接驱动本地服务；页面增加问答/教学共用的单请求 busy 锁，并对仓库、模型与学习者文本统一转义。
- 新增 `tests/test_learning_frontend.py` 并更新前端回归，离线覆盖连续 action、预检、模型失败、篡改 session、HTTP 200 业务失败、413 请求上限、密钥不泄露、只读边界和页面内存状态。

## 2026-08-19

- 扩展 `coding_rag/learning/models.py`：新增严格的提交、逐项审查、修改反馈、审查/反思记录、会话事件、三类 action 请求和稳定结果模型；单步验收项与逐项 findings 最多 20 条；可恢复 session 只持久化 `awaiting_submission`、`needs_revision`、`awaiting_reflection`、`completed` 四种阶段，`failed` 仅表示本回合结果。
- 新增 `coding_rag/learning/session.py`：用一张单回合 LangGraph 统一分发 `start/submit/reflect`；`start` 复用项目拆解路线，`submit` 在预检或模型审查后等待修改/反思，`reflect` 只推进一个步骤或完成，最终图状态清除临时 `submission/reflection` 输入槽，合法反思仍保存在签名 session 中。
- 会话由调用方保存和回传，服务端不建 checkpoint 或数据库。session 使用至少 32 字节服务端密钥计算 HMAC-SHA256，签名覆盖除 token 外的完整 canonical JSON；结构或签名无效的 session 不回显，携带有效 session 的其他处理失败会保留上一个已验证等待点。
- 步骤审查只依据学习者报告，审查记录固定 `execution_performed=false`。模型自由文本在持久化前由服务端按当前验收项生成具体、定长的缺口与提示，原始提交不写入 session，仅保存域分离 keyed HMAC 摘要和字符数；完整请求、签名 session 与模型审查输出分别设 500,000、450,000 和 50,000 canonical JSON/字符上限，签发前还验证剩余步骤至少保留一条最小完成路径。
- 新增 `tests/test_learning_session.py`，覆盖严格模型、启动、提交重试、反思推进、完成、阶段错配、HMAC 篡改、模型坏输出、JSON 状态和命令文本不执行；`coding_rag/learning/__init__.py` 导出稳定的会话模型、依赖和四个公开运行入口。
- 本次仍未接入前端、HTTP API、checkpoint、数据库、文件写入或代码执行。无状态 HMAC 不阻止合法旧 session 的重放或分叉；服务重启若更换签名密钥，旧 session 会失效。

## 2026-08-13

- 新增 `coding_rag/learning/models.py`：以严格 Pydantic schema 定义项目画像、源码证据、4 到 6 步复现路线、教学必填字段和稳定失败结果；步骤依赖只能指向前序。
- 新增 `coding_rag/learning/workflow.py`：使用 LangGraph 显式编排 `collect_evidence -> analyze_and_plan -> present_step`，用 LangChain `ChatPromptTemplate` 与 `PydanticOutputParser` 校验模型输出，并复用 RepoPilot Hybrid Search、只读 `read_file` 与统一 trace；executor 延迟到证据节点内构造，使损坏 policy 等初始化错误收敛为稳定 `repository_error`。
- Learning Mode 要求每条引用写明模型所声明的 claim，并执行第二层位置校验：路径必须精确匹配，引用范围必须完整落在本次已观察范围内；未知文件、真实行号越界或未观察范围均失败且不展示部分计划。claim 的自然语言蕴含关系本次依靠提示词和人工审查，不宣称自动验证。
- 统一 trace 记录执行器实际 `dry_run/safe_mode` 与固定的 `readonly/execution_requested/execution_enabled` 权限边界，Learning Mode 始终不请求也不启用写执行。
- 新增 `tests/test_learning_workflow.py`，离线覆盖空白教学项与 claim、路径和步骤依赖、三节点路线、默认真实 Hybrid Search、只调用 search/read、模型异常与坏 JSON、空仓库、越界证据、损坏 policy、非有限 provider 数据清洗、安全 flags 和严格 JSON 状态。
- 本次保持 `frontend.py`、`main.py`、现有 `runtime.py` 与 `graph_runtime.py` 不变；不开放写工具，也不需要真实 API key。

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
