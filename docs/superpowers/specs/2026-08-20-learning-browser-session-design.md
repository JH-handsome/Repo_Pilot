# RepoPilot Learning Mode 浏览器接入设计

日期：2026-08-20
任务：第 32 次，Learning Mode 浏览器/API 接入与单用户演示闭环

## 目标

把已经通过离线测试的 `LearningPlan` 与 `LearningSession` 接到现有浏览器界面，使新手可以在同一页面完成：

```text
start -> 查看项目拆解与第一步
submit -> 得到基于提交证据的验收或修改提示
reflect -> 回答优化问题并推进一步
```

每次请求只运行到一个人机等待点。教学模式保持只读，不执行学习者粘贴的命令、测试输出或代码。

## 方案选择

采用独立的 `POST /api/learning/session` 端点。请求外层保存仓库与模型配置，内层 `request` 原样遵守第 31 次的 `start | submit | reflect` 严格契约。

不采用以下方案：

- 三个独立端点：会重复路由、校验和错误映射，并掩盖统一状态机。
- 扩展 `/api/run`：普通 Agent 与教学会话的输入、输出和安全边界不同，混用会破坏现有契约。

## 数据流

```text
浏览器学习面板
  -> POST /api/learning/session
  -> HTTP 适配层校验请求大小与配置
  -> LearningWorkflowConfig + LearningSessionDependencies
  -> run_learning_session(request)
  -> LangGraph 运行到下一等待点
  -> LearningSessionResult JSON
  -> 浏览器替换内存中的 session 并按 next_action 渲染
```

`start` 请求示例：

```json
{
  "repo_path": ".",
  "provider": "deepseek",
  "top_k": 8,
  "recall_window": 2,
  "request": {
    "action": "start",
    "learning_goal": "从零复现 RepoPilot",
    "learner_level": "beginner"
  }
}
```

`submit` 和 `reflect` 必须携带上一响应中的完整签名 session。

## 会话与密钥边界

- 用户选择“当前页面有效”：session 只保存在页面内存，不写入 `localStorage`、`sessionStorage`、文件或数据库。
- 服务启动时生成一次随机 HMAC 密钥，并在该服务进程内复用；密钥不进入 HTML、响应、trace 或日志。
- 刷新页面或重启服务后重新开始。旧 session 在服务重启后因密钥变化而失效。
- HMAC 只防止内容被篡改，不阻止合法旧 session 的重放、回滚或分叉。因此页面必须阻止同一时间重复请求。

## 页面行为

保留原“代码问答”能力，并增加“项目学习”模式：

- `start`：输入仓库路径、学习目标和模型配置，展示项目画像与第一步。
- `next_action=submit`：显示实现说明和测试输出输入框。
- `needs_revision`：显示具体缺口与最小提示，仍停留在当前步骤。
- `next_action=reflect`：显示当前步骤的优化问题与反思输入框。
- `completed`：显示路线完成状态，不再提供推进按钮。

当前步骤必须展示学习目标、任务、为什么这样做、收益、验收项、源码证据、常见问题和优化问题。所有仓库、模型与学习者文本必须经过 HTML 转义。

## HTTP 与错误语义

- 原 `/api/run` 契约和双重执行授权保持不变。
- Learning 业务结果统一返回 JSON；`LearningSessionResult.status/error` 表达业务失败，使有效旧 session 可以被页面继续使用。
- HTTP 层只接受 `application/json`，并在读取完整正文前检查 `Content-Length`；媒体类型错误返回 415，超限返回 413。
- 非 JSON、配置错误或缺少服务依赖返回稳定的 4xx/5xx 响应，不回显密钥或底层异常细节。
- Learning Mode 不接受或转发 `execute_tools`，即使服务以 `--allow-tool-execution` 启动也始终只读。

## 修改范围

允许修改：

- `frontend.py`
- `tests/test_frontend.py`
- 新增 `tests/test_learning_frontend.py`
- `README.md`
- `FUNCTION_MAP.md`
- `CHANGELOG.md`

不修改：

- `main.py`、`web_ui.py`
- `coding_rag/learning/models.py`
- `coding_rag/learning/workflow.py`
- `coding_rag/learning/session.py`
- 现有 Agent/ASK 运行时

## 验收

- 浏览器与 HTTP 测试覆盖 start、submit 不通过、submit 通过、reflect 推进和完成展示。
- 篡改 session、action/phase 错配、模型坏输出均不能误推进。
- 命令文本只作为提示数据，教学路径从不调用写工具或命令执行。
- 重复请求锁、请求体上限、HTML 转义和密钥不泄露有确定性测试。
- 原 `/api/run`、执行双授权、第 30/31 次 Learning 测试、运行时回归和全仓测试全部通过。

## 明确不做

- 页面刷新恢复、服务重启恢复、数据库、checkpoint、账户或多用户隔离。
- 文件上传、跨设备同步、流式输出、生产部署、TLS、鉴权或限流。
- 自动创建学习项目、执行测试、应用补丁或运行学习者命令。
