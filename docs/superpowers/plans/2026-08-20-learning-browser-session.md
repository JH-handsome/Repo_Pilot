# Learning Mode 浏览器接入实施计划

依据：`docs/superpowers/specs/2026-08-20-learning-browser-session-design.md`

## 任务 1：固定 HTTP 适配契约

- 在 `frontend.py` 中新增教学端点、`application/json` 限制、1 MiB 请求体上限和稳定的 415/413 错误。
- 给 `RepoPilotServer` 增加进程级随机签名密钥。
- 新增 `run_learning_frontend_request(payload, *, signing_key)`：
  - 校验外层配置和内层 `request`；
  - 构造现有 LLM client、`LearningWorkflowConfig` 与 `LearningSessionDependencies`；
  - 调用 `run_learning_session()`；
  - 返回严格 JSON 的 `LearningSessionResult`。
- 保持 `/api/run` 路径、响应和双重工具授权不变。

验证：前端适配器单元测试、HTTP start 测试、旧端点回归。

## 任务 2：实现学习页面状态机

- 在同一首页增加“代码问答 / 项目学习”模式选择。
- 页面内存保存当前 session，不使用浏览器存储。
- 根据 `next_action` 构造 `start`、`submit`、`reflect` 请求。
- 根据 phase/next_action 渲染项目画像、当前步骤、反馈、优化问题和完成状态。
- 所有不可信文本统一经过 `escapeHtml()`；增加单请求 busy 锁，阻止按钮和快捷键并发提交。
- 教学请求不包含 `execute_tools`。

验证：首页契约测试、HTML 安全断言、连续 action 测试。

## 任务 3：补齐安全与失败测试

- 覆盖信息不足、模型通过、反思推进、action/phase 错配和 session 篡改。
- 验证业务失败仍通过 HTTP 200 返回稳定结果，并保留可恢复的已验证 session。
- 验证超过 1 MiB 的请求在读取正文前以 413 拒绝。
- 验证签名密钥不进入 HTML、响应或 trace。
- 把命令/HTML 文本作为恶意输入，证明既不执行也不原样注入页面。

验证：`tests.test_frontend` 与 `tests.test_learning_frontend`。

## 任务 4：同步文档和任务书

- 更新 RepoPilot `README.md`、`FUNCTION_MAP.md`、`CHANGELOG.md`。
- 归档第31次任务；创建并最终归档第32次任务。
- 记录真实命令、测试数字、分支提交、回滚方式和残余风险。

验证：`git diff --check`、修改范围审计、文档与代码契约对照。

## 任务 5：完整回归与分支交付

按顺序运行：

1. `python -m unittest tests.test_frontend tests.test_learning_frontend -v`
2. `python -m unittest tests.test_learning_workflow tests.test_learning_session`
3. `python -m unittest tests.test_graph_runtime tests.test_unified_runtime tests.test_agent_tools`
4. `python -m unittest`
5. `python -m compileall coding_rag main.py frontend.py web_ui.py scripts tests`
6. `git diff --check` 与 `git status --short`

全部通过后提交功能分支，并按分支完成流程向用户提供合并、PR、保留或丢弃四个选项。
