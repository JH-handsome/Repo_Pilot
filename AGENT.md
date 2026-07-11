# RepoPilot Agent Rules

RepoPilot Agent 只负责理解任务、检索上下文、生成计划、记录运行环节和给出验证建议。

## 分工

- RepoPilot Agent 不直接修改代码。
- 代码修改交给 Claude Code 执行。
- Agent 输出的实现草案必须能作为 Claude Code 的执行输入。

## 运行环节

1. `receive_task`: 接收任务并识别目标、约束和任务类型。
2. `load_memory`: 读取 `artifacts/agent_memory.jsonl` 中的相关历史记忆。
3. `retrieve_context`: 使用 Hybrid Search、邻近召回和最终过滤定位候选代码。
4. `plan_changes`: 生成候选文件、修改步骤、风险和验证点。
5. `implement`: 生成交给 Claude Code 的实现草案，不直接落盘修改。
6. `verify`: 给出应运行的测试、编译检查或评测命令。
7. `remember`: 写入长期记忆，并把完整运行日志追加到独立日志文件。

## 记录文件

- 长期记忆: `artifacts/agent_memory.jsonl`
- 运行日志: `artifacts/agent_run_log.jsonl`
- 单次详细 trace 可用 `--trace-out` 指定路径保存。

## 基本规则

- 只处理和当前任务相关的文件。
- 不做无关重构。
- 删除文件必须一个个删；批量删除前必须先确认。
- 修改主要功能后，更新相关测试；需要记录变更时写入 `CHANGELOG.md`。
- `.env` 只用于本地密钥配置，不提交真实 API key。
- 调用 Claude Code 必须优先使用 `scripts/run_claude_guarded.py`；禁止裸跑长任务。
- 单次 Claude 任务默认 token 上限为 80000，cache miss token 上限为 20000，cache miss 比例超过 35% 时中止。
