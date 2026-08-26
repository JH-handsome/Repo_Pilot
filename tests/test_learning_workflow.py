"""RepoPilot Learning Mode 结构化计划、证据边界和三节点图测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from coding_rag.agent.executor import AgentExecutor
from coding_rag.learning import (
    EvidenceRef,
    LearningPlan,
    LearningWorkflowConfig,
    ReproductionStep,
    build_learning_graph,
    build_learning_input,
    run_learning_workflow,
)


class SequenceLearningClient:
    """按顺序返回字符串或异常，并记录收到的聊天消息。"""

    def __init__(self, responses: list[str | Exception]):
        """保存离线响应并初始化调用记录。"""
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        """记录消息并返回下一条响应或抛出预设异常。"""
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StaticSearchProvider:
    """返回固定 Hybrid Search 形状的离线搜索 provider。"""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        retrieval_trace: dict[str, Any] | None = None,
    ):
        """保存固定搜索行并初始化查询记录。"""
        self.rows = rows
        self.calls: list[tuple[str, int]] = []
        self.retrieval_trace = retrieval_trace or {
            "summary": {"final_count": len(self.rows)}
        }

    def __call__(self, query: str, top_k: int) -> dict[str, Any]:
        """记录查询并返回带计数与子 trace 的搜索结果。"""
        self.calls.append((query, top_k))
        return {
            "results": self.rows[:top_k],
            "counts": {"seed": len(self.rows), "recalled": len(self.rows), "final": len(self.rows)},
            "retrieval_trace": self.retrieval_trace,
        }


class RecordingExecutor:
    """记录 Learning Graph 实际使用的工具名，并委托真实只读能力。"""

    def __init__(self, root: Path, provider: StaticSearchProvider):
        self.delegate = AgentExecutor(
            root,
            safe_mode=True,
            search_provider=provider,
        )
        self.safety = self.delegate.safety
        self.calls: list[str] = []

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append(name)
        return self.delegate.call(name, arguments)


def valid_plan_payload(
    *,
    path: str = "app.py",
    start_line: int = 1,
    end_line: int = 4,
    step_count: int = 4,
) -> dict[str, Any]:
    """生成字段完整、依赖按顺序排列的离线 LearningPlan。"""
    evidence = {
        "claim": f"{path} 定义并调用了 target 函数。",
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
    }
    step_templates = [
        {
            "title": "实现最小业务函数",
            "learning_goal": "理解纯函数的输入输出边界",
            "files_to_create": ["src/app.py"],
            "tasks": ["实现 target()，让它稳定返回字符串 'ok'。"],
            "why": "先复现最小业务行为，后续入口和测试才有稳定依赖。",
            "benefits": ["函数可单独运行，出错时定位范围只有一个文件。"],
            "verification": [
                "运行 python -c \"from src.app import target; assert target() == 'ok'\"。"
            ],
            "common_pitfalls": ["把调用 target() 写进函数体会造成递归。"],
            "optimization_question": "如果 target 后续需要参数，你会怎样保持接口清晰？",
        },
        {
            "title": "为业务函数建立回归测试",
            "learning_goal": "用断言固定 target 的可观察行为",
            "files_to_create": ["tests/test_app.py"],
            "tasks": ["编写 unittest，断言 target() 等于 'ok'。"],
            "why": "入口接线前先锁定核心行为，可以区分函数错误和调用错误。",
            "benefits": ["后续重构能够快速发现返回值回归。"],
            "verification": ["运行 python -m unittest tests.test_app。"],
            "common_pitfalls": ["只验证函数能运行，却没有断言返回值。"],
            "optimization_question": "除了正常返回值，你还会为 target 增加哪类边界测试？",
        },
        {
            "title": "连接程序入口",
            "learning_goal": "理解定义层和调用层的依赖方向",
            "files_to_create": ["src/main.py"],
            "tasks": ["从 src.app 导入 target，调用后把结果保存为 result。"],
            "why": "把调用放在独立入口中，可以让业务函数继续保持可复用。",
            "benefits": ["调用链清楚，测试可以分别覆盖函数和入口。"],
            "verification": [
                "运行 python -c \"from src.main import result; assert result == 'ok'\"。"
            ],
            "common_pitfalls": ["模块路径写错会导致 ModuleNotFoundError。"],
            "optimization_question": "如果入口增加 CLI 参数，你会把解析逻辑放在哪一层？",
        },
        {
            "title": "验证完整调用链",
            "learning_goal": "区分单元测试与入口集成测试",
            "files_to_create": ["tests/test_main.py"],
            "tasks": ["导入 src.main 并断言 result 为 'ok'，覆盖 target 的实际调用。"],
            "why": "函数测试通过不代表入口导入和调用一定正确。",
            "benefits": ["同时保护模块路径、函数调用和结果传递。"],
            "verification": ["运行 python -m unittest，确认两组测试全部通过。"],
            "common_pitfalls": ["只跑新测试，遗漏已有测试的回归。"],
            "optimization_question": "入口出现副作用后，你会怎样让集成测试仍保持确定性？",
        },
        {
            "title": "补充最小运行说明",
            "learning_goal": "把可运行条件转化为可复现文档",
            "files_to_create": ["README.md"],
            "tasks": ["记录目录结构、运行入口和两条测试命令。"],
            "why": "代码能运行不等于其他人能从空目录复现。",
            "benefits": ["减少环境与命令理解偏差。"],
            "verification": ["按 README 在新终端依次运行入口和完整测试。"],
            "common_pitfalls": ["文档命令依赖未说明的当前工作目录。"],
            "optimization_question": "你会怎样让 README 中的命令持续与代码保持一致？",
        },
        {
            "title": "声明项目元数据",
            "learning_goal": "理解 Python 项目配置与源码边界",
            "files_to_create": ["pyproject.toml"],
            "tasks": ["声明项目名、Python 版本和 src 包发现配置。"],
            "why": "显式配置可以避免运行环境依赖本机隐式路径。",
            "benefits": ["安装方式和包结构更稳定。"],
            "verification": ["在干净虚拟环境安装项目后再次运行 unittest。"],
            "common_pitfalls": ["配置包目录时忘记 src 布局。"],
            "optimization_question": "项目增加依赖后，你会如何区分运行依赖与开发依赖？",
        },
        {
            "title": "建立最小性能基线",
            "learning_goal": "学习用可重复测量判断优化是否有效",
            "files_to_create": ["benchmarks/benchmark_target.py"],
            "tasks": ["重复调用 target 并记录固定轮数的耗时。"],
            "why": "没有基线就无法判断后续优化是提升还是波动。",
            "benefits": ["性能讨论可以落到可比较数据。"],
            "verification": ["连续运行三次基线脚本并保存结果。"],
            "common_pitfalls": ["把单次极短调用的偶然波动当成结论。"],
            "optimization_question": "你会怎样设置预热和重复次数来降低测量噪声？",
        },
    ]
    steps: list[dict[str, Any]] = []
    for index in range(1, step_count + 1):
        step_id = f"step-{index}"
        template = step_templates[index - 1]
        steps.append(
            {
                "step_id": step_id,
                "depends_on": [] if index == 1 else [f"step-{index - 1}"],
                "evidence": [evidence],
                **template,
            }
        )
    return {
        "project_profile": {
            "project_name": "Demo Target",
            "summary": "一个定义并调用 Python 函数的最小项目。",
            "tech_stack": ["Python"],
            "prerequisites": ["Python 基础"],
            "entry_points": [evidence],
            "components": [
                {
                    "name": "入口",
                    "responsibility": "定义并调用 target 函数。",
                    "evidence": [evidence],
                }
            ],
        },
        "steps": steps,
    }


def search_row(
    *,
    path: str = "app.py",
    start_line: int = 1,
    end_line: int = 4,
) -> dict[str, Any]:
    """生成一个标准化搜索结果行。"""
    return {
        "rank": 1,
        "score": 1.0,
        "source": "hybrid",
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "text": "unused because read_file re-validates evidence",
    }


class LearningModelValidationTest(unittest.TestCase):
    """验证模型形状、相对路径和跨步骤依赖约束。"""

    def test_evidence_ref_normalizes_windows_path(self):
        """Windows 分隔符会转换为稳定的仓库相对 POSIX 路径。"""
        reference = EvidenceRef(
            claim="入口定义在这里。",
            path="src\\app.py",
            start_line=1,
            end_line=2,
        )

        self.assertEqual(reference.path, "src/app.py")

    def test_evidence_ref_rejects_unsafe_or_invalid_values(self):
        """绝对路径、穿越、错误行号、隐式类型和额外字段都会被拒绝。"""
        invalid_payloads = [
            {"path": "app.py", "start_line": 1, "end_line": 2},
            {"claim": "   ", "path": "app.py", "start_line": 1, "end_line": 2},
            {"claim": "x", "path": "../app.py", "start_line": 1, "end_line": 2},
            {"claim": "x", "path": "C:/repo/app.py", "start_line": 1, "end_line": 2},
            {"claim": "x", "path": "/repo/app.py", "start_line": 1, "end_line": 2},
            {"claim": "x", "path": "bad:name.py", "start_line": 1, "end_line": 2},
            {"claim": "x", "path": "app.py", "start_line": 0, "end_line": 2},
            {"claim": "x", "path": "app.py", "start_line": 3, "end_line": 2},
            {"claim": "x", "path": "app.py", "start_line": "1", "end_line": 2},
            {
                "claim": "x",
                "path": "app.py",
                "start_line": 1,
                "end_line": 2,
                "quote": "fake",
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                EvidenceRef.model_validate(payload)

    def test_plan_accepts_four_to_six_ordered_steps(self):
        """任务允许的四、五、六步路线都能通过严格校验。"""
        for step_count in (4, 5, 6):
            with self.subTest(step_count=step_count):
                plan = LearningPlan.model_validate(valid_plan_payload(step_count=step_count))
                self.assertEqual(len(plan.steps), step_count)

    def test_plan_rejects_step_count_and_dependency_errors(self):
        """拒绝三/七步、重复标识以及未知、自身或未来依赖。"""
        invalid_payloads = [
            valid_plan_payload(step_count=3),
            valid_plan_payload(step_count=7),
        ]
        duplicate = valid_plan_payload()
        duplicate["steps"][1]["step_id"] = "step-1"
        invalid_payloads.append(duplicate)
        future = valid_plan_payload()
        future["steps"][0]["depends_on"] = ["step-2"]
        invalid_payloads.append(future)
        self_dependency = valid_plan_payload()
        self_dependency["steps"][0]["depends_on"] = ["step-1"]
        invalid_payloads.append(self_dependency)
        unknown = valid_plan_payload()
        unknown["steps"][1]["depends_on"] = ["missing-step"]
        invalid_payloads.append(unknown)
        repeated_dependency = valid_plan_payload()
        repeated_dependency["steps"][1]["depends_on"] = ["step-1", "step-1"]
        invalid_payloads.append(repeated_dependency)
        invalid_dependency_id = valid_plan_payload()
        invalid_dependency_id["steps"][1]["depends_on"] = ["Step 1"]
        invalid_payloads.append(invalid_dependency_id)

        for payload in invalid_payloads:
            with self.subTest(steps=len(payload["steps"])), self.assertRaises(ValidationError):
                LearningPlan.model_validate(payload)

    def test_step_requires_all_teaching_fields(self):
        """每个教学字段都必须显式出现，不能依赖模型默认值。"""
        step_payload = valid_plan_payload()["steps"][0]
        required_fields = (
            "learning_goal",
            "files_to_create",
            "tasks",
            "why",
            "benefits",
            "verification",
            "evidence",
            "common_pitfalls",
            "optimization_question",
        )
        for field in required_fields:
            invalid = dict(step_payload)
            invalid.pop(field)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ReproductionStep.model_validate(invalid)

    def test_teaching_fields_reject_empty_or_blank_content(self):
        """非空列表不能用空列表或空白字符串伪装成具体教学内容。"""
        for field in (
            "files_to_create",
            "tasks",
            "benefits",
            "verification",
            "evidence",
            "common_pitfalls",
        ):
            invalid = valid_plan_payload()["steps"][0]
            invalid[field] = []
            with self.subTest(field=field, value=[]), self.assertRaises(ValidationError):
                ReproductionStep.model_validate(invalid)

        for field in ("tasks", "benefits", "verification", "common_pitfalls"):
            invalid = valid_plan_payload()["steps"][0]
            invalid[field] = ["   "]
            with self.subTest(field=field, value="blank"), self.assertRaises(ValidationError):
                ReproductionStep.model_validate(invalid)

        for field in ("title", "learning_goal", "why"):
            invalid = valid_plan_payload()["steps"][0]
            invalid[field] = "   "
            with self.subTest(field=field, value="blank"), self.assertRaises(ValidationError):
                ReproductionStep.model_validate(invalid)

        invalid_question = valid_plan_payload()["steps"][0]
        invalid_question["optimization_question"] = "？"
        with self.assertRaises(ValidationError):
            ReproductionStep.model_validate(invalid_question)

    def test_profile_requires_real_entry_component_and_list_content(self):
        """项目画像也拒绝空技术栈、空入口和没有证据的组件。"""
        cases: list[tuple[str, dict[str, Any]]] = []
        for field in ("tech_stack", "prerequisites", "entry_points", "components"):
            payload = valid_plan_payload()
            payload["project_profile"][field] = []
            cases.append((field, payload))
        for field in ("tech_stack", "prerequisites"):
            payload = valid_plan_payload()
            payload["project_profile"][field] = ["   "]
            cases.append((f"{field}-blank", payload))
        component_evidence = valid_plan_payload()
        component_evidence["project_profile"]["components"][0]["evidence"] = []
        cases.append(("component-evidence", component_evidence))

        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ValidationError):
                LearningPlan.model_validate(payload)

    def test_step_normalizes_created_file_paths_and_rejects_non_question(self):
        """新项目文件也必须是相对路径，优化字段必须明确提问。"""
        step_payload = valid_plan_payload()["steps"][0]
        step_payload["files_to_create"] = ["src\\reader.py"]
        step = ReproductionStep.model_validate(step_payload)
        self.assertEqual(step.files_to_create, ["src/reader.py"])

        step_payload["optimization_question"] = "考虑性能优化"
        with self.assertRaises(ValidationError):
            ReproductionStep.model_validate(step_payload)


class LearningWorkflowTest(unittest.TestCase):
    """验证正常路线、失败收敛、证据校验和 JSON 状态。"""

    def setUp(self):
        """建立一个四行、可被安全读取的最小 Python 仓库。"""
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "app.py").write_text(
            'def target():\n    return "ok"\n\nresult = target()\n',
            encoding="utf-8",
        )
        self.config = LearningWorkflowConfig(repo_path=self.root, top_k=4)

    def tearDown(self):
        """清理本测试创建的临时仓库。"""
        self.temp_dir.cleanup()

    def test_success_returns_profile_route_first_step_and_trace(self):
        """正常请求经过三个节点，并返回四步路线和第一步。"""
        provider = StaticSearchProvider([search_row()])
        client = SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)])

        result = run_learning_workflow(
            "从零复现最小 Python 函数调用项目",
            self.config,
            client,
            search_provider=provider,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.graph_steps, ["collect_evidence", "analyze_and_plan", "present_step"])
        self.assertEqual(len(result.steps), 4)
        self.assertEqual(result.first_step, result.steps[0])
        self.assertEqual([event["step"] for event in result.trace["events"]], result.graph_steps)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(provider.calls), 1)
        user_message = next(message for message in client.calls[0] if message["role"] == "user")
        self.assertIn("从零复现最小 Python 函数调用项目", user_message["content"])
        self.assertEqual(result.first_step.title, "实现最小业务函数")
        self.assertIn("target()", result.first_step.tasks[0])
        self.assertIn('"path": "app.py"', user_message["content"])
        self.assertIn("project_profile", user_message["content"])
        flags = result.trace["run"]["flags"]
        self.assertTrue(flags["safe_mode"])
        self.assertTrue(flags["readonly"])
        self.assertFalse(flags["execution_requested"])
        self.assertFalse(flags["execution_enabled"])

    def test_default_provider_reuses_real_hybrid_search(self):
        """未注入 provider 时复用 RepoPilot 的完整 Hybrid Search 流水线。"""
        client = SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)])

        result = run_learning_workflow(
            "复现 target 入口",
            self.config,
            client,
        )

        self.assertEqual(result.status, "success")
        retrieval = result.trace["artifacts"]["retrieval"]
        self.assertEqual(retrieval["summary"]["final_count"], 1)
        self.assertEqual(result.first_step.evidence[0].path, "app.py")

    def test_result_and_graph_state_are_json_serializable(self):
        """模型、client 和 executor 留在闭包中，不会污染可序列化状态。"""
        provider = StaticSearchProvider([search_row()])
        client = SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)])
        graph = build_learning_graph(self.config, client, search_provider=provider)

        state = graph.invoke(build_learning_input("复现检索器"), {"recursion_limit": 10})
        result = run_learning_workflow(
            "复现检索器",
            self.config,
            SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)]),
            search_provider=provider,
        )

        json.dumps(state, ensure_ascii=False, allow_nan=False)
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
        self.assertNotIn("client", state)
        self.assertNotIn("executor", state)

    def test_empty_python_repository_stops_before_model(self):
        """没有 Python 文件时停在证据节点，且不会浪费模型调用。"""
        empty_root = self.root / "empty"
        empty_root.mkdir()
        (empty_root / "README.md").write_text("empty", encoding="utf-8")
        client = SequenceLearningClient([json.dumps(valid_plan_payload())])
        provider = StaticSearchProvider([search_row()])

        result = run_learning_workflow(
            "复现项目",
            LearningWorkflowConfig(repo_path=empty_root),
            client,
            search_provider=provider,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "empty_repository")
        self.assertEqual(result.graph_steps, ["collect_evidence"])
        self.assertEqual(client.calls, [])
        self.assertEqual(provider.calls, [])

    def test_model_exception_returns_failed_without_plan(self):
        """模型异常被收敛为 model_error，未校验计划字段保持为空。"""
        client = SequenceLearningClient([RuntimeError("secret provider detail")])

        result = run_learning_workflow(
            "复现项目",
            self.config,
            client,
            search_provider=StaticSearchProvider([search_row()]),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "model_error")
        self.assertIsNone(result.project_profile)
        self.assertEqual(result.steps, [])
        self.assertIsNone(result.first_step)
        serialized = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        self.assertNotIn("secret provider detail", serialized)

    def test_bad_json_and_missing_fields_are_rejected(self):
        """坏 JSON 或缺少教学字段都返回 invalid_model_output。"""
        missing_field = valid_plan_payload()
        missing_field["steps"][0].pop("why")
        responses = ["{not-json", json.dumps(missing_field, ensure_ascii=False)]

        for response in responses:
            with self.subTest(response=response[:20]):
                result = run_learning_workflow(
                    "复现项目",
                    self.config,
                    SequenceLearningClient([response]),
                    search_provider=StaticSearchProvider([search_row()]),
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error.code, "invalid_model_output")
                self.assertEqual(result.graph_steps, ["collect_evidence", "analyze_and_plan"])
                self.assertEqual(result.steps, [])

    def test_unknown_file_and_unobserved_range_are_rejected(self):
        """结构正确但引用虚构文件或未观察行范围的计划不能展示。"""
        payloads = [
            valid_plan_payload(path="missing.py"),
            valid_plan_payload(start_line=1, end_line=4),
        ]
        providers = [
            StaticSearchProvider([search_row()]),
            StaticSearchProvider([search_row(start_line=1, end_line=2)]),
        ]

        for payload, provider in zip(payloads, providers):
            with self.subTest(path=payload["project_profile"]["entry_points"][0]["path"]):
                result = run_learning_workflow(
                    "复现项目",
                    self.config,
                    SequenceLearningClient([json.dumps(payload, ensure_ascii=False)]),
                    search_provider=provider,
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error.code, "invalid_evidence")
                self.assertEqual(result.steps, [])

    def test_model_evidence_beyond_real_file_is_rejected(self):
        """检索范围合法时，模型自行扩大到真实总行数之外仍会安全失败。"""
        payload = valid_plan_payload(end_line=99)

        result = run_learning_workflow(
            "复现项目",
            self.config,
            SequenceLearningClient([json.dumps(payload, ensure_ascii=False)]),
            search_provider=StaticSearchProvider([search_row()]),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "invalid_evidence")
        self.assertEqual(result.graph_steps, ["collect_evidence", "analyze_and_plan"])
        self.assertIsNone(result.project_profile)
        self.assertEqual(result.steps, [])
        self.assertIsNone(result.first_step)

    def test_learning_graph_calls_only_readonly_tools(self):
        """即使底层执行器支持更多工具，Learning Graph 也只触达搜索和读取。"""
        provider = StaticSearchProvider([search_row()])
        executor = RecordingExecutor(self.root, provider)
        client = SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)])

        with patch(
            "coding_rag.learning.workflow.build_learning_executor",
            return_value=executor,
        ):
            result = run_learning_workflow("复现项目", self.config, client)

        self.assertEqual(result.status, "success")
        self.assertEqual(executor.calls, ["search_code", "read_file"])
        self.assertTrue(executor.safety.policy.safe_mode)
        self.assertEqual(
            set(executor.calls),
            {"search_code", "read_file"},
        )

    def test_non_finite_provider_data_cannot_pollute_json_state(self):
        """NaN 分数和非法子 trace 会被丢弃，最终状态保持严格 JSON。"""
        row = search_row()
        row["score"] = float("nan")
        provider = StaticSearchProvider(
            [row],
            retrieval_trace={"summary": {"score": float("nan")}},
        )

        graph = build_learning_graph(
            self.config,
            SequenceLearningClient([json.dumps(valid_plan_payload(), ensure_ascii=False)]),
            search_provider=provider,
        )
        state = graph.invoke(build_learning_input("复现项目"), {"recursion_limit": 10})

        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["evidence"][0]["score"])
        self.assertIsNone(state["retrieval_trace"])
        json.dumps(state, ensure_ascii=False, allow_nan=False)

    def test_invalid_repository_policy_becomes_stable_failure(self):
        """图节点内延迟构造执行器，使损坏 policy 不再从公开入口裸抛异常。"""
        policy_dir = self.root / ".repopilot"
        policy_dir.mkdir()
        (policy_dir / "policy.json").write_text("{bad-json", encoding="utf-8")
        client = SequenceLearningClient([json.dumps(valid_plan_payload())])

        result = run_learning_workflow("复现项目", self.config, client)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "repository_error")
        self.assertEqual(result.graph_steps, ["collect_evidence"])
        self.assertEqual(client.calls, [])
        self.assertTrue(result.trace["run"]["flags"]["readonly"])

    def test_search_result_outside_real_file_is_rejected_before_model(self):
        """read_file 的静默截断会被新证据层识别为仓库错误。"""
        client = SequenceLearningClient([json.dumps(valid_plan_payload())])

        result = run_learning_workflow(
            "复现项目",
            self.config,
            client,
            search_provider=StaticSearchProvider([search_row(end_line=99)]),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "repository_error")
        self.assertEqual(result.graph_steps, ["collect_evidence"])
        self.assertEqual(client.calls, [])

    def test_invalid_plan_dependency_is_not_presented(self):
        """模型给出未来依赖时在 Pydantic 解析阶段终止。"""
        payload = valid_plan_payload()
        payload["steps"][0]["depends_on"] = ["step-2"]

        result = run_learning_workflow(
            "复现项目",
            self.config,
            SequenceLearningClient([json.dumps(payload, ensure_ascii=False)]),
            search_provider=StaticSearchProvider([search_row()]),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.code, "invalid_model_output")
        self.assertNotIn("present_step", result.graph_steps)

    def test_input_and_config_validation_fail_before_running_graph(self):
        """空目标、不支持的级别和危险切片参数会被同步拒绝。"""
        with self.assertRaises(ValueError):
            build_learning_input("   ")
        with self.assertRaises(ValueError):
            build_learning_input("goal", "advanced")
        with self.assertRaises(ValueError):
            LearningWorkflowConfig(repo_path=self.root, overlap=40, chunk_size=40)
        with self.assertRaises(ValueError):
            LearningWorkflowConfig(repo_path=self.root, max_context_chars=100)


if __name__ == "__main__":
    unittest.main()
