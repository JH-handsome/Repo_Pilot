"""LangChain 工具适配器与 LangGraph 只读运行时测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import json
import unittest

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from coding_rag.agent.executor import AgentExecutor
from coding_rag.agent.graph_runtime import (
    build_graph_input,
    build_readonly_graph,
    run_graph_query,
)
from coding_rag.agent.langchain_tools import (
    READONLY_LANGCHAIN_TOOL_NAMES,
    build_readonly_langchain_tool_map,
    build_readonly_langchain_tools,
)
from coding_rag.agent.runtime import UnifiedRunConfig, unified_run_to_dict
from coding_rag.tools.agent_readonly import AgentToolError


class RecordingExecutor:
    """记录适配器调用，并返回或抛出预先配置的结果。"""

    def __init__(self, result: Any = None, error: Exception | None = None):
        """配置假执行器的返回值或异常。"""
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """记录一次窄执行入口调用。"""
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


class SequenceToolCallingModel:
    """按顺序返回 AIMessage 或异常的离线工具调用模型。"""

    def __init__(self, responses: list[BaseMessage | Exception]):
        """保存待返回响应，并初始化绑定与调用记录。"""
        self.responses = list(responses)
        self.bound_tools: list[BaseTool] = []
        self.bind_kwargs: dict[str, Any] = {}
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "SequenceToolCallingModel":
        """记录模型可见工具和绑定选项。"""
        self.bound_tools = list(tools)
        self.bind_kwargs = dict(kwargs)
        return self

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """记录消息，并返回下一条响应或抛出配置的异常。"""
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool_request(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call-1",
) -> AIMessage:
    """构造一个提供商无关的 LangChain 工具调用消息。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": name,
                "args": arguments,
                "type": "tool_call",
            }
        ],
    )


class LangChainReadOnlyToolAdapterTest(unittest.TestCase):
    """验证参数 schema 与 RepoPilot 执行入口委托。"""

    def test_builds_exactly_two_readonly_structured_tools(self):
        """只按稳定顺序暴露 search_code 和 read_file。"""
        executor = RecordingExecutor()

        tools = build_readonly_langchain_tools(executor)

        self.assertEqual([tool.name for tool in tools], list(READONLY_LANGCHAIN_TOOL_NAMES))
        self.assertTrue(all(isinstance(tool, StructuredTool) for tool in tools))
        self.assertNotIn("apply_patch", {tool.name for tool in tools})
        self.assertNotIn("run_command", {tool.name for tool in tools})
        self.assertNotIn("inspect_diff", {tool.name for tool in tools})

    def test_search_code_schema_has_required_query_and_bounded_top_k(self):
        """验证模型可见的搜索参数契约。"""
        tool = build_readonly_langchain_tool_map(RecordingExecutor())["search_code"]
        schema = tool.args_schema.model_json_schema()

        self.assertEqual(schema["required"], ["query"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["top_k"]["default"], 5)
        self.assertEqual(schema["properties"]["top_k"]["minimum"], 1)
        self.assertEqual(schema["properties"]["top_k"]["maximum"], 20)

    def test_read_file_schema_has_repo_path_and_positive_line_numbers(self):
        """验证模型可见的文件读取参数契约。"""
        tool = build_readonly_langchain_tool_map(RecordingExecutor())["read_file"]
        schema = tool.args_schema.model_json_schema()
        start_line_schema = next(
            option
            for option in schema["properties"]["start_line"]["anyOf"]
            if option.get("type") == "integer"
        )
        end_line_schema = next(
            option
            for option in schema["properties"]["end_line"]["anyOf"]
            if option.get("type") == "integer"
        )

        self.assertEqual(schema["required"], ["path"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(start_line_schema["minimum"], 1)
        self.assertEqual(end_line_schema["minimum"], 1)

    def test_search_code_delegates_to_executor_call(self):
        """把校验后的搜索参数交给 RepoPilot 执行入口。"""
        expected = {"results": [], "counts": {"final": 0}}
        executor = RecordingExecutor(result=expected)
        tool = build_readonly_langchain_tool_map(executor)["search_code"]

        result = tool.invoke({"query": "runtime", "top_k": 3})

        self.assertEqual(result, expected)
        self.assertEqual(executor.calls, [("search_code", {"query": "runtime", "top_k": 3})])

    def test_search_code_uses_schema_default_top_k(self):
        """把 schema 声明的默认 top_k 传给执行入口。"""
        executor = RecordingExecutor(result={"results": []})
        tool = build_readonly_langchain_tool_map(executor)["search_code"]

        tool.invoke({"query": "runtime"})

        self.assertEqual(executor.calls, [("search_code", {"query": "runtime", "top_k": 5})])

    def test_read_file_delegates_to_executor_call(self):
        """把校验后的文件参数交给 RepoPilot 执行入口。"""
        expected = {"path": "app.py", "start_line": 2, "end_line": 4, "text": "2: value = 1"}
        executor = RecordingExecutor(result=expected)
        tool = build_readonly_langchain_tool_map(executor)["read_file"]

        result = tool.invoke({"path": "app.py", "start_line": 2, "end_line": 4})

        self.assertEqual(result, expected)
        self.assertEqual(
            executor.calls,
            [
                (
                    "read_file",
                    {"path": "app.py", "start_line": 2, "end_line": 4},
                )
            ],
        )

    def test_invalid_search_arguments_do_not_reach_executor(self):
        """拒绝空白、越界、类型错误和未声明的搜索参数。"""
        invalid_inputs = [
            {"query": "   "},
            {"query": "runtime", "top_k": 0},
            {"query": "runtime", "top_k": 21},
            {"query": "runtime", "top_k": "3"},
            {"query": "runtime", "top_k": 3.0},
            {"query": "runtime", "top_k": True},
            {"query": "runtime", "unexpected": True},
        ]
        for arguments in invalid_inputs:
            with self.subTest(arguments=arguments):
                executor = RecordingExecutor()
                tool = build_readonly_langchain_tool_map(executor)["search_code"]

                with self.assertRaises(ValidationError):
                    tool.invoke(arguments)

                self.assertEqual(executor.calls, [])

    def test_invalid_read_arguments_do_not_reach_executor(self):
        """拒绝无效路径、行范围、严格类型和额外字段。"""
        invalid_inputs = [
            {},
            {"path": "   "},
            {"path": "app.py", "start_line": 0},
            {"path": "app.py", "end_line": 0},
            {"path": "app.py", "start_line": 5, "end_line": 4},
            {"path": "app.py", "start_line": "1"},
            {"path": "app.py", "end_line": 2.0},
            {"path": "app.py", "start_line": True},
            {"path": "app.py", "unexpected": True},
        ]
        for arguments in invalid_inputs:
            with self.subTest(arguments=arguments):
                executor = RecordingExecutor()
                tool = build_readonly_langchain_tool_map(executor)["read_file"]

                with self.assertRaises(ValidationError):
                    tool.invoke(arguments)

                self.assertEqual(executor.calls, [])

    def test_repo_path_escape_is_still_rejected_by_agent_executor(self):
        """证明适配器没有绕过 RepoPilot 的仓库路径边界。"""
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "repo"
            root.mkdir()
            outside = base / "outside.py"
            outside.write_text("SECRET = True\n", encoding="utf-8")
            tool = build_readonly_langchain_tool_map(AgentExecutor(root))["read_file"]

            with self.assertRaises(AgentToolError):
                tool.invoke({"path": "../outside.py"})

    def test_executor_errors_remain_available_to_graph_error_handling(self):
        """保留执行器异常，供后续图节点转换为 ToolMessage。"""
        executor = RecordingExecutor(error=AgentToolError("read denied"))
        tool = build_readonly_langchain_tool_map(executor)["read_file"]

        with self.assertRaisesRegex(AgentToolError, "read denied"):
            tool.invoke({"path": "app.py"})

        self.assertEqual(
            executor.calls,
            [("read_file", {"path": "app.py", "start_line": None, "end_line": None})],
        )


class LangGraphReadOnlyRuntimeTest(unittest.TestCase):
    """验证只读图的状态、路由、错误收敛和统一结果结构。"""

    def test_graph_has_minimal_business_nodes_and_serializable_state(self):
        """图仅含四个业务节点，最终状态可以直接 JSON 序列化。"""
        config = UnifiedRunConfig(repo_path=".")
        model = SequenceToolCallingModel([AIMessage(content="直接回答。")])
        graph = build_readonly_graph(
            config,
            model,
            executor=RecordingExecutor(),
        )

        node_names = set(graph.get_graph().nodes)
        state = graph.invoke(build_graph_input("问题", config))

        self.assertTrue(
            {"model", "tool", "finalize", "safe_fallback"}.issubset(node_names)
        )
        self.assertEqual(state["graph_steps"], ["model", "finalize"])
        json.dumps(state, ensure_ascii=False)

    def test_direct_answer_uses_no_tool_and_keeps_public_result_shape(self):
        """直接回答路径不执行工具，并复用 UnifiedRun 的公开序列化结构。"""
        model = SequenceToolCallingModel([AIMessage(content="可以直接回答。")])

        run = run_graph_query(
            "普通问题",
            UnifiedRunConfig(repo_path="."),
            model,
            executor=RecordingExecutor(),
        )
        payload = unified_run_to_dict(run)

        self.assertEqual(run.status, "success")
        self.assertEqual(run.answer, "可以直接回答。")
        self.assertEqual(
            [tool.name for tool in model.bound_tools],
            ["search_code", "read_file"],
        )
        self.assertEqual(model.bind_kwargs, {"parallel_tool_calls": False})
        self.assertEqual(
            set(payload),
            {"query", "status", "answer", "execution", "summary", "trace"},
        )
        self.assertEqual(payload["summary"]["tool_call_count"], 0)
        self.assertFalse(payload["execution"]["requested"])
        self.assertFalse(payload["execution"]["enabled"])
        self.assertEqual(
            payload["trace"]["artifacts"]["agent"]["graph_steps"],
            ["model", "finalize"],
        )

    def test_successful_search_round_trip_preserves_trace_and_citation(self):
        """搜索结果经过 ToolMessage 返回模型，并保留检索 trace 与有效引用。"""
        retrieval_trace = {
            "query": "target",
            "events": [{"step": "retrieve", "status": "success"}],
        }
        executor = RecordingExecutor(
            result={
                "results": [
                    {
                        "path": "app.py",
                        "start_line": 1,
                        "end_line": 2,
                        "text": "1: def target():\n2:     return True",
                    }
                ],
                "retrieval_trace": retrieval_trace,
            }
        )
        model = SequenceToolCallingModel(
            [
                tool_request("search_code", {"query": "target", "top_k": 1}),
                AIMessage(content="定义位于 app.py:1-2。"),
            ]
        )

        run = run_graph_query(
            "target 在哪里？",
            UnifiedRunConfig(repo_path=".", max_steps=2),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "success")
        self.assertEqual(
            executor.calls,
            [("search_code", {"query": "target", "top_k": 1})],
        )
        self.assertEqual(
            run.trace["artifacts"]["agent"]["graph_steps"],
            ["model", "tool", "model", "finalize"],
        )
        self.assertEqual(
            [event["step"] for event in run.trace["events"]],
            ["model_decision", "search_code", "model_decision", "final_answer"],
        )
        self.assertEqual(
            run.trace["artifacts"]["retrieval"]["searches"],
            [retrieval_trace],
        )
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertIn("不可信数据", str(model.calls[1][-1].content))
        self.assertNotIn("引用校验", run.answer)

    def test_tool_argument_validation_becomes_partial_tool_observation(self):
        """无效工具参数不会进入执行器，并以 ToolMessage 反馈给模型。"""
        executor = RecordingExecutor(result={"results": []})
        model = SequenceToolCallingModel(
            [
                tool_request("search_code", {"query": "target", "top_k": 0}),
                AIMessage(content="搜索参数无效，无法确认。"),
            ]
        )

        run = run_graph_query(
            "查找 target",
            UnifiedRunConfig(repo_path=".", max_steps=2),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "partial")
        self.assertEqual(executor.calls, [])
        self.assertTrue(run.observations[0]["error"]["recoverable"])
        self.assertEqual(run.trace["events"][1]["step"], "search_code")
        self.assertEqual(run.trace["events"][1]["status"], "failed")
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertEqual(model.calls[1][-1].status, "error")
        self.assertEqual(model.calls[1][-1].tool_call_id, "call-1")

    def test_path_escape_error_is_returned_without_leaving_repository(self):
        """真实执行入口拒绝仓库外路径，图把拒绝结果收敛为部分成功。"""
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            outside = Path(temp_dir) / "outside.py"
            outside.write_text("SECRET = True\n", encoding="utf-8")
            model = SequenceToolCallingModel(
                [
                    tool_request("read_file", {"path": "../outside.py"}),
                    AIMessage(content="该路径不在仓库内，无法读取。"),
                ]
            )

            run = run_graph_query(
                "读取仓库外文件",
                UnifiedRunConfig(repo_path=root, max_steps=2),
                model,
            )

        self.assertEqual(run.status, "partial")
        self.assertEqual(run.observations[0]["tool"], "read_file")
        self.assertEqual(run.trace["events"][1]["status"], "failed")
        self.assertNotIn("SECRET", run.answer)

    def test_unknown_tool_routes_to_failed_safe_fallback(self):
        """模型请求未授权工具时不调用执行器，并返回结构化失败。"""
        executor = RecordingExecutor()
        model = SequenceToolCallingModel(
            [tool_request("apply_patch", {"diff": "not allowed"})]
        )

        run = run_graph_query(
            "修改文件",
            UnifiedRunConfig(repo_path="."),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "failed")
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            run.trace["artifacts"]["agent"]["graph_steps"],
            ["model", "safe_fallback"],
        )
        self.assertEqual(run.trace["events"][-1]["step"], "unknown_tool")
        self.assertIn("未授权工具", run.answer)

    def test_multiple_tool_calls_are_rejected_before_execution(self):
        """单轮多个工具调用在路由层失败，不选择性执行其中任何一个。"""
        executor = RecordingExecutor()
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "search_code",
                    "args": {"query": "target"},
                    "type": "tool_call",
                },
                {
                    "id": "call-2",
                    "name": "read_file",
                    "args": {"path": "app.py"},
                    "type": "tool_call",
                },
            ],
        )

        run = run_graph_query(
            "同时搜索和读取",
            UnifiedRunConfig(repo_path="."),
            SequenceToolCallingModel([message]),
            executor=executor,
        )

        self.assertEqual(run.status, "failed")
        self.assertEqual(executor.calls, [])
        self.assertEqual(run.trace["events"][-1]["step"], "invalid_tool_calls")

    def test_tool_step_limit_ends_in_partial_state(self):
        """达到工具步数上限后再次请求工具会确定性结束为 partial。"""
        executor = RecordingExecutor(result={"results": []})
        model = SequenceToolCallingModel(
            [
                tool_request("search_code", {"query": "first"}),
                tool_request("search_code", {"query": "second"}, call_id="call-2"),
            ]
        )

        run = run_graph_query(
            "连续搜索",
            UnifiedRunConfig(repo_path=".", max_steps=1),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "partial")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(
            run.trace["artifacts"]["agent"]["graph_steps"],
            ["model", "tool", "model", "safe_fallback"],
        )
        self.assertEqual(run.trace["events"][-1]["step"], "tool_limit")

    def test_initial_model_failure_ends_in_failed_state(self):
        """首次模型调用异常时不执行工具，并返回 failed。"""
        executor = RecordingExecutor()

        run = run_graph_query(
            "问题",
            UnifiedRunConfig(repo_path="."),
            SequenceToolCallingModel([RuntimeError("model offline")]),
            executor=executor,
        )

        self.assertEqual(run.status, "failed")
        self.assertEqual(executor.calls, [])
        self.assertIn("model offline", run.answer)
        self.assertEqual(
            [event["step"] for event in run.trace["events"]],
            ["model_decision", "safe_fallback"],
        )

    def test_model_failure_after_tool_observation_ends_in_partial_state(self):
        """已有工具观察后模型异常时保留观察并返回 partial。"""
        executor = RecordingExecutor(result={"results": []})
        model = SequenceToolCallingModel(
            [
                tool_request("search_code", {"query": "target"}),
                RuntimeError("model offline"),
            ]
        )

        run = run_graph_query(
            "查找 target",
            UnifiedRunConfig(repo_path=".", max_steps=2),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "partial")
        self.assertEqual(len(run.observations), 1)
        self.assertEqual(run.trace["events"][-1]["step"], "safe_fallback")

    def test_empty_model_answer_is_a_structured_failure(self):
        """模型既不回答也不调用工具时，进入 empty_model_answer 错误态。"""
        run = run_graph_query(
            "问题",
            UnifiedRunConfig(repo_path="."),
            SequenceToolCallingModel([AIMessage(content="")]),
            executor=RecordingExecutor(),
        )

        self.assertEqual(run.status, "failed")
        self.assertIn("既没有给出回答", run.answer)
        self.assertEqual(run.trace["events"][0]["error"]["code"], "empty_model_answer")

    def test_large_tool_output_is_compacted_before_model_context(self):
        """超长工具输出被压缩，但原始可引用范围仍用于引用校验。"""
        large_text = "1: value = 1\n" + ("x" * 5000)
        executor = RecordingExecutor(
            result={
                "path": "large.py",
                "start_line": 1,
                "end_line": 1,
                "total_lines": 1,
                "text": large_text,
            }
        )
        model = SequenceToolCallingModel(
            [
                tool_request(
                    "read_file",
                    {"path": "large.py", "start_line": 1, "end_line": 1},
                ),
                AIMessage(content="值位于 large.py:1-1。"),
            ]
        )

        run = run_graph_query(
            "读取大文件",
            UnifiedRunConfig(repo_path=".", max_context_chars=300, max_steps=2),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "success")
        self.assertTrue(run.observations[0]["output_truncated"])
        self.assertLess(len(str(model.calls[1][-1].content)), len(large_text))
        self.assertNotIn("引用校验", run.answer)

    def test_non_json_model_metadata_and_tool_values_are_normalized(self):
        """模型元数据和工具结果含特殊对象时，GraphState 仍可序列化。"""
        executor = RecordingExecutor(
            result={
                "path": "app.py",
                "start_line": 1,
                "end_line": 1,
                "total_lines": 1,
                "text": "1: value = 1",
                "source": Path("app.py"),
            }
        )
        model = SequenceToolCallingModel(
            [
                tool_request("read_file", {"path": "app.py"}),
                AIMessage(
                    content="值位于 app.py:1-1。",
                    response_metadata={"provider_value": object()},
                ),
            ]
        )
        config = UnifiedRunConfig(repo_path=".", max_steps=2)
        graph = build_readonly_graph(config, model, executor=executor)

        state = graph.invoke(build_graph_input("读取值", config))

        serialized = json.dumps(state, ensure_ascii=False)
        self.assertIn("provider_value", serialized)
        self.assertEqual(state["observations"][0]["output"]["source"], "app.py")

    def test_missing_citation_adds_existing_validation_report(self):
        """回答遗漏已观察代码引用时，复用现有引用校验报告。"""
        executor = RecordingExecutor(
            result={
                "path": "app.py",
                "start_line": 1,
                "end_line": 2,
                "total_lines": 2,
                "text": "1: def target():\n2:     return True",
            }
        )
        model = SequenceToolCallingModel(
            [
                tool_request("read_file", {"path": "app.py"}),
                AIMessage(content="target 返回真值。"),
            ]
        )

        run = run_graph_query(
            "target 做什么？",
            UnifiedRunConfig(repo_path=".", max_steps=2),
            model,
            executor=executor,
        )

        self.assertEqual(run.status, "success")
        self.assertIn("引用校验", run.answer)
        self.assertTrue(
            run.trace["events"][-1]["output_summary"]["citation_issues"]
        )

    def test_run_input_validation_rejects_invalid_limits(self):
        """入口拒绝空问题和非正的工具步数、上下文长度。"""
        model = SequenceToolCallingModel([AIMessage(content="unused")])

        with self.assertRaisesRegex(ValueError, "query is required"):
            run_graph_query(
                "   ",
                UnifiedRunConfig(repo_path="."),
                model,
                executor=RecordingExecutor(),
            )
        with self.assertRaisesRegex(ValueError, "max_steps"):
            run_graph_query(
                "问题",
                UnifiedRunConfig(repo_path=".", max_steps=0),
                model,
                executor=RecordingExecutor(),
            )
        with self.assertRaisesRegex(ValueError, "max_context_chars"):
            run_graph_query(
                "问题",
                UnifiedRunConfig(repo_path=".", max_context_chars=0),
                model,
                executor=RecordingExecutor(),
            )


if __name__ == "__main__":
    unittest.main()
