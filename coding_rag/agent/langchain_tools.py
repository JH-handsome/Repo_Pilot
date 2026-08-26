"""把 RepoPilot 绑定仓库的工具适配为 LangChain 工具。

本模块只负责模型可见的工具名称和 Pydantic 参数校验。
仓库访问与路径安全仍委托给 RepoPilot 的 ``AgentExecutor.call`` 入口。
只读运行时继续使用 ``build_readonly_langchain_tools``；普通 Agent 使用
完整工具集合，并由执行器保留逐调用授权、快照和安全策略边界。
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, model_validator


READONLY_LANGCHAIN_TOOL_NAMES = ("search_code", "read_file")
AGENT_LANGCHAIN_TOOL_NAMES = (
    "search_code",
    "read_file",
    "apply_patch",
    "run_command",
    "inspect_diff",
)


class _ToolExecutor(Protocol):
    """描述 RepoPilot 窄执行入口的结构化类型。"""

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """分派一次已经完成参数校验的仓库工具调用。"""
        ...


class ReadOnlyToolInput(BaseModel):
    """只读工具参数基类，拒绝模型提供的未声明参数。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class SearchCodeInput(ReadOnlyToolInput):
    """仓库代码搜索工具的参数约束。"""

    query: str = Field(min_length=1, description="Repository code search query.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of ranked results.",
    )


class ReadFileInput(ReadOnlyToolInput):
    """仓库文件读取工具的参数约束。"""

    path: str = Field(min_length=1, description="Repository-relative file path.")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional first line, inclusive.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional last line, inclusive.",
    )

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReadFileInput":
        """在访问仓库前拒绝起止行倒置的显式范围。"""
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ApplyPatchInput(ReadOnlyToolInput):
    """代码补丁工具的参数约束。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)

    diff: str = Field(min_length=1, description="Unified diff to apply inside the repository.")


class RunCommandInput(ReadOnlyToolInput):
    """仓库命令工具的参数约束。"""

    cmd: str = Field(min_length=1, description="Command allowed by the repository safety policy.")
    affected_files: list[str] = Field(
        min_length=1,
        description="Repository-relative files that the command may modify.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        le=300,
        description="Optional command timeout in seconds.",
    )


class InspectDiffInput(ReadOnlyToolInput):
    """查看当前仓库 diff；该工具不接受参数。"""


def build_readonly_langchain_tool_map(
    executor: _ToolExecutor,
) -> dict[str, StructuredTool]:
    """构建只读运行时允许使用的两个 LangChain 工具。"""
    if executor is None:
        raise ValueError("executor is required")

    def search_code(query: str, top_k: int = 5) -> Any:
        """通过 RepoPilot 执行入口检索并排序仓库代码。"""
        return executor.call("search_code", {"query": query, "top_k": top_k})

    def read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> Any:
        """通过 RepoPilot 执行入口读取仓库内文件。"""
        return executor.call(
            "read_file",
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    tools = [
        StructuredTool.from_function(
            func=search_code,
            name="search_code",
            description=(
                "Search the current repository for relevant code. "
                "Returns ranked file paths, line ranges, scores, and code text."
            ),
            args_schema=SearchCodeInput,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "Read one file inside the current repository, optionally limited "
                "to an inclusive line range."
            ),
            args_schema=ReadFileInput,
            infer_schema=False,
        ),
    ]
    return {tool.name: tool for tool in tools}


def build_readonly_langchain_tools(executor: _ToolExecutor) -> list[StructuredTool]:
    """按稳定顺序返回可绑定到模型的只读工具。"""
    tool_map = build_readonly_langchain_tool_map(executor)
    return [tool_map[name] for name in READONLY_LANGCHAIN_TOOL_NAMES]


def build_agent_langchain_tool_map(
    executor: _ToolExecutor,
) -> dict[str, StructuredTool]:
    """构建普通 Agent 的完整工具集合，执行控制仍由 executor 负责。"""
    tool_map = build_readonly_langchain_tool_map(executor)

    def apply_patch(diff: str) -> Any:
        """在执行器通过安全检查和一次性授权后应用补丁。"""
        return executor.call("apply_patch", {"diff": diff})

    def run_command(
        cmd: str,
        affected_files: list[str],
        timeout_seconds: int | None = None,
    ) -> Any:
        """在执行器允许且安全策略通过时运行仓库命令。"""
        arguments: dict[str, Any] = {
            "cmd": cmd,
            "affected_files": affected_files,
        }
        if timeout_seconds is not None:
            arguments["timeout_seconds"] = timeout_seconds
        return executor.call("run_command", arguments)

    def inspect_diff() -> Any:
        """读取当前仓库变更，不修改任何文件。"""
        return executor.call("inspect_diff", {})

    write_tools = [
        StructuredTool.from_function(
            func=apply_patch,
            name="apply_patch",
            description=(
                "Apply a unified diff inside the repository after safety checks and "
                "an exact one-time approval for this call."
            ),
            args_schema=ApplyPatchInput,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=run_command,
            name="run_command",
            description=(
                "Run one repository command if it passes the command allowlist, all "
                "safety checks, and an exact one-time approval for this call."
            ),
            args_schema=RunCommandInput,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=inspect_diff,
            name="inspect_diff",
            description="Inspect the current repository diff without modifying files.",
            args_schema=InspectDiffInput,
            infer_schema=False,
        ),
    ]
    tool_map.update({tool.name: tool for tool in write_tools})
    return tool_map


def build_agent_langchain_tools(executor: _ToolExecutor) -> list[StructuredTool]:
    """按稳定顺序返回普通 Agent 的完整 LangChain 工具集合。"""
    tool_map = build_agent_langchain_tool_map(executor)
    return [tool_map[name] for name in AGENT_LANGCHAIN_TOOL_NAMES]
