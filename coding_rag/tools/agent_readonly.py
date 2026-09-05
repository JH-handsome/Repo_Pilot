"""Read-only tools for Agent planning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from coding_rag.repository.chunks import CodeChunk, split_python_files
from coding_rag.repository.files import IGNORED_DIRS, IGNORED_ROOT_DIRS, load_python_files, should_skip
from coding_rag.repository.index import RepoIndex, build_repo_index
from coding_rag.tools.bm25 import BM25Retriever


class AgentToolError(Exception):
    """Raised when a read-only Agent tool call is invalid or rejected."""


class ReadOnlyAgentTools:
    """Repo-bound read-only tools used before full Agent execution exists."""

    def __init__(self, repo_path: str | Path, *, chunk_size: int = 40, overlap: int = 5):
        """绑定仓库路径并初始化工具缓存。"""
        self.repo_path = Path(repo_path).resolve()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: list[CodeChunk] | None = None
        self._retriever: BM25Retriever | None = None
        self._repo_index: RepoIndex | None = None

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """调度支持的只读工具并拒绝未知工具。"""
        args = arguments or {}
        if name == "search_code":
            return self.search_code(
                query=str(args.get("query") or ""),
                top_k=int(args.get("top_k") or 5),
            )
        if name == "read_file":
            return self.read_file(
                path=str(args.get("path") or ""),
                start_line=optional_int(args.get("start_line")),
                end_line=optional_int(args.get("end_line")),
            )
        if name == "list_files":
            return self.list_files(
                pattern=str(args.get("pattern") or "**/*.py"),
                limit=int(args.get("limit") or 200),
            )
        if name == "inspect_symbol":
            return self.inspect_symbol(
                symbol=str(args.get("symbol") or ""),
                context_lines=int(args.get("context_lines") or 3),
            )
        if name in {"run_checks", "propose_patch"}:
            raise AgentToolError(f"{name} is not available in read-only Agent tools")
        raise AgentToolError(f"Unknown read-only Agent tool: {name}")

    def search_code(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """对仓库进行排序检索。"""
        if not query.strip() or top_k <= 0:
            return []
        results = self.retriever.search(query, top_k=top_k)
        rows: list[dict[str, Any]] = []
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            rows.append(
                {
                    "rank": rank,
                    "score": float(result.score),
                    "path": self.display_path(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text.rstrip(),
                }
            )
        return rows

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """验证仓库路径并返回所请求行范围及元数据。"""
        file_path = self.resolve_repo_path(path)
        if not file_path.exists():
            raise AgentToolError(f"File not found: {path}")
        if not file_path.is_file():
            raise AgentToolError(f"Path is not a file: {path}")

        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        if end < start:
            selected: list[str] = []
        else:
            selected = lines[start - 1 : end]
        numbered = [f"{line_no}: {line}" for line_no, line in enumerate(selected, start=start)]
        return {
            "path": self.display_path(file_path),
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "text": "\n".join(numbered),
        }

    def list_files(self, pattern: str = "**/*.py", limit: int = 200) -> list[str]:
        """列出仓库文件，排除忽略目录，遵守数量限制。"""
        if limit <= 0:
            return []
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise AgentToolError("list_files pattern must stay inside the repository")

        files: list[str] = []
        for path in sorted(self.repo_path.glob(pattern)):
            if not path.is_file() or should_skip(path, self.repo_path):
                continue
            files.append(self.display_path(path))
            if len(files) >= limit:
                break
        return files

    def inspect_symbol(self, symbol: str, context_lines: int = 3) -> list[dict[str, Any]]:
        """定位指定符号并返回其源代码上下文与元数据。"""
        needle = symbol.strip().casefold()
        if not needle:
            return []

        matches: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for file_index in self.repo_index.files.values():
            for record in file_index.symbols:
                if needle not in record.name.casefold() and needle not in record.signature.casefold():
                    continue
                key = (self.display_path(file_index.path), record.name, record.start_line)
                if key in seen:
                    continue
                seen.add(key)
                context = self.read_file(
                    self.display_path(file_index.path),
                    start_line=max(1, record.start_line - context_lines),
                    end_line=record.end_line + context_lines,
                )
                matches.append(
                    {
                        "name": record.name,
                        "kind": record.kind,
                        "parent": record.parent,
                        "signature": record.signature,
                        "path": self.display_path(file_index.path),
                        "start_line": record.start_line,
                        "end_line": record.end_line,
                        "context": context["text"],
                    }
                )
        return matches

    def resolve_repo_path(self, path: str) -> Path:
        """解析仓库内路径并拒绝越权访问。"""
        if not path:
            raise AgentToolError("path is required")
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.repo_path / candidate).resolve()
        try:
            resolved.relative_to(self.repo_path)
        except ValueError as exc:
            raise AgentToolError(f"Access denied outside repository: {path}") from exc
        return resolved

    def display_path(self, path: Path) -> str:
        """返回相对于仓库根目录的路径，无法相对时返回绝对路径。"""
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self.repo_path.resolve()).as_posix()
        except ValueError:
            return resolved_path.as_posix()

    @property
    def chunks(self) -> list[CodeChunk]:
        """懒加载并缓存仓库代码片段。"""
        if self._chunks is None:
            python_files = load_python_files(self.repo_path)
            self._chunks = split_python_files(
                python_files,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
        return self._chunks

    @property
    def retriever(self) -> BM25Retriever:
        """惰性构建并缓存基于仓库代码块的 BM25 检索器。"""
        if self._retriever is None:
            if not self.chunks:
                raise AgentToolError("No Python code chunks found in repository")
            self._retriever = BM25Retriever(self.chunks)
        return self._retriever

    @property
    def repo_index(self) -> RepoIndex:
        """惰性构建并缓存仓库符号索引。"""
        if self._repo_index is None:
            self._repo_index = build_repo_index(self.chunks)
        return self._repo_index


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
STOP_WORDS = {
    "add",
    "and",
    "bug",
    "class",
    "code",
    "def",
    "fix",
    "implement",
    "new",
    "refactor",
    "test",
    "the",
    "with",
}


def extract_task_identifiers(task: str, *, limit: int = 3) -> list[str]:
    """从任务中提取有意义的规范化标识符，同时过滤常见词。"""
    identifiers: list[str] = []
    seen: set[str] = set()
    for match in IDENTIFIER_RE.finditer(task):
        token = match.group(0)
        lowered = token.casefold()
        if lowered in STOP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        identifiers.append(token)
        if len(identifiers) >= limit:
            break
    return identifiers


def format_observations(observations: list[dict[str, Any]]) -> str:
    """将只读工具观察结果字典渲染为用于提示的文本。"""
    if not observations:
        return "(no observations)"
    lines: list[str] = []
    for observation in observations:
        tool = observation.get("tool", "unknown")
        tool_input = observation.get("input", {})
        output = observation.get("output")
        lines.append(f"- {tool}: input={tool_input}")
        if isinstance(output, list):
            lines.append(f"  count={len(output)}")
            for item in output[:5]:
                if isinstance(item, dict):
                    location = format_location(item)
                    label = item.get("name") or item.get("path") or item.get("rank")
                    lines.append(f"  - {label}{location}")
                else:
                    lines.append(f"  - {item}")
        elif isinstance(output, dict):
            lines.append(f"  {output.get('path', '')} lines={output.get('total_lines', '?')}")
        else:
            lines.append(f"  {output}")
    return "\n".join(lines)


def format_location(item: dict[str, Any]) -> str:
    """将路径和可选的起止行号格式化为源码位置字符串。"""
    path = item.get("path")
    start = item.get("start_line")
    end = item.get("end_line")
    if path and start and end:
        return f" ({path}:{start}-{end})"
    return ""


def optional_int(value: object) -> int | None:
    """将非空值转换为 int，并将 None/空字符串映射为 None。"""
    if value in (None, ""):
        return None
    return int(value)
