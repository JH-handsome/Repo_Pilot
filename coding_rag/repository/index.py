"""Lightweight AST index for repository-aware code retrieval."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from coding_rag.repository.chunks import CodeChunk


CALL_INTENT_TOKENS = {"call", "caller", "callee"}
IMPORT_INTENT_TOKENS = {"import", "from"}
SIGNATURE_INTENT_TOKENS = {"signature", "function", "method", "def", "class"}
TREE_INTENT_TOKENS = {"filetree", "tree", "path", "module"}


@dataclass(frozen=True)
class SymbolRecord:
    """不可变符号记录，描述一个AST符号的基本信息。"""
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    parent: str | None = None


@dataclass(frozen=True)
class FileIndex:
    """不可变文件索引记录，存储模块路径、目录层级、符号、导入和调用信息。"""
    path: Path
    module: str
    directories: tuple[str, ...]
    symbols: tuple[SymbolRecord, ...] = field(default_factory=tuple)
    imports: tuple[str, ...] = field(default_factory=tuple)
    calls: tuple[str, ...] = field(default_factory=tuple)

    def symbols_in_range(self, start_line: int, end_line: int) -> tuple[SymbolRecord, ...]:
        """选择与指定行范围重叠的符号。"""
        return tuple(
            symbol
            for symbol in self.symbols
            if ranges_overlap(start_line, end_line, symbol.start_line, symbol.end_line)
        )


@dataclass(frozen=True)
class RepoIndex:
    """仓库级文件索引。"""

    files: dict[str, FileIndex]

    def for_chunk(self, chunk: CodeChunk) -> FileIndex | None:
        """获取代码块对应的文件索引。"""
        return self.files.get(normalize_path(chunk.file_path))

    def metadata_for_chunk(self, chunk: CodeChunk) -> str:
        """为代码块组合文件、符号、导入及调用元数据。"""
        file_index = self.for_chunk(chunk)
        if file_index is None:
            return ""

        symbols = file_index.symbols_in_range(chunk.start_line, chunk.end_line)
        symbol_docs = [
            symbol_metadata(symbol)
            for symbol in symbols
        ]
        if not symbol_docs:
            symbol_docs = [symbol.name for symbol in file_index.symbols[:8]]

        return " ".join(
            [
                "module",
                file_index.module,
                "filetree",
                " ".join(file_index.directories),
                "symbols",
                " ".join(symbol_docs),
                "imports",
                " ".join(file_index.imports),
                "calls",
                " ".join(file_index.calls),
            ]
        )

    def structural_score(self, query_tokens: set[str], chunk: CodeChunk) -> float:
        """根据代码块的结构化元数据与查询词元的匹配程度进行评分。"""
        file_index = self.for_chunk(chunk)
        if file_index is None:
            return 0.0

        score = 0.0
        symbols = file_index.symbols_in_range(chunk.start_line, chunk.end_line)
        if query_tokens & CALL_INTENT_TOKENS and terms_match(query_tokens, file_index.calls):
            score += 1.0
        if query_tokens & IMPORT_INTENT_TOKENS and terms_match(query_tokens, file_index.imports):
            score += 1.0
        if query_tokens & SIGNATURE_INTENT_TOKENS and terms_match(
            query_tokens,
            [symbol.signature for symbol in symbols] + [symbol.name for symbol in symbols],
        ):
            score += 1.0
        if query_tokens & TREE_INTENT_TOKENS and terms_match(
            query_tokens,
            [file_index.module, *file_index.directories],
        ):
            score += 0.5

        return min(score, 1.5) / 1.5


def build_repo_index(chunks: list[CodeChunk]) -> RepoIndex:
    """将代码块按文件分组并构建仓库索引。"""
    files: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        files.setdefault(normalize_path(chunk.file_path), []).append(chunk)

    return RepoIndex(
        files={
            file_key: build_file_index(file_chunks)
            for file_key, file_chunks in files.items()
        }
    )


def build_file_index(chunks: list[CodeChunk]) -> FileIndex:
    """从 chunks 重建源码并提取文件结构索引。"""
    first_chunk = chunks[0]
    source = rebuild_source(chunks)
    path = first_chunk.file_path
    module = module_name(path)
    directories = tuple(normalize_path(path).split("/")[:-1])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileIndex(path=path, module=module, directories=directories)

    return FileIndex(
        path=path,
        module=module,
        directories=directories,
        symbols=tuple(extract_symbols(tree)),
        imports=tuple(unique_preserve_order(extract_imports(tree))),
        calls=tuple(unique_preserve_order(extract_calls(tree))),
    )


def rebuild_source(chunks: list[CodeChunk]) -> str:
    """从有序代码块重建文件源文本。"""
    lines_by_number: dict[int, str] = {}
    for chunk in chunks:
        for offset, line in enumerate(chunk.text.splitlines(), start=chunk.start_line):
            lines_by_number.setdefault(offset, line)

    if not lines_by_number:
        return ""

    return "\n".join(lines_by_number.get(index, "") for index in range(1, max(lines_by_number) + 1))


def extract_symbols(tree: ast.AST) -> list[SymbolRecord]:
    """从 AST 树中提取所有符号记录（类、函数、异步函数）。"""
    symbols: list[SymbolRecord] = []

    class Visitor(ast.NodeVisitor):
        """遍历 AST 并收集符号记录的访问器。"""
        def __init__(self) -> None:
            """初始化父级名称栈为空列表。"""
            self.parents: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """处理类定义节点，记录类符号信息。"""
            parent = self.parents[-1] if self.parents else None
            symbols.append(
                SymbolRecord(
                    name=node.name,
                    kind="class",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=class_signature(node),
                    parent=parent,
                )
            )
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """处理普通函数定义节点。"""
            self._visit_function(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """访问异步函数定义节点，委托给通用函数处理方法。"""
            self._visit_function(node, "async_function")

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
            """访问函数定义节点，记录符号信息。"""
            parent = self.parents[-1] if self.parents else None
            symbols.append(
                SymbolRecord(
                    name=node.name,
                    kind=kind,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=function_signature(node),
                    parent=parent,
                )
            )
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

    Visitor().visit(tree)
    return symbols


def extract_imports(tree: ast.AST) -> list[str]:
    """遍历AST提取所有import语句中的模块名称，返回字符串列表。"""
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.extend(import_terms(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                imports.extend(import_terms(module))
            for alias in node.names:
                imports.extend(import_terms(alias.name))
                if module:
                    imports.append(f"{module}.{alias.name}")
    return imports


def extract_calls(tree: ast.AST) -> list[str]:
    """提取AST中所有函数调用的点分名称及简短名称列表。"""
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = dotted_name(node.func)
            if call_name:
                calls.append(call_name)
                calls.append(call_name.rsplit(".", 1)[-1])
    return calls


def dotted_name(node: ast.AST) -> str | None:
    """从 AST 节点中提取点分名称字符串，若为 Name 或 Attribute 链则返回，否则返回 None。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """从AST函数节点提取形如 ``name(arg1, arg2)`` 的签名字符串。"""
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg.arg)
    if node.args.kwarg:
        args.append(node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def class_signature(node: ast.ClassDef) -> str:
    """返回类的签名，包含类名和基类名称。"""
    bases = [dotted_name(base) or getattr(base, "id", "") for base in node.bases]
    bases = [base for base in bases if base]
    if not bases:
        return node.name
    return f"{node.name}({', '.join(bases)})"


def symbol_metadata(symbol: SymbolRecord) -> str:
    """将符号记录格式化为元数据字符串。"""
    parent = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
    return f"{symbol.kind} {symbol.name} {parent} signature {symbol.signature}"


def import_terms(name: str) -> list[str]:
    """将点分隔的名称拆分为完整名称及其各组成部分的列表。"""
    parts = [part for part in name.split(".") if part]
    return [name, *parts]


def module_name(path: Path) -> str:
    """将路径转换为 Python 模块名（去除 .py 后缀并将路径分隔符替换为点号）。"""
    normalized = normalize_path(path)
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".")


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """判断两个整数区间是否有交集。"""
    return start_a <= end_b and start_b <= end_a


def normalize_path(path: object) -> str:
    """将路径统一转为小写正斜杠分隔的字符串形式。"""
    return str(path).replace("\\", "/").casefold()


def unique_preserve_order(items: list[str]) -> list[str]:
    """去重并保持原始顺序，忽略空白项及空字符串。"""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def terms_match(query_tokens: set[str], values: tuple[str, ...] | list[str]) -> bool:
    """检查查询词元与值列表中任一标识符词元是否存在交集。"""
    value_tokens: set[str] = set()
    for value in values:
        value_tokens.update(identifier_terms(value))
    return bool(query_tokens & value_tokens)


def identifier_terms(value: str) -> set[str]:
    """将标识符字符串归一化并拆分为词条集合。"""
    normalized = value.replace("\\", "/").replace(".", "_").replace("(", "_").replace(")", "_")
    normalized = normalized.replace(",", "_").strip("_").casefold()
    if not normalized:
        return set()

    terms = {normalized}
    for part in normalized.split("_"):
        if part:
            terms.add(part)
    return terms
