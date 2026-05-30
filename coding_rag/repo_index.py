"""Lightweight AST index for repository-aware code retrieval."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from coding_rag.code_splitter import CodeChunk


CALL_INTENT_TOKENS = {"call", "caller", "callee"}
IMPORT_INTENT_TOKENS = {"import", "from"}
SIGNATURE_INTENT_TOKENS = {"signature", "function", "method", "def", "class"}
TREE_INTENT_TOKENS = {"filetree", "tree", "path", "module"}


@dataclass(frozen=True)
class SymbolRecord:
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    parent: str | None = None


@dataclass(frozen=True)
class FileIndex:
    path: Path
    module: str
    directories: tuple[str, ...]
    symbols: tuple[SymbolRecord, ...] = field(default_factory=tuple)
    imports: tuple[str, ...] = field(default_factory=tuple)
    calls: tuple[str, ...] = field(default_factory=tuple)

    def symbols_in_range(self, start_line: int, end_line: int) -> tuple[SymbolRecord, ...]:
        return tuple(
            symbol
            for symbol in self.symbols
            if ranges_overlap(start_line, end_line, symbol.start_line, symbol.end_line)
        )


@dataclass(frozen=True)
class RepoIndex:
    files: dict[str, FileIndex]

    def for_chunk(self, chunk: CodeChunk) -> FileIndex | None:
        return self.files.get(normalize_path(chunk.file_path))

    def metadata_for_chunk(self, chunk: CodeChunk) -> str:
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
    lines_by_number: dict[int, str] = {}
    for chunk in chunks:
        for offset, line in enumerate(chunk.text.splitlines(), start=chunk.start_line):
            lines_by_number.setdefault(offset, line)

    if not lines_by_number:
        return ""

    return "\n".join(lines_by_number.get(index, "") for index in range(1, max(lines_by_number) + 1))


def extract_symbols(tree: ast.AST) -> list[SymbolRecord]:
    symbols: list[SymbolRecord] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
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
            self._visit_function(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node, "async_function")

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
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
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = dotted_name(node.func)
            if call_name:
                calls.append(call_name)
                calls.append(call_name.rsplit(".", 1)[-1])
    return calls


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg.arg)
    if node.args.kwarg:
        args.append(node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def class_signature(node: ast.ClassDef) -> str:
    bases = [dotted_name(base) or getattr(base, "id", "") for base in node.bases]
    bases = [base for base in bases if base]
    if not bases:
        return node.name
    return f"{node.name}({', '.join(bases)})"


def symbol_metadata(symbol: SymbolRecord) -> str:
    parent = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
    return f"{symbol.kind} {symbol.name} {parent} signature {symbol.signature}"


def import_terms(name: str) -> list[str]:
    parts = [part for part in name.split(".") if part]
    return [name, *parts]


def module_name(path: Path) -> str:
    normalized = normalize_path(path)
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".")


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a <= end_b and start_b <= end_a


def normalize_path(path: object) -> str:
    return str(path).replace("\\", "/").casefold()


def unique_preserve_order(items: list[str]) -> list[str]:
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
    value_tokens: set[str] = set()
    for value in values:
        value_tokens.update(identifier_terms(value))
    return bool(query_tokens & value_tokens)


def identifier_terms(value: str) -> set[str]:
    normalized = value.replace("\\", "/").replace(".", "_").replace("(", "_").replace(")", "_")
    normalized = normalized.replace(",", "_").strip("_").casefold()
    if not normalized:
        return set()

    terms = {normalized}
    for part in normalized.split("_"):
        if part:
            terms.add(part)
    return terms
