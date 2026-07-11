"""Generate a lightweight function map from Python source files."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOTS = ["main.py", "frontend.py", "web_ui.py", "coding_rag", "scripts"]


@dataclass(frozen=True)
class SymbolInfo:
    kind: str
    name: str
    lineno: int
    doc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RepoPilot function map")
    parser.add_argument("--root", action="append", dest="roots", help="File or directory to scan")
    parser.add_argument("--output", help="Write markdown to this path instead of stdout")
    return parser.parse_args()


def discover_python_files(paths: list[str], base: Path = Path(".")) -> list[Path]:
    base = base.resolve()
    files: list[Path] = []
    for value in paths:
        path = (base / value).resolve()
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    ignored_parts = {"__pycache__", ".git", ".venv"}
    return [path for path in sorted(files) if not (set(path.parts) & ignored_parts)]


def inspect_file(path: Path) -> list[SymbolInfo]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except SyntaxError:
        return []
    symbols: list[SymbolInfo] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(SymbolInfo("class", node.name, node.lineno, first_sentence(ast.get_docstring(node))))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        SymbolInfo(
                            "method",
                            f"{node.name}.{child.name}",
                            child.lineno,
                            first_sentence(ast.get_docstring(child)),
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(SymbolInfo("function", node.name, node.lineno, first_sentence(ast.get_docstring(node))))
    return symbols


def first_sentence(text: str | None) -> str:
    if not text:
        return ""
    normalized = " ".join(text.strip().split())
    for sep in ("。", ". "):
        if sep in normalized:
            sentence = normalized.split(sep, 1)[0].strip()
            return sentence + ("。" if sep == "。" else ".")
    return normalized


def generate_function_map(paths: list[str] | None = None, base: Path = Path(".")) -> str:
    base = base.resolve()
    roots = paths or DEFAULT_ROOTS
    lines = [
        "# RepoPilot Function Map",
        "",
        "本文件由 `scripts/generate_function_map.py` 基于 AST 生成，列出主要 Python 文件中的类和函数。",
        "",
    ]
    for path in discover_python_files(roots, base=base):
        symbols = inspect_file(path)
        if not symbols:
            continue
        rel = path.resolve().relative_to(base).as_posix()
        lines.extend([f"## `{rel}`", ""])
        for symbol in symbols:
            detail = f" - {symbol.doc}" if symbol.doc else ""
            lines.append(f"- `{symbol.kind}` `{symbol.name}` (line {symbol.lineno}){detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    text = generate_function_map(args.roots)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
