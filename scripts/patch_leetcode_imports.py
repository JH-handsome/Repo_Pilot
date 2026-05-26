from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PRELUDE = "from leetcode_types import *\n"
NEEDS_PRELUDE = re.compile(
    r"\b("
    r"ListNode|TreeNode|Node|Optional|List|Dict|Set|Tuple|Deque|"
    r"defaultdict|deque|Counter|lru_cache|cache|heappush|heappop|heapify|inf"
    r")\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch copied LeetCode solutions so they can run locally."
    )
    parser.add_argument("repo_path", help="Path to the downloaded LeetCode repo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"Repo path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repo path is not a directory: {repo_path}")

    shim_source = Path(__file__).resolve().parents[1] / "leetcode_types.py"
    if not shim_source.exists():
        raise FileNotFoundError(f"Cannot find shim file: {shim_source}")

    patched_files = []
    shim_dirs = set()

    for path in sorted(repo_path.rglob("*.py")):
        if should_skip(path):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if "from leetcode_types import" in text:
            shim_dirs.add(path.parent)
            continue
        if not NEEDS_PRELUDE.search(text):
            continue

        lines = text.splitlines(keepends=True)
        insert_at = find_insert_index(lines)
        lines.insert(insert_at, PRELUDE)
        path.write_text("".join(lines), encoding="utf-8")

        patched_files.append(path)
        shim_dirs.add(path.parent)

    for directory in sorted(shim_dirs):
        shutil.copy2(shim_source, directory / "leetcode_types.py")

    print(f"Patched {len(patched_files)} Python files.")
    print(f"Copied leetcode_types.py into {len(shim_dirs)} directories.")


def should_skip(path: Path) -> bool:
    ignored_dirs = {".git", ".venv", "venv", "__pycache__"}
    if path.name == "leetcode_types.py":
        return True
    return any(part in ignored_dirs for part in path.parts)


def find_insert_index(lines: list[str]) -> int:
    index = 0

    if index < len(lines) and lines[index].startswith("#!"):
        index += 1
    if index < len(lines) and "coding" in lines[index]:
        index += 1

    index = skip_blank_and_comment_lines(lines, index)
    index = skip_module_docstring(lines, index)
    index = skip_blank_and_comment_lines(lines, index)

    while index < len(lines) and lines[index].startswith("from __future__ import "):
        index += 1

    return index


def skip_blank_and_comment_lines(lines: list[str], index: int) -> int:
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "" or stripped.startswith("#"):
            index += 1
            continue
        break
    return index


def skip_module_docstring(lines: list[str], index: int) -> int:
    if index >= len(lines):
        return index

    stripped = lines[index].lstrip()
    quote = None
    if stripped.startswith('"""'):
        quote = '"""'
    elif stripped.startswith("'''"):
        quote = "'''"

    if quote is None:
        return index

    if stripped.count(quote) >= 2:
        return index + 1

    index += 1
    while index < len(lines):
        if quote in lines[index]:
            return index + 1
        index += 1

    return index


if __name__ == "__main__":
    main()
