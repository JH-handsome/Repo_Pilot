"""早期文件加载原型。

当前主流程已经迁移到 `coding_rag.file_loader`。这个文件仅保留作学习和对照，
不要再作为新功能入口。
"""

from pathlib import Path


IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
}


def should_ignore_path(path: Path) -> bool:
    """
    判断一个路径是否应该被忽略。

    例如：
    - .git 目录不需要读
    - .venv 目录不需要读
    - __pycache__ 不需要读
    """
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    return False


def load_python_files(repo_path: str) -> list[dict]:
    """
    读取一个代码仓库中的所有 Python 文件。

    参数：
        repo_path: 仓库路径，例如 "./examples/sample_repo"

    返回：
        [
            {
                "path": "rag/loader.py",
                "absolute_path": "/Users/xxx/repo/rag/loader.py",
                "content": "...文件内容...",
                "line_count": 120
            }
        ]
    """
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"不是文件夹: {root}")

    python_files = []

    for file_path in root.rglob("*.py"):
        if should_ignore_path(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk", errors="ignore")

        relative_path = file_path.relative_to(root)

        python_files.append(
            {
                "path": str(relative_path),
                "absolute_path": str(file_path),
                "content": content,
                "line_count": len(content.splitlines()),
            }
        )

    return python_files
