"""
文件加载模块
负责从指定路径加载所有 Python 文件
"""

from dataclasses import dataclass
from pathlib import Path


IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
IGNORED_ROOT_DIRS = {"legacy"}


@dataclass(frozen=True)
class PythonFile:
    """表示一个 Python 文件的不可变数据类"""
    path: Path  # 文件路径
    text: str  # 文件内容


def load_python_files(repo_path: str | Path) -> list[PythonFile]:
    """
    读取指定仓库路径下的所有 .py 文件
    
    Args:
        repo_path: 仓库路径，可以是字符串或 Path 对象
        
    Returns:
        PythonFile 对象列表
        
    Raises:
        FileNotFoundError: 路径不存在
        NotADirectoryError: 路径不是目录
    """
    root = Path(repo_path).resolve()  # 解析为绝对路径
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    python_files: list[PythonFile] = []
    # 递归查找所有 .py 文件并按路径排序
    for path in sorted(root.rglob("*.py")):
        if should_skip(path, root):
            continue

        # 以 UTF-8 编码读取文件内容，忽略编码错误
        text = path.read_text(encoding="utf-8", errors="ignore")
        python_files.append(PythonFile(path=path, text=text))

    return python_files


def should_skip(path: Path, root: Path | None = None) -> bool:
    """
    检查是否应该跳过某个路径
    
    跳过常见的生成目录和环境目录
    
    Args:
        path: 要检查的文件路径
        
    Returns:
        如果路径包含被忽略的目录，则返回 True
    """
    # 常见的需要忽略的目录名称
    if any(part in IGNORED_DIRS for part in path.parts):
        return True

    if root is not None:
        try:
            relative_parts = path.resolve().relative_to(root.resolve()).parts
        except ValueError:
            relative_parts = ()
        if relative_parts and relative_parts[0] in IGNORED_ROOT_DIRS:
            return True

    return False
