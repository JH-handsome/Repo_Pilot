"""
文件加载模块
负责从指定路径加载所有 Python 文件
"""

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path


IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
IGNORED_ROOT_DIRS = {"legacy"}
SENSITIVE_DIRS = {".aws", ".azure", ".gnupg", ".kube", ".ssh"}
SENSITIVE_FILES = {
    ".netrc", ".npmrc", ".pypirc", "id_ed25519", "id_rsa",
    "credentials.json", "credentials.py", "credentials.toml", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.py", "secrets.toml", "secrets.yaml", "secrets.yml",
    "service-account.json", "service_account.json", "token.json", "tokens.json",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx", ".pkcs12"}


def is_env_name(name: str) -> bool:
    """Match local environment config files and directories."""
    return name == ".env" or name.startswith(".env.")


def is_sensitive_name(name: str) -> bool:
    """Conservatively exclude common credential files from model-visible source."""
    lowered = name.casefold()
    return (is_env_name(lowered) or lowered in SENSITIVE_DIRS
            or lowered in SENSITIVE_FILES or Path(lowered).suffix in SENSITIVE_SUFFIXES)


@dataclass(frozen=True)
class PythonFile:
    """表示一个 Python 文件的不可变数据类"""
    path: Path  # 文件路径
    text: str  # 文件内容


def load_python_files(
    repo_path: str | Path,
    exclude_roots: tuple[str, ...] = (),
) -> list[PythonFile]:
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
    python_files: list[PythonFile] = []
    for path in iter_python_paths(repo_path, exclude_roots=exclude_roots):
        # 以 UTF-8 编码读取文件内容，忽略编码错误
        text = path.read_text(encoding="utf-8", errors="ignore")
        python_files.append(PythonFile(path=path, text=text))

    return python_files


def iter_python_paths(
    repo_path: str | Path,
    exclude_roots: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield Python files deterministically while pruning ignored directory trees."""
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")
    if is_sensitive_name(root.name):
        return

    excluded_roots = IGNORED_ROOT_DIRS | set(exclude_roots)
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        excluded_here = IGNORED_DIRS | (excluded_roots if current == root else set())
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in excluded_here
            and not is_sensitive_name(name)
            and not (current / name).is_symlink()
        )
        for filename in sorted(filenames):
            if not filename.endswith(".py") or is_sensitive_name(filename):
                continue
            path = current / filename
            if not path.is_symlink() and path.is_file():
                yield path


def should_skip(
    path: Path,
    root: Path | None = None,
    exclude_roots: tuple[str, ...] = (),
) -> bool:
    """
    检查是否应该跳过某个路径
    
    跳过常见的生成目录和环境目录
    
    Args:
        path: 要检查的文件路径
        
    Returns:
        如果路径包含被忽略的目录，则返回 True
    """
    # 机密配置、忽略目录和符号链接都不应进入 Agent 的读取视野。
    if any(is_sensitive_name(part) for part in path.parts):
        return True
    if path.is_symlink() or any(part in IGNORED_DIRS for part in path.parts):
        return True

    if root is not None:
        try:
            relative_parts = path.resolve().relative_to(root.resolve()).parts
        except ValueError:
            return True
        excluded = IGNORED_ROOT_DIRS | set(exclude_roots)
        if relative_parts and relative_parts[0] in excluded:
            return True

    return False
