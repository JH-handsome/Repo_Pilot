"""
文件加载模块
负责遍历仓库并读取任意语言的源码文件；保留 Python 专用兼容入口。
"""

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path


IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "vendor", "dist", "build", "target", ".next", ".nuxt",
    ".gradle", ".idea", ".engramory-memory",
}
IGNORED_ROOT_DIRS = {"legacy"}
SENSITIVE_DIRS = {".aws", ".azure", ".gnupg", ".kube", ".ssh"}
SENSITIVE_FILES = {
    ".netrc", ".npmrc", ".pypirc", "id_ed25519", "id_rsa",
    "credentials.json", "credentials.py", "credentials.toml", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.py", "secrets.toml", "secrets.yaml", "secrets.yml",
    "service-account.json", "service_account.json", "token.json", "tokens.json",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx", ".pkcs12"}

# Unknown extensions remain eligible: a new language needs no allowlist entry.
NON_SOURCE_SUFFIXES = {
    ".md", ".rst", ".txt", ".log", ".csv", ".tsv", ".jsonl", ".lock",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".jar", ".war", ".exe", ".dll",
    ".so", ".dylib", ".o", ".a", ".class", ".pyc", ".pyo", ".wasm",
    ".sqlite", ".sqlite3", ".db", ".map", ".mp3", ".mp4", ".woff", ".woff2",
}
SOURCE_FILENAMES = {"cmakelists.txt", "requirements.txt"}
MAX_SOURCE_BYTES = 2 * 1024 * 1024
LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".kt": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".scala": "scala", ".r": "r",
    ".sh": "bash", ".bash": "bash", ".ps1": "powershell", ".sql": "sql",
    ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue",
    ".svelte": "svelte", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".lua": "lua", ".dart": "dart",
}


def source_language(path: str | Path) -> str:
    """Return a Markdown language label, using plain text for unknown languages."""
    return LANGUAGE_BY_SUFFIX.get(Path(path).suffix.casefold(), "text")


def read_source_text(path: Path) -> str | None:
    """读取 UTF-8／带 BOM 的 UTF-16 源码文件，拒绝二进制、超限与编码错误。"""
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            return None
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = raw.decode(encoding)
    except (OSError, UnicodeError):
        return None
    if any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
        return None
    return text


def is_env_name(name: str) -> bool:
    """Match local environment config files and directories."""
    return name == ".env" or name.startswith(".env.")


def is_sensitive_name(name: str) -> bool:
    """Conservatively exclude common credential files from model-visible source."""
    lowered = name.casefold()
    return (is_env_name(lowered) or lowered in SENSITIVE_DIRS
            or Path(lowered).stem in {"credentials", "secrets"}
            or lowered in SENSITIVE_FILES or Path(lowered).suffix in SENSITIVE_SUFFIXES)


@dataclass(frozen=True)
class SourceFile:
    """表示任意语言文本源码的不可变数据类。"""
    path: Path  # 文件路径
    text: str  # 文件内容


PythonFile = SourceFile  # Backward-compatible public type.


def load_source_files(
    repo_path: str | Path,
    exclude_roots: tuple[str, ...] = (),
) -> list[SourceFile]:
    """从仓库读取各类语言的文本源码文件，不导入或执行代码。"""
    files: list[SourceFile] = []
    for path in iter_source_paths(repo_path, exclude_roots=exclude_roots):
        text = read_source_text(path)
        if text is not None:
            files.append(SourceFile(path=path, text=text))
    return files


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
        text = read_source_text(path)
        if text is not None:
            python_files.append(PythonFile(path=path, text=text))

    return python_files


def iter_python_paths(
    repo_path: str | Path,
    exclude_roots: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Preserve the Python-only discovery API for existing integrations."""
    for path in iter_source_paths(repo_path, exclude_roots=exclude_roots):
        if path.suffix.casefold() == ".py":
            yield path


def iter_source_paths(
    repo_path: str | Path,
    exclude_roots: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield source candidates deterministically, pruning dependencies and secrets."""
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
            if is_sensitive_name(filename):
                continue
            if (Path(filename).suffix.casefold() in NON_SOURCE_SUFFIXES
                    and filename.casefold() not in SOURCE_FILENAMES):
                continue
            path = current / filename
            try:
                eligible = not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES
            except OSError:
                continue
            if eligible:
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
