"""
代码分割模块
负责将 Python 文件分割成固定大小的代码块，用于索引和检索
"""

from dataclasses import dataclass
from pathlib import Path

from coding_rag.file_loader import PythonFile


@dataclass(frozen=True)
class CodeChunk:
    """
    表示一个代码块的不可变数据类
    
    Attributes:
        file_path: 所属文件路径
        start_line: 起始行号（从1开始）
        end_line: 结束行号
        text: 代码块文本内容
    """
    file_path: Path
    start_line: int
    end_line: int
    text: str


def split_python_files(
    python_files: list[PythonFile],
    chunk_size: int = 40,
    overlap: int = 5,
) -> list[CodeChunk]:
    """
    将多个 Python 文件分割成基于行的代码块
    
    Args:
        python_files: PythonFile 对象列表
        chunk_size: 每个代码块的行数，默认为40
        overlap: 相邻代码块之间的重叠行数，默认为5
        
    Returns:
        CodeChunk 对象列表
    """
    chunks: list[CodeChunk] = []
    for python_file in python_files:
        chunks.extend(split_code_by_lines(python_file.path, python_file.text, chunk_size, overlap))
    return chunks


def split_code_by_lines(
    file_path: Path,
    text: str,
    chunk_size: int = 40,
    overlap: int = 5,
) -> list[CodeChunk]:
    """
    将单个 Python 文件分割成基于行的代码块
    
    使用滑动窗口方式分割，支持代码块之间的重叠
    
    Args:
        file_path: 文件路径
        text: 文件内容
        chunk_size: 每个代码块的行数
        overlap: 相邻代码块之间的重叠行数
        
    Returns:
        CodeChunk 对象列表
        
    Raises:
        ValueError: 参数值不合法时抛出
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # 将文本按行分割
    lines = text.splitlines()
    chunks: list[CodeChunk] = []
    step = chunk_size - overlap  # 滑动窗口步长

    # 使用滑动窗口方式遍历所有行
    for start_index in range(0, len(lines), step):
        chunk_lines = lines[start_index : start_index + chunk_size]
        if not chunk_lines:
            continue

        # 行号从1开始
        start_line = start_index + 1
        end_line = start_index + len(chunk_lines)
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                # 将代码行重新连接成文本
                text="\n".join(chunk_lines),
            )
        )

        # 如果已经到达文件末尾，提前退出
        if end_line == len(lines):
            break

    return chunks
