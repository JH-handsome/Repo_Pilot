"""早期代码切片原型。

当前主流程使用 `coding_rag.code_splitter` 中的 `CodeChunk` 数据类和切片函数。
这里保留旧的 dict 风格实现，方便理解项目演进。
"""


def split_code_into_chunks(
    file: dict,
    chunk_size: int = 80,
    overlap: int = 20,
) -> list[dict]:
    """
    按行把一个代码文件切成多个 chunk。

    参数：
        file: load_python_files 返回的单个文件字典
        chunk_size: 每个 chunk 最多包含多少行
        overlap: 相邻 chunk 之间重叠多少行

    返回：
        [
            {
                "chunk_id": "path/to/file.py:1-80",
                "file_path": "path/to/file.py",
                "start_line": 1,
                "end_line": 80,
                "content": "...",
            }
        ]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    file_path = file["path"]
    content = file["content"]

    lines = content.splitlines()

    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))

        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        start_line = start + 1
        end_line = end

        chunk = {
            "chunk_id": f"{file_path}:{start_line}-{end_line}",
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "content": chunk_content,
        }

        chunks.append(chunk)

        if end == len(lines):
            break

        start = end - overlap

    return chunks


def split_files_into_chunks(
    files: list[dict],
    chunk_size: int = 80,
    overlap: int = 20,
) -> list[dict]:
    """
    把多个文件全部切成 chunks。
    """
    all_chunks = []

    for file in files:
        chunks = split_code_into_chunks(
            file=file,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    return all_chunks
