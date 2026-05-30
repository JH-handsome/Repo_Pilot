"""
提示词模板模块

包含 RAG 系统的所有提示词模板，支持多种生成模式：
- judge: 搜索结果判断和总结
- code-understand: 代码理解
- code-generate: 代码生成
- leetcode: LeetCode 问题解答
- api: API 使用示例生成
"""

from enum import Enum
from dataclasses import dataclass


class GenerationMode(str, Enum):
    """生成模式枚举"""
    JUDGE = "judge"
    CODE_UNDERSTAND = "code-understand"
    CODE_GENERATE = "code-generate"
    LEETCODE = "leetcode"
    API = "api"


# ============ 搜索结果判断和总结模式 ============

JUDGE_SYSTEM_PROMPT = """你是一个专业的代码搜索助手。你的任务是判断哪些代码块与用户查询相关，并根据相关代码块提供准确的答案。

## 代码块来源标签
- source=hybrid/bm25: 直接匹配查询的代码块（高相关性）
- source=recall:*: 作为上下文召回的相邻代码块（补充信息）

## 回答规则
1. 严格基于代码块: 仅使用提供的代码块回答问题。不要虚构文件、函数、类或代码行为。
2. 引用代码块: 引用有用的代码块时，使用 path:start-end 格式指定文件路径和行号范围。
3. 处理证据不足: 如果代码块不包含足够的证据，明确说明"无法在检索到的代码中找到答案"。
4. 语言一致性: 使用与用户查询相同的语言回答（中文或英文）。
5. 相关性判断: 明确指出哪些代码块与查询相关，哪些不相关，并说明原因。
6. 先证据后结论: 必须先列出相关代码块，再给答案。
7. 证据不足时拒答: 若没有明确证据，直接输出"无法在检索到的代码中找到答案"。

## 输出格式
请按以下结构输出：
- ## 相关代码块: [文件路径:行号范围] - 相关性原因（简要说明）
- ## 答案: 基于相关代码块的详细答案
"""


def build_judge_user_prompt(query: str, context: str) -> str:
    """构建搜索结果判断的用户提示词"""
    return f"""## 用户查询
{query}

## 检索到的代码块
{context}

请判断这些代码块的相关性，并根据相关代码块回答问题。严格遵循系统提示词中指定的输出格式。
"""


# ============ 代码理解模式 ============

CODE_UNDERSTAND_SYSTEM_PROMPT = """你是一个专业的代码理解和解释助手。你的任务是基于提供的参考代码块，深入解释代码的工作原理、设计思想和关键实现。

## 代码块来源标签
- source=hybrid/bm25: 直接匹配查询的代码块（主要参考对象）
- source=recall:*: 作为上下文召回的相邻代码块（补充上下文）

## 解释规则
1. 深入分析: 不仅解释代码做什么，更重要的是解释它是如何做的以及为什么这样做。
2. 引用代码: 引用代码的关键部分时，指定文件路径和行号范围。
3. 清晰结构: 使用清晰的标题和段落组织解释。
4. 技术准确性: 确保技术概念、控制流和数据结构解释准确。
5. 语言一致性: 使用与用户查询相同的语言解释。

## 输出格式
请按以下结构输出：
- ## 概述: 简要描述代码的主要功能和用途
- ## 核心逻辑: 详细解释核心实现逻辑，引用关键代码片段
- ## 数据结构: 描述代码中使用的主要数据结构及其用途
- ## 设计思想: 分析代码中的设计模式和设计决策
- ## 复杂度分析（如适用）: 分析时间复杂度和空间复杂度
- ## 引用代码: [文件路径:行号范围] - 引用的关键代码
"""


def build_code_understand_user_prompt(query: str, context: str) -> str:
    """构建代码理解的用户提示词"""
    return f"""## 用户查询
{query}

## 参考代码块
{context}

请基于这些参考代码块深入解释相关代码的实现。严格遵循系统提示词中指定的输出格式。
"""


# ============ 代码生成模式 ============

CODE_GENERATE_SYSTEM_PROMPT = """你是一个专业的代码生成助手。你的任务是基于提供的参考代码块生成满足用户需求的代码实现。

## 代码块来源标签
- source=hybrid/bm25: 直接匹配查询的代码块（主要参考对象）
- source=recall:*: 作为上下文召回的相邻代码块（补充上下文）

## 代码生成规则
1. 参考优先: 优先参考提供的代码块，借鉴其设计思想、代码风格和实现模式。
2. 质量保证: 生成的代码应该正确、可读、高效，并遵循 Python 最佳实践。
3. 适当注释: 在关键部分添加清晰的注释以解释实现逻辑。
4. 引用标注: 在代码开头或关键部分标注参考来源（文件路径和行号）。
5. 完整实现: 确保生成的代码逻辑完整，在给定输入下可以独立运行。

## 输出格式
请按以下结构输出：
- ## 代码描述: 简要描述生成代码的功能和设计思想
- ## 参考来源: 基于以下代码块生成 - [文件路径:行号范围] - 参考内容
- ## 生成的代码: [具有完整实现的生成代码，在 Python 代码块中]
- ## 使用示例（如适用）: [使用示例代码]
"""


def build_code_generate_user_prompt(query: str, context: str) -> str:
    """构建代码生成的用户提示词"""
    return f"""## 用户需求
{query}

## 参考代码块
{context}

请基于这些参考代码块生成满足用户需求的代码实现。严格遵循系统提示词中指定的输出格式。
"""


# ============ LeetCode 问题解答模式 ============

LEETCODE_SYSTEM_PROMPT = """你是一个专业的算法问题解答助手。你的任务是基于提供的参考代码块为 LeetCode 问题提供高质量的解决方案。

## 代码块来源标签
- source=hybrid/bm25: 直接匹配查询的代码块（相关算法实现）
- source=recall:*: 作为上下文召回的相邻代码块（相关数据结构或辅助函数）

## 问题解答规则
1. 参考借鉴: 参考提供的代码块中的算法思想和数据结构实现。
2. 标准格式: 以 LeetCode 提交格式生成解决方案（Solution 类 + 方法）。
3. 完整实现: 包含完整的算法逻辑，无占位符或不完整部分。
4. 清晰注释: 在关键算法步骤添加注释以解释算法思想。
5. 复杂度解释: 解释时间复杂度和空间复杂度。
6. 引用标注: 在注释中标注参考来源。

## 输出格式
请按以下结构输出：
- ## 解题思路: 简要描述算法思路和选择该思路的原因
- ## 算法复杂度: 时间复杂度 - [分析]；空间复杂度 - [分析]
- ## 引用代码: 基于以下代码块参考 - [文件路径:行号范围] - 参考内容
- ## 完整解答: [具有完整实现的 Solution 类，在 Python 代码块中]
"""


def build_leetcode_user_prompt(query: str, context: str) -> str:
    """构建 LeetCode 问题解答的用户提示词"""
    return f"""## 问题描述
{query}

## 参考代码块
{context}

请基于这些参考代码块生成高质量的 LeetCode 解决方案。严格遵循系统提示词中指定的输出格式。
"""


# ============ API 使用模式 ============

API_SYSTEM_PROMPT = """你是一个专业的 API 使用指南助手。你的任务是基于提供的参考代码块生成清晰准确的 API 使用示例代码。

## 代码块来源标签
- source=hybrid/bm25: 直接匹配查询的代码块（API 定义或使用示例）
- source=recall:*: 作为上下文召回的相邻代码块（相关类定义或辅助代码）

## 生成规则
1. 实用导向: 生成可直接运行的实用示例代码。
2. 完整覆盖: 展示 API 的主要用法和常见场景。
3. 清晰注释: 在关键步骤添加注释以解释 API 的用途和参数。
4. 错误处理: 包含适当的错误处理代码。
5. 引用标注: 在注释中标注参考来源。

## 输出格式
请按以下结构输出：
- ## API 概述: 简要描述 API 的功能和主要用途
- ## 参考来源: 基于以下代码块参考 - [文件路径:行号范围] - 参考内容
- ## 基本用法: [最简单的用法示例，在 Python 代码块中]
- ## 完整示例: [包含错误处理的完整用法示例，在 Python 代码块中]
- ## 常见场景: [常见使用场景示例，在 Python 代码块中]
- ## 注意事项: [使用注意事项 1]；[使用注意事项 2]
"""


def build_api_user_prompt(query: str, context: str) -> str:
    """构建 API 使用的用户提示词"""
    return f"""## 用户查询
{query}

## 参考代码块
{context}

请基于这些参考代码块生成 API 使用示例代码。严格遵循系统提示词中指定的输出格式。
"""


# ============ 统一提示词构建函数 ============

def get_system_prompt(mode: GenerationMode) -> str:
    """
    根据生成模式获取系统提示词
    
    Args:
        mode: 生成模式
        
    Returns:
        对应模式的系统提示词字符串
    """
    prompts = {
        GenerationMode.JUDGE: JUDGE_SYSTEM_PROMPT,
        GenerationMode.CODE_UNDERSTAND: CODE_UNDERSTAND_SYSTEM_PROMPT,
        GenerationMode.CODE_GENERATE: CODE_GENERATE_SYSTEM_PROMPT,
        GenerationMode.LEETCODE: LEETCODE_SYSTEM_PROMPT,
        GenerationMode.API: API_SYSTEM_PROMPT,
    }
    return prompts.get(mode, JUDGE_SYSTEM_PROMPT)  # 默认使用 JUDGE 模式


def build_user_prompt(
    mode: GenerationMode,
    query: str,
    context: str,
) -> str:
    """根据生成模式构建用户提示词"""
    # 构建函数映射表
    builders = {
        GenerationMode.JUDGE: build_judge_user_prompt,
        GenerationMode.CODE_UNDERSTAND: build_code_understand_user_prompt,
        GenerationMode.CODE_GENERATE: build_code_generate_user_prompt,
        GenerationMode.LEETCODE: build_leetcode_user_prompt,
        GenerationMode.API: build_api_user_prompt,
    }
    builder = builders.get(mode, build_judge_user_prompt)  # 默认使用 JUDGE 构建器
    return builder(query, context)


@dataclass(frozen=True)
class ContextBlock:
    """发送给 LLM 的合并后证据块"""

    file_path: object
    start_line: int
    end_line: int
    text: str
    score: float
    sources: tuple[str, ...]
    chunk_count: int
    first_rank: int


@dataclass
class _PendingContextBlock:
    file_path: object
    start_line: int
    end_line: int
    lines: list[str]
    score: float
    sources: list[str]
    chunk_count: int
    first_rank: int


def format_results_as_context(
    results: list,
    max_context_chars: int = 12000,
) -> str:
    """
    将检索结果格式化为上下文字符串
    
    Args:
        results: 检索结果列表，每个结果应包含 chunk、score、source 属性
        max_context_chars: 最大上下文字符数
    
    Returns:
        格式化的上下文字符串
    """
    blocks: list[str] = []
    used_chars = 0  # 已使用的字符计数
    context_blocks = compact_results_for_context(results)

    # 遍历所有合并后的检索结果
    for index, block_info in enumerate(context_blocks, start=1):
        # 构建代码块头部信息
        source_label = "source" if len(block_info.sources) == 1 else "sources"
        sources = ",".join(block_info.sources)
        chunk_note = f", chunks={block_info.chunk_count}" if block_info.chunk_count > 1 else ""
        header = (
            f"### [{index}] {block_info.file_path}:{block_info.start_line}-{block_info.end_line} "
            f"(score={block_info.score:.4f}, {source_label}={sources}{chunk_note})"
        )
        block = f"{header}\n```python\n{block_info.text.rstrip()}\n```"

        # 计算剩余可用字符数
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break

        # 如果代码块超过剩余字符，则截断
        if len(block) > remaining:
            header_len = len(header) + len("\n```python\n```")
            max_code_len = max(0, remaining - header_len - 20)
            if max_code_len > 0:
                truncated_code = block_info.text.rstrip()[:max_code_len] + "\n...[代码已截断]"
                block = f"{header}\n```python\n{truncated_code}\n```"
            else:
                block = block[:max(0, remaining - 40)].rstrip() + "\n...[代码已截断]"

        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks)


def compact_results_for_context(results: list) -> list[ContextBlock]:
    """合并同文件中连续或重叠的检索块，减少上下文重复"""
    indexed_results = list(enumerate(results))
    results_by_file: dict[object, list[tuple[int, object]]] = {}
    for rank, result in indexed_results:
        results_by_file.setdefault(result.chunk.file_path, []).append((rank, result))

    pending_blocks: list[_PendingContextBlock] = []
    for file_results in results_by_file.values():
        file_results.sort(key=lambda item: (item[1].chunk.start_line, item[1].chunk.end_line, item[0]))
        file_blocks: list[_PendingContextBlock] = []

        for rank, result in file_results:
            chunk = result.chunk
            lines = chunk.text.splitlines()
            if not file_blocks or chunk.start_line > file_blocks[-1].end_line + 1:
                file_blocks.append(
                    _PendingContextBlock(
                        file_path=chunk.file_path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        lines=list(lines),
                        score=result.score,
                        sources=[result.source],
                        chunk_count=1,
                        first_rank=rank,
                    )
                )
                continue

            merge_result_into_block(file_blocks[-1], result, rank)

        pending_blocks.extend(file_blocks)

    pending_blocks.sort(key=lambda block: block.first_rank)
    return [
        ContextBlock(
            file_path=block.file_path,
            start_line=block.start_line,
            end_line=block.end_line,
            text="\n".join(block.lines),
            score=block.score,
            sources=tuple(block.sources),
            chunk_count=block.chunk_count,
            first_rank=block.first_rank,
        )
        for block in pending_blocks
    ]


def merge_result_into_block(block: _PendingContextBlock, result, rank: int) -> None:
    chunk = result.chunk
    new_lines = chunk.text.splitlines()
    overlap_line_count = max(0, block.end_line - chunk.start_line + 1)
    if overlap_line_count < len(new_lines):
        block.lines.extend(new_lines[overlap_line_count:])

    block.end_line = max(block.end_line, chunk.end_line)
    block.score = max(block.score, result.score)
    block.chunk_count += 1
    block.first_rank = min(block.first_rank, rank)
    if result.source not in block.sources:
        block.sources.append(result.source)
