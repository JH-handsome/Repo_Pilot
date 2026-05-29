"""旧版 LLM 判断器。

新代码默认使用 `rag.answer_generator.AnswerGenerator`。这个模块保留给
`main.py --legacy-judge`，用于兼容最早的“检索结果相关性判断”流程。
"""

from __future__ import annotations

from dataclasses import dataclass

from coding_rag.bm25_retriever import SearchResult
from coding_rag.llm_client import OpenAICompatibleChatClient


SYSTEM_PROMPT = """你是一个严谨的代码搜索判断助手。

只能使用提供的代码块。请先判断哪些代码块与用户问题相关，再基于相关代码块回答。

代码块来源标签：
- source=bm25 表示该代码块直接匹配用户查询。
- source=recall:* 表示该代码块是作为相邻上下文被召回的。

规则：
- 不要虚构代码块中没有出现的文件、函数或行为。
- 引用有用代码块时，使用 path:start-end 格式标出文件和行号。
- 如果证据不足，请说明“无法在检索到的代码中找到答案”。
- 使用与用户问题相同的语言回答。
"""


@dataclass
class LLMJudge:
    """基于检索结果调用 LLM 做相关性判断和回答。"""

    client: OpenAICompatibleChatClient
    max_context_chars: int = 12000

    def judge(self, query: str, results: list[SearchResult]) -> str:
        messages = build_judge_messages(query, results, self.max_context_chars)
        return self.client.complete(messages)


def build_judge_messages(
    query: str,
    results: list[SearchResult],
    max_context_chars: int,
) -> list[dict[str, str]]:
    """把用户问题和检索结果拼成 legacy judge 模式使用的消息列表。"""
    context = format_results_for_prompt(results, max_context_chars)
    user_prompt = f"""用户问题：
{query}

检索到的代码块：
{context}

请判断相关性，并按以下结构回答：
相关代码块：
- path:start-end - 相关原因

答案：
..."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_results_for_prompt(results: list[SearchResult], max_context_chars: int) -> str:
    """把检索结果格式化为带路径、行号和代码块的 prompt 上下文。"""
    blocks: list[str] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = (
            f"[{index}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(score={result.score:.4f}, source={result.source})"
        )
        block = f"{header}\n```python\n{chunk.text.rstrip()}\n```"

        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break

        if len(block) > remaining:
            block = block[: max(0, remaining - 40)].rstrip() + "\n...[代码已截断]"

        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks)
