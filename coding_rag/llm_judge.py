from __future__ import annotations

from dataclasses import dataclass

from coding_rag.bm25_retriever import SearchResult
from coding_rag.llm_client import OpenAICompatibleChatClient


SYSTEM_PROMPT = """You are a careful code-search judge.

Use only the provided code chunks. Decide which chunks are relevant to the
query, then answer based on those chunks.

Chunk source labels:
- source=bm25 means the chunk directly matched the query.
- source=recall:* means the chunk was recalled as nearby context.

Rules:
- Do not invent files, functions, or behavior not shown in the chunks.
- Cite useful chunks with path and line range.
- If the chunks do not contain enough evidence, say that the answer was not
  found in the retrieved code.
- Answer in the same language as the user's query.
"""


@dataclass
class LLMJudge:
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
    context = format_results_for_prompt(results, max_context_chars)
    user_prompt = f"""Query:
{query}

Retrieved code chunks:
{context}

Please judge relevance and answer with this structure:
Relevant chunks:
- path:start-end - reason

Answer:
..."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_results_for_prompt(results: list[SearchResult], max_context_chars: int) -> str:
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
            block = block[: max(0, remaining - 40)].rstrip() + "\n...[truncated]"

        blocks.append(block)
        used_chars += len(block)

    return "\n\n".join(blocks)
