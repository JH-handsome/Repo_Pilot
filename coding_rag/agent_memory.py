"""轻量级 Agent 记忆存储。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from coding_rag.tokenizer import CodeTokenizer


@dataclass(frozen=True)
class AgentMemory:
    """一次 Agent 任务的长期记忆记录。"""

    id: str
    created_at: str
    task: str
    status: str
    summary: str
    files: list[str]
    decisions: list[str]


class AgentMemoryStore:
    """JSONL 本地记忆库。"""

    def __init__(self, path: str | Path, tokenizer: CodeTokenizer | None = None):
        self.path = Path(path)
        self.tokenizer = tokenizer or CodeTokenizer()

    def load_all(self) -> list[AgentMemory]:
        if not self.path.exists():
            return []

        memories: list[AgentMemory] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                memories.append(AgentMemory(**payload))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{self.path}: invalid memory record at line {line_number}") from error
        return memories

    def append(self, memory: AgentMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(memory), ensure_ascii=False) + "\n")

    def append_if_new(self, memory: AgentMemory) -> bool:
        """写入非重复记忆；如果已有等价任务和文件集合，则跳过。"""
        if self.has_duplicate(memory):
            return False
        self.append(memory)
        return True

    def has_duplicate(self, memory: AgentMemory) -> bool:
        memory_key = memory_identity(memory)
        return any(memory_identity(existing) == memory_key for existing in self.load_all())

    def search(self, task: str, limit: int = 5) -> list[AgentMemory]:
        if limit <= 0:
            return []

        query_tokens = set(self.tokenizer.tokenize(task))
        scored: list[tuple[float, int, AgentMemory]] = []
        memories = self.load_all()
        for index, memory in enumerate(memories):
            memory_text = " ".join([memory.task, memory.summary, " ".join(memory.files), " ".join(memory.decisions)])
            memory_tokens = set(self.tokenizer.tokenize(memory_text))
            score = token_overlap(query_tokens, memory_tokens)
            if score <= 0:
                continue
            scored.append((score, index, memory))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for _, _, memory in scored[:limit]]


def build_memory(
    task: str,
    status: str,
    summary: str,
    files: list[str],
    decisions: list[str],
) -> AgentMemory:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return AgentMemory(
        id=f"agent-{created_at}",
        created_at=created_at,
        task=task,
        status=status,
        summary=summary,
        files=dedupe(files),
        decisions=dedupe(decisions),
    )


def token_overlap(query_tokens: set[str], memory_tokens: set[str]) -> float:
    if not query_tokens or not memory_tokens:
        return 0.0
    return len(query_tokens & memory_tokens) / len(query_tokens)


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def memory_identity(memory: AgentMemory) -> tuple[str, tuple[str, ...]]:
    return (normalize_memory_text(memory.task), tuple(sorted(memory.files)))


def normalize_memory_text(text: str) -> str:
    return " ".join(text.casefold().split())
