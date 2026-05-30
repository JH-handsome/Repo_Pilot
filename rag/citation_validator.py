"""Validate path:start-end citations in generated RAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.prompt import compact_results_for_context

if TYPE_CHECKING:
    from coding_rag.bm25_retriever import SearchResult


CITATION_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[\w./\\-]+?\.py):(?P<start>\d+)-(?P<end>\d+)"
)


@dataclass(frozen=True)
class Citation:
    path: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class CitationValidationResult:
    citations: list[Citation]
    invalid_citations: list[Citation]
    missing_citations: bool

    @property
    def has_issues(self) -> bool:
        return self.missing_citations or bool(self.invalid_citations)


def extract_citations(answer: str) -> list[Citation]:
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(answer):
        start_line = int(match.group("start"))
        end_line = int(match.group("end"))
        citations.append(
            Citation(
                path=match.group("path"),
                start_line=start_line,
                end_line=end_line,
                text=match.group(0),
            )
        )
    return citations


def validate_answer_citations(answer: str, results: list["SearchResult"]) -> CitationValidationResult:
    citations = extract_citations(answer)
    available_ranges = [
        (str(block.file_path), block.start_line, block.end_line)
        for block in compact_results_for_context(results)
    ]
    invalid = [
        citation
        for citation in citations
        if not citation_is_supported(citation, available_ranges)
    ]
    missing = bool(results) and not citations and not is_refusal_answer(answer)

    return CitationValidationResult(
        citations=citations,
        invalid_citations=invalid,
        missing_citations=missing,
    )


def citation_is_supported(
    citation: Citation,
    available_ranges: list[tuple[str, int, int]],
) -> bool:
    if citation.start_line <= 0 or citation.end_line < citation.start_line:
        return False

    for path, start_line, end_line in available_ranges:
        if not paths_match(citation.path, path):
            continue
        if citation.start_line >= start_line and citation.end_line <= end_line:
            return True

    return False


def paths_match(cited_path: str, available_path: str) -> bool:
    cited = normalize_path(cited_path)
    available = normalize_path(available_path)
    return cited == available or available.endswith(f"/{cited}") or cited.endswith(f"/{available}")


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().casefold()


def is_refusal_answer(answer: str) -> bool:
    normalized = answer.casefold()
    return (
        "无法在检索到的代码中找到答案" in answer
        or "cannot find" in normalized
        or "not enough evidence" in normalized
    )


def append_citation_validation_report(
    answer: str,
    validation: CitationValidationResult,
) -> str:
    if not validation.has_issues:
        return answer

    lines = ["", "## 引用校验"]
    if validation.missing_citations:
        lines.append("- 未检测到 path:start-end 格式引用，请补充来自检索上下文的代码位置。")
    for citation in validation.invalid_citations:
        lines.append(f"- 无效引用: {citation.text}")

    return answer.rstrip() + "\n" + "\n".join(lines)
