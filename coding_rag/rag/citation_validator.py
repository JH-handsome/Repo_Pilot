"""Validate path:start-end citations in generated RAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from coding_rag.rag.prompt import compact_results_for_context

if TYPE_CHECKING:
    from coding_rag.tools.bm25 import SearchResult


CITATION_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[\w./\\-]+?\.py):(?P<start>\d+)-(?P<end>\d+)"
)


@dataclass(frozen=True)
class Citation:
    """表示解析后的代码引用，包含文件路径与行范围。"""

    path: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class CitationValidationResult:
    """保存引用校验结果，包含有效引用、无效引用和缺失标记。"""

    citations: list[Citation]
    invalid_citations: list[Citation]
    missing_citations: bool

    @property
    def has_issues(self) -> bool:
        """当存在无效引用或缺失引用时返回 True。"""
        return self.missing_citations or bool(self.invalid_citations)


def extract_citations(answer: str) -> list[Citation]:
    """从回答文本中通过正则提取 path:start-end 格式的引用。"""
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
    """根据可用检索结果范围校验回答中的引用。"""
    available_ranges = [
        (str(block.file_path), block.start_line, block.end_line)
        for block in compact_results_for_context(results)
    ]
    return validate_answer_citations_against_ranges(answer, available_ranges)


def validate_answer_citations_against_ranges(
    answer: str,
    available_ranges: list[tuple[str, int, int]],
) -> CitationValidationResult:
    """Validate citations against code ranges observed through Agent tools."""
    citations = extract_citations(answer)
    invalid = [
        citation
        for citation in citations
        if not citation_is_supported(citation, available_ranges)
    ]
    missing = bool(available_ranges) and not citations and not is_refusal_answer(answer)

    return CitationValidationResult(
        citations=citations,
        invalid_citations=invalid,
        missing_citations=missing,
    )


def citation_is_supported(
    citation: Citation,
    available_ranges: list[tuple[str, int, int]],
) -> bool:
    """检查引用行范围是否被任一可用范围包含。"""
    if citation.start_line <= 0 or citation.end_line < citation.start_line:
        return False

    for path, start_line, end_line in available_ranges:
        if not paths_match(citation.path, path):
            continue
        if citation.start_line >= start_line and citation.end_line <= end_line:
            return True

    return False


def paths_match(cited_path: str, available_path: str) -> bool:
    """归一化后若路径相等，或其一为另一以斜杠分隔的后缀（支持相对/绝对路径混用），则返回 True。"""
    cited = normalize_path(cited_path)
    available = normalize_path(available_path)
    return cited == available or available.endswith(f"/{cited}") or cited.endswith(f"/{available}")


def normalize_path(path: str) -> str:
    """归一化文件路径以支持跨平台大小写不敏感比较。"""
    return path.replace("\\", "/").strip().casefold()


def is_refusal_answer(answer: str) -> bool:
    """检测回答是否表明无法找到相关代码。"""
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
    """当存在校验问题时追加中文引用校验报告段落。"""
    if not validation.has_issues:
        return answer

    lines = ["", "## 引用校验"]
    if validation.missing_citations:
        lines.append("- 未检测到 path:start-end 格式引用，请补充来自检索上下文的代码位置。")
    for citation in validation.invalid_citations:
        lines.append(f"- 无效引用: {citation.text}")

    return answer.rstrip() + "\n" + "\n".join(lines)
