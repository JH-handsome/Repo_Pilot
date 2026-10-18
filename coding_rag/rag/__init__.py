"""RAG orchestration, prompting, tracing, and answer generation."""

from coding_rag.rag.ask import RetrievalConfig, retrieve_code
from coding_rag.rag.answer_generator import AnswerGenerator, build_generator
from coding_rag.rag.prompt import GenerationMode, format_results_as_context
from coding_rag.rag.trace import build_retrieval_trace, render_trace_report

__all__ = [
    "RetrievalConfig",
    "AnswerGenerator",
    "GenerationMode",
    "build_generator",
    "build_retrieval_trace",
    "format_results_as_context",
    "render_trace_report",
    "retrieve_code",
]
