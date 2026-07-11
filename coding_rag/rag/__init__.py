"""RAG orchestration, prompting, tracing, and answer generation."""

from coding_rag.rag.ask import AskModeConfig, AskModeRun, build_ask_messages, retrieve_for_ask
from coding_rag.rag.answer_generator import AnswerGenerator, build_generator
from coding_rag.rag.prompt import GenerationMode, format_results_as_context
from coding_rag.rag.trace import build_retrieval_trace, render_trace_report

__all__ = [
    "AskModeConfig",
    "AskModeRun",
    "AnswerGenerator",
    "GenerationMode",
    "build_ask_messages",
    "build_generator",
    "build_retrieval_trace",
    "format_results_as_context",
    "render_trace_report",
    "retrieve_for_ask",
]
