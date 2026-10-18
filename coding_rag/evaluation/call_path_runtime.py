"""Offline, query-only call-path navigation contract through the unified Agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from coding_rag.agent.runtime import UnifiedRunConfig, run_unified_query
from coding_rag.rag.citation_validator import extract_citations

from .agent_runtime import _decision, _last_observation, _normalize
from .multirepo import MultiRepoCase


ENTRY_PATTERN = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\(\s*\)")


class CallPathScriptedPolicy:
    """Exercise search → static trace → two reads with no gold or model access.

    This deliberately narrow policy only handles questions explicitly naming an
    entry function/method and a target call. It is not a natural-language Agent.
    """

    def __init__(self, query: str):
        """Bind only the user's question and its first explicit callable name."""
        self.query = query
        match = ENTRY_PATTERN.search(query)
        self.entry_symbol = match.group(1) if match else None
        self.stage = 0
        self.entry: dict[str, Any] | None = None
        self.target: dict[str, Any] | None = None
        self.entry_read: dict[str, Any] | None = None

    def _target_edge(self, edges: object) -> dict[str, Any] | None:
        """Choose one query-mentioned call; ties and unlocated edges fail closed."""
        if not isinstance(edges, list):
            return None
        explicit_targets = {
            match.group(1).casefold()
            for match in list(ENTRY_PATTERN.finditer(self.query))[1:]
            if match.group(1).casefold() != self.entry_symbol.casefold()
        }
        if len(explicit_targets) > 1:
            return None
        candidates: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            if edge.get("kind") == "same_class_method_candidate":
                continue
            start, end = edge.get("target_start_line"), edge.get("target_end_line")
            source_line = edge.get("source_line")
            if (not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start
                    or not isinstance(source_line, int) or source_line < 1 or not edge.get("target_path")):
                continue
            names = (edge.get("call"), edge.get("target_symbol"))
            if explicit_targets:
                score = max(
                    (len(name) for name in names if isinstance(name, str)
                     and name.casefold() in explicit_targets), default=0,
                )
            else:
                score = max(
                    (len(name) for name in names if isinstance(name, str)
                     and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", self.query, re.I)
                     and name.casefold() != self.entry_symbol.casefold()),
                    default=0,
                )
            if score:
                key = (str(edge.get("target_path")), str(edge.get("target_symbol")))
                if key not in candidates or score > candidates[key][0]:
                    candidates[key] = (score, edge)
        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None
        return ranked[0][1]

    @staticmethod
    def _read_arguments(path: str, start: int, end: int) -> dict:
        """Limit evidence reads to forty source lines per file."""
        return {"path": path, "start_line": start, "end_line": min(end, start + 39)}

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Make the next tool decision from the original query and last observation."""
        if not self.entry_symbol:
            return _decision(action="answer", answer="问题未明确指定入口函数，无法执行此离线契约。")
        if self.stage == 0:
            self.stage = 1
            return _decision(action="tool", tool="search_code", arguments={"query": self.query, "top_k": 5})
        observation = _last_observation(messages)
        if not observation or "error" in observation:
            return _decision(action="answer", answer="调用链工具未返回可用证据。")
        output = observation.get("output") or {}
        if self.stage == 1 and observation.get("tool") == "search_code":
            rows = output.get("results") if isinstance(output, dict) else None
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                return _decision(action="answer", answer="检索未找到入口。")
            symbol_name = self.entry_symbol.rsplit(".", 1)[-1]
            definition = re.compile(rf"\b(?:async\s+)?def\s+{re.escape(symbol_name)}\s*\(")
            self.entry = next(
                (row for row in rows if isinstance(row, dict)
                 and definition.search(str(row.get("text") or ""))),
                rows[0],
            )
            self.stage = 2
            return _decision(action="tool", tool="trace_call_path", arguments={
                "path": self.entry["path"], "symbol": self.entry_symbol,
            })
        if self.stage == 2 and observation.get("tool") == "trace_call_path":
            self.target = self._target_edge(output.get("edges") if isinstance(output, dict) else None)
            if self.target is None or self.entry is None:
                return _decision(action="answer", answer="静态调用边没有唯一的查询目标。")
            self.stage = 3
            source_start = max(output["start_line"], self.target["source_line"] - 20)
            return _decision(action="tool", tool="read_file", arguments=self._read_arguments(
                self.entry["path"], source_start, output["end_line"],
            ))
        if self.stage == 3 and observation.get("tool") == "read_file":
            if not isinstance(output, dict) or not output.get("path") or not output.get("text"):
                return _decision(action="answer", answer="入口源码未能读取。")
            self.entry_read = output
            self.stage = 4
            return _decision(action="tool", tool="read_file", arguments=self._read_arguments(
                self.target["target_path"],
                self.target["target_start_line"], self.target["target_end_line"],
            ))
        if self.stage == 4 and observation.get("tool") == "read_file":
            if not isinstance(output, dict) or not output.get("path") or not output.get("text"):
                return _decision(action="answer", answer="目标源码未能读取。")
            entry = self.entry_read
            return _decision(action="answer", answer=(
                f"入口代码：{entry['path']}:{entry['start_line']}-{entry['end_line']}；"
                f"被调目标：{output['path']}:{output['start_line']}-{output['end_line']}。"
            ))
        return _decision(action="answer", answer="调用链观察结果与预期顺序不符。")


def observed_line(output: dict, path: str | None, line: int | None) -> bool:
    """Require non-empty source text at the expected path and numbered line."""
    start, end = output.get("start_line"), output.get("end_line")
    return bool(
        path and isinstance(line, int) and isinstance(start, int) and isinstance(end, int)
        and _normalize(output.get("path")) == _normalize(path)
        and start <= line <= end
        and re.search(rf"(?m)^{line}: ", str(output.get("text") or ""))
    )


def evaluate_case(case: MultiRepoCase, repo_root: Path, config: dict) -> dict:
    """Score real graph observations and citations only after the policy finishes."""
    policy = CallPathScriptedPolicy(case.query)
    run = run_unified_query(case.query, UnifiedRunConfig(
        repo_path=repo_root, top_k=5, chunk_size=int(config["chunk_size"]),
        overlap=int(config["overlap"]), recall_window=int(config["recall_window"]),
        max_recall_results=config.get("max_recall_results"),
        min_final_score=float(config["min_final_score"]),
        graph_weight=float(config["graph_weight"]), path_weight=float(config["path_weight"]),
        index_exclude_roots=tuple(config["exclude_roots"]), max_steps=4,
    ), policy)
    gold = {_normalize(path) for path in case.gold_files}
    read_outputs = [
        observation["output"]
        for observation in run.observations if observation.get("tool") == "read_file"
        and "error" not in observation and isinstance(observation.get("output"), dict)
    ]
    read_paths = [_normalize(output.get("path")) for output in read_outputs]
    source_call_observed = bool(
        policy.target and policy.entry and len(read_outputs) >= 1
        and observed_line(read_outputs[0], policy.entry.get("path"), policy.target["source_line"])
    )
    target_definition_observed = bool(
        policy.target and len(read_outputs) >= 2
        and observed_line(read_outputs[1], policy.target["target_path"], policy.target["target_start_line"])
    )
    final = next((event for event in reversed(run.trace.get("events", []))
                  if event.get("step") == "final_answer"), {})
    citations = extract_citations(run.answer)
    citation_paths = {_normalize(citation.path) for citation in citations}
    return {
        "id": case.id,
        "entry_symbol_from_query": policy.entry_symbol,
        "selected_entry": (policy.entry or {}).get("path"),
        "selected_target": (policy.target or {}).get("target_path"),
        "tool_sequence": [decision.tool for decision in run.decisions if decision.action == "tool"],
        "read_paths": read_paths,
        "gold_files": sorted(gold),
        "read_gold_coverage": len(gold.intersection(read_paths)) / len(gold),
        "all_gold_read": gold.issubset(read_paths),
        "source_call_observed": source_call_observed,
        "target_definition_observed": target_definition_observed,
        "citation_valid": bool(citations) and not (final.get("output_summary") or {}).get("citation_issues", True),
        "all_gold_cited": gold.issubset(citation_paths),
        "terminal_success": run.status == "success",
        "contract_success": (
            [decision.tool for decision in run.decisions if decision.action == "tool"]
            == ["search_code", "trace_call_path", "read_file", "read_file"]
            and gold.issubset(read_paths) and bool(citations)
            and source_call_observed and target_definition_observed
            and not (final.get("output_summary") or {}).get("citation_issues", True)
            and gold.issubset(citation_paths)
        ),
    }
