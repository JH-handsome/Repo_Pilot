"""Baseline compatibility and regression policy checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    """Result of comparing one scorecard with its compatible baseline."""
    compatible: bool
    passed: bool
    failures: tuple[str, ...]


DEFAULT_POLICY = {
    "max_drop": {
        "overall_score": 2.0,
        "retrieval.hit_at_1": 0.05,
        "retrieval.mrr_at_5": 0.02,
        "retrieval.recall_at_5": 0.02,
        "agent.fixture_diagnostic.dataset_coverage": 0.0,
        "agent.offline_contract.trace_complete_rate": 0.0,
        "agent.offline_contract.safe_tool_interface_rate": 0.0,
    },
    "max_ratio": {"efficiency.query_latency_ms_p95": 1.5},
}


def _value(payload: dict, dotted: str) -> float:
    """Read a numeric scorecard value using a dotted field path."""
    current = payload
    for part in dotted.split("."):
        current = current[part]
    return float(current)


def _raw_value(payload: dict, dotted: str):
    """Read any nested scorecard value using a dotted field path."""
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def compare_scorecards(current: dict, baseline: dict, policy: dict) -> GateResult:
    """Check dataset compatibility and all configured regression thresholds."""
    incompatibilities: list[str] = []
    compatibility_fields = (
        "schema_version",
        "experiment.datasets.retrieval.sha256",
        "experiment.datasets.retrieval.case_count",
        "experiment.datasets.agent.sha256",
        "experiment.datasets.agent.case_count",
        "experiment.config",
        "experiment.retriever.name",
    )
    for key in compatibility_fields:
        before = _raw_value(baseline, key)
        after = _raw_value(current, key)
        if before != after:
            incompatibilities.append(f"incompatible {key}: baseline={before!r}, current={after!r}")
    if incompatibilities:
        return GateResult(False, False, tuple(incompatibilities))

    failures: list[str] = []
    for metric, allowed in (policy.get("max_drop") or {}).items():
        before = _value(baseline, metric)
        after = _value(current, metric)
        drop = before - after
        if drop > float(allowed) + 1e-12:
            failures.append(f"{metric}: baseline={before:.6f}, current={after:.6f}, drop={drop:.6f}, allowed={float(allowed):.6f}")
    for metric, allowed in (policy.get("max_ratio") or {}).items():
        before = _value(baseline, metric)
        after = _value(current, metric)
        ratio = after / before if before > 0 else (1.0 if after <= 0 else float("inf"))
        if ratio > float(allowed) + 1e-12:
            failures.append(f"{metric}: baseline={before:.6f}, current={after:.6f}, ratio={ratio:.3f}, allowed={float(allowed):.3f}")
    return GateResult(True, not failures, tuple(failures))
