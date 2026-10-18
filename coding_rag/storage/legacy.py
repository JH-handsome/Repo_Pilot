"""Normalize archived legacy traces without running models or repository tools."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
import sqlite3

from coding_rag.storage.persistence import sanitize_trace
from coding_rag.storage.sqlite import SQLiteTraceStore


STAGES = ("initial_search", "neighbor_recall", "final_filter", "context_compaction")


def _event(step: str, *, status: str = "unknown", summary: dict | None = None,
           artifacts: dict | None = None) -> dict:
    """Unknown historical timestamps stay unknown, including synthesized events."""
    return {"step": step, "status": status, "input": {}, "output_summary": summary or {},
            "artifacts": artifacts or {}, "error": None, "duration_ms": None,
            "occurred_at": None}


def normalize_legacy_trace(record: object) -> tuple[dict, dict | None] | None:
    """Recognize trace records by structure; summaries and stage sidecars are skipped."""
    if not isinstance(record, dict):
        return None
    original = sanitize_trace(deepcopy(record))
    candidate = original
    context = None
    if isinstance(original.get("trace"), dict):
        candidate = original["trace"]
        if isinstance(original.get("context"), dict):
            context = original["context"]
    elif isinstance(original.get("trajectory"), dict):
        candidate = original["trajectory"]

    run = candidate.get("run")
    if isinstance(run, dict) and isinstance(candidate.get("events"), list):
        trace = deepcopy(candidate)
        if "run_id" in run:
            if not isinstance(run["run_id"], str) or not run["run_id"]:
                raise ValueError("invalid historical run_id")
            if "revision" in run and (type(run["revision"]) is not int or run["revision"] < 1):
                raise ValueError("invalid historical revision")
            # Native snapshots retain their shape so same-revision conflicts remain visible.
            trace["run"].setdefault("revision", 1)
            return trace, context
        mode = run.get("mode") or "legacy"
    else:
        stages = candidate.get("stages")
        retrieval_row = ("query" in original and all(key in original for key in ("seed", "recalled", "final")))
        evaluation_row = all(key in original for key in ("id", "trace_complete", "result_files"))
        if not isinstance(stages, dict) and not retrieval_row and not evaluation_row:
            return None
        if retrieval_row and not isinstance(stages, dict):
            stages = {name: original[key] for name, key in
                      zip(STAGES, ("seed", "recalled", "final", "context")) if key in original}
        if isinstance(stages, dict):
            if not any(key in stages for key in STAGES):
                return None
            events = []
            for name in STAGES:
                if name not in stages:
                    continue
                rows = stages[name]
                if not isinstance(rows, (list, dict)):
                    raise ValueError("invalid historical retrieval stage")
                events.append(_event(name, summary={"result_count": len(rows)}, artifacts={"results": rows}))
            mode = "retrieval_eval" if retrieval_row else "retrieval"
        else:
            mode = "agent_eval"
            events = [_event("evaluation", summary={key: original[key] for key in
                             ("trace_complete", "candidate_file_hit", "candidate_recall_at_5") if key in original})]
        agent_steps = original.get("agent_trace")
        if isinstance(agent_steps, list):
            mode = "agent_exec"
            for item in agent_steps:
                if not isinstance(item, dict) or not isinstance(item.get("step"), str):
                    raise ValueError("invalid historical agent step")
                status = item.get("status") or "unknown"
                if status in {"done", "drafted", "planned"}:
                    status = "success"
                events.append(_event(item["step"], status=status, artifacts=item.get("artifacts")))
        elif "workflow" in original and "task" in original:
            mode = "agent_exec"
        params = candidate.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("invalid historical parameters")
        trace = {"trace_version": "1.0", "run": {
            "mode": mode, "status": "unknown",
            "query": original.get("query", original.get("task", candidate.get("query"))),
            "repo_path": params.get("repo_path", original.get("repo")),
        }, "events": events, "artifacts": {}}

    # No path, filename, line number, import time or random UUID enters this identity.
    canonical = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    run = trace["run"]
    run["run_id"] = "legacy-" + digest
    run["revision"] = 1
    run.setdefault("mode", mode)
    run.setdefault("status", "unknown")
    run.setdefault("started_at", None)
    run.setdefault("finished_at", None)
    trace.setdefault("trace_version", "1.0")
    artifacts = trace.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("invalid historical artifacts")
    artifacts["legacy_import"] = {"identity_version": 1, "content_sha256": digest,
                                "original_record": original}
    return trace, context


def import_archived_traces(store: SQLiteTraceStore) -> dict:
    """Convert every archived record; existing run identity and revision rules apply."""
    counts = {"saved": 0, "unchanged": 0, "skipped": 0, "failed": 0}
    errors = []
    for archived in store.iter_json_files():
        try:
            content = archived["content"]
            if archived["format"] == "jsonl":
                records = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                payload = json.loads(content)
                records = payload if isinstance(payload, list) else [payload]
        except (ValueError, TypeError) as error:
            counts["failed"] += 1
            errors.append({"file_id": archived["file_id"], "error": str(error)})
            continue
        for index, record in enumerate(records, start=1):
            try:
                normalized = normalize_legacy_trace(record)
                if normalized is None:
                    counts["skipped"] += 1
                    continue
                trace, context = normalized
                run = trace["run"]
                for event in trace.get("events", []):
                    if not isinstance(event, dict) or not isinstance(event.get("step"), str):
                        raise ValueError("invalid historical event")
                    if event.get("error") is not None and not isinstance(event["error"], dict):
                        raise ValueError("invalid historical event error")
                existing = store.get_trace(run["run_id"])
                if run["run_id"].startswith("legacy-") and "legacy_import" in trace.get("artifacts", {}):
                    # Reusing the first archive timestamp keeps copies and reimports identical.
                    if not run.get("updated_at"):
                        run["updated_at"] = existing["run"]["updated_at"] if existing else archived["saved_at"]
                        trace["artifacts"]["legacy_import"]["updated_at_fallback"] = "first_archive_saved_at"
                elif not run.get("updated_at"):
                    run["updated_at"] = existing["run"]["updated_at"] if existing else archived["saved_at"]
                if existing is not None and (trace == existing or run["revision"] < existing["run"]["revision"]):
                    # Copies may carry a different export context; neither equal nor older
                    # snapshots should rewrite the existing run's identity or history.
                    counts["unchanged"] += 1
                    continue
                if context is None:
                    context = store.get_trace_context(run["run_id"])
                if context is None:
                    repo = run.get("repo_path") or (trace.get("params") or {}).get("repo_path")
                    context = {"source": "legacy_import", "repo_key": os.path.normcase(str(repo)) if repo else "unknown"}
                saved = store.save_trace(trace, context)
                counts["saved" if saved else "unchanged"] += 1
            except (ValueError, TypeError, KeyError, sqlite3.Error) as error:
                counts["failed"] += 1
                errors.append({"file_id": archived["file_id"], "record": index,
                               "error": str(error)})
    return {**counts, "errors": errors}
