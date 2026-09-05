"""Trace configuration, credential filtering and save-only failure recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from coding_rag.storage.sqlite import SQLiteTraceStore


OMITTED_KEYS = {
    "approval", "approval_id", "approval_fingerprint", "fingerprint", "write_approval",
    "resume_state", "messages", "raw_text", "raw_output", "integrity_token",
    "api_key", "password", "secret", "token", "authorization", "cookie",
    "signing_key", "access_token", "refresh_token", "private_key",
}
SECRET_ASSIGNMENT = re.compile(
    r'''(?i)(\b(?:[a-z_]*(?:api[_-]?key|token|password|secret)|authorization|cookie)\b["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}]+)'''
)


def configured_trace_db(explicit: str | Path | None = None) -> Path | None:
    """Resolve a server/CLI setting; 'off' disables storage without changing tools."""
    value = explicit if explicit is not None else os.environ.get("REPOPILOT_TRACE_DB")
    if value is not None and str(value).casefold() == "off":
        return None
    if value is None:
        return Path(__file__).resolve().parents[2] / "artifacts" / "traces.sqlite3"
    if not str(value).strip():
        raise ValueError("trace database path cannot be empty")
    return Path(value).expanduser().resolve()


def sanitize_trace(trace: dict) -> dict:
    """Copy a trace, omitting execution state and masking recognizable credentials."""
    known_secrets = sorted({
        value for key, value in os.environ.items()
        if len(value) >= 8 and (
            key.upper().endswith(("API_KEY", "TOKEN", "PASSWORD", "SECRET"))
        )
    }, key=len, reverse=True)

    def clean(value):
        """Recursively clean JSON values without mutating the live result."""
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key.casefold() not in OMITTED_KEYS}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for secret in known_secrets:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(
                r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
                "[REDACTED PRIVATE KEY]", value, flags=re.DOTALL,
            )
            value = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
            value = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
            return SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", value)
        return value

    return clean(trace)


def persist_trace(trace: dict, db_path: str | Path | None, *, source: str,
                  repo_path: str | Path) -> dict:
    """Save once, or spool a sanitized snapshot; never retry model or tool execution."""
    if db_path is None:
        return {"status": "disabled"}
    context = {"source": source, "repo_key": os.path.normcase(str(Path(repo_path).resolve()))}
    clean = sanitize_trace(trace)
    try:
        saved = SQLiteTraceStore(db_path).save_trace(clean, context)
        return {"status": "saved" if saved else "unchanged"}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        result = {"status": "failed", "error_type": type(error).__name__, "pending": False}
        try:
            pending_dir = Path(db_path).expanduser().resolve().parent / "trace_pending"
            pending_dir.mkdir(parents=True, exist_ok=True)
            # Random file names avoid interpreting run identifiers as filesystem paths.
            pending_path = pending_dir / f"{uuid4()}.json"
            with pending_path.open("x", encoding="utf-8") as handle:
                json.dump({"trace": clean, "context": context}, handle,
                          ensure_ascii=False, allow_nan=False)
            result["pending"] = True
        except (OSError, ValueError, TypeError):
            pass
        return result


def replay_pending(path: str | Path, db_path: str | Path) -> dict:
    """Replay one pending snapshot without executing code or removing its source file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = SQLiteTraceStore(db_path).save_trace(payload["trace"], payload["context"])
    return {"status": "saved" if saved else "unchanged"}
