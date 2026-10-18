"""Write generated files and archive diagnostic JSON in the configured trace DB."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3

from coding_rag.storage.persistence import configured_trace_db
from coding_rag.storage.sqlite import SQLiteTraceStore


LOGGER = logging.getLogger(__name__)


def archive_json_file(path: str | Path, *, trace_db: str | Path | None = None) -> dict:
    """Best-effort archive; a database failure leaves the generated file usable."""
    try:
        db_path = configured_trace_db(trace_db)
        if db_path is None:
            return {"status": "disabled"}
        return SQLiteTraceStore(db_path).save_json_file(path)
    except (OSError, ValueError, TypeError, sqlite3.Error) as error:
        LOGGER.warning("JSON archive failed (%s); source file retained: %s", type(error).__name__, path)
        return {"status": "failed", "error_type": type(error).__name__}


def write_json_artifact(path: str | Path, payload: object, *, jsonl: bool = False,
                        trace_db: str | Path | None = None) -> None:
    """Keep normal UTF-8 file output and archive a filtered copy in SQLite."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in payload)
    else:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
    target.write_text(content, encoding="utf-8")
    archive_json_file(target, trace_db=trace_db)
