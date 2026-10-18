"""Archive generated JSON/JSONL files without executing tasks or deleting sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding_rag.storage.persistence import configured_trace_db
from coding_rag.storage.sqlite import SQLiteTraceStore
from coding_rag.storage.legacy import import_archived_traces


def main(argv: list[str] | None = None) -> int:
    """Initialize the DB and import each selected file, reporting partial failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="JSON/JSONL files or recursive directories")
    parser.add_argument("--trace-db", help="SQLite database path; defaults to REPOPILOT_TRACE_DB")
    parser.add_argument("--convert-traces", action="store_true",
                        help="also convert archived legacy traces into trace_runs")
    parser.add_argument("--archives-only", action="store_true",
                        help="convert already archived data without reading source files")
    args = parser.parse_args(argv)
    if args.archives_only and (not args.convert_traces or args.paths):
        parser.error("--archives-only requires --convert-traces and accepts no paths")
    counts = {"saved": 0, "unchanged": 0, "failed": 0}
    try:
        db_path = configured_trace_db(args.trace_db)
        if db_path is None:
            raise ValueError("database storage is disabled")
        store = SQLiteTraceStore(db_path)
        store.initialize()
    except (OSError, ValueError, sqlite3.Error) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    seen = set()
    selections = [] if args.archives_only else (args.paths or [PROJECT_ROOT / "artifacts"])
    for selection in selections:
        candidates = sorted(selection.rglob("*")) if selection.is_dir() else [selection]
        for path in candidates:
            if path.is_dir() or (selection.is_dir() and path.suffix.lower() not in {".json", ".jsonl"}):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                result = store.save_json_file(path)
            except (OSError, ValueError, TypeError, sqlite3.Error) as error:
                result = {"status": "failed", "error_type": type(error).__name__}
            counts[result["status"]] += 1
            if result["status"] == "failed":
                print(json.dumps({"path": str(path), **result}, ensure_ascii=True))
    if args.convert_traces:
        try:
            converted = import_archived_traces(store)
        except (OSError, ValueError, sqlite3.Error) as error:
            converted = {"failed": 1, "error_type": type(error).__name__}
        counts["traces"] = converted
    print(json.dumps({"database": str(db_path), **counts}, ensure_ascii=True))
    if counts.get("traces", {}).get("failed"):
        return 1
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
