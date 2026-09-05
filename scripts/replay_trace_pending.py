"""Replay local pending trace snapshots into SQLite without rerunning Agent work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coding_rag.storage.persistence import configured_trace_db, replay_pending


def main() -> int:
    """Save each explicitly supplied pending file; retain all source files."""
    parser = argparse.ArgumentParser(description="补录 trace，不执行模型或工具，不删除源文件")
    parser.add_argument("paths", nargs="+", help="一个或多个待补录 JSON 文件")
    parser.add_argument("--trace-db", help="目标 SQLite 路径；默认使用 REPOPILOT_TRACE_DB")
    args = parser.parse_args()
    database = configured_trace_db(args.trace_db)
    if database is None:
        parser.error("trace storage is disabled")
    failed = False
    for path in args.paths:
        try:
            result = replay_pending(path, database)
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
            result = {"status": "failed", "error_type": type(error).__name__}
            failed = True
        print(json.dumps({"path": path, **result}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
