"""Transactional SQLite trace snapshots and searchable event summaries."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from time import monotonic, sleep
from typing import Any, Iterator, Protocol


SCHEMA_VERSION = 1
SCHEMA = (
    """CREATE TABLE trace_runs (
        run_id TEXT PRIMARY KEY, session_id TEXT, source TEXT NOT NULL,
        repo_key TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
        query TEXT, started_at TEXT, updated_at TEXT NOT NULL, finished_at TEXT,
        trace_version TEXT NOT NULL, revision INTEGER NOT NULL CHECK (revision > 0),
        trace_json TEXT NOT NULL
    )""",
    """CREATE TABLE trace_events (
        run_id TEXT NOT NULL REFERENCES trace_runs(run_id), seq INTEGER NOT NULL,
        step TEXT NOT NULL, status TEXT NOT NULL, duration_ms INTEGER,
        occurred_at TEXT, error_type TEXT, summary_json TEXT NOT NULL,
        PRIMARY KEY (run_id, seq)
    )""",
    "CREATE INDEX trace_repo_time ON trace_runs(repo_key, updated_at, run_id)",
    "CREATE INDEX trace_status_time ON trace_runs(status, updated_at, run_id)",
    "CREATE INDEX trace_time ON trace_runs(updated_at, run_id)",
    "CREATE INDEX trace_session ON trace_runs(session_id)",
    "CREATE INDEX trace_event_status ON trace_events(step, status)",
)


class TraceSchemaError(ValueError):
    """The configured database is not compatible with this trace store."""


def utc_now() -> str:
    """Return an unambiguous UTC timestamp for new records."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class TraceStore(Protocol):
    """Backend-independent contract for sanitized diagnostic snapshots."""

    def save_trace(self, trace: dict, context: dict) -> bool:
        """Atomically save a newer snapshot; return false for an old revision."""
        ...

    def get_trace(self, run_id: str) -> dict | None:
        """Return the latest trace or none when it does not exist."""
        ...

    def list_traces(self, filters: dict | None = None, cursor: str | None = None,
                    limit: int = 50) -> dict:
        """Return a bounded page of run summaries and an optional next cursor."""
        ...


class SQLiteTraceStore:
    """Use one short-lived connection per operation, safe across HTTP threads."""

    def __init__(self, path: str | Path, *, timeout: float = 2.0):
        """Bind a local database file without creating it until the first save."""
        self.path = Path(path).expanduser().resolve()
        self.timeout = timeout

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Open and close a connection, checking schema before accessing tables."""
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = self.path.as_uri() + ("?mode=rwc" if write else "?mode=ro")
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                # Read connections never mutate schema; migrations are serialized.
                self._enable_wal(connection)
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0 and write:
                existing = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()
                if existing:
                    raise TraceSchemaError("refusing to initialize a nonempty unversioned database")
                for statement in SCHEMA:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION:
                raise TraceSchemaError("unsupported trace database schema version")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _enable_wal(self, connection: sqlite3.Connection) -> None:
        """Bound retries for the first WAL transition, which can fail without waiting."""
        deadline = monotonic() + self.timeout
        while True:
            try:
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode != "wal":
                    raise ValueError("trace database requires WAL journal mode")
                return
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", 0) & 0xFF
                if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or monotonic() >= deadline:
                    raise
                sleep(min(0.02, max(0, deadline - monotonic())))

    def save_trace(self, trace: dict, context: dict) -> bool:
        """Save a cumulative snapshot and only append events absent from the store."""
        run = trace["run"]
        run_id = run["run_id"]
        revision = run["revision"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        if type(revision) is not int or revision < 1:
            raise ValueError("revision must be a positive integer")
        events = trace.get("events", [])
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        encoded = json.dumps(trace, ensure_ascii=False, allow_nan=False)
        source, repo_key = context["source"], context["repo_key"]
        with self._connection(write=True) as connection:
            old = connection.execute(
                "SELECT revision, trace_json, source, repo_key FROM trace_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            previous_count = 0
            if old:
                if old["source"] != source or old["repo_key"] != repo_key:
                    raise ValueError("run identity cannot change")
                old_trace = json.loads(old["trace_json"])
                if revision == old["revision"] and trace != old_trace:
                    raise ValueError("conflicting snapshots have the same revision")
                if revision <= old["revision"]:
                    return False
                if any(run.get(key) != old_trace["run"].get(key)
                       for key in ("mode", "query", "started_at")):
                    raise ValueError("run metadata cannot change")
                previous = old_trace.get("events", [])
                previous_count = len(previous)
                if events[:previous_count] != previous:
                    raise ValueError("a newer snapshot must preserve existing events")
            values = (
                run_id, context.get("session_id"), source, repo_key, run["mode"],
                run["status"], run.get("query"), run.get("started_at"),
                run.get("updated_at") or utc_now(), run.get("finished_at"),
                str(trace.get("trace_version", "1.0")), revision, encoded,
            )
            connection.execute(
                """INSERT INTO trace_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status, updated_at=excluded.updated_at,
                    finished_at=excluded.finished_at, revision=excluded.revision,
                    trace_version=excluded.trace_version, trace_json=excluded.trace_json""",
                values,
            )
            for seq, event in enumerate(events[previous_count:], start=previous_count + 1):
                error = event.get("error") or {}
                connection.execute(
                    "INSERT INTO trace_events VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, seq, event["step"], event.get("status", "success"),
                     event.get("duration_ms"), event.get("occurred_at"), error.get("type"),
                     json.dumps(event.get("output_summary") or {}, ensure_ascii=False,
                                allow_nan=False)),
                )
        return True

    def get_trace(self, run_id: str) -> dict | None:
        """Read a complete sanitized trace without returning any execution state."""
        if not self.path.exists():
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT trace_json FROM trace_runs WHERE run_id=?", (run_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_traces(self, filters: dict | None = None, cursor: str | None = None,
                    limit: int = 50) -> dict:
        """Filter and paginate summaries using an opaque timestamp/run ID cursor."""
        import base64

        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        clauses, values = [], []
        for field, value in (filters or {}).items():
            if field in {"repo_key", "status", "mode", "source", "session_id"}:
                clauses.append(f"{field}=?")
            elif field in {"since", "until"}:
                operator = ">=" if field == "since" else "<="
                clauses.append(f"updated_at {operator} ?")
            else:
                raise ValueError(f"unknown trace filter: {field}")
            values.append(value)
        if cursor:
            try:
                if len(cursor) > 2048:
                    raise ValueError("cursor too long")
                decoded = json.loads(base64.urlsafe_b64decode(cursor).decode("utf-8"))
                if not isinstance(decoded, list) or len(decoded) != 2:
                    raise ValueError("invalid cursor shape")
                timestamp, run_id = decoded
                if not all(isinstance(item, str) for item in decoded):
                    raise ValueError("invalid cursor values")
            except (ValueError, UnicodeError, TypeError) as error:
                raise ValueError("invalid trace cursor") from error
            clauses.append("(updated_at < ? OR (updated_at = ? AND run_id < ?))")
            values.extend((timestamp, timestamp, run_id))
        if not self.path.exists():
            return {"items": [], "next_cursor": None}
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id,session_id,source,repo_key,mode,status,query,started_at,"
                "updated_at,finished_at,trace_version,revision FROM trace_runs" + where
                + " ORDER BY updated_at DESC,run_id DESC LIMIT ?", (*values, limit + 1),
            ).fetchall()
        items = [dict(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            last = items[-1]
            next_cursor = base64.urlsafe_b64encode(json.dumps(
                [last["updated_at"], last["run_id"]],
            ).encode()).decode()
        return {"items": items, "next_cursor": next_cursor}
