"""Transactional SQLite trace snapshots and searchable event summaries."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from hashlib import sha256
from pathlib import Path
import sqlite3
from time import monotonic, sleep
from typing import Any, Iterator, Protocol

from coding_rag.rag.llm_client import normalize_llm_call_observation


SCHEMA_VERSION = 3
JSON_FILE_SCHEMA = (
    """CREATE TABLE json_files (
        file_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
        format TEXT NOT NULL CHECK (format IN ('json', 'jsonl')),
        sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
        record_count INTEGER NOT NULL, saved_at TEXT NOT NULL,
        content TEXT NOT NULL,
        UNIQUE (source_path, sha256)
    )""",
    "CREATE INDEX json_file_path_time ON json_files(source_path, saved_at)",
)
MAX_LLM_CALLS_PER_RUN = 100
LLM_CALL_SCHEMA = (
    """CREATE TABLE llm_calls (
        run_id TEXT NOT NULL REFERENCES trace_runs(run_id) ON DELETE CASCADE,
        call_index INTEGER NOT NULL CHECK (call_index BETWEEN 1 AND 100),
        event_seq INTEGER NOT NULL CHECK (event_seq > 0),
        operation TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
        attempts INTEGER NOT NULL CHECK (attempts BETWEEN 1 AND 5),
        latency_ms REAL NOT NULL CHECK (latency_ms BETWEEN 0 AND 86400000),
        outcome TEXT NOT NULL CHECK (outcome IN ('success', 'error')),
        error_class TEXT,
        PRIMARY KEY (run_id, call_index), UNIQUE (run_id, event_seq)
    )""",
    "CREATE INDEX llm_call_outcome ON llm_calls(run_id, outcome)",
)
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
    *LLM_CALL_SCHEMA,
    *JSON_FILE_SCHEMA,
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
            elif version == 1 and write:
                for statement in LLM_CALL_SCHEMA:
                    connection.execute(statement)
                for statement in JSON_FILE_SCHEMA:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version == 2 and write:
                for statement in JSON_FILE_SCHEMA:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version in (1, 2) and not write:
                # Legacy trace tables remain readable; the next write migrates atomically.
                pass
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

    def initialize(self) -> None:
        """Create or migrate a database without inserting synthetic trace records."""
        with self._connection(write=True):
            pass

    def save_json_file(self, path: str | Path) -> dict:
        """Archive sanitized JSON/JSONL content, retaining distinct file versions."""
        from coding_rag.storage.persistence import sanitize_trace

        source = Path(path).expanduser().resolve()
        file_format = source.suffix.lower().lstrip(".")
        if file_format not in {"json", "jsonl"}:
            raise ValueError("only JSON and JSONL files can be archived")
        raw = source.read_text(encoding="utf-8-sig")
        if file_format == "json":
            content = json.dumps(sanitize_trace(json.loads(raw)), ensure_ascii=False,
                                 allow_nan=False, indent=2)
            record_count = 1
        else:
            records = [sanitize_trace(json.loads(line)) for line in raw.splitlines() if line.strip()]
            content = "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in records)
            record_count = len(records)
        encoded = content.encode("utf-8")
        digest = sha256(encoded).hexdigest()
        file_id = sha256((str(source) + "\0" + digest).encode("utf-8")).hexdigest()
        with self._connection(write=True) as connection:
            result = connection.execute(
                """INSERT INTO json_files
                (file_id,source_path,format,sha256,size_bytes,record_count,saved_at,content)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source_path,sha256) DO NOTHING""",
                (file_id, str(source), file_format, digest, len(encoded), record_count, utc_now(), content),
            )
            saved = result.rowcount == 1
        return {"status": "saved" if saved else "unchanged", "file_id": file_id}

    def get_json_file(self, file_id: str) -> dict | None:
        """Return file metadata and archived UTF-8 content for local export."""
        if not self.path.exists():
            return None
        with self._connection() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] < 3:
                return None
            row = connection.execute("SELECT * FROM json_files WHERE file_id=?", (file_id,)).fetchone()
        return dict(row) if row else None

    def iter_json_files(self) -> Iterator[dict]:
        """Stream a stable archive snapshot while WAL allows trace writes."""
        if not self.path.exists():
            return
        with self._connection() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] < 3:
                return
            for row in connection.execute("SELECT * FROM json_files ORDER BY saved_at,file_id"):
                yield dict(row)

    def get_trace_context(self, run_id: str) -> dict | None:
        """Read immutable identity for importing an exported existing run."""
        if not self.path.exists():
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT source,repo_key,session_id FROM trace_runs WHERE run_id=?", (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_json_files(self, *, source_path: str | Path | None = None,
                        limit: int = 50, offset: int = 0) -> list[dict]:
        """List bounded file metadata, optionally restricting to one source path."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if not self.path.exists():
            return []
        where, values = "", []
        if source_path is not None:
            where = " WHERE source_path=?"
            values.append(str(Path(source_path).expanduser().resolve()))
        with self._connection() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] < 3:
                return []
            rows = connection.execute(
                "SELECT file_id,source_path,format,sha256,size_bytes,record_count,saved_at "
                "FROM json_files" + where + " ORDER BY saved_at DESC,file_id DESC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

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
        call_records = _llm_call_records(events)
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
            previous_call_count = connection.execute(
                "SELECT count(*) FROM llm_calls WHERE run_id=?", (run_id,),
            ).fetchone()[0]
            for record in call_records:
                if record["event_seq"] <= previous_count:
                    continue
                previous_call_count += 1
                connection.execute(
                    """INSERT INTO llm_calls (
                        run_id,call_index,event_seq,operation,provider,model,attempts,
                        latency_ms,outcome,error_class
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id, previous_call_count, record["event_seq"],
                        record["operation"], record["provider"], record["model"],
                        record["attempts"], record["latency_ms"], record["outcome"],
                        record["error_class"],
                    ),
                )
        return True

    def get_llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        """Return bounded sanitized logical-call records for one run."""
        _validate_run_id(run_id)
        if not self.path.exists():
            return []
        with self._connection() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] == 1:
                return []
            rows = connection.execute(
                """SELECT call_index,event_seq,operation,provider,model,attempts,
                          latency_ms,outcome,error_class
                   FROM llm_calls WHERE run_id=? ORDER BY call_index""",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_llm_summary(self, run_id: str) -> dict[str, Any]:
        """Aggregate attempts, retries, latency, outcomes and errors for one run."""
        calls = self.get_llm_calls(run_id)
        outcomes: dict[str, int] = {}
        error_classes: dict[str, int] = {}
        for item in calls:
            outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
            if item["error_class"] is not None:
                name = item["error_class"]
                error_classes[name] = error_classes.get(name, 0) + 1
        latency = round(sum(item["latency_ms"] for item in calls), 3)
        return {
            "call_count": len(calls),
            "attempt_count": sum(item["attempts"] for item in calls),
            "retry_count": sum(item["attempts"] - 1 for item in calls),
            "outcomes": outcomes,
            "latency_ms_total": latency,
            "latency_ms_avg": round(latency / len(calls), 3) if calls else 0.0,
            "error_classes": error_classes,
        }

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


def _validate_run_id(run_id: object) -> None:
    """Reject malformed identifiers before using them in storage queries."""
    if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
        raise ValueError("run_id must be a nonempty string of at most 256 characters")


def _llm_call_records(events: list[object]) -> list[dict[str, Any]]:
    """Extract and strictly validate fixed-schema call evidence from trace events."""
    records: list[dict[str, Any]] = []
    for event_seq, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        artifacts = event.get("artifacts") or {}
        if not isinstance(artifacts, dict) or "llm_call" not in artifacts:
            continue
        if event.get("step") != "model_decision":
            raise ValueError("llm_call observation must belong to model_decision")
        observation = normalize_llm_call_observation(artifacts["llm_call"])
        if observation is None:
            raise ValueError("invalid llm_call observation")
        records.append({"event_seq": event_seq, **observation})
        if len(records) > MAX_LLM_CALLS_PER_RUN:
            raise ValueError("too many llm_call observations in one run")
    return records
