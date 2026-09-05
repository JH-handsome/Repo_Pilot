"""Persist completed chat turns separately from diagnostic traces and tool state."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import UUID


# Bounded lock pool serializes same-session requests in the local threaded server.
SESSION_LOCKS = tuple(RLock() for _ in range(64))


def conversation_path(explicit=None):
    """Resolve the chat database independently of optional trace persistence."""
    value = explicit or os.environ.get("REPOPILOT_CHAT_DB")
    return Path(value).expanduser().resolve() if value else (
        Path(__file__).resolve().parents[2] / "artifacts" / "conversations.sqlite3"
    )


def session_key(repo_path, session_id):
    """Validate a UUID capability and scope it to the canonical repository."""
    try:
        session = str(UUID(str(session_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("session_id must be a UUID") from exc
    return os.path.normcase(str(Path(repo_path).resolve())), session


class ConversationStore:
    """Small SQLite turn log; never stores graph checkpoints or write approvals."""

    def __init__(self, path, repo_path, session_id):
        """Bind a validated session to a local database and repository."""
        self.path = conversation_path(path)
        self.key = session_key(repo_path, session_id)
        self.lock = SESSION_LOCKS[hash((str(self.path), self.key)) % len(SESSION_LOCKS)]

    def connect(self):
        """Create a short-lived connection with an independent chat schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("""CREATE TABLE IF NOT EXISTS chat_turns (
            repo TEXT NOT NULL, session TEXT NOT NULL, turn INTEGER NOT NULL,
            run_id TEXT NOT NULL, query TEXT NOT NULL, answer TEXT NOT NULL,
            PRIMARY KEY(repo, session, turn), UNIQUE(repo, session, run_id))""")
        return conn

    def load(self):
        """Read complete turns in order, returning their optimistic revision."""
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT query, answer FROM chat_turns WHERE repo=? AND session=? ORDER BY turn",
                self.key,
            ).fetchall()
        return rows, len(rows)

    def append(self, query, answer, run_id, revision):
        """Append once; reject stale state instead of overwriting concurrent history."""
        with closing(self.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM chat_turns WHERE repo=? AND session=? AND run_id=?",
                (*self.key, run_id),
            ).fetchone():
                return
            count = conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE repo=? AND session=?", self.key
            ).fetchone()[0]
            if count != revision:
                raise RuntimeError("Conversation changed; start a new turn before continuing")
            conn.execute("INSERT INTO chat_turns VALUES (?,?,?,?,?,?)",
                         (*self.key, count + 1, run_id, query, answer))


def history_messages(turns, max_chars):
    """Keep the initial context and latest complete turns; report every omission."""
    if max_chars <= 0:
        raise ValueError("max_history_chars must be greater than zero")
    selected = set()
    remaining = max_chars
    # Keep the first turn if it fits, then a contiguous suffix of recent turns.
    order = ([0] + list(range(len(turns) - 1, 0, -1))) if turns else []
    for index in order:
        size = sum(len(text) for text in turns[index])
        if size > remaining:
            if index == 0:
                continue
            break
        selected.add(index)
        remaining -= size
    messages = []
    for index in sorted(selected):
        query, answer = turns[index]
        messages.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps({
                "action": "answer", "reason": "previous turn", "answer": answer,
                "tool": None, "arguments": {}, "expected_observation": None,
            }, ensure_ascii=False)},
        ])
    return messages, len(turns) - len(selected)
