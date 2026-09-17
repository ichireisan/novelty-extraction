"""SQLite beliefs, bounded deferred observations, and an append-only decision audit."""

import json
import sqlite3
from pathlib import Path

from .types import Assessment, Experience, Status


class Memory:
    def __init__(self, path: str | Path = ":memory:", pending_limit: int = 128):
        if pending_limit < 0:
            raise ValueError("pending_limit must be nonnegative")
        self.pending_limit = pending_limit
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                claim_key TEXT NOT NULL,
                value TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_key ON memories(claim_key, status);
            CREATE TABLE IF NOT EXISTS pending (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def answer(self, key: str) -> str | None:
        row = self.db.execute(
            "SELECT value FROM memories WHERE claim_key=? AND status='verified' "
            "ORDER BY sequence DESC LIMIT 1", (key,)
        ).fetchone()
        return row[0] if row else None

    def retrieve(self, event: Experience, limit: int = 8) -> list[dict]:
        """Exact claim-key matches first, then recency; intentionally not semantic search."""
        key = event.claim.key if event.claim else ""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE status != 'superseded' "
            "ORDER BY (claim_key=?) DESC, sequence DESC LIMIT ?", (key, limit)
        ).fetchall()
        return [dict(row) for row in rows]

    def remember(self, event: Experience, assessment: Assessment) -> bool:
        if event.claim is None or assessment.status == Status.REJECTED:
            return False
        previous = self.db.execute(
            "SELECT * FROM memories WHERE event_id=?", (event.event_id,)
        ).fetchone()
        if previous:
            if (previous["claim_key"], previous["value"], previous["text"], previous["source"]) != (
                event.claim.key, event.claim.value, event.text, event.source
            ):
                raise ValueError("event_id was reused for a different experience")
            if previous["status"] in ("verified", "superseded") or assessment.status != Status.VERIFIED:
                return False
        with self.db:
            if assessment.status == Status.VERIFIED:
                self.db.execute(
                    "UPDATE memories SET status='superseded' "
                    "WHERE claim_key=? AND value!=? AND status='verified'",
                    (event.claim.key, event.claim.value),
                )
            self.db.execute(
                "INSERT INTO memories(event_id,claim_key,value,text,source,status,evidence,reason) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(event_id) DO UPDATE SET "
                "status=excluded.status,evidence=excluded.evidence,reason=excluded.reason",
                (event.event_id, event.claim.key, event.claim.value, event.text, event.source,
                 assessment.status.value, json.dumps(assessment.evidence_refs), assessment.reason),
            )
            self.db.execute("DELETE FROM pending WHERE event_id=?", (event.event_id,))
        return True

    def defer(self, event: Experience):
        with self.db:
            self.db.execute(
                "INSERT INTO pending(event_id,payload) VALUES(?,?) "
                "ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload",
                (event.event_id, json.dumps(event.to_dict())),
            )
            self.db.execute(
                "DELETE FROM pending WHERE sequence NOT IN "
                "(SELECT sequence FROM pending ORDER BY sequence DESC LIMIT ?)",
                (self.pending_limit,),
            )

    def deferred(self) -> list[Experience]:
        return [Experience.from_dict(json.loads(row[0])) for row in
                self.db.execute("SELECT payload FROM pending ORDER BY sequence")]

    def log(self, data: dict):
        with self.db:
            self.db.execute("INSERT INTO audit(payload) VALUES(?)", (json.dumps(data),))

    def records(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM memories ORDER BY sequence")]

    def clone(self) -> "Memory":
        result = Memory(pending_limit=self.pending_limit)
        self.db.backup(result.db)
        return result
