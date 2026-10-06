"""Durable learning store on ``plugin_db("bayes-check")`` (SQLite, WAL).

Tables: ``claims`` (append-only outcome ledger, one row per scored claim), ``evidence`` (source
types per claim, for refit), ``calibration`` (fitted LRs and claim-type base rates), ``events``
(gate skips, refits). The caller owns transactions in ``plugin_db``; every write here commits
before returning so the transform hook's separate connection sees it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from .scorer import LabelledClaim, refit_base_rates, refit_lrs

PLUGIN_NAME = "bayes-check"
MAX_DB_BYTES = 50 * 1024 * 1024
ROTATE_CHUNK = 1000
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_hash TEXT NOT NULL,
    model_claim_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,        -- '' until post_tool_call binds the call to its turn
    call_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    text TEXT NOT NULL,
    prior REAL NOT NULL,
    posterior REAL NOT NULL,
    verdict TEXT NOT NULL,
    outcome INTEGER,              -- NULL = unlabelled, 1 true, 0 false
    labelled_at REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS claims_hash ON claims (claim_hash);
CREATE INDEX IF NOT EXISTS claims_turn ON claims (turn_id);
CREATE INDEX IF NOT EXISTS claims_call ON claims (call_id);
CREATE TABLE IF NOT EXISTS evidence (
    claim_row INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    stance TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_claim ON evidence (claim_row);
CREATE TABLE IF NOT EXISTS calibration (
    kind TEXT NOT NULL,           -- 'lr' (per source_type) | 'base_rate' (per claim_type)
    key TEXT NOT NULL,
    value TEXT NOT NULL,          -- JSON
    updated_at REAL NOT NULL,
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    session_id TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _default_opener() -> sqlite3.Connection:
    from plugins.plugin_storage import plugin_db
    return plugin_db(PLUGIN_NAME)


class Store:
    """Thin, thread-safe facade; one connection per Store, opened lazily."""

    def __init__(self, opener: Optional[Callable[[], sqlite3.Connection]] = None,
                 max_db_bytes: int = MAX_DB_BYTES) -> None:
        self._opener = opener or _default_opener
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self.max_db_bytes = max_db_bytes

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = self._opener()
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                         (str(SCHEMA_VERSION),))
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def db_path(self) -> Optional[Path]:
        row = self._connect().execute("PRAGMA database_list").fetchone()
        return Path(row[2]) if row and row[2] else None

    # --- writes ---------------------------------------------------------------------------------

    def append_claims(self, session_id: str, call_id: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """Append scored claims (+ their evidence types); commits; rotates when over budget."""
        now = time.time()
        with self._tx() as conn:
            for r in rows:
                cur = conn.execute(
                    "INSERT INTO claims (claim_hash, model_claim_id, session_id, turn_id, call_id, claim_type, text,"
                    " prior, posterior, verdict, created_at) VALUES (?,?,?,'',?,?,?,?,?,?,?)",
                    (r["hash"], r["id"], session_id, call_id, r["claim_type"], r["text"],
                     r["prior"], r["posterior"], r["verdict"], now))
                conn.executemany(
                    "INSERT INTO evidence (claim_row, source_type, stance) VALUES (?,?,?)",
                    [(cur.lastrowid, e["source_type"], e["stance"]) for e in r.get("evidence", [])])
        self.rotate_if_needed()
        return len(rows)

    def bind_turn(self, call_id: str, turn_id: str) -> int:
        with self._tx() as conn:
            return conn.execute("UPDATE claims SET turn_id = ? WHERE call_id = ?", (turn_id, call_id)).rowcount

    def label(self, claim_ref: str, outcome: bool, *, session_id: str = "", turn_id: str = "") -> int:
        """Label the newest claim matching a hash prefix (>= 6 hex) or, with ``turn_id``, a model claim id."""
        ref = claim_ref.strip()
        with self._tx() as conn:
            row = None
            if turn_id:
                row = conn.execute("SELECT id FROM claims WHERE turn_id = ? AND model_claim_id = ?"
                                   " ORDER BY id DESC LIMIT 1", (turn_id, ref)).fetchone()
            if row is None and len(ref) >= 6 and all(c in "0123456789abcdef" for c in ref.lower()):
                row = conn.execute("SELECT id FROM claims WHERE claim_hash LIKE ? ORDER BY id DESC LIMIT 1",
                                   (ref.lower() + "%",)).fetchone()
            if row is None:
                return 0
            conn.execute("UPDATE claims SET outcome = ?, labelled_at = ? WHERE id = ?",
                         (1 if outcome else 0, time.time(), row[0]))
            conn.execute("INSERT INTO events (kind, session_id, detail, created_at) VALUES (?,?,?,?)",
                         ("label", session_id, json.dumps({"claim_row": row[0], "outcome": outcome}), time.time()))
        return 1

    def record_event(self, kind: str, session_id: str, detail: Mapping[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO events (kind, session_id, detail, created_at) VALUES (?,?,?,?)",
                         (kind, session_id, json.dumps(dict(detail)), time.time()))

    # --- calibration ----------------------------------------------------------------------------

    def labelled_claims(self) -> List[LabelledClaim]:
        conn = self._connect()
        with self._lock:
            rows = conn.execute("SELECT id, claim_type, outcome FROM claims WHERE outcome IS NOT NULL").fetchall()
            ev: Dict[int, List[str]] = {}
            for claim_row, st in conn.execute(
                    "SELECT e.claim_row, e.source_type FROM evidence e JOIN claims c ON c.id = e.claim_row"
                    " WHERE c.outcome IS NOT NULL AND e.stance = 'supports'"):
                ev.setdefault(claim_row, []).append(st)
        return [LabelledClaim(ct, bool(o), tuple(ev.get(i, ()))) for i, ct, o in rows]

    def refit(self) -> Dict[str, Any]:
        """CPU-only refit over labelled outcomes; persists fitted LRs and base rates."""
        labelled = self.labelled_claims()
        lrs, base = refit_lrs(labelled), refit_base_rates(labelled)
        now = time.time()
        with self._tx() as conn:
            for kind, table in (("lr", lrs), ("base_rate", base)):
                for key, value in table.items():
                    conn.execute("INSERT OR REPLACE INTO calibration (kind, key, value, updated_at) VALUES (?,?,?,?)",
                                 (kind, key, json.dumps(value), now))
            conn.execute("INSERT INTO events (kind, session_id, detail, created_at) VALUES (?,?,?,?)",
                         ("refit", "", json.dumps({"labelled": len(labelled), "lrs": len(lrs)}), now))
        return {"labelled": len(labelled), "lrs": lrs, "base_rates": base}

    def calibration(self, kind: str) -> Dict[str, Dict[str, float]]:
        conn = self._connect()
        with self._lock:
            rows = conn.execute("SELECT key, value FROM calibration WHERE kind = ?", (kind,)).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def lr_table(self, defaults: Mapping[str, float]) -> Dict[str, float]:
        table = dict(defaults)
        for key, fitted in self.calibration("lr").items():
            if key in table:   # fitted values only recalibrate known types
                table[key] = float(fitted["lr"])
        return table

    def base_rate(self, claim_type: str) -> Optional[float]:
        fitted = self.calibration("base_rate").get(claim_type)
        return float(fitted["mean"]) if fitted else None

    # --- stats / rotation -----------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        conn = self._connect()
        with self._lock:
            total, labelled = conn.execute(
                "SELECT COUNT(*), COUNT(outcome) FROM claims").fetchone()
            skips = conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'gate_skipped'").fetchone()[0]
        return {"claims": int(total), "labelled": int(labelled), "gate_skips": int(skips)}

    def _db_bytes(self) -> int:
        path = self.db_path()
        if path is None:
            return 0
        return sum(os.path.getsize(p) for p in (path, Path(f"{path}-wal")) if p.exists())

    def rotate_if_needed(self) -> int:
        """Delete the oldest UNLABELLED claims in chunks until under budget (labels are the signal)."""
        removed = 0
        with self._lock:
            conn = self._connect()
            while self._db_bytes() > self.max_db_bytes:
                cur = conn.execute(
                    "DELETE FROM claims WHERE id IN (SELECT id FROM claims WHERE outcome IS NULL"
                    " ORDER BY id LIMIT ?)", (ROTATE_CHUNK,))
                conn.execute("DELETE FROM evidence WHERE claim_row NOT IN (SELECT id FROM claims)")
                conn.commit()
                if cur.rowcount <= 0:
                    break
                removed += cur.rowcount
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("VACUUM")
        return removed
