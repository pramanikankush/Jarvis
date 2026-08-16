"""SQLite persistence (documents, chunks+embeddings, chat sessions) + search.

Everything lives in data/app.db — free, local, survives restarts.

Three search signals live here:
  * vector: numpy cosine similarity over stored embeddings (see `search`)
  * keyword: SQLite FTS5 BM25 (see `keyword_search`) — the hybrid partner
  * memory: durable user facts + per-session summary/task (see memory/session_meta)
"""
import json
import os
import re
import sqlite3
import threading
import time
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    chunks INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    doc_name TEXT NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    emb BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_meta (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    fact TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    rows INTEGER NOT NULL DEFAULT 0,
    cols INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_fact ON memory(fact);
"""

# FTS5 needs the virtual table + backfill to be created after the plain tables.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(doc_id UNINDEXED, doc_name, page UNINDEXED, text);
"""


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: quoted alnum tokens, OR-joined.

    FTS5 throws on syntax errors / special characters, so every token is
    sanitised to [A-Za-z0-9_] and double-quoted. Short tokens (<2 chars) are
    dropped — they add noise and cost BM25 ranking quality.
    """
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:24])


class Store:
    """SQLite persistence. A single instance is shared across requests; use
    `for_user(user_id)` to get a per-user scoped view of the same database.

    user_id "" is the legacy/local workspace — existing rows keep working.
    """

    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.executescript(FTS_SCHEMA)
        self._migrate()
        self._uid = ""  # "" = local/legacy workspace (current single-user behaviour)
        self._shared = {"version": 0, "cache": None}  # (version, uid, rows, matrix)
        self.reindex_fts()  # keeps FTS in sync with pre-existing dbs / crashes

    def for_user(self, user_id: str):
        """Return a view of this store scoped to one user. Every query is
        filtered by user_id; rows created through the view carry it. The view
        shares the connection + lock + search cache so it stays consistent
        with the base store and other views (multi-user on one DB)."""
        view = object.__new__(Store)
        view._lock = self._lock
        view._conn = self._conn
        view._uid = user_id or ""
        view._shared = self._shared
        return view

    def _migrate(self):
        """In-place schema upgrades for databases created before this version.
        New columns are added via ALTER TABLE; missing tables are handled by
        SCHEMA's CREATE TABLE IF NOT EXISTS."""
        try:
            cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)").fetchall()]
            if "pinned" not in cols:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
            for table, col in (("docs", "user_id"), ("sessions", "user_id"),
                               ("memory", "user_id"), ("sheets", "user_id")):
                tcols = [r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if col not in tcols:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )
            # user_id indexes live here (not in SCHEMA) so old databases get
            # them only after the column has been added
            for table in ("docs", "sessions", "memory", "sheets"):
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_user ON {table}(user_id)"
                )
            self._conn.commit()
        except sqlite3.Error:
            pass

    # ---------------- docs ----------------
    def add_doc(self, name: str, size: int, pages_chunks: list[tuple[int | None, str]], embeddings: list[np.ndarray]) -> dict:
        """pages_chunks: [(page, chunk_text)] parallel to embeddings (list of float32 vecs)."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO docs(user_id, name, size, chunks, uploaded_at) VALUES(?,?,?,?,?)",
                (self._uid, name, size, len(pages_chunks), time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            doc_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO chunks(doc_id, doc_name, page, text, emb) VALUES(?,?,?,?,?)",
                [
                    (doc_id, name, page, text, np.asarray(emb, dtype="<f4").tobytes())
                    for (page, text), emb in zip(pages_chunks, embeddings)
                ],
            )
            self._conn.executemany(
                "INSERT INTO chunks_fts(doc_id, doc_name, page, text) VALUES(?,?,?,?)",
                [(doc_id, name, page, text) for page, text in pages_chunks],
            )
            self._conn.commit()
            self._shared["version"] += 1
        return self.get_doc(doc_id)

    def get_doc(self, doc_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE id=? AND user_id=?", (doc_id, self._uid)
        ).fetchone()
        return dict(row) if row else None

    def list_docs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM docs WHERE user_id=? ORDER BY uploaded_at DESC", (self._uid,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_doc(self, doc_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM docs WHERE id=? AND user_id=?", (doc_id, self._uid)
            )
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))
            self._conn.commit()
            if cur.rowcount:
                self._shared["version"] += 1
                return True
            return False

    # ---------------- FTS5 keyword search ----------------
    def reindex_fts(self) -> None:
        """Bring chunks_fts in sync with chunks; no-op when counts already match."""
        try:
            with self._lock:
                n_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                n_fts = self._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                if n_chunks == n_fts:
                    return
                self._conn.execute("DELETE FROM chunks_fts")
                self._conn.execute(
                    "INSERT INTO chunks_fts(doc_id, doc_name, page, text) "
                    "SELECT id, doc_name, page, text FROM chunks"
                )
                self._conn.commit()
        except sqlite3.Error:
            # FTS5 unavailable (e.g. some embedded builds) — hybrid degrades to vector-only.
            pass

    def keyword_search(self, query: str, k: int = 8) -> list[dict]:
        """BM25 keyword search over chunks, scoped to this view's user.
        Returns [] on empty/sanitised query or FTS errors."""
        match = fts_query(query)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                "SELECT f.doc_id, f.doc_name, f.page, f.text, bm25(chunks_fts) AS bm25 "
                "FROM chunks_fts f JOIN docs d ON d.id = f.doc_id "
                "WHERE chunks_fts MATCH ? AND d.user_id = ? "
                "ORDER BY bm25 LIMIT ?",
                (match, self._uid, k),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "doc_id": r["doc_id"],
                "doc_name": r["doc_name"],
                "page": r["page"],
                "text": r["text"],
                "bm25": float(r["bm25"]),
            }
            for r in rows
        ]

    # ---------------- memory (long-term user facts) ----------------
    def add_memory(self, fact: str, kind: str = "fact", session_id: str | None = None) -> dict | None:
        """Insert a fact, de-duplicated by exact normalized text. Returns row or None if dup."""
        norm = re.sub(r"\s+", " ", fact.strip().lower())
        if not norm:
            return None
        row = self._conn.execute(
            "SELECT id FROM memory WHERE fact=? AND user_id=?", (norm, self._uid)
        ).fetchone()
        if row:
            return None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memory(user_id, fact, kind, session_id, created_at) VALUES(?,?,?,?,?)",
                (self._uid, norm, kind, session_id, time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            self._conn.commit()
            return {"id": cur.lastrowid, "fact": norm, "kind": kind}

    def list_memory(self, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, fact, kind, created_at FROM memory WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self._uid, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def recall_memory(self, query: str, limit: int = 5) -> list[dict]:
        """Cheap relevance recall: overlap between query tokens and fact tokens."""
        toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for m in self.list_memory(limit=500):
            ftoks = set(re.findall(r"[a-z0-9]+", m["fact"]))
            overlap = len(toks & ftoks)
            if overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:limit]]

    def update_memory(self, mid: int, fact: str, kind: str | None = None) -> bool:
        """Replace a stored fact's text (and optionally its kind). False if missing."""
        norm = re.sub(r"\s+", " ", (fact or "").strip())
        if not norm:
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memory SET fact=?, kind=COALESCE(?, kind) WHERE id=? AND user_id=?",
                (norm, kind, mid, self._uid),
            )
            self._conn.commit()
            return bool(cur.rowcount)

    def delete_memory(self, mid: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memory WHERE id=? AND user_id=?", (mid, self._uid)
            )
            self._conn.commit()
            return bool(cur.rowcount)

    # ---------------- session_meta (summary + current task context) ----------------
    def get_meta(self, sid: str) -> dict:
        row = self._conn.execute(
            "SELECT m.summary, m.task FROM session_meta m "
            "JOIN sessions s ON s.id = m.session_id AND s.user_id = ? "
            "WHERE m.session_id=?", (self._uid, sid)
        ).fetchone()
        return {"summary": row["summary"] if row else "", "task": row["task"] if row else ""}

    def set_meta(self, sid: str, summary: str | None = None, task: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_meta(session_id, summary, task, updated_at) "
                "SELECT ?, ?, ?, ? WHERE EXISTS "
                "(SELECT 1 FROM sessions WHERE id = ? AND user_id = ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "summary=COALESCE(?, summary), task=COALESCE(?, task), updated_at=?",
                (sid, summary or "", task or "", time.strftime("%Y-%m-%d %H:%M:%S"),
                 sid, self._uid, summary, task, time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            self._conn.commit()

    # ---------------- sheets (spreadsheets, file stored under data/sheets/) ----------------
    def add_sheet(self, name: str, kind: str, rows: int, cols: int) -> dict:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sheets(user_id, name, kind, rows, cols, uploaded_at) VALUES(?,?,?,?,?,?)",
                (self._uid, name, kind, rows, cols, time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            self._conn.commit()
            sid = cur.lastrowid
        return self.get_sheet(sid)

    def get_sheet(self, sid: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sheets WHERE id=? AND user_id=?", (sid, self._uid)
        ).fetchone()
        return dict(row) if row else None

    def update_sheet_rows(self, sid: int, rows: int, cols: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sheets SET rows=?, cols=? WHERE id=? AND user_id=?",
                (rows, cols, sid, self._uid),
            )
            self._conn.commit()

    def list_sheets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sheets WHERE user_id=? ORDER BY uploaded_at DESC", (self._uid,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_sheet(self, sid: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sheets WHERE id=? AND user_id=?", (sid, self._uid)
            )
            self._conn.commit()
            return bool(cur.rowcount)

    # ---------------- sessions/messages ----------------
    def create_session(self, title: str) -> dict:
        sid = uuid.uuid4().hex
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(id, user_id, title, created_at, updated_at, pinned) VALUES(?,?,?,?,?,0)",
                (sid, self._uid, title, now, now),
            )
            self._conn.commit()
        return {"id": sid, "title": title, "created_at": now, "updated_at": now, "pinned": 0}

    def set_title(self, sid: str, title: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=? AND user_id=?",
                (title, time.strftime("%Y-%m-%d %H:%M:%S"), sid, self._uid),
            )
            self._conn.commit()

    def set_pinned(self, sid: str, pinned: bool) -> bool:
        """Pin/unpin a session. Returns False if the session does not exist."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET pinned=?, updated_at=? WHERE id=? AND user_id=?",
                (1 if pinned else 0, time.strftime("%Y-%m-%d %H:%M:%S"), sid, self._uid),
            )
            self._conn.commit()
            return bool(cur.rowcount)

    def search_sessions(self, q: str, limit: int = 50) -> list[dict]:
        """Case-insensitive title match on chat sessions (client-side filter also
        exists; this is the server-side counterpart for API users)."""
        q = (q or "").strip()
        if not q:
            return self.list_sessions(limit)
        like = f"%{q}%"
        rows = self._conn.execute(
            "SELECT s.id, s.title, s.updated_at, s.pinned, "
            "(SELECT content FROM messages m WHERE m.session_id=s.id ORDER BY m.id DESC LIMIT 1) AS last "
            "FROM sessions s WHERE s.title LIKE ? COLLATE NOCASE AND s.user_id = ? "
            "ORDER BY s.pinned DESC, s.updated_at DESC LIMIT ?",
            (like, self._uid, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.id, s.title, s.updated_at, s.pinned, "
            "(SELECT content FROM messages m WHERE m.session_id=s.id ORDER BY m.id DESC LIMIT 1) AS last "
            "FROM sessions s WHERE s.user_id = ? ORDER BY s.pinned DESC, s.updated_at DESC LIMIT ?",
            (self._uid, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def export_session(self, sid: str) -> dict | None:
        """Full conversation export (title, meta, ordered messages). None if missing."""
        s = self.get_session(sid)
        if not s:
            return None
        return {
            "id": sid,
            "title": s["title"],
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
            "pinned": bool(s.get("pinned")),
            "meta": self.get_meta(sid),
            "messages": self.messages(sid),
        }

    def get_session(self, sid: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id=? AND user_id=?", (sid, self._uid)
        ).fetchone()
        return dict(row) if row else None

    def add_message(self, sid: str, role: str, content: str, sources: list | None = None) -> None:
        import json

        with self._lock:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            # only write when the session belongs to this user (defence in depth:
            # a guessed session id must not let another user append messages)
            ok = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id=? AND user_id=?", (sid, self._uid)
            ).fetchone()
            if not ok:
                return
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, sources, created_at) VALUES(?,?,?,?,?)",
                (sid, role, content, json.dumps(sources or [], ensure_ascii=False), now),
            )
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
            self._conn.commit()

    def messages(self, sid: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT m.role, m.content, m.sources FROM messages m "
            "JOIN sessions s ON s.id = m.session_id AND s.user_id = ? "
            "WHERE m.session_id=? ORDER BY m.id", (self._uid, sid)
        ).fetchall()
        out = []
        for r in rows:
            m = dict(r)
            try:
                import json

                m["sources"] = json.loads(m["sources"] or "[]")
            except (ValueError, TypeError):
                m["sources"] = []
            out.append(m)
        return out

    def delete_session(self, sid: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id=? AND user_id=?", (sid, self._uid)
            )
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            self._conn.commit()
            return bool(cur.rowcount)

    # ---------------- search ----------------
    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def total_chunks(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE d.user_id = ?",
            (self._uid,),
        ).fetchone()[0]

    def _all(self):
        """Lazily-cached (chunk_rows, embedding_matrix), scoped to this view's
        user and invalidated when the shared version counter changes."""
        version, cache = self._shared["version"], self._shared["cache"]
        if cache is not None and cache[0] == version and cache[1] == self._uid:
            return cache[2], cache[3]
        rows = self._conn.execute(
            "SELECT c.doc_name, c.page, c.text, c.emb FROM chunks c "
            "JOIN docs d ON d.id = c.doc_id WHERE d.user_id = ?",
            (self._uid,),
        ).fetchall()
        if not rows:
            self._shared["cache"] = (version, self._uid, [], None)
            return [], None
        matrix = np.vstack([np.frombuffer(r["emb"], dtype="<f4") for r in rows])
        self._shared["cache"] = (version, self._uid, rows, matrix)
        return rows, matrix

    def search(self, query_vec: np.ndarray, k: int = 8) -> list[dict]:
        """query_vec must already be normalized."""
        rows, matrix = self._all()
        if not rows or query_vec is None or matrix.shape[1] != query_vec.size:
            if query_vec is None:
                return []
            # dimension mismatch (model switched) — treat as no match rather than crash
            return []
        scores = matrix @ query_vec
        top = np.argsort(-scores)[:k]
        return [
            {
                "doc_name": rows[i]["doc_name"],
                "page": rows[i]["page"],
                "text": rows[i]["text"],
                "score": float(scores[i]),
            }
            for i in top
        ]