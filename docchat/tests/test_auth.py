"""Tests for the identity layer and per-user data isolation.

Run: python tests/test_auth.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ragchat import auth
from ragchat.store import Store


def _vecs(n, dim=8, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.normal(size=dim).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return out


def test_resolve_user_priority_clerk_over_guest():
    # no Clerk configured -> bearer token is unverifiable and degrades
    u = auth.resolve_user(authorization="Bearer garbage.token.here", guest_id="abc123")
    assert u.source in ("guest", "local")  # never crashes; degrades safely


def test_resolve_user_guest_and_local():
    g = auth.resolve_user(guest_id="device-abc-12345")
    assert g.source == "guest"
    assert g.uid == "guest:device-abc-12345"

    local = auth.resolve_user(guest_id="")
    assert local.source == "local"
    assert local.uid == ""

    # malformed guest ids are rejected -> local
    bad = auth.resolve_user(guest_id="<script>")
    assert bad.source == "local"


def test_clerk_enabled_only_with_key():
    saved = auth.CLERK_PUBLISHABLE_KEY
    try:
        auth.CLERK_PUBLISHABLE_KEY = ""
        assert auth.clerk_enabled() is False
        auth.CLERK_PUBLISHABLE_KEY = "pk_test_abc"
        assert auth.clerk_enabled() is True
    finally:
        auth.CLERK_PUBLISHABLE_KEY = saved


def test_publishable_domain_extraction():
    # pk_<base64url JSON {"d": "https://x.clerk.accounts.dev"}>
    import base64, json
    payload = base64.urlsafe_b64encode(
        json.dumps({"d": "https://demo.clerk.accounts.dev"}).encode()
    ).decode().rstrip("=")
    saved_pk, saved_api = auth.CLERK_PUBLISHABLE_KEY, auth.CLERK_FRONTEND_API
    try:
        auth.CLERK_PUBLISHABLE_KEY = f"pk_test_{payload}"
        auth.CLERK_FRONTEND_API = ""
        assert auth._publishable_domain() == "https://demo.clerk.accounts.dev"
        assert auth._jwks_url() == "https://demo.clerk.accounts.dev/.well-known/jwks.json"
    finally:
        auth.CLERK_PUBLISHABLE_KEY, auth.CLERK_FRONTEND_API = saved_pk, saved_api


# ---------------- store isolation ----------------

def test_store_users_are_isolated():
    with tempfile.TemporaryDirectory() as td:
        base = Store(os.path.join(td, "t.db"))
        a = base.for_user("guest:aaa")
        b = base.for_user("guest:bbb")
        local = base.for_user("")

        # each user sees only their own docs
        a.add_doc("a.txt", 10, [(None, "alpha")], _vecs(1))
        b.add_doc("b.txt", 10, [(None, "beta")], _vecs(1))
        assert [d["name"] for d in a.list_docs()] == ["a.txt"]
        assert [d["name"] for d in b.list_docs()] == ["b.txt"]
        assert local.list_docs() == []

        # vector search is scoped too
        q = _vecs(1, seed=9)[0]
        assert all(h["doc_name"] == "a.txt" for h in a.search(q, k=3))
        assert all(h["doc_name"] == "b.txt" for h in b.search(q, k=3))
        assert a.search(q, k=3) and not local.search(q, k=3)

        # keyword search scoped via docs join
        assert any("alpha" in h["text"] for h in a.keyword_search("alpha"))
        assert b.keyword_search("alpha") == []

        # memory isolation
        a.add_memory("prefers tea", "preference")
        assert [m["fact"] for m in a.list_memory()] == ["prefers tea"]
        assert b.list_memory() == []

        # sessions + messages isolation
        sa = a.create_session("a's chat")
        a.add_message(sa["id"], "user", "hello from a")
        assert len(a.messages(sa["id"])) == 1
        assert b.get_session(sa["id"]) is None          # cannot read another's session
        assert b.messages(sa["id"]) == []               # nor its messages
        assert [s["title"] for s in a.list_sessions()] == ["a's chat"]
        assert b.list_sessions() == []

        # sheets isolation
        a.add_sheet("s.csv", "csv", 3, 2)
        assert len(a.list_sheets()) == 1
        assert b.list_sheets() == []
        assert b.get_sheet(a.list_sheets()[0]["id"]) is None

        base.close()


def test_store_total_chunks_scoped():
    with tempfile.TemporaryDirectory() as td:
        base = Store(os.path.join(td, "t.db"))
        a = base.for_user("guest:aaa")
        b = base.for_user("guest:bbb")
        a.add_doc("a.txt", 10, [(None, "alpha"), (None, "beta")], _vecs(2))
        b.add_doc("b.txt", 10, [(None, "gamma")], _vecs(1))
        assert a.total_chunks() == 2
        assert b.total_chunks() == 1
        base.close()


def test_delete_is_scoped():
    with tempfile.TemporaryDirectory() as td:
        base = Store(os.path.join(td, "t.db"))
        a = base.for_user("guest:aaa")
        b = base.for_user("guest:bbb")
        doc = a.add_doc("a.txt", 10, [(None, "alpha")], _vecs(1))
        # b cannot delete a's doc; a can
        assert b.delete_doc(doc["id"]) is False
        assert a.list_docs()  # still there
        assert a.delete_doc(doc["id"]) is True
        assert a.list_docs() == []
        base.close()


def test_memory_ops_scoped():
    with tempfile.TemporaryDirectory() as td:
        base = Store(os.path.join(td, "t.db"))
        a = base.for_user("guest:aaa")
        b = base.for_user("guest:bbb")
        row = a.add_memory("user likes dark mode", "preference")
        # b cannot update or delete a's fact
        assert b.update_memory(row["id"], "changed") is False
        assert b.delete_memory(row["id"]) is False
        assert [m["fact"] for m in a.list_memory()] == ["user likes dark mode"]
        assert a.update_memory(row["id"], "user likes light mode") is True
        assert a.delete_memory(row["id"]) is True
        base.close()


def test_session_patch_scoped():
    with tempfile.TemporaryDirectory() as td:
        base = Store(os.path.join(td, "t.db"))
        a = base.for_user("guest:aaa")
        b = base.for_user("guest:bbb")
        s = a.create_session("my chat")
        # b cannot rename or pin a's chat
        b.set_title(s["id"], "hacked")
        b.set_pinned(s["id"], True)
        assert a.get_session(s["id"])["title"] == "my chat"
        assert a.get_session(s["id"])["pinned"] == 0
        # b cannot delete it either
        assert b.delete_session(s["id"]) is False
        assert a.get_session(s["id"]) is not None
        base.close()


def test_migration_backfills_user_id():
    """Pre-existing databases (no user_id column) must keep working in local
    mode: existing rows are visible, new rows get the caller's uid."""
    import sqlite3
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "old.db")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE docs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                size INTEGER NOT NULL, chunks INTEGER NOT NULL DEFAULT 0, uploaded_at TEXT NOT NULL);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL,
                doc_name TEXT NOT NULL, page INTEGER, text TEXT NOT NULL, emb BLOB NOT NULL);
            CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                sources TEXT, created_at TEXT NOT NULL);
            CREATE TABLE memory (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact', session_id TEXT, created_at TEXT NOT NULL);
            CREATE TABLE sheets (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                kind TEXT NOT NULL, rows INTEGER NOT NULL DEFAULT 0, cols INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL);
        """)
        conn.commit()
        conn.close()
        st = Store(path)
        local = st.for_user("")
        guest = st.for_user("guest:zzz")
        # user_id column was added by migration; legacy rows default to '' (local)
        cols = [r["name"] for r in st._conn.execute("PRAGMA table_info(docs)").fetchall()]
        assert "user_id" in cols
        # guest workspace starts empty; local workspace is independent
        assert guest.list_docs() == []
        assert local.list_docs() == []
        st.close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
