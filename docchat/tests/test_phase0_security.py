"""Phase 0 security gate: fail-closed auth, signed chart URLs, store scoping,
upload guards, SSRF protection, and the hosted-mode sandbox switch."""
import io
import zipfile

import numpy as np
import pytest

from ragchat import auth, parsing, signing, tools
from ragchat.store import Store

pytestmark = pytest.mark.unit


# ---------------- auth ----------------

def test_auth_required_fails_closed(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("ALLOW_GUESTS", "0")
    with pytest.raises(auth.AuthError):
        auth.resolve_user("", "")
    with pytest.raises(auth.AuthError):
        auth.resolve_user("Bearer not-a-real-token", "")


def test_local_mode_is_explicit_owner(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "local")
    user = auth.resolve_user("", "")
    assert user.uid == "" and user.source == "local" and user.is_owner


def test_guest_disabled_in_required_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("ALLOW_GUESTS", "0")
    with pytest.raises(auth.AuthError):
        auth.resolve_user("", "device-abc-12345")


def test_guest_allowed_in_required_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("ALLOW_GUESTS", "1")
    user = auth.resolve_user("", "device-abc-12345")
    assert user.uid == "guest:device-abc-12345" and not user.is_owner


def test_owner_is_marked_for_clerk(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("OWNER_USER_ID", "clerk:user_owner")
    u = auth.User(uid="clerk:user_owner", source="clerk")
    assert auth._mark_owner(u).is_owner


# ---------------- signed chart URLs ----------------

def test_signed_chart_roundtrip_and_tamper(monkeypatch):
    monkeypatch.setenv("CHART_SIGNING_KEY", "test-chart-key")
    token = signing.sign_chart(7, "guest:device-abc-12345", ttl=60)
    data = signing.verify_chart(token)
    assert data is not None
    assert data["sid"] == 7 and data["uid"] == "guest:device-abc-12345"
    assert signing.verify_chart(token + "x") is None
    assert signing.verify_chart("") is None
    assert signing.verify_chart("not-a-token") is None


# ---------------- store scoping ----------------

def _doc(store, name="a.txt", text="hello world"):
    return store.add_doc(name, len(text), [(None, text)], [np.ones(4, dtype=np.float32)])


def test_delete_doc_does_not_cross_workspaces(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    try:
        a, b = st.for_user("guest:aaa11111"), st.for_user("guest:bbb22222")
        doc = _doc(a)
        assert b.delete_doc(doc["id"]) is False
        assert a.get_doc(doc["id"]) is not None
        assert a.total_chunks() == 1
    finally:
        st.close()


def test_delete_session_does_not_cross_workspaces(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    try:
        a, b = st.for_user("guest:aaa11111"), st.for_user("guest:bbb22222")
        sid = a.create_session("mine")["id"]
        a.add_message(sid, "user", "hello")
        assert b.delete_session(sid) is False
        assert len(a.messages(sid)) == 1
    finally:
        st.close()


def test_reindex_fts_keeps_doc_join(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    try:
        u = st.for_user("guest:aaa11111")
        _doc(u, "notes.txt", "hello searchable world")
        st._conn.execute("DELETE FROM chunks_fts")
        st._conn.commit()
        u.reindex_fts()
        hits = u.keyword_search("hello")
        assert hits and hits[0]["doc_name"] == "notes.txt"
    finally:
        st.close()


def test_schema_version_and_wal(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    try:
        assert st._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        version = st._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 1
    finally:
        st.close()


# ---------------- calculator / sandbox ----------------

def test_calculator_rejects_nested_exponents():
    assert tools.calculate("2**10") == "1024"
    assert tools.calculate("2**(2**10)").startswith("Error")


def test_run_python_disabled_in_hosted_mode(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "required")
    assert "disabled" in tools.run_python("print(1)")


def test_safe_env_scrubs_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk-clerk")
    env = tools.safe_env()
    assert "GROQ_API_KEY" not in env and "CLERK_SECRET_KEY" not in env
    assert "PATH" in env


# ---------------- SSRF ----------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost/",
    "http://10.0.0.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "file:///etc/passwd",
    "http://user:pass@example.com/",
])
def test_ssrf_guard_rejects_private_or_unsafe_urls(url):
    with pytest.raises(ValueError):
        tools._assert_public_url(url)


# ---------------- archive guards ----------------

def _zip_bytes(size: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x" * size)
    return buf.getvalue()


def test_zip_bomb_guard():
    data = _zip_bytes(1000)
    with pytest.raises(ValueError):
        parsing.check_zip_safety(data, max_uncompressed=100)
    assert parsing.check_zip_safety(data, max_uncompressed=10_000) is True
    assert parsing.check_zip_safety(b"not a zip") is False


# ---------------- API gate ----------------

def test_api_requires_auth_in_required_mode(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("ALLOW_GUESTS", "0")
    client = TestClient(server.app)
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/health").status_code == 200

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["me"] is None
    assert state.json()["auth_required"] is True


def test_config_rejects_non_owner(monkeypatch):
    from fastapi.testclient import TestClient

    import server

    monkeypatch.setenv("AUTH_MODE", "required")
    monkeypatch.setenv("ALLOW_GUESTS", "1")
    monkeypatch.setenv("OWNER_USER_ID", "clerk:someone-else")

    class _Guest:
        def resolve_user(self, authorization="", guest_id="", required=None):
            return auth.User(uid=f"guest:{guest_id}", source="guest", name="Guest")

    monkeypatch.setattr(auth, "resolve_user", _Guest().resolve_user)
    client = TestClient(server.app)
    r = client.post("/api/config", headers={"X-Guest-Id": "device-abc-12345"},
                    json={"model": "openai/gpt-oss-120b"})
    assert r.status_code == 403
