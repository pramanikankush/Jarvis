"""HTTP-level tests for the project API (TestClient, temp DB, fake embeddings)."""
import io
import zipfile

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("src/app.py", "def hello():\n    return 'world'\n")
        z.writestr("README.md", "# Demo project\n")
    return buf.getvalue()


def _client_with_temp_db(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import server
    from ragchat.store import Store

    real_db, real_embed = server.DB, server.llm.embed_texts
    st = Store(str(tmp_path / "api.db"))
    server.DB = st
    server.llm.embed_texts = lambda texts: np.ones((len(texts), 8), dtype=np.float32)
    monkeypatch.setenv("AUTH_MODE", "local")
    client = TestClient(server.app, raise_server_exceptions=False)
    return client, st, real_db, real_embed


def _restore(st, real_db, real_embed):
    import server

    server.DB = real_db
    server.llm.embed_texts = real_embed
    st.close()


def test_project_api_roundtrip(tmp_path, monkeypatch):
    client, st, real_db, real_embed = _client_with_temp_db(tmp_path, monkeypatch)
    try:
        r = client.post("/api/projects",
                        files={"file": ("demo.zip", _zip(), "application/zip")},
                        data={"name": "Demo"})
        assert r.status_code == 200, r.text
        pid = r.json()["project"]["id"]
        assert r.json()["stats"]["files"] >= 2

        assert client.get("/api/projects").status_code == 200
        detail = client.get(f"/api/projects/{pid}")
        assert detail.status_code == 200
        assert detail.json()["project"]["name"] == "Demo"
        assert detail.json()["file_count"] >= 2

        ins = client.put(f"/api/projects/{pid}/instructions", json={"content": "Use pytest."})
        assert ins.status_code == 200, ins.text
        assert client.get(f"/api/projects/{pid}/instructions").json()["instruction"]["version"] == 1

        files = client.get(f"/api/projects/{pid}/files")
        assert files.status_code == 200
        assert any(f["path"] == "src/app.py" for f in files.json()["files"])

        deleted = client.delete(f"/api/projects/{pid}")
        assert deleted.status_code == 200
        assert client.get(f"/api/projects/{pid}").status_code == 404
    finally:
        _restore(st, real_db, real_embed)


def test_project_api_is_per_user(tmp_path, monkeypatch):
    client, st, real_db, real_embed = _client_with_temp_db(tmp_path, monkeypatch)
    try:
        a = {"X-Guest-Id": "device-aaaa1111"}
        b = {"X-Guest-Id": "device-bbbb2222"}
        r = client.post("/api/projects", headers=a,
                        files={"file": ("demo.zip", _zip(), "application/zip")},
                        data={"name": "A only"})
        assert r.status_code == 200, r.text
        pid = r.json()["project"]["id"]
        assert client.get(f"/api/projects/{pid}", headers=b).status_code == 404
        assert client.get("/api/projects", headers=b).json()["projects"] == []
        assert client.delete(f"/api/projects/{pid}", headers=b).status_code == 404
    finally:
        _restore(st, real_db, real_embed)


def test_project_chat_injects_context_and_links_session(tmp_path, monkeypatch):
    client, st, real_db, real_embed = _client_with_temp_db(tmp_path, monkeypatch)
    import server

    captured = []

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def run(self, question, history, session_id, attachments=None):
            yield {"type": "done", "answer": "project answer", "sources": []}

    real_agent = server.agent_mod.Agent
    real_key = server._config.get("groq_key")
    server.agent_mod.Agent = FakeAgent
    server._config["groq_key"] = "test-key"
    try:
        r = client.post("/api/projects",
                        files={"file": ("demo.zip", _zip(), "application/zip")},
                        data={"name": "Demo"})
        assert r.status_code == 200, r.text
        pid = r.json()["project"]["id"]
        assert client.put(f"/api/projects/{pid}/instructions",
                          json={"content": "Prefer small functions."}).status_code == 200

        chat = client.post("/api/chat", json={"message": "Explain app.py", "project_id": pid})
        assert chat.status_code == 200, chat.text
        assert "project answer" in chat.text
        assert captured and captured[-1]["project_id"] == pid
        block = captured[-1]["project_block"]
        assert "PROJECT CONTEXT" in block and "Demo" in block
        assert "Prefer small functions." in block

        sessions = client.get("/api/sessions").json()["sessions"]
        assert sessions and sessions[0]["project_id"] == pid
    finally:
        server.agent_mod.Agent = real_agent
        server._config["groq_key"] = real_key
        _restore(st, real_db, real_embed)


def test_project_upload_rejects_traversal(tmp_path, monkeypatch):
    client, st, real_db, real_embed = _client_with_temp_db(tmp_path, monkeypatch)
    try:
        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../evil.txt", "nope")
        r = client.post("/api/projects",
                        files={"file": ("bad.zip", bad.getvalue(), "application/zip")})
        assert r.status_code == 400, r.text
        assert "traversal" in r.json()["detail"].lower() or "escapes" in r.json()["detail"].lower()
    finally:
        _restore(st, real_db, real_embed)
