"""Tests for chat attachments: ingest pipeline, per-user validation, and the
agent's ATTACHED FILES behavior (all offline: fake embed + fake LLM).
Run: python tests/test_attach.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ragchat import ingest
from ragchat.agent import Agent
from ragchat.store import Store


class FakeLLM:
    """Records decision prompts so tests can assert the agent's instructions."""

    def __init__(self, script):
        self.script = list(script)
        self.decision_prompts = []

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        prompt = " ".join(m["content"] for m in messages)
        self.decision_prompts.append(prompt)
        if not self.script:
            raise IndexError("script exhausted")
        return self.script.pop(0)

    async def stream(self, key, model, messages):
        if not self.script:
            raise IndexError("script exhausted")
        text = self.script.pop(0)
        for i in range(0, len(text), 8):
            yield text[i:i + 8]


def _embed(text):
    return np.ones(8, dtype=np.float32) / np.sqrt(8)


def _run(agent, q, history=None, sid=None, attachments=None):
    events = []

    async def collect():
        async for e in agent.run(q, history or [], sid, attachments=attachments):
            events.append(e)

    asyncio.run(collect())
    return events


def test_ingest_doc_parses_chunks_and_stores():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        doc, n = asyncio.run(ingest.ingest_doc(st, "notes.txt", b"alpha beta\ngamma delta", _embed))
        assert doc["name"] == "notes.txt" and doc["size"] == len(b"alpha beta\ngamma delta")
        assert n >= 1 and st.total_chunks() >= 1
        st.close()


def test_ingest_doc_rejects_unsupported_and_empty():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        try:
            asyncio.run(ingest.ingest_doc(st, "virus.exe", b"MZ...", _embed))
            assert False, "should have raised"
        except ValueError as e:
            assert "Unsupported" in str(e)
        try:
            asyncio.run(ingest.ingest_doc(st, "empty.txt", b"", _embed))
            assert False, "should have raised"
        except ValueError:
            pass
        assert st.list_docs() == []
        st.close()


def test_ingest_doc_duplicate_names_get_suffix():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        d1, _ = asyncio.run(ingest.ingest_doc(st, "report.txt", b"first copy", _embed))
        d2, _ = asyncio.run(ingest.ingest_doc(st, "report.txt", b"second copy", _embed))
        assert d1["name"] == "report.txt" and d2["name"].startswith("report (2)")
        st.close()


def test_ingest_doc_wraps_embedding_failure():
    def boom(_texts):
        raise RuntimeError("model download failed")

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        try:
            try:
                asyncio.run(ingest.ingest_doc(st, "a.txt", b"hello world", boom))
                assert False, "should have raised"
            except ingest.EmbedError as e:
                assert "download" in str(e)
            assert st.list_docs() == []
        finally:
            st.close()


def test_resolve_attachments_scopes_to_user():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        doc, _ = asyncio.run(ingest.ingest_doc(st, "mine.txt", b"my content", _embed))
        rows = ingest.resolve_attachments(st, [{"id": doc["id"], "name": "mine.txt"}])
        assert len(rows) == 1 and rows[0]["name"] == "mine.txt" and rows[0]["chunks"] >= 1
        # foreign / garbage / wrong-type ids are skipped, never crash
        assert ingest.resolve_attachments(st, [{"id": 99999, "name": "ghost"}]) == []
        assert ingest.resolve_attachments(st, [{"id": "abc"}, None, {"name": "no-id"}]) == []
        assert ingest.resolve_attachments(st, "not-a-list") == []
        assert ingest.resolve_attachments(st, None) == []
        st.close()


def test_resolve_attachments_caps_at_five():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        docs = [asyncio.run(ingest.ingest_doc(st, f"f{i}.txt", f"content {i}".encode(), _embed))[0]
                for i in range(8)]
        rows = ingest.resolve_attachments(st, [{"id": d["id"]} for d in docs])
        assert len(rows) == 5, rows
        st.close()


def test_agent_announces_attachments_and_searches_first():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        try:
            doc, _ = asyncio.run(ingest.ingest_doc(
                st, "guide.txt",
                b"The guide states the launch code is ZEBRA-42 and explains the setup steps.",
                _embed))
            llm = FakeLLM([
                '{"thought":"read the attached file","tool":"search_documents","tool_input":{"query":"launch code"}}',
                '{"thought":"enough evidence","tool":null,"tool_input":{}}',
                "The attached guide states the launch code is ZEBRA-42 [1].",
            ])
            agent = Agent(key="k", model="m", store=st, llm_chat=llm.chat, llm_stream=llm.stream,
                          embed_fn=_embed, is_disconnected=lambda: False)
            evts = _run(agent, "What is the launch code?", attachments=[{"id": doc["id"], "name": "guide.txt"}])
            types = [e["type"] for e in evts]
            assert "sources" in types, types
            done = [e for e in evts if e["type"] == "done"][0]
            assert "ZEBRA-42" in done["answer"]
            # the decision prompt must announce the attachment
            assert any("guide.txt" in p and "ATTACHED FILES" in p for p in llm.decision_prompts), \
                llm.decision_prompts[:1]
            # ...and the prompt must instruct searching the attachment first
            assert any("search_documents FIRST" in p for p in llm.decision_prompts)
        finally:
            st.close()


def test_agent_without_attachments_leaves_prompt_unchanged():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM(['{"thought":"none","tool":null,"tool_input":{}}', "Hello!"])
        agent = Agent(key="k", model="m", store=st, llm_chat=llm.chat, llm_stream=llm.stream,
                      embed_fn=_embed, is_disconnected=lambda: False)
        _run(agent, "hi")
        assert any("no files attached to this message" in p for p in llm.decision_prompts)
        st.close()


def test_embed_cache_is_durable_not_the_os_temp_dir():
    """503s on attachments came from the embedding model being re-downloaded
    inside the upload request whenever fastembed's OS-temp cache was wiped.
    Resolution order: FASTEMBED_CACHE > data/models > legacy temp cache."""
    from ragchat import llm

    saved_env = os.environ.pop("FASTEMBED_CACHE", None)
    saved_durable, saved_legacy = llm.DEFAULT_EMBED_CACHE, llm.legacy_embed_cache_dir
    try:
        with tempfile.TemporaryDirectory() as td:
            durable = os.path.join(td, "data", "models")
            legacy = os.path.join(td, "fastembed_cache")
            os.makedirs(legacy)
            llm.DEFAULT_EMBED_CACHE = durable
            llm.legacy_embed_cache_dir = lambda: legacy
            # nothing cached yet -> download into the durable (volume) dir
            assert llm.embed_cache_dir() == durable
            # a model already in fastembed's old temp cache is reused
            open(os.path.join(legacy, "model.onnx"), "wb").close()
            assert llm.embed_cache_dir() == legacy
            # ...unless the durable cache has one too, which wins
            os.makedirs(durable)
            open(os.path.join(durable, "model.onnx"), "wb").close()
            assert llm.embed_cache_dir() == durable
            # an explicit FASTEMBED_CACHE (Docker: /opt/fastembed) always wins
            os.environ["FASTEMBED_CACHE"] = os.path.join(td, "pinned")
            assert llm.embed_cache_dir() == os.environ["FASTEMBED_CACHE"]
    finally:
        llm.DEFAULT_EMBED_CACHE, llm.legacy_embed_cache_dir = saved_durable, saved_legacy
        os.environ.pop("FASTEMBED_CACHE", None)
        if saved_env is not None:
            os.environ["FASTEMBED_CACHE"] = saved_env


def test_embed_failure_fails_fast_then_retries_after_the_cooldown():
    """A failed embedding load is reported (state + 503 body) and uploads
    fail fast instead of stalling ~40 s on every attempt, yet the load is
    genuinely retried after the cooldown so recovery is automatic."""
    import time

    from ragchat import llm

    saved = (llm._embedder, llm._embed_state, llm._embed_failed_at, llm._embed_fatal,
             llm._load_embedder)
    try:
        llm._embedder = None
        llm._embed_fatal = False
        llm._embed_state = "error: Could not load model X from any source."
        llm._embed_failed_at = time.monotonic()
        assert llm.embed_error() == "Could not load model X from any source."
        assert llm.embed_state()["state"].startswith("error")
        start = time.monotonic()
        try:
            llm._get_embedder()
            assert False, "should have raised"
        except RuntimeError as e:
            assert "Could not load model X from any source." in str(e), e
            assert "retrying" in str(e), e
        assert time.monotonic() - start < 1, "must not re-attempt the download inside the cooldown"

        # once the cooldown expires the load is attempted again (offline double)
        attempts = []

        def _boom():
            attempts.append(1)
            raise RuntimeError("still offline")

        llm._load_embedder = _boom
        llm._embed_failed_at = time.monotonic() - llm.EMBED_RETRY_AFTER - 1
        try:
            llm._get_embedder()
            assert False, "should have raised"
        except RuntimeError as e:
            assert "still offline" in str(e), e
        assert attempts == [1], attempts
        assert llm.embed_error() == "still offline"
    finally:
        (llm._embedder, llm._embed_state, llm._embed_failed_at, llm._embed_fatal,
         llm._load_embedder) = saved


def test_embed_state_reports_the_cache_and_reason():
    """A bare 503 is useless: /api/state must carry the model's state so the UI
    can say why uploads fail."""
    from ragchat import llm

    st = llm.embed_state()
    assert st["model"] and st["cache_dir"], st
    assert st["state"] == "ready" or st["state"] == "cold" or st["state"].startswith("error"), st


def test_upload_endpoints_return_400_json_for_corrupt_files_not_500():
    """End-to-end upload contract: corrupt files must be HTTP 400 with a JSON
    detail, never a raw 500 — hosting proxies surface bare 500s as opaque
    HTTP 502s, which is what users saw on every bad upload."""
    import server
    from fastapi.testclient import TestClient
    from ragchat import llm

    def _embed(texts):
        return np.array([_embed_one() for _ in texts], dtype=np.float32)

    def _embed_one():
        v = np.ones(8, dtype=np.float32)
        return v / np.linalg.norm(v)

    real_db, real_embed = server.DB, llm.embed_texts
    with tempfile.TemporaryDirectory() as td:
        tmp = Store(os.path.join(td, "t.db"))
        server.DB, llm.embed_texts = tmp, _embed
        try:
            client = TestClient(server.app, raise_server_exceptions=False)
            corrupt = [
                ("broken.pdf", b"%PDF-1.4 garbage not a real pdf trailer", "application/pdf"),
                ("broken.docx", b"PK not a real docx content at all",
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ]
            for url in ("/api/docs", "/api/attach"):
                for name, data, mt in corrupt:
                    r = client.post(url, files={"file": (name, data, mt)})
                    assert r.status_code == 400, (url, name, r.status_code, r.text[:200])
                    assert r.json().get("detail"), (url, name, r.text[:200])
            # ...and a valid upload on the same endpoints still succeeds
            r = client.post("/api/docs", files={"file": ("ok.txt", b"hello world", "text/plain")})
            assert r.status_code == 200, r.text[:200]
        finally:
            server.DB, llm.embed_texts = real_db, real_embed
            tmp.close()


def test_doc_stats_descriptor():
    from ragchat import parsing

    assert parsing.doc_stats("a.pdf") == {"kind": "pdf", "split_by": "page"}
    assert parsing.doc_stats("b.docx")["kind"] == "word"
    assert parsing.doc_stats("c.csv")["kind"] == "csv"
    assert parsing.doc_stats("d.md")["kind"] == "markdown"
    assert parsing.doc_stats("e.txt")["kind"] == "text"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
