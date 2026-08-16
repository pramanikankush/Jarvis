"""Tests for ragchat.agent with a scripted fake LLM (no network).
Run: python tests/test_agent.py
"""
import asyncio
import contextlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ragchat import tools
from ragchat.agent import Agent
from ragchat.store import Store


@contextlib.contextmanager
def _patch_web_search(fn):
    """Swap tools.web_search so agent tests never touch the network."""
    orig = tools.web_search
    tools.web_search = fn
    try:
        yield
    finally:
        tools.web_search = orig


class FakeLLM:
    """Pops responses from a script; IndexError when exhausted is treated as
    'no response' (memory extraction is designed to fail silently)."""

    def __init__(self, script):
        self.script = list(script)

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
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


def _store_with_doc(tmp):
    st = Store(os.path.join(tmp, "t.db"))
    vec = _embed("x")
    st.add_doc("manual.txt", 42,
               [(None, "The documentation states the answer is 42.")],
               [vec])
    return st


def _agent(st, llm, disconnected=False):
    return Agent(key="k", model="m", store=st, llm_chat=llm.chat, llm_stream=llm.stream,
                 embed_fn=_embed, is_disconnected=lambda: disconnected)


def _run(agent, q="What is the answer?", history=None, sid=None):
    events = []
    async def collect():
        async for e in agent.run(q, history or [], sid):
            events.append(e)
    asyncio.run(collect())
    return events


def _types(events):
    return [e["type"] for e in events]


def test_plain_chat_streams_live():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM(['{"thought":"none","tool":null,"tool_input":{}}', "Hi! What can I do for you?"])
        evts = _run(_agent(st, llm))
        types = _types(evts)
        assert types.count("token") > 1, types
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == "Hi! What can I do for you?"
        st.close()


class FailingLLM:
    """LLM whose stream raises — models a Groq outage / rate limit."""

    def __init__(self, chat_script):
        self.script = list(chat_script)

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        if not self.script:
            raise IndexError("script exhausted")
        return self.script.pop(0)

    async def stream(self, key, model, messages):
        raise RuntimeError("Groq request failed: rate limit reached")


def test_plain_stream_failure_yields_error_then_done():
    """Guaranteed terminal state: when the LLM stream fails mid-chat, the agent
    must emit an error event AND a done event (never hang without either)."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FailingLLM(['{"thought":"none","tool":null,"tool_input":{}}'])
        evts = _run(_agent(st, llm), q="hi")
        types = _types(evts)
        assert "error" in types, types
        assert "done" in types, types
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == ""
        st.close()


def test_final_tool_answer_llm_failure_yields_error():
    """Tool path: if the final answer LLM call fails, emit an error event
    instead of crashing the stream without a terminal event."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FailingLLM([
            '{"thought":"calc","tool":"calculate","tool_input":{"expression":"2+2"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
        ])
        evts = _run(_agent(st, llm), q="what is 2+2?")
        types = _types(evts)
        assert "error" in types, types
        st.close()


def test_unparseable_decision_falls_back_to_chat():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM(["not json at all", "Sorry, I didn't catch that."])
        evts = _run(_agent(st, llm))
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == "Sorry, I didn't catch that."
        st.close()


def test_rag_path_sources_citations_verdict_and_memory():
    with tempfile.TemporaryDirectory() as td:
        st = _store_with_doc(td)
        llm = FakeLLM([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"the answer"}}',
            '{"thought":"enough","tool":null,"tool_input":{}}',
            "According to the documentation, the answer is [1] 42.",
            '{"relevant": true, "supported": true, "missing_info": null}',
            '{"facts":[{"fact":"the user asked about the answer","kind":"goal"}],'
            '"task":"understand the docs","summary":null}',
        ])
        sid = st.create_session("s")["id"]
        evts = _run(_agent(st, llm), sid=sid)
        types = _types(evts)
        assert "tool" in types and "sources" in types and "done" in types
        done = [e for e in evts if e["type"] == "done"][0]
        assert "42" in done["answer"] and "[1]" in done["answer"]
        assert len(done["sources"]) == 1
        assert done["sources"][0]["doc_name"] == "manual.txt"
        assert done["verdict"]["supported"] is True
        # memory extraction persisted the fact
        assert st.list_memory() and "answer" in st.list_memory()[0]["fact"]
        st.close()


def test_self_rag_corrects_missing_citations():
    with tempfile.TemporaryDirectory() as td:
        st = _store_with_doc(td)
        llm = FakeLLM([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"answer"}}',
            '{"thought":"enough","tool":null,"tool_input":{}}',
            "The answer is 42 and it is fully explained inside the uploaded manual.",  # no [n]
            "The answer is [1] 42, as stated in the uploaded manual.",               # corrected
            '{"relevant": true, "supported": true, "missing_info": null}',
        ])
        evts = _run(_agent(st, llm))
        done = [e for e in evts if e["type"] == "done"][0]
        assert "[1]" in done["answer"], done["answer"]
        assert done["verdict"]["corrected"] is True
        st.close()


def test_self_rag_unsupported_answer_gets_corrected():
    with tempfile.TemporaryDirectory() as td:
        st = _store_with_doc(td)
        llm = FakeLLM([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"answer"}}',
            '{"thought":"enough","tool":null,"tool_input":{}}',
            "The answer is [1] 9000, which is definitely correct.",
            '{"relevant": true, "supported": false, "missing_info": "the actual value"}',
            "The answer is [1] 42 per the documentation.",
        ])
        evts = _run(_agent(st, llm))
        done = [e for e in evts if e["type"] == "done"][0]
        assert "42" in done["answer"], done["answer"]
        assert done["verdict"]["corrected"] is True
        assert done["verdict"]["supported"] is True
        st.close()


def test_no_documents_tool_returns_useful_error():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))  # no docs
        llm = FakeLLM([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"anything"}}',
            '{"thought":"no docs","tool":null,"tool_input":{}}',
            "I don't have any documents uploaded yet — please add files first.",
        ])
        evts = _run(_agent(st, llm))
        done = [e for e in evts if e["type"] == "done"][0]
        assert "documents" in done["answer"]
        assert done["sources"] == []
        st.close()


def test_calculator_tool_used_and_reported():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"calc","tool":"calculate","tool_input":{"expression":"2+2"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "2 + 2 is 4.",
        ])
        evts = _run(_agent(st, llm))
        assert "tool" in _types(evts)
        done = [e for e in evts if e["type"] == "done"][0]
        assert "4" in done["answer"]
        st.close()


def test_decision_messages_include_tool_results():
    """The decision loop must pass prior tool results back to the LLM,
    otherwise the agent cannot chain tool calls (e.g. info -> stats)."""
    seen = {"calls": []}

    class RecorderLLM:
        async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
            seen["calls"].append("\n".join(m["content"] for m in messages if "content" in m))
            if len(seen["calls"]) == 1:
                return '{"thought":"calc","tool":"calculate","tool_input":{"expression":"2+2"}}'
            return '{"thought":"done","tool":null,"tool_input":{}}'

        async def stream(self, key, model, messages):
            yield "ok"

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        _run(_agent(st, RecorderLLM()))
        # a decision call (not memory extraction) must include the tool results
        tool_calls = [c for c in seen["calls"] if "TOOL RESULTS" in c]
        assert tool_calls, "no decision call saw the tool results"
        assert "2+2" in tool_calls[0]
        assert "Result:" in tool_calls[0]
        assert "4" in tool_calls[0]  # the calculator output reached the model
        st.close()


def test_unknown_tool_is_ignored_without_crash():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"?","tool":"definitely_not_real","tool_input":{}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "Fine.",
        ])
        evts = _run(_agent(st, llm))
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == "Fine."
        st.close()


# ---------------- web search decision tests (user spec scenarios) ----------------
def test_simple_question_does_not_search():
    """Scenario 1: a simple factual question must not trigger web search."""
    calls = []

    def fake_ws(q, n=5):
        calls.append(q)
        return "[1] unexpected"

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM(['{"thought":"known","tool":null,"tool_input":{}}', "The capital of France is Paris."])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="What is the capital of France?")
        assert "tool" not in _types(evts), _types(evts)
        assert calls == [], "web_search must not be invoked for simple questions"
        st.close()


def test_current_question_triggers_web_search():
    """Scenario 2: a current/latest question triggers web search and the
    answer uses the retrieved results."""
    def fake_ws(q, n=5):
        return ("[Source: web search via Tavily]\n[1] Python 3.13 Release\n"
                "Python 3.13 is the current stable release.\nhttps://www.python.org/downloads/")

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"needs current info","tool":"web_search","tool_input":{"query":"latest python version"}}',
            '{"thought":"enough","tool":null,"tool_input":{}}',
            "According to current web search results, Python 3.13 is the latest stable release.",
        ])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="What is the latest version of Python?")
        tools_used = [e for e in evts if e["type"] == "tool"]
        assert any(e["tool"] == "web_search" for e in tools_used), tools_used
        done = [e for e in evts if e["type"] == "done"][0]
        assert "3.13" in done["answer"]
        st.close()


def test_docs_answer_avoids_web_search():
    """Scenario 4: a question answerable from uploaded documents must use the
    documents and NOT search the web again."""
    calls = []

    def fake_ws(q, n=5):
        calls.append(q)
        return "[1] unexpected web result"

    with tempfile.TemporaryDirectory() as td:
        st = _store_with_doc(td)
        llm = FakeLLM([
            '{"thought":"search docs","tool":"search_documents","tool_input":{"query":"answer"}}',
            '{"thought":"enough from docs","tool":null,"tool_input":{}}',
            "The documentation says the answer is [1] 42.",
            '{"relevant": true, "supported": true, "missing_info": null}',
        ])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="What does the manual say the answer is?")
        tools_used = [e["tool"] for e in evts if e["type"] == "tool"]
        assert tools_used == ["search_documents"], tools_used
        assert calls == [], "must not re-search the web when documents suffice"
        st.close()


def test_duplicate_web_queries_are_deduplicated():
    """The per-turn cache must suppress repeat searches of the same query
    (even with different max_results or casing)."""
    calls = []

    def fake_ws(q, n=5):
        calls.append((q, n))
        return f"[1] result for {q}"

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"s1","tool":"web_search","tool_input":{"query":"python 3.13 release"}}',
            '{"thought":"s2","tool":"web_search","tool_input":{"query":"PYTHON 3.13 RELEASE","max_results":8}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "Answer.",
        ])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="tell me about python 3.13")
        tools_used = [e for e in evts if e["type"] == "tool"]
        assert len(tools_used) == 2, tools_used  # both decisions still recorded
        assert len(calls) == 1, f"expected 1 underlying search, got {calls}"
        st.close()


def test_web_search_failure_answers_with_limitation():
    """Scenario 5: when web search is unavailable the agent still answers and
    clearly states the limitation."""
    def fake_ws(q, n=5):
        return "Error: live web search is unavailable right now — no results could be retrieved."

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"try web","tool":"web_search","tool_input":{"query":"latest news"}}',
            '{"thought":"failed","tool":null,"tool_input":{}}',
            "I couldn't retrieve live results, so my answer may be outdated: based on my knowledge, the sky is blue.",
        ])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="what is the latest news?")
        assert "tool" in _types(evts)
        done = [e for e in evts if e["type"] == "done"][0]
        assert "outdated" in done["answer"], done["answer"]
        st.close()


# ---------------- memory / time tools ----------------
def test_memory_tool_add_and_list():
    """'Remember that...' goes through the memory tool and persists."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"remember","tool":"memory","tool_input":{"op":"add",'
            '"fact":"the user prefers concise answers","kind":"preference"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "Got it — I'll keep answers concise.",
        ])
        evts = _run(_agent(st, llm), q="Remember that I prefer concise answers")
        assert any(e["tool"] == "memory" for e in evts if e["type"] == "tool")
        facts = st.list_memory()
        assert any("concise" in f["fact"] for f in facts), facts
        st.close()


def test_memory_tool_forget_deletes():
    """'Forget...' deletes the matching stored fact."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        st.add_memory("the user prefers concise answers", "preference")
        llm = FakeLLM([
            '{"thought":"forget","tool":"memory","tool_input":{"op":"forget","text":"concise"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "Forgotten.",
        ])
        _run(_agent(st, llm), q="Forget my preference for concise answers")
        assert st.list_memory() == [], st.list_memory()
        st.close()


def test_time_tool_returns_current_time():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"need time","tool":"time","tool_input":{}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "The current UTC time is provided above.",
        ])
        evts = _run(_agent(st, llm), q="What time is it?")
        assert any(e["tool"] == "time" for e in evts if e["type"] == "tool")
        st.close()


def test_registry_drives_tool_list():
    """TOOL_DOCS (the LLM prompt) must derive from the registry, so a
    registered tool is automatically callable and documented."""
    from ragchat.agent import TOOL_DOCS
    from ragchat import registry

    assert set(TOOL_DOCS) == set(registry.TOOLS)
    for name in ("search_documents", "web_search", "calculate", "analyze_spreadsheet",
                 "run_python", "memory", "time"):
        assert name in TOOL_DOCS, f"{name} missing from tool list"
        assert TOOL_DOCS[name].strip(), f"{name} has no description"


# ---------------- RAG confidence / fallback (spec: uncertainty handling) ----------------
def _store_with_weak_doc(tmp):
    """A doc whose text shares no meaningful terms with the test query."""
    st = Store(os.path.join(tmp, "t.db"))
    st.add_doc("protocol.txt", 42,
               [(None, "The networking protocol handbook covers cabling, switches and routers.")],
               [_embed("x")])
    return st


def _captured_tool_results(llm_script, st, q="quantum entanglement research funding"):
    """Run the agent and return the 'TOOL RESULTS' text the model saw."""
    seen = {"calls": []}

    class RecorderLLM:
        async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
            seen["calls"].append("\n".join(m["content"] for m in messages if "content" in m))
            if not llm_script:
                return '{"thought":"done","tool":null,"tool_input":{}}'
            return llm_script.pop(0)

        async def stream(self, key, model, messages):
            yield "ok"

    _run(_agent(st, RecorderLLM()), q=q)
    for c in seen["calls"]:
        if "TOOL RESULTS" in c:
            return c
    return ""


def test_low_confidence_rag_flags_web_fallback():
    """Weak lexical match -> the tool result carries a LOW-confidence note so
    the model falls back to web search instead of guessing."""
    with tempfile.TemporaryDirectory() as td:
        st = _store_with_weak_doc(td)
        res = _captured_tool_results([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"quantum entanglement research funding"}}',
        ], st)
        assert "RETRIEVAL CONFIDENCE: LOW" in res, res
        assert "web_search" in res, res
        st.close()


def test_high_confidence_rag_has_no_fallback_note():
    """Docs that actually answer the query -> no fallback note, so the model
    answers from documents without an unnecessary web search."""
    with tempfile.TemporaryDirectory() as td:
        st = _store_with_doc(td)  # text contains "answer" / "42"
        res = _captured_tool_results([
            '{"thought":"search","tool":"search_documents","tool_input":{"query":"what is the answer"}}',
        ], st, q="what is the answer")
        assert "RETRIEVAL CONFIDENCE" not in res, res
        st.close()


def test_rag_to_web_fallback_path():
    """End-to-end: weak docs -> web_search -> answer. The agent must be able to
    chain the fallback (both tools used, answer grounded in web results)."""
    def fake_ws(q, n=5):
        return "[Source: web search via Tavily]\n[1] Entanglement explained\nQuantum entanglement is a physical phenomenon.\nhttps://example.com/q"

    with tempfile.TemporaryDirectory() as td:
        st = _store_with_weak_doc(td)
        llm = FakeLLM([
            '{"thought":"docs weak","tool":"search_documents","tool_input":{"query":"quantum entanglement"}}',
            '{"thought":"fallback to web","tool":"web_search","tool_input":{"query":"quantum entanglement basics"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "According to current web search results, quantum entanglement is a physical phenomenon.",
        ])
        with _patch_web_search(fake_ws):
            evts = _run(_agent(st, llm), q="Explain quantum entanglement")
        tools_used = [e["tool"] for e in evts if e["type"] == "tool"]
        assert tools_used == ["search_documents", "web_search"], tools_used
        done = [e for e in evts if e["type"] == "done"][0]
        assert "web search" in done["answer"].lower(), done["answer"]
        st.close()


def test_memory_tool_update():
    """'Change my preference' -> op=update replaces the stored fact in place."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        row = st.add_memory("the user prefers concise answers", "preference")
        llm = FakeLLM([
            '{"thought":"update","tool":"memory","tool_input":{"op":"update","id":%d,'
            '"fact":"the user prefers very short answers"}}' % row["id"],
            '{"thought":"done","tool":null,"tool_input":{}}',
            "Updated.",
        ])
        _run(_agent(st, llm), q="Actually, I prefer very short answers now")
        facts = st.list_memory()
        assert len(facts) == 1, facts
        assert "very short" in facts[0]["fact"], facts
        st.close()


def test_final_prompt_requires_web_attribution():
    """Web answers must be distinguished from document answers (no fake doc
    citations for web info)."""
    from ragchat.agent import FINAL_RAG_TPL

    assert "web search" in FINAL_RAG_TPL.lower()


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
