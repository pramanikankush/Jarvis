"""Tests for the everyday toolkit: tasks, notes, converter, image URL builder,
and the store-level scoping of the new tables. No network (currency path is
exercised only through its failure/cache branches). No key required.
Run: python tests/test_toolkit.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from ragchat import tools
from ragchat.agent import Agent
from ragchat.store import Store


def test_unit_conversions():
    assert "3.1" in tools.convert("km", "mi", 5)
    assert tools.convert("m", "m", 3).startswith("3 m")
    r = tools.convert("kg", "lb", 70)
    assert "154" in r, r
    r = tools.convert("gb", "mb", 1.5)
    assert "1500" in r, r


def test_temperature_conversions():
    r = tools.convert("c", "f", 37)
    assert "98.6" in r, r
    r = tools.convert("f", "c", 212)
    assert "100" in r, r
    r = tools.convert("k", "c", 0)
    assert "-273" in r, r


def test_unknown_unit_is_a_clean_error():
    r = tools.convert("bananas", "apples", 2)
    assert r.startswith("Error:"), r
    assert "currency" in r


def test_image_url_builds_pollinations_link():
    url = tools.image_url("a robot butler, watercolor")
    assert url.startswith("https://image.pollinations.ai/prompt/")
    assert "nologo=true" in url
    assert "a%20robot" in url  # prompt is URL-encoded


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        return self.responses.pop(0)

    async def stream(self, key, model, messages):
        yield "ok"


def _agent(st, llm, images=True):
    return Agent(key="k", model="m", store=st, llm_chat=llm.chat, llm_stream=llm.stream,
                 embed_fn=lambda t: __import__("numpy").ones(8, dtype="float32") / 3,
                 images_enabled=images)


def _collect(agent, q, sid=None):
    events = []

    async def go():
        async for e in agent.run(q, [], sid):
            events.append(e)
    asyncio.run(go())
    return events


def test_tasks_tool_add_and_list():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = ScriptedLLM(["Answered."])
        _collect(_agent(st, llm), q="irrelevant")  # agent boot not needed; call tool directly
        a = _agent(st, llm)
        out = asyncio.run(a._tool_tasks({"op": "add", "text": "pay rent", "due": "friday"}))
        assert "pay rent" in out and "friday" in out
        out = asyncio.run(a._tool_tasks({"op": "list"}))
        assert "[1]" in out and "pay rent" in out
        st.close()


def test_tasks_done_and_delete():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        a = _agent(st, ScriptedLLM([]))
        asyncio.run(a._tool_tasks({"op": "add", "text": "call mom"}))
        out = asyncio.run(a._tool_tasks({"op": "done", "id": 1}))
        assert "completed" in out
        out = asyncio.run(a._tool_tasks({"op": "done", "id": 99}))
        assert out.startswith("Error:")
        out = asyncio.run(a._tool_tasks({"op": "delete", "id": 1}))
        assert "Deleted" in out
        assert st.list_tasks() == []
        st.close()


def test_notes_add_list_search():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        a = _agent(st, ScriptedLLM([]))
        asyncio.run(a._tool_notes({"op": "add", "text": "wifi password is hunter2"}))
        asyncio.run(a._tool_notes({"op": "add", "text": "standup moved to 10am"}))
        out = asyncio.run(a._tool_notes({"op": "list", "query": "wifi"}))
        assert "hunter2" in out and "standup" not in out
        st.close()


def test_generate_image_respects_toggle():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        a = _agent(st, ScriptedLLM([]), images=False)
        out = asyncio.run(a._tool_generate_image({"prompt": "a cat"}))
        assert "disabled" in out.lower()
        st.close()


def test_generate_image_emits_event():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        a = _agent(st, ScriptedLLM([]), images=True)
        out = asyncio.run(a._tool_generate_image({"prompt": "a cat"}))
        assert "Image generated" in out
        assert a._chart_event and a._chart_event["type"] == "image"
        assert a._chart_event["url"].startswith("https://image.pollinations.ai")
        st.close()


def test_toolkit_tools_in_registry():
    from ragchat.agent import TOOL_DOCS
    for name in ("summarize_url", "generate_image", "tasks", "notes", "quiz_me", "convert"):
        assert name in TOOL_DOCS, f"{name} missing from registry"
        assert TOOL_DOCS[name].strip()


def test_quiz_me_requires_documents():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))  # no docs
        a = _agent(st, ScriptedLLM([]))
        out = asyncio.run(a._tool_quiz_me({"topic": "anything"}))
        assert "no documents" in out.lower()
        st.close()


def test_quiz_me_builds_quiz_from_docs():
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        st.add_doc("notes.txt", 10, [(None, "Photosynthesis converts light into chemical energy in plants.")],
                   [np.ones(8, dtype="float32") / 3])
        a = _agent(st, ScriptedLLM(["Q1: What does photosynthesis convert? --- A: light to energy"]))
        out = asyncio.run(a._tool_quiz_me({"topic": "photosynthesis"}))
        assert "Q1" in out and "answer key" in out.lower() or "---" in out
        st.close()


def test_store_task_note_scoping():
    with tempfile.TemporaryDirectory() as td:
        s = Store(os.path.join(td, "t.db"))
        u1, u2 = s.for_user("guest:aaa"), s.for_user("guest:bbb")
        u1.add_task("private task")
        u1.add_note("private note")
        assert u2.list_tasks() == [] and u2.list_notes() == []
        assert len(u1.list_tasks()) == 1 and len(u1.list_notes()) == 1
        s.close()


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
