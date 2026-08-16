"""Tests for ragchat.memory. Run: python tests/test_memory.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat import memory
from ragchat.store import Store


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        self.calls += 1
        return self.response


def test_parse_extract_handles_fences_and_prose():
    assert memory._parse_extract('```json\n{"facts": [], "task": "x"}\n```')["task"] == "x"
    assert memory._parse_extract('Sure! {"facts": [{"fact": "hi", "kind": "other"}], "task": null}')["facts"][0]["fact"] == "hi"
    assert memory._parse_extract("no json here") == {}
    assert memory._parse_extract("") == {}


def test_extract_and_remember_saves_and_dedupes():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        sid = st.create_session("s")["id"]
        payload = ('{"facts": [{"fact": "the user is a data scientist", "kind": "personal"},'
                   ' {"fact": "the user prefers python", "kind": "preference"}],'
                   ' "task": "analyze sales data", "summary": null}')
        llm = _FakeLLM(payload)
        history = [{"role": "user", "content": "I'm a data scientist and I prefer Python."}]

        import asyncio
        asyncio.run(memory.extract_and_remember(st, sid, history, llm.chat, "k", "m"))

        facts = st.list_memory()
        assert len(facts) == 2, facts
        kinds = {f["kind"] for f in facts}
        assert kinds == {"personal", "preference"}

        # same extraction again -> dedup, still 2 rows
        asyncio.run(memory.extract_and_remember(st, sid, history, llm.chat, "k", "m"))
        assert len(st.list_memory()) == 2

        meta = st.get_meta(sid)
        assert meta["task"] == "analyze sales data"

        # recall finds the python fact for a python question
        recalled = st.recall_memory("do you prefer python?", limit=5)
        assert recalled and "python" in recalled[0]["fact"]
        st.close()


def test_extract_failure_is_silent():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        sid = st.create_session("s")["id"]

        class Boom:
            async def chat(self, *a, **k):
                raise RuntimeError("llm down")

        import asyncio
        asyncio.run(memory.extract_and_remember(st, sid, [{"role": "user", "content": "hi"}],
                                                Boom().chat, "k", "m"))
        assert st.list_memory() == []  # no crash, no memory
        st.close()


def test_build_memory_context_and_block():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        sid = st.create_session("s")["id"]
        st.add_memory("the user likes coffee", "preference", sid)
        st.set_meta(sid, task="build a report")
        ctx = memory.build_memory_context(st, sid, "do you like coffee?")
        assert "coffee" in ctx["facts"]
        assert "[id=" in ctx["facts"], "fact ids must be exposed so op='update' can target them"
        assert ctx["task"] == "build a report"
        block = memory.format_memory_block(ctx)
        assert "KNOWN FACTS" in block and "coffee" in block and "CURRENT TASK" in block
        st.close()


def test_changed_preference_replaces_not_duplicates():
    """A conflicting extracted fact ('concise' -> 'detailed') must update the
    existing fact in place, never create a second row."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        st.add_memory("the user prefers concise answers", "preference")
        # simulate extraction of the changed preference
        memory._store_fact(st, "the user prefers detailed answers", "preference", None)
        facts = st.list_memory()
        assert len(facts) == 1, facts
        assert "detailed" in facts[0]["fact"], facts
        st.close()


def test_unrelated_fact_is_added_not_merged():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        st.add_memory("the user prefers concise answers", "preference")
        memory._store_fact(st, "the user works at acme corp", "personal", None)
        facts = st.list_memory()
        assert len(facts) == 2, facts
        st.close()


def test_extraction_input_is_user_statements_only():
    """Assistant replies must never reach the extraction prompt, so a misread
    of our own answer can't become a stored fact about the user."""
    import asyncio

    captured = {}

    class Rec:
        async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
            captured["content"] = messages[-1]["content"]
            return '{"facts": [], "task": "x", "summary": null}'

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        sid = st.create_session("s")["id"]
        history = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex! I'll remember that."},
            {"role": "user", "content": "I prefer bullet points in answers."},
        ]
        asyncio.run(memory.extract_and_remember(st, sid, history, Rec().chat, "k", "m"))
        content = captured.get("content", "")
        assert "My name is Alex" in content
        assert "I prefer bullet points" in content
        assert "Nice to meet you" not in content, "assistant text leaked into extraction input"
        st.close()


def test_extraction_sees_existing_facts_for_contradictions():
    """The extraction prompt must show stored facts so a change of mind
    (concise -> detailed) is emitted as an update, not a fresh fact."""
    import asyncio

    captured = {}

    class Rec:
        async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
            captured["content"] = messages[-1]["content"]
            return '{"facts": [], "task": "x", "summary": null}'

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        st.add_memory("the user prefers concise answers", "preference")
        sid = st.create_session("s")["id"]
        history = [{"role": "user", "content": "I now prefer detailed answers"}]
        asyncio.run(memory.extract_and_remember(st, sid, history, Rec().chat, "k", "m"))
        content = captured.get("content", "")
        assert "EXISTING FACTS" in content
        assert "concise answers" in content, "existing fact must be shown for contradiction checks"
        st.close()


def test_compact_history():
    short = [{"role": "user", "content": str(i)} for i in range(5)]
    win, need = memory.compact_history(short)
    assert win == short and need is False
    long = [{"role": "user", "content": str(i)} for i in range(30)]
    win, need = memory.compact_history(long)
    assert need is True and len(win) == memory.SHORT_WINDOW
    assert win[-1]["content"] == "29"


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
