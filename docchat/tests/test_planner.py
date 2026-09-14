"""Tests for the Jarvis planner (multi-step plans) with a fake LLM. No network.
Run: python tests/test_planner.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat.agent import Agent
from ragchat.store import Store


class FakeLLM:
    """Pops chat responses from a script; streams from a separate script."""

    def __init__(self, chat_script, stream_text="ok"):
        self.chat_script = list(chat_script)
        self.stream_text = stream_text

    async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        if not self.chat_script:
            raise IndexError("script exhausted")
        return self.chat_script.pop(0)

    async def stream(self, key, model, messages):
        text = self.stream_text
        for i in range(0, len(text), 8):
            yield text[i:i + 8]


def _run(agent, q="plan my day", history=None, sid=None):
    events = []

    async def collect():
        async for e in agent.run(q, history or [], sid):
            events.append(e)
    asyncio.run(collect())
    return events


def _agent(st, llm):
    return Agent(key="k", model="m", store=st, llm_chat=llm.chat, llm_stream=llm.stream,
                 embed_fn=lambda t: __import__("numpy").ones(8, dtype="float32") / 3)


def test_plan_event_emitted_and_steps_tracked():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"multi-step","plan":["check the time","add a task"],"tool":"time","tool_input":{}}',
            '{"thought":"step 2","tool":"tasks","tool_input":{"op":"add","text":"review notes"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            'Here is your plan and the task I added.',
        ], stream_text="All planned out.")
        evts = _run(_agent(st, llm))
        plans = [e for e in evts if e["type"] == "plan"]
        assert plans, "no plan events emitted"
        assert plans[0]["plan"] == ["check the time", "add a task"]
        # after two tool calls the plan tracker advanced to step 2 (0-indexed)
        assert plans[-1]["step"] == 2, plans[-1]
        # the tasks tool actually ran
        assert any(t["text"] == "review notes" for t in st.list_tasks())
        st.close()


def test_replan_resets_step_counter():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        try:
            llm = FakeLLM([
                '{"thought":"p1","plan":["step a","step b"],"tool":"calculate","tool_input":{"expression":"1+1"}}',
                '{"thought":"results contradict the plan, replanning","plan":["new direction"],"tool":"calculate","tool_input":{"expression":"2*3"}}',
                '{"thought":"done","tool":null,"tool_input":{}}',
                "Replanned and computed both.",
            ], stream_text="done")
            evts = _run(_agent(st, llm))
            plans = [e for e in evts if e["type"] == "plan"]
            # each plan emits on revision AND after each completed step
            assert plans[0]["step"] == 0
            replans = [e for e in plans if e["plan"] == ["new direction"]]
            assert replans, plans
            assert replans[0]["step"] == 0, "replan must restart step tracking"
        finally:
            st.close()


def test_no_plan_field_degrades_to_single_step():
    """Decisions without a plan (legacy models) must behave exactly like before."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"calc","tool":"calculate","tool_input":{"expression":"2+2"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "4",
        ], stream_text="4")
        evts = _run(_agent(st, llm), q="what is 2+2?")
        assert not [e for e in evts if e["type"] == "plan"]
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == "4"
        st.close()


def test_malformed_plan_field_ignored():
    """A non-list or empty plan must be ignored, not crash the loop."""
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = FakeLLM([
            '{"thought":"weird","plan":"just do it","tool":"calculate","tool_input":{"expression":"2+2"}}',
            '{"thought":"empty","plan":[],"tool":null,"tool_input":{}}',
            "4",
        ], stream_text="4")
        evts = _run(_agent(st, llm))
        assert not [e for e in evts if e["type"] == "plan"]
        done = [e for e in evts if e["type"] == "done"][0]
        assert done["answer"] == "4"
        st.close()


def test_tool_log_includes_plan_step_labels():
    """Tool results shown to the model carry the plan-step context."""
    seen = {"calls": []}

    class RecorderLLM(FakeLLM):
        async def chat(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
            seen["calls"].append("\n".join(m["content"] for m in messages if "content" in m))
            return await super().chat(key, model, messages, json_mode, temperature, max_tokens)

    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        llm = RecorderLLM([
            '{"thought":"p","plan":["do math"],"tool":"calculate","tool_input":{"expression":"2+2"}}',
            '{"thought":"done","tool":null,"tool_input":{}}',
            "4",
        ], stream_text="4")
        _run(_agent(st, llm))
        assert any("plan step 1" in c for c in seen["calls"]), seen["calls"]
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
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
