"""Three-tier memory (spec: short-term conversation / long-term user memory /
current task context) without ever sending the full conversation to the LLM.

  * short-term : last N messages of the session, plus a compact rolling summary
  * long-term  : durable user facts stored in the `memory` table, de-duplicated
  * task       : one line describing what the user is currently trying to do

A single non-streaming LLM call per turn extracts all three (facts + task +
summary-when-needed). If it fails, memory is skipped silently — memory must
never break chat (failure-handling requirement).
"""
import json
import logging
import re

log = logging.getLogger("jarvis.memory")

SHORT_WINDOW = 8  # most recent messages kept verbatim
KINDS = ("preference", "personal", "project", "goal", "other")

EXTRACT_PROMPT = """You are the memory system of a personal assistant. From the conversation below, extract ONLY durable, useful information. Ignore greetings, small talk, and one-off questions.

Return JSON (json object) with:
- "facts": list of {{ "fact": str, "kind": "preference|personal|project|goal|other" }} — max 5. Facts are statements about the user that remain true later (name, preferences, constraints, projects, goals, skills, tools they use). Short, self-contained, 3rd person ("the user ..."). Do NOT include facts that are obvious from this single question.
- "task": one short phrase (max 12 words) describing what the user is currently trying to accomplish right now.
{summary_instruction}
HARD RULES:
- Extract facts ONLY from statements the USER made about themselves. Never derive facts from the assistant's replies or from tool results.
- If a user statement contradicts a stored fact, prefer the most recent statement (the user may have changed their mind).
Only output the JSON, no other text."""


def _summary_instruction(ask: bool) -> str:
    if not ask:
        return '- "summary": null'
    return ('- "summary": a 3-bullet recap of the WHOLE conversation so far '
            '(who the user is, what they asked, what was answered/decided), max 40 words total, '
            "or null if the conversation is trivial")


def build_memory_context(store, session_id: str, question: str, window: int = 5) -> dict:
    """Assemble the memory block injected into the agent's system prompt."""
    ctx: dict[str, str] = {}
    facts = store.recall_memory(question, limit=window)
    if facts:
        # ids are included so the memory tool can target an op='update'/'delete'
        ctx["facts"] = "\n".join(f"- [id={f['id']}] ({f['kind']}) {f['fact']}" for f in facts)
    meta = store.get_meta(session_id) if session_id else {"summary": "", "task": ""}
    if meta.get("task"):
        ctx["task"] = meta["task"]
    if meta.get("summary"):
        ctx["summary"] = meta["summary"]
    return ctx


def format_memory_block(ctx: dict) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get("facts"):
        parts.append("KNOWN FACTS ABOUT THE USER:\n" + ctx["facts"])
    if ctx.get("task"):
        parts.append(f"CURRENT TASK: {ctx['task']}")
    if ctx.get("summary"):
        parts.append(f"EARLIER IN THIS CONVERSATION:\n{ctx['summary']}")
    return "\n\n".join(parts)


def compact_history(history: list[dict], window: int = SHORT_WINDOW) -> tuple[list[dict], bool]:
    """Keep the last `window` messages verbatim; flag whether a summary is due."""
    if len(history) <= window + 2:
        return history, False
    return history[-window:], True


async def extract_and_remember(store, session_id: str, history: list[dict], llm_json, key: str, model: str) -> None:
    """One LLM call: extract facts + task (+ summary when history is long), persist."""
    if not history:
        return
    try:
        ask_summary = len(history) > SHORT_WINDOW + 4
        # Facts come from the USER's own statements only — assistant replies are
        # excluded from the extraction input so a misread of our own answer can
        # never become a stored fact about the user.
        user_part = [f"user: {m['content'][:600]}" for m in history[-14:] if m["role"] == "user"]
        if not user_part:
            return
        content = "CONVERSATION:\n" + "\n".join(user_part)
        # Show the extracted model what we already know, so a changed preference
        # ("concise" -> "detailed") is recognised as an update, not a new fact.
        existing = store.recall_memory(user_part[-1][6:], limit=5)
        if existing:
            lines = "\n".join(f"- ({f['kind']}) {f['fact']}" for f in existing)
            content += ("\n\nEXISTING FACTS ABOUT THE USER (check these for contradictions — "
                        "when the user's latest statement contradicts one, output the UPDATED "
                        "fact; the most recent statement wins):\n" + lines)
        messages = [
            {"role": "system", "content": EXTRACT_PROMPT.format(summary_instruction=_summary_instruction(ask_summary))},
            {"role": "user", "content": content},
        ]
        raw = await llm_json(key, model, messages)
        log.info("extraction input: %s", messages[-1]["content"][:400])
        log.info("extraction raw: %s", (raw or "")[:300])
        data = _parse_extract(raw)
    except Exception as e:
        log.warning("memory extraction skipped: %s", e)
        return

    for f in data.get("facts", [])[:5]:
        fact = str(f.get("fact", "")).strip()
        kind = str(f.get("kind", "other")).strip().lower()
        if fact and (kind in KINDS or len(fact) < 400):
            _store_fact(store, fact, kind if kind in KINDS else "other", session_id)

    task = str(data.get("task", "")).strip()
    summary = str(data.get("summary") or "").strip()
    store.set_meta(session_id, summary=summary or None, task=task or None)


def _store_fact(store, fact: str, kind: str, session_id: str | None) -> None:
    """Persist an extracted fact, REPLACING a substantially overlapping stored
    fact instead of duplicating it (a changed preference like "concise" ->
    "detailed" must update in place, never coexist as two facts)."""
    toks = set(re.findall(r"[a-z0-9]+", fact.lower()))
    best, best_overlap = None, 0
    for m in store.list_memory(limit=500):
        mtoks = set(re.findall(r"[a-z0-9]+", m["fact"].lower()))
        overlap = len(toks & mtoks)
        if overlap > best_overlap:
            best, best_overlap = m, overlap
    if best and best_overlap >= 3 and best_overlap / max(1, len(toks)) >= 0.5:
        store.update_memory(best["id"], fact, kind)
    else:
        store.add_memory(fact, kind, session_id)


def _parse_extract(raw: str) -> dict:
    raw = (raw or "").strip()
    # models sometimes wrap JSON in ```json fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        raw = raw.lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # tolerate stray prose: find the first {...} block
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        return {}
