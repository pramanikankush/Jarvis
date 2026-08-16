"""The Jarvis agent: one loop, no multi-agent machinery (spec: "Do not create
unnecessary multi-agent architecture").

Flow per turn:
  1. decide (LLM, JSON): which tool, if any — search_documents / calculate /
     web_search / analyze_spreadsheet / run_python
  2. act: run the tool, yield a "tool" event, append the observation
  3. repeat (bounded) until the model says no tool is needed
  4. answer: if tools/sources were used -> generate in full, run Self-RAG
     (relevant? uses sources? supported? -> re-retrieve / correct once),
     then stream the verified text to the UI. Plain chat streams live.

Why JSON-mode routing instead of native tool-calling: one code path that works
across every Groq chat model, and it is trivially testable with a fake LLM.
"""
import asyncio
import datetime
import json
import logging
import os
import re

from . import memory, registry, retrieval, tools
from .llm import parse_json

log = logging.getLogger("jarvis.agent")

MAX_STEPS = 5
RAG_K = 6
ANSWER_MAX_TOKENS = 1500
CORRECTION_NOTE = ("\n\n> ⚠ _Parts of the above may not be fully supported by your "
                   "documents — I could not verify every claim against the sources._")

SYSTEM_TPL = """You are Jarvis, a personal AI assistant. Help the user with questions, their uploaded documents, spreadsheets, calculations, and web lookups. Be concise and direct.

{memory_block}

AVAILABLE TOOLS — choose at most one per step; respond with ONLY a JSON object:
{{
  "thought": "short reasoning",
  "tool": "tool_name or null",
  "tool_input": {{ ...arguments... }}
}}

- search_documents: {search_documents}
- calculate: {calculate}
- web_search: {web_search}
- analyze_spreadsheet: {analyze_spreadsheet}
- run_python: {run_python}
- memory: {memory}
- time: {time}

RULES:
1. For questions about the user's uploaded files, call search_documents first.
2. If search_documents finds nothing relevant, you may call it again with a rewritten query, or use web_search.
3. Never claim facts from documents you have not searched.
4. For spreadsheets: if you don't know the columns, call analyze_spreadsheet with op='info' first, then use the exact column names shown.
5. Do NOT repeat the same call with the same arguments. It IS correct to call the same tool again with NEW arguments when you need more data (e.g. a spreadsheet op other than 'info', or a rewritten search query).
6. Use web_search ONLY when the question genuinely needs live information: current news, prices, versions, live/real-time data, recent events, or unfamiliar topics where your knowledge may be outdated or incomplete. Do NOT search for simple, general, or well-known questions you can answer reliably from your own knowledge (basic math, common facts, definitions, coding basics) — and never search again when search_documents already gave you what you need.
7. If the user tells you a new fact about themselves ("I prefer...", "my name is...", "I work on...") or CHANGES something you know about them, call the memory tool (op='add' for new facts, op='update' for changes). Do not just acknowledge it — persist it.
8. When you have enough information — or no tool applies — set "tool" to null. The final answer is generated afterwards, so do not include the answer text here.
9. Output must be valid JSON only (json object)."""

FINAL_RAG_TPL = """You are Jarvis, a personal AI assistant.

{memory_block}

TOOL ACTIVITY THIS TURN (results from tools you chose):
{tool_log}

Answer the user's question using the tool results above. RULES:
1. Base your answer on the tool results; cite document sources inline as [n] (the numbers in the search results).
2. If the tool results do not contain the answer, say clearly: "I couldn't find that in your documents." and, only if relevant, suggest a web search.
3. Be concise but complete; quote exact figures from sources.
4. For spreadsheet results, summarize the numbers; mention a chart only if one was produced.
5. If the tool results came from live web search (they are tagged "web search via"), say so in the answer — e.g. "According to current web search results…" — so the user can tell live information from your own knowledge."""

PLAIN_TPL = """You are Jarvis, a personal AI assistant. Help the user with questions, their uploaded documents, spreadsheets, calculations, and web lookups. Be concise and direct.

{memory_block}"""

VERIFY_PROMPT = """You verify an assistant answer against retrieved sources.

Question: {question}

Retrieved sources:
{sources}

Assistant answer:
{answer}

Return ONLY a JSON object:
{{"relevant": true_or_false, "supported": true_or_false, "missing_info": "what info is missing to answer, or null"}}
- relevant: do the retrieved sources actually address the question?
- supported: is every claim in the answer grounded in the sources (no hallucination)?
- If relevant is false, set supported to false too."""

CORRECT_PROMPT = (
    "Your previous answer may contain claims not supported by the sources. "
    "Answer the question again using ONLY the sources, citing [n] for every claim. "
    "If the sources cannot support an answer, say so plainly."
)


class Agent:
    """Single-agent assistant. Dependencies are injected so every piece is
    testable in isolation (fake llm_chat / llm_stream / embed_fn)."""

    def __init__(
        self,
        *,
        key: str,
        model: str,
        store,
        llm_chat=None,
        llm_stream=None,
        embed_fn=None,
        get_sheet_path=None,
        is_disconnected=None,
    ):
        self.key = key
        self.model = model
        self.store = store
        self.llm_chat = llm_chat or self._chat_impl
        self.llm_stream = llm_stream or self._stream_impl
        self.embed_fn = embed_fn
        self.get_sheet_path = get_sheet_path or (lambda _sid: None)
        self.is_disconnected = is_disconnected or (lambda: False)
        self._last_sources: list[dict] = []
        self._chart_event: dict | None = None
        self._web_cache: dict[str, str] = {}  # per-turn dedup of web queries

    # ---------------- injected defaults (real Groq via ragchat.llm) ----------------
    async def _chat_impl(self, key, model, messages, json_mode=False, temperature=0.2, max_tokens=1200):
        from . import llm

        return await llm.groq_chat(key, model, messages, json_mode=json_mode,
                                   temperature=temperature, max_tokens=max_tokens)

    async def _stream_impl(self, key, model, messages):
        from . import llm

        async for delta in llm.groq_stream(key, model, messages, max_tokens=ANSWER_MAX_TOKENS):
            yield delta

    async def _check_disconnect(self) -> bool:
        """Accept both sync and async disconnect callables (tests vs FastAPI)."""
        r = self.is_disconnected()
        if asyncio.iscoroutine(r):
            r = await r
        return bool(r)

    # ---------------- main entry ----------------
    async def run(self, question: str, history: list[dict], session_id: str | None):
        """Yield events: tool / status / sources / chart / token / memory / done."""
        yield {"type": "status", "label": "Understanding request"}
        mem_block = memory.format_memory_block(
            memory.build_memory_context(self.store, session_id, question)) if session_id else ""
        tool_log: list[str] = []
        sources: list[dict] = []
        steps = 0
        last_call: str | None = None
        self._web_cache = {}  # fresh dedup cache per turn

        while steps < MAX_STEPS:
            steps += 1
            if await self._check_disconnect():
                return
            decision = await self._decide(question, history, mem_block, tool_log)
            if decision is None:
                break  # model output unparseable -> answer with what we have
            tool = decision.get("tool")
            if not tool or tool in ("answer", "respond", "null"):
                break
            if tool not in TOOL_DOCS:
                tool_log.append(f"Tool '{tool}' unknown; ignored. Pick from: {', '.join(TOOL_DOCS)}.")
                continue
            args = decision.get("tool_input") or {}
            call_sig = f"{tool}({json.dumps(args, default=str, sort_keys=True)})"
            if call_sig == last_call:
                # the model repeated the identical failing call -> stop looping
                log.warning("repeated identical tool call, stopping: %s", call_sig)
                break
            last_call = call_sig
            yield {"type": "tool", "tool": tool, "args": args}
            result, evt = await self._run_tool(tool, args)
            if evt:
                yield evt
            tool_log.append(f"Tool call {steps}: {tool}({json.dumps(args, default=str)[:300]})\nResult:\n{result[:2500]}")
            if tool == "search_documents":
                sources = self._last_sources

        if await self._check_disconnect():
            return

        if sources:
            async for evt in self._answer_with_sources(question, history, mem_block, tool_log, sources):
                yield evt
        elif tool_log:
            async for evt in self._answer_with_tools(question, history, mem_block, tool_log):
                yield evt
        else:
            yield {"type": "status", "label": "Composing answer…"}
            answer = ""
            async for evt in self._stream_plain(question, history, mem_block):
                if evt.get("type") == "token":
                    answer += evt["text"]
                yield evt
            yield {"type": "done", "answer": answer, "sources": []}

        if await self._check_disconnect():
            return
        # memory: one combined extraction call (facts + task + optional summary)
        await memory.extract_and_remember(
            self.store, session_id,
            history + [{"role": "user", "content": question}],
            self.llm_chat, self.key, self.model)
        yield {"type": "memory", "message": "Memory updated"}

    # ---------------- decision step ----------------
    async def _decide(self, question, history, mem_block, tool_log) -> dict | None:
        messages = self._messages(
            SYSTEM_TPL.format(memory_block=mem_block, **TOOL_DOCS), history, question, tool_log)
        try:
            raw = await self.llm_chat(self.key, self.model, messages, json_mode=True,
                                      temperature=0.1, max_tokens=600)
        except Exception as e:
            log.warning("decision call failed: %s", e)
            return None
        data = parse_json(raw)
        return data if isinstance(data, dict) else None

    # ---------------- tool execution ----------------
    async def _run_tool(self, tool: str, args: dict):
        """Execute a tool with one retry on failure. Returns (result_text, event_or_None)."""
        try:
            result = await self._exec_tool(tool, args)
        except Exception as e:
            log.warning("tool %s failed (retrying once): %s", tool, e)
            await asyncio.sleep(0.2)
            try:
                result = await self._exec_tool(tool, args)
            except Exception as e2:
                log.error("tool %s failed twice: %s", tool, e2)
                return f"Error: {tool} failed — {e2}", None
        evt = self._chart_event
        self._chart_event = None
        return result, evt

    async def _exec_tool(self, tool: str, args: dict):
        """Dispatch to the registered tool handler. Unknown tool -> clear error.

        Tools are declared with @registry.register on the _tool_* methods below;
        the decision prompt (TOOL_DOCS) and this dispatcher both derive from the
        registry, so adding a tool never touches this function.
        """
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            return f"Error: unknown tool '{tool}'."
        return await handler(args or {})

    # ---------------- tool handlers (registered in ragchat.registry) ----------------
    @registry.register(
        "search_documents",
        "Search the user's uploaded documents for information. Use this FIRST for questions "
        "about their files. Args: {\"query\": str, \"k\": int (default 6)}",
        category="rag",
    )
    async def _tool_search_documents(self, args: dict):
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: empty search query."
        k = min(int(args.get("k") or RAG_K), 10)
        if self.store.total_chunks() == 0:
            return "Error: no documents uploaded yet. Tell the user to upload files first."
        hits, meta = await asyncio.to_thread(retrieval.retrieve, self.store, query, k,
                                             embed_fn=self.embed_fn)
        log.info("retrieval query=%r hits=%d meta=%s", query[:80], len(hits), meta)
        if not hits:
            return ("No matches found in the documents for that query. "
                    "Consider a rewritten query or web_search.")
        self._last_sources = [
            {"doc_name": h["doc_name"], "page": h.get("page"), "text": h["text"],
             "score": round(float(h.get("score", 0)), 3)}
            for h in hits
        ]
        blocks = []
        for i, h in enumerate(hits, 1):
            loc = f" (page {h.get('page')})" if h.get("page") else ""
            blocks.append(f"[{i}] File: {h['doc_name']}{loc} Score: {h['score']:.2f}\n{h['text'][:900]}")
        weak = self._retrieval_is_weak(query, hits, meta)
        if weak:
            # deterministic low-confidence signal: tells the model to fall back
            # to web_search instead of guessing from weak matches
            blocks.append("")
            blocks.append("RETRIEVAL CONFIDENCE: LOW — the uploaded documents matched this "
                          "query weakly. Only answer from the excerpts above if the information "
                          "is clearly present; otherwise call web_search for reliable, current "
                          "information instead of guessing.")
        return "\n\n".join(blocks)

    @staticmethod
    def _retrieval_is_weak(query: str, hits: list[dict], meta: dict) -> bool:
        """Deterministic, model-independent confidence signal.

        Two weak signals:
          1. lexical coverage — the fraction of query terms that appear in the
             top retrieved chunk texts (sparse-dense hybrids treat term overlap
             as the conservative floor for relevance);
          2. retrieval degraded to vector-only (FTS found no exact terms).
        The fused RRF 'score' is normalised (best hit is always 1.0), so it
        cannot be thresholded — coverage is the honest, stable signal here.
        """
        tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 2}
        if not tokens:
            return False
        joined = " ".join(h["text"].lower() for h in hits[:4])
        covered = sum(1 for t in tokens if t in joined)
        return (covered / len(tokens)) < 0.4 or str(meta.get("fallback") or "").startswith("vector only")

    @registry.register(
        "calculate",
        'Evaluate a math expression. Args: {"expression": "e.g. (1250*0.85)/12"}',
        category="math",
    )
    async def _tool_calculate(self, args: dict):
        return await asyncio.to_thread(tools.calculate, str(args.get("expression") or ""))

    @registry.register(
        "web_search",
        'Search the web for current information (news, prices, versions, live data, or when '
        'your knowledge may be outdated). Args: {"query": str, "max_results": int (default 5)}',
        category="web",
    )
    async def _tool_web_search(self, args: dict):
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: empty search query."
        try:
            n = int(args.get("max_results") or 5)
        except (TypeError, ValueError):
            n = 5
        cache_key = " ".join(query.lower().split())  # normalize whitespace/case
        if cache_key in self._web_cache:
            log.info("web_search deduplicated (already searched this turn): %r", query[:120])
            return self._web_cache[cache_key]
        result = await asyncio.to_thread(tools.web_search, query, n)
        self._web_cache[cache_key] = result
        return result

    @registry.register(
        "analyze_spreadsheet",
        "Analyze an uploaded spreadsheet. Args: {\"sheet_id\": int, \"op\": \"info|stats|groupby|"
        "filter|anomalies|chart\", \"column\": str, \"by\": str (groupby), \"agg\": "
        "\"mean|sum|count|min|max\" (groupby), \"condition\": str (filter, e.g. 'price > 100'), "
        "\"chart_type\": \"bar|line|hist|box\" (chart)}",
        category="data",
    )
    async def _tool_analyze_spreadsheet(self, args: dict):
        return await self._spreadsheet(args)

    @registry.register(
        "run_python",
        "Run pandas/numpy Python code (restricted sandbox). `df` holds the sheet when sheet_id is "
        "given. Args: {\"code\": str, \"sheet_id\": int or null}",
        category="data",
    )
    async def _tool_run_python(self, args: dict):
        return await self._run_python(args)

    @registry.register(
        "memory",
        'Manage long-term user memory. Args: {"op": "list"} to list stored facts; '
        '{"op": "add", "fact": "the user prefers concise answers", "kind": '
        '"preference|personal|project|goal|other"} to remember; {"op": "update", "fact": '
        '"new text"} to change a stored fact (optionally with "id": N if several match); '
        '{"op": "forget", "text": "concise"} to delete matching fact(s); {"op": "delete", '
        '"id": N} to delete by id. Use when the user says "remember...", "change/update...", '
        '"forget...", or asks what you know about them.',
        category="memory",
    )
    async def _tool_memory(self, args: dict):
        op = str(args.get("op") or "list").strip().lower()
        if op == "list":
            facts = self.store.list_memory(limit=30)
            if not facts:
                return "No facts stored yet."
            return "\n".join(f"[{f['id']}] ({f['kind']}) {f['fact']}" for f in facts)
        if op == "add":
            fact = str(args.get("fact") or "").strip()
            if not fact:
                return "Error: 'fact' is required for op='add'."
            kind = str(args.get("kind") or "other").strip().lower()
            if kind not in memory.KINDS:
                kind = "other"
            row = self.store.add_memory(fact, kind)
            return f"Already remembered: {fact}" if row is None else f"Remembered: {fact} ({kind})"
        if op == "update":
            fact = str(args.get("fact") or "").strip()
            if not fact:
                return "Error: 'fact' is required for op='update'."
            kind = str(args.get("kind") or "").strip().lower()
            if kind and kind not in memory.KINDS:
                kind = ""
            if args.get("id") not in (None, "", 0):
                try:
                    mid = int(args.get("id"))
                except (TypeError, ValueError):
                    return "Error: 'id' must be a number for op='update'."
                ok = self.store.update_memory(mid, fact, kind or None)
                return f"Updated memory {mid}: {fact}" if ok else f"Error: no fact with id {mid}."
            # no id -> match by text (same disambiguation as forget)
            text = str(args.get("text") or fact).strip()
            matches = [f for f in self.store.list_memory(limit=500)
                       if text.lower() in f["fact"].lower()]
            if not matches:
                return "No matching fact found. Use op='list' to see what I remember."
            if len(matches) > 1:
                lines = "\n".join(f"[{f['id']}] ({f['kind']}) {f['fact']}" for f in matches)
                return f"Multiple facts match — call op='update' with one id:\n{lines}"
            ok = self.store.update_memory(matches[0]["id"], fact, kind or None)
            return f"Updated memory {matches[0]['id']}: {fact}" if ok else f"Error: could not update."
        if op in ("forget", "delete"):
            if op == "delete":
                try:
                    mid = int(args.get("id") or 0)
                except (TypeError, ValueError):
                    return "Error: 'id' must be a number for op='delete'."
                ok = self.store.delete_memory(mid)
                return f"Forgotten (id {mid})." if ok else f"Error: no fact with id {mid}."
            text = str(args.get("text") or "").strip()
            if not text:
                return "Error: 'text' is required for op='forget'."
            matches = [f for f in self.store.list_memory(limit=500)
                       if text.lower() in f["fact"].lower()]
            if not matches:
                return "No matching facts found. Use op='list' to see what I remember."
            if len(matches) > 1:
                lines = "\n".join(f"[{f['id']}] ({f['kind']}) {f['fact']}" for f in matches)
                return f"Multiple facts match — call op='delete' with one id:\n{lines}"
            self.store.delete_memory(matches[0]["id"])
            return f"Forgotten: {matches[0]['fact']}"
        return "Error: unknown memory op (list|add|update|forget|delete)."

    @registry.register(
        "time",
        'Get the current UTC date and time. Useful for "what time is it", "what day is it", '
        "or deciding whether a question needs current information. Args: {}",
        category="general",
    )
    async def _tool_time(self, args: dict):
        now = datetime.datetime.now(datetime.timezone.utc)
        return (f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')}).")

    async def _spreadsheet(self, args: dict) -> str:
        from . import spreadsheet

        sid = int(args.get("sheet_id") or 0)
        path = self.get_sheet_path(sid)
        if not path:
            return (f"Error: spreadsheet {sid} not found — ask the user to upload it "
                    "(.csv or .xlsx) first.")
        df = await asyncio.to_thread(spreadsheet.load_sheet, path,
                                     os.path.splitext(path)[1].lower())
        op = str(args.get("op") or "info").strip()
        column = str(args.get("column") or "")
        if op == "info":
            info = await asyncio.to_thread(spreadsheet.sheet_info, df)
            lines = [f"Sheet has {info['rows']} rows × {info['cols']} columns."]
            lines.append("| column | dtype | non-null | unique | sample |")
            lines.append("|---|---|---|---|---|")
            for c in info["columns"]:
                lines.append(f"| {c['name']} | {c['dtype']} | {c['non_null']} | {c['unique']} | {c['sample'] or '—'} |")
            lines.append("")
            lines.append("NEXT STEPS: to answer questions, call again with op='stats' (with column) for summary stats, "
                         "op='groupby' (with by and agg='sum'/'mean'/'count') for aggregations, "
                         "op='filter' (with condition) to filter rows, or op='chart' (with chart_type and column) for a chart. "
                         "Use the EXACT column names listed above.")
            return "\n".join(lines)
        if op == "stats":
            return await asyncio.to_thread(spreadsheet.describe, df, [column] if column else None)
        if op == "groupby":
            return await asyncio.to_thread(spreadsheet.groupby, df, str(args.get("by") or ""),
                                           column or None, str(args.get("agg") or "mean"),
                                           int(args.get("limit") or 10))
        if op == "filter":
            return await asyncio.to_thread(spreadsheet.filter_rows, df,
                                           str(args.get("condition") or ""))
        if op == "anomalies":
            return await asyncio.to_thread(spreadsheet.anomalies, df, column)
        if op == "chart":
            chart_type = str(args.get("chart_type") or "bar")
            group = str(args.get("group") or "") or None
            if column not in df.columns:
                return f"Error: column '{column}' not found. Columns: {', '.join(map(str, df.columns))}"
            # validate the chart renders before promising a URL
            await asyncio.to_thread(spreadsheet.chart, df, chart_type, column, group)
            url = f"/api/sheets/{sid}/chart?type={chart_type}&column={column}&group={group or ''}"
            self._chart_event = {"type": "chart", "url": url}
            return f"Chart ready: {url}"
        return f"Error: unknown op '{op}' (info|stats|groupby|filter|anomalies|chart)."

    async def _run_python(self, args: dict) -> str:
        from . import spreadsheet, tools as t

        code = str(args.get("code") or "").strip()
        if not code:
            return "Error: empty code."
        df_json = None
        sid = args.get("sheet_id")
        if sid:
            path = self.get_sheet_path(int(sid))
            if not path:
                return f"Error: spreadsheet {sid} not found."
            df = await asyncio.to_thread(spreadsheet.load_sheet, path,
                                         os.path.splitext(path)[1].lower())
            df_json = await asyncio.to_thread(spreadsheet.to_split_json, df)
        return await asyncio.to_thread(t.run_python, code, df_json)

    # ---------------- final answers ----------------
    def _source_blocks(self, sources: list[dict]) -> str:
        blocks = []
        for i, s in enumerate(sources, 1):
            loc = f" (page {s.get('page')})" if s.get("page") else ""
            blocks.append(f"[{i}] File: {s['doc_name']}{loc} Score: {s.get('score', 0):.2f}\n{s['text'][:1200]}")
        return "\n\n".join(blocks)

    async def _answer_with_sources(self, question, history, mem_block, tool_log, sources):
        yield {"type": "sources", "sources": [dict(s) for s in sources]}
        yield {"type": "status", "label": "Reranking results…"}
        yield {"type": "status", "label": "Verifying sources…"}
        messages = self._messages(
            FINAL_RAG_TPL.format(memory_block=mem_block,
                                 tool_log="\n".join(tool_log[-6:])), history, question, [])
        # pin the sources explicitly so generation is grounded regardless of tool log
        messages[-1] = {"role": "user", "content":
            f"QUESTION: {question}\n\nSOURCES:\n{self._source_blocks(sources)}"}
        try:
            answer = await self.llm_chat(self.key, self.model, messages, temperature=0.2,
                                         max_tokens=ANSWER_MAX_TOKENS)
        except Exception as e:
            log.warning("final RAG answer failed: %s", e)
            yield {"type": "error", "message": f"Error generating answer: {e}"}
            return
        final_answer, final_sources, verdict = await self._verify(question, answer, sources, messages)
        async for evt in self._stream_text(final_answer):
            yield evt
        yield {"type": "done", "answer": final_answer, "sources": final_sources, "verdict": verdict}

    async def _answer_with_tools(self, question, history, mem_block, tool_log):
        yield {"type": "status", "label": "Composing answer…"}
        messages = self._messages(
            FINAL_RAG_TPL.format(memory_block=mem_block, tool_log="\n".join(tool_log[-6:])),
            history, question, [])
        try:
            answer = await self.llm_chat(self.key, self.model, messages, temperature=0.2,
                                         max_tokens=ANSWER_MAX_TOKENS)
        except Exception as e:
            log.warning("final tool answer failed: %s", e)
            yield {"type": "error", "message": f"Error generating answer: {e}"}
            return
        async for evt in self._stream_text(answer):
            yield evt
        yield {"type": "done", "answer": answer, "sources": []}

    async def _stream_plain(self, question, history, mem_block):
        messages = self._messages(PLAIN_TPL.format(memory_block=mem_block), history, question, [])
        try:
            async for delta in self.llm_stream(self.key, self.model, messages):
                if await self._check_disconnect():
                    return
                yield {"type": "token", "text": delta}
        except Exception as e:
            log.warning("plain stream failed: %s", e)
            yield {"type": "error", "message": f"Error generating answer: {e}"}

    # ---------------- self-RAG verification ----------------
    async def _verify(self, question, answer, sources, messages) -> tuple[str, list[dict], dict]:
        """Spec self-RAG chain: sources relevant? answer uses sources? supported?
        On failure -> correct / retrieve again (bounded to one correction)."""
        verdict = {"relevance": None, "supported": None, "corrected": False}
        if not sources:
            return answer, sources, verdict
        used = self._used_sources(answer, sources)
        # step 2 (deterministic): does the answer actually use the sources (citations)?
        if not used and len(answer) > 30:
            corrected = await self._regenerate(messages, CORRECT_PROMPT)
            if corrected:
                answer = corrected
                used = self._used_sources(answer, sources)
                verdict["corrected"] = True
        # steps 1+3 (LLM judge): relevance of sources, support of answer
        judge = await self._judge(question, sources, answer)
        verdict["relevance"] = bool(judge.get("relevant")) if judge else None
        verdict["supported"] = bool(judge.get("supported")) if judge else None
        if judge and not judge.get("supported"):
            if not verdict["corrected"]:
                corrected = await self._regenerate(messages, CORRECT_PROMPT)
                if corrected:
                    answer = corrected
                    verdict["corrected"] = True
                    verdict["supported"] = True  # regeneration replaces the answer
                else:
                    answer = answer + CORRECTION_NOTE
            else:
                answer = answer + CORRECTION_NOTE
        elif judge and not judge.get("relevant"):
            answer = (answer + "\n\n> ⚠ I could not find sources in your documents that "
                              "directly cover this question.")
        return answer, sources, verdict

    async def _judge(self, question, sources, answer) -> dict | None:
        prompt = VERIFY_PROMPT.format(
            question=question[:1500],
            sources=self._source_blocks(sources)[:6000],
            answer=answer[:4000])
        try:
            raw = await self.llm_chat(self.key, self.model,
                                      [{"role": "user", "content": prompt}],
                                      json_mode=True, temperature=0.0, max_tokens=300)
        except Exception as e:
            log.warning("self-RAG judge failed: %s", e)
            return None
        data = parse_json(raw)
        return data if isinstance(data, dict) else None

    async def _regenerate(self, messages, instruction) -> str | None:
        """One corrective regeneration; returns new answer or None on failure."""
        try:
            m = [dict(x) for x in messages]
            m.append({"role": "user", "content": instruction})
            return await self.llm_chat(self.key, self.model, m, temperature=0.2,
                                       max_tokens=ANSWER_MAX_TOKENS)
        except Exception as e:
            log.warning("corrective regeneration failed: %s", e)
            return None

    # ---------------- helpers ----------------
    def _messages(self, system: str, history: list[dict], question: str, tool_log: list[str]) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        for m in history[-6:]:
            messages.append({"role": m["role"], "content": m["content"][:2000]})
        messages.append({"role": "user", "content": question[:4000]})
        if tool_log:
            # the model must SEE the tool results to decide the next step
            messages.append({"role": "assistant", "content": "I called tools to gather information."})
            messages.append({"role": "user", "content": "TOOL RESULTS:\n" + "\n\n".join(tool_log[-4:])})
        return messages

    @staticmethod
    def _used_sources(answer: str, sources: list[dict]) -> list[dict]:
        cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer or "")} & set(range(1, len(sources) + 1))
        if cited:
            return [sources[i - 1] for i in sorted(cited)]
        return []

    async def _stream_text(self, text: str):
        """Stream a held answer out as token events (fast, sub-100ms)."""
        for i in range(0, len(text), 6):
            if await self._check_disconnect():
                return
            yield {"type": "token", "text": text[i:i + 6]}
            await asyncio.sleep(0.004)


# Tool list for the decision prompt, derived from the registry (populated by the
# @registry.register decorations on the Agent's _tool_* methods above). Adding a
# tool to the registry automatically adds it to the prompt.
TOOL_DOCS = {name: t.description for name, t in registry.all_tools().items()}
