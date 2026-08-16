# Jarvis — personal AI assistant (MVP)

A single-agent personal assistant that answers from **your documents (RAG)**, analyzes **spreadsheets**, does **web search**, **math**, and **voice input/output** — all through one chat interface. Built as an evolution of the original DocChat app (same stack, same UI patterns).

- **100% local** except the Groq API (your key): embeddings and storage run on this machine.
- One agent loop, no multi-agent machinery, no microservices.
- Works with any Groq chat model; voice uses Groq Whisper (STT) + Groq Orpheus (TTS).

## Features

| Capability | How |
|---|---|
| Text chat | Streaming SSE, ChatGPT-style UI with markdown + sources |
| Voice input | 🎤 button → MediaRecorder → Groq Whisper (`whisper-large-v3-turbo`) → same agent |
| Voice output | Groq Orpheus TTS (`canopylabs/orpheus-v1-english`); falls back to browser SpeechSynthesis |
| UI theme | Claude-inspired system from `DESIGN.md`: cream canvas, coral primary, dark navy sidebar, serif display + Inter |
| RAG | Upload PDF/DOCX/TXT/CSV/MD → parse → chunk → embed (local `bge-small-en-v1.5`) |
| Hybrid search | Vector cosine + SQLite FTS5 BM25, fused with **Reciprocal Rank Fusion** |
| Agentic RAG | The agent decides *whether* to search, rewrites queries, searches again if needed; low-confidence document matches carry a deterministic signal that steers the model to web search |
| Self-RAG | Verifies: sources relevant? answer cites them? answer supported? → corrects once, bounded |
| Spreadsheet | .csv / .xlsx via pandas: columns, stats, groupby, filters, anomaly detection, charts |
| Tools | Safe calculator (AST allowlist), web search (Tavily + `ddgs` fallback), sandboxed pandas execution |
| Memory | Long-term facts, per-session task context, rolling summary — never the full conversation. Explicit commands: "remember…", "update…", "forget…" plus a UI add box. Facts are extracted from the user's statements only, and a changed preference replaces the old fact instead of duplicating it |
| Chat management | New / history / continue / rename (double-click) / pin / search / export (JSON) |
| Tool system | Declarative registry (`ragchat/registry.py`) — tools are registered once, appear in the LLM's tool list automatically; adding a tool is one method |
| Failure handling | Every module degrades gracefully (keyword fallback, clear errors, one retry) |

## Setup & run

```bash
cd docchat
python -m venv .venv
.venv\Scripts\activate        # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # opens http://127.0.0.1:8000
```

- `http://127.0.0.1:8000/` — marketing landing page (Try now opens the app).
- `http://127.0.0.1:8000/app` — the Jarvis chat application itself.


1. **⚙ Settings → paste your Groq API key** (or set `GROQ_API_KEY`; env var wins).
2. (Optional) paste a **Tavily API key** in Settings (or set `TAVILY_API_KEY`) for higher-quality web search; without it search falls back to keyless DuckDuckGo.
2. Drop files into **Documents** (indexed with embeddings), spreadsheets into **Spreadsheets**.
3. Chat. Ask about your files, your spreadsheet, the web, or quick math. Tap 🎤 for voice.

> TTS note: Groq requires accepting terms for the Orpheus model once per org
> (https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english).
> Until then the UI falls back to browser speech synthesis automatically.

## Environment variables

Create a `.env` file next to `app.py` (see `.env.example`) — the app loads it
automatically at startup via `envfile.py` (zero-dependency loader; real
environment variables always win over the file). All of these can also just
be exported in your shell.

| Var | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Overrides the stored key (config.json) |
| `TAVILY_API_KEY` | — | Web search provider (optional; falls back to DuckDuckGo when unset) |
| `GROQ_FALLBACK_MODEL` | — | Retry model if the primary fails (resilience net) |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model |
| `GROQ_TTS_MODEL` | `canopylabs/orpheus-v1-english` | Text-to-speech model |
| `GROQ_TTS_VOICE` | `troy` | Default TTS voice |
| `DOCCHAT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `PORT` | 8000 | Server port |

## Architecture

```
Frontend (vanilla JS) ── SSE ──> FastAPI (server.py)
                                  └── Agent (ragchat/agent.py)          # one loop
                                        ├── search_documents  → retrieval.py (vector + BM25 + RRF)
                                        ├── calculate         → tools.py (AST allowlist)
                                        ├── web_search        → websearch.py (Tavily → ddgs fallback)
                                        ├── analyze_spreadsheet → spreadsheet.py (pandas)
                                        ├── run_python        → tools.py (sandboxed subprocess)
                                        ├── memory            → store (list/add/forget) + extraction
                                        ├── time              → UTC clock (current-info decisions)
                                        └── Self-RAG verify   → judge + bounded correction
                                  Store: SQLite (data/app.db) — chunks, FTS5, memory, sessions, sheets
                                  Voice: llm.py → Groq /audio/transcriptions + /audio/speech
```

### Deliberate trade-offs (why, not just what)

1. **SQLite + FTS5 instead of PostgreSQL + pgvector.** The spec's reference
   architecture lists Postgres/pgvector, but the hard requirements are
   "minimal dependencies", "no microservices", and "runnable locally". SQLite
   gives BM25 (FTS5) and vector cosine with zero extra services, reusing the
   existing `data/app.db`. The retrieval layer is isolated in
   `ragchat/retrieval.py`, so swapping in pgvector later is contained.

2. **Hand-rolled agent loop instead of LangGraph.** The whole agent is a
   linear `decide → act → observe → verify → answer` state machine with one
   LLM call per step. LangGraph would add a dependency without changing the
   control flow at this scale; the loop is deliberately small and unit-tested
   with a fake LLM. Porting to LangGraph later is mechanical.

3. **JSON-mode tool routing instead of native tool-calling.** One code path
   that works across every Groq chat model, and it is trivially testable.
   Decision calls are non-streamed (they are short); the final answer streams.

4. **"Reranking" = RRF fusion + LLM relevance gate, not a cross-encoder.**
   A cross-encoder (e.g. sentence-transformers) would add a large dependency
   for marginal gains at MVP scale. The self-RAG judge plays the reranker
   role: it prunes irrelevant retrieval before the answer is streamed.

5. **RAG answers are generated in full, verified, then streamed to the UI.**
   Verification (self-RAG) must happen *before* you see the answer, so RAG
   turns trade first-token latency for correctness. Plain chat streams live.
   This is the honest way to do self-RAG with a single stream.

6. **Sandbox is best-effort, by design.** `run_python` runs in a subprocess
   with a timeout and blocked imports/file I/O, but a determined local user
   can escape a Python sandbox. That is accepted: the app binds to
   `127.0.0.1` only and is single-user. Do not expose it to a network.

7. **One memory-extraction LLM call per turn** produces facts + task +
   summary together, and never sends the whole conversation: context = last 8
   messages + rolling summary + recalled facts + current task.

## Layout

```
app.py            launcher (port + browser open)
server.py         FastAPI: agent chat (SSE), docs, sheets, voice, memory, config
ragchat/
  agent.py        the agent loop + self-RAG verification + tool handlers
  registry.py     declarative tool registry (tool list + dispatcher derive from it)
  retrieval.py    hybrid retrieval (vector + BM25 + RRF) with fallbacks
  llm.py          Groq client: chat (stream/JSON), STT, TTS, fallback model
  store.py        SQLite: chunks+embeddings, FTS5, memory, sessions, sheets
  parsing.py      PDF/DOCX/TXT/CSV/MD extraction + chunking
  spreadsheet.py  pandas: stats, groupby, filter, anomalies, charts (matplotlib PNG)
  tools.py        calculator, web search (delegates to websearch.py), sandboxed pandas execution
  websearch.py    Tavily client (env key only) + DuckDuckGo fallback, per-query logging
  usagetrack.py   local per-model daily token tracker (persisted to data/usage.json)
  memory.py       three-tier memory (facts / task / summary)
static/           the web UI (vanilla JS)
data/             created at runtime: app.db, config.json, sheets/
tests/            per-module suites + run_all.py
```

## Tests

```bash
python tests/run_all.py          # all suites
python tests/test_agent.py       # agent loop with a scripted fake LLM (no network)
```

Tests cover: chunking/parsing, store+FTS, RRF fusion and fallbacks, the
calculator allowlist, the sandbox, spreadsheet ops + charts, memory
extraction/dedupe, the web-search provider (Tavily request format, fallback,
failures, key never leaked), the usage tracker (record/sync/rollover/corrupt
file), and the agent loop (routing, RAG citations, self-RAG correction,
tool-error fallback, tool-log visibility, duplicate-search suppression).

## Notes & limits

- Scanned/image-only PDFs have no text layer → upload fails with a clear message (no OCR).
- Files > 50 MB rejected; very large files truncated at ~300k chars.
- Web search: Tavily when `TAVILY_API_KEY` is set (see
  https://docs.tavily.com/documentation/api-reference/endpoint/search);
  otherwise keyless DuckDuckGo (`ddgs`). The agent decides *when* to search
  (current/live info, uncertain topics) and is told not to search for simple
  questions it can answer from knowledge. Searches are logged with the query
  and result count; duplicate queries within a turn are suppressed.
- Groq has no public usage/limits endpoint (verified against the API reference),
  so daily token usage is tracked locally: every response's `usage.total_tokens`
  is accumulated per model in `data/usage.json`, and 429 rate-limit bodies
  (which carry the account's own `Limit/Used` counters) correct the local count
  upward so a fresh install converges on the real usage. The sidebar footer and
  Settings show a meter; limits are the documented per-model TPD values.
- TTS voice list is a curated subset of Orpheus voices; invalid voices produce a clear error.
- Local app — bind to `127.0.0.1` only. Don't expose it to a network.
