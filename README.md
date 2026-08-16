# Jarvis — Agentic AI Workspace

> One agent. Your documents, your data, your web, your voice.
> A personal AI workspace that answers from **your uploaded documents** (hybrid RAG),
> **analyzes spreadsheets**, **searches the web when it truly needs live information**,
> does **math**, remembers **what you tell it**, and talks back — all through a single
> streaming chat interface.

Jarvis is a production-minded, single-agent AI workspace. Instead of a chatbot bolted
onto a vector store, it is built around one **intent router**: every message goes through
an agent that decides *what actually needs to happen* — search documents, analyze a
spreadsheet, calculate, check the web, consult memory, or just answer — then executes,
verifies, and streams a grounded answer with citations.

- **Local-first**: embeddings (ONNX) and all data live on your machine; the only cloud
  dependency is the Groq LLM API (and optionally Tavily/Clerk).
- **One agent loop** — no multi-agent machinery, no microservices, no orphan services.
- **Works with any Groq chat model**; voice uses Groq Whisper (STT) + Groq Orpheus (TTS).
- **Multi-user ready**: optional Clerk authentication with per-user workspace isolation,
  plus a guest mode and a legacy local mode — one database, scoped queries.
- **108 tests across 11 suites**, all runnable offline.

The code lives in [`docchat/`](docchat/). [`docchat/README.md`](docchat/README.md) is the
developer deep-dive (module layout, trade-offs, internals).

---

## Table of contents

1. [What it can do](#what-it-can-do)
2. [Quick start](#quick-start)
3. [Configuration](#configuration)
4. [Architecture](#architecture)
5. [How the agent decides](#how-the-agent-decides)
6. [Capabilities in depth](#capabilities-in-depth)
7. [Authentication & multi-tenant isolation](#authentication--multi-tenant-isolation)
8. [Reliability & failure handling](#reliability--failure-handling)
9. [Security](#security)
10. [Deployment](#deployment)
11. [Operations & observability](#operations--observability)
12. [Testing](#testing)
13. [Project structure](#project-structure)
14. [Design decisions](#design-decisions)
15. [Known limits](#known-limits)
16. [Roadmap](#roadmap)

---

## What it can do

| Capability | What happens | Tech |
|---|---|---|
| **Text chat** | Streaming answers (SSE), markdown, tool-activity status | FastAPI, vanilla JS |
| **Document RAG** | Upload PDF/DOCX/TXT/CSV/MD → parse → chunk → embed → hybrid retrieve → cite | pypdf, python-docx, fastembed |
| **Hybrid retrieval** | Vector cosine **+** FTS5 BM25, fused with **Reciprocal Rank Fusion**, relevance-gated | SQLite FTS5, numpy |
| **Agentic RAG** | The agent decides *whether* to search, rewrites queries, re-searches when evidence is thin | JSON-mode routing |
| **Self-RAG verification** | Sources relevant? answer cites them? supported? → one bounded correction, then stream | LLM judge |
| **Citations** | `[n]` inline, clickable source list with document name + page number; never fabricated | — |
| **Spreadsheet agent** | CSV/XLSX: column inspection, stats, groupby, filters, anomaly detection, charts | pandas, matplotlib, openpyxl |
| **Safe Python/data analysis** | Restricted pandas/numpy sandbox with timeout — no arbitrary unsafe execution | subprocess sandbox |
| **Web search** | Tavily when keyed; keyless DuckDuckGo fallback; agent only searches when needed | Tavily API, `ddgs` |
| **Calculator** | AST allowlist — no `eval` | Python `ast` |
| **Memory** | Long-term facts + per-session task context + rolling summary — never the full conversation | SQLite |
| **Voice** | 🎤 → MediaRecorder → Groq Whisper → same agent → Groq Orpheus TTS (browser speech fallback) | Groq audio APIs |
| **Chat management** | New / history / continue / rename / pin / search / export to JSON | SQLite |
| **Usage meter** | Per-model daily token tracking surfaced in the sidebar + Settings | local `usage.json` |

---

## Quick start

```bash
cd docchat
python -m venv .venv
.venv\Scripts\activate        # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # starts uvicorn and opens http://127.0.0.1:8000
```

Then:

1. **Paste your Groq API key** in **Settings** (gear icon, bottom-left) — or set
   `GROQ_API_KEY` in the environment (env var always wins).
2. Optional: add a **Tavily key** (`TAVILY_API_KEY`) for higher-quality web search.
   Without it, search automatically falls back to keyless DuckDuckGo.
3. Drop files into **Documents** (they're parsed, chunked, and embedded locally) and
   spreadsheets into **Spreadsheets**.
4. Chat. Ask about your files, your spreadsheet, the web, or quick math. Tap 🎤 for voice.

### Routes

| URL | Page |
|---|---|
| `http://127.0.0.1:8000/` | Marketing landing page — "Try now" opens the app |
| `http://127.0.0.1:8000/app` | The Jarvis workspace itself |
| `http://127.0.0.1:8000/sign-in` `/sign-up` | Standalone Clerk auth pages (fallback when the modal can't mount) |
| `http://127.0.0.1:8000/api/state` | Health/config probe (used by Render's health check) |

> **TTS note:** Groq requires accepting terms for the Orpheus model once per org
> (https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english).
> Until then, the UI falls back to browser speech synthesis automatically.

---

## Configuration

All configuration is via environment variables. Create `docchat/.env` from
[`docchat/.env.example`](docchat/.env.example) — the app loads it automatically at startup
(zero-dependency loader). **Real environment variables always win over the file.**

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Groq LLM key; overrides the key saved via the Settings UI |
| `TAVILY_API_KEY` | — | Web search provider (optional; DuckDuckGo fallback when unset) |
| `CLERK_PUBLISHABLE_KEY` | — | Clerk publishable key — enables auth (public by design) |
| `CLERK_SECRET_KEY` | — | Clerk secret key (server-side only; fallback token verifier) |
| `CLERK_FRONTEND_API` | *(derived from the publishable key)* | Override the Clerk FAPI domain |
| `DEMO_MAX_CHATS` | `3` | Max chats for guests/Clerk users; `0` = unlimited. Local owner never limited |
| `GROQ_FALLBACK_MODEL` | — | Retry model if the primary fails (resilience net) |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model |
| `GROQ_TTS_MODEL` | `canopylabs/orpheus-v1-english` | Text-to-speech model |
| `GROQ_TTS_VOICE` | `troy` | Default TTS voice |
| `DOCCHAT_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (first run downloads ~100 MB) |
| `PORT` | `8000` | Server port |

Secrets are never committed: `docchat/.env`, `docchat/data/`, and `docchat/server_*.log`
are gitignored. See [Security](#security).

---

## Architecture

```
Frontend (vanilla JS, /app)
   │  SSE streaming (text/event-stream)
   ▼
FastAPI (server.py) ── identity: Clerk JWT > guest UUID > local
   │
   ▼
Agent (ragchat/agent.py)  ─ one loop: decide → act → observe → verify → answer
   │
   ├── search_documents     → retrieval.py  (vector + BM25 + RRF + confidence signal)
   ├── calculate            → tools.py      (AST allowlist)
   ├── web_search           → websearch.py  (Tavily → DuckDuckGo fallback, per-turn dedup)
   ├── analyze_spreadsheet  → spreadsheet.py (pandas: info/stats/groupby/filter/anomalies/chart)
   ├── run_python           → tools.py      (restricted subprocess sandbox)
   ├── memory               → store.py      (list/add/update/forget/delete + extraction)
   └── time                 → UTC clock     (current-information decisions)
   │
   └── Self-RAG verify      → relevance/support judge + bounded correction
   │
Store: SQLite data/app.db — chunks + embeddings, FTS5, memory, sessions, sheets (per user_id)
Voice: llm.py → Groq /audio/transcriptions + /audio/speech
```

The agent's tools are declared once in a **declarative registry**
(`ragchat/registry.py`); the decision prompt and the dispatcher both derive from it.
Adding a tool = one decorated method — nothing else to touch.

---

## How the agent decides

Every message runs through the same loop (bounded, `MAX_STEPS = 5`):

```
User message
   │
   ▼
1. DECIDE   — LLM returns JSON: {thought, tool, tool_input}   (JSON-mode; works on every Groq model)
   │
   ▼
2. ACT      — run the chosen tool, stream a "tool" status event, append the observation
   │
   ▼
3. OBSERVE  — repeat until the model says no tool is needed (identical repeated calls are suppressed)
   │
   ▼
4. VERIFY   — if sources were retrieved: self-RAG (relevant? cited? supported?) → correct once
   │
   ▼
5. ANSWER   — RAG answers stream after verification; plain chat streams live
```

Routing rules baked into the system prompt:

- **Document questions** → `search_documents` first.
- **Spreadsheet questions** → inspect columns (`op='info'`) before querying — exact column
  names are required.
- **Math** → `calculate` (AST allowlist, no `eval`).
- **Live/current information** (news, prices, versions, recent events, unfamiliar topics) →
  `web_search`. The model is explicitly told **not** to search for simple, well-known
  questions it can answer from its own knowledge — and **never** to search again when
  documents already answered the question.
- **Personal facts** ("I prefer…", "my name is…") → `memory` persist commands.
- **Enough information / no tool applies** → answer directly.

**Uncertainty fallback:** when document retrieval is weak, a deterministic signal is
appended to the tool result (`RETRIEVAL CONFIDENCE: LOW…`) — computed from lexical term
coverage of the top chunks and whether FTS degraded to vector-only. This steers the model
to `web_search` instead of hallucinating from weak matches.

---

## Capabilities in depth

### 1. Hybrid RAG

```
Upload → parse (PDF/DOCX/TXT/CSV/MD) → chunk → embed (fastembed, local ONNX)
   → store in SQLite → hybrid retrieval → rerank → generate → cite
```

- **Vector** — numpy cosine similarity over stored embeddings (fastembed `bge-small-en-v1.5`).
- **Keyword** — SQLite FTS5 BM25, token-sanitized to avoid FTS syntax errors.
- **Fusion** — Reciprocal Rank Fusion; results are relevance-gated by the self-RAG judge
  before the answer is streamed.
- **Citations** — every RAG answer cites `[n]` inline; the UI renders a clickable source
  list with **document name**, **page number** when available, and the retrieved passage.
  Citations are only ever generated from actually-retrieved chunks — never fabricated.
- **Multiple documents** — cross-document questions work; retrieval is scoped per user.

### 2. Spreadsheet agent (pandas)

CSV/XLSX via pandas/openpyxl. The agent inspects columns and dtypes first, then runs one of
the validated operations: `info` · `stats` · `groupby` · `filter` · `anomalies` · `chart`
(bar/line/hist/box rendered as matplotlib PNG). Natural-language questions
("Which region generated the highest revenue?") are answered by chaining these ops.
For anything beyond the predefined ops, a **restricted `run_python` sandbox** executes
pandas/numpy code in a subprocess with a timeout and blocked imports/file I/O.

### 3. Memory

Three tiers, deliberately separated:
- **Long-term facts** — extracted only from what the user states; deduplicated; updated
  when a preference changes (no duplicates).
- **Per-session task context** — the current task in this conversation.
- **Rolling summary** — a compact summary instead of the full transcript.

The context sent to the LLM is **last 8 messages + rolling summary + recalled facts +
current task** — never the whole conversation. Explicit user control: *"remember…"*,
*"update…"*, *"forget…"*, plus a UI box and list/delete.

### 4. Web search (Tavily + fallback)

Tavily when `TAVILY_API_KEY` is set; otherwise keyless DuckDuckGo (`ddgs`). The agent
decides *when* to search; queries are deduplicated within a turn; every search is logged
with query and result count. Live answers are tagged ("According to current web search
results…") so the user can tell web-derived facts from the model's own knowledge.

### 5. Voice

Same agent as text — no separate voice brain. Mic → MediaRecorder → Groq Whisper STT →
agent router → answer → Groq Orpheus TTS (browser SpeechSynthesis fallback). Voice is
optional and never breaks text chat.

### 6. Usage tracking

Groq has no public usage endpoint, so Jarvis tracks tokens locally per model in
`data/usage.json` (from each response's `usage.total_tokens`). 429 rate-limit bodies carry
the account's own `Limit/Used` counters, which correct the local count upward so a fresh
install converges on real usage. The sidebar footer and Settings show the meter.

---

## Authentication & multi-tenant isolation

Jarvis supports **three identity modes** — one stable `user_id` each, resolved per request
(precedence: **clerk > guest > local**):

| Mode | How identified | Workspace key |
|---|---|---|
| **Clerk** | `Authorization: Bearer <JWT>` verified (RS256) against Clerk's JWKS, cached 1 h; API-token fallback | `clerk:<sub>` |
| **Guest** | `X-Guest-Id: <uuid>` from the browser (no account needed) | `guest:<uuid>` |
| **Local** | No identity headers — the legacy single-user workspace | `""` (never limited) |

Every store query goes through a per-user scoped view (`store.for_user(uid)`), so
documents, chats, memory, and spreadsheets are **isolated per user** — on a single SQLite
database. An invalid/expired Clerk token degrades gracefully to guest/local instead of
crashing.

**Free-demo chat limit:** `DEMO_MAX_CHATS` (default 3, `0` = unlimited) caps guests and
Clerk users at N chats. Enforced server-side at both creation points (`POST /api/sessions`
→ HTTP 429; `/api/chat` new-session branch → SSE error + terminal `done`). Existing chats
keep working; deleting a chat frees a slot. The UI shows a "x/3" meter and blocks New chat
with a clear toast. The local owner workspace is **never** limited.

**Frontend auth:** Clerk's modal is the primary sign-in path; if the modal can't mount
(some browsers), Clerk redirects to `signInUrl` — which Jarvis serves itself
(`/sign-in`, `/sign-up`), so auth never lands on a dead hosted page. Signed-in users get a
profile avatar/name chip, a profile button (account management), and Sign out; the landing
page mirrors this live via a Clerk listener.

---

## Reliability & failure handling

Every external capability degrades instead of crashing:

- **Retrieval** — FTS unavailable → vector-only; no FTS matches → keyword fallback.
- **No relevant documents** — the agent says "I couldn't find that in your documents"
  and suggests a web search; a low-confidence signal steers to `web_search`.
- **Tool failure** — one automatic retry, then a useful error in the tool log.
- **LLM failure** — plain-chat/RAG errors surface with a message; `GROQ_FALLBACK_MODEL`
  provides a retry net when configured.
- **Embedding failure** — 503 with a clear hint (first run downloads ~100 MB; needs internet).
- **Spreadsheet parse failure** — the upload is rolled back and explained (never a crash).
- **Web search failure** — Tavily fails → DuckDuckGo; both fail → graceful note.
- **Malformed JSON bodies** → 400 with a clear message (never a raw 500).
- **Guaranteed terminal state** — `/api/chat` always emits a final SSE `done` event, even
  on error or spurious disconnect, so the UI can never hang on "Thinking…".

All failures are logged with `log.exception`/`log.warning` context for debugging.

---

## Security

- **Keys live only in `docchat/.env` or environment variables** (Settings saves to
  `data/config.json`, also gitignored). Never in the repo; the only Clerk value served to
  the frontend is the **publishable** key, which is public by design. `CLERK_SECRET_KEY`
  stays server-side.
- **Upload validation** — extension allowlists (docs: PDF/DOCX/TXT/CSV/MD; sheets:
  CSV/XLSX), size caps (50 MB docs, 20 MB sheets), filename sanitization, duplicate-name
  handling, and parse-before-accept for spreadsheets.
- **No arbitrary code execution** — calculator uses an AST allowlist (no `eval`); the
  pandas sandbox runs in a subprocess with a timeout and blocked imports/file I/O.
- **Best-effort sandbox, by design** — a determined local user can escape a Python
  sandbox. That's accepted: the app binds to `127.0.0.1` only and is single-tenant when
  run locally. **Do not expose it directly to a public network** — deploy behind a proper
  host (see below) or keep it local.
- **JWT verification** — Clerk tokens verified against JWKS with PyJWT (RS256), keys
  cached with a 1 h TTL; secret key never leaves the server.

---

## Deployment

### Option A — Render (native Python, no Docker) — recommended free path

1. Push this repo to GitHub (`main`).
2. Render → **New + → Web Service** → connect the repo.
3. Settings:
   - **Root Directory:** `docchat`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -m uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
   - **Health Check Path:** `/api/state`
4. Add env vars in the Render dashboard (never in the repo): `GROQ_API_KEY`,
   `TAVILY_API_KEY`, `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `DEMO_MAX_CHATS`.
5. Deploy. First build takes a few minutes; the ~100 MB embedding model downloads on first
   use — send one "hi" right after deploy to warm it up.

**Free-tier expectations (honest):** ephemeral disk (documents/chats/memory reset on each
redeploy), ~15 min idle sleep with a ~30–60 s cold start on wake, 512 MB RAM / 0.1 CPU,
and 750 instance-hours/month (one always-on service ≈ 720). Fine for a demo/portfolio.

### Option B — Docker (for any host with Docker, or Render's Docker path)

A `Dockerfile` is included (`docchat/Dockerfile`) that installs dependencies and
**pre-caches the embedding model during the image build** — no runtime download. A
`render.yaml` blueprint documents the Render Docker path with `sync: false` env vars
(values pasted in the dashboard, never committed).

```bash
cd docchat
docker build -t jarvis .
docker run -p 8000:8000 --env-file .env jarvis
```

### Option C — Local production-ish

```bash
cd docchat
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

---

## Operations & observability

- **Logging** — structured, leveled logs from `jarvis.server`, `jarvis.agent`,
  `jarvis.auth`, `jarvis.retrieval` (`%(asctime)s %(levelname)s %(name)s`). Everything
  notable is logged: retrieval queries + hit counts, web-search triggers + result counts,
  tool failures with retry state, auth degradation.
- **Usage meter** — see [Usage tracking](#6-usage-tracking).
- **Data** — everything is one SQLite file: `docchat/data/app.db` (plus `sheets/` and
  `config.json`). Back up that directory; restore by replacing it.
- **Chat export** — any conversation can be exported to JSON (`/api/sessions/{id}/export`).

---

## Testing

```bash
cd docchat
python tests/run_all.py          # all 11 suites (108 tests)
python tests/test_agent.py       # agent loop with a scripted fake LLM — no network
```

Coverage: parsing/chunking, store + FTS, RRF fusion and fallbacks, calculator allowlist,
sandbox, spreadsheet ops + charts, memory extraction/dedupe, web-search provider (request
format, fallback, failures, key never leaked), usage tracker (record/sync/rollover/corrupt
file), auth (Clerk JWT + guest + local + degradation), demo chat limit (cap, exempt local,
delete frees slot, disable, per-user isolation), and the agent loop (routing, RAG citations,
self-RAG correction, tool-error fallback, duplicate-search suppression).

---

## Project structure

```
README.md             this document
docchat/
  app.py              launcher (port + browser open)
  server.py           FastAPI: chat (SSE), docs, sheets, voice, memory, config, auth routes
  envfile.py          zero-dependency .env loader
  requirements.txt    dependencies (pyjwt/cryptography for Clerk JWT included)
  Dockerfile          container build with pre-cached embedding model
  render.yaml         Render blueprint (env vars sync:false)
  ragchat/
    agent.py          the agent loop + self-RAG verification + tool handlers
    registry.py       declarative tool registry (prompt + dispatcher derive from it)
    retrieval.py      hybrid retrieval (vector + BM25 + RRF) with fallbacks
    llm.py            Groq client: chat (stream/JSON), STT, TTS, fallback model, embeddings
    store.py          SQLite: chunks+embeddings, FTS5, memory, sessions, sheets (per-user)
    parsing.py        PDF/DOCX/TXT/CSV/MD extraction + chunking
    spreadsheet.py    pandas: stats, groupby, filter, anomalies, charts (matplotlib PNG)
    tools.py          calculator (AST allowlist), web search, sandboxed pandas execution
    websearch.py      Tavily client + DuckDuckGo fallback, per-query logging
    usagetrack.py     per-model daily token tracker (persisted to data/usage.json)
    memory.py         three-tier memory (facts / task / summary)
    auth.py           Clerk JWT verification (JWKS) + guest/local fallback
  static/             the web UI (vanilla JS: landing page, app, sign-in/up)
  tests/              per-module suites + run_all.py
  data/               created at runtime (gitignored): app.db, config.json, sheets/, usage.json
```

---

## Design decisions

The full rationale lives in [`docchat/README.md`](docchat/README.md). The short version:

1. **SQLite + FTS5 instead of PostgreSQL + pgvector** — BM25 + vector cosine with zero
   extra services; retrieval is isolated in `retrieval.py` so a pgvector swap is contained.
2. **Hand-rolled agent loop instead of LangGraph** — a linear `decide → act → observe →
   verify → answer` state machine; one LLM call per step; unit-tested with a fake LLM.
3. **JSON-mode tool routing instead of native tool-calling** — one code path that works
   across every Groq chat model and is trivially testable.
4. **"Reranking" = RRF fusion + LLM relevance gate** — no cross-encoder dependency; the
   self-RAG judge prunes irrelevant retrieval before the answer streams.
5. **RAG answers verified before streaming** — correctness over first-token latency for
   document answers; plain chat streams live.
6. **One memory-extraction LLM call per turn** — facts + task + summary together; the full
   conversation is never sent to the model.

---

## Known limits

- Scanned/image-only PDFs have no text layer → upload fails with a clear message (no OCR).
- Files > 50 MB rejected; very large files truncated at ~300k chars.
- Local Groq embedding downloads (~100 MB) on first run — requires internet once.
- Free-tier hosting resets data on redeploy (ephemeral disk) — see [Deployment](#deployment).
- Voice TTS needs one-time Groq Orpheus terms acceptance per org; falls back to browser TTS.

---

## Roadmap

- [ ] PostgreSQL + pgvector backend (the retrieval layer is already isolated for this)
- [ ] Cross-encoder reranker for higher retrieval precision
- [ ] Per-user persistent storage on Render (paid disk) for true multi-user persistence
- [ ] Streaming charts and richer spreadsheet visualizations
- [ ] Organization/team workspaces via Clerk Organizations

---

*Built with FastAPI, Groq, fastembed, SQLite, pandas, and a lot of plain vanilla JS.
Local-first, agent-first, and honest about its limits.*
