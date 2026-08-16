"""FastAPI server: agent chat (SSE), documents, spreadsheets, voice, memory,
config. Serves the web UI. Every external capability fails with a clear
message instead of crashing (spec: failure handling).
"""
import asyncio
import json
import logging
import os

import envfile

envfile.load_env()  # MUST run before ragchat imports (llm.py reads env at import time)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from ragchat import agent as agent_mod  # noqa: E402
from ragchat import llm, parsing, spreadsheet, store, usagetrack, websearch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jarvis.server")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT, "static")
CONFIG_PATH = os.path.join(store.DATA_DIR, "config.json")
SHEETS_DIR = os.path.join(store.DATA_DIR, "sheets")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_SHEET_BYTES = 20 * 1024 * 1024

DB = store.Store()
_config = {"groq_key": "", "model": llm.DEFAULT_MODEL, "tts_voice": llm.TTS_VOICE,
           "tavily_key": ""}


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k in ("groq_key", "model", "tts_voice", "tavily_key"):
            if k in saved and saved[k]:
                _config[k] = saved[k]
    except (FileNotFoundError, ValueError):
        pass
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:  # environment variable always wins (docker/CI friendly)
        _config["groq_key"] = env_key
    # Tavily: the env var (TAVILY_API_KEY) wins over the UI-saved key. We hand
    # the saved key to websearch as a fallback — never write to os.environ,
    # so a UI clear can actually take effect.
    env_tavily = os.environ.get("TAVILY_API_KEY")
    if env_tavily:
        _config["tavily_key"] = env_tavily
    websearch.set_configured_key(_config.get("tavily_key") or "")
    if _config["tts_voice"] not in llm.TTS_VOICES:
        _config["tts_voice"] = llm.TTS_VOICE


def save_config() -> None:
    os.makedirs(store.DATA_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_config, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


load_config()

app = FastAPI(title="Jarvis")


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_error(message: str) -> str:
    return sse({"type": "error", "message": message})


async def json_body(request: Request) -> dict:
    """Parse a JSON request body; return 400 with a useful message instead of
    letting a malformed/undecodable body crash the endpoint with a raw 500."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON request body.")
    return body if isinstance(body, dict) else {}


def get_key() -> str:
    key = _config.get("groq_key") or ""
    if not key:
        raise HTTPException(400, "No Groq API key configured — open Settings and paste your key.")
    return key


def sheet_path(sid: int) -> str | None:
    s = DB.get_sheet(sid)
    if not s:
        return None
    ext = ".xlsx" if s["kind"] == "xlsx" else ".csv"
    return os.path.join(SHEETS_DIR, f"{sid}{ext}")


def sheet_filepath(sid: int, ext: str) -> str:
    return os.path.join(SHEETS_DIR, f"{sid}{ext}")


# ---------------- state & config ----------------
@app.get("/api/state")
async def api_state():
    return {
        "key_set": bool(_config["groq_key"]),
        "tavily_set": bool(_config.get("tavily_key")),
        "embedding": llm.embed_state(),
        "model": _config.get("model") or llm.DEFAULT_MODEL,
        "tts_voice": _config.get("tts_voice") or llm.TTS_VOICE,
        "voices": llm.TTS_VOICES,
        "usage": usagetrack.report(_config.get("model") or llm.DEFAULT_MODEL),
        "docs": DB.list_docs(),
        "sessions": DB.list_sessions(),
        "sheets": DB.list_sheets(),
        "memory_count": len(DB.list_memory()),
    }


@app.get("/api/config")
async def get_config():
    return {
        "key_set": bool(_config["groq_key"]),
        "tavily_set": bool(_config.get("tavily_key")),
        "model": _config.get("model") or llm.DEFAULT_MODEL,
        "tts_voice": _config.get("tts_voice") or llm.TTS_VOICE,
        "voices": llm.TTS_VOICES,
        "usage": usagetrack.report(_config.get("model") or llm.DEFAULT_MODEL),
    }


@app.post("/api/config")
async def post_config(request: Request):
    body = await json_body(request)
    if "groq_key" in body:
        _config["groq_key"] = (body.get("groq_key") or "").strip()
    if "tavily_key" in body:
        _config["tavily_key"] = (body.get("tavily_key") or "").strip()
    if body.get("model"):
        _config["model"] = str(body["model"]).strip()
    if body.get("tts_voice") in llm.TTS_VOICES:
        _config["tts_voice"] = str(body["tts_voice"]).strip()
    save_config()
    load_config()  # re-applies env overrides (GROQ_API_KEY / TAVILY_API_KEY) on top of file
    return {"ok": True, "key_set": bool(_config["groq_key"]), "model": _config["model"],
            "tts_voice": _config["tts_voice"],
            "tavily_set": bool(_config.get("tavily_key"))}


@app.get("/api/models")
async def models():
    key = _config.get("groq_key")
    if key:
        live = await llm.groq_models(key)
        if live:
            return {"models": live}
    return {"models": llm.DEFAULT_GROQ_MODELS}


# ---------------- documents ----------------
@app.post("/api/docs")
async def upload_doc(request: Request):
    form = await request.form()
    up = form.get("file")
    if not up or not up.filename:
        raise HTTPException(400, "No file received")
    name = os.path.basename(str(up.filename).replace("\\", "/"))
    if not name:
        raise HTTPException(400, "Empty filename")
    data = await up.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB)")
    try:
        pages = await asyncio.to_thread(parsing.parse, name, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    chunks = [(page, c) for page, text in pages for c in parsing.chunk_text(text)]
    if not chunks:
        raise HTTPException(400, "No extractable text found in this file")
    try:
        vecs = await asyncio.to_thread(llm.embed_texts, [c for _, c in chunks])
    except Exception as e:
        raise HTTPException(
            503, f"Embedding model failed to start: {e} (first run downloads ~100 MB; needs internet)"
        )
    existing = {d["name"] for d in DB.list_docs()}
    base, ext = os.path.splitext(name)
    i = 2
    while name in existing:
        name = f"{base} ({i}){ext}"
        i += 1
    doc = await asyncio.to_thread(DB.add_doc, name, len(data), chunks, list(vecs))
    return JSONResponse({"ok": True, "doc": doc, "docs": DB.list_docs()})


@app.delete("/api/docs/{doc_id}")
async def delete_doc(doc_id: int):
    if not DB.delete_doc(doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True, "docs": DB.list_docs()}


# ---------------- spreadsheets ----------------
@app.post("/api/sheets")
async def upload_sheet(request: Request):
    form = await request.form()
    up = form.get("file")
    if not up or not up.filename:
        raise HTTPException(400, "No file received")
    name = os.path.basename(str(up.filename).replace("\\", "/"))
    ext = os.path.splitext(name)[1].lower()
    if ext not in spreadsheet.ALLOWED_KINDS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: .csv, .xlsx")
    data = await up.read()
    if len(data) > MAX_SHEET_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_SHEET_BYTES // (1024 * 1024)} MB)")
    os.makedirs(SHEETS_DIR, exist_ok=True)
    row = DB.add_sheet(name, ext.lstrip("."), 0, 0)
    path = sheet_filepath(row["id"], ext)
    with open(path, "wb") as f:
        f.write(data)
    # validate it actually parses — explain instead of crashing (spec)
    try:
        df = await asyncio.to_thread(spreadsheet.load_sheet, path, ext)
        info = await asyncio.to_thread(spreadsheet.sheet_info, df)
        DB.update_sheet_rows(row["id"], info["rows"], info["cols"])
        row = DB.get_sheet(row["id"])
    except Exception as e:
        DB.delete_sheet(row["id"])
        os.unlink(path)
        raise HTTPException(400, f"Could not parse spreadsheet: {e}")
    return {"ok": True, "sheet": row, "sheets": DB.list_sheets()}


@app.get("/api/sheets")
async def list_sheets():
    return {"sheets": DB.list_sheets()}


@app.delete("/api/sheets/{sid}")
async def delete_sheet(sid: int):
    if not DB.delete_sheet(sid):
        raise HTTPException(404, "Spreadsheet not found")
    for ext in (".csv", ".xlsx"):
        p = sheet_filepath(sid, ext)
        if os.path.exists(p):
            os.unlink(p)
    return {"ok": True, "sheets": DB.list_sheets()}


@app.get("/api/sheets/{sid}/chart")
async def sheet_chart(sid: int, type: str = "bar", column: str = "", group: str = ""):
    path = sheet_path(sid)
    if not path:
        raise HTTPException(404, "Spreadsheet not found")
    if type not in ("bar", "line", "hist", "box"):
        raise HTTPException(400, f"Unknown chart type '{type}' (bar|line|hist|box)")
    try:
        df = await asyncio.to_thread(spreadsheet.load_sheet, path, os.path.splitext(path)[1].lower())
        if type != "box" and not column:
            raise ValueError("'column' is required for this chart type")
        png = await asyncio.to_thread(spreadsheet.chart, df, type, column, group or None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-cache"})


# ---------------- voice ----------------
@app.post("/api/stt")
async def stt(request: Request):
    key = get_key()
    form = await request.form()
    up = form.get("file")
    if not up or not up.filename:
        raise HTTPException(400, "No audio file received")
    data = await up.read()
    if not data:
        raise HTTPException(400, "Empty audio")
    try:
        text = await llm.groq_stt(key, data, str(up.filename))
    except Exception as e:
        raise HTTPException(502, f"Speech-to-text failed: {e}")
    return {"text": text}


@app.post("/api/tts")
async def tts(request: Request):
    key = get_key()
    body = await json_body(request)
    text = (body.get("text") or "").strip()
    voice = body.get("voice") or _config.get("tts_voice") or llm.TTS_VOICE
    if not text:
        raise HTTPException(400, "No text to speak")
    try:
        audio = await llm.groq_tts(key, text, voice)
    except Exception as e:
        raise HTTPException(502, f"Text-to-speech failed: {e}")
    return Response(content=audio, media_type="audio/wav")


# ---------------- memory ----------------
@app.get("/api/memory")
async def get_memory():
    return {"memory": DB.list_memory()}


@app.post("/api/memory")
async def add_memory(request: Request):
    body = await json_body(request)
    fact = (body.get("fact") or "").strip()
    if not fact:
        raise HTTPException(400, "Empty fact")
    kind = (body.get("kind") or "fact").strip().lower()
    if kind not in ("preference", "personal", "project", "goal", "other", "fact"):
        kind = "fact"
    row = DB.add_memory(fact, kind)
    if row is None:
        return {"ok": True, "duplicate": True, "memory": DB.list_memory()}
    return {"ok": True, "memory": DB.list_memory()}


@app.delete("/api/memory/{mid}")
async def delete_memory(mid: int):
    if not DB.delete_memory(mid):
        raise HTTPException(404, "Memory not found")
    return {"ok": True, "memory": DB.list_memory()}


# ---------------- sessions ----------------
@app.post("/api/sessions")
async def new_session():
    return DB.create_session("New chat")


@app.get("/api/sessions")
async def list_sessions(q: str = ""):
    sessions = DB.search_sessions(q) if q.strip() else DB.list_sessions()
    return {"sessions": sessions}


@app.patch("/api/sessions/{sid}")
async def patch_session(sid: str, request: Request):
    body = await json_body(request)
    if not DB.get_session(sid):
        raise HTTPException(404, "Session not found")
    title = (body.get("title") or "").strip()
    if title:
        DB.set_title(sid, title[:200])
    if "pinned" in body:
        DB.set_pinned(sid, bool(body["pinned"]))
    return {"ok": True, "session": DB.get_session(sid)}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if not DB.delete_session(sid):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
async def session_messages(sid: str):
    if not DB.get_session(sid):
        raise HTTPException(404, "Session not found")
    return {"messages": DB.messages(sid)}


@app.get("/api/sessions/{sid}/export")
async def export_session(sid: str):
    data = DB.export_session(sid)
    if not data:
        raise HTTPException(404, "Session not found")
    filename = f"jarvis-chat-{sid[:8]}.json"
    return Response(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------- chat (agent, streaming SSE) ----------------
@app.post("/api/chat")
async def chat(request: Request):
    body = await json_body(request)
    question = (body.get("message") or "").strip()
    if not question:
        raise HTTPException(400, "Empty question")
    sid = body.get("session_id") or None

    async def gen():
        try:
            if sid and DB.get_session(sid):
                session_id = sid
                history = [m for m in DB.messages(sid) if m["role"] in ("user", "assistant")][-20:]
            else:
                session_id = DB.create_session(question[:70] or "New chat")["id"]
                history = []

            key = _config.get("groq_key")
            if not key:
                yield sse_error("No Groq API key configured — open Settings (gear icon, bottom-left) and paste your key.")
                yield sse({"type": "done", "answer": "", "sources": []})
                return

            DB.add_message(session_id, "user", question)
            yield sse({"type": "start", "session_id": session_id})

            a = agent_mod.Agent(
                key=key,
                model=_config.get("model") or llm.DEFAULT_MODEL,
                store=DB,
                embed_fn=llm.embed_query,
                get_sheet_path=sheet_path,
                is_disconnected=lambda: request.is_disconnected(),
            )
            answer, sources = "", []
            saw_done = False
            async for evt in a.run(question, history, session_id):
                if evt["type"] == "done":
                    saw_done = True
                    answer, sources = evt.get("answer", ""), evt.get("sources") or []
                    yield sse({"type": "done", "answer": answer, "sources": sources,
                               "session_id": session_id, "verdict": evt.get("verdict")})
                elif evt["type"] == "tool":
                    yield sse({"type": "tool", "tool": evt["tool"], "args": evt.get("args") or {}})
                elif evt["type"] == "chart":
                    yield sse({"type": "chart", "url": evt["url"]})
                elif evt["type"] == "sources":
                    yield sse({"type": "sources", "sources": evt["sources"]})
                elif evt["type"] == "status":
                    yield sse({"type": "status", "label": evt.get("label", "")})
                elif evt["type"] == "memory":
                    yield sse({"type": "memory", "message": evt.get("message", "")})
                elif evt["type"] == "error":
                    yield sse_error(evt.get("message", "Unknown error"))
                elif evt["type"] == "token":
                    yield sse({"type": "token", "text": evt["text"]})

            # guaranteed terminal state: the agent returned without a done event
            # (e.g. spurious disconnect) -> emit one so the UI can never hang
            if not saw_done:
                yield sse({"type": "done", "answer": answer, "sources": sources,
                           "session_id": session_id})

            if answer.strip():
                kept = [{**s, "text": (s.get("text") or "")[:500]} for s in sources]
                DB.add_message(session_id, "assistant", answer, kept)
        except Exception as e:
            log.exception("chat failed")
            yield sse_error(f"Chat error: {e}")
            yield sse({"type": "done", "answer": "", "sources": []})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- static UI ----------------
@app.get("/")
async def landing():
    """Marketing entry page; the app itself lives at /app (Try now opens it)."""
    return FileResponse(os.path.join(STATIC_DIR, "landing.html"))


@app.get("/app")
async def app_index():
    """The Jarvis chat application. Served at /app so the landing page can
    own the root URL. All app assets/APIs use absolute paths (/style.css,
    /app.js, /api/...), so nothing breaks at this sub-path."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
