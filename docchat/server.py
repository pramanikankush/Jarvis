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
from ragchat import auth, llm, parsing, spreadsheet, store, usagetrack, websearch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jarvis.server")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT, "static")
CONFIG_PATH = os.path.join(store.DATA_DIR, "config.json")
SHEETS_DIR = os.path.join(store.DATA_DIR, "sheets")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_SHEET_BYTES = 20 * 1024 * 1024
# Free-demo chat limit: guests and Clerk users may keep at most this many
# chats. 0 = unlimited. The local workspace (owner) is never limited, so
# pre-existing local workflows are unaffected. Set DEMO_MAX_CHATS=0 to disable.
DEMO_MAX_CHATS = int(os.environ.get("DEMO_MAX_CHATS", "3") or "0")

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


def chat_limit_reached(db, uid: str, limit: int = DEMO_MAX_CHATS) -> bool:
    """True when a *new* chat must be refused for this user. The free-demo cap
    applies only to non-local identities (uid != "" — the owner's workspace is
    never limited) and only when the user already holds `limit` chats.
    Existing chats always keep working (they are not "new")."""
    return limit > 0 and bool(uid) and len(db.list_sessions()) >= limit


def identity(request: Request) -> auth.User:
    """Resolve the current user from request headers (clerk > guest > local)."""
    return auth.resolve_user(
        authorization=request.headers.get("Authorization", ""),
        guest_id=request.headers.get("X-Guest-Id", ""),
    )


def scoped(request: Request):
    """The current user's store view — every query is scoped to their data."""
    return DB.for_user(identity(request).uid)


def sheet_path(sid: int, uid: str = "") -> str | None:
    s = DB.for_user(uid).get_sheet(sid)
    if not s:
        return None
    ext = ".xlsx" if s["kind"] == "xlsx" else ".csv"
    return os.path.join(SHEETS_DIR, f"{sid}{ext}")


def sheet_filepath(sid: int, ext: str) -> str:
    return os.path.join(SHEETS_DIR, f"{sid}{ext}")


# ---------------- identity & state ----------------
@app.get("/api/me")
async def api_me(request: Request):
    """The current identity: uid (stable workspace key), source, name.
    Frontend uses this to render the right account chip / headers."""
    return {"me": identity(request).to_dict(), "clerk_enabled": auth.clerk_enabled()}


@app.get("/api/state")
async def api_state(request: Request):
    db = scoped(request)
    return {
        "key_set": bool(_config["groq_key"]),
        "tavily_set": bool(_config.get("tavily_key")),
        "embedding": llm.embed_state(),
        "model": _config.get("model") or llm.DEFAULT_MODEL,
        "tts_voice": _config.get("tts_voice") or llm.TTS_VOICE,
        "voices": llm.TTS_VOICES,
        "usage": usagetrack.report(_config.get("model") or llm.DEFAULT_MODEL),
        "me": identity(request).to_dict(),
        "clerk_enabled": auth.clerk_enabled(),
        "clerk_pk": auth.CLERK_PUBLISHABLE_KEY if auth.clerk_enabled() else "",
        "clerk_domain": auth._publishable_domain() if auth.clerk_enabled() else "",
        "docs": db.list_docs(),
        "sessions": db.list_sessions(),
        "sheets": db.list_sheets(),
        "memory_count": len(db.list_memory()),
        "demo_max_chats": DEMO_MAX_CHATS if identity(request).uid else 0,
    }


@app.get("/api/config")
async def get_config(request: Request):
    return {
        "key_set": bool(_config["groq_key"]),
        "tavily_set": bool(_config.get("tavily_key")),
        "model": _config.get("model") or llm.DEFAULT_MODEL,
        "tts_voice": _config.get("tts_voice") or llm.TTS_VOICE,
        "voices": llm.TTS_VOICES,
        "usage": usagetrack.report(_config.get("model") or llm.DEFAULT_MODEL),
        "me": identity(request).to_dict(),
        "clerk_enabled": auth.clerk_enabled(),
        "clerk_pk": auth.CLERK_PUBLISHABLE_KEY if auth.clerk_enabled() else "",
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
    db = scoped(request)
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
    existing = {d["name"] for d in db.list_docs()}
    base, ext = os.path.splitext(name)
    i = 2
    while name in existing:
        name = f"{base} ({i}){ext}"
        i += 1
    doc = await asyncio.to_thread(db.add_doc, name, len(data), chunks, list(vecs))
    return JSONResponse({"ok": True, "doc": doc, "docs": db.list_docs()})


@app.delete("/api/docs/{doc_id}")
async def delete_doc(doc_id: int, request: Request):
    db = scoped(request)
    if not db.delete_doc(doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True, "docs": db.list_docs()}


# ---------------- spreadsheets ----------------
@app.post("/api/sheets")
async def upload_sheet(request: Request):
    db = scoped(request)
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
    row = db.add_sheet(name, ext.lstrip("."), 0, 0)
    path = sheet_filepath(row["id"], ext)
    with open(path, "wb") as f:
        f.write(data)
    # validate it actually parses — explain instead of crashing (spec)
    try:
        df = await asyncio.to_thread(spreadsheet.load_sheet, path, ext)
        info = await asyncio.to_thread(spreadsheet.sheet_info, df)
        db.update_sheet_rows(row["id"], info["rows"], info["cols"])
        row = db.get_sheet(row["id"])
    except Exception as e:
        db.delete_sheet(row["id"])
        os.unlink(path)
        raise HTTPException(400, f"Could not parse spreadsheet: {e}")
    return {"ok": True, "sheet": row, "sheets": db.list_sheets()}


@app.get("/api/sheets")
async def list_sheets(request: Request):
    return {"sheets": scoped(request).list_sheets()}


@app.delete("/api/sheets/{sid}")
async def delete_sheet(sid: int, request: Request):
    db = scoped(request)
    if not db.delete_sheet(sid):
        raise HTTPException(404, "Spreadsheet not found")
    for ext in (".csv", ".xlsx"):
        p = sheet_filepath(sid, ext)
        if os.path.exists(p):
            os.unlink(p)
    return {"ok": True, "sheets": db.list_sheets()}


@app.get("/api/sheets/{sid}/chart")
async def sheet_chart(sid: int, request: Request, type: str = "bar", column: str = "", group: str = "", uid: str = ""):
    # charts are embedded via <img src>, which cannot send auth headers, so the
    # identity may also arrive as a query param (uid is not a secret — it is the
    # same stable key the client already holds)
    user = identity(request)
    if user.source == "local" and uid:
        user = auth.User(uid=uid, source="guest" if uid.startswith("guest:") else "clerk", name="")
    path = sheet_path(sid, user.uid)
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
async def get_memory(request: Request):
    return {"memory": scoped(request).list_memory()}


@app.post("/api/memory")
async def add_memory(request: Request):
    db = scoped(request)
    body = await json_body(request)
    fact = (body.get("fact") or "").strip()
    if not fact:
        raise HTTPException(400, "Empty fact")
    kind = (body.get("kind") or "fact").strip().lower()
    if kind not in ("preference", "personal", "project", "goal", "other", "fact"):
        kind = "fact"
    row = db.add_memory(fact, kind)
    if row is None:
        return {"ok": True, "duplicate": True, "memory": db.list_memory()}
    return {"ok": True, "memory": db.list_memory()}


@app.delete("/api/memory/{mid}")
async def delete_memory(mid: int, request: Request):
    db = scoped(request)
    if not db.delete_memory(mid):
        raise HTTPException(404, "Memory not found")
    return {"ok": True, "memory": db.list_memory()}


# ---------------- sessions ----------------
@app.post("/api/sessions")
async def new_session(request: Request):
    db = scoped(request)
    u = identity(request)
    if chat_limit_reached(db, u.uid):
        raise HTTPException(
            429,
            f"Free demo limited to {DEMO_MAX_CHATS} chats — delete an old chat to start a new one.",
        )
    return db.create_session("New chat")


@app.get("/api/sessions")
async def list_sessions(request: Request, q: str = ""):
    db = scoped(request)
    sessions = db.search_sessions(q) if q.strip() else db.list_sessions()
    return {"sessions": sessions}


@app.patch("/api/sessions/{sid}")
async def patch_session(sid: str, request: Request):
    db = scoped(request)
    body = await json_body(request)
    if not db.get_session(sid):
        raise HTTPException(404, "Session not found")
    title = (body.get("title") or "").strip()
    if title:
        db.set_title(sid, title[:200])
    if "pinned" in body:
        db.set_pinned(sid, bool(body["pinned"]))
    return {"ok": True, "session": db.get_session(sid)}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str, request: Request):
    if not scoped(request).delete_session(sid):
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
async def session_messages(sid: str, request: Request):
    db = scoped(request)
    if not db.get_session(sid):
        raise HTTPException(404, "Session not found")
    return {"messages": db.messages(sid)}


@app.get("/api/sessions/{sid}/export")
async def export_session(sid: str, request: Request):
    db = scoped(request)
    data = db.export_session(sid)
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
    user = identity(request)
    db = DB.for_user(user.uid)

    async def gen():
        try:
            if sid and db.get_session(sid):
                session_id = sid
                history = [m for m in db.messages(sid) if m["role"] in ("user", "assistant")][-20:]
            else:
                # free-demo cap: a new chat needs a free slot (existing chats
                # keep working); the owner's local workspace is never limited
                if chat_limit_reached(db, user.uid):
                    yield sse_error(
                        f"Free demo limited to {DEMO_MAX_CHATS} chats — "
                        "delete an old chat to start a new one."
                    )
                    yield sse({"type": "done", "answer": "", "sources": []})
                    return
                session_id = db.create_session(question[:70] or "New chat")["id"]
                history = []

            key = _config.get("groq_key")
            if not key:
                yield sse_error("No Groq API key configured — open Settings (gear icon, bottom-left) and paste your key.")
                yield sse({"type": "done", "answer": "", "sources": []})
                return

            db.add_message(session_id, "user", question)
            yield sse({"type": "start", "session_id": session_id})

            a = agent_mod.Agent(
                key=key,
                model=_config.get("model") or llm.DEFAULT_MODEL,
                store=db,
                embed_fn=llm.embed_query,
                get_sheet_path=lambda s: sheet_path(s, user.uid),
                is_disconnected=lambda: request.is_disconnected(),
                uid=user.uid,
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
                db.add_message(session_id, "assistant", answer, kept)
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


@app.get("/sign-in")
async def sign_in_page():
    """Standalone Clerk sign-in page. The app/landing open Clerk's modal by
    default; when the modal cannot mount (some browsers), Clerk redirects to
    signInUrl/signUpUrl — we serve our own page so auth never hits Clerk's
    hosted pages (which can 404 for some instances)."""
    return FileResponse(os.path.join(STATIC_DIR, "signin.html"))


@app.get("/sign-up")
async def sign_up_page():
    """Standalone Clerk sign-up page (see sign-in for rationale)."""
    return FileResponse(os.path.join(STATIC_DIR, "signin.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
