"""Local free embeddings (fastembed, ONNX CPU) + Groq client (chat/STT/TTS).

The only external service is api.groq.com — everything else runs locally.
"""
import json
import logging
import os
import re
import tempfile
import time

import httpx
import numpy as np

from . import usagetrack

log = logging.getLogger("jarvis.llm")

# Groq 429 bodies embed the account's daily counters, e.g.
#   ... on tokens per day (TPD): Limit 100000, Used 98669, Requested 1570. ...
_TPD_RE = re.compile(r"(?:tokens per day|TPD)[^\n]*?Limit\s+(\d+)[^\d]+Used\s+(\d+)", re.I)


def _track_429(body: str, model: str) -> None:
    """Feed the account's own daily counters from a 429 body into the tracker.
    Best-effort: a body we cannot parse is simply ignored."""
    m = _TPD_RE.search(body or "")
    if m:
        usagetrack.sync_429(model, used=int(m.group(2)), limit=int(m.group(1)))


def _track_usage(model: str, payload: dict | None) -> None:
    """Record total_tokens from a response body (chat + streaming usage)."""
    if not isinstance(payload, dict):
        return
    usage = payload.get("usage")
    total = 0
    if isinstance(usage, dict):
        try:
            total = int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            total = 0
    if total > 0:
        usagetrack.record(model, total)

EMBED_MODEL = os.environ.get("DOCCHAT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
# Live model IDs (verified 2026-09 against https://console.groq.com/docs/models).
# The legacy llama-3.3/3.1 IDs were decommissioned by Groq on 2026-08-16
# (https://console.groq.com/docs/deprecations); DEAD_MODELS below keeps old
# saved configs from selecting them again.
DEFAULT_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",  # agentic system: built-in web search + code execution
]
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")  # resilience net

# Model IDs Groq has decommissioned (deprecations page, fetched 2026-09). Any
# of these found in a saved config is auto-migrated to DEFAULT_MODEL so a stale
# config.json or .env can never select a model that only returns 404s.
DEAD_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "qwen/qwen3-32b",
    "playai-tts",
    "playai-tts-arabic",
}


def migrate_model(model: str) -> str:
    """Map a deprecated model ID to the current default. Live IDs pass through."""
    return DEFAULT_MODEL if (model or "") in DEAD_MODELS else (model or "")

# Voice (verified against https://console.groq.com/docs/text-to-speech)
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.environ.get("GROQ_TTS_VOICE", "troy")
TTS_VOICES = ["troy", "austin", "hannah", "jessica", "sam", "leo", "mia"]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Durable home for the embedding model next to the SQLite data (data/ is the
# mounted volume in Docker), NOT fastembed's default OS-temp cache: Windows
# Disk Cleanup / Storage Sense wipe Temp, and the next upload then had to
# re-download ~100 MB *inside the request* — which is what turned the first
# attachment into an HTTP 503 when that download was slow or blocked.
DEFAULT_EMBED_CACHE = os.path.join(_PROJECT_ROOT, "data", "models")

_embedder = None
_embed_state = "cold"  # cold | loading | ready | error: <reason>
_embed_fatal = False  # True only when retrying cannot help (fastembed missing)
_embed_failed_at = 0.0  # monotonic time of the last failed load
# A failed load spends ~40 s retrying the download; answering every following
# upload with that same stall would hit proxy timeouts and hide the reason.
# For this many seconds uploads fail fast with the cached reason instead, then
# the load is genuinely retried so an offline machine recovers by itself.
EMBED_RETRY_AFTER = float(os.environ.get("DOCCHAT_EMBED_RETRY_AFTER", "60") or "60")


def _has_cache(path: str) -> bool:
    """True when a directory holds at least one entry (a downloaded model)."""
    try:
        with os.scandir(path) as it:
            return any(True for _ in it)
    except OSError:
        return False


def legacy_embed_cache_dir() -> str:
    """fastembed's own default cache (inside the OS temp dir). Only consulted
    so an install that already downloaded the model there is not forced to
    re-download it now that the default cache moved to data/models."""
    return os.path.join(tempfile.gettempdir(), "fastembed_cache")


def embed_cache_dir() -> str:
    """Where the embedding model is stored: FASTEMBED_CACHE when set (the
    Docker image pre-downloads into /opt/fastembed), else the durable
    data/models; a model already sitting (only) in fastembed's legacy temp
    cache is reused rather than re-downloaded."""
    env = os.environ.get("FASTEMBED_CACHE")
    if env:
        return env
    if _has_cache(DEFAULT_EMBED_CACHE):
        return DEFAULT_EMBED_CACHE
    legacy = legacy_embed_cache_dir()
    if _has_cache(legacy):
        log.info("using the existing embedding cache at %s", legacy)
        return legacy
    return DEFAULT_EMBED_CACHE


def embed_state() -> dict:
    """The embedding model's state for the UI / the health check:
    cold | loading | ready | error: <why>, plus the model and cache location."""
    global _embed_state, _embed_fatal
    if _embed_state == "cold":
        try:
            import fastembed  # noqa: F401 (verifies install)
        except ImportError:
            _embed_state = "error: 'fastembed' not installed — run: pip install -r requirements.txt"
            _embed_fatal = True
    return {"state": _embed_state, "model": EMBED_MODEL, "cache_dir": embed_cache_dir()}


def embed_error() -> str:
    """The reason the embedding model is not ready ("" when there is none),
    with the display "error: " prefix stripped."""
    return _embed_state.removeprefix("error: ") if _embed_state.startswith("error") else ""


def _load_embedder():
    """Build the local ONNX embedding model (a seam so the retry behaviour is
    testable without a 100 MB download)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL, cache_dir=embed_cache_dir())


def _get_embedder():
    global _embedder, _embed_state, _embed_fatal, _embed_failed_at
    if _embedder is not None:
        return _embedder
    if _embed_fatal:
        raise RuntimeError(
            "'fastembed' is not installed. Run: pip install -r requirements.txt"
        )
    if _embed_failed_at and time.monotonic() - _embed_failed_at < EMBED_RETRY_AFTER:
        left = int(EMBED_RETRY_AFTER - (time.monotonic() - _embed_failed_at)) + 1
        raise RuntimeError(f"{embed_error()} (retrying in ~{left}s)")
    _embed_state = "loading"
    try:
        _embedder = _load_embedder()
    except Exception as e:
        # Record the real reason (the UI shows it instead of a bare 503) but
        # stay retryable: a download that failed while offline must be able to
        # succeed once the machine is online again.
        _embed_state = f"error: {e}"
        _embed_failed_at = time.monotonic()
        raise
    _embed_state = "ready"
    _embed_failed_at = 0.0
    return _embedder


def warm_embeddings() -> bool:
    """Load the embedding model ahead of the first upload. Never raises — the
    server calls this at startup so the one-time ~100 MB download happens then,
    not in the middle of a user's request."""
    try:
        _get_embedder()
        log.info("embedding model %s ready (cache: %s)", EMBED_MODEL, embed_cache_dir())
        return True
    except Exception as e:
        log.warning("embedding model is not ready (%s) — uploads will fail until it loads", e)
        return False


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (n, dim) float32 matrix, rows normalized."""
    _get_embedder()
    vecs = np.asarray(list(_get_embedder().embed(list(texts), batch_size=32)), dtype=np.float32)
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def build_messages(history: list[dict], question: str, sources: list[dict]) -> list[dict]:
    """history: [{'role','content'}] previous turns; sources: retrieval results."""
    blocks = []
    for i, s in enumerate(sources, 1):
        loc = f" (page {s['page']})" if s.get("page") else ""
        blocks.append(f"[{i}] File: {s['doc_name']}{loc} Score: {s['score']:.2f}\n{s['text']}")
    system = (
        "You are a precise assistant that answers questions ONLY from the user's uploaded documents.\n\n"
        "SOURCES:\n"
        + "\n\n".join(blocks)
        + "\n\nRULES:\n"
        "1. Answer using ONLY the SOURCES above. Never use outside knowledge.\n"
        "2. After every claim, cite the supporting source inline, e.g. [1] or [2][3].\n"
        "3. If the SOURCES do not contain the answer, reply exactly: \"I couldn't find that in your documents.\"\n"
        "4. For greetings or small talk, answer briefly in one line without citations.\n"
        "5. Be concise but complete. Quote figures and exact values from the sources."
    )
    messages = [{"role": "system", "content": system}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    return messages


async def groq_models(key: str) -> list[str] | None:
    """Live model list from Groq; None if unreachable."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)) as client:
            resp = await client.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
            if resp.status_code != 200:
                return None
            ids = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
        # keep it tidy: prioritize known-stable models, then everything else
        stable = [m for m in DEFAULT_GROQ_MODELS if m in ids]
        rest = [m for m in ids if m not in DEFAULT_GROQ_MODELS]
        return stable + rest
    except Exception:
        return None


# ---------------- non-streaming chat (agent decisions, memory, judges) ----------------
async def groq_chat(
    key: str, model: str, messages: list[dict],
    json_mode: bool = False, temperature: float = 0.2, max_tokens: int = 1200,
) -> str:
    """Non-streaming completion. Retries once on the fallback model if configured.

    json_mode uses Groq's response_format json_object; the caller must still
    parse defensively (see parse_json).
    """
    for attempt, mdl in enumerate(_model_chain(model)):
        try:
            payload = {
                "model": mdl,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)) as client:
                resp = await client.post(GROQ_URL, json=payload, headers=_headers(key))
            body = resp.text or ""
            if resp.status_code == 200:
                data = resp.json()
                _track_usage(mdl, data)
                return data["choices"][0]["message"]["content"] or ""
            if resp.status_code == 429:
                _track_429(body, mdl)
            if attempt == 0 and FALLBACK_MODEL:
                log.warning("model %s failed (%s); retrying with fallback %s", mdl, resp.status_code, FALLBACK_MODEL)
                continue
            raise RuntimeError(f"Groq API error {resp.status_code}: {body}")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt == 0 and FALLBACK_MODEL:
                log.warning("model %s error (%s); retrying with fallback", mdl, e)
                continue
            raise RuntimeError(f"Groq request failed: {e}")
    raise RuntimeError("Groq request failed: no models left to try")


def _model_chain(model: str):
    chain = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        chain.append(FALLBACK_MODEL)
    return chain


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


async def groq_stream(key: str, model: str, messages: list[dict], max_tokens: int = 1500):
    """Yield answer text deltas from Groq (streaming). Falls back to the fallback model once."""
    for attempt, mdl in enumerate(_model_chain(model)):
        try:
            payload = {
                "model": mdl,
                "messages": messages,
                "stream": True,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream_options": {"include_usage": True},
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)) as client:
                async with client.stream("POST", GROQ_URL, json=payload, headers=_headers(key)) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:400]
                        if resp.status_code == 429:
                            _track_429(body, mdl)
                        if attempt == 0 and FALLBACK_MODEL:
                            log.warning("stream model %s failed (%s); fallback", mdl, resp.status_code)
                            continue
                        raise RuntimeError(f"Groq API error {resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        # stream_options.include_usage puts a usage object in the
                        # final chunk (choices may be empty) — record it once.
                        if isinstance(chunk, dict) and chunk.get("usage"):
                            _track_usage(mdl, chunk)
                        try:
                            delta = chunk["choices"][0]["delta"].get("content")
                        except (KeyError, IndexError, TypeError):
                            continue
                        if delta:
                            yield delta
                    return
        except Exception as e:
            if attempt == 0 and FALLBACK_MODEL:
                log.warning("stream failed (%s); fallback model", e)
                continue
            raise RuntimeError(f"Groq request failed: {e}")
    raise RuntimeError("Groq request failed: no models left to try")


def parse_json(text: str) -> dict | list | None:
    """Tolerant JSON parse: strips code fences and stray prose, returns None on failure."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------- voice (STT + TTS via Groq) ----------------
async def groq_stt(key: str, audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio via Groq Whisper. Returns text or raises with a clear message."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)) as client:
        resp = await client.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {key}"},
            data={"model": STT_MODEL},
            files={"file": (filename, audio_bytes, _mime_for(filename))},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Speech-to-text error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = (data.get("text") or "").strip()
    if not text:
        raise RuntimeError("Speech-to-text returned no text (could not hear anything).")
    return text


async def groq_tts(key: str, text: str, voice: str | None = None, model: str | None = None) -> bytes:
    """Synthesize speech via Groq (Orpheus). Returns wav bytes or raises."""
    text = (text or "").strip()[:2000]
    if not text:
        raise ValueError("Nothing to speak")
    payload = {
        "model": model or TTS_MODEL,
        "input": text,
        "voice": voice or TTS_VOICE,
        "response_format": "wav",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)) as client:
        resp = await client.post(GROQ_TTS_URL, json=payload, headers=_headers(key))
    if resp.status_code != 200:
        raise RuntimeError(f"Text-to-speech error {resp.status_code}: {resp.text[:300]}")
    return resp.content


def _mime_for(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}.get(ext, "audio/webm")