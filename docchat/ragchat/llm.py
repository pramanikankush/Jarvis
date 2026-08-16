"""Local free embeddings (fastembed, ONNX CPU) + Groq client (chat/STT/TTS).

The only external service is api.groq.com — everything else runs locally.
"""
import json
import logging
import os
import re

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
DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "")  # optional resilience net

# Voice (verified against https://console.groq.com/docs/text-to-speech)
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.environ.get("GROQ_TTS_VOICE", "troy")
TTS_VOICES = ["troy", "austin", "hannah", "jessica", "sam", "leo", "mia"]

_embedder = None
_embed_state = "cold"  # cold | loading | ready | error


def embed_state() -> dict:
    global _embed_state
    if _embed_state == "cold":
        try:
            import fastembed  # noqa: F401 (verifies install)
        except ImportError:
            _embed_state = "error: 'fastembed' not installed — run: pip install -r requirements.txt"
    return {"state": _embed_state, "model": EMBED_MODEL}


def _get_embedder():
    global _embedder, _embed_state
    if _embedder is None:
        if _embed_state.startswith("error"):
            raise RuntimeError(
                "'fastembed' is not installed. Run: pip install -r requirements.txt"
            )
        _embed_state = "loading"
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE"))
        _embed_state = "ready"
    return _embedder


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