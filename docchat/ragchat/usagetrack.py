"""Local daily token-usage tracker for the Groq account.

Groq exposes no public usage/limits endpoint (the OpenAI-compatible API only
offers chat, responses, audio, models, batches, files, fine-tuning — verified
against https://console.groq.com/docs/api-reference). So this module tracks
usage locally from the two authoritative sources we DO have:

  1. `usage.total_tokens` in every chat-completion response body, and
  2. the `Limit X, Used Y` numbers inside 429 rate-limit error bodies, which
     are the account's per-model daily counters (they reflect ALL apps on the
     key, not just ours) — so we use them to correct/raise our local count.

Daily limits are PER MODEL (e.g. llama-3.3-70b-versatile TPD=100K,
llama-3.1-8b-instant TPD=500K per https://console.groq.com/docs/rate-limits),
so the tracker is keyed by model. The UI shows the report for the model that
is currently selected.

Design notes:
  * Best-effort by contract: if the file is corrupt or unwritable the app
    must keep working — every public method swallows its own I/O errors.
  * `used` is the max of (locally observed tokens, last 429-reported used),
    so a fresh install that missed earlier usage converges upward on the
    account's real number without double-counting.
  * Rollover is UTC-midnight, matching Groq's daily reset window.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone

log = logging.getLogger("jarvis.usagetrack")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
USAGE_PATH = os.path.join(DATA_DIR, "usage.json")

# Known tokens-per-day limits from https://console.groq.com/docs/rate-limits.
# Unknown models fall back to limit=None -> UI shows raw usage without a bar.
KNOWN_TPD = {
    "llama-3.3-70b-versatile": 100_000,
    "llama-3.1-8b-instant": 500_000,
    "llama3-8b-8192": 500_000,
    "mixtral-8x7b-32768": 500_000,
    "gemma2-9b-it": 500_000,
}

# RLock (not Lock): record/sync_429 acquire it and then call _current(),
# which acquires it again — a plain Lock would deadlock on the second entry.
_lock = threading.RLock()
_state: dict | None = None  # {"date": "YYYY-MM-DD", "models": {model: {"used": int, "limit": int|None}}}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _empty_bucket() -> dict:
    return {"used": 0, "limit": None}


def _load() -> dict:
    """Load state; returns a fresh empty state on missing/corrupt file."""
    global _state
    try:
        with open(USAGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("usage file is not a dict")
        models = {}
        raw_models = data.get("models")
        if isinstance(raw_models, dict):
            for model, b in raw_models.items():
                if not isinstance(b, dict):
                    continue
                used = 0
                try:
                    used = int(b.get("used") or 0)
                except (TypeError, ValueError):
                    used = 0
                limit = b.get("limit")
                models[model] = {
                    "used": used,
                    "limit": int(limit) if isinstance(limit, (int, float)) else None,
                }
        state = {"date": str(data.get("date", "")), "models": models}
    except Exception as e:
        log.warning("usage file unreadable (%s); starting fresh", e)
        state = {"date": _today(), "models": {}}
    _state = state
    return state


def _persist(state: dict) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = USAGE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, USAGE_PATH)
    except Exception as e:
        log.warning("could not persist usage (%s)", e)


def _current() -> dict:
    """Return today's state, rolling over model buckets if the date changed."""
    global _state
    with _lock:
        if _state is None:
            _state = _load()
        if _state.get("date") != _today():
            _state = {"date": _today(), "models": {}}
            _persist(_state)
        return _state


def record(model: str, tokens: int) -> None:
    """Add tokens to today's bucket for `model` (from a response usage body)."""
    if not tokens or tokens <= 0:
        return
    with _lock:
        state = _current()
        bucket = state["models"].setdefault(model, _empty_bucket())
        bucket["used"] += int(tokens)
        _persist(state)


def sync_429(model: str, used: int | None, limit: int | None) -> None:
    """Correct today's bucket for `model` from a 429 rate-limit error body.

    `used`/`limit` are the per-model daily counters Groq reported. The local
    count only ever moves upward (max), so a fresh install that missed earlier
    usage converges on the account's real number.
    """
    with _lock:
        state = _current()
        bucket = state["models"].setdefault(model, _empty_bucket())
        if isinstance(used, (int, float)) and int(used) > 0:
            bucket["used"] = max(bucket["used"], int(used))
        if isinstance(limit, (int, float)) and int(limit) > 0:
            bucket["limit"] = int(limit)
        _persist(state)


def report(model: str = "") -> dict:
    """Return today's usage for `model` for the UI.

    Returns {date, model, used, limit, remaining, pct}; limit is None when
    unknown (no 429 seen yet and the model is not in KNOWN_TPD).
    """
    with _lock:
        state = _current()
        bucket = state["models"].get(model, _empty_bucket())
        used = int(bucket.get("used") or 0)
        limit = bucket.get("limit") or KNOWN_TPD.get(model) or None
        remaining = max(limit - used, 0) if limit else None
        pct = round(100.0 * used / limit, 1) if limit else None
        return {
            "date": state.get("date", _today()),
            "model": model,
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "pct": pct,
        }


def reset() -> None:
    """Test helper: clear in-memory state (file untouched)."""
    global _state
    with _lock:
        _state = None
