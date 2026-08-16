"""Web search: Tavily (primary, when a key is configured) with a keyless
DuckDuckGo fallback.

Security: the Tavily API key comes from (in order) the process environment
(TAVILY_API_KEY, which always wins) or a configured key the server sets from
config.json via set_configured_key(). It is never read from the browser and
never included in tool output, logs, or API responses.

Failure handling: every provider is wrapped so that a network failure, timeout,
HTTP error, or empty result set degrades gracefully:
  - Tavily configured + works        -> Tavily results (tagged as such)
  - Tavily configured + fails        -> logged, fall back to DuckDuckGo
  - No Tavily key / both providers   -> returns an explicit error string the
    fail                               agent turns into a best-effort answer
                                       with a stated limitation

API reference: POST https://api.tavily.com/search with
`Authorization: Bearer <key>`, body {query, search_depth, max_results,
include_answer, topic}. Verified against
https://docs.tavily.com/documentation/api-reference/endpoint/search
"""
import logging
import os

import httpx

log = logging.getLogger("jarvis.websearch")

TAVILY_URL = "https://api.tavily.com/search"
SEARCH_TIMEOUT = 10.0  # seconds; long enough for real searches, short enough to fail fast
MAX_RESULTS = 8


class WebSearchError(RuntimeError):
    """A search provider failed (network, timeout, HTTP error, bad response)."""


_configured_key: str = ""  # UI-saved key, set by server.load_config()


def set_configured_key(key: str) -> None:
    """Set the fallback key from config.json (never overrides the env var)."""
    global _configured_key
    _configured_key = (key or "").strip()


def tavily_key() -> str:
    return (os.environ.get("TAVILY_API_KEY") or _configured_key).strip()


def tavily_configured() -> bool:
    return bool(tavily_key())


# ---------------- Tavily ----------------
def tavily_search(query: str, max_results: int = 5) -> tuple[str, int]:
    """Run one Tavily search. Returns (formatted_text, result_count).

    Raises WebSearchError on any failure (unconfigured, network, timeout,
    non-200 response, unparseable JSON). Never raises for an empty result set —
    that is a valid search outcome and is returned as 'No web results found.'
    """
    key = tavily_key()
    if not key:
        raise WebSearchError("Tavily is not configured (set TAVILY_API_KEY)")
    try:
        resp = httpx.post(
            TAVILY_URL,
            json={
                "query": query,
                "search_depth": "basic",      # faster/cheaper; depth is a quality knob, not a feature
                "max_results": max_results,
                "include_answer": False,      # we synthesize the answer ourselves from raw results
                "topic": "general",
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=SEARCH_TIMEOUT,
        )
    except httpx.HTTPError as e:  # covers ConnectError, TimeoutException, etc.
        raise WebSearchError(f"Tavily request failed: {e}")
    if resp.status_code != 200:
        # Tavily returns {"detail": {"error": "..."}} on API/plan errors
        detail = resp.text[:200]
        try:
            detail = resp.json().get("detail", {}).get("error") or detail
        except ValueError:
            pass
        raise WebSearchError(f"Tavily API error {resp.status_code}: {detail}")
    try:
        data = resp.json()
    except ValueError as e:
        raise WebSearchError(f"Tavily returned invalid JSON: {e}")
    return _format_results(data.get("results") or [], "Tavily")


# ---------------- DuckDuckGo fallback (keyless) ----------------
def _ddgs_search(query: str, max_results: int) -> tuple[str, int]:
    try:
        from ddgs import DDGS
    except ImportError:
        raise WebSearchError("'ddgs' is not installed (pip install ddgs)")
    try:
        results = DDGS().text(query, max_results=max_results, safesearch="moderate")
    except Exception as e:  # rate limits, captchas, network
        raise WebSearchError(f"DuckDuckGo search failed ({e})")
    return _format_results(results or [], "DuckDuckGo")


def _format_results(results: list[dict], provider: str) -> tuple[str, int]:
    """Normalize heterogeneous provider dicts into the compact [n] blocks the
    agent prompt already knows how to cite. Empty -> explicit 'no results'."""
    lines = [f"[Source: web search via {provider}]"]
    kept = 0
    for i, r in enumerate(results, 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("content") or "").strip().replace("\n", " ")
        url = (r.get("href") or r.get("url") or "").strip()
        if not title and not body:
            continue
        date = (r.get("published_date") or "").strip()
        lines.append(f"[{i}] {title}" + (f" ({date})" if date else ""))
        if body:
            lines.append(body)
        lines.append(url)
        kept += 1
    if not kept:
        return "No web results found.", 0
    return "\n".join(lines), kept


# ---------------- public entry ----------------
def web_search(query: str, max_results: int = 5) -> str:
    """Best-effort web search: Tavily first (when configured), else fall back
    to DuckDuckGo. Never raises — returns a clear error string on total
    failure so the agent can answer from knowledge with a stated limitation."""
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    max_results = min(max(1, int(max_results)), MAX_RESULTS)
    if tavily_configured():
        try:
            text, count = tavily_search(query, max_results)
            log.info("web_search triggered query=%r provider=tavily results=%d", query[:120], count)
            return text
        except WebSearchError as e:
            log.warning("web_search tavily failed, falling back to ddgs: %s", e)
    try:
        text, count = _ddgs_search(query, max_results)
        log.info("web_search triggered query=%r provider=ddgs results=%d", query[:120], count)
        return text
    except WebSearchError as e:
        log.error("web_search failed (no provider available): %s", e)
        return (
            "Error: live web search is unavailable right now — no results could be "
            f"retrieved ({e}). Answer from your own knowledge and clearly state that "
            "the information may be outdated or unverified."
        )
