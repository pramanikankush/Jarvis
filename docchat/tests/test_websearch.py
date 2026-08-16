"""Tests for ragchat.websearch: Tavily client, DuckDuckGo fallback, graceful
failure, and key confidentiality. No real network calls — httpx.post and the
ddgs fallback are mocked.
Run: python tests/test_websearch.py
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from ragchat import websearch


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


SAMPLE_RESULTS = {
    "results": [
        {"title": "Latest Python Release", "url": "https://www.python.org/downloads/",
         "content": "Python 3.13 is the current stable release.", "score": 0.9,
         "published_date": "2026-08-01"},
        {"title": "Another page", "url": "https://example.com/x", "content": "Some body text."},
    ],
    "response_time": "0.8",
}


def _set_key(val):
    if val is None:
        os.environ.pop("TAVILY_API_KEY", None)
        websearch.set_configured_key("")
    else:
        os.environ["TAVILY_API_KEY"] = val


@contextlib.contextmanager

def _patch_post(resp_or_exc):
    """Patch websearch.httpx.post (the module's reference to httpx.post)."""
    orig = websearch.httpx.post

    def fake_post(url, **kwargs):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    websearch.httpx.post = fake_post
    try:
        yield
    finally:
        websearch.httpx.post = orig


def test_tavily_request_format_and_output():
    _set_key("tvly-secret-test-key")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["body"] = kwargs.get("json", {})
        return FakeResp(200, SAMPLE_RESULTS)

    orig = websearch.httpx.post
    websearch.httpx.post = fake_post
    try:
        text, count = websearch.tavily_search("latest python version", 3)
    finally:
        websearch.httpx.post = orig
    assert captured["url"] == websearch.TAVILY_URL
    assert captured["headers"]["Authorization"] == "Bearer tvly-secret-test-key"
    assert captured["body"]["query"] == "latest python version"
    assert captured["body"]["max_results"] == 3
    assert captured["body"]["search_depth"] == "basic"
    assert captured["body"]["include_answer"] is False
    assert captured["body"]["topic"] == "general"
    assert count == 2
    assert "[Source: web search via Tavily]" in text
    assert "Latest Python Release" in text and "https://www.python.org/downloads/" in text
    assert "2026-08-01" in text
    # the key must never appear in tool output
    assert "tvly-secret-test-key" not in text


def test_tavily_unconfigured_raises():
    _set_key(None)
    try:
        websearch.tavily_search("anything")
        assert False, "expected WebSearchError"
    except websearch.WebSearchError as e:
        assert "TAVILY_API_KEY" in str(e)


def test_tavily_http_error_raises_with_detail():
    _set_key("tvly-x")
    try:
        with _patch_post(FakeResp(500, {"detail": {"error": "Internal Server Error"}})):
            try:
                websearch.tavily_search("q")
                assert False, "expected WebSearchError"
            except websearch.WebSearchError as e:
                assert "500" in str(e) and "Internal Server Error" in str(e)
    finally:
        _set_key(None)


def test_tavily_malformed_results_treated_as_empty():
    _set_key("tvly-x")
    try:
        with _patch_post(FakeResp(200, payload={"results": "not-a-list"})):
            # non-list results degrade to a clean 'no results', not a crash
            text, count = websearch.tavily_search("q")
            assert count == 0 and "No web results" in text
    finally:
        _set_key(None)


def test_web_search_no_key_falls_back_to_ddgs():
    _set_key(None)
    calls = []

    def fake_ddgs(query, max_results):
        calls.append((query, max_results))
        return ("[Source: web search via DuckDuckGo]\n[1] T\nb\nu", 1)

    orig = websearch._ddgs_search
    websearch._ddgs_search = fake_ddgs
    try:
        out = websearch.web_search("what is the weather", 7)
    finally:
        websearch._ddgs_search = orig
    assert calls == [("what is the weather", 7)]
    assert "[Source: web search via DuckDuckGo]" in out


def test_web_search_tavily_failure_falls_back_to_ddgs():
    _set_key("tvly-x")
    called_ddgs = []

    def fake_ddgs(query, max_results):
        called_ddgs.append(query)
        return ("[Source: web search via DuckDuckGo]\n[1] fallback", 1)

    orig = websearch._ddgs_search
    websearch._ddgs_search = fake_ddgs
    try:
        with _patch_post(httpx.TimeoutException("took too long", request=None)):
            out = websearch.web_search("current news")
    finally:
        websearch._ddgs_search = orig
        _set_key(None)
    assert called_ddgs == ["current news"], "ddgs fallback should run after Tavily timeout"
    assert "fallback" in out


def test_web_search_both_fail_returns_clear_error():
    _set_key("tvly-x")

    def fake_ddgs(query, max_results):
        raise websearch.WebSearchError("DuckDuckGo rate limited")

    orig = websearch._ddgs_search
    websearch._ddgs_search = fake_ddgs
    try:
        with _patch_post(httpx.ConnectError("no network")):
            out = websearch.web_search("anything")
    finally:
        websearch._ddgs_search = orig
        _set_key(None)
    assert out.startswith("Error:"), out
    assert "unavailable" in out  # agent is told to answer with a stated limitation


def test_web_search_empty_results_is_not_an_error():
    _set_key("tvly-x")
    try:
        with _patch_post(FakeResp(200, {"results": []})):
            out = websearch.web_search("nothing here")
        assert out == "No web results found.", out
    finally:
        _set_key(None)


def test_web_search_empty_query():
    _set_key(None)
    assert "Error: empty search query" in websearch.web_search("   ")


def test_configured_key_used_when_env_unset():
    """UI-saved key (set_configured_key) enables Tavily when no env var."""
    _set_key(None)
    websearch.set_configured_key("tvly-from-config")
    try:
        assert websearch.tavily_key() == "tvly-from-config"
        assert websearch.tavily_configured()
    finally:
        websearch.set_configured_key("")


def test_env_var_wins_over_configured_key():
    _set_key("tvly-env")
    websearch.set_configured_key("tvly-from-config")
    try:
        assert websearch.tavily_key() == "tvly-env"
    finally:
        websearch.set_configured_key("")
        _set_key(None)


def test_clearing_configured_key_disables_tavily():
    """A UI clear must actually take effect (the server never writes env)."""
    _set_key(None)
    websearch.set_configured_key("tvly-x")
    websearch.set_configured_key("")
    try:
        assert websearch.tavily_key() == ""
        assert not websearch.tavily_configured()
    finally:
        pass


def test_max_results_capped():
    _set_key("tvly-x")
    captured = {}

    def fake_post(url, **kwargs):
        captured["body"] = kwargs.get("json", {})
        return FakeResp(200, {"results": []})

    orig = websearch.httpx.post
    websearch.httpx.post = fake_post
    try:
        websearch.web_search("q", 999)
    finally:
        websearch.httpx.post = orig
        _set_key(None)
    assert captured["body"]["max_results"] == websearch.MAX_RESULTS


def test_key_never_leaked_anywhere():
    _set_key("tvly-SUPER-SECRET-123")
    calls = []

    def fake_ddgs(query, max_results):
        calls.append(query)
        raise websearch.WebSearchError("no ddgs either")

    orig = websearch._ddgs_search
    websearch._ddgs_search = fake_ddgs
    try:
        with _patch_post(FakeResp(500, {"detail": {"error": "boom"}})):
            try:
                websearch.web_search("q")
            except Exception as e:
                assert "SUPER-SECRET" not in str(e)
    finally:
        websearch._ddgs_search = orig
        _set_key(None)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
