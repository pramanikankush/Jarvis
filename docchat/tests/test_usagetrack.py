"""Tests for ragchat.usagetrack: local per-model daily token tracking.
No network. Each test redirects the tracker to its own temp file so tests
never touch the app's real data/usage.json.
Run: python tests/test_usagetrack.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ragchat.usagetrack as ut


def _fresh(tmp):
    ut.reset()
    ut.USAGE_PATH = os.path.join(tmp, "usage.json")
    return ut


def _read_file(tmp):
    with open(os.path.join(tmp, "usage.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def test_record_accumulates_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 100)
        m.record("llama-3.3-70b-versatile", 250)
        rep = m.report("llama-3.3-70b-versatile")
        assert rep["used"] == 350
        assert rep["limit"] == 100_000  # KNOWN_TPD fallback
        assert rep["remaining"] == 100_000 - 350
        assert rep["pct"] == round(100.0 * 350 / 100_000, 1)  # 0.3
        # persisted to disk
        data = _read_file(tmp)
        assert data["models"]["llama-3.3-70b-versatile"]["used"] == 350


def test_models_are_tracked_separately():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 90_000)
        m.record("llama-3.1-8b-instant", 5_000)
        assert m.report("llama-3.3-70b-versatile")["used"] == 90_000
        assert m.report("llama-3.3-70b-versatile")["pct"] == 90.0
        assert m.report("llama-3.1-8b-instant")["used"] == 5_000
        # unknown model -> limit None, pct None (no false alarm)
        rep = m.report("some-future-model")
        assert rep["used"] == 0 and rep["limit"] is None and rep["pct"] is None


def test_sync_429_corrects_usage_and_limit():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 500)
        # a 429 says the account used 60k today (other apps counted too)
        m.sync_429("llama-3.3-70b-versatile", used=60_000, limit=100_000)
        rep = m.report("llama-3.3-70b-versatile")
        assert rep["used"] == 60_000
        assert rep["limit"] == 100_000
        # subsequent local usage adds on top
        m.record("llama-3.3-70b-versatile", 100)
        assert m.report("llama-3.3-70b-versatile")["used"] == 60_100


def test_sync_429_never_lowers_usage():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 20_000)
        m.sync_429("llama-3.3-70b-versatile", used=5_000, limit=100_000)
        assert m.report("llama-3.3-70b-versatile")["used"] == 20_000  # max() semantics


def test_rollover_at_utc_midnight():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 90_000)
        # simulate a new UTC day by forcing the stored date back
        m._current()["date"] = "2000-01-01"
        m._persist(m._current())
        m.reset()
        m.USAGE_PATH = os.path.join(tmp, "usage.json")
        # next access rolls the bucket over
        rep = m.report("llama-3.3-70b-versatile")
        assert rep["used"] == 0
        assert rep["limit"] == 100_000  # known limit carries via KNOWN_TPD


def test_corrupt_file_starts_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "usage.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ not valid json !!!")
        m = _fresh(tmp)
        rep = m.report("llama-3.3-70b-versatile")
        assert rep["used"] == 0
        # a subsequent record still works and rewrites the file
        m.record("llama-3.3-70b-versatile", 10)
        assert _read_file(tmp)["models"]["llama-3.3-70b-versatile"]["used"] == 10


def test_nonpositive_tokens_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.record("llama-3.3-70b-versatile", 0)
        m.record("llama-3.3-70b-versatile", -5)
        assert m.report("llama-3.3-70b-versatile")["used"] == 0


def test_limit_sync_without_used():
    """A 429 that only yields a limit (unusual) still records the limit."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _fresh(tmp)
        m.sync_429("custom-model", used=None, limit=250_000)
        rep = m.report("custom-model")
        assert rep["limit"] == 250_000
        assert rep["used"] == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
