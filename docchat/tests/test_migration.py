"""Tests for the 2026 model migration: default model, dead-model mapping,
fallback chain, and usagetrack limits. No network.
Run: python tests/test_migration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat import llm


def test_default_model_is_2026_qwen():
    # Groq decommissioned llama-3.3-70b-versatile on 2026-08-16; the default
    # must be a live model (user-selected: qwen/qwen3.8-27b).
    assert llm.DEFAULT_MODEL == "qwen/qwen3.8-27b"
    assert llm.DEFAULT_MODEL not in llm.DEAD_MODELS


def test_fallback_model_is_live_and_differs():
    assert llm.FALLBACK_MODEL == "openai/gpt-oss-120b"
    assert llm.FALLBACK_MODEL not in llm.DEAD_MODELS
    assert llm.FALLBACK_MODEL != llm.DEFAULT_MODEL


def test_dead_models_map_to_default():
    for dead in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                 "mixtral-8x7b-32768", "gemma2-9b-it", "qwen/qwen3-32b"):
        assert llm.migrate_model(dead) == llm.DEFAULT_MODEL, dead


def test_live_models_pass_through():
    for live in ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                 "groq/compound-mini"):
        assert llm.migrate_model(live) == live


def test_empty_and_none_model_pass_through():
    assert llm.migrate_model("") == ""
    assert llm.migrate_model(None) == ""


def test_default_model_list_has_no_dead_models():
    assert not (set(llm.DEFAULT_GROQ_MODELS) & llm.DEAD_MODELS)
    assert "qwen/qwen3.8-27b" in llm.DEFAULT_GROQ_MODELS
    assert "groq/compound-mini" in llm.DEFAULT_GROQ_MODELS


def test_model_chain_for_fallback():
    chain = llm._model_chain("qwen/qwen3.8-27b")
    assert chain == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    # if primary == fallback, no duplicate attempt
    assert llm._model_chain("openai/gpt-oss-120b") == ["openai/gpt-oss-120b"]


def test_usagetrack_knows_new_models():
    from ragchat import usagetrack
    assert usagetrack.KNOWN_TPD.get("qwen/qwen3.8-27b") == 250_000
    assert usagetrack.KNOWN_TPD.get("openai/gpt-oss-120b") == 250_000
    # legacy rows still resolve
    assert usagetrack.KNOWN_TPD.get("llama-3.3-70b-versatile") == 100_000


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
