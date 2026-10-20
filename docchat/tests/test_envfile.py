"""Tests for envfile (zero-dependency .env loader). Run: python tests/test_envfile.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envfile import load_env


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_loads_plain_quoted_and_export_lines():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".env")
        _write(p, "# comment\nGROQ_FALLBACK_MODEL=llama-3.1-8b-instant\nexport PORT=8888\n"
                  "TTS_VOICE='troy'\n\nEMPTY=\n")
        saved = {k: os.environ.pop(k, None) for k in ("GROQ_FALLBACK_MODEL", "PORT", "TTS_VOICE", "EMPTY")}
        try:
            assert load_env(p) is True
            assert os.environ["GROQ_FALLBACK_MODEL"] == "llama-3.1-8b-instant"
            assert os.environ["PORT"] == "8888"
            assert os.environ["TTS_VOICE"] == "troy"  # quotes stripped
            assert os.environ["EMPTY"] == ""
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_real_environment_wins_over_file():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".env")
        _write(p, "GROQ_FALLBACK_MODEL=from_file\n")
        os.environ["GROQ_FALLBACK_MODEL"] = "from_shell"
        try:
            assert load_env(p) is True
            assert os.environ["GROQ_FALLBACK_MODEL"] == "from_shell"
        finally:
            os.environ.pop("GROQ_FALLBACK_MODEL", None)


def test_default_path_points_at_project_root():
    """load_env() with no args must use ENV_PATH; point that at a temp file so
    the test never writes the developer's real .env."""
    import envfile as env_module

    real_path = env_module.ENV_PATH
    key = "GROQ_FALLBACK_MODEL"
    saved = os.environ.pop(key, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, ".env")
            _write(temp_path, f"{key}=llama-3.1-8b-instant\n")
            env_module.ENV_PATH = temp_path
            assert env_module.load_env() is True
            assert os.environ[key] == "llama-3.1-8b-instant"
    finally:
        env_module.ENV_PATH = real_path
        os.environ.pop(key, None)
        if saved is not None:
            os.environ[key] = saved


def test_missing_file_and_malformed_lines():
    assert load_env("/nonexistent/.env") is False
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, ".env")
        _write(p, "this line has no equals\n=novalue\nGROQ_TTS_VOICE=austin\n")
        saved = os.environ.pop("GROQ_TTS_VOICE", None)
        try:
            assert load_env(p) is True  # valid line still loads
            assert os.environ["GROQ_TTS_VOICE"] == "austin"
        finally:
            if saved is None:
                os.environ.pop("GROQ_TTS_VOICE", None)
            else:
                os.environ["GROQ_TTS_VOICE"] = saved


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
