"""Tiny .env loader — zero dependencies.

Reads KEY=VALUE lines from .env (project root) into os.environ, WITHOUT
overriding variables that are already set in the real environment (a shell
export always wins over the file). Supports blank lines, # comments, an
optional `export ` prefix, and single/double-quoted values.

Why not python-dotenv? One less dependency for ~25 lines; the subset we need
is small and pinned by tests.
"""
import os

# envfile.py lives at the project root, so the .env sits right next to it
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def load_env(path: str | None = None) -> bool:
    """Load the .env file at `path` (default: project root). Returns True if
    the file existed and was parsed. Never raises on a malformed file — a bad
    line is skipped and logged via a print (startup is too early for logging)."""
    path = path or ENV_PATH
    if not os.path.exists(path):
        return False
    loaded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                print(f"[env] skipping malformed line in {path}: {line[:40]!r}")
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key:
                continue
            if key in os.environ:
                continue  # real environment wins over the file
            os.environ[key] = value
            loaded += 1
    if loaded:
        print(f"[env] loaded {loaded} variable(s) from {path}")
    return True
