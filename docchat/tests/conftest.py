"""Shared pytest setup: make the project root importable and keep tests hermetic."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep imports of server/store away from the developer's real data directory.
os.environ.setdefault("DOCCHAT_DATA_DIR", tempfile.mkdtemp(prefix="jarvis-pytest-"))

# Default to the explicit local mode so importing server/auth never requires a
# Clerk configuration. Tests that exercise hosted behavior override this.
os.environ.setdefault("AUTH_MODE", "local")
