"""Basic tools: safe calculator, web search, sandboxed pandas execution.

Security posture: this app binds to 127.0.0.1 and is single-user. The
calculator and python sandbox are *best-effort* isolation, not a security
boundary — they stop accidental damage and the classic eval() foot-guns, but a
determined local attacker is out of scope by design. The python sandbox runs
in a subprocess with a timeout and restricted builtins (no imports, no file
I/O) so a bad script cannot crash the server process.

web_search lives in ragchat.websearch (Tavily primary + DuckDuckGo fallback);
this module just re-exports it so all tools stay behind one import.
"""
import ast
import json
import logging
import math
import os
import subprocess
import sys
import tempfile

log = logging.getLogger("jarvis.tools")

# ---------------- calculator (safe AST eval) ----------------
_MATH_FUNCS = {
    "abs", "ceil", "floor", "round", "sqrt", "exp", "log", "log2", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "degrees", "radians",
    "pi", "e", "tau", "hypot", "pow", "fabs", "isnan", "isinf",
}
_ALLOWED_OPS = (ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Expression)


def calculate(expression: str) -> str:
    """Evaluate a math expression with a strict AST allowlist. Never exec()."""
    expr = expression.strip()
    if not expr:
        return "Error: empty expression"
    if len(expr) > 200:
        return "Error: expression too long"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"Error: invalid expression ({e.msg})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS
                    and not node.keywords and len(node.args) <= 2):
                return f"Error: function '{getattr(node.func, 'id', '?')}' not allowed"
        elif isinstance(node, ast.Attribute):
            return "Error: attribute access not allowed"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            # guard against result blow-up (2 ** 100000000 would allocate GBs)
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                if abs(node.right.value) > 10000:
                    return "Error: exponent too large"
        elif not isinstance(node, _ALLOWED_OPS) and not isinstance(node, (ast.Name, ast.Load)):
            return f"Error: construct '{type(node).__name__}' not allowed"
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if names - _MATH_FUNCS:
        return f"Error: unknown name(s): {', '.join(sorted(names - _MATH_FUNCS))}"
    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, vars(math))
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return f"Error: {e}"
    if isinstance(result, float):
        result = round(result, 8)
    return str(result)


# ---------------- web search (Tavily + ddgs fallback) ----------------
def web_search(query: str, max_results: int = 5) -> str:
    """Best-effort web search. Delegates to ragchat.websearch: Tavily when a
    key is configured, DuckDuckGo otherwise. Returns compact [n]-tagged text
    or a clear error string (never raises)."""
    from .websearch import web_search as _ws

    return _ws(query, max_results)


# ---------------- sandboxed pandas execution ----------------
_SANDBOX_SNIPPET = r"""
import io, json, sys, builtins, contextlib
import math, statistics, re, datetime
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("IMPORT_ERROR:", e); sys.exit(1)

def _blocked(*_a, **_k):
    raise RuntimeError("blocked: file/import access is disabled in the analysis sandbox")

df = None
df_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
if df_path:
    with open(df_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
code = sys.stdin.read()
# Keep the real builtins (pandas/numpy internals need getattr etc.) but block
# the dangerous entry points: imports and file I/O. The hard isolation boundary
# is the subprocess itself + timeout; this is defense-in-depth for accidents.
_globs = {
    "__name__": "__main__",
    "df": df, "pd": pd, "np": np, "math": math, "statistics": statistics, "json": json,
    "re": re, "datetime": datetime,
    # globals shadows catch direct calls; the builtins override catches `import`
    # statements (Python 3.12 resolves those via __builtins__, not globals)
    "__import__": _blocked,
    "open": _blocked,
    "input": _blocked,
    "__builtins__": {
        k: v for k, v in vars(builtins).items() if k not in ("open", "input")
    } | {"__import__": _blocked},
}
out = io.StringIO()
try:
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<sandbox>", "exec"), _globs)
except SystemExit:
    pass
except Exception as e:
    print("RUNTIME_ERROR:", type(e).__name__, ":", e)
    sys.exit(1)
text = out.getvalue()
if text.strip():
    print(text)
"""


def run_python(code: str, df_json: dict | None = None, timeout: int = 15) -> str:
    """Execute `code` in a restricted subprocess.

    `df_json` is a pandas split-format dict ({"columns": [...], "data": [...]})
    exposed to the script as `df`. Returns captured stdout or an error string.
    """
    if len(code) > 4000:
        return "Error: code too long (max 4000 chars)"
    df_path = "-"
    tmp = None
    if df_json is not None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(df_json, tmp)
        tmp.close()
        df_path = tmp.name
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SANDBOX_SNIPPET, df_path],
            input=code, capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return "Error: script timed out after %ds" % timeout
    finally:
        if tmp:
            os.unlink(tmp.name)
    if proc.returncode != 0:
        detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if "RUNTIME_ERROR:" in detail or "IMPORT_ERROR:" in detail or "blocked" in detail:
            return f"Error: {detail[:300]}"
        return f"Error: script exited with {proc.returncode}: {detail[:300] or 'unknown error'}"
    out = (proc.stdout or "").strip()
    return out or "(script produced no output)"
