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


# ---------------- unit + currency converter ----------------
# Length/mass/temperature/data only — the units people actually convert daily.
# Everything is stdlib; currency is best-effort via a keyless free API with a
# 1-hour cache and never raises (failures return an explanatory string).
_UNIT_FACTOR = {
    # length (base: meter)
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "inch": 0.0254,
    "inches": 0.0254, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "yd": 0.9144,
    "yard": 0.9144, "yards": 0.9144, "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    # mass (base: kilogram)
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0, "tonne": 1000.0, "oz": 0.028349523125,
    "ounce": 0.028349523125, "ounces": 0.028349523125, "lb": 0.45359237, "lbs": 0.45359237,
    "pound": 0.45359237, "pounds": 0.45359237,
    # data (base: byte)
    "b": 1.0, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
}
_TEMP_UNITS = {"c", "°c", "celsius", "f", "°f", "fahrenheit", "k", "kelvin"}


def _norm_unit(u: str) -> str:
    return (u or "").strip().lower().rstrip(".")


def _temp_to_c(v: float, unit: str) -> float:
    if unit in ("f", "°f", "fahrenheit"):
        return (v - 32.0) * 5.0 / 9.0
    if unit in ("k", "kelvin"):
        return v - 273.15
    return v


def _temp_from_c(v: float, unit: str) -> float:
    if unit in ("f", "°f", "fahrenheit"):
        return v * 9.0 / 5.0 + 32.0
    if unit in ("k", "kelvin"):
        return v + 273.15
    return v


def convert(from_unit: str, to_unit: str, value: float) -> str:
    """Convert `value` between everyday units; temperature and currency get
    special paths. Returns a ready-to-show string or a clear error."""
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f == t:
        return f"{value:g} {f} = {value:g} {t}"
    if f in _TEMP_UNITS and t in _TEMP_UNITS:
        c = _temp_to_c(value, f)
        return f"{value:g}° {f.upper()} = {_temp_from_c(c, t):.4g}° {t.upper()}"
    fu, tu = _UNIT_FACTOR.get(f), _UNIT_FACTOR.get(t)
    if fu is not None and tu is not None:
        out = value * fu / tu
        pretty = f"{out:.6g}"
        return f"{value:g} {f} = {pretty} {t}"
    # currency: same-symbol codes (usd, eur, inr, ...) via a free keyless API
    if len(f) == 3 and len(t) == 3 and f.isalpha() and t.isalpha():
        return _convert_currency(f, t, value)
    known = sorted(set(_UNIT_FACTOR) | _TEMP_UNITS)
    return (f"Error: I don't know the unit(s) '{from_unit}'/'{to_unit}'. "
            f"Try units like: {', '.join(known[:18])}…, temperatures (c/f/k), "
            "or 3-letter currency codes (usd, eur, inr).")


_FX_URL = "https://open.er-api.com/v6/latest/{base}"
_fx_cache: dict[str, tuple[float, dict]] = {}  # base -> (fetched_at, rates)
_FX_TTL = 3600.0  # seconds; rates move slowly, one fetch/hour is plenty


def _convert_currency(f: str, t: str, value: float) -> str:
    import time as _time

    try:
        import httpx

        now = _time.time()
        cached = _fx_cache.get(f)
        if cached is None or now - cached[0] > _FX_TTL:
            resp = httpx.get(_FX_URL.format(base=f.upper()), timeout=8.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("result") != "success":
                return f"Error: currency service returned an error for {f.upper()}."
            _fx_cache[f] = (now, data["rates"])
        rate = _fx_cache[f][1].get(t.upper())
        if not rate:
            return f"Error: unknown currency code '{t}'."
        out = value * float(rate)
        return f"{value:g} {f.upper()} = {out:,.4g} {t.upper()} (live rate, 1h cache)"
    except Exception as e:
        return (f"Error: live currency conversion failed ({e}). "
                "Use web_search for current rates, or try again later.")


# ---------------- URL reading (fetch + readable text) ----------------
_BLOCK_TAGS = ("script", "style", "noscript", "svg", "iframe", "nav", "footer", "form")


def extract_url_text(html: str, max_chars: int = 12000) -> str:
    """Crude but effective HTML -> readable text: drop script/style/nav junk,
    strip tags, collapse whitespace. stdlib-only (no bs4 dependency)."""
    import re as _re

    text = html or ""
    for tag in _BLOCK_TAGS:
        text = _re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", " ", text, flags=_re.I | _re.S)
    text = _re.sub(r"<head\b[^>]*>.*?</head>", " ", text, flags=_re.I | _re.S)
    text = _re.sub(r"<[^>]+>", " ", text)
    # common entities (order matters: ampersand last)
    for ent, ch in (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                    ("&#39;", "'"), ("&amp;", "&")):
        text = text.replace(ent, ch)
    text = _re.sub(r"[ \t\r\f\v]+", " ", text)
    text = _re.sub(r" ?\n ?", "\n", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def fetch_url_text(url: str, timeout: float = 15.0) -> str:
    """Fetch a page and return its readable text. Raises ValueError with a
    clear message on bad input or fetch failure (caller decides fallback)."""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    try:
        import httpx

        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; JarvisAssistant/1.0)",
        })
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not fetch the page: {e}") from e
    ctype = resp.headers.get("content-type", "")
    if ctype and "html" not in ctype and "text" not in ctype and "json" not in ctype:
        raise ValueError(f"Unsupported content type '{ctype.split(';')[0]}' — I can only read web pages.")
    return extract_url_text(resp.text)


# ---------------- image generation (free, keyless) ----------------
def image_url(prompt: str, width: int = 768, height: int = 512) -> str:
    """Build a Pollinations.ai image URL for `prompt` (free service, no key,
    no signup — verified 2026-09 at https://pollinations.ai). The URL is
    embedded as markdown; the service renders on request."""
    from urllib.parse import quote

    return (f"https://image.pollinations.ai/prompt/{quote(prompt.strip()[:300])}"
            f"?width={width}&height={height}&nologo=true")


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
