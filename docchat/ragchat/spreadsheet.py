"""Spreadsheet analysis on top of pandas (spec: "use Pandas instead of building
a custom spreadsheet engine"). Every operation returns plain data (dicts /
markdown strings) so the agent can narrate it, plus a chart endpoint that
renders PNGs server-side (matplotlib, Agg backend — no display needed).
"""
import io
import logging
import os
import re

import numpy as np
import pandas as pd

log = logging.getLogger("jarvis.spreadsheet")

# matplotlib must pick the Agg backend before pyplot is imported (headless)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

ALLOWED_KINDS = {".csv", ".xlsx"}
MAX_CHART_ROWS = 5000
MAX_PREVIEW_ROWS = 2000
_cache: dict[str, tuple[float, object]] = {}  # path -> (mtime, DataFrame)


def load_sheet(path: str, kind: str) -> "pd.DataFrame":
    """Load a csv/xlsx file into a DataFrame, cached by path+mtime."""
    import pandas as pd

    mtime = os.path.getmtime(path)
    if path in _cache and _cache[path][0] == mtime:
        return _cache[path][1]
    try:
        if kind == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        raise ValueError(f"Could not parse spreadsheet: {e}")
    df.columns = [str(c).strip() if str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)]
    _cache[path] = (mtime, df)
    return df


def clear_cache(path: str | None = None) -> None:
    if path:
        _cache.pop(path, None)
    else:
        _cache.clear()


def sheet_info(df) -> dict:
    """Column summary: name, dtype, non-null count, unique count + head rows."""
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": str(c),
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "unique": int(s.nunique()),
            "sample": _fmt_value(s.dropna().iloc[0]) if s.notna().any() else None,
        })
    head = df.head(5).astype(object).where(df.head(5).notna(), None)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": cols,
        "head": [[_fmt_value(v) for v in row] for row in head.values.tolist()],
    }


def _numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def describe(df, columns: list[str] | None = None) -> str:
    """Markdown table of summary stats for numeric columns + top values for text."""
    import pandas as pd

    cols = [c for c in (columns or _numeric_cols(df)) if c in df.columns]
    if not cols:
        if columns:
            return (f"Error: column(s) not found: {', '.join(columns)}. "
                    f"Available columns: {', '.join(map(str, df.columns))}")
        return "No numeric columns found to summarise."
    lines = ["| column | count | mean | std | min | 25% | median | 75% | max |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cols:
        s = df[c].dropna()
        if not len(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        q = s.quantile([0.25, 0.5, 0.75])
        lines.append(
            f"| {c} | {len(s)} | {s.mean():.3g} | {s.std():.3g} | {s.min():.3g} "
            f"| {q[0.25]:.3g} | {q[0.5]:.3g} | {q[0.75]:.3g} | {s.max():.3g} |"
        )
    text_cols = [c for c in df.columns if df[c].dtype == object and df[c].notna().any()]
    if text_cols:
        lines.append("")
        lines.append("**Text columns — most common values:**")
        for c in text_cols[:3]:
            top = df[c].dropna().astype(str).value_counts().head(3)
            lines.append(f"- `{c}`: " + ", ".join(f"{v} ({n})" for v, n in top.items()))
    return "\n".join(lines)


def groupby(df, by: str, agg_column: str | None = None, agg: str = "mean", limit: int = 10) -> str:
    """Group by `by`, aggregate `agg_column` with `agg` (mean/sum/count/min/max)."""
    import pandas as pd

    if by not in df.columns:
        return f"Error: column '{by}' not found. Available: {', '.join(map(str, df.columns))}"
    agg = agg or "mean"
    if agg == "count":
        g = df.groupby(by, dropna=True).size().reset_index(name="count")
    else:
        col = agg_column or next(iter(_numeric_cols(df)), None)
        if col is None:
            return "Error: no numeric column to aggregate. Use agg='count' instead."
        if agg not in ("mean", "sum", "min", "max", "median"):
            return f"Error: aggregation '{agg}' not supported (mean/sum/count/min/max/median)."
        g = df.groupby(by, dropna=True)[col].agg(agg).reset_index()
        g.columns = [by, f"{agg}({col})"]
    g = g.sort_values(g.columns[-1], ascending=False).head(min(limit, 50))
    lines = ["| " + " | ".join(map(str, g.columns)) + " |", "|" + "---|" * len(g.columns)]
    for _, row in g.iterrows():
        lines.append("| " + " | ".join(_fmt_value(v) for v in row) + " |")
    return "\n".join(lines)


_SAFE_QUERY = re.compile(r"^[A-Za-z0-9_ .<>=!&|()'\"\[\]\-]+$")
_CALL_LIKE = re.compile(r"[A-Za-z_]\w*\s*\(")


def filter_rows(df, condition: str, limit: int = 20) -> str:
    """Filter rows with a sanitised pandas query expression (e.g. `price > 100`)."""
    import pandas as pd

    cond = (condition or "").strip()
    if not cond:
        return "Error: empty filter condition"
    if not _SAFE_QUERY.match(cond) or _CALL_LIKE.search(cond) and "(" in cond:
        return "Error: filter condition contains disallowed characters (letters+functions or symbols)."
    try:
        out = df.query(cond, engine="python")
    except Exception as e:
        return f"Error: invalid filter condition ({e})"
    if out.empty:
        return f"No rows match the condition `{cond}`."
    lines = ["| " + " | ".join(map(str, out.columns)) + " |", "|" + "---|" * len(out.columns)]
    for _, row in out.head(limit).iterrows():
        lines.append("| " + " | ".join(_fmt_value(v) for v in row) + " |")
    lines.append(f"\n_Matched {len(out)} of {len(df)} rows (showing {min(limit, len(out))})._")
    return "\n".join(lines)


def anomalies(df, column: str, limit: int = 20) -> str:
    """Detect outliers via IQR (robust) and z-score; return the offending rows."""
    import pandas as pd

    if column not in df.columns:
        return f"Error: column '{column}' not found."
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(s) < 4:
        return "Error: need at least 4 numeric values to detect anomalies."
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    if not mask.any():
        mean, std = s.mean(), s.std()
        if std > 0:
            z = (s - mean) / std
            mask = z.abs() > 3
    if not mask.any():
        return f"No anomalies detected in `{column}` (IQR and z>3 checks)."
    idx = s[mask].index.tolist()[:limit]
    lines = [f"**{len(s[mask])} potential anomaly(ies) in `{column}`** (IQR bounds: "
             f"{_fmt_value(lo)} … {_fmt_value(hi)}):"]
    for i in idx:
        row = df.loc[i].to_dict()
        lines.append(f"- row {i}: `{column}` = {_fmt_value(s.loc[i])}  " +
                     ", ".join(f"{k}={_fmt_value(v)}" for k, v in list(row.items())[:4] if k != column))
    return "\n".join(lines)


def chart(df, chart_type: str, column: str, group: str | None = None, limit: int = 10) -> bytes:
    """Render a PNG chart. Types: bar, line, hist, box."""
    import pandas as pd

    df = df.head(MAX_CHART_ROWS)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    try:
        if chart_type == "hist":
            s = pd.to_numeric(df[column], errors="coerce").dropna()
            ax.hist(s, bins=min(30, max(5, s.nunique() if s.nunique() else 10)), color="#10a37f")
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("count")
        elif chart_type == "line":
            ax.plot(df[column].astype(str), _num(df, group or column), color="#10a37f")
            ax.set_title(f"{group or column} over {column}")
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "box":
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:8]
            ax.boxplot([df[c].dropna() for c in cols], labels=[str(c) for c in cols])
            ax.set_title("Box plot")
            plt.xticks(rotation=45, ha="right")
        else:  # bar
            if group and group in df.columns and pd.api.types.is_numeric_dtype(df[group]):
                g = df.groupby(column, dropna=True)[group].mean().sort_values(ascending=False).head(limit)
                ax.bar([str(i) for i in g.index], g.values, color="#10a37f")
                ax.set_ylabel(f"mean({group})")
            else:
                vc = df[column].astype(str).value_counts().head(limit)
                ax.bar([str(i)[:30] for i in vc.index], vc.values, color="#10a37f")
                ax.set_ylabel("count")
            ax.set_title(f"{'mean(' + group + ') by' if group else 'Counts of'} {column}")
            plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
    finally:
        plt.close(fig)


def _num(df, column):
    s = pd.to_numeric(df[column], errors="coerce")
    return s.ffill().bfill().fillna(0).tolist()


def _fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)[:60]


def to_split_json(df, max_rows: int = MAX_PREVIEW_ROWS) -> dict:
    """Pandas split-format dict for the sandbox tool (capped at max_rows)."""
    df = df.head(max_rows)
    return {
        "columns": [str(c) for c in df.columns],
        "data": df.astype(object).where(df.notna(), None).values.tolist(),
    }
