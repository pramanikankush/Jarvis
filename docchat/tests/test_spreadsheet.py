"""Tests for ragchat.spreadsheet. Run: python tests/test_spreadsheet.py"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat import spreadsheet

CSV = ("region,product,price,units\n"
       "north,widget,10.0,100\n"
       "north,gadget,25.5,40\n"
       "south,widget,12.0,80\n"
       "south,gadget,30.0,15\n"
       "east,widget,11.5,200\n")


def _write_csv(path, text=CSV):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def test_load_and_info():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        info = spreadsheet.sheet_info(df)
        assert info["rows"] == 5 and info["cols"] == 4
        cols = {c["name"] for c in info["columns"]}
        assert cols == {"region", "product", "price", "units"}
        spreadsheet.clear_cache(p)


def test_describe_stats():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        out = spreadsheet.describe(df)
        assert "price" in out and "mean" in out and "units" in out
        assert "| price | 5 |" in out
        # unknown explicit column -> useful error listing real columns
        err = spreadsheet.describe(df, ["units_sold"])
        assert "units_sold" in err and "units" in err and "Error" in err
        spreadsheet.clear_cache(p)


def test_groupby():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        out = spreadsheet.groupby(df, "region", "units", "sum")
        assert "north" in out and "south" in out
        # north = 140 units, east = 200, south = 95 -> east first
        lines = [l for l in out.splitlines() if l.startswith("| north")]
        assert "140" in lines[0], out
        out_bad = spreadsheet.groupby(df, "missing", "units", "sum")
        assert "Error" in out_bad
        spreadsheet.clear_cache(p)


def test_filter_rows():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        out = spreadsheet.filter_rows(df, "price > 20")
        assert "Matched 2 of 5 rows" in out
        out_bad = spreadsheet.filter_rows(df, "open('/etc/passwd')")
        assert "Error" in out_bad
        spreadsheet.clear_cache(p)


def test_anomalies():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        # price column [10, 25.5, 12, 30, 11.5] has no IQR outliers
        out = spreadsheet.anomalies(df, "price")
        assert "No anomalies" in out
        spreadsheet.clear_cache(p)


def test_anomalies_detect_outlier():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "odd.csv"),
                       "id,value\n1,10\n2,11\n3,9\n4,10\n5,12\n6,1000\n7,11\n8,10\n")
        df = spreadsheet.load_sheet(p, ".csv")
        out = spreadsheet.anomalies(df, "value")
        assert "anomal" in out.lower() and "1000" in out
        spreadsheet.clear_cache(p)


def test_chart_png():
    with tempfile.TemporaryDirectory() as td:
        p = _write_csv(os.path.join(td, "sales.csv"))
        df = spreadsheet.load_sheet(p, ".csv")
        png = spreadsheet.chart(df, "bar", "product")
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG header"
        png2 = spreadsheet.chart(df, "hist", "price")
        assert png2[:4] == b"\x89PNG"
        png3 = spreadsheet.chart(df, "line", "region", "units")
        assert png3[:4] == b"\x89PNG"
        spreadsheet.clear_cache(p)


def test_xlsx_roundtrip():
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "book.xlsx")
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(path, index=False)
        df = spreadsheet.load_sheet(path, ".xlsx")
        info = spreadsheet.sheet_info(df)
        assert info["rows"] == 3 and info["cols"] == 2
        spreadsheet.clear_cache(path)


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
