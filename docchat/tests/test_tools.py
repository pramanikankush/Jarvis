"""Tests for ragchat.tools: safe calculator + sandboxed python.
Run: python tests/test_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat import tools


def test_calculator_arithmetic():
    assert tools.calculate("2+3*4") == "14"
    assert tools.calculate("(1250*0.85)/12") == "88.54166667"
    assert tools.calculate("2**10") == "1024"
    assert tools.calculate("sqrt(16)") == "4.0"
    assert tools.calculate("sin(0) + cos(0)") == "1.0"
    assert tools.calculate("pi").startswith("3.14159")  # rounded to 8 decimals


def test_calculator_rejects_dangerous_input():
    assert "Error" in tools.calculate("__import__('os').system('rm -rf /')")
    assert "Error" in tools.calculate("open('/etc/passwd')")
    assert "Error" in tools.calculate("lambda x: x")
    assert "Error" in tools.calculate("1; print('hi')")
    assert "Error" in tools.calculate("a.b")
    assert "Error" in tools.calculate("foo(1)")
    assert "Error" in tools.calculate("x + 1")  # unknown name
    assert "Error" in tools.calculate("2 ** 100000000")  # exponent blow-up guard
    assert "Error" in tools.calculate("2 ** 99999")


def test_calculator_edge_cases():
    assert tools.calculate("") == "Error: empty expression"
    assert tools.calculate("1/0") == "Error: division by zero"
    assert tools.calculate("2 + 3 " * 100) == "Error: expression too long"


def test_sandbox_basic_math():
    out = tools.run_python("print(2 + 2)")
    assert out.strip() == "4", out


def test_sandbox_with_dataframe():
    df_json = {"columns": ["name", "value"], "data": [["a", 1], ["b", 2], ["c", 3]]}
    out = tools.run_python("print(df['value'].sum())", df_json=df_json)
    assert out.strip() == "6", out
    out = tools.run_python("print(df.shape)", df_json=df_json)
    assert out.strip() == "(3, 2)", out


def test_sandbox_blocks_imports_and_file_io():
    out = tools.run_python("import os\nprint(os.getcwd())")
    assert "Error" in out, out
    out = tools.run_python("open('/tmp/evil', 'w').write('x')")
    assert "Error" in out, out
    out = tools.run_python("__import__('subprocess').run(['echo', 'hi'])")
    assert "Error" in out, out


def test_sandbox_timeout_and_length():
    out = tools.run_python("print(1)\n" * 2000)  # 8k chars > 4000 limit
    assert "too long" in out, out
    out = tools.run_python("while True:\n    pass", timeout=2)
    assert "timed out" in out, out


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
