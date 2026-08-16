"""Run the full Jarvis test suite: python tests/run_all.py"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULES = ["test_check", "test_envfile", "test_retrieval", "test_tools", "test_spreadsheet",
           "test_memory", "test_websearch", "test_usagetrack", "test_agent", "test_auth",
           "test_demo_limit"]


def main():
    failed = 0
    for name in MODULES:
        print(f"\n=== {name} ===")
        try:
            mod = importlib.import_module(name)
            mod.main()
        except SystemExit as e:
            if e.code:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"  MODULE FAILED TO IMPORT: {e}")
    print(f"\n{'=' * 40}\n{'ALL SUITES PASSED' if not failed else f'{failed} suite(s) failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
