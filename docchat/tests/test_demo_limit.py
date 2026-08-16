"""Tests for the free-demo chat limit.

Run: python tests/test_demo_limit.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat.store import Store
import server  # noqa: E402  (imports the chat_limit_reached helper + constants)


def _fresh_store(uid: str) -> Store:
    tmp = tempfile.mkdtemp()
    return Store(db_path=os.path.join(tmp, "test.db")).for_user(uid)


def _fill(st: Store, n: int):
    for i in range(n):
        st.create_session(f"chat {i}")


def test_local_workspace_never_limited():
    # uid "" (owner) is exempt regardless of how many chats exist
    st = _fresh_store("")
    _fill(st, 10)
    assert server.chat_limit_reached(st, "", limit=3) is False


def test_guest_capped_at_three():
    st = _fresh_store("guest:device-abc")
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is False
    _fill(st, 2)
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is False
    _fill(st, 1)  # now 3 chats
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is True
    # a 4th is refused
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is True


def test_deleting_a_chat_frees_a_slot():
    st = _fresh_store("guest:device-abc")
    _fill(st, 3)
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is True
    sid = st.list_sessions()[0]["id"]
    assert st.delete_session(sid) is True
    assert server.chat_limit_reached(st, "guest:device-abc", limit=3) is False


def test_zero_limit_disables_cap():
    st = _fresh_store("guest:device-abc")
    _fill(st, 5)
    assert server.chat_limit_reached(st, "guest:device-abc", limit=0) is False


def test_limit_is_per_user():
    a = _fresh_store("guest:aaa")
    b = _fresh_store("guest:bbb")
    _fill(a, 3)
    assert server.chat_limit_reached(a, "guest:aaa", limit=3) is True
    # user b still has a free slot
    assert server.chat_limit_reached(b, "guest:bbb", limit=3) is False


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
