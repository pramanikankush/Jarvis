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


def test_bad_numeric_env_vars_do_not_break_startup():
    """A typo'd numeric env var must not stop the app from booting (that would
    be a 503 on every endpoint); it warns and falls back."""
    saved = os.environ.get("DOCCHAT_WARM_MIN_MB")
    try:
        os.environ["DOCCHAT_WARM_MIN_MB"] = "seven hundred"
        assert server.env_number("DOCCHAT_WARM_MIN_MB", 700) == 700
        os.environ["DOCCHAT_WARM_MIN_MB"] = "900"
        assert server.env_number("DOCCHAT_WARM_MIN_MB", 700) == 900
        os.environ["DOCCHAT_WARM_MIN_MB"] = ""
        assert server.env_number("DOCCHAT_WARM_MIN_MB", 700) == 700
        assert server.env_number("NOT_SET_ANYWHERE", 5) == 5
    finally:
        os.environ.pop("DOCCHAT_WARM_MIN_MB", None)
        if saved is not None:
            os.environ["DOCCHAT_WARM_MIN_MB"] = saved


def test_memory_headroom_uses_the_container_budget_not_the_host():
    """In a container /proc/meminfo reports the host's RAM, so a 512 MB plan
    would look roomy and pin ~200 MB of embedding model it cannot afford."""
    saved = (server.CGROUP_LIMIT_PATHS, server.CGROUP_USED_PATHS, server.host_free_memory_mb)
    saved_env = os.environ.pop("DOCCHAT_WARM_EMBEDDINGS", None)
    saved_min = server.WARM_MIN_MB
    try:
        with tempfile.TemporaryDirectory() as td:
            limit = os.path.join(td, "memory.max")
            used = os.path.join(td, "memory.current")
            with open(limit, "w", encoding="utf-8") as f:
                f.write("536870912")  # 512 MB plan
            with open(used, "w", encoding="utf-8") as f:
                f.write("83886080")  # 80 MB already used
            server.CGROUP_LIMIT_PATHS = (limit,)
            server.CGROUP_USED_PATHS = (used,)
            server.host_free_memory_mb = lambda: 30000.0  # the host has 30 GB free
            server.WARM_MIN_MB = 700.0
            free = server.available_memory_mb()
            assert free is not None and 420 <= free <= 440, free  # 512 - 80 MB
            assert server.warm_up_enabled() is False
            # an unlimited cgroup ("max") falls back to the host figure
            with open(limit, "w", encoding="utf-8") as f:
                f.write("max")
            assert server.available_memory_mb() == 30000.0
            # and a roomy box still warms up
            server.host_free_memory_mb = lambda: 4000.0
            assert server.warm_up_enabled() is True
    finally:
        (server.CGROUP_LIMIT_PATHS, server.CGROUP_USED_PATHS, server.host_free_memory_mb) = saved
        server.WARM_MIN_MB = saved_min
        if saved_env is not None:
            os.environ["DOCCHAT_WARM_EMBEDDINGS"] = saved_env


def test_warm_up_policy_respects_env_and_free_memory():
    """The embedding model holds ~200 MB resident, so a 512 MB host must not
    preload it: an explicit DOCCHAT_WARM_EMBEDDINGS wins, else free memory does."""
    saved_env = os.environ.pop("DOCCHAT_WARM_EMBEDDINGS", None)
    saved_free = server.available_memory_mb
    try:
        server.available_memory_mb = lambda: None       # unknown -> dev machine
        assert server.warm_up_enabled() is True
        server.available_memory_mb = lambda: 2000.0
        assert server.warm_up_enabled() is True
        server.available_memory_mb = lambda: 300.0      # a 512 MB free tier
        assert server.warm_up_enabled() is False
        os.environ["DOCCHAT_WARM_EMBEDDINGS"] = "1"
        assert server.warm_up_enabled() is True         # explicit force wins
        os.environ["DOCCHAT_WARM_EMBEDDINGS"] = "0"
        server.available_memory_mb = lambda: 8000.0
        assert server.warm_up_enabled() is False        # explicit disable wins
    finally:
        server.available_memory_mb = saved_free
        os.environ.pop("DOCCHAT_WARM_EMBEDDINGS", None)
        if saved_env is not None:
            os.environ["DOCCHAT_WARM_EMBEDDINGS"] = saved_env


def test_sheet_stack_is_not_imported_by_the_server():
    """pandas + matplotlib are ~94 MB resident, so the chat/upload paths must not
    pay for them — the sheet endpoints import the module on first use instead."""
    assert not hasattr(server, "spreadsheet"), "server imports pandas/matplotlib eagerly again"


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
