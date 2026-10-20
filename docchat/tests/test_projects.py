"""Unit tests for project import, indexing, and scoped project storage."""
import io
import os
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragchat import projects  # noqa: E402
from ragchat.store import Store  # noqa: E402


def _zip(files: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def _embed(texts):
    return np.ones((len(texts), 4), dtype=np.float32) / 2.0


def _store(tmp_path) -> Store:
    return Store(str(tmp_path / "projects.db")).for_user("guest:aaa11111")


def test_extract_archive_rejects_zip_slip(tmp_path):
    data = _zip({"../evil.txt": "nope"})
    try:
        projects.extract_archive(data, str(tmp_path / "src"))
    except ValueError as e:
        assert "traversal" in str(e) or "escapes" in str(e)
    else:
        raise AssertionError("zip-slip archive was accepted")


def test_extract_archive_rejects_symlink(tmp_path):
    buf = io.BytesIO()
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(info, "target")
    try:
        projects.extract_archive(buf.getvalue(), str(tmp_path / "src"))
    except ValueError as e:
        assert "symlink" in str(e)
    else:
        raise AssertionError("symlink archive was accepted")


def test_index_workspace_skips_secrets_binaries_and_deps(tmp_path):
    root = tmp_path / "src"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (root / ".env").write_text("GROQ_API_KEY=sk-should-not-index\n", encoding="utf-8")
    (root / "leak.py").write_text("KEY = 'AKIA1234567890ABCDEF'\n", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("console.log('dep')", encoding="utf-8")
    st = _store(tmp_path)
    try:
        proj = st.create_project("Demo", "upload", status="importing")
        stats = projects.index_workspace(st, proj["id"], str(root), _embed)
        files = [f["path"] for f in st.list_project_files(proj["id"])]
        assert "pkg/app.py" in files
        assert ".env" not in files
        assert "leak.py" not in files
        assert "blob.bin" not in files
        assert all("node_modules" not in f for f in files)
        assert stats["files"] == 1 and stats["chunks"] >= 1 and stats["skipped_secret"] == 1
        vec = np.ones(4, dtype=np.float32) / 2.0
        assert st.project_vector_search(proj["id"], vec, k=3)
        assert st.project_keyword_search(proj["id"], "hello", k=3)
    finally:
        st.close()


def test_create_from_archive_is_scoped(tmp_path):
    st = _store(tmp_path)
    try:
        data = _zip({"src/app.py": "def greet():\n    return 'hello project'\n"})
        result = projects.create_from_archive(st, "Demo", "demo.zip", data, _embed)
        pid = result["project"]["id"]
        assert result["project"]["status"] == "ready"
        assert st.get_project(pid)["name"] == "Demo"
        assert st.project_total_chunks(pid) >= 1
        other = Store(str(tmp_path / "other.db")).for_user("guest:bbb22222")
        try:
            assert other.get_project(pid) is None
            assert other.project_vector_search(pid, np.ones(4, dtype=np.float32), k=3) == []
            assert other.project_keyword_search(pid, "greet", k=3) == []
        finally:
            other.close()
        assert projects.read_project_file(st, pid, "src/app.py").startswith("def greet")
        try:
            projects.read_project_file(st, pid, "../secret.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("path traversal read was allowed")
        assert projects.delete_project(st, pid) is True
        assert st.get_project(pid) is None
        assert not os.path.exists(projects.project_root(pid))
    finally:
        st.close()


def test_project_instruction_versioning(tmp_path):
    st = _store(tmp_path)
    try:
        pid = st.create_project("Demo", "upload")["id"]
        v1 = st.add_instruction(pid, "Use pytest.", created_by="guest:aaa11111")
        v2 = st.add_instruction(pid, "Use pytest and ruff.", created_by="guest:aaa11111")
        assert v1["version"] == 1 and v2["version"] == 2
        assert st.active_instruction(pid)["content"] == "Use pytest and ruff."
        assert [i["version"] for i in st.list_instructions(pid)] == [2, 1]
        block = projects.format_project_block(st.get_project(pid), st.active_instruction(pid),
                                               len(st.list_project_files(pid)))
        assert "PROJECT CONTEXT" in block and "Demo" in block and "v2" in block
    finally:
        st.close()


def test_unsupported_archive_is_clear(tmp_path):
    try:
        projects.extract_archive(b"not an archive", str(tmp_path / "src"))
    except ValueError as e:
        assert "Unsupported project archive" in str(e)
    else:
        raise AssertionError("invalid archive was accepted")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                import pathlib
                t(pathlib.Path(td))
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
