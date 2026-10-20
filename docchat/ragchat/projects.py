"""Project import + indexing.

A project is a source tree (uploaded archive for v1; GitHub/companion later)
that is indexed into project_files / project_chunks. Archives are extracted
with explicit path, type, size, and member-count checks — never `extractall`.
"""
import hashlib
import io
import logging
import os
import re
import shutil
import tarfile
import zipfile

from . import parsing
from .store import DATA_DIR

log = logging.getLogger("jarvis.projects")


class ProjectEmbedError(Exception):
    """The local embedding model could not run; the upload should return 503."""


PROJECTS_DIR = os.path.join(DATA_DIR, "projects")
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000
MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024
MAX_INDEX_CHARS = 20_000_000
CHUNK_LINES = 80
CHUNK_OVERLAP = 20

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build", "target",
    "vendor", ".next", ".nuxt", "coverage", ".idea", ".vscode",
}
SKIP_FILE_NAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "credentials", "credentials.json", "secrets", "secrets.json",
}
TEXT_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs",
    ".java", ".kt", ".kts", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb",
    ".php", ".swift", ".scala", ".sh", ".bash", ".zsh", ".ps1", ".sql",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".md", ".markdown",
    ".txt", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".xml", ".gradle", ".dockerfile", ".make", ".mk", ".gitignore",
    ".gitattributes", ".editorconfig",
}
LANGUAGE_BY_EXT = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".sql": "sql", ".html": "html", ".htm": "html", ".css": "css",
    ".scss": "scss", ".sass": "sass", ".less": "less", ".md": "markdown",
    ".json": "json", ".jsonc": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".xml": "xml",
}
_SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)


def project_root(pid: str) -> str:
    return os.path.join(PROJECTS_DIR, pid, "source")


def _safe_target(root: str, name: str) -> str:
    name = (name or "").replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Archive contains an absolute path: {name!r}")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"Archive contains a path traversal entry: {name!r}")
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, *parts))
    if os.path.commonpath([root_real, target]) != root_real:
        raise ValueError(f"Archive entry escapes the workspace: {name!r}")
    return target


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def extract_archive(data: bytes, dest: str) -> None:
    """Safely extract a zip or tar archive into `dest`.

    Raises ValueError for traversal, symlinks, device files, oversized members,
    too many members, or an unsupported archive format.
    """
    os.makedirs(dest, exist_ok=True)
    if zipfile.is_zipfile(io.BytesIO(data)):
        _extract_zip(data, dest)
        return
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*"):
            pass
    except tarfile.TarError as e:
        raise ValueError("Unsupported project archive: upload a .zip, .tar, .tar.gz, or .tgz file.") from e
    _extract_tar(data, dest)


def _extract_zip(data: bytes, dest: str) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            raise ValueError(f"Archive has too many files ({len(infos)}; max {MAX_ARCHIVE_FILES}).")
        total = sum(max(0, i.file_size) for i in infos)
        if total > MAX_EXTRACTED_BYTES:
            raise ValueError(
                f"Archive expands to {total // (1024 * 1024)} MB "
                f"(max {MAX_EXTRACTED_BYTES // (1024 * 1024)} MB)."
            )
        for info in infos:
            if _zip_is_symlink(info):
                raise ValueError(f"Archive contains a symlink: {info.filename!r}")
            if info.is_dir():
                _safe_target(dest, info.filename)
                continue
            target = _safe_target(dest, info.filename)
            if info.file_size > MAX_EXTRACTED_BYTES:
                raise ValueError(f"Archive member '{info.filename}' is too large.")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            written = 0
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_EXTRACTED_BYTES:
                        raise ValueError(f"Archive member '{info.filename}' expands too large.")
                    out.write(chunk)


def _extract_tar(data: bytes, dest: str) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        members = tf.getmembers()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError(f"Archive has too many files ({len(members)}; max {MAX_ARCHIVE_FILES}).")
        total = sum(max(0, m.size) for m in members if m.isfile())
        if total > MAX_EXTRACTED_BYTES:
            raise ValueError(
                f"Archive expands to {total // (1024 * 1024)} MB "
                f"(max {MAX_EXTRACTED_BYTES // (1024 * 1024)} MB)."
            )
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Archive contains a link: {member.name!r}")
            if member.isdev() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Archive contains an unsupported entry: {member.name!r}")
            target = _safe_target(dest, member.name)
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            written = 0
            with src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_EXTRACTED_BYTES:
                        raise ValueError(f"Archive member '{member.name}' expands too large.")
                    out.write(chunk)


def _secret_hit(text: str) -> bool:
    sample = text[:200_000]
    return any(p.search(sample) for p in _SECRET_PATTERNS)


def _chunk_lines(text: str) -> list[dict]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        block = lines[start:start + CHUNK_LINES]
        if not block:
            continue
        body = "\n".join(block).strip()
        if not body:
            continue
        chunks.append({"start_line": start + 1, "end_line": start + len(block), "text": body})
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks


def index_workspace(db, pid: str, root: str, embed_fn, batch_size: int = 64) -> dict:
    """Walk a project tree, skip binaries/secrets, chunk text, embed, and store."""
    if not db.get_project(pid):
        raise ValueError("Project not found")
    if not os.path.isdir(root):
        raise ValueError("Project workspace is missing — re-import the project.")
    files: list[dict] = []
    chunks: list[dict] = []
    total_chars = 0
    skipped_secret = 0
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(current, d)))
        for name in sorted(names):
            path = os.path.join(current, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if name.lower() in SKIP_FILE_NAMES or name.startswith(".env."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in TEXT_EXTS:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > MAX_TEXT_FILE_BYTES:
                continue
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = parsing.decode_text(raw)
            if _secret_hit(text):
                skipped_secret += 1
                log.warning("project index skipped a file that looks like it contains a secret: %s", rel)
                continue
            if total_chars + len(text) > MAX_INDEX_CHARS:
                log.warning("project index hit the text cap; remaining files are not indexed")
                break
            total_chars += len(text)
            sha = hashlib.sha256(raw).hexdigest()
            files.append({"path": rel, "language": LANGUAGE_BY_EXT.get(ext), "size": size, "sha256": sha})
            for chunk in _chunk_lines(text):
                chunks.append({**chunk, "path": rel, "sha256": sha})
        if total_chars >= MAX_INDEX_CHARS:
            break
    if not chunks:
        db.replace_project_index(pid, files, [], [])
        return {"files": len(files), "chunks": 0, "skipped_secret": skipped_secret}
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        batch = [c["text"] for c in chunks[start:start + batch_size]]
        try:
            vectors = embed_fn(batch)
        except Exception as e:
            raise ProjectEmbedError(str(e)) from e
        embeddings.extend(vectors)
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count did not match the indexed chunks.")
    db.replace_project_index(pid, files, chunks, embeddings)
    return {"files": len(files), "chunks": len(chunks), "skipped_secret": skipped_secret}


def create_from_archive(db, name: str, filename: str, data: bytes, embed_fn) -> dict:
    """Create a project from an uploaded archive, extract it, and index it."""
    name = (name or "").strip() or os.path.splitext(os.path.basename(filename))[0] or "Project"
    project = db.create_project(name[:120], "upload", source_ref=filename, status="importing")
    pid = project["id"]
    root = project_root(pid)
    try:
        os.makedirs(root, exist_ok=True)
        extract_archive(data, root)
        stats = index_workspace(db, pid, root, embed_fn)
        updated = db.update_project(pid, status="ready") or project
        return {"project": updated, "stats": stats}
    except Exception:
        db.update_project(pid, status="error")
        log.exception("project import failed: %s", filename)
        raise


def sync_project(db, pid: str, embed_fn) -> dict:
    project = db.get_project(pid)
    if not project:
        raise ValueError("Project not found")
    root = project_root(pid)
    if not os.path.isdir(root):
        raise ValueError("Project workspace is missing — re-import the project.")
    stats = index_workspace(db, pid, root, embed_fn)
    updated = db.update_project(pid, status="ready") or project
    return {"project": updated, "stats": stats}


def delete_project(db, pid: str) -> bool:
    if not db.delete_project(pid):
        return False
    shutil.rmtree(os.path.join(PROJECTS_DIR, pid), ignore_errors=True)
    return True


def read_project_file(db, pid: str, path: str, start_line: int | None = None,
                      end_line: int | None = None, max_chars: int = 20_000) -> str:
    if not db.get_project(pid):
        raise ValueError("Project not found")
    root = project_root(pid)
    target = _safe_target(root, path)
    if not os.path.isfile(target) or os.path.islink(target):
        raise ValueError(f"File not found in project: {path}")
    if os.path.getsize(target) > MAX_TEXT_FILE_BYTES:
        raise ValueError("File is too large to read.")
    with open(target, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if start_line or end_line:
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line or start + 200))
        lines = lines[start - 1:end]
    return "".join(lines)[:max_chars]


def format_project_block(project: dict, instruction: dict | None, file_count: int) -> str:
    if not project:
        return ""
    parts = [
        "PROJECT CONTEXT:",
        f"- Name: {project.get('name', '')}",
        f"- Source: {project.get('source_type', '')}"
        + (f" ({project.get('source_ref')})" if project.get("source_ref") else ""),
        f"- Status: {project.get('status', '')}",
        f"- Indexed files: {file_count}",
    ]
    if project.get("default_branch"):
        parts.append(f"- Default branch: {project['default_branch']}")
    if instruction and instruction.get("content"):
        parts.append(f"- Instructions (v{instruction.get('version', 1)}):\n{instruction['content'][:4000]}")
    parts.append(
        "Use project_search / read_project_file to inspect the code before answering. "
        "Never claim to have run tests or changed files unless a job result says so."
    )
    return "\n".join(parts)
