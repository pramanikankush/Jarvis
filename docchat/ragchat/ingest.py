"""Shared document ingestion: parse -> chunk -> embed -> store.

One pipeline for both upload paths — the sidebar Documents upload
(`POST /api/docs`) and chat attachments (`POST /api/attach`) — so validation,
duplicate-name handling, and error messages never drift apart.

Every failure raises with a message meant for the UI:
  * ValueError   -> bad extension / no extractable text        -> HTTP 400
  * EmbedError   -> embedding model failed to start            -> HTTP 503
"""
import asyncio
import os

from . import parsing


class EmbedError(Exception):
    """The embedding model failed to start (first run downloads ~100 MB)."""


async def ingest_doc(db, name: str, data: bytes, embed_fn=None) -> tuple[dict, int]:
    """Parse, chunk, embed, and store `data` as a document for `db` (a
    per-user store view). Returns (doc_row, chunk_count). Raises ValueError
    for unsupported/binary-empty input and EmbedError when embeddings fail."""
    try:
        pages = await asyncio.to_thread(parsing.parse, name, data)
    except ValueError:
        raise
    chunks = [(page, c) for page, text in pages for c in parsing.chunk_text(text)]
    if not chunks:
        raise ValueError("No extractable text found in this file")
    embed = embed_fn
    if embed is None:
        from . import llm

        embed = llm.embed_texts
    try:
        vecs = await asyncio.to_thread(embed, [c for _, c in chunks])
    except Exception as e:  # noqa: BLE001 — any startup/IO failure is a 503 for the UI
        raise EmbedError(str(e)) from e
    # duplicate names get " (2)", " (3)"… suffixes so both stay addressable
    existing = {d["name"] for d in db.list_docs()}
    base, ext = os.path.splitext(name)
    i = 2
    while name in existing:
        name = f"{base} ({i}){ext}"
        i += 1
    doc = await asyncio.to_thread(db.add_doc, name, len(data), chunks, list(vecs))
    return doc, len(chunks)


def resolve_attachments(db, raw) -> list[dict]:
    """Validate a chat request's `attachments` field: [{"id": int, "name": str}].
    Only documents the caller actually owns are accepted — a foreign or garbage
    id is skipped with a log line, never a crash. Returns prompt-ready rows
    ({id, name, chunks}); at most 5 attachments per message."""
    import logging

    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    log = logging.getLogger("jarvis.ingest")
    for a in raw[:5]:
        if not isinstance(a, dict):
            continue
        try:
            doc_id = int(a.get("id") or 0)
        except (TypeError, ValueError):
            continue
        doc = db.get_doc(doc_id)
        if doc:
            out.append({"id": doc["id"], "name": doc["name"],
                        "chunks": doc.get("chunks", 0)})
        else:
            log.warning("chat attachment skipped (not found for this user): %r", a)
    return out
