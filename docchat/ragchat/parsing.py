"""Parse uploaded files into (page, text) segments, then chunk them."""
import csv
import io
import os
import re

from docx import Document as _Docx
from docx.oxml.ns import qn
from docx.table import Table as _DocxTable
from pypdf import PdfReader

SUPPORTED_EXT = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}
# hard caps on text kept per file — protects chunking/embedding time on huge files
MAX_FILE_CHARS = 300_000
MAX_ROW_TEXT = 150_000
MIN_PDF_PAGE_CHARS = 20


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def parse(name: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number_or_None, text), ...] for the file."""
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXT))}")
    if ext == ".pdf":
        return _parse_pdf(data)
    if ext == ".docx":
        text = _parse_docx(data)[:MAX_FILE_CHARS]
        return [] if not text else [(None, text)]
    if ext == ".csv":
        text = _parse_csv(data)
        return [] if not text else [(None, text)]
    text = decode_text(data)[:MAX_FILE_CHARS]
    return [] if not text.strip() else [(None, text)]


def _parse_pdf(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if len(text) >= MIN_PDF_PAGE_CHARS:
            pages.append((i, text[:MAX_FILE_CHARS]))
    if not pages:
        raise ValueError("No readable text found in this PDF (it may be a scanned/image-only document).")
    return pages


def _parse_docx(data: bytes) -> str:
    doc = _Docx(io.BytesIO(data))
    parts = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = "".join(node.text or "" for node in child.iter(qn("w:t"))).strip()
            if text:
                parts.append(text)
        elif child.tag == qn("w:tbl"):
            table = _DocxTable(child, doc)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                line = " | ".join(c for c in cells if c)
                if line:
                    parts.append(line)
    return "\n\n".join(parts)


def _parse_csv(data: bytes) -> str:
    text = decode_text(data)
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ""
    header = rows[0]
    # treat first row as header only if it doesn't look like data (mostly empty or all numbers)
    nonempty = [c for c in header if c.strip()]
    looks_like_header = bool(nonempty) and not all(_is_numeric(c) for c in nonempty) and any(
        any(ch.isalpha() for ch in c) for c in nonempty
    )
    lines, total = [], 0
    for row in rows[1:] if looks_like_header else rows:
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if looks_like_header:
            line = " | ".join(
                f"{header[i]}: {cell}" for i, cell in enumerate(cells) if i < len(header) and cell
            )
        else:
            line = ", ".join(cells)
        if not line:
            continue
        total += len(line)
        if total > MAX_ROW_TEXT:
            lines.append("[rows truncated: file is very large]")
            break
        lines.append(line)
    return "\n".join(lines)


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks: paragraphs first, then sentences, then words."""
    if not text:
        return []
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        cur = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                out.extend(_hard_split(s, max_chars, overlap))
                continue
            if cur and len(cur) + len(s) + 1 > max_chars:
                out.append(cur)
                cur = cur[-overlap:] if overlap else ""
            cur = f"{cur} {s}".strip()
        if cur:
            out.append(cur)
    return out


def _hard_split(s: str, max_chars: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + len(w) + 1 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else ""
        cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks