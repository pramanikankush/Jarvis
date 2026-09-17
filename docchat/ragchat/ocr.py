"""OCR fallback for image-only PDFs and image files (scans, screenshots).

Deliberately lazy: RapidOCR (~60 MB onnxruntime) and pypdfium2 (~3 MB) are
imported only when a file actually needs OCR, so installs that never see a
scanned document pay nothing at startup. Models ship inside the RapidOCR
wheel — no download happens at runtime.

Every function returns plain text or "" and never raises past a logged
warning for a single failed page — OCR is a fallback, not a hard dependency.
"""
import io
import logging

import numpy as np

log = logging.getLogger("jarvis.ocr")

_engine = None  # cached RapidOCR instance (model load is ~1 s on first use)

# hard cap on OCR work per document (Render free tier: 0.1 CPU)
MAX_OCR_PAGES = 30
RENDER_SCALE = 2.0  # 144 dpi — enough for OCR without blowing up memory
MAX_PAGE_CHARS = 30_000


def available() -> bool:
    """True when the optional OCR stack (engine, PDF renderer, Pillow) is
    importable. Cheap and safe to call on every upload."""
    try:
        import PIL  # noqa: F401
        import pypdfium2  # noqa: F401
    except ImportError:
        return False
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import rapidocr  # noqa: F401  (newer unified package name)
        return True
    except ImportError:
        return False


def _get_engine():
    """Lazy-load (once) the RapidOCR engine; raises ImportError with a pip
    hint when the optional dependency is missing."""
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR  # newer unified package
        _engine = RapidOCR()
    return _engine


def _call_engine(engine, img: np.ndarray):
    """Run the engine and normalize its return shape to a single result.

    RapidOCR 1.x returns ``(rows, elapse)`` where rows are [box, text, score];
    RapidOCR 3.x returns one result object (``RapidOCROutput``). Unpacking the
    3.x object raised TypeError, which the caller swallowed — every image and
    scanned page then OCR'd to "" and looked like an unreadable file.
    """
    out = engine(img)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]  # 1.x: (rows, elapse)
    return out  # 3.x: the result object itself


def _lines_from_result(result) -> list[str]:
    """Recognized text lines from a RapidOCR result, shape-agnostically:
    a 3.x object exposing ``txts``, or 1.x rows shaped [box, text, score].
    Anything unexpected is logged and skipped — never raised."""
    if result is None:
        return []
    items = getattr(result, "txts", None)
    if items is None:
        items = result
    lines: list[str] = []
    try:
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                text = item[1]
            else:
                continue
            text = text.strip()
            if text:
                lines.append(text)
    except TypeError as e:
        log.warning("unexpected OCR result shape: %s", e)
    return lines


def ocr_image(img: np.ndarray) -> str:
    """Run OCR on an RGB uint8 image array; return the recognized lines
    joined with newlines ("" when nothing is recognized)."""
    engine = _get_engine()
    try:
        result = _call_engine(engine, img)
    except Exception as e:  # engine-level failure (corrupt frame, model error)
        log.warning("ocr_image failed: %s", e)
        return ""
    return "\n".join(_lines_from_result(result))


def ocr_image_bytes(data: bytes) -> str:
    """Decode an image (PNG/JPG/WEBP/TIFF/…) and OCR it. Undecodable bytes
    return "" (a fallback must not raise past its caller)."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        log.warning("could not decode image for OCR: %s", e)
        return ""
    return ocr_image(np.asarray(img))


def ocr_pdf(data: bytes) -> list[tuple[int, str]]:
    """Render each PDF page to an image and OCR it. Returns
    [(page_number, text), ...] — text may be "" for pages that fail; a page
    failure never aborts the document (logged, skipped)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(io.BytesIO(data))
    out: list[tuple[int, str]] = []
    try:
        total = min(len(pdf), MAX_OCR_PAGES)
        if len(pdf) > MAX_OCR_PAGES:
            log.warning("PDF has %d pages; OCR capped at %d", len(pdf), MAX_OCR_PAGES)
        for i in range(total):
            page = pdf[i]
            try:
                bitmap = page.render(scale=RENDER_SCALE)
                pil = bitmap.to_pil().convert("RGB")
                text = ocr_image(np.asarray(pil))[:MAX_PAGE_CHARS]
            except Exception as e:
                log.warning("OCR failed on page %d: %s", i + 1, e)
                text = ""
            finally:
                try:
                    page.close()
                except Exception:  # close() is best-effort cleanup
                    pass
            out.append((i + 1, text))
    finally:
        try:
            pdf.close()
        except Exception:
            pass
    return out
