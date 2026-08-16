"""Hybrid retrieval: vector (cosine) + BM25 (FTS5) fused with Reciprocal Rank Fusion.

Why RRF instead of weighted score sums? Weights for cosine vs BM25 are
dataset-dependent and fragile; RRF needs no tuning and is robust across
corpora (Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods", SIGIR 2009). The constant 60 dampens the
rank advantage of the single best list.

Failure handling (spec requirement):
  * embedding fails  -> keyword-only fallback
  * FTS/BM25 fails   -> vector-only fallback
  * both empty       -> []
The caller (agent) treats an empty result as "not covered" and can fall back
to web search or a clear not-found answer.
"""
import logging

log = logging.getLogger("jarvis.retrieval")

RRF_K = 60  # RRF damping constant


def rrf_fuse(vector_hits: list[dict], keyword_hits: list[dict], k: int) -> list[dict]:
    """Fuse two ranked lists (each with a 'text' key) into one RRF-ranked list.

    Each returned item carries both signals ('vector_score', 'bm25') plus the
    fused 'score' (normalised RRF, in [0, 1]).
    """
    # key by chunk text: the same chunk appears as *different dict objects* in
    # the vector and keyword result lists, so object identity would double it
    scores: dict[str, dict] = {}
    for rank, hit in enumerate(vector_hits):
        key = hit["text"]
        item = scores.setdefault(
            key, {"doc_name": hit["doc_name"], "page": hit.get("page"), "text": hit["text"]}
        )
        item["vector_score"] = float(hit.get("score", 0.0))
        item["rrf"] = item.get("rrf", 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, hit in enumerate(keyword_hits):
        key = hit["text"]
        item = scores.setdefault(
            key, {"doc_name": hit["doc_name"], "page": hit.get("page"), "text": hit["text"]}
        )
        item["bm25"] = float(hit.get("bm25", 0.0))
        item["rrf"] = item.get("rrf", 0.0) + 1.0 / (RRF_K + rank + 1)

    items = sorted(scores.values(), key=lambda x: -x["rrf"])[:k]
    # normalise for display: best item gets 1.0
    best = items[0]["rrf"] if items else 1.0
    for it in items:
        it["score"] = round(it["rrf"] / best, 4)
        it.setdefault("vector_score", 0.0)
        it.setdefault("bm25", 0.0)
    return items


def retrieve(
    store,
    query: str,
    k: int = 8,
    embed_fn=None,
    keyword_only: bool = False,
) -> tuple[list[dict], dict]:
    """Hybrid retrieval for `query`. Returns (hits, meta) where meta describes
    which signals succeeded (for logging / debugging).

    `embed_fn(query) -> np.ndarray` is injected so tests can stub it and the
    server can pass the real embedder; when it raises, we degrade to keyword.
    """
    meta = {"vector": False, "keyword": False, "fallback": None}
    vector_hits: list[dict] = []
    keyword_hits: list[dict] = []

    if not keyword_only and embed_fn is not None:
        try:
            q_vec = embed_fn(query)
            vector_hits = store.search(q_vec, k=k * 2) or []
            meta["vector"] = True
        except Exception as e:  # embedding model failed to load/run
            log.warning("vector search failed, falling back to keyword: %s", e)
            meta["fallback"] = f"vector failed ({e}); used keyword only"

    try:
        keyword_hits = store.keyword_search(query, k=k * 2) or []
        meta["keyword"] = True
    except Exception as e:
        log.warning("keyword search failed: %s", e)

    if not vector_hits and not keyword_hits:
        return [], meta
    if not vector_hits:
        meta["fallback"] = meta["fallback"] or "keyword only (no vector results)"
        return keyword_hits[:k], meta
    if not keyword_hits:
        meta["fallback"] = meta["fallback"] or "vector only (no keyword results)"
        return vector_hits[:k], meta
    return rrf_fuse(vector_hits, keyword_hits, k), meta
