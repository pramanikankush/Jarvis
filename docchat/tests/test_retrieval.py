"""Tests for ragchat.retrieval (RRF fusion, hybrid fallbacks) + store keyword search.
Run: python tests/test_retrieval.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ragchat import retrieval
from ragchat.store import Store


def test_rrf_fuse_merges_and_ranks():
    vec = [
        {"doc_name": "a", "page": 1, "text": "alpha", "score": 0.9},
        {"doc_name": "b", "page": 1, "text": "beta", "score": 0.7},
        {"doc_name": "c", "page": 1, "text": "gamma", "score": 0.5},
    ]
    kw = [
        {"doc_name": "b", "page": 1, "text": "beta", "bm25": -1.0},
        {"doc_name": "c", "page": 1, "text": "gamma", "bm25": -2.0},
        {"doc_name": "d", "page": 1, "text": "delta", "bm25": -3.0},
    ]
    # gamma appears in both lists (rank 3 vec + rank 2 kw) so it outranks the
    # single-list alpha; beta is top of both lists
    fused = retrieval.rrf_fuse(vec, kw, k=4)
    assert [f["text"] for f in fused] == ["beta", "gamma", "alpha", "delta"], fused
    assert fused[0]["score"] == 1.0  # normalized
    assert fused[0]["vector_score"] == 0.7 and fused[0]["bm25"] == -1.0


def test_rrf_fuse_rank_damping():
    # a hit at rank 1 in only one list must outrank nothing — verify k bound works
    vec = [{"doc_name": "x", "page": 1, "text": t, "score": 1.0} for t in "abcdefghij"]
    kw = []
    fused = retrieval.rrf_fuse(vec, kw, k=3)
    assert [f["text"] for f in fused] == ["a", "b", "c"]


class _FakeStore:
    def __init__(self, vec, kw, n=3):
        self._vec, self._kw, self._n = vec, kw, n

    def search(self, qv, k=8):
        return list(self._vec)[:k]

    def keyword_search(self, q, k=8):
        return list(self._kw)[:k]

    def total_chunks(self):
        return self._n


def _stub_embed(text):
    return np.ones(4, dtype=np.float32)


def test_retrieve_hybrid_fusion():
    vec = [{"doc_name": "v.pdf", "page": 1, "text": "vector text", "score": 0.9}]
    kw = [{"doc_name": "k.pdf", "page": 2, "text": "keyword text", "bm25": -1.0}]
    hits, meta = retrieval.retrieve(_FakeStore(vec, kw), "query", k=5, embed_fn=_stub_embed)
    assert meta["vector"] and meta["keyword"]
    assert len(hits) == 2 and {h["text"] for h in hits} == {"vector text", "keyword text"}


def test_retrieve_embedding_failure_falls_back_to_keyword():
    kw = [{"doc_name": "k.pdf", "page": 2, "text": "keyword text", "bm25": -1.0}]

    def broken_embed(text):
        raise RuntimeError("embedding model not downloaded")

    hits, meta = retrieval.retrieve(_FakeStore([], kw), "query", k=5, embed_fn=broken_embed)
    assert meta["vector"] is False and meta["keyword"] is True
    assert meta["fallback"], "expected a fallback note"
    assert hits and hits[0]["text"] == "keyword text"


def test_retrieve_all_empty():
    hits, meta = retrieval.retrieve(_FakeStore([], []), "nothing here", k=5, embed_fn=_stub_embed)
    assert hits == [] and meta["vector"] is True and meta["keyword"] is True


def test_store_keyword_search_and_fts_query():
    with tempfile.TemporaryDirectory() as td:
        st = Store(os.path.join(td, "t.db"))
        rng = np.random.default_rng(3)
        vec = rng.normal(size=8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        st.add_doc("notes.txt", 50,
                   [(None, "The capital of France is Paris and the river is the Seine."),
                    (None, "Jarvis answers questions about your documents."),
                    (None, "The price of tea in China is hard to measure.")],
                   [vec, vec, vec])
        assert st.total_chunks() == 3

        hits = st.keyword_search("capital France Paris", k=5)
        assert hits and hits[0]["text"].startswith("The capital of France"), hits

        assert st.keyword_search("!!!" ) == []  # sanitised away
        assert st.keyword_search("a b") == []  # single-char tokens dropped

        # reindex keeps FTS in sync when a doc is deleted
        doc_id = st.list_docs()[0]["id"]
        st.delete_doc(doc_id)
        assert st.keyword_search("France") == []
        st.close()


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
