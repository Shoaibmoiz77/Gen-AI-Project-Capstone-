import numpy as np
import pytest

from groundwork.bm25 import BM25
from groundwork.embeddings import HashingEmbedder
from groundwork.retrieval import HybridRetriever, reciprocal_rank_fusion


def test_bm25_ranks_relevant_doc_first():
    docs = ["the cat sat on the mat", "dogs chase cats in the park", "stock prices fell today"]
    scores = BM25(docs).scores("stock market prices")
    assert int(np.argmax(scores)) == 2
    assert scores[0] == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    # docs 1 and 2 are near the top of both lists; 3 and 4 appear only once
    assert fused[1] == pytest.approx(fused[2])
    assert fused[2] > fused[3] == pytest.approx(fused[4])
    assert fused[3] == pytest.approx(1 / 63)


def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder(dim=256)
    m = e.embed_documents(["hello world", "goodbye"])
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert np.allclose(e.embed_query("hello world"), m[0])


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes_find_the_right_document(retriever, mode):
    hits = retriever.search("How long does the Kestrel X2 battery last?", k=3, mode=mode)
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "kestrel-x2-faq"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_invalid_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = HybridRetriever.load(tmp_path)
    q = "minimum password length"
    assert [h.chunk.id for h in loaded.search(q)] == [h.chunk.id for h in retriever.search(q)]


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        HybridRetriever.load(tmp_path / "nope")
