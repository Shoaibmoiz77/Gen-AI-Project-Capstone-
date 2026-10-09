"""Hybrid retrieval: BM25 and dense search, merged with Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) scores each document as sum(1 / (k + rank)) across the ranked
lists it appears in. It needs no score calibration between retrievers, which is exactly the
problem with naively adding a BM25 score (unbounded) to a cosine similarity (in [-1, 1]).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from groundwork.bm25 import BM25
from groundwork.chunking import Chunk
from groundwork.embeddings import Embedder, get_embedder

MODES = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int  # 1-based


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        embeddings: np.ndarray | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks")
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.search_text for c in chunks]
        self.bm25 = BM25(texts)
        self.embeddings = (
            embeddings if embeddings is not None else embedder.embed_documents(texts)
        )

    # -- individual retrievers -------------------------------------------------------------
    def _bm25_ranking(self, query: str, depth: int) -> list[int]:
        scores = self.bm25.scores(query)
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order[:depth] if scores[i] > 0]

    def _dense_ranking(self, query: str, depth: int) -> list[int]:
        q = self.embedder.embed_query(query)
        sims = self.embeddings @ q
        order = np.argsort(-sims, kind="stable")
        return [int(i) for i in order[:depth]]

    # -- public API --------------------------------------------------------------------------
    def search(self, query: str, k: int = 6, mode: str = "hybrid") -> list[ScoredChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        depth = max(k * 4, 20)
        if mode == "bm25":
            ranking = self._bm25_ranking(query, depth)
            scored = [(i, 1.0 / (r + 1)) for r, i in enumerate(ranking)]
        elif mode == "dense":
            ranking = self._dense_ranking(query, depth)
            scored = [(i, 1.0 / (r + 1)) for r, i in enumerate(ranking)]
        else:
            fused = reciprocal_rank_fusion(
                [self._bm25_ranking(query, depth), self._dense_ranking(query, depth)]
            )
            scored = sorted(fused.items(), key=lambda kv: -kv[1])
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(s), rank=r)
            for r, (i, s) in enumerate(scored[:k], start=1)
        ]

    # -- persistence -------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        np.save(d / "embeddings.npy", self.embeddings)
        (d / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)}, indent=2)
        )

    @classmethod
    def load(cls, directory: str | Path) -> HybridRetriever:
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No index at {d}. Run `groundwork ingest` first.")
        meta = json.loads((d / "meta.json").read_text())
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        embeddings = np.load(d / "embeddings.npy")
        return cls(chunks, get_embedder(meta["embedder"]), embeddings=embeddings)
