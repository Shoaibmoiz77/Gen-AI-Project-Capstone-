"""The end-to-end pipeline: retrieve, then generate a grounded answer."""

from __future__ import annotations

import time
from dataclasses import dataclass

from groundwork.chunking import Chunk, load_corpus
from groundwork.embeddings import get_embedder
from groundwork.generation import Answer, generate_answer
from groundwork.llm import LLM
from groundwork.retrieval import HybridRetriever, ScoredChunk


@dataclass
class AskResult:
    question: str
    answer: Answer
    retrieved: list[ScoredChunk]
    retrieval_ms: float
    total_ms: float

    def sources(self) -> list[dict]:
        """The chunks the answer actually cited, in citation order."""
        by_id = {sc.chunk.id: sc.chunk for sc in self.retrieved}
        out = []
        for n, cid in enumerate(self.answer.cited_chunk_ids, start=1):
            c: Chunk = by_id[cid]
            out.append(
                {"n": n, "chunk_id": cid, "doc_id": c.doc_id, "title": c.title,
                 "section": c.section, "text": c.text}
            )
        return out

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer.render(),
            "answerable": self.answer.answerable,
            "sentences": [
                {"text": s.text, "chunk_ids": s.chunk_ids} for s in self.answer.sentences
            ],
            "sources": self.sources(),
            "retrieved": [
                {"chunk_id": sc.chunk.id, "doc_id": sc.chunk.doc_id, "score": round(sc.score, 5)}
                for sc in self.retrieved
            ],
            "timing_ms": {
                "retrieval": round(self.retrieval_ms, 1),
                "total": round(self.total_ms, 1),
            },
            "usage": {
                "input_tokens": self.answer.usage.input_tokens,
                "output_tokens": self.answer.usage.output_tokens,
            },
        }


class RAGPipeline:
    def __init__(
        self, retriever: HybridRetriever, llm: LLM, top_k: int = 6, mode: str = "hybrid"
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.mode = mode

    @classmethod
    def from_corpus(
        cls, src: str, llm: LLM, embedder: str = "hashing", chunk_chars: int = 900, **kw
    ) -> RAGPipeline:
        chunks = load_corpus(src, chunk_chars)
        return cls(HybridRetriever(chunks, get_embedder(embedder)), llm, **kw)

    def retrieve(self, question: str, k: int | None = None) -> list[ScoredChunk]:
        return self.retriever.search(question, k=k or self.top_k, mode=self.mode)

    def ask(self, question: str, k: int | None = None) -> AskResult:
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, k)
        t1 = time.perf_counter()
        answer = generate_answer(self.llm, question, retrieved)
        t2 = time.perf_counter()
        return AskResult(
            question=question,
            answer=answer,
            retrieved=retrieved,
            retrieval_ms=(t1 - t0) * 1000,
            total_ms=(t2 - t0) * 1000,
        )
