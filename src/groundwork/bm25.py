"""Okapi BM25, implemented from scratch over a small in-memory corpus."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from groundwork.text import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_lens = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(documents)
        # BM25+ style idf floor keeps very common terms from going negative.
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(len(self.doc_tokens), dtype=float)
        if not q_terms or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens / self.avgdl)
        for term in set(q_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            f = np.array([tf.get(term, 0) for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out
