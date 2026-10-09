"""Dense embedders behind one small interface.

* ``hashing``   - zero-dependency, offline. Signed feature hashing of word unigrams, bigrams
                  and character trigrams. Not semantic, but it catches morphology and typos
                  that BM25 misses, which makes it a useful second signal for fusion and lets
                  the whole project run (and CI pass) without network access.
* ``voyage``    - Voyage AI embeddings (Anthropic's recommended embedding provider).
                  Needs ``VOYAGE_API_KEY``.
* ``fastembed`` - local ONNX models such as BAAI/bge-small-en-v1.5. ``pip install .[fastembed]``.

All embedders return L2-normalised float32 rows, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from groundwork.text import tokenize


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"<{w}>"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f in self._features(text):
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = 0.5 if f.startswith("c:") else 1.0
            v[idx] += sign * weight
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._embed(text)[None, :])[0]


class VoyageEmbedder:
    name = "voyage"
    url = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, model: str = "voyage-3.5", batch_size: int = 64) -> None:
        import httpx

        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})

    def _call(self, texts: list[str], input_type: str) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = self._client.post(
                self.url, json={"input": batch, "model": self.model, "input_type": input_type}
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
        return _normalize(np.array(rows))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._call(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "query")[0]


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the extra: pip install '.[fastembed]'") from e
        self._model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self._model.embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.stack(list(self._model.query_embed(text))))[0]


def get_embedder(name: str) -> Embedder:
    name = name.lower()
    if name == "hashing":
        return HashingEmbedder()
    if name == "voyage":
        return VoyageEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"Unknown embedder {name!r}; choose hashing, voyage or fastembed")
