"""Pure metric functions. No I/O, no model calls, easy to unit test."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _dedupe(seq: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for s in seq:
        seen.setdefault(s, None)
    return list(seen)


def recall_at_k(retrieved_docs: Sequence[str], gold_docs: Sequence[str], k: int) -> float:
    """Fraction of gold documents that appear in the top-k retrieved chunks."""
    if not gold_docs:
        return float("nan")
    top = set(retrieved_docs[:k])
    return sum(1 for g in set(gold_docs) if g in top) / len(set(gold_docs))


def hit_at_k(retrieved_docs: Sequence[str], gold_docs: Sequence[str], k: int) -> float:
    if not gold_docs:
        return float("nan")
    return 1.0 if set(retrieved_docs[:k]) & set(gold_docs) else 0.0


def reciprocal_rank(retrieved_docs: Sequence[str], gold_docs: Sequence[str]) -> float:
    """1 / rank of the first chunk from any gold document (0 if none retrieved)."""
    if not gold_docs:
        return float("nan")
    gold = set(gold_docs)
    for rank, d in enumerate(_dedupe(retrieved_docs), start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def citation_precision(cited_docs: Sequence[str], gold_docs: Sequence[str]) -> float:
    """Fraction of cited chunks that come from a gold document."""
    if not cited_docs or not gold_docs:
        return float("nan")
    gold = set(gold_docs)
    return sum(1 for d in cited_docs if d in gold) / len(cited_docs)


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: Sequence[float], p: float) -> float:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return float("nan")
    idx = (len(vals) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)
