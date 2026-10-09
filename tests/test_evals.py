import json
import math
from pathlib import Path

import pytest

from groundwork.evals import load_golden, run_eval
from groundwork.evals import metrics as M
from groundwork.llm import FakeLLM

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden.jsonl"


def test_recall_and_mrr():
    retrieved = ["a", "b", "a", "c"]
    assert M.recall_at_k(retrieved, ["a", "c"], 2) == 0.5
    assert M.recall_at_k(retrieved, ["a", "c"], 4) == 1.0
    assert M.reciprocal_rank(retrieved, ["c"]) == pytest.approx(1 / 3)  # deduped ranks
    assert M.reciprocal_rank(retrieved, ["z"]) == 0.0
    assert math.isnan(M.recall_at_k(retrieved, [], 3))


def test_citation_precision_and_percentile():
    assert M.citation_precision(["a", "b"], ["a"]) == 0.5
    assert M.percentile([1, 2, 3, 4, 5], 50) == 3
    assert M.mean([1.0, float("nan"), 3.0]) == 2.0


def test_golden_set_is_well_formed():
    cases = load_golden(GOLDEN)
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert c.answerable == bool(c.gold_docs)
        assert c.answerable == bool(c.reference_answer)


def test_full_eval_offline(pipeline, tmp_path):
    result = run_eval(pipeline, GOLDEN, tmp_path, judge=FakeLLM(), workers=2)
    s = result["summary"]
    assert s["retrieval"]["recall@6"] >= 0.9
    assert set(result["retrieval_ablation"]) == {"bm25", "dense", "hybrid"}
    assert (tmp_path / "report.md").read_text().startswith("# Evaluation Report")
    rows = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert len(rows) == s["n_cases"]
    json.loads((tmp_path / "results.json").read_text())  # valid JSON, no NaN
