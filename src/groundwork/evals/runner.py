"""Run the golden set through the pipeline and write a report.

Outputs (in ``out_dir``):
  results.json   - aggregate metrics, machine readable (CI can gate on these)
  cases.jsonl    - one line per question with retrieval, answer, citations and grades
  report.md      - a human-readable summary, including the retrieval ablation table
"""

from __future__ import annotations

import json
import math
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from groundwork.evals import metrics as M
from groundwork.evals.judge import grade_answer
from groundwork.llm import LLM
from groundwork.pipeline import RAGPipeline
from groundwork.retrieval import MODES

ABLATION_KS = (1, 3, 6)


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    reference_answer: str
    gold_docs: list[str]
    answerable: bool
    type: str = "lookup"


def load_golden(path: str | Path) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(EvalCase(**json.loads(line)))
    return cases


# --------------------------------------------------------------------------------------------
# Retrieval ablation: no LLM calls, so it's cheap to run on every commit
# --------------------------------------------------------------------------------------------
def retrieval_ablation(pipeline: RAGPipeline, cases: list[EvalCase]) -> dict:
    answerable = [c for c in cases if c.answerable]
    out: dict[str, dict] = {}
    depth = max(ABLATION_KS)
    for mode in MODES:
        rows = []
        for c in answerable:
            hits = pipeline.retriever.search(c.question, k=depth, mode=mode)
            docs = [h.chunk.doc_id for h in hits]
            row = {"mrr": M.reciprocal_rank(docs, c.gold_docs)}
            for k in ABLATION_KS:
                row[f"recall@{k}"] = M.recall_at_k(docs, c.gold_docs, k)
            rows.append(row)
        out[mode] = {key: M.mean([r[key] for r in rows]) for key in rows[0]}
    return out


# --------------------------------------------------------------------------------------------
# Full end-to-end evaluation
# --------------------------------------------------------------------------------------------
def _run_case(pipeline: RAGPipeline, judge: LLM | None, case: EvalCase) -> dict:
    res = pipeline.ask(case.question)
    retrieved_docs = [sc.chunk.doc_id for sc in res.retrieved]
    by_id = {sc.chunk.id: sc.chunk for sc in res.retrieved}
    cited_docs = [by_id[cid].doc_id for cid in res.answer.cited_chunk_ids]
    k = pipeline.top_k

    row: dict = {
        "id": case.id,
        "type": case.type,
        "question": case.question,
        "answerable": case.answerable,
        "model_answered": res.answer.answerable,
        "answer": res.answer.render(),
        "reference_answer": case.reference_answer,
        "gold_docs": case.gold_docs,
        "retrieved": [sc.chunk.id for sc in res.retrieved],
        "cited": res.answer.cited_chunk_ids,
        "recall@k": M.recall_at_k(retrieved_docs, case.gold_docs, k),
        "mrr": M.reciprocal_rank(retrieved_docs, case.gold_docs),
        "citation_precision": M.citation_precision(cited_docs, case.gold_docs),
        "unsupported_sentences": sum(1 for s in res.answer.sentences if not s.supported),
        "n_sentences": len(res.answer.sentences),
        "dropped_citations": res.answer.dropped_citations,
        "latency_ms": res.total_ms,
        "input_tokens": res.answer.usage.input_tokens,
        "output_tokens": res.answer.usage.output_tokens,
        "faithfulness": None,
        "correctness": None,
        "judge_reasoning": None,
    }

    if judge is not None and case.answerable and res.answer.answerable:
        try:
            g = grade_answer(
            judge,
            case.question,
            case.reference_answer,
                res.answer.render(),
                [by_id[cid].text for cid in res.answer.cited_chunk_ids],
            )
        except Exception as e:  # a failed grade shouldn't sink the whole run
            row["error"] = f"judge: {e}"[:500]
        else:
            row.update(
                faithfulness=g.faithfulness,
                correctness=g.correctness,
                judge_reasoning=g.reasoning,
                judge_input_tokens=g.usage.input_tokens,
                judge_output_tokens=g.usage.output_tokens,
            )
    return row


def _run_case_safe(pipeline: RAGPipeline, judge: LLM | None, case: EvalCase) -> dict:
    """Run one case; if generation fails, record the error and keep going."""
    try:
        return _run_case(pipeline, judge, case)
    except Exception as e:
        retrieved = pipeline.retrieve(case.question)
        docs = [sc.chunk.doc_id for sc in retrieved]
        return {
            "id": case.id,
            "type": case.type,
            "question": case.question,
            "answerable": case.answerable,
            "model_answered": False,
            "answer": "",
            "reference_answer": case.reference_answer,
            "gold_docs": case.gold_docs,
            "retrieved": [sc.chunk.id for sc in retrieved],
            "cited": [],
            "recall@k": M.recall_at_k(docs, case.gold_docs, pipeline.top_k),
            "mrr": M.reciprocal_rank(docs, case.gold_docs),
            "citation_precision": float("nan"),
            "unsupported_sentences": 0,
            "n_sentences": 0,
            "dropped_citations": 0,
            "latency_ms": float("nan"),
            "input_tokens": 0,
            "output_tokens": 0,
            "faithfulness": None,
            "correctness": None,
            "judge_reasoning": None,
            "error": f"generation: {e}"[:500],
        }


def summarize(rows: list[dict], k: int) -> dict:
    errors = [r for r in rows if str(r.get("error", "")).startswith("generation")]
    ok = [r for r in rows if r not in errors]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    all_ans = [r for r in rows if r["answerable"]]
    graded = [r for r in ans if r["faithfulness"] is not None]
    total_sentences = sum(r["n_sentences"] for r in rows)
    return {
        "n_cases": len(rows),
        "n_answerable": len(all_ans),
        "n_unanswerable": sum(1 for r in rows if not r["answerable"]),
        "n_errors": len(errors),
        "n_judge_errors": sum(1 for r in rows if str(r.get("error", "")).startswith("judge")),
        "retrieval": {
            f"recall@{k}": M.mean([r["recall@k"] for r in all_ans]),
            "mrr": M.mean([r["mrr"] for r in all_ans]),
        },
        "generation": {
            "answer_rate": M.mean([1.0 if r["model_answered"] else 0.0 for r in ans]),
            "abstention_accuracy": M.mean(
                [0.0 if r["model_answered"] else 1.0 for r in unans]
            ),
            "citation_precision": M.mean([r["citation_precision"] for r in ans]),
            "unsupported_sentence_rate": (
                sum(r["unsupported_sentences"] for r in rows) / total_sentences
                if total_sentences
                else float("nan")
            ),
            "faithfulness_mean": M.mean([r["faithfulness"] for r in graded]),
            "correctness_mean": M.mean([r["correctness"] for r in graded]),
            "correct_rate": M.mean([1.0 if r["correctness"] >= 4 else 0.0 for r in graded]),
            "n_graded": len(graded),
        },
        "cost": {
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "judge_input_tokens": sum(r.get("judge_input_tokens", 0) for r in rows),
            "judge_output_tokens": sum(r.get("judge_output_tokens", 0) for r in rows),
        },
        "latency_ms": {
            "p50": M.percentile([r["latency_ms"] for r in rows], 50),
            "p95": M.percentile([r["latency_ms"] for r in rows], 95),
        },
        "by_type": {
            t: {
                "n": len(sub),
                "correctness_mean": M.mean([r["correctness"] for r in sub if r["correctness"]]),
                f"recall@{k}": M.mean([r["recall@k"] for r in sub]),
            }
            for t in sorted({r["type"] for r in rows})
            for sub in [[r for r in rows if r["type"] == t]]
        },
    }


def _fmt(v: object, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.2f}"
    return str(v)


def render_report(summary: dict, ablation: dict, rows: list[dict], meta: dict) -> str:
    k = meta["top_k"]
    g, r = summary["generation"], summary["retrieval"]
    lines = [
        "# Evaluation Report",
        "",
        f"- **Run:** {meta['timestamp']}",
        f"- **Generator:** `{meta['generator']}` · **Judge:** `{meta['judge']}`",
        f"- **Embedder:** `{meta['embedder']}` · **Retrieval:** `{meta['mode']}` · **k:** {k}",
        f"- **Cases:** {summary['n_cases']} ({summary['n_answerable']} answerable, "
        f"{summary['n_unanswerable']} unanswerable)",
        f"- **Errors:** {summary['n_errors']} generation, {summary['n_judge_errors']} judge "
        "(errored generations are excluded from generation metrics)",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval recall@{k} | {_fmt(r[f'recall@{k}'], True)} |",
        f"| Retrieval MRR | {_fmt(r['mrr'])} |",
        f"| Answer rate (answerable questions) | {_fmt(g['answer_rate'], True)} |",
        f"| Abstention accuracy (unanswerable questions) | "
        f"{_fmt(g['abstention_accuracy'], True)} |",
        f"| Citation precision | {_fmt(g['citation_precision'], True)} |",
        f"| Unsupported sentence rate | {_fmt(g['unsupported_sentence_rate'], True)} |",
        f"| Judge faithfulness (1-5) | {_fmt(g['faithfulness_mean'])} |",
        f"| Judge correctness (1-5) | {_fmt(g['correctness_mean'])} |",
        f"| Correct (score >= 4) | {_fmt(g['correct_rate'], True)} |",
        f"| Latency p50 / p95 | {_fmt(summary['latency_ms']['p50'])} ms / "
        f"{_fmt(summary['latency_ms']['p95'])} ms |",
        f"| Tokens in / out (generation) | {summary['cost']['input_tokens']:,} / "
        f"{summary['cost']['output_tokens']:,} |",
        "",
        "## Retrieval ablation",
        "",
        "Same questions, three retrieval strategies. No LLM involved.",
        "",
        "| Mode | " + " | ".join(f"Recall@{kk}" for kk in ABLATION_KS) + " | MRR |",
        "|---|" + "---|" * (len(ABLATION_KS) + 1),
    ]
    for mode, vals in ablation.items():
        cells = " | ".join(_fmt(vals[f"recall@{kk}"], True) for kk in ABLATION_KS)
        lines.append(f"| {mode} | {cells} | {_fmt(vals['mrr'])} |")

    lines += [
        "",
        "## By question type",
        "",
        f"| Type | N | Recall@{k} | Correctness |",
        "|---|---|---|---|",
    ]
    for t, v in summary["by_type"].items():
        lines.append(
            f"| {t} | {v['n']} | {_fmt(v[f'recall@{k}'], True)} | "
            f"{_fmt(v['correctness_mean'])} |"
        )

    errored = [x for x in rows if x.get("error")]
    if errored:
        lines += ["", f"## Errors ({len(errored)})", ""]
        for x in errored:
            lines.append(f"- **{x['id']}**: {x['error'][:200]}")
    failures = [
        x
        for x in rows
        if not str(x.get("error", "")).startswith("generation")
        and (
            (x["answerable"] and (not x["model_answered"] or (x["correctness"] or 5) < 4))
            or (not x["answerable"] and x["model_answered"])
        )
    ]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for x in failures:
        if not x["answerable"]:
            why = "answered an unanswerable question"
        elif not x["model_answered"]:
            why = "abstained on an answerable question"
        else:
            why = f"correctness {x['correctness']}/5"
        lines += [
            f"**{x['id']}** ({x['type']}): {x['question']}  ",
            f"*Why it failed:* {why}  ",
            f"*Answer:* {x['answer']}  ",
            f"*Reference:* {x['reference_answer'] or '(should abstain)'}",
            "",
        ]
    return "\n".join(lines) + "\n"


def run_eval(
    pipeline: RAGPipeline,
    golden_path: str | Path,
    out_dir: str | Path,
    judge: LLM | None = None,
    workers: int = 4,
    embedder_name: str = "",
) -> dict:
    cases = load_golden(golden_path)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda c: _run_case_safe(pipeline, judge, c), cases))

    summary = summarize(rows, pipeline.top_k)
    ablation = retrieval_ablation(pipeline, cases)
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "generator": pipeline.llm.name,
        "judge": judge.name if judge else "none",
        "embedder": embedder_name or pipeline.retriever.embedder.name,
        "mode": pipeline.mode,
        "top_k": pipeline.top_k,
        "python": platform.python_version(),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _clean(o):  # JSON has no NaN
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    result = {"meta": meta, "summary": summary, "retrieval_ablation": ablation}
    (out / "results.json").write_text(json.dumps(_clean(result), indent=2))
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_clean(r)) + "\n")
    (out / "report.md").write_text(render_report(summary, ablation, rows, meta))
    return result
