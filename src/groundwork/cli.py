"""Command line interface.

    groundwork ingest  --src data/corpus
    groundwork ask     "How many PTO days do I get?"
    groundwork eval    --llm anthropic          # full eval with Claude + LLM judge
    groundwork eval    --llm fake               # offline: retrieval metrics + extractive baseline
    groundwork serve   --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from groundwork.chunking import load_corpus
from groundwork.config import get_settings
from groundwork.embeddings import get_embedder
from groundwork.llm import get_llm
from groundwork.pipeline import RAGPipeline
from groundwork.retrieval import MODES, HybridRetriever


def _pipeline(args, settings) -> RAGPipeline:
    retriever = HybridRetriever.load(args.index)
    llm = get_llm(args.llm, settings.model, settings.max_tokens)
    return RAGPipeline(retriever, llm, top_k=args.k, mode=args.mode)


def _default_llm() -> str:
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "fake"


def cmd_ingest(args, settings) -> None:
    chunks = load_corpus(args.src, settings.chunk_chars)
    retriever = HybridRetriever(chunks, get_embedder(args.embedder))
    retriever.save(args.index)
    n_docs = len({c.doc_id for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {n_docs} documents into {args.index}/")


def cmd_ask(args, settings) -> None:
    result = _pipeline(args, settings).ask(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(result.answer.render())
    for s in result.sources():
        print(f"  [{s['n']}] {s['title']} > {s['section']}  ({s['chunk_id']})")


def cmd_eval(args, settings) -> None:
    from groundwork.evals import run_eval

    pipeline = _pipeline(args, settings)
    judge = None
    if not args.no_judge:
        judge = get_llm(args.llm, settings.judge_model, 512)
    result = run_eval(pipeline, args.golden, args.out, judge=judge, workers=args.workers)
    s = result["summary"]
    print(json.dumps({"retrieval": s["retrieval"], "generation": s["generation"]}, indent=2))
    print(f"\nReport written to {args.out}/report.md")

    if args.min_recall is not None:
        recall = s["retrieval"][f"recall@{args.k}"]
        if recall < args.min_recall:
            print(f"FAIL: recall@{args.k} {recall:.3f} < {args.min_recall}", file=sys.stderr)
            sys.exit(1)


def cmd_serve(args, settings) -> None:
    import uvicorn

    os.environ["GROUNDWORK_INDEX_DIR"] = args.index
    os.environ.setdefault("GROUNDWORK_LLM", args.llm)
    uvicorn.run("groundwork.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(prog="groundwork", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_llm: bool = True):
        sp.add_argument("--index", default=settings.index_dir)
        sp.add_argument("--k", type=int, default=settings.top_k)
        sp.add_argument("--mode", choices=MODES, default=settings.retrieval_mode)
        if needs_llm:
            sp.add_argument("--llm", choices=["anthropic", "fake"], default=_default_llm())

    sp = sub.add_parser("ingest", help="Chunk and index a folder of .md/.txt files")
    sp.add_argument("--src", default="data/corpus")
    sp.add_argument("--index", default=settings.index_dir)
    sp.add_argument("--embedder", default=settings.embedder)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("ask", help="Ask a question")
    sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="Run the evaluation suite")
    sp.add_argument("--golden", default="data/eval/golden.jsonl")
    sp.add_argument("--out", default="reports/latest")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--no-judge", action="store_true")
    sp.add_argument("--min-recall", type=float, default=None,
                    help="Exit non-zero if recall@k falls below this (for CI)")
    common(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("serve", help="Run the HTTP API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    common(sp)
    sp.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args, settings)


if __name__ == "__main__":
    main()
