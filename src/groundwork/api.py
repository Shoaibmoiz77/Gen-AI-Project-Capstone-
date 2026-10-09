"""HTTP API.

    POST /ask      {"question": "...", "k": 6}  -> cited answer + sources
    POST /search   {"query": "...", "k": 6, "mode": "hybrid"}  -> ranked chunks (no LLM)
    GET  /health
    GET  /         -> a small demo UI
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from groundwork.config import get_settings
from groundwork.llm import get_llm
from groundwork.pipeline import RAGPipeline
from groundwork.retrieval import MODES, HybridRetriever

UI_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=50)
    mode: str = "hybrid"


def _build_pipeline() -> RAGPipeline:
    s = get_settings()
    kind = os.environ.get("GROUNDWORK_LLM") or (
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "fake"
    )
    return RAGPipeline(
        HybridRetriever.load(s.index_dir),
        get_llm(kind, s.model, s.max_tokens),
        top_k=s.top_k,
        mode=s.retrieval_mode,
    )


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or _build_pipeline()
        yield

    app = FastAPI(
        title="Groundwork RAG",
        version="0.1.0",
        description="Grounded, cited answers over your documents.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return UI_HTML

    @app.get("/health")
    def health() -> dict:
        p: RAGPipeline = app.state.pipeline
        return {
            "status": "ok",
            "chunks": len(p.retriever.chunks),
            "embedder": p.retriever.embedder.name,
            "llm": p.llm.name,
        }

    # Sync handlers run in FastAPI's threadpool, so a slow model call never blocks the loop.
    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        p: RAGPipeline = app.state.pipeline
        try:
            return p.ask(req.question, k=req.k).to_dict()
        except Exception as e:  # surface upstream model errors as 502, not 500
            raise HTTPException(status_code=502, detail=f"Generation failed: {e}") from e

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        if req.mode not in MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
        p: RAGPipeline = app.state.pipeline
        hits = p.retriever.search(req.query, k=req.k, mode=req.mode)
        return {
            "results": [
                {"rank": h.rank, "score": round(h.score, 5), **h.chunk.to_dict()} for h in hits
            ]
        }

    return app


app = create_app()
