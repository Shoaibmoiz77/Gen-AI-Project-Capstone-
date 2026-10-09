from pathlib import Path

import pytest

from groundwork.chunking import load_corpus
from groundwork.embeddings import HashingEmbedder
from groundwork.llm import FakeLLM
from groundwork.pipeline import RAGPipeline
from groundwork.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(CORPUS)


@pytest.fixture(scope="session")
def retriever(chunks):
    return HybridRetriever(chunks, HashingEmbedder())


@pytest.fixture()
def pipeline(retriever):
    return RAGPipeline(retriever, FakeLLM(), top_k=6)
