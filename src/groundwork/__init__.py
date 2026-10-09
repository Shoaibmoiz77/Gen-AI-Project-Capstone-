"""Groundwork: grounded, cited, measurable retrieval-augmented generation."""

from groundwork.chunking import Chunk, chunk_document
from groundwork.pipeline import AskResult, RAGPipeline

__all__ = ["Chunk", "chunk_document", "RAGPipeline", "AskResult"]
__version__ = "0.1.0"
