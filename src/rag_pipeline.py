"""RAG retrieval + screening pipeline (compatibility alias of :mod:`src.pipeline`).

`RAGPipeline` is the same class as :class:`src.pipeline.SecureRAGPipeline`: TF-IDF retrieval, query and chunk
screening, quarantine, boundary-protected prompt assembly and an EXTRACTIVE answer (no generative model).
"""
from .pipeline import RunResult, SecureRAGPipeline

RAGPipeline = SecureRAGPipeline

__all__ = ["RAGPipeline", "RunResult", "SecureRAGPipeline"]
