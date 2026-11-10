"""Document loader (compatibility alias).

The implementation lives in :mod:`src.ingest`; this module only exposes the same functionality under the
name `document_loader` so that the project layout matches the conventional RAG module list.
It does NOT replicate the author's original (unavailable) loader API.
"""
from .ingest import Chunk, chunk_text, chunks_from_text, extract_text, load_directory

load_documents = load_directory

__all__ = ["Chunk", "chunk_text", "chunks_from_text", "extract_text", "load_directory", "load_documents"]
