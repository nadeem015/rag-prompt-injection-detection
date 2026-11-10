import pytest
from src.ingest import chunk_text, chunks_from_text, load_directory, extract_text
from src.retriever import TfidfRetriever
from src.config import ROOT

def test_chunk_empty():
    assert chunk_text("") == [] and chunk_text(None) == []

def test_chunk_size_respected():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    ch = chunk_text(text, 300, 50)
    assert len(ch) > 3 and all(len(c) <= 400 for c in ch)

def test_chunk_long_unbroken():
    assert all(len(c) <= 100 for c in chunk_text("x" * 1000, 100, 10))

def test_chunk_bad_params():
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)

def test_load_directory_and_retrieve():
    chunks = load_directory(ROOT / "data/knowledge_base")
    assert len(chunks) >= 7
    hits = TfidfRetriever(chunks).retrieve("benefits of renewable energy solar", k=2)
    assert hits and hits[0].chunk.doc_id.startswith(("renewable", "POISONED_DEMO_renewable"))

def test_retrieve_empty_query():
    assert TfidfRetriever(chunks_from_text("a.txt", "Hello world.")).retrieve("  ") == []

def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.exe"; p.write_text("x")
    with pytest.raises(ValueError):
        extract_text(p)
