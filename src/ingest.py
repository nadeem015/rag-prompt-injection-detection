"""Document ingestion, text extraction and chunking."""
from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

SUPPORTED = {".txt", ".md", ".csv"}


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def extract_text(path: Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            return "\n".join(" ".join(r) for r in csv.reader(f))
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PDF support needs `pip install pypdf`") from e
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, size: int = 600, overlap: int = 100) -> List[str]:
    """Paragraph-/sentence-aware sliding chunks of at most ~``size`` characters."""
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    units = [u.strip() for p in re.split(r"\n\s*\n", text) for u in re.split(r"(?<=[.!?])\s+", p) if u.strip()]
    chunks, cur = [], ""
    for u in units:
        while len(u) > size:                       # hard-split very long units
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(u[:size]); u = u[size - overlap:]
        if len(cur) + len(u) + 1 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] + " " + u if overlap else u
        else:
            cur = (cur + " " + u).strip()
    if cur:
        chunks.append(cur)
    return chunks


def load_directory(directory: Path | str, size: int = 600, overlap: int = 100) -> List[Chunk]:
    directory = Path(directory)
    out: List[Chunk] = []
    for p in sorted(directory.glob("*")):
        if p.suffix.lower() in SUPPORTED:
            out.extend(chunks_from_text(p.name, extract_text(p), size, overlap))
    return out


def chunks_from_text(doc_id: str, text: str, size: int = 600, overlap: int = 100) -> List[Chunk]:
    return [Chunk(doc_id, i, c) for i, c in enumerate(chunk_text(text, size, overlap))]
