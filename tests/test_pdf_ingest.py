"""PDF ingestion test with a minimal, hand-built one-page PDF (needs the optional `pypdf` package)."""
import pytest

pytest.importorskip("pypdf")
from src.ingest import chunks_from_text, extract_text  # noqa: E402


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_text_is_extracted_and_chunked(tmp_path):
    p = tmp_path / "policy.pdf"
    p.write_bytes(_minimal_pdf("International orders ship within ten business days."))
    text = extract_text(p)
    assert "International orders ship" in text
    chunks = chunks_from_text("policy.pdf", text, 400, 50)
    assert chunks and chunks[0].doc_id == "policy.pdf"
