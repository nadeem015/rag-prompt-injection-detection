"""Streamlit front end for the RAG prompt-injection screening prototype.

Run (PowerShell, from the project root):
    streamlit run app.py

IMPORTANT: the answer stage is EXTRACTIVE (it selects sentences). No generative LLM is called.
The detectors are research baselines, not a production security product.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PipelineConfig, ROOT, VALIDATION_SELECTED_THRESHOLDS
from src.detector import MODES, Detector
from src.events import EventLogger
from src.ingest import chunks_from_text, extract_text, load_directory
from src.pipeline import SecureRAGPipeline

st.set_page_config(page_title="RAG Prompt-Injection Screening", layout="wide")
KB_DIR = ROOT / "data" / "knowledge_base"
FINAL = ROOT / "results" / "final"

# ---------------------------------------------------------------- session state
if "uploaded" not in st.session_state:
    st.session_state.uploaded = {}   # name -> text
if "include_poisoned" not in st.session_state:
    st.session_state.include_poisoned = True


def build_chunks(include_poisoned: bool, chunk_chars: int, overlap: int):
    chunks = []
    for c in load_directory(KB_DIR, chunk_chars, overlap):
        if not include_poisoned and c.doc_id.startswith("POISONED_DEMO"):
            continue
        chunks.append(c)
    for name, text in st.session_state.uploaded.items():
        chunks.extend(chunks_from_text(name, text, chunk_chars, overlap))
    return chunks


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Detection mode", MODES, index=MODES.index("hybrid_fusion"))
ml_model = st.sidebar.selectbox("ML component", ["word_char_tfidf", "word_tfidf"])
default_thr = VALIDATION_SELECTED_THRESHOLDS[ml_model]
use_default = st.sidebar.checkbox(f"Use validation-selected threshold ({default_thr})", value=True)
thr = default_thr if use_default else st.sidebar.slider("ML threshold (manual override)", 0.05, 0.95, default_thr, 0.01)
top_k = st.sidebar.slider("Top-k retrieved chunks", 1, 6, 3)
screen_q = st.sidebar.checkbox("Screen the user query", True)
screen_c = st.sidebar.checkbox("Screen retrieved chunks", True)
st.session_state.include_poisoned = st.sidebar.checkbox(
    "Include synthetic POISONED_DEMO documents", st.session_state.include_poisoned)
st.sidebar.caption("Overriding the threshold departs from the validation-selected operating point "
                   "used in the paper; do not report such runs as the paper's results.")

cfg = PipelineConfig(detector_mode=mode, ml_model=ml_model, ml_threshold=thr, top_k=top_k,
                     screen_query=screen_q, screen_chunks=screen_c)

st.title("RAG prompt-injection screening prototype")
st.warning("Research prototype. Extractive answers only (no LLM). Detectors are lexical TF-IDF/rule baselines "
           "with substantial false-positive rates on ordinary prose; not a production security control.")

tab_dash, tab_kb, tab_query, tab_events, tab_eval = st.tabs(
    ["Dashboard", "Knowledge base", "Query", "Event history", "Evaluation"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    chunks = build_chunks(st.session_state.include_poisoned, cfg.chunk_chars, cfg.chunk_overlap)
    docs = sorted({c.doc_id for c in chunks})
    events = EventLogger(cfg.event_log).read(5000)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(docs))
    c2.metric("Chunks", len(chunks))
    c3.metric("Quarantined chunks (logged)", sum(e["event"] == "chunk_quarantined" for e in events))
    c4.metric("Blocked queries (logged)", sum(e["event"] == "query_blocked" for e in events))
    st.write("Active configuration")
    st.code(cfg.to_json(), language="json")

# ---------------------------------------------------------------- KB management
with tab_kb:
    st.subheader("Upload documents (txt, md, csv, pdf)")
    files = st.file_uploader("Add to knowledge base", accept_multiple_files=True,
                             type=["txt", "md", "csv", "pdf"])
    if files:
        import tempfile
        for f in files:
            if f.name not in st.session_state.uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.getvalue())
                try:
                    st.session_state.uploaded[f.name] = extract_text(Path(tmp.name))
                except Exception as exc:  # noqa: BLE001 - show any parser error to the user
                    st.error(f"Could not read {f.name}: {exc}")
    pasted_name = st.text_input("Or paste a document name")
    pasted = st.text_area("Pasted document text", height=120)
    if st.button("Add pasted document") and pasted_name and pasted.strip():
        st.session_state.uploaded[pasted_name] = pasted
    rows = [{"document": d, "origin": "uploaded" if d in st.session_state.uploaded else "data/knowledge_base",
             "chunks": sum(c.doc_id == d for c in chunks)} for d in docs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    rm = st.selectbox("Remove an uploaded document", ["-"] + list(st.session_state.uploaded))
    if rm != "-" and st.button("Remove"):
        st.session_state.uploaded.pop(rm, None)
        st.rerun()

# ---------------------------------------------------------------- query
with tab_query:
    q = st.text_input("Question", "What is the shipping policy for international orders?")
    if st.button("Run pipeline", type="primary"):
        try:
            pipe = SecureRAGPipeline(build_chunks(st.session_state.include_poisoned,
                                                  cfg.chunk_chars, cfg.chunk_overlap), cfg)
            res = pipe.run(q)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Query screening")
            st.json(res.query_detection or {"screening": "disabled"})
            if res.query_blocked:
                st.error(res.answer)
            else:
                st.subheader("Answer (extractive)")
                st.success(res.answer)
                st.subheader(f"Retrieved context ({len(res.retrieved)})")
                for it in res.retrieved:
                    flagged = it in res.quarantined
                    label = "QUARANTINED" if flagged else "allowed"
                    with st.expander(f"[{label}] {it['source']} (similarity {it['similarity']:.3f})",
                                     expanded=flagged):
                        st.write(it["text"])
                        if "detection" in it:
                            st.json(it["detection"])
                st.subheader("Filtered context sent to the prompt builder")
                st.write(f"{len(res.allowed)} chunk(s) passed; {len(res.quarantined)} quarantined.")
                with st.expander("Boundary-protected prompt (what a generative model would receive)"):
                    st.code(res.prompt)

# ---------------------------------------------------------------- events
with tab_events:
    ev = EventLogger(cfg.event_log).read(500)
    if ev:
        st.dataframe(pd.DataFrame(ev).iloc[::-1], use_container_width=True)
    else:
        st.info("No events logged yet.")

# ---------------------------------------------------------------- evaluation dashboard
with tab_eval:
    st.caption("Reads stored result files. Nothing is recomputed or retrained here.")
    for title, rel in [("All detectors, all scopes", "metrics/all_detectors_all_scopes.csv"),
                       ("ROC AUC / average precision", "metrics/auc_ap.csv"),
                       ("Paired statistical tests", "statistical_tests/paired_tests.csv"),
                       ("Context-screening simulation (summary)",
                        "metrics/context_screening_simulation_summary.csv")]:
        p = FINAL / rel
        st.subheader(title)
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True)
        else:
            st.info(f"Missing: {rel}")
    for fig in sorted((FINAL / "figures").glob("*.png")):
        st.image(str(fig), caption=fig.name)
    man = FINAL / "manifests" / "analysis_manifest.json"
    if man.exists():
        with st.expander("Analysis manifest"):
            st.json(json.loads(man.read_text()))
