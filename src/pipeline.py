"""Secure RAG pipeline: query screening -> retrieval -> chunk screening -> boundary-safe prompt -> extractive answer.

The response stage is EXTRACTIVE (sentence selection).  No generative LLM is called;
``result['prompt']`` shows the boundary-protected prompt that a generative model WOULD receive.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import PipelineConfig
from .detector import Detector
from .events import EventLogger
from .ingest import Chunk
from .prompt_builder import build_prompt
from .retriever import TfidfRetriever

REFUSAL = "The request was blocked by the input screening layer because it resembles a prompt-injection attempt."
NO_ANSWER = "No safe, relevant context was available to answer this question."


@dataclass
class RunResult:
    query: str
    query_detection: dict
    query_blocked: bool
    retrieved: List[dict] = field(default_factory=list)
    allowed: List[dict] = field(default_factory=list)
    quarantined: List[dict] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""


class SecureRAGPipeline:
    def __init__(self, chunks: List[Chunk], config: Optional[PipelineConfig] = None,
                 detector: Optional[Detector] = None):
        self.cfg = config or PipelineConfig()
        self.detector = detector or Detector(self.cfg.detector_mode, self.cfg.ml_model,
                                             self.cfg.ml_threshold, self.cfg.rule_threshold)
        self.retriever = TfidfRetriever(chunks)
        self.log = EventLogger(self.cfg.event_log)

    # --------------------------------------------------------------------
    def run(self, query: str) -> RunResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        query = query.strip()[: self.cfg.max_query_chars]
        qd = self.detector.detect(query) if self.cfg.screen_query else None
        res = RunResult(query, qd.as_dict() if qd else {}, bool(qd and qd.flagged))
        if res.query_blocked:
            res.answer = REFUSAL
            self.log.log("query_blocked", mode=self.cfg.detector_mode, score=qd.score, why=qd.explanation(),
                         query=query[:200])
            return res

        for r in self.retriever.retrieve(query, self.cfg.top_k, self.cfg.min_similarity):
            item = {"source": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "text": r.chunk.text,
                    "similarity": r.similarity}
            res.retrieved.append(item)
            if self.cfg.screen_chunks:
                d = self.detector.detect(r.chunk.text)
                item["detection"] = d.as_dict()
                if d.flagged:
                    res.quarantined.append(item)
                    self.log.log("chunk_quarantined", mode=self.cfg.detector_mode, source=r.chunk.doc_id,
                                 chunk=r.chunk.chunk_id, score=d.score, why=d.explanation())
                    continue
            res.allowed.append(item)

        res.prompt = build_prompt(query, res.allowed)
        res.answer = self._extractive_answer(query, res.allowed)
        self.log.log("query_answered", mode=self.cfg.detector_mode, retrieved=len(res.retrieved),
                     quarantined=len(res.quarantined), allowed=len(res.allowed), query=query[:200])
        return res

    @staticmethod
    def _extractive_answer(query: str, contexts: List[dict], n: int = 2) -> str:
        if not contexts:
            return NO_ANSWER
        qwords = set(re.findall(r"[a-z]{3,}", query.lower()))
        sents = [(s.strip(), c["source"]) for c in contexts
                 for s in re.split(r"(?<=[.!?])\s+", c["text"]) if s.strip()]
        scored = sorted(sents, key=lambda sc: -len(qwords & set(re.findall(r"[a-z]{3,}", sc[0].lower()))))
        top = [f"{s} [{src}]" for s, src in scored[:n]]
        return " ".join(top) if top else NO_ANSWER
