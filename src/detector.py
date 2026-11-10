"""Unified detection interface: rule / ML / hybrid modes with fixed a-priori fusion rules."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .config import FUSION_RULE_BONUS
from .ml_detector import MLDetector
from .rules import RuleDetector

MODES = ("rule", "word_tfidf", "word_char_tfidf", "hybrid_or", "hybrid_fusion")


@dataclass
class Detection:
    mode: str
    flagged: bool
    score: float
    rule_score: Optional[float] = None
    rule_matches: List[dict] = field(default_factory=list)
    ml_probability: Optional[float] = None
    ml_threshold: Optional[float] = None
    ml_top_features: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def explanation(self) -> str:
        parts = []
        if self.rule_matches:
            parts.append("rules: " + ", ".join(f"{m['category']}/{m['indicator']}" for m in self.rule_matches))
        if self.ml_probability is not None:
            parts.append(f"ML p={self.ml_probability:.3f} (thr {self.ml_threshold:.2f})")
        if self.ml_top_features:
            parts.append("top features: " + ", ".join(f["feature"] for f in self.ml_top_features[:3]))
        return "; ".join(parts) or "no indicators"


def fuse(mode: str, rule_flag: bool, p: float, thr: float) -> bool:
    """Decision rule for hybrid modes (pure function, unit-tested).

    hybrid_or     : flag if the rule detector flags OR p >= thr.
    hybrid_fusion : flag if p + 0.10*[rule flagged] (capped at 1) >= thr.
    """
    if mode == "hybrid_or":
        return bool(rule_flag or p >= thr)
    if mode == "hybrid_fusion":
        return bool(min(1.0, p + (FUSION_RULE_BONUS if rule_flag else 0.0)) >= thr)
    raise ValueError(mode)


class Detector:
    def __init__(self, mode: str = "hybrid_fusion", ml_model: str = "word_char_tfidf",
                 ml_threshold: Optional[float] = None, rule_threshold: float = 0.5):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.rules = RuleDetector(rule_threshold)
        name = mode if mode in ("word_tfidf", "word_char_tfidf") else ml_model
        self.ml = None if mode == "rule" else MLDetector(name, ml_threshold)

    def detect(self, text: str, explain: bool = True) -> Detection:
        text = text if isinstance(text, str) else ""
        r = self.rules.detect(text)
        if self.mode == "rule":
            return Detection("rule", r.flagged, r.score, r.score, r.matches)
        m = self.ml.detect(text, explain=explain)
        if self.mode in ("word_tfidf", "word_char_tfidf"):
            return Detection(self.mode, m.flagged, m.probability, r.score, r.matches,
                             m.probability, m.threshold, m.top_features)
        flagged = fuse(self.mode, r.flagged, m.probability, m.threshold)
        score = m.probability if self.mode == "hybrid_or" else min(1.0, m.probability + (0.10 if r.flagged else 0.0))
        return Detection(self.mode, flagged, max(score, r.score) if self.mode == "hybrid_or" else score,
                         r.score, r.matches, m.probability, m.threshold, m.top_features)
