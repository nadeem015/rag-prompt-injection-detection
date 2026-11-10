"""Central configuration (no secrets, no external services)."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "final_experiment"

# Thresholds selected on the VALIDATION split only (see FINAL experiment manifest).
VALIDATION_SELECTED_THRESHOLDS = {"word_tfidf": 0.47, "word_char_tfidf": 0.60}
# A-priori hybrid constants carried over from the project's earlier phases
# (README Phase 3 / Phase 5).  They were NOT tuned on any final-experiment data.
FUSION_RULE_BONUS = 0.10


@dataclass
class PipelineConfig:
    detector_mode: str = "hybrid_fusion"   # rule | word_tfidf | word_char_tfidf | hybrid_or | hybrid_fusion
    top_k: int = 3
    chunk_chars: int = 600
    chunk_overlap: int = 100
    min_similarity: float = 0.05
    ml_model: str = "word_char_tfidf"      # ML component used by hybrid modes
    ml_threshold: float | None = None      # None -> validation-selected value
    rule_threshold: float = 0.50
    screen_query: bool = True
    screen_chunks: bool = True
    event_log: str = str(ROOT / "results" / "security_events.jsonl")
    max_query_chars: int = 2000

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
