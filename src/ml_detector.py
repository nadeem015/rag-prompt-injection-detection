"""Wrapper around the fitted TF-IDF + Logistic Regression pipelines."""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from .config import MODEL_DIR, VALIDATION_SELECTED_THRESHOLDS


@dataclass
class MLResult:
    probability: float
    flagged: bool
    threshold: float
    top_features: List[dict] = field(default_factory=list)


class MLDetector:
    """Loads word_tfidf or word_char_tfidf and scores text.

    ``explain=True`` returns the strongest positive (injection-pushing) features,
    computed as coefficient x TF-IDF value for the transformed sample.
    """

    def __init__(self, name: str = "word_char_tfidf", threshold: Optional[float] = None,
                 model_dir: Path | str = MODEL_DIR):
        self.name = name
        self.threshold = VALIDATION_SELECTED_THRESHOLDS[name] if threshold is None else float(threshold)
        path = Path(model_dir) / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pickles were written with scikit-learn 1.8.0
            self.pipeline = joblib.load(path)
        self._names = None

    def _feature_names(self):
        if self._names is None:
            feats = self.pipeline.named_steps["features"]
            self._names = np.asarray(feats.get_feature_names_out())
        return self._names

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def detect(self, text: str, explain: bool = True, k: int = 5) -> MLResult:
        p = float(self.predict_proba([text])[0])
        top: List[dict] = []
        if explain:
            X = self.pipeline.named_steps["features"].transform([text])
            coef = self.pipeline.named_steps["classifier"].coef_[0]
            row = X.tocsr()
            contrib = row.data * coef[row.indices]
            order = np.argsort(-contrib)[:k]
            names = self._feature_names()
            top = [{"feature": str(names[row.indices[i]]), "contribution": float(contrib[i])}
                   for i in order if contrib[i] > 0]
        return MLResult(probability=p, flagged=p >= self.threshold, threshold=self.threshold, top_features=top)
