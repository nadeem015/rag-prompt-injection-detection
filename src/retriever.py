"""TF-IDF cosine-similarity retriever over chunks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    similarity: float


class TfidfRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = list(chunks)
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def retrieve(self, query: str, k: int = 3, min_similarity: float = 0.0) -> List[Retrieved]:
        if not self.chunks or not query.strip():
            return []
        q = self.vec.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()   # rows are L2-normalised => cosine
        order = np.argsort(-sims)[:k]
        return [Retrieved(self.chunks[i], float(sims[i])) for i in order if sims[i] > min_similarity]
