"""RECONSTRUCTED training script for the final experiment.  >>> NOT EXECUTED in this package <<<

The author's original run_final_experiment.py was not available.  This file re-creates the
training/threshold-selection procedure from (i) the settings read out of the stored joblib
pipelines, (ii) final_experiment_manifest.json and (iii) the column layout of
all_test_predictions.csv.  It needs Hugging Face access, which was blocked where this package was
prepared, so it has NEVER been run here.  Do not treat its output as the paper's results; the paper's
numbers come from the stored historical predictions in results/final_experiment/.

Run (PowerShell, from project root, with internet access):
    python experiments/run_final_experiment.py --out results/reproduction_run

It writes to a NEW directory and refuses to overwrite results/final_experiment/.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

DATASET = "3nesdeniz/agentic-prompt-injection-5k"   # NOTE: card not verified; see docs/reference_verification_report.md
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)   # 91 thresholds (matches validation_threshold_sweep.csv)
MAX_FPR = 0.10


def make_model(kind: str) -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50_000, min_df=2,
                           sublinear_tf=True, lowercase=True, strip_accents="unicode")
    if kind == "word_tfidf":
        feats = word
    elif kind == "word_char_tfidf":
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60_000, min_df=2,
                               sublinear_tf=True, strip_accents="unicode")
        feats = FeatureUnion([("word", word), ("char", char)])
    else:
        raise ValueError(kind)
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)
    return Pipeline([("features", feats), ("clf", clf)])


def select_threshold(y, p) -> float:
    """max F1 s.t. FPR <= MAX_FPR; ties prefer recall, then the LOWER threshold."""
    best = None
    for t in THRESHOLD_GRID:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        fpr = fp / max(fp + tn, 1)
        if fpr > MAX_FPR:
            continue
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        key = (round(f1, 12), round(rec, 12), -t)
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.resolve().name == "final_experiment":
        raise SystemExit("Refusing to write into the historical results directory.")
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset          # requires `pip install datasets` and network access
    ds = load_dataset(DATASET)
    # Column names ('text','label') are assumed from all_test_predictions.csv; verify against the card.
    tr, va, te = ds["train"], ds["validation"], ds["test"]
    summary = {}
    for kind in ("word_tfidf", "word_char_tfidf"):
        m = make_model(kind).fit(tr["text"], np.array(tr["label"]))
        thr = select_threshold(np.array(va["label"]), m.predict_proba(va["text"])[:, 1])
        refit = make_model(kind).fit(list(tr["text"]) + list(va["text"]),
                                     np.array(list(tr["label"]) + list(va["label"])))
        p = refit.predict_proba(te["text"])[:, 1]
        np.save(out / f"{kind}_test_probabilities.npy", p)
        summary[kind] = {"threshold": thr}
    (out / "reproduction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
