"""Additional metrics requested for security reporting: macro-F1, weighted-F1, false-negative rate.

Deterministic post-hoc computation from the stored per-example predictions
(results/final/predictions/all_predictions_with_rules_and_hybrids.csv + the stored ML probabilities).
Output: results/final/metrics/extra_metrics.csv.   No randomness, no retraining.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HEAD = "agentic5k_test"
hist = pd.read_csv(ROOT / "results/final_experiment/all_test_predictions.csv")
new = pd.read_csv(ROOT / "results/final/predictions/all_predictions_with_rules_and_hybrids.csv")
y = hist.label.values.astype(bool)
pw, pc = hist.word_tfidf_probability.values, hist.word_char_tfidf_probability.values
rule = new.rule_pred.values.astype(bool)
det = {"always_flag": np.ones(len(y), bool), "rule_only": rule, "word_tfidf": pw >= 0.47, "word_char_tfidf": pc >= 0.60,
       "hybrid_or_word_char": rule | (pc >= 0.60), "hybrid_fusion_word_char": np.minimum(1, pc + 0.10 * rule) >= 0.60}

def f1_class(yt, yp, pos):
    tp = ((yt == pos) & (yp == pos)).sum(); fp = ((yt != pos) & (yp == pos)).sum(); fn = ((yt == pos) & (yp != pos)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

rows = []
scopes = {HEAD: hist.dataset.values == HEAD, "external_diagnostic_pooled": hist.dataset.values != HEAD}
for sc, m in scopes.items():
    for d, p in det.items():
        yt, yp = y[m], p[m]
        f1p, f1n = f1_class(yt, yp, True), f1_class(yt, yp, False)
        npos, nneg = yt.sum(), (~yt).sum()
        rows.append(dict(scope=sc, detector=d, n=int(m.sum()), macro_f1=(f1p + f1n) / 2,
                         weighted_f1=(f1p * npos + f1n * nneg) / (npos + nneg),
                         fnr=((yt & ~yp).sum() / npos), fpr=((~yt & yp).sum() / nneg),
                         f1_injection=f1p, f1_benign=f1n))
out = pd.DataFrame(rows); out.to_csv(ROOT / "results/final/metrics/extra_metrics.csv", index=False)
print(out.round(4).to_string(index=False))
