"""Independent re-computation of the paper's headline numbers.

Purpose: a second, deliberately different implementation (plain numpy / scipy.stats.binomtest /
sklearn.metrics) that does NOT import experiments/run_final_analysis.py, so that a bug in that
script cannot silently propagate.  It reads the *historical* per-example predictions
(results/final_experiment/all_test_predictions.csv, produced by the author's original run) and
the rule flags stored by run_final_analysis.py, rebuilds the hybrids by direct boolean algebra,
and compares every value with results/final/metrics/*.csv.

Usage:  python experiments/independent_verification.py        (exit code 1 on any mismatch)
Writes: results/final/metrics/independent_verification.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
HEAD = "agentic5k_test"
THR = {"word": 0.47, "word_char": 0.60}          # validation-selected (results/final_experiment/validation_summary.csv)
TOL = 1e-9

hist = pd.read_csv(ROOT / "results/final_experiment/all_test_predictions.csv")
new = pd.read_csv(ROOT / "results/final/predictions/all_predictions_with_rules_and_hybrids.csv")
assert len(hist) == len(new) and (hist.label.values == new.label.values).all() and (hist.dataset.values == new.dataset.values).all()
assert np.allclose(hist.word_tfidf_probability, new.word_tfidf_probability) and np.allclose(hist.word_char_tfidf_probability, new.word_char_tfidf_probability)

y = hist.label.values.astype(bool)
pw, pc = hist.word_tfidf_probability.values, hist.word_char_tfidf_probability.values
# Rule scores: the stored CSV rounds rule_score to 3 decimals (this changes ties in AUC/AP at ~1e-7),
# so re-score the stored texts with the frozen reconstructed rule module instead.
sys.path.insert(0, str(ROOT))
from src.rules import RuleDetector  # noqa: E402
_rd = RuleDetector()
_rr = [_rd.detect(t) for t in hist.text.astype(str)]
rule_score = np.array([r.score for r in _rr])
rule = np.array([r.flagged for r in _rr], dtype=bool)
assert (rule == new.rule_pred.values.astype(bool)).all(), "stored rule flags differ from re-scoring"
det = {
    "always_flag": np.ones(len(y), bool),
    "rule_only": rule,
    "word_tfidf": pw >= THR["word"],
    "word_char_tfidf": pc >= THR["word_char"],
    "hybrid_or_word": rule | (pw >= THR["word"]),
    "hybrid_or_word_char": rule | (pc >= THR["word_char"]),
    "hybrid_fusion_word": np.minimum(1, pw + 0.10 * rule) >= THR["word"],
    "hybrid_fusion_word_char": np.minimum(1, pc + 0.10 * rule) >= THR["word_char"],
}

def counts(yt, yp):
    return (int(((~yt) & (~yp)).sum()), int(((~yt) & yp).sum()), int((yt & ~yp).sum()), int((yt & yp).sum()))

def metrics(yt, yp):
    tn, fp, fn, tp = counts(yt, yp)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tp + tn) / len(yt), precision=P, recall=R,
                f1=2 * P * R / (P + R) if P + R else 0.0, fpr=fp / (fp + tn))

def mcnemar(yt, a, b):
    ca, cb = (a == yt), (b == yt)
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n10 + n01
    return n10, n01, (1.0 if n == 0 else min(1.0, binomtest(min(n10, n01), n, 0.5).pvalue * 1.0))

scopes = {s: (hist.dataset.values == s) for s in sorted(hist.dataset.unique())}
scopes["external_diagnostic_pooled"] = hist.dataset.values != HEAD

ref = pd.read_csv(ROOT / "results/final/metrics/all_detectors_all_scopes.csv")
rows, bad = [], 0
def check(name, mine, theirs):
    global bad
    ok = abs(float(mine) - float(theirs)) <= TOL
    bad += (not ok)
    rows.append(dict(check=name, independent=float(mine), stored=float(theirs), ok=ok))

for sc, m in scopes.items():
    for d, pred in det.items():
        r = ref[(ref.scope == sc) & (ref.detector == d)].iloc[0]
        mm = metrics(y[m], pred[m])
        for k, v in mm.items():
            check(f"{sc}|{d}|{k}", v, r[k])

auc = pd.read_csv(ROOT / "results/final/metrics/auc_ap.csv")
score_cols = {"word_tfidf": pw, "word_char_tfidf": pc, "rule_score": rule_score}
for sc, m in scopes.items():
    for name, s in score_cols.items():
        r = auc[(auc.scope == sc) & (auc.scorer == name)]
        if r.empty: continue
        r = r.iloc[0]
        check(f"{sc}|{name}|roc_auc", roc_auc_score(y[m], s[m]), r.roc_auc)
        check(f"{sc}|{name}|avg_precision", average_precision_score(y[m], s[m]), r.avg_precision)

pt = pd.read_csv(ROOT / "results/final/statistical_tests/paired_tests.csv")
for _, r in pt.iterrows():
    m = scopes[r.scope]
    n10, n01, p = mcnemar(y[m], det[r.A][m], det[r.B][m])
    check(f"{r.scope}|{r.A}->{r.B}|b", n10, r.A_right_B_wrong)
    check(f"{r.scope}|{r.A}->{r.B}|c", n01, r.A_wrong_B_right)
    check(f"{r.scope}|{r.A}->{r.B}|mcnemar_p", np.log10(max(p, 1e-300)), np.log10(max(r.mcnemar_exact_p, 1e-300)) )
    check(f"{r.scope}|{r.A}->{r.B}|dF1", metrics(y[m], det[r.B][m])["f1"] - metrics(y[m], det[r.A][m])["f1"], r.f1_diff_B_minus_A)

# Holm re-implementation (statsmodels-free): step-down, monotone
prim = pt[(pt.A == "word_tfidf") & (pt.B == "word_char_tfidf")].copy()
p = prim.mcnemar_exact_p.values; order = np.argsort(p); m_ = len(p); adj = np.empty(m_); run = 0.0
for rank, ix in enumerate(order):
    run = max(run, min(1.0, (m_ - rank) * p[ix])); adj[ix] = run
for a, b in zip(adj, prim.holm_adj_p_primary.values):
    check("holm_adj_p", np.log10(max(a, 1e-300)), np.log10(max(b, 1e-300)))

# always-flag closed form  F1 = 2*pi/(1+pi)
for sc, m in scopes.items():
    pi = y[m].mean()
    check(f"{sc}|always_flag|closed_form_f1", 2 * pi / (1 + pi), metrics(y[m], det['always_flag'][m])["f1"])

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/final/metrics/independent_verification.csv", index=False)
print(f"{len(out)} independent checks, {int(bad)} mismatches (tolerance {TOL})")
sys.exit(1 if bad else 0)
