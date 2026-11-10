"""Post-hoc analysis of the FINAL experiment from its stored predictions (no retraining).

Inputs  : results/final_experiment/*  (historical, never modified)
Outputs : results/final/{metrics,predictions,curves,statistical_tests,ablation,manifests}/

Everything here is computed from stored per-example probabilities and texts.
Rule/hybrid detectors use the FROZEN rules (src/rules.py) and fixed a-priori fusion
constants; nothing is tuned on any test data.
"""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from src.rules import RuleDetector            # noqa: E402
from src.detector import fuse                 # noqa: E402

SRC = ROOT / "results" / "final_experiment"
OUT = ROOT / "results" / "final"
for d in ("metrics", "predictions", "curves", "statistical_tests", "ablation", "manifests"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

THR = {"word_tfidf": 0.47, "word_char_tfidf": 0.60}
MODELS = list(THR)
SEED, B = 42, 1000
HEADLINE = "agentic5k_test"


def counts(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    return int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum()), \
           int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())


def metrics_from_counts(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    sd = lambda a, b: a / b if b else 0.0
    prec, rec = sd(tp, tp + fp), sd(tp, tp + fn)
    return dict(n=n, accuracy=sd(tp + tn, n), precision=prec, recall=rec,
                f1=sd(2 * prec * rec, prec + rec), fpr=sd(fp, fp + tn),
                tn=tn, fp=fp, fn=fn, tp=tp)


def f1_of(y, p):
    return metrics_from_counts(*counts(y, p))["f1"]


def boot_ci(y, p, fn=f1_of, b=B, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(b))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_exact(y, pa, pb):
    """Exact two-sided McNemar test on discordant pairs."""
    ca, cb = (np.asarray(pa) == np.asarray(y)), (np.asarray(pb) == np.asarray(y))
    b01 = int((ca & ~cb).sum()); b10 = int((~ca & cb).sum())   # A right/B wrong ; A wrong/B right
    n = b01 + b10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5)))
    return b01, b10, p


# ----------------------------------------------------------------------------------------------
df = pd.read_csv(SRC / "all_test_predictions.csv")
df["row_id"] = np.arange(len(df))
rd = RuleDetector()
rr = [rd.detect(t) for t in df.text]
df["rule_score"] = [r.score for r in rr]
df["rule_pred"] = [int(r.flagged) for r in rr]
df["rule_categories"] = [";".join(r.categories) for r in rr]
for m in MODELS:
    p, t = df[f"{m}_probability"].values, THR[m]
    df[f"{m}_pred_recomputed"] = (p >= t).astype(int)
    df[f"hybrid_or_{m}_pred"] = [int(fuse("hybrid_or", bool(r), pp, t)) for r, pp in zip(df.rule_pred, p)]
    df[f"hybrid_fusion_{m}_pred"] = [int(fuse("hybrid_fusion", bool(r), pp, t)) for r, pp in zip(df.rule_pred, p)]

DETECTORS = {
    "rule_only": "rule_pred",
    "word_tfidf": "word_tfidf_prediction",
    "word_char_tfidf": "word_char_tfidf_prediction",
    "hybrid_or_word": "hybrid_or_word_tfidf_pred",
    "hybrid_or_word_char": "hybrid_or_word_char_tfidf_pred",
    "hybrid_fusion_word": "hybrid_fusion_word_tfidf_pred",
    "hybrid_fusion_word_char": "hybrid_fusion_word_char_tfidf_pred",
}
df.drop(columns=["text"]).to_csv(OUT / "predictions" / "all_predictions_with_rules_and_hybrids.csv", index=False)
(df[df.dataset == HEADLINE][["row_id", "text", "label", "technique", "attack_family"] +
                           [f"{m}_probability" for m in MODELS] + ["rule_score", "rule_categories"] +
                           [c for c in DETECTORS.values()]]
 ).to_csv(OUT / "predictions" / "headline_test_predictions_with_text.csv", index=False)

datasets = [HEADLINE] + sorted(d for d in df.dataset.unique() if d != HEADLINE)
pool = df[df.dataset != HEADLINE]

# ---- 1. VERIFICATION of original numbers ----------------------------------------------------
orig_sum = pd.read_csv(SRC / "final_summary_metrics.csv")
orig_ds = pd.read_csv(SRC / "per_dataset_metrics.csv")
ver = []
for m in MODELS:
    assert (df[f"{m}_pred_recomputed"] == df[f"{m}_prediction"]).all(), "stored prediction != prob>=thr"
    for scope, sub in [("agentic5k_test", df[df.dataset == HEADLINE]), ("external_diagnostic_pooled", pool)]:
        mine = metrics_from_counts(*counts(sub.label, sub[f"{m}_prediction"]))
        o = orig_sum[(orig_sum.dataset == scope) & (orig_sum.model == m)].iloc[0]
        for k in ("accuracy", "precision", "recall", "f1", "fpr", "tn", "fp", "fn", "tp"):
            ver.append(dict(scope=scope, model=m, metric=k, original=float(o[k]), recomputed=float(mine[k]),
                            abs_diff=abs(float(o[k]) - float(mine[k]))))
    for ds in datasets:
        sub = df[df.dataset == ds]
        mine = metrics_from_counts(*counts(sub.label, sub[f"{m}_prediction"]))
        o = orig_ds[(orig_ds.dataset == ds) & (orig_ds.model == m)]
        if len(o):
            o = o.iloc[0]
            for k in ("accuracy", "precision", "recall", "f1", "fpr"):
                ver.append(dict(scope=ds, model=m, metric=k, original=float(o[k]), recomputed=float(mine[k]),
                                abs_diff=abs(float(o[k]) - float(mine[k]))))
ver = pd.DataFrame(ver)
ver.to_csv(OUT / "metrics" / "verification_original_vs_recomputed.csv", index=False)
print("verification: rows", len(ver), "max abs diff", ver.abs_diff.max())

# ---- 2. Metrics for all detectors, all scopes ------------------------------------------------
rows = []
scopes = [(ds, df[df.dataset == ds]) for ds in datasets] + [("external_diagnostic_pooled", pool)]
for scope, sub in scopes:
    prev = float(sub.label.mean())
    af = metrics_from_counts(*counts(sub.label, np.ones(len(sub))))
    rows.append(dict(scope=scope, detector="always_flag", **af, prevalence=prev, f1_ci_low=np.nan, f1_ci_high=np.nan))
    for name, col in DETECTORS.items():
        mt = metrics_from_counts(*counts(sub.label, sub[col]))
        lo, hi = boot_ci(sub.label.values, sub[col].values)
        rows.append(dict(scope=scope, detector=name, **mt, prevalence=prev, f1_ci_low=lo, f1_ci_high=hi))
allm = pd.DataFrame(rows)
allm.to_csv(OUT / "metrics" / "all_detectors_all_scopes.csv", index=False)

# ---- 3. AUC / AP ------------------------------------------------------------------------------
auc = []
for scope, sub in scopes:
    for m in MODELS + ["rule_score"]:
        col = f"{m}_probability" if m in MODELS else m
        auc.append(dict(scope=scope, scorer=m, roc_auc=roc_auc_score(sub.label, sub[col]),
                        avg_precision=average_precision_score(sub.label, sub[col]), prevalence=sub.label.mean()))
auc = pd.DataFrame(auc)
# bootstrap CI for headline AUC
sub = df[df.dataset == HEADLINE]; rng = np.random.default_rng(SEED)
for i, m in enumerate(MODELS):
    v = []
    for _ in range(B):
        ix = rng.integers(0, len(sub), len(sub)); yy = sub.label.values[ix]
        if yy.min() != yy.max():
            v.append(roc_auc_score(yy, sub[f"{m}_probability"].values[ix]))
    auc.loc[(auc.scope == HEADLINE) & (auc.scorer == m), ["auc_ci_low", "auc_ci_high"]] = np.percentile(v, [2.5, 97.5])
auc.to_csv(OUT / "metrics" / "auc_ap.csv", index=False)

# operating-point-free diagnostic: recall at FPR<=10% *on the evaluated set itself* (post-hoc, NOT a selected threshold)
oracle = []
for scope, sub in scopes:
    for m in MODELS:
        fpr, tpr, thr = roc_curve(sub.label, sub[f"{m}_probability"])
        ok = fpr <= 0.10
        oracle.append(dict(scope=scope, model=m, posthoc_recall_at_fpr10=float(tpr[ok].max()),
                           note="POST-HOC on the evaluated set; diagnostic only, not a deployable threshold"))
pd.DataFrame(oracle).to_csv(OUT / "ablation" / "posthoc_recall_at_fpr10.csv", index=False)

# curve data for figures
curves = {}
for scope in (HEADLINE, "external_diagnostic_pooled"):
    sub = df[df.dataset == HEADLINE] if scope == HEADLINE else pool
    for m in MODELS + ["rule_score"]:
        col = f"{m}_probability" if m in MODELS else m
        f, t, _ = roc_curve(sub.label, sub[col]); pr, rc, _ = precision_recall_curve(sub.label, sub[col])
        curves[f"{scope}|{m}"] = dict(fpr=f.tolist(), tpr=t.tolist(), precision=pr.tolist(), recall=rc.tolist())
json.dump(curves, open(OUT / "curves" / "curve_data.json", "w"))

# ---- 4. Statistical tests ---------------------------------------------------------------------
st = []
for scope, sub in scopes:
    for a, b in [("word_tfidf", "word_char_tfidf"), ("word_char_tfidf", "hybrid_or_word_char"),
                 ("word_char_tfidf", "hybrid_fusion_word_char"), ("word_tfidf", "rule_only")]:
        ca, cb = DETECTORS[a], DETECTORS[b]
        b01, b10, p = mcnemar_exact(sub.label, sub[ca], sub[cb])
        # paired bootstrap of F1 difference (b - a)
        rng = np.random.default_rng(SEED); y = sub.label.values; pa, pb = sub[ca].values, sub[cb].values
        diffs = []
        for _ in range(B):
            ix = rng.integers(0, len(y), len(y)); diffs.append(f1_of(y[ix], pb[ix]) - f1_of(y[ix], pa[ix]))
        st.append(dict(scope=scope, A=a, B=b, A_right_B_wrong=b01, A_wrong_B_right=b10, mcnemar_exact_p=p,
                       f1_diff_B_minus_A=f1_of(y, pb) - f1_of(y, pa),
                       diff_ci_low=float(np.percentile(diffs, 2.5)), diff_ci_high=float(np.percentile(diffs, 97.5))))
st = pd.DataFrame(st)
# Holm correction across the primary comparison (word vs word+char) over the 8 scopes
prim = st[(st.A == "word_tfidf") & (st.B == "word_char_tfidf")].sort_values("mcnemar_exact_p")
k = len(prim); holm = {}; run = 0.0
for i, (ix, r) in enumerate(prim.iterrows()):
    run = max(run, min(1.0, (k - i) * r.mcnemar_exact_p)); holm[ix] = run
st["holm_adj_p_primary"] = pd.Series(holm)
st.to_csv(OUT / "statistical_tests" / "paired_tests.csv", index=False)

# ---- 5. Error breakdowns -------------------------------------------------------------------------
h = df[df.dataset == HEADLINE]
rows = []
for (cls, tech), g in h.groupby(["class", "technique"]):
    r = dict(cls=cls, technique=tech, n=len(g))
    for name, col in DETECTORS.items():
        r[f"flagged_{name}"] = int(g[col].sum())
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "ablation" / "headline_by_technique.csv", index=False)

lang_rows = []
for ds in ["synthetic_direct_test", "synthetic_docs_test", "synthetic_v2_test"]:
    g0 = df[df.dataset == ds]
    for tag, g in [("english", g0[g0.lang == "en"]), ("non_english", g0[g0.lang != "en"])]:
        for name, col in DETECTORS.items():
            mt = metrics_from_counts(*counts(g.label, g[col]))
            lang_rows.append(dict(dataset=ds, subset=tag, detector=name, **mt, prevalence=g.label.mean()))
pd.DataFrame(lang_rows).to_csv(OUT / "ablation" / "language_breakdown.csv", index=False)

kind_rows = []
for kind, g in pool.groupby("kind"):
    for name, col in DETECTORS.items():
        kind_rows.append(dict(kind=kind, detector=name, **metrics_from_counts(*counts(g.label, g[col])),
                              prevalence=g.label.mean()))
pd.DataFrame(kind_rows).to_csv(OUT / "ablation" / "kind_breakdown.csv", index=False)

# rule-category hits by class on headline (explainability)
cat = []
for c in sorted({x for s in h.rule_categories for x in s.split(";") if x}):
    has = h.rule_categories.str.contains(c)
    cat.append(dict(category=c, hits_injection=int((has & (h.label == 1)).sum()), hits_benign=int((has & (h.label == 0)).sum())))
pd.DataFrame(cat).to_csv(OUT / "ablation" / "rule_categories_headline.csv", index=False)

# ---- 6. Feature inspection ----------------------------------------------------------------------
import joblib
feat = []
for m in MODELS:
    pipe = joblib.load(SRC / f"{m}.joblib")
    names = np.asarray(pipe.named_steps["features"].get_feature_names_out())
    coef = pipe.named_steps["classifier"].coef_[0]
    o = np.argsort(coef)
    for rank, i in enumerate(o[::-1][:20]): feat.append(dict(model=m, direction="injection", rank=rank + 1, feature=names[i], coef=float(coef[i])))
    for rank, i in enumerate(o[:20]): feat.append(dict(model=m, direction="benign", rank=rank + 1, feature=names[i], coef=float(coef[i])))
pd.DataFrame(feat).to_csv(OUT / "ablation" / "top_features.csv", index=False)

# ---- 7. Manifest ------------------------------------------------------------------------------------
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
json.dump({
    "script": "experiments/run_final_analysis.py",
    "kind": "post-hoc analysis of stored predictions; NO retraining; NO new data downloaded",
    "inputs_sha256": {p.name: sha(p) for p in sorted(SRC.glob("*")) if p.is_file()},
    "frozen_rules_sha256": sha(ROOT / "src" / "rules.py"),
    "rule_threshold": 0.50, "hybrid_or": "rule OR p>=thr", "hybrid_fusion": "min(1,p+0.10*rule)>=thr",
    "ml_thresholds": THR, "bootstrap_replicates": B, "bootstrap_seed": SEED,
    "verification_max_abs_diff": float(ver.abs_diff.max()),
    "numpy": np.__version__, "pandas": pd.__version__,
}, open(OUT / "manifests" / "analysis_manifest.json", "w"), indent=2)
print("done")
