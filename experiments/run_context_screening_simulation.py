"""Context-screening simulation (NEW synthetic experiment, not part of the original final experiment).

Question: when an injection payload is embedded inside otherwise benign retrieved text
(realistic RAG chunk), does the screening layer keep the payload out of the prompt?

Construction (seeded):  composite chunk = benign carrier paragraph + payload  (payload 'suffix')
                                       or payload + benign carrier             (payload 'prefix')
  * payloads : 280 injection texts and 265 benign texts (hard negatives etc.) from the
               HEADLINE test split (stored in results/final_experiment/all_test_predictions.csv)
  * carriers : paragraphs from this project's own demo knowledge base (author-written, benign)
  * seeds    : 42, 123, 456 (carrier assignment)
Metrics   : payload-exposure rate (injection composite NOT quarantined, i.e. would reach the prompt),
            false-quarantine rate on benign composites and on carriers alone.
IMPORTANT : "exposure" is a PROXY.  No generative model is run, so this does NOT measure
            attack success rate, exfiltration or harmful actions.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
from src.config import MODEL_DIR                      # noqa
from src.detector import fuse                         # noqa
from src.ml_detector import MLDetector                # noqa
from src.rules import RuleDetector                    # noqa

OUT = ROOT / "results" / "final" / "metrics"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456]
df = pd.read_csv(MODEL_DIR / "all_test_predictions.csv")
h = df[df.dataset == "agentic5k_test"].reset_index(drop=True)
carriers = []
for p in sorted((ROOT / "data" / "knowledge_base").glob("*.txt")):
    if p.name.startswith("POISONED"):
        continue
    carriers += [c.strip() for c in p.read_text(encoding="utf-8").split("\n\n") if c.strip()]
assert len(carriers) >= 8, len(carriers)

rules = RuleDetector()
ml = {n: MLDetector(n, explain=False) if False else MLDetector(n) for n in ("word_tfidf", "word_char_tfidf")}

def score_all(texts):
    out = {"rule": np.array([rules.detect(t).flagged for t in texts], dtype=int)}
    rs = out["rule"].astype(bool)
    for n, m in ml.items():
        p = m.predict_proba(texts); thr = m.threshold
        out[n] = (p >= thr).astype(int)
        out[f"hybrid_or_{n}"] = np.array([fuse("hybrid_or", r, pp, thr) for r, pp in zip(rs, p)], dtype=int)
        out[f"hybrid_fusion_{n}"] = np.array([fuse("hybrid_fusion", r, pp, thr) for r, pp in zip(rs, p)], dtype=int)
    return out

rows, pred_rows = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    ci = rng.integers(0, len(carriers), len(h))
    for pos in ("suffix", "prefix"):
        texts = [(carriers[c] + "\n\n" + t) if pos == "suffix" else (t + "\n\n" + carriers[c]) for c, t in zip(ci, h.text)]
        sc = score_all(texts)
        for det, pred in sc.items():
            inj, ben = h.label.values == 1, h.label.values == 0
            rows.append(dict(seed=seed, position=pos, detector=det,
                             injection_exposure_rate=1 - pred[inj].mean(),
                             benign_false_quarantine_rate=pred[ben].mean(),
                             n_inj=int(inj.sum()), n_benign=int(ben.sum())))
    sc = score_all(carriers)
    for det, pred in sc.items():
        rows.append(dict(seed=seed, position="carrier_only", detector=det, injection_exposure_rate=np.nan,
                         benign_false_quarantine_rate=pred.mean(), n_inj=0, n_benign=len(carriers)))
res = pd.DataFrame(rows)
res.to_csv(OUT / "context_screening_simulation_per_seed.csv", index=False)
agg = (res.groupby(["position", "detector"]).agg(
        exposure_mean=("injection_exposure_rate", "mean"), exposure_sd=("injection_exposure_rate", "std"),
        false_quarantine_mean=("benign_false_quarantine_rate", "mean"),
        false_quarantine_sd=("benign_false_quarantine_rate", "std"),
        n_inj=("n_inj", "first"), n_benign=("n_benign", "first")).reset_index())
agg.to_csv(OUT / "context_screening_simulation_summary.csv", index=False)
json.dump({"seeds": SEEDS, "n_carriers": len(carriers), "payload_source": "agentic5k_test (stored)",
           "no_defence_exposure": 1.0, "caveat": "proxy; no LLM executed"},
          open(ROOT / "results" / "final" / "manifests" / "context_simulation_manifest.json", "w"), indent=2)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(agg.to_string())
