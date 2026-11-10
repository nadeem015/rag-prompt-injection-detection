"""Inference-cost benchmark of the screening layer (single CPU process, no GPU).

Measures wall-clock latency per text for the 545 headline test texts: reconstructed rule detector, word TF-IDF + LR,
word+char TF-IDF + LR, and the full Detector in hybrid_fusion mode (rules + word+char model + explanation off).
Protocol: models are loaded once; one warm-up pass; then 5 timed passes over all texts, one text per call
(time.perf_counter_ns); we report the median and 95th percentile over all 5 x 545 calls, plus batch throughput
(one predict_proba call on all 545 texts, median of 5), model file sizes and load time.
Timings depend on the machine and are NOT bit-reproducible; the environment is recorded in the output.
Output: results/final/metrics/latency_benchmark.csv and results/final/manifests/latency_environment.json
"""
import json, platform, statistics, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.rules import RuleDetector
from src.ml_detector import MLDetector
from src.detector import Detector

df = pd.read_csv(ROOT / "results/final_experiment/all_test_predictions.csv")
texts = df[df.dataset == "agentic5k_test"].text.astype(str).tolist()
assert len(texts) == 545
lens = np.array([len(t) for t in texts])

t0 = time.perf_counter(); rd = RuleDetector(); load_rule = time.perf_counter() - t0
t0 = time.perf_counter(); mw = MLDetector("word_tfidf"); load_w = time.perf_counter() - t0
t0 = time.perf_counter(); mc = MLDetector("word_char_tfidf"); load_c = time.perf_counter() - t0
t0 = time.perf_counter(); det = Detector(mode="hybrid_fusion"); load_d = time.perf_counter() - t0

calls = {
    "rules (reconstructed)": lambda t: rd.detect(t),
    "word TF-IDF + LR": lambda t: mw.detect(t, explain=False),
    "word+char TF-IDF + LR": lambda t: mc.detect(t, explain=False),
    "Hybrid-fusion (rules + word+char)": lambda t: det.detect(t, explain=False),
}
sizes = {"rules (reconstructed)": 0, "word TF-IDF + LR": (ROOT / "results/final_experiment/word_tfidf.joblib").stat().st_size,
         "word+char TF-IDF + LR": (ROOT / "results/final_experiment/word_char_tfidf.joblib").stat().st_size}
sizes["Hybrid-fusion (rules + word+char)"] = sizes["word+char TF-IDF + LR"]
loads = {"rules (reconstructed)": load_rule, "word TF-IDF + LR": load_w, "word+char TF-IDF + LR": load_c, "Hybrid-fusion (rules + word+char)": load_d}
batch_fn = {"word TF-IDF + LR": lambda: mw.predict_proba(texts), "word+char TF-IDF + LR": lambda: mc.predict_proba(texts)}

rows = []
for name, fn in calls.items():
    for t in texts: fn(t)                                    # warm-up
    lat = []
    for _ in range(5):
        for t in texts:
            a = time.perf_counter_ns(); fn(t); lat.append((time.perf_counter_ns() - a) / 1e6)
    lat = np.array(lat)
    row = dict(detector=name, calls=len(lat), median_ms=float(np.median(lat)), p95_ms=float(np.percentile(lat, 95)), max_ms=float(lat.max()),
               mean_ms=float(lat.mean()), model_file_kib=sizes[name] / 1024, load_time_s=loads[name])
    if name in batch_fn:
        bt = []
        for _ in range(5):
            a = time.perf_counter(); batch_fn[name](); bt.append(time.perf_counter() - a)
        row["batch545_median_s"] = statistics.median(bt); row["batch_texts_per_s"] = 545 / statistics.median(bt)
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(ROOT / "results/final/metrics/latency_benchmark.csv", index=False)
import sklearn
env = dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(), cpu_count=__import__("os").cpu_count(),
           sklearn=sklearn.__version__, numpy=np.__version__, mean_text_chars=float(lens.mean()), median_text_chars=float(np.median(lens)),
           note="shared cloud container, CPU only, single process; not the author's hardware; timings vary run to run")
try:
    env["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception: pass
(ROOT / "results/final/manifests/latency_environment.json").write_text(json.dumps(env, indent=1))
print(out.round(3).to_string(index=False)); print(env)
