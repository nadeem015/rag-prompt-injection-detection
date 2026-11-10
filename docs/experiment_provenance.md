# Experiment provenance (round 2)

Every number that was **added after the author's original final experiment** (rows 1-9 in round 2, rows 10-11 in round 3) is listed here with its exact script, input data, output file and random seed. All of them are *post-hoc analyses of the author's stored predictions* or *new synthetic experiments*; **none retrains a model**.

**Important.** The rule detector is a **reconstruction** written for this study (`src/rules.py`) because the author's original rule module was not available. All rows labelled *rules*, *Hybrid-OR* and *Hybrid-fusion* therefore come from the reconstructed detector and must not be presented as results of the author's original implementation. The author's earlier rule-detector results (15-sample, 46-row and 30-case sets in `results/*.csv`) were produced on different, tiny data and are not comparable.

| # | Experiment | Source script | Input data | Output files | Seed | Reproduction status |
|---|---|---|---|---|---|---|
| 1 | Reconstructed rule detector (noisy-OR, threshold 0.50) | src/rules.py (SHA-256 b7574694…83677); applied by experiments/run_final_analysis.py | results/final_experiment/all_test_predictions.csv (texts, labels) | results/final/predictions/all_predictions_with_rules_and_hybrids.csv; metrics/all_detectors_all_scopes.csv (detector=rule_only); ablation/rule_categories_headline.csv | none (deterministic) | Exact (clean-copy re-run identical; independent_verification.py 768/768) |
| 2 | Hybrid-OR (rule OR p>=tau) | src/detector.py:fuse + run_final_analysis.py | same + stored word/word+char probabilities; tau 0.47 / 0.60 | metrics/all_detectors_all_scopes.csv (hybrid_or_*) | none | Exact; also rebuilt by direct boolean algebra in independent_verification.py |
| 3 | Hybrid-fusion (min(1,p+0.10*rule)>=tau) | same | same | metrics/all_detectors_all_scopes.csv (hybrid_fusion_*) | none | Exact; independent rebuild |
| 4 | ROC / PR (AUC, AP, curves, AUC bootstrap CI) | run_final_analysis.py (sklearn roc_auc_score, average_precision_score, roc_curve, precision_recall_curve) | stored probabilities; rule score re-computed | metrics/auc_ap.csv; curves/curve_data.json; figures/fig3_roc_pr.* | 42 (AUC CI: 1000 resamples, headline split only) | Exact; independent recomputation of AUC/AP (rule score re-scored from text) |
| 5 | McNemar exact tests | run_final_analysis.py:mcnemar_exact (binomial); Holm over 8 scopes | predictions above | statistical_tests/paired_tests.csv | none (exact test) | Exact; independent scipy.stats.binomtest |
| 6 | Paired bootstrap of delta-F1 and F1 CIs | run_final_analysis.py (boot_ci, paired loop) | predictions above | statistical_tests/paired_tests.csv (diff_ci_*); metrics/all_detectors_all_scopes.csv (f1_ci_*) | 42, 1000 resamples (rng re-seeded per comparison) | Exact re-run; NOT independently recomputed (resampling code is the only copy) |
| 7 | Context-screening simulation (545 payloads x 10 carriers x prefix/suffix) | experiments/run_context_screening_simulation.py | results/final_experiment/all_test_predictions.csv (agentic5k_test texts); data/knowledge_base/*.txt (non-POISONED) | metrics/context_screening_simulation_per_seed.csv, ..._summary.csv; manifests/context_simulation_manifest.json; figures/fig6_* | 42, 123, 456 (carrier assignment) | Exact re-run; models load with a sklearn version warning (1.8.0 pickles in 1.9.1) |
| 8 | Always-flag baseline | run_final_analysis.py (all-ones prediction); closed form 2pi/(1+pi) | labels | metrics/all_detectors_all_scopes.csv (detector=always_flag) | none | Exact; closed form checked for all 8 scopes |
| 9 | Hard-negative / technique analysis | run_final_analysis.py lines 205-213 (groupby class, technique) | headline split predictions | ablation/headline_by_technique.csv; ablation/kind_breakdown.csv | none | Exact re-run; counts (212, 75, 164, 1) asserted by paper/verify_paper_numbers.py |
| 10 | FNR, macro-F1, weighted-F1 (added in round 3) | experiments/run_extra_metrics.py | results/final_experiment/all_test_predictions.csv; results/final/predictions/all_predictions_with_rules_and_hybrids.csv | results/final/metrics/extra_metrics.csv | none | Exact; recomputed from confusion counts by paper/verify_paper_numbers.py |
| 11 | Inference latency / model size (added in round 3) | experiments/run_latency_benchmark.py | headline-split texts; stored joblib models | results/final/metrics/latency_benchmark.csv; manifests/latency_environment.json | none | **Timing, not bit-reproducible** (shared 2-vCPU container); values in the paper are those of the delivered CSV |

## Reproduction evidence

* A clean copy of the project had `results/final/{metrics,ablation,statistical_tests,curves,predictions,manifests}` deleted and was re-run with `python experiments/run_final_analysis.py` and `python experiments/run_context_screening_simulation.py`; `diff -rq` against the delivered outputs reported **no difference** (4 Nov 2026, Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3).
* `python experiments/independent_verification.py` (a separate implementation that does not import `run_final_analysis.py`) recomputed **768 values** — confusion counts, accuracy/precision/recall/F1/FPR for 8 detectors x 8 scopes, ROC AUC and AP, McNemar counts and exact p-values, dF1, Holm-adjusted p-values and the always-flag closed form — and found **0 mismatches** (tolerance 1e-9). Result file: `results/final/metrics/independent_verification.csv`.
* Note: `all_predictions_with_rules_and_hybrids.csv` stores `rule_score` rounded to three decimals; AUC/AP in `auc_ap.csv` use the unrounded scores (differences ~1e-7 appear if the rounded column is used). The independent script therefore re-scores the stored texts with the frozen rule module.

## What was not reproduced

* The author's original training run (`results/final_experiment/*`) was **not re-executed**: dataset download was blocked. `experiments/run_final_experiment.py` is a reconstructed training script that has never been run; it was only verified to reproduce the thresholds 0.47 and 0.60 from the stored validation sweep.
* Bootstrap intervals for the two ML baselines in the headline table are the author's originally reported values (marked with a dagger); this project's re-computation gives slightly different bounds (e.g. 0.7380-0.8043 vs 0.7352-0.8027).

## Input and script hashes

| File | SHA-256 |
|---|---|
| `src/rules.py` | `b757469428dcac0d60adeb920ba4a480d1bce728132b9d4a60fc18caa9683677` |
| `experiments/run_final_analysis.py` | `17820001616ca9d2d8cbed725a240e6a817b5ac33611b6005f285b7d44ccc083` |
| `experiments/run_context_screening_simulation.py` | `64028b458ecf5b5ddd534249356b956b86dd2c582df28a270555a1af35995cfd` |
| `experiments/independent_verification.py` | `66314d06368be5aa4eb3dd9f7c58c0e0c6f0b4875f4c525e5729b01c65436587` |
| `experiments/make_figures.py` | `5bf82209d20ba0e9ba8f2bb702467369bfbdc63df44b61cbcdb19a133c633159` |
| `experiments/run_final_experiment.py` | `fbc3bbd7c2ddd28cee472bfee21741afad38e087c0ea35da3d40072b7dc2b119` |
| `results/final_experiment/all_test_predictions.csv` | `65fda4516897bc5f7fd2a6f22e1fb5ad2063ad8d526fb8546be8a1f17c48999f` |
| `results/final/predictions/all_predictions_with_rules_and_hybrids.csv` | `5814699d24622b9295c8409d3bc93e3a60c21df814c17da1dc3a56d5bb813f22` |
| `results/final/metrics/all_detectors_all_scopes.csv` | `454e200537dc19a1c8cb0b9da6365eba851ebaba27321e2e4254439fd4cdf247` |
| `results/final/metrics/auc_ap.csv` | `9d777028d38657981161136a1f833cfb7a79d613ebd9bdecea112db150cab08b` |
| `results/final/statistical_tests/paired_tests.csv` | `862039cd3bc663d1c6013cc34eb67a488d75e42294445f07d1b04c93ce138961` |
| `results/final/metrics/context_screening_simulation_per_seed.csv` | `fac77cda302a5651728dbbef4f08cfe43e4ef2eea662292e9fcf024eebb86fe5` |
| `results/final/metrics/context_screening_simulation_summary.csv` | `5ea396501d9a91d50de3d34cb1d8731b6eadf443af05ae10d9f1beeda995604c` |
| `results/final/metrics/independent_verification.csv` | `4e3df476d2978668906403240cc5928c81ba908ed064e8f209c1636f3adb0485` |
| `experiments/run_extra_metrics.py` | `27385c3be458c81597d80bb5d9ea7821c23ae78b5390ac7a17028afde71dceba` |
| `experiments/run_latency_benchmark.py` | `e38c2a19a8e728f1aec6ca7758eba304aaf0ae35f488e4cd88eaa25f9ed90bd8` |
| `results/final/metrics/extra_metrics.csv` | `0ed7dc0960c6b5c7b5f30de4ec45dc91db43a6e59c6b5b92124d0771efe3f017` |
| `results/final/metrics/latency_benchmark.csv` | `5074fd4646192956127abfab282eeb391469f2f6985fae016c3a3d295fbc916a` |
