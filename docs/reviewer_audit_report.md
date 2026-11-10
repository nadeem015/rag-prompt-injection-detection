# Simulated reviewer audit

Three simulated reviewers read the manuscript. Findings are listed with the action taken. Items marked OPEN could not be resolved with the available evidence.

## Reviewer 1 - cybersecurity / prompt injection
| # | Concern | Action |
|---|---|---|
| 1 | No end-to-end attack evaluation; 'mitigation' unsupported | Retitled; abstract/conclusion state no ASR measured; simulation labelled a proxy. OPEN (needs an LLM experiment) |
| 2 | Threat model silent on adaptive attackers | Out-of-scope statement added; evasion listed as untested |
| 3 | Rule detector provenance | Disclosed as a *reconstruction* (labelled in every table), frozen and hashed; results not equated with earlier rule results; provenance table in `docs/experiment_provenance.md` |
| 4 | Exposure simulation could reward blanket flagging | Reported with false-quarantine and plain-carrier check (n=10, anecdotal) |
| 5 | Boundary delimiting effect not measured | Stated explicitly; not claimed as protection |

## Reviewer 2 - ML / NLP
| # | Concern | Action |
|---|---|---|
| 1 | F1 gain may not reflect better model | Added ROC AUC/AP; word-only ranks better (0.9877 vs 0.9455); gain attributed to calibration |
| 2 | Always-flag baseline missing | Added to all tables; word+char beats it on 3 of 7 sets, word on 6 of 7 |
| 3 | Validation far easier than test | Reported (FPR 0.0189/0.0063 vs 0.6226/0.2830); thresholds not retuned |
| 4 | Bootstrap CIs differ from original | Original CIs printed for baselines with marker; difference disclosed |
| 5 | McNemar vs F1 disagreement | Explained; both reported; independence caveat noted |
| 6 | Possible stylistic shortcut | Presented as observation, no causal claim; experiment listed as future work |
| 7 | Training/diagnostic overlap | Limitation added; unaudited. OPEN (training texts unavailable) |

## Reviewer 3 - methodology / IEEE quality
| # | Concern | Action |
|---|---|---|
| 1 | Dataset identity/licence unverifiable | Flagged in Appendix and reference report. OPEN |
| 2 | Pooled metrics mislead | Per-dataset table first; pooled labelled diagnostic |
| 3 | Prevalence ~50% unrealistic | Limitation added |
| 4 | Template compliance | Built with the official IEEEtran.cls v1.8b; venue-specific template/page limit and PDF eXpress check remain OPEN |
| 5 | References order/verification | Ordered by first citation; verification report; unverifiable items omitted or flagged |
| 6 | Figures readable in print | Wide figures spanned both columns; hatching/markers used |
| 7 | Reproducibility | Appendix, manifests, hashes, unexecuted items listed |

## Round 2 (technical correction pass)
| # | Concern | Action |
|---|---|---|
| 1 | Equations missing in DOCX | 17 numbered equations rebuilt (decision rule, TF-IDF, LR objective and output, noisy-OR, hybrids, threshold selection, accuracy/precision/recall/FPR/F1, always-flag F1, AUC, AP, McNemar, Holm, exposure/false quarantine); native-OMML and image variants; equation numbers match the LaTeX aux file |
| 2 | New experiments lack provenance | `docs/experiment_provenance.md` (script, input, output, seed); clean-copy re-run identical; `experiments/independent_verification.py` 768/768 |
| 3 | Dataset claims unverified | Horizon-Labs card read; per-config rows/positives/licences recorded; independence of boundary_pairs from training data stated as NOT established |
| 4 | PIDS-Bench record | cited as arXiv preprint; DOI reported only as listed on arXiv; IEEE Access status unconfirmed (OPEN) |
| 5 | App never run | stub smoke test added and labelled as such; Streamlit install failure recorded verbatim; launch remains OPEN |
