# Software testing report

Date: 2026-11-04. Environment: Linux container, Python 3.13.x, pytest 9.1.1, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, pypdf 6.17.0.

## Executed

| Item | Command | Result |
|---|---|---|
| Unit and integration tests | `python3 -m pytest -q` | **52 passed** (test_rules 17, test_detector 13, test_pipeline_integration 13, test_ingest_retrieval 7, test_pdf_ingest 1, test_app_smoke_stub 1) |
| Real-PDF ingestion | `tests/test_pdf_ingest.py` (hand-built one-page PDF, `pypdf`) | passes: text extracted and chunked |
| App logic smoke test (**stub, not a launch**) | `tests/test_app_smoke_stub.py` | passes: `app.py` runs top to bottom against a fake `streamlit` module, including a simulated click on "Run pipeline" (query screening, retrieval, quarantine and event logging execute; event log redirected to a temporary file) |
| Syntax check of the UI | `python3 -m py_compile app.py` | compiles |
| Syntax check of reconstructed training script | `python3 -m py_compile experiments/run_final_experiment.py` | compiles |
| Threshold-rule check | the selection rule applied to the stored validation sweep | reproduces 0.47 (word) and 0.60 (word+char) |
| Model inference reproducibility | stored joblib models applied to stored texts | max abs difference 2.2e-16 vs stored probabilities (scikit-learn 1.9.1 loading 1.8.0 pickles gives a version warning) |
| Metric re-derivation | `experiments/run_final_analysis.py` | 106 comparisons, max abs diff 1.1e-16 |
| Independent recomputation | `experiments/independent_verification.py` | 768 values (counts, accuracy/precision/recall/F1/FPR for 8 detectors x 8 scopes, AUC, AP, McNemar counts and exact p, delta-F1, Holm, always-flag closed form): 0 mismatches at tolerance 1e-9 |
| Clean-copy reproduction | outputs of `run_final_analysis.py` and `run_context_screening_simulation.py` deleted in a copy and regenerated | `diff -rq` found no difference |
| Paper-number audit | `paper/verify_paper_numbers.py` | 152/152 checks pass |
| Equation numbering | LaTeX `.aux` labels vs DOCX builder labels | 17/17 equation numbers identical |

Covered by tests: rule indicators and normalisation (zero-width, NFKC), noisy-OR scoring, fusion logic, ML detector loading/scoring/explanations, ingestion and chunking (txt/md/csv/pdf), retrieval ranking, end-to-end pipeline behaviour (query refusal, chunk quarantine, boundary neutralisation of forged delimiters, empty-query error, event log including truncated lines), and that planted `POISONED_DEMO` documents are quarantined.

## Streamlit launch attempt (verbatim record)

`python3 -c "import streamlit"` -> `ModuleNotFoundError: No module named 'streamlit'`.

`pip install --break-system-packages streamlit` (4 Nov 2026) ended with:

```
WARNING: Retrying (...) after connection broken by 'NameResolutionError("HTTPSConnection(host='pypi.org', port=443): Failed to resolve 'pypi.org' ([Errno -2] Name or service not known)")': /simple/streamlit/
ERROR: Could not find a version that satisfies the requirement streamlit (from versions: none)
ERROR: No matching distribution found for streamlit
```

## NOT executed (stated exactly)

* **The Streamlit application has never been launched or used in a browser.** Upload, tabs, sliders, charts, reruns and layout are untested. The stub test above checks only that the Python logic of `app.py` runs against a fake module; it must not be described as a successful launch.
* **Models were not retrained.** `experiments/run_final_experiment.py` is a reconstruction that needs Hugging Face access and has never been executed; it was only verified to compile and to contain a threshold-selection routine that reproduces 0.47 and 0.60 from the stored validation sweep.
* No end-to-end test with a generative LLM exists; the answer stage is extractive.
* No performance, load or adversarial-robustness testing.
* Not tested on Windows/PowerShell; the README commands are untested there.
* The paper's DOCX equations: the native (OMML) variant was checked structurally (pandoc round trip recovers 17 display and 107 inline equations; 13 fractions, 6 n-ary operators) but was **not opened in Microsoft Word**; LibreOffice in this environment cannot render OMML. The viewer-safe variant (formulas as images) was rendered via LibreOffice and inspected.
