# Research methodology report

## Question
How well do lexical detectors (rules, word TF-IDF + LR, word+char TF-IDF + LR, two hybrids) screen text for injected instructions in a small RAG pipeline, and what does screening cost in over-blocked benign content?

## Retitling
The earlier title claimed 'mitigation'. No experiment measures whether attacks are prevented end to end against a generative model, so the paper is retitled around detection and screening ('Prompt Injection Detection in Retrieval-Augmented Generation Systems: A Lexical-Baseline Study with Rule and Hybrid Screening and a Context-Screening Simulation'). The phrase 'mitigation effectiveness' is not used.

## Evidence base
* Historical author results (read-only): `results/final_experiment/` (13,183 stored test predictions with texts, metrics, validation sweep, two fitted pipelines).
* The author's original `src/` was missing. The rule detector, hybrid code and pipeline were reconstructed in this project; the re-implemented rules were frozen (SHA-256 b757469428dc...3677) before being evaluated.

## Protocol
1. Reproduce: re-derive all 106 reported metrics from stored predictions (max diff 1.1e-16); re-score stored texts with stored models (max diff 2.2e-16); confirm predictions equal p >= threshold on all rows.
2. Thresholds: validation only, policy 'max F1 subject to FPR <= 0.10, ties prefer recall then lower threshold'; models refit on train+validation (author's run). No threshold was chosen or changed using test data. Hybrid constants (OR; +0.10 bonus) were fixed a priori.
3. Add rule and hybrid detectors, an always-flag baseline (F1 = 2p/(1+p)), ROC AUC, average precision, exact McNemar, paired bootstrap on F1 difference, Holm correction on the primary family, percentile bootstrap CIs (1000 replicates, seed 42).
4. Per-dataset reporting; pooled results labelled as diagnostic.
5. Context-screening simulation: 545 headline payloads in 10 benign carrier paragraphs, 3 seeds, prefix/suffix; exposure and false quarantine. Proxy only; no LLM.
6. Error analysis: false-positive categories, top coefficients, rule-category hits, direct/indirect and language breakdowns.

## Deviations and disclosures
* Bootstrap CIs recomputed here differ slightly from the originals; the paper prints the original CIs for the two baselines (marked) and this project's CIs for new detectors.
* Training-split class counts (2,162/1,761) come from the manuscript; no training file was available.
* Post-hoc recall at FPR <= 0.10 is diagnostic only and labelled so.
* Dataset cards could not be verified (see reference report).
* The word-only finding (higher AUC than word+char) was not anticipated; it is reported as found.
