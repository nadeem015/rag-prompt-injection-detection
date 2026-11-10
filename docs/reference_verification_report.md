# Reference verification report

Verification date: 2026-11-04 (round 2 re-checked the dataset entries, PIDS-Bench and the BIPIA author list). Method: WebSearch (and WebFetch where the host was allowlisted) against publisher, proceedings, arXiv, OWASP/NIST and library-catalogue pages. Shell access to arXiv, Crossref, Hugging Face and CTAN was blocked, so DOIs were **not** resolved programmatically. Only identifiers that appeared on a retrieved authoritative or catalogue page are listed; anything not seen is omitted from the bibliography rather than guessed.

Status legend: *peer reviewed* = appeared in a proceedings/journal record; *preprint* = arXiv only; *dataset/web* = not a publication.

| Key | Status | Identifiers in bibliography | Verification note |
|---|---|---|---|
| `owasp2025` | web resource (not peer reviewed) | URL | Page located via search 2026-11-04; Top 10 v2025 release dated 2024-11-18 per OWASP. |
| `greshake2023` | peer reviewed (workshop) | DOI 10.1145/3605764.3623985, arXiv 2302.12173, pp. 79--90 | DOI, pages, venue confirmed via search of ACM record. |
| `perez2022` | workshop paper, non-archival; preprint | arXiv 2211.09527 | Venue and arXiv ID confirmed via search. |
| `liu2024formalizing` | peer reviewed | pp. 1831--1847 | Venue and pages confirmed via search; DOI not listed. |
| `yi2025bipia` | peer reviewed (KDD 2025); author list verified | DOI 10.1145/3690624.3709179, arXiv 2312.14197 | Author list resolved: the KDD'25 version has 7 authors (Yi, Xie, Zhu, Kiciman, Sun, Xie, Wu); K. Hines appears only in the v1 preprint. Prints as 'J. Yi et al.'. ACM page numbers not seen, omitted. |
| `zhan2024injecagent` | peer reviewed (Findings) | DOI 10.18653/v1/2024.findings-acl.624, pp. 10471--10506 | DOI and pages confirmed via search. |
| `debenedetti2024agentdojo` | peer reviewed | arXiv 2406.13352 | Venue and arXiv ID confirmed via search; page range not verified and omitted. |
| `zou2025poisonedrag` | peer reviewed | arXiv 2402.07867, pp. 3827--3844 | Venue, pages and arXiv ID confirmed via search. |
| `liu2025datasentinel` | peer reviewed | arXiv 2504.11358, pp. 2190--2208 | Venue and pages confirmed via search; DOI not verified and omitted. |
| `abdelnabi2024tasktracker` | preprint (arXiv comment mentions SaTML 2025; proceedings entry not verified) | arXiv 2406.00799 | arXiv listing only. |
| `jacob2025promptshield` | preprint | arXiv 2501.15145 | arXiv listing only. |
| `hung2025attention` | peer reviewed (Findings) | DOI 10.18653/v1/2025.findings-naacl.123, pp. 2309--2322 | DOI and pages confirmed via search. |
| `alon2023perplexity` | preprint | arXiv 2308.14132 | arXiv listing only. |
| `hines2024spotlighting` | preprint | arXiv 2403.14720 | arXiv listing only. |
| `chen2025struq` | peer reviewed | arXiv 2402.06363, pp. 2383--2400 | Venue, pages and arXiv ID confirmed via search. |
| `wallace2024hierarchy` | preprint | arXiv 2404.13208 | arXiv listing only. |
| `shire2026pids` | arXiv preprint verified; journal publication NOT verified | arXiv 2609.15017 (v1, 14 Sep 2026); DOI field on the arXiv record: 10.1109/ACCESS.2026.3728186 | arXiv abs page re-fetched 2026-11-04: title, authors (Yusuf Khalid Shire, Sang-Chul Kim), subjects cs.CR/cs.AI and the DOI field are as listed; the page has no journal-ref, comments or 'published in' statement. The bibliography therefore cites the arXiv preprint and prints the DOI only as 'listed on the arXiv record'; volume, pages and date are NOT given. Earlier secondary sources (ResearchGate, OpenTAI) mentioned IEEE Access vol. 14, while the authors' repository README said 'under review'; unresolved without IEEE Xplore. |
| `vassilev2025nist` | government technical report | DOI 10.6028/NIST.AI.100-2e2025 | DOI confirmed via search. |
| `lewis2020rag` | peer reviewed | pp. 9459--9474 | Venue and pages confirmed via search. |
| `salton1988` | peer reviewed | DOI 10.1016/0306-4573(88)90021-0, pp. 513--523 | DOI, volume, pages confirmed via search (this session). |
| `pedregosa2011` | peer reviewed | pp. 2825--2830 | Volume and pages confirmed via search. |
| `saito2015` | peer reviewed | DOI 10.1371/journal.pone.0118432, pp. e0118432 | DOI, volume, article number confirmed via search (this session). |
| `mcnemar1947` | peer reviewed | DOI 10.1007/BF02295996, pp. 153--157 | Confirmed via Springer record (this session). |
| `dietterich1998` | peer reviewed | DOI 10.1162/089976698300017197, pp. 1895--1923 | Confirmed via search. |
| `holm1979` | peer reviewed | pp. 65--70 | Volume, issue, pages confirmed via search (this session); DOI not verified and omitted. |
| `deniz5k` | dataset; existence and test split corroborated indirectly; own card NOT opened | URL | The Horizon-Labs card (config agentic5k_test) names 3nesdeniz/agentic-prompt-injection-5k (test) as its source with 545 rows, 280 positives, CC-BY-4.0, matching the analysed headline split. The dataset's own card could not be fetched, and a profile search listed other datasets by the same author but no '5k' repository. Training/validation sizes (3,923/467), scenario isolation and the training-data licence come from the project's manifest and need the author's confirmation. |
| `horizon2026suite` | dataset card read 2026-11-04; no citation block on the card | URL | 16 configurations, 24,773 rows, licence field 'mixed-per-config'. Configs used here (rows/positives/licence): boundary_pairs_test 240/120/CC-BY-4.0; neuralchemy_test 942/552/Apache-2.0; slabs_test 2101/1051/MIT; synthetic_docs_test 3354/1113, synthetic_direct_test 3162/1585, synthetic_v2_test 2839/1412 (all Apache-2.0); sum 12,638 = pooled diagnostic set; headline agentic5k_test 545/280/CC-BY-4.0. Upstream sources are named per config; synthetic configs are LLM-generated by the publisher. The card states no overlap analysis between configs and the 5K training data; independence of boundary_pairs_test (same author as the 5K data) is NOT established. |

## Items needing the author's attention before submission

1. **`deniz5k` (primary dataset).** The dataset's own card could not be fetched. Indirect corroboration (Horizon-Labs card): repository name, test-split size (545), positives (280) and licence (CC-BY-4.0) match. Still to be confirmed from the card: the train/validation sizes (3,923/467), the scenario-isolation statement, and the training-data licence.
2. **`horizon2026suite`.** The card was read and the six diagnostic configurations are described per config in the paper. Two points remain open: (a) no overlap/deduplication statement between configs and the 5K training split exists on the card, so train/test independence of the diagnostics (especially `boundary_pairs_test`) is untested; (b) the three synthetic configs were generated by the suite publisher, who describes them as in-distribution for the publisher's own models. The paper's wording reflects both.
3. **`shire2026pids` (PIDS-Bench).** Cited as an arXiv preprint. The IEEE Access publication (DOI 10.1109/ACCESS.2026.3728186, listed on the arXiv record) could not be confirmed because IEEE Xplore and doi.org are not reachable here; the arXiv page itself carries no journal reference. If the author confirms publication, update the entry with volume, article number and date from IEEE Xplore.
4. **Preprints.** `abdelnabi2024tasktracker`, `jacob2025promptshield`, `alon2023perplexity`, `hines2024spotlighting`, `wallace2024hierarchy` are cited as arXiv preprints. TaskTracker's arXiv comment mentions SaTML 2025, but no proceedings record was verified.
5. **Not cited (unverified).** A search hit for a TF-IDF + linear-SVM prompt-injection paper (arXiv 2512.19011) could not be fetched and is not cited. Informal repositories and theses on TF-IDF injection classifiers were found but are not citable evidence.
6. **Page ranges not verified (omitted):** BIPIA (KDD '25), AgentDojo (NeurIPS '24 D&B).
7. **DOIs were not resolved programmatically** (doi.org, Crossref and arXiv are not reachable from the shell; WebFetch rejects doi.org). Every DOI printed was seen on a retrieved publisher, arXiv or catalogue page.

## Methods references
McNemar (1947), Holm (1979), Dietterich (1998), Salton & Buckley (1988), Saito & Rehmsmeier (2015), scikit-learn (2011), Lewis et al. (2020) were checked against catalogue/publisher pages as listed above; Holm's DOI was not seen and is omitted.

## Novelty statement
The paper makes no novelty claim for the detectors. The related-work table is qualitative and based only on what the retrieved records state. The literature search was limited to the sources above; it is not a systematic review.
