# Response to the updated requirements (both papers)

All code and results are at https://github.com/6866561/knowability-illusory-edge, commit `d6f5aad`.

## Done

**1. Reproduction (partial: author-run, not independent).**
- `reproduce.py` and `requirements-cpu.txt` are now published, along with the prediction datasets for all tasks except stocks (stock labels derive from Yahoo prices and stay rebuildable by script).
- A first clean-room run, from a fresh GitHub clone in a new environment, found two portability bugs: a missing output folder, and a crash when the press-release texts are absent. Both were fixed without changing any result.
- A second clean-room run reproduced **551 of 551** compared values exactly (relative tolerance 1e-9; largest difference 0). It covered the unit tests, audit bounds, earnings cohort, readout ceilings, time-block and two-way intervals, the issuer breakdown, the qubit model term, the reported-EPS cross-check and the coverage suite.
- The report and log are in the repository (`results/reproduction.json`, `reproduction_log.txt`).
- Not covered: the GPU chaos studies, the quantum comparison, the raw-data builders, the stock rows, and the guidance text screen.
- **An independent party has not yet reproduced it.**

**3. Dependence-method validation.**
- `coverage_suite.py` covers the LSS ratio itself, with 1,000 panels per scenario; the Monte Carlo SE is at most 1.2 points.
- Two-way coverage was at least 98.6% at nominal 95% in every scenario: independent shocks, AR(1) calendar shocks, unequal weights, 30% missing cells, and 8 time clusters.
- One-way resampling covered 83.9–92.4%.
- The two-way intervals are conservative, 1.7–1.8 times wider.

**5. Figure/table consistency.** The atlas figures now use the two-way intervals, with the hardware task's qubit-only interval drawn dashed. Legends and captions are updated, and every value comes from the same results file.

**Illusory Edge, item 2: mechanism made inspectable.** `issuer_breakdown.py`, a new figure, and the row-level table `data/d7_issuer_breakdown.csv`:
- The 76 invalid rows come from 31 of 98 issuers. 50 of them sit in 9 issuers whose invalid rows recur across two or more fiscal periods: Pentair 15 rows over 11 periods; Teleflex 11 rows over 3 periods, all with the same recorded outcome.
- On invalid rows, the entity rule matched the recorded outcome 63% of the time against 38% for the base rate. On valid rows the two were 61% vs 59%.
- Invalid rows supplied +2.48 of the +2.88-point pooled edge.

**Illusory Edge, item 3: audit implementation checks.**
- The hypergeometric step has an exact minimum coverage of 95.1%, computed rather than simulated.
- Monte Carlo coverage of the implemented bound is at least 96.2% with highly unequal weights and errors on the heaviest rows, and 100% with unresolved rows counted as wrong.
- If the adjudicator misses half the errors, coverage falls to 92.8% (plain) and 84.9% (tightened). This is the reason the plain bound is primary and human adjudication matters.

**Knowability, item 2: uncertainty through the full scaling analysis.** 20 independent datasets per system were run, regenerating trajectories, observations, learners, crossings and regressions. Bootstrap intervals are over datasets, at tolerance 0.05:

| System | 1-NN | k-NN |
|---|---|---|
| Lorenz-63 | 2.22 (2.19–2.25) | 3.42 (3.36–3.49) |
| Rössler | 1.85 (1.76–1.95) | 3.38 (3.23–3.52) |

- Lorenz-63 rates are stable when the smallest or largest sample size is dropped, or when crossings are interpolated linearly.
- The Rössler k-NN rate falls to 2.81 when the smallest sample size is dropped, so it depends on the sample-size range; the paper says so.
- Error in the approximate ensemble reference is not included.

## Not done (needs people, a new method, or new data)
- **An independent reproduction by someone else.**
- **Independent human adjudication of all 40 audit rows.** The worksheet is `data/d7_human_audit_sheet.csv`.
- **Chaos reference in the tiny-noise, long-horizon regime.** This needs an attractor-aware posterior method; it is a research step.
- **Hardware two-way inference.** The row-level drift predictions were not saved, so this needs `qdrift.py` re-run with predictions kept.
- **Point-in-time verification against source records for every domain.**
- **A second earnings dataset or a locked forward evaluation.**
- **A DOI.** This requires linking Zenodo to the GitHub account.
- **Expert review:** an independent proof check, and a specialist check of the earnings definitions.
