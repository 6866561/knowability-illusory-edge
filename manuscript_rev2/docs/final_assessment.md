# Final assessment (revision 2)

Every number below is generated from `repo/results/*.json` by `make_macros_rev2.py`, `add_stats*.py` and `make_tables.py`; none is typed by hand in the manuscript.

## Supported (claims retained, with stated assumptions)
- **Inflation identity** for fixed decision rules with row-varying label noise: Δ̃ − Δ = 2E[(f − g)(q̃ − q)]. Exact; verified numerically (`test_inflation_identity_row_varying`).
- **Symmetric noise shrinks** a fixed-rule edge by (1 − 2ρ). Exact.
- **Sign reversal** under entity-persistent noise. Holds under the stated baseline condition (1 − ω)m > ½.
- **Time-ordered validation does not detect entity-persistent noise.** Holds under stationarity, SLLN, no ties, the learner restriction and the fallback dependence, all stated.
- **Disagreement audit.** An exact hypergeometric finite-population bound (at most 10 wrong rows among the 82 disagreement rows, vs 0.153 by Clopper–Pearson). The weighted worst-case lower bound on the true edge is −6.10 points, or −7.14 with the unverifiable row counted as wrong. The audit therefore certifies no positive edge. Scope: a fixed population, not future edge.
- **Earnings mechanism.** The 76 removed rows carried +20.1 points of apparent edge under their original labels. On a fixed cohort of 587 correctly labelled rows, the archived and retrained models both have edges indistinguishable from zero.
- **Qubit readout ceiling.** Closed form for asymmetric readout, with shot-bootstrap CIs on the live values.
- **Drift forecasting on IBM devices.** Readout and two-qubit errors are partly forecastable one calibration ahead; T1/T2 are at most weakly forecastable. Exploratory, with an incomplete literature check.
- **1-NN floor.** E[r(1 − 2r)] is ≤ 0.0056 in every chaos setting, so the 1-NN crossings at tolerance 0.05 are not artefacts of an inconsistent learner.

## Weakened (claims narrowed)
- **Chaos scaling.**
  - The rate depends on the learner (Lorenz-63: 2.20 for 1-NN vs 3.31 for k-NN) and on the tolerance.
  - "Close to exponential" holds for Lorenz-63 only. For Rössler, exponential and power-law forms are not distinguished.
  - The γ_L = D₂ correspondence is specific to 1-NN at tolerance 0.05 in three-variable flows. It is no longer stated as a law.
- **Atlas.** Achieved score improvements against point-in-time base rates only. Time-block CIs make stock skill indistinguishable from zero. No "unknowable" or ceiling language.
- **Quantum ML.** Selection is now on validation data; the result is "no significant advantage detected", not equivalence.
  - Across 8 domains no quantum model is significantly better (smallest Holm-adjusted p = 0.576). The classical model is better under both bootstraps in 5 of them.
  - Stocks depends on the dependence model: the circuit wins under an entity bootstrap (p = 0.0025) but not under a month-block bootstrap (p = 0.072). Both skills are near zero.
- **Ensemble reference.** Now a reference estimate, not a lower bound. It was validated for Lorenz-63 only, at L ≤ 2:
  - Against an importance-sampled posterior at σ ∈ {0.2, 0.5}: ≥ 96.6 % decision agreement and accuracy difference ≤ 0.008.
  - At the study's noise level, going from 500 to 2,000 particles changes accuracy by 0 on 1,000 test states.
  - Halving dt leaves ≥ 99.6 % of decisions unchanged.
- **Preregistration.** The hash identifies content, not time. H1a and H1b are scored separately.
- **CHSH / side information.** 0.75 is a game-specific bound. The Delft literal and heralded conventions are both reported.

## Removed
- Lorenz-96 learnability rates. They rested on one seed, larger samples and extrapolation; they are not identifiable under the revised protocol.
- Cover–Hart and Hellman–Raviv "upper bounds" and the atlas whiskers.
- "Near-zero stock LSS ⇒ near-zero ceiling" and "simple models near the attainable frontier".
- "Best learner in a consistent class" and "physical ceiling" as defined objects.
- Novelty claims for κ (Theil's U), the ladder and the beta-binomial formulas. These are now classified as established or synthesis.

## Needs new data or inputs (not obtainable from the archive)
1. Corrected labels for the 76 removed earnings rows. Needs relabelling from the original press releases.
2. An independent second adjudicator for the 40-row audit, giving inter-rater agreement.
3. Midpoint-proximity sensitivity for earnings labels. Needs the guidance ranges at the row level.
4. Larger training sets (≫10⁶) for Lorenz-96 to identify rates. GPU time is the only cost, but it exceeds this session.
5. Direct validation of the ensemble posterior in the tiny-noise regime (σ = 10⁻⁴·std). Importance sampling is infeasible there without a much larger library or a different scheme (e.g. particle MCMC).
6. A third-party preregistration timestamp (OSF, AsPredicted). The existing preregistration can only be documented, not re-timestamped.
7. A persistent DOI for the release. Needs the author's Zenodo (or similar) account.
8. Multi-day and multi-device live hardware runs. The live run is one device on one day, and the free plan allows more.
9. A held-out period of self-polled IBM calibrations. Public datasets only, so publication delay is unknown.
