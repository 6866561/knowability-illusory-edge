# Response to review (revision 2)

Each concern was treated as a hypothesis and checked against the manuscript, code and outputs. Item numbers follow the review; "Ledger #" refers to docs/issue_ledger.md.

**1. Issue ledger.** Provided (docs/issue_ledger.md, 31 entries) with status, evidence, change, validation and limitation.

**2. Contribution and novelty.** Agreed in substance. A prior-art audit (refs_prior.bib, verified against Crossref/arXiv) classifies κ as Theil's uncertainty coefficient (established), the ladder and beta-binomial formulas as synthesis, the quantum ceiling as a folklore corollary, the chaos method as classical with new numbers, and drift forecasting as a new empirical finding with an incomplete literature check. The central contribution is now persistent label errors and the disagreement audit (new title, abstract, Sections I and IV). We recommend separate papers for the chaos and hardware studies; in this manuscript they are labelled exploratory. *Location:* Abstract, Sec. I-A, II, Discussion "Scope". Ledger #30.

**3. Definitions.** Agreed. "Best learner in a consistent class" removed; algorithm-specific learnability A_n(𝒜) and gap G_n defined with independent test example and explicit randomization; "physical ceiling" replaced by "Bayes ceiling relative to F"; H(Y)>0, the evaluation distribution and the role of training history stated. *Location:* Sec. III, Appendix A. Ledger #7.

**4. Theorems.** All propositions restated with assumptions and proved in Appendix A. Lemma 1: measurability and conditional independence of randomization. Prop. 1: depolarizing convention stated; asymmetric readout derived. Prop. 2: oracle hyperparameters, weighting, monotonicity proof; real-data use labelled approximate. Props. 3–4: fixed rules; general row-varying identity; retrained pipelines treated separately. Prop. 5: correct; baseline condition explicit. Prop. 6: stationarity, SLLN, ties, learner restriction, fallback dependence. Prop. 7: fixed finite population, sampling design, hypergeometric bound, weighted worst case. Prop. 8: conditional on training data. Numerical checks: tests/test_rev2.py. Ledger #8–15.

**5. Nearest neighbours.** Agreed in principle; numerically the floors are small in our settings. Floors E[r(1−2r)] are now estimated from saved posteriors (≤0.0056 across all systems, seeds and horizons; far below the 0.05 tolerance); a consistent k-NN was added; floor and zero-floor fits compared; crossing status reported; claims restricted to named learners. The learner matters: Lorenz-63 rates 2.20±0.09 (1-NN) vs 3.31±0.11 (k-NN); Rössler 1.95±0.32 vs 3.31±0.17. *Location:* Sec. VI-D, Fig. law3, Table law3. Ledger #1–2.

**6. Cover–Hart / Hellman–Raviv.** Agreed. Relabelled as asymptotic plug-in diagnostics; upper-bound language and atlas whiskers removed; silent clipping identified (values ≥0.5 are undefined, not clipped, in revised code paths). We did not reinterpret failures as bounds on "learnable predictability". *Location:* Sec. III, Tables controls/atlas, Fig. atlas. Ledger #6.

**7. Chaos reference.** Generative model fully specified. Particles were verified to be centred on the observation (no leakage). The local flat-prior approximation was compared with an importance-sampled posterior over 2×10⁷ invariant-measure samples at moderate noise, and particle-count and step-size convergence were checked at the study's noise level (Lorenz-63, L ≤ 2: decision agreement ≥0.966 and accuracy difference ≤0.008 at σ∈{0.2,0.5}; 500 vs 2,000 particles changes accuracy by 0; halving dt leaves ≥99.6% of decisions unchanged). Not covered: tiny noise directly, L=3, other systems. Accuracy of the reference is now described as an estimate, not a bound; entropy is not derived from accuracy. *Location:* Sec. VI-D. Ledger #4–5.

**8. Scaling re-estimation.** Three independent seeds per system; crossings classified; per-seed slopes; exponential vs power-law comparison; sensitivity to tolerance (0.02/0.05/0.10) stored; D₂ sensitivity across scaling windows and samples; λ standard errors. Lorenz-96 rates are not identifiable within 10⁶ samples under the revised protocol and are excluded from headline claims. Language bounded to "close to exponential over the tested range" for Lorenz-63 with specified learners; for Rössler exponential and power-law forms fit equally well (R² 0.960 vs 0.965 for 1-NN), so no functional form is claimed. The γ_L–1/β̂ comparison was dropped from the claims because both come from the same curves. Not done: block resampling across horizons beyond seed replication; larger n for Lorenz-96. Ledger #2–3.

**9. Ladder and atlas.** Agreed. Increments recast as achieved score improvements; clipping removed (negative increments reported); denominators stated; references visually distinct from fitted skill; "unknowable" language removed. Ledger #16.

**10. LSS claims.** Agreed. Population decomposition (Eq. lss); expectation vs realized; constant vs varying baselines; stock ceiling and "simple models near frontier" claims removed. Ledger #17–18.

**11. Disagreement audit.** Exact hypergeometric bound (K_U=10 of 82, vs Clopper–Pearson 0.153); unverifiable row treated as wrong in sensitivity (K_U=12); weighted worst-case bound; sampling seed and exclusions documented; scope (fixed population, not future edge) stated; power calculation re-derived (8,235 under independence; ×design effect). No post-audit rule change was applied to the audited data. Not done: independent second adjudicator. Ledger #19.

**12. Earnings fixed cohort.** Done for what the archive allows: (1) archived predictions on original labels for removed rows (+20.1 points); (2)–(3) archived vs retrained predictions on a fixed cohort of 587 correctly labelled rows (+0.46 vs −0.81 points, both CIs including zero). Corrected labels for removed rows, independent adjudication and midpoint-proximity sensitivity require relabelling from press releases and were not completed. Ledger #20.

**13. Point-in-time audit.** docs/leakage_spec.md covers every task; no demonstrated leakage; unresolved risks listed (Sec. Threats). Polling duplicates are deduplicated on calibration timestamp before splitting. Ledger #28–29.

**14. Inference.** Time-block bootstraps added (stock skill CI now includes zero; presidential skill imprecise); DM construction specified and marked approximate; model selection moved to validation (8 domains rerun: no quantum model significantly better, smallest Holm p 0.576; the stock comparison favours the circuit under an entity bootstrap, p 0.0025, but not under a month-block bootstrap, p 0.072, and both skills are near zero, +0.0002 and −0.0012 — reported in the text); "no significant advantage detected" replaces any absence claim; no equivalence margin is claimed. Hardware generalisation limited to qubits of three processors over the period. Ledger #21–23.

**15. Preregistration.** Hash identifies content only; timing rests on local records; H1a/H1b scored separately; failure-to-reject language corrected; confirmatory vs exploratory labelled throughout. Ledger #24.

**16. Quantum claims.** Prop. 8 scope stated; CHSH defined; 0.75 identified as a game bound; simulation's classical side-information protocol defined; quantum-memory caveat (basis knowledge; no simultaneous incompatible outcomes); Delft heralded convention and literal win rate both reported; differing p-value procedures distinguished; live-hardware limitations stated. Ledger #25–27.

**17. Readout ceilings.** Optimal-threshold closed form derived; symmetric formula identified as a special case; reported values were already optimal-threshold (difference ≤2e-5); live ceilings now have shot-bootstrap CIs; the noise-model figure's agreement with its line is by construction and described as such. Ledger #9.

**18. Figures.** Atlas redrawn (achieved skill only, references distinct, no ceiling whiskers); new scaling figure shows measured points, crossing status and seeds; earnings mechanisms in a table; quantum table states selection rule; calibration heatmaps added in the main text with 3D renderings moved to an appendix. Philosophical framing reduced.

**19. Reproducibility.** release/ contains lockfile, environment, SHA-256 hashes of data, results and raw caches, a table/figure-to-script manifest, exact commands (light and full), data-provenance and redistribution notes, chaos settings, quantum job details. No persistent DOI was minted (requires the author's account on a repository such as Zenodo).

## Change log (revision 2, code)
- `core.cover_hart_upper`: silent clipping identified; new code paths report values ≥ 0.5 as undefined.
- `atlas.py`: ladder increments signed (no clipping).
- `run_law3.py` (new): 3 seeds, 1-NN and k-NN, saved posteriors, 1-NN floors, floor/zero-floor fits with AIC, crossing classification, D₂ window sensitivity.
- `validate_posterior.py` (new): importance-sampled posterior over 2×10⁷ invariant-measure samples; particle and step-size convergence.
- `rev2_analyses.py` (new): hypergeometric audit bound (two scenarios), power, fixed-cohort earnings, asymmetric readout ceilings with shot bootstrap, time-block bootstraps.
- `make_quantum2.py` → `run_quantum2.py` (new): validation selection, DM, entity and month-block bootstraps. A wrong-tail bootstrap p in the first draft was found and fixed before any number reached the manuscript (ledger #32).
- `make_macros_rev2.py`, `fig_rev2.py`, `make_release.py` (new); `make_tables.py`, `add_stats*.py` take `KN_PAPER`.
- `tests/test_rev2.py`: asymmetric readout closed form, row-varying inflation identity, 1-NN asymptotic error, hypergeometric coverage, beta-binomial finite-history accuracy (15 tests pass).
