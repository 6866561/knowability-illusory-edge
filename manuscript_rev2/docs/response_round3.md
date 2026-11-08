# Response to review, round 3 (Illusory Edge)

**1. Audit weights not reproducible.** Agreed and fixed. New Table I (Audit Population, Weights and Bounds) and an expanded audit paragraph set out the full accounting:
- **Population.** The cleaned universe has 972 rows. Of these, 340 were hand-checked earlier and lie outside the audit population. The audit population is the 632 harvested rows from 90 issuers.
- **Weights.** Each row is weighted 1/(rows forecasting the same outcome). Total weight is 383 outcome groups; weights range from 0.167 to 1.
- **Disagreement rate.** There are 82 disagreement rows: 0.130 of rows, or 0.105 by weight (40.2 of 383). The paper now explains why 82/972 is not the relevant rate.
- **Bound.** W_(K_U) is 10.0 for K_U = 10 and 12.0 for K_U = 12. The bound −0.8747 − 2·10/383 = −6.10 points can be recomputed from the table. The unverified-row sensitivity analysis is kept beside it.
- **Code.** All fields are written by `rev2_analyses.py` under `audit.accounting` and `W_top_KU`.

**2. Empirical result vs theorem.** Agreed.
- The abstract, contributions, results, discussion and conclusion now state the defensible conclusion: the apparent advantage was concentrated in invalid or target-incompatible evaluation rows, and the retained cohort showed no established advantage.
- The paper says explicitly that the application does not demonstrate the sign reversal, which Proposition 3 establishes only as a possibility.
- "Correcting the training labels changed little" is replaced by the paired change on the same issuer resamples: −1.27 points (−4.47 to +1.88). No effect was detected, and effects of a few points either way are not excluded. The paired change is added to Table II.

**3. Chaos reference.** Agreed.
- Attainability is now stated as conditional on the reference model; the floors inherit its approximation.
- The paper states that particle-number and step-size checks test numerical stability, not the posterior assumption.
- All results are described as gaps to the ensemble reference.
- Validating the tiny-noise regime directly remains open: it needs a different posterior scheme.

**4. Statistics.**
- **Table III** now shows entity and time-block intervals side by side. A new earnings quarter-block interval is included: 0.032 to 0.094 over 18 quarters.
- **Hardware task:** only qubit resampling is available, and the table, text and threats section now say it ignores shared device-time shocks, with coverage not validated.
- **Chaos seeds:** three-seed standard deviations are labelled a descriptive stability measure, not a pipeline interval.

**5. Access point.** Partly addressed.
- The exact code commit (1a25d144b1acfd1f43a0ff2355368544637f510d) is now given.
- The title footnote and Appendix C state that no public URL or DOI exists yet.
- A persistent deposit needs the author's GitHub or Zenodo account and has not been made.

**6. Smaller corrections.**
- **1-NN excess error:** the condition is corrected (zero when η ∈ {0, ½, 1}, including η ≡ ½).
- **Eq. (1):** population LSS is defined as a ratio of expected losses and distinguished from empirical LSS, in the main text and the appendix.
- **Proposition 7:** the noninformative-weighting assumption is stated, with the alternative for rate-dependent sample sizes.
- **Fig. 1(c):** redrawn as a forest plot with every estimate labelled by its population. It uses the hypergeometric weighted bound and drops the Clopper–Pearson comparison.
- **Appendix:** the superseded single-seed learning surface is removed.
- **Quantum and hardware material:** moved to supplementary Appendix B, with a one-paragraph summary in the main text.

**Also found and fixed during this round.**
- The paper had said "nine tasks", but the atlas has nine real-data tasks; corrected in both papers.
- In the companion knowability paper, two sentences overstated the earnings result and one misattributed earthquake skill to location; both corrected.
