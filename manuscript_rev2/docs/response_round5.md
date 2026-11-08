# Responses to review (round 5: Illusory Edge; round 2: Knowability)

## Illusory Edge

**1. The audit guarantee assumes correct adjudication.** Agreed.
- Propositions 5 and 12 now assume that sampled rows are adjudicated correctly, and the text says the guarantee covers sampling error only.
- The adjudication protocol is documented in full:
  - one reviewer, an AI system (a Claude model);
  - a fixed four-point checklist applied to the guidance side of each label: (a) a GAAP diluted EPS range whose midpoint equals the recorded midpoint, (b) matching period, (c) total rather than continuing-operations EPS, (d) not a pre-announcement;
  - a failed check and an evidence quote recorded for each row;
  - "unverifiable" used rather than a guess.
- Not done, and stated: the reported-EPS side was not re-checked, and there has been no independent second adjudication.
- The acknowledgment and threats section say the AI system served as adjudicator.
- The 340 rows outside the audit population are no longer described as "checked by hand"; they are the original package's previously audited set.
- **Needs you:** independent second adjudication of the 40 sampled rows by a person. The rows, checklist, verdicts and evidence quotes are in `data/d7_audit_results.csv` and `data/d7_harvested_audit_sample.csv`.

**2. Tighter bound (optional).** Done.
- Rows verified as correct cannot be wrong. So after the three sampled problem rows (total weight 1.75), at most 9 further errors can lie among the 42 unaudited rows (top-9 weight 9.0).
- The primary bound tightens from −7.14 to −6.49 points, and from −6.10 to −5.84 when the unresolved row is excluded.
- −7.14 stays as the simple conservative bound; the tightened values are reported alongside it in the text and Tables I–II.
- Computed in `rev2_analyses.py` under `audit.scenarios.*.tight`.

**3. Reproducibility.** The public copy is prepared and waits only on the author's GitHub login. The acknowledgment no longer says "released code"; it now refers to the code identified in Appendix C.

**4. Unresolved limitations.** Acknowledged; no further caveats were added.

**5. Wording and layout.**
- Appendix A now says the audit bounds the clean-label edge on the fixed evaluation population, starting from the recorded-label edge.
- The references are no longer interrupted by floating figures.

## How Much of the Future Is Knowable?

**1. Model term vs accuracy.** Agreed, and now measured. `qubit_modelterm.py` computes E KL(η‖p̂) exactly from the known η(θ) on 200,000 held-out angles, with three training seeds:

| Training examples | Model term (nats) |
|---|---|
| 10³ | 0.090 |
| 10⁴ | 0.014 |
| 10⁵ | 0.00072 (largest upper 95% limit 0.00079) |

At 10⁵ this is 0.19% of the exact conditional entropy (0.386). The text now distinguishes accuracy from probability quality.

**2. Integrability.** Proposition 2 now assumes finite expected log losses against both Y and Ỹ (for example p̂ ∈ [ε, 1 − ε]), and states that all expectations are conditional on the training data.

**3. "Achieved skill bounds knowability from below."** Replaced with the suggested wording.

**4. Hardware causal language.** The paper now says the split is consistent with prior work on defect-driven fluctuations, and that the experiment establishes differences in forecast performance, not mechanism. It also says weak coherence-time forecasts do not rule out exploitable persistence.

**5. Uncertainty limitations.** Acknowledged; the corresponding estimates are labelled exploratory throughout.

**6. Reproducibility and acknowledgment.** Same as Illusory Edge item 3.

**Wording fixes.**
- **"The quantity we measure is the Bayes ceiling"** → "the quantity of interest … computed exactly or approximately in controlled systems and bounded only indirectly, through achieved skill, on real tasks".
- **"Best available experiment"** → "the loophole-free experiment reanalysed here".
- **Audit theorem:** correct adjudication assumed, clean-label target stated, protocol summarised, and the tightened bound reported.
- **NWP:** replaced "almost all achieved skill" with the numbers: 0.297 alone, against 0.346 combined and 0.103 for the best observation-only model.
