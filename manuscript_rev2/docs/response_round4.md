# Responses to review (round 4: Illusory Edge; round 1: Knowability)

## Illusory Edge

**1. Make −7.14 the primary audit bound.** Agreed. Excluding the unresolved row is justified only if its unresolvability is unrelated to its label, which we cannot justify.
- The primary analysis now counts that row as potentially wrong: 3 errors in 40, K_U = 12, W_(12) = 12.0, bound −7.14 points.
- −6.10 is reported as conditional on excluding the row.
- Changed in the abstract, the audit paragraph, Table I (primary row first), Table II, and Fig. 1(c) and its caption.

**2. Proposition 7 generalisation.** Agreed.
- The proposition now assumes noninformative weighting and a Beta rate law.
- A separate remark covers a general outcome-weighted rate law G:
  - only the oracle expectations E_G[max(p, 1 − p)] and E_G[H₂(p)] carry over;
  - the predictive probability is derived from G, m_k = ∫p^{k+1}(1−p)^{n−k} dG / ∫p^k(1−p)^{n−k} dG;
  - A₁*(n) follows from those predictive probabilities;
  - monotonicity and martingale convergence still hold.
- The conjugate formulas are no longer claimed outside the Beta case.

**3. Release the research package.** In progress.
- A public copy is prepared:
  - code, derived result JSONs, the 82-row audit sample with weights and outcome groups, the 40 verdicts with evidence;
  - scripts, the lockfile, hashes, a README and an MIT license;
  - local paths removed and no secrets.
- It waits only on the author's GitHub login. The papers will cite the repository URL and commit once it is pushed.

**Smaller.**
- **Power:** now labelled an equal-weight, independent-outcome McNemar approximation that is not justified for weighted, repeated rows. It is compared with the 569 distinct outcomes from 110 issuers, not with 972 rows, and the text states the study is underpowered by a wide margin.
- **LSS convergence:** convergence is now stated under i.i.d. sampling from a fixed distribution, or under stationarity and ergodicity with a fixed forecast. The text says it need not hold under drift or strong dependence (main text and appendix).

## How Much of the Future Is Knowable?

**1. Full measured-loss identity.** Eq. (2) now shows all four terms: E ℓ(p̂, Ỹ) = H(Y|F∞) + I(Y;Z|X) + E KL(η‖p̂) + M, with M = E[(Ỹ − Y) ln((1 − p̂)/p̂)].
- The appendix proof is extended.
- The text states that M is a signed distortion, with examples of both signs, unlike the three nonnegative components.
- The text states that the identity organises components but does not identify their values without further information (a known ceiling, an exact reference, audited labels). This also appears in the abstract and the introduction.

**2. Ontic interpretation.** "Ontic" is defined as irreducible under the stated quantum model and information access.
- Bell violations are described as ruling out locally causal completions under the tests' assumptions, not as proving indeterminism; Norsen (2011, arXiv:0707.0401) is cited.
- Categorical sentences in the abstract, introduction, perspectives, Table I, results, discussion and conclusion are made conditional.

**3. Inferring information limits from fitted models.** The four flagged claims are replaced with claims about the tested models and inputs:
- **Stocks:** "the tested models extracted no reliably positive skill under time-block resampling; this does not establish that the price history contains no information."
- **Quantum section:** the claim that the model term is small is removed. The text now says that the remaining model term is not identified because the ceilings are unknown.
- **Benchmarks:** they "carried most of the achieved skill", not "most of what is knowable".
- **Claimed vs knowable:** replaced by "claimed or expected skill exceeded achieved skill; we cannot say how far achieved skill falls short of what is knowable".
- Section headings and the conclusion were adjusted to match.

**4. Audit wording and bound.** The abstract and Table I say the audit "could not certify a positive edge". −7.14 is primary, as in the companion paper.

**5. The 0.490 value.** It is now called the finite-sample accuracy of the 400-particle ensemble rule, with a Wilson 95% interval over 1,500 test states:
- 0.490 (0.465–0.515), indistinguishable from a coin;
- 0.981 (0.972–0.987).

The text notes that population Bayes accuracy cannot fall below 0.5, and that the posterior approximation is not validated in this tiny-noise, long-horizon regime. The abstract, Table I, Fig. 1's axis labels and caption, and the discussion were updated.

**6. Hardware noise.** The universal claim is replaced with "the fitted readout channels and the depolarizing noise considered here lower the average ceiling relative to the ideal preparation ensemble". The reset-channel counterexample is now given in the text.

**7. Remaining issues.** The Proposition 4 fix is the same as Illusory Edge item 2. The release is the same as Illusory Edge item 3.
