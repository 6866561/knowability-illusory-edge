# Round 9: remaining fixes (both papers)

Public repository: https://github.com/6866561/knowability-illusory-edge

**Hardware: shared device-time shocks.** `domains/qdrift.py` now saves row-level predictions, and `qdrift_twoway.py` resamples qubits and calendar blocks together, with block lengths of 1, 7, 14 and 28 days pooled over the three processors.
- **Model selection fixed.** The earlier analysis reported the better of ridge and gradient boosting on the test data, a mild selection. The pre-specified model is now gradient boosting, with ridge reported as a sensitivity.
- **Results**, with two-way intervals at 7-day blocks:

| Forecast | Skill (7-day CI) | 28-day CI |
|---|---|---|
| Readout error, 1 step ahead | 0.305 (0.251–0.360) | 0.231–0.369 |
| CZ error, 1 step ahead | 0.322 (0.262–0.382) | — |
| Out-of-tolerance task | 0.324 (0.274–0.370) | 0.271–0.375 |
| T₁ | 0.010, includes zero | — |
| T₂ | 0.044 (0.009–0.078) | includes zero |

- The readout, CZ and out-of-tolerance results are robust to block length.
- The T₂ interval includes zero with 28-day blocks, so T₂ is now described as not robust to the dependence model.
- The drift figure, macros and both papers are updated.

**Quantum comparison: two-way dependence.** A two-way (entity × month) bootstrap was added. Holm correction now uses the largest of the entity, time and two-way p-values.
- No quantum advantage in any domain; the smallest Holm-adjusted p is 1.000.
- Stocks: p is 0.0025 (entity), 0.072 (month block) and 0.134 (two-way).
- The re-run reproduced the earlier results to within 0.4% relative. The largest change was in a non-selected kernel model, from GPU and SVM nondeterminism.

**Decomposition demonstrations, verified separately.** `decomposition_check.py` and `tests/test_decomposition.py` compute everything exactly by quadrature on the qubit.
- **Conditional entropy:** matches 2 ln 2 − 1 to 1e-13.
- **Model term:** the identity holds exactly for a trained forecast, conditional on its training data (model term 0.0033 nats).
- **Epistemic term:** positive (0.0062).
- **Signed measurement term:** −0.026 when label errors agree with the forecast and +0.281 when they oppose it, each equal to the change in measured loss.
- 20 unit tests pass.

**Point-in-time audit, every domain.** `pit_audit.py` covers all 1,718,666 rows of the 7 task datasets:
- **Label timing:** 0 rows where the label is available before the decision time.
- **Entity rates:** permuting every label that becomes available after a cutoff changed none of the earlier entity-rate forecasts (largest change 0).
- **Model fits:** the same permutation changed none of the predictions of models refitted at that cutoff (largest change 0).
- **Source recomputation:** 300 earthquake rows and 300 stock rows recomputed from cached source files matched in features and labels, with 0 mismatches.
- **Earnings:** every guidance release precedes its decision, and every label arrives at least 50 days after it.
- **Not recomputed from source:** weather, football and election features. This is stated in the papers.

**Chaos reference in the regime used.** `chaos_attractor_check.py` compares the study's flat-prior reference with an attractor-aware reference: particles from the state's own past, integrated onto the attractor and weighted by the likelihood.
- **Up to 3 Lyapunov times**, the scaling-analysis horizons, the two agree in at least 99.7% of decisions in both systems, with mean probability differences at most 0.0021.
- **At 5 Lyapunov times**, agreement is 99.2%.
- **At 7.2 Lyapunov times**, agreement is only 91.8%. The flat-prior rule is the less accurate one there (0.890 vs 0.942), so the long-horizon accuracies in the knowability paper's Fig. 1 are conservative estimates, not bounds.
- The attractor-aware reference is not the exact posterior either.

**Still requires people or new data.**
- An independent reproduction by someone else.
- Human adjudication of all 40 audit rows.
- Expert review of the proofs and of the earnings definitions.
- A second earnings dataset or a locked forward evaluation.
- A DOI, which needs a Zenodo link to the GitHub account.
