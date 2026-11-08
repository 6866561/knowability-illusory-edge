# Two-way dependence (both papers)

Response to "incomplete dependence modelling" and "stronger validation of the empirical uncertainty".

**Method.** Achieved-skill intervals now include a two-way, entity × time-block pigeonhole bootstrap (Owen 2007):
- Entities and calendar blocks are resampled independently with replacement.
- Each observation is weighted by the product of its entity's and its block's resampling counts.
- The method is implemented on cell sums in `kn.core.pigeonhole_ratio_bootstrap` and computed in `rev2_analyses.py` (`two_way`).
- It is the primary interval in Table III (Illusory Edge) and Table IV (Knowability).

**Validation.** A new unit test simulates data with independent entity and time effects (30 × 30, 120 replications). Two-way intervals covered the mean in 97% of replications at nominal 95%, against 80% for entity-only resampling. All 16 tests pass.

**Results.** Two-way intervals are wider than both one-way intervals for every task:

| Task | LSS | Time-block CI | Two-way CI |
|---|---|---|---|
| Rain + NWP | 0.346 | 0.317–0.373 | 0.300–0.389 |
| President | 0.345 | 0.050–0.603 | 0.043–0.641 |
| House | 0.167 | 0.127–0.201 | 0.113–0.215 |
| Football + odds | 0.115 | 0.108–0.121 | 0.098–0.131 |
| Earnings | 0.063 | 0.032–0.094 | 0.003–0.128 |
| Rain | 0.057 | 0.055–0.060 | 0.048–0.067 |
| Earthquakes | 0.049 | 0.045–0.053 | 0.041–0.056 |
| Stocks | 0.0003 | −0.0007–0.0012 | −0.0007–0.0013 |

Every task except stocks stays above zero; earnings does so only narrowly. The hardware out-of-tolerance task has no saved row-level predictions, so it keeps qubit-only resampling, and the table flags this.

**Removed.** The κ_U (Hellman–Raviv) plug-in diagnostic column, which has no coverage guarantee, is dropped from the atlas tables to make room. Its values remain in the release.

**Remaining limits (stated in the threats sections).**
- Blocks are treated as exchangeable; serial dependence across blocks is not modelled.
- The quantum and hardware comparisons still use one-way resampling.
