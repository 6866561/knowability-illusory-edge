# Preregistration: How Much of the Future Is Knowable?

Locked 2026-11-07, before any data for domains D3, D4, D5, D6, D8 or D9 was downloaded or fitted.
D7 (earnings guidance) results were partly seen during the model's design (disclosed); its clean prospective check is the separate locked test scored after 1 August 2027.
D1 and D2 are simulated controls whose truth is known by construction.

## Hypotheses and falsification rules
- H1 (estimators): on controls, the best-model accuracy is within 0.02 of the true ceiling at n ≥ 10^4, and the Cover–Hart value is ≥ the true ceiling minus 0.01 (i.e. behaves as an upper bound) in at least 80% of control settings. Fails otherwise.
- H2 (ranking): Spearman correlation between the predicted and measured rank of LSS across the seven ranked rows below is ≥ 0.6. Fails otherwise.
- H3 (base rate): S_acc < 0.15 for earnings (D7) and stock direction (D8). Fails if either ≥ 0.15.
- H4 (illusory edge): in simulation, every confirmatory cell (r = 1, (1−π)m > 1/2, persistent noise, π > 0) has mean measured edge > mean true edge; independent-noise cells show |measured| ≤ |true| on average; audit-bound coverage ≥ 0.95 overall. In earnings, measured eb_beta edge over base on new tickers falls after label cleaning. Fails otherwise.
- H5 (quantum vs classical): no quantum model (VQC, quantum kernel, projected kernel) beats the best classical model on log loss with Holm-adjusted p < 0.05 in any domain. Fails if any does (reported as found).
- H6 (epistemic vs ontic): chaos ceiling at h = 1 rises toward 1 as σ_o → 10^-4; quantum full-θ ceiling stays at 1/2 + 1/π for all n.

## Primary models and metric
Primary metric for ranking: LSS of the primary model vs the domain M0 (log-loss skill score). Accuracy, S_acc, BSS also reported.
- D1 exact η; D2 particle-ensemble η; D3 exact η with fitted device noise.
- D4, D5, D6, D8, D9: M4 (L2 logistic, C = 0.5, on logit of the empirical-Bayes entity rate plus domain features).
- D7: combo (spec), with eb_beta as the entity-only rung.
- Benchmarks (F3): NWP (D4), betting odds (D5).
Hyper-parameters fixed as in `kn/walk.py` at the locking commit. No tuning on any test period.

## M0 per domain
D1–D3 pooled frequency; D4 station × month expanding climatology; D5 expanding league home-win share (football) / higher-rank-win share (tennis); D6 expanding pooled hold rate; D7 expanding pooled beat rate; D8 expanding pooled up-day share; D9 expanding per-cell weekly event share.

## Predicted LSS of the primary model (ranked rows)
| Rank | Domain / setting | Predicted LSS |
|---|---|---|
| 1 | D2 chaos, h = 0.25, σ_o = 1e-3 | 0.80 – 0.99 |
| 2 | D1 quantum, full θ | 0.443 (exact) |
| 3 | D6 contested US House seats | 0.30 – 0.60 |
| 4 | D4 rain tomorrow, observation features | 0.10 – 0.30 |
| 5 | D5 football home win | 0.03 – 0.12 |
| 6 | D7 earnings guidance | 0.02 – 0.08 |
| 7 | D8 stock direction | −0.005 – 0.005 |
Unranked: D3 quantum hardware/noise model 0.35 – 0.44; D4 with NWP benchmark 0.25 – 0.45; D9 earthquakes 0.05 – 0.30; D5 tennis 0.05 – 0.15; D6 president-by-state 0.4 – 0.8.

## Analysis plan
Walk-forward point-in-time evaluation; entity-clustered bootstrap (1,000–2,000 reps); half-split win rates; McNemar and Diebold–Mariano for paired comparisons; Holm across the quantum-vs-classical family; Benjamini–Hochberg for secondary analyses. All deviations logged with date and reason.
