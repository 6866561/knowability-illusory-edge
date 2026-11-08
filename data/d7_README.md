# d7: earnings-guidance (Abraham model package) standardized datasets

Built by `scratchpad/kn/build_d7.py` from a fresh rerun in `scratchpad/kn/pkg_rerun` (original in `kn/pkg/Abraham_model_package`, untouched).
Task: predict whether first-reported GAAP diluted EPS beats the guidance-range midpoint (y=1 if actual > midpoint + 1e-10).

## Reproduction status
Scripts rerun in documented order: abraham_model, run_experts, robustness, eval_expanded, combined_backtest, rescale_gaap_rows, fix_labels,
charge_filings, text_signals, nongaap (`build eval`), abstention. All exited 0, no tracebacks.
Every *_report.json, forecasts_*.csv and experts_*.csv is byte-identical to the original (25 files). One order effect: nongaap.py's eval step
calls eval_expanded.evaluate() and overwrites forecasts_expanded_new_tickers.csv with a 165-row file; rerunning eval_expanded.py once more restores the
original 950-row file (identical). So the original was produced with eval_expanded last.

Deviations (all in the copy only):
1. `combined_backtest.py` and `text_signals.py` read raw HTML from data/edgar/docs (not extracted). Patched to read the cached text `data/edgar/text/{adsh}_{doc}.txt`
   (those files are `to_text()` output of the same HTML). Results are byte-identical, so equivalent.
2. `nongaap.py build eval` was run (no rebuild from text): a rebuild overwrites the cleaned nongaap_rows.csv with unfixed quarantine flags.
3. rescale_gaap_rows.py / fix_labels.py edit data/edgar/expanded_rows.csv and nongaap_rows.csv in place. The packaged files are already in final
   state; both scripts are idempotent, and the rerun files are byte-identical to the originals.
4. Skipped (no network / no raw HTML): harvest_guidance.py, fetch_parallel.py, extract_guidance.py harvest, nongaap.py cache, lock_forecasts.py,
   score_locked.py (never run). Prices/XBRL/submissions came from the packaged caches; no network request was needed (px file count unchanged, 338).
   locked/ unchanged (checked identical).

| Anchor | Reported | Rerun |
|---|---|---|
| 20-ticker walk-forward freq / eb_beta / eb_beta log loss | 72.3 / 73.5 / 0.558 | 72.34 / 73.52 / 0.5576 |
| unseen-ticker eb_beta | 74.1 | 74.13 (freq 72.79) |
| combined n | 972 | 972 (569 outcome groups, 110 issuers) |
| combined base / freq / eb_beta / combo / ctrl_lite | 66.0 / 66.0 / 66.2 / 66.8 / 67.1 | 66.02 / 66.02 / 66.23 / 66.81 / 67.07 |
| new tickers after cleaning freq / eb_beta (expanded_report, 587 versions, 82 tickers) | 63.4 / 62.7 | 63.36 / 62.71 (ctrl_lite 65.78) |
| new tickers before cleaning freq / eb_beta (98 tickers, 679 versions) | 60.7 / 63.5 | 60.66 / 63.54 (ctrl_lite 63.25) |

All anchors reproduce. No mismatches found.

## d7_combined.parquet (972 rows = forecasts_combined.csv, scored rows with n_groups>=20)
decision_time (t), entity (issuer ticker), y (label), weight, period_kind (J; 1=annual, 0=quarter), src (audited20 | new), label_available,
freq, eb_beta, combo, ctrl_lite, struct_t (probabilities), z, fund_ok, k_known, nhist, M (midpoint), P (prior close), h (horizon, years), W (range width), q,
plus outcome_group (`TICKER|fiscal_end|kind[...]`), fiscal_end, period_kind_name, pair_id, n_groups, p_struct, eb_kind.
Decision rule everywhere: predict 1 if p >= 0.5.
Weight = 1 / (number of rows in the CSV with the same outcome_group); this is exactly how the package scores (weighted accuracy / log loss).
Reproduces 66.02/66.23/66.81/67.07 exactly. Note: weights sum to 569 (outcome groups), not n.

## d7_audited_wf.parquet / d7_unseen.parquet (375 rows each)
forecasts_walk_forward.csv / forecasts_unseen_ticker_walk_forward.csv (20 audited tickers) joined with training_data outcome_group and the expert columns
(eb_beta, ew_freq, kalman_t, struct_t, bma, fixed_share, label_available from experts_*.csv). `scored` = not fallback (299 / 297 rows); `weight` is
defined for scored rows only (NaN otherwise). Columns: pair_id, issuer, t, label, annual, freq, n_groups, ctrl7, spec10, v2, v3, v4, fallback,
outcome_group, scored, weight, + experts. Weighted accuracy reproduced: wf freq 72.34, eb_beta 73.52, ctrl7 68.92, bma 70.53; unseen eb_beta 74.13.

## d7_before_after.parquet (822 rows: GAAP new-ticker rows present before cleaning)
Protocol = eval_expanded.build(new_only=True) + sweep + ctrl_lite. BEFORE = only the original |A-M| quarantine (`quarantine_v1`, MODEL_REPORT round 2);
AFTER = final quarantine (adds scale check, quarter-ordinal check, continuing-ops exclusion). Rows already quarantined by the |A-M| rule (v1) are in neither set.
Key: entity, decision_time, outcome_group, M. Suffix `_before` / `_after` columns hold each state's own sweep outputs (freq, eb_beta, eb_kind, ctrl_lite,
nhist, n_groups, weight, scored); after-values are NaN where the row was removed. Because both sweeps retrain on their own history, probabilities for a
kept row differ between before and after. `scored_*` = n_groups>=20; `weight_*` = scoring weight within that state.
`removed_by`: scale_check (42 rows; 33 scored-before), quarter_ordinal (12; 9), continuing_ops (44; 34), none (724 kept). Priority scale > ordinal > contin;
10 rows fail more than one test and are labelled by the first. Also in_before (always True), in_after, y, M, low, high, actual, fiscal_end, kind, adsh, doc.
Scored-before rows: base rate of removed groups is low (continuing_ops 26.5%, quarter_ordinal 22.2%, scale_check 54.5%) vs 59.0% for kept rows.
Recompute: weighted accuracy of freq/eb_beta over scored_before / scored_after reproduces 60.66/63.54 and 63.36/62.71.
Limit: the before state is a reconstruction from the quarantine_v1 column, validated against the reported anchors (matches to 0.1 points); nothing was
re-extracted from the raw filings.

## d7_harvested_audit_sample.csv (82 rows, 21 issuers)
Harvested (src=new) rows in d7_combined where eb_beta and freq make different 0/1 decisions. Columns: entity, decision_time, file_date (release date),
period_kind_name, fiscal_end, M, low, high, actual (first-reported EPS used), y, freq, eb_beta, decisions and correctness (eb_correct 42, freq_correct 40),
weight, outcome_group, adsh, doc, text_file (absolute path in pkg_rerun/data/edgar/text; identical copy exists in the original package), text_exists (all 82 True).
Open text_file, find the guidance range low-high, verify period, basis (GAAP, total vs continuing) and the actual (A = `actual`) before trusting y.

## Caveats
- The d7_combined `weight` and `outcome_group` weighting treat each outcome group equally; raw n=972.
- d4_*..d8_* parquets in this folder belong to other tasks and were not touched.
