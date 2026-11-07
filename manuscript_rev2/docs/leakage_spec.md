# Point-in-time specification and leakage audit (revision 2)

For every domain: decision time `t`, label availability `avail`, features and their timestamps, preprocessing scope, and the audit verdict. "Demonstrated leakage" means a feature or step verified to use information after `t`; "unresolved risk" means it could not be excluded with the available data.

## Shared pipeline (`kn/walk.py`, `run_domain.py`)
- Entity rates (M0, M1, M2): computed from labels with `avail < t` only (`entity_rates`, verified by unit test `test_walk_forward_no_future_labels`).
- Method-of-moments prior: re-estimated at each decision time from matured labels only.
- Feature models: refit at each cutoff on rows with `avail < cutoff`; winsorisation bounds, means and standard deviations computed on the training rows of that refit only (`_prep`).
- Base-rate forecast as an input to nested models: point-in-time M0 value (no future data).
- No hyper-parameter search on test periods; fixed settings.
- Quantum comparison (`run_quantum*.py`): standardisation, PCA and the [0, π] scaling fitted on the fit split only; model selection on the validation split (revision 2; the original selected on test — corrected).
- **Verdict:** no demonstrated leakage in the shared pipeline.

| Domain | t (decision) | avail (label) | Features (timestamp) | Verdict |
|---|---|---|---|---|
| Weather (obs) | end of day d | d + 2 days | day-d observations; past-only station-month anomaly | No demonstrated leakage. **Unresolved:** Meteostat bulk files may contain values revised after first publication. |
| Weather (+NWP) | end of day d | d + 2 days | Open-Meteo *previous-day1* forecast for d+1 (issued ≈24 h earlier per hour) | Calibration of the forecast-to-probability map used only the first 25% of rows, which were then dropped. **Unresolved:** exact model-run cadence of "best_match" undocumented. |
| Football | match date | +1 day | Elo, rest, form from earlier matches; closing odds as benchmark | Closing odds are set at kick-off; a forecast issued at kick-off may use them. **Unresolved:** kick-off time ignored; same-day matches share t. |
| House | 1 Nov of election year | +30 days | previous elections; midterm/president indicators; open seat from ballot names (filings precede the election) | Contested-seat selection uses the current ballot (known before the election). **Unresolved:** redistricting changes boundaries while keeping district ids; election-wide shocks are shared by all seats in a year (addressed by election-level resampling). |
| President | 1 Nov | +30 days | previous state and national margins; nominee home states | No demonstrated leakage; only 9 test elections. |
| Earnings | press-release time (+1 day for date-only filings) | first-reported actual (+24 h / +1 day) | guidance midpoint, structural diffusion score, issuer history | As in the package spec. **Unresolved:** revised vs original guidance (audit found revised ranges); labels machine-extracted. |
| Stocks | close of day d | d + 3 days | returns, volatility, volume ending d | **Unresolved risks:** survivorship (current S&P 100 members); split/dividend-adjusted prices revised retroactively (affects levels, negligible for returns). |
| Earthquakes | week start | +8 days | counts and magnitudes before t | Cell selection used 1990–2004 only. **Unresolved:** ComCat is the revised catalogue (magnitudes/locations not first-reported); completeness changes over time. |
| Hardware drift | calibration time of the target's preceding calibration | next calibration | lagged values, rolling statistics, environmental covariates at forecast time | Repeated 30-min polls of the same calibration deduplicated on (device, property, qubit(s), `calibrated_time`), so a calibration event appears once and cannot straddle a train/test boundary. Walk-forward by calendar day. **Unresolved:** publication delay between calibration and polling; unit change in May 2026 (handled by filtering). |
| Chaos controls | observation time | — | noisy observation of the full state | Ensemble particles are centred on the observation z (not the true state): verified in code (`parts = obs + sigma * noise`). |

## Demonstrated leakage
None found in the pipelines. One **evaluation** flaw (not leakage) was found and corrected: test-set selection of the "best" quantum and classical models.
