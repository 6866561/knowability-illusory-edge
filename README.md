# Knowability and Illusory Edge: code, results and manuscripts

Code and derived results for two papers by Johan Abraham:

- **How Much of the Future Is Knowable? Information Ceilings, Chaos, Label Errors and Quantum Resources**: `manuscript_knowability/`
- **Illusory Edge: Persistent Label Errors, Disagreement Audits, and the Limits of Measured Predictability**: `manuscript_rev2/`

Branch `revision-2` holds the current analyses; `master` holds the original analyses, which are kept for comparison.

## Layout
- `kn/`: shared library (ceilings, skill scores, bootstraps, audit bounds, quantum simulator)
- `domains/`: one builder per forecasting task (downloads public data)
- `run_*.py`: studies. `run_law3.py` and `validate_posterior.py` (chaos), `rev2_analyses.py` (audit, earnings cohort, readout ceilings, time-block intervals), `run_quantum2.py` (validation-selected quantum comparison), `run_controls.py`
- `make_*.py`, `fig_*.py`: every table, figure and number in the papers (numbers are written to LaTeX macros; none are typed by hand)
- `results/`: derived results (JSON) that the manuscripts are built from
- `data/`: the earnings disagreement-audit sample (82 rows, with weights and outcome groups) and the 40 audit verdicts with evidence
- `tests/`: unit tests (`python -m pytest -q tests`)
- `manuscript_*/`: LaTeX sources, generated tables and macros, figures and PDFs; `manuscript_rev2/docs/` contains the issue ledger, the responses to review, the final assessment and the leakage specification; `manuscript_rev2/release/` contains the environment lockfile, file hashes and the table/figure-to-script manifest

## Reproducing
```
pip install -r manuscript_rev2/release/environment.lock.txt
python -m pytest -q tests
python rev2_analyses.py && python make_macros_rev2.py
```
The full compute steps, data sources and redistribution notes are in `manuscript_rev2/release/README.md`. Raw data whose licences forbid redistribution (e.g. Yahoo Finance prices, the unlicensed replication calibration dataset) are not included; the builders re-download them.

## License
Code: MIT (see `LICENSE`). Third-party data keep their own licences (listed in the release README).
