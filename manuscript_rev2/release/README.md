# Reproducibility package (revision 2)

Repository: git branch `revision-2` (commit 5272db4721b2); the original analyses are on `master` and are preserved.
Byte-identical reproduction of archived outputs validates consistency with the archive, not substantive correctness.

## Environment
`environment.lock.txt` (pip freeze), `environment.json` (Python, OS, GPU). Built with NVIDIA GeForce RTX 5070 Ti Laptop GPU, 621.14.

## Lightweight verification (CPU, < 10 min)
```
python -m pytest -q tests                      # closed forms, inflation identity, 1-NN asymptotics, audit coverage, simulator vs Qiskit
python rev2_analyses.py                        # audit bounds, power, fixed-cohort earnings, readout ceilings, time-block CIs
python make_tables.py && python make_macros_rev2.py   # regenerate every table and number (set KN_PAPER=paper_rev)
```

## Full compute (GPU recommended; about 3-4 h on an RTX 5070 Ti laptop GPU, 32 GB RAM)
```
python run_controls.py ; python run_noise_sim.py
python domains/build_<task>.py ; python run_domain.py <task>      # each real-data task (downloads; see data notes)
python run_d7.py                                                  # earnings (needs the user's archived package)
python run_law3.py ; python validate_posterior.py                 # chaos (GPU)
python make_quantum2.py ; python run_quantum2.py d1_qubit d2_lorenz d4_weather d5_football d6_house d7_earnings d8_stocks d9_quakes
python domains/qdrift.py primary ; python domains/qdrift.py secondary
python domains/bell_delft.py ; python run_d3.py
python run_ibm_real.py --dry-run                                  # live run needs an IBM Quantum account (free Open plan)
python fig_rev2.py ; python fig_hw.py ; python figs.py
```

## Data provenance and redistribution
`hashes.json` lists SHA-256 digests of every derived dataset, result file and cached raw file used.
- Meteostat bulk daily (CC BY-NC 4.0) and football-data.co.uk: fetched by the builders; redistribute derived results only.
- MIT Election Lab (CC0), USGS ComCat (public domain): redistributable.
- Yahoo Finance via yfinance: not redistributable; re-download with `build_d8_stocks.py` (survivorship-biased current constituents).
- SEC EDGAR and the earnings package: the package is the user's; EDGAR is public.
- IBM calibration drift: Hugging Face `phanerozoic/qiskit-calibration-drift` (CC BY 4.0); the replication dataset (GitHub HUANG-KUAN-HSIANG) has no licence and is analysed but not redistributed.
- Delft Bell-test data: 4TU.ResearchData DOI 10.4121/uuid:53644d31-d862-4f9f-9ad2-0b571874b829 (4TU terms of use); md5 recorded in results/bell_delft.json.
- Live IBM run: job dbnk4cc4148s73915tpg; circuits, layouts, counts and calibration snapshot in results/ibm_real_ibm_kingston_20261107.json (transpiled with optimization_level=1).

## Chaos numerical settings
RK4 with dt = 0.01 (Lorenz systems) or 0.02 (Roessler); invariant-measure samples from independent trajectories after burn-in plus random extra integration; observation noise 1e-4 x coordinate std; 500-particle Gaussian cloud centred on the observation (local flat-prior approximation, validated in validate_posterior.py); three seeds; settings recorded in each results/law3_<system>.json.

## Split and timestamp manifests
Decision and label-availability timestamps are columns `t` and `avail` of every data/*.parquet; walk-forward cutoffs are derived from each `*.meta.json`. The feature-and-timestamp specification and leakage audit are in docs/leakage_spec.md.

## Label provenance
data/d7_audit_results.csv (audit verdicts with evidence quotes), data/d7_before_after.parquet (`removed_by` per row), data/d7_README.md.
