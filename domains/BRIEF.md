# Domain agent brief (read fully)

Project: research paper measuring how predictable different kinds of future events are. You build ONE domain dataset and run the shared evaluator. Never invent or adjust numbers; report exactly what the scripts print. Be concise.

Paths (all under scratchpad/kn): repo/ (code, run from here), data/ (outputs), raw/<domain>/ (download cache).
Full scratchpad: C:/Users/johan/AppData/Local/Temp/claude/b7dc35862fdf9617/084886cb-1afb-424d-85d4-d856b33f3240/scratchpad/kn

Rules
- Do NOT edit repo/kn/*, repo/run_domain.py, repo/prereg/*. If you find a bug there, stop and report it.
- Write your builder as repo/domains/build_<name>.py. It must download (free, public sources only, no API keys, no accounts), cache raw files under raw/<domain>/, and be rerunnable from cache. Be polite: ≤ 2 requests/s, retries with backoff, a generic User-Agent like "knowability-research/1.0" (never an email).
- Output data/<name>.parquet with columns:
  t (datetime64, decision time), avail (datetime64, when the label becomes known; must be > t), entity (str), y (0/1 int), base_key (str; the base-rate group defined in your task), x_* feature columns (float; known strictly before or at t), optional bench (probability in (0,1), known at t).
- Output data/<name>.meta.json, e.g. {"refit":"YS","xcols":["x_a","x_b"],"bench":false,"leo_folds":5,"min_train":500,"reps":500,"lz":false,"start":"2008-01-01"}. "start" = first test cutoff (leave ≥ 3 years of training history before it). If bench=true every row must have a bench value (filter rows or make a separate dataset).
- Leakage checklist (verify each, state in report): every feature uses data ≤ t only; labels avail > t; no same-period aggregates; first-reported/raw values only; entity ids don't encode outcomes; no future-year normalisation.
- Then run: cd repo && python run_domain.py <name>   (one dataset at a time; the laptop is shared, so never run two heavy jobs at once; use background run + wait if > 2 min).
- Final report (≤ 30 lines): data source URLs + licence, n rows, entities, period, base rate, the printed summary line, LSS/accuracy for p_M2, p_M4, p_M5 (and bench/p_M4b if any) from results/<name>.json, ceilings (hindsight_entity_majority, beta_ceiling.A, cover_hart.A_upper, kappa_upper), tau, problems and how you solved them, and anything a reviewer would question.
