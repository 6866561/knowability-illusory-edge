"""Generic domain runner. Usage: python run_domain.py <name>

Reads ../data/<name>.parquet (schema in kn/walk.py) and ../data/<name>.meta.json:
  {"refit": "YS"|"MS"|"QS"|<int step for numeric t>, "xcols": [...], "bench": false,
   "leo_folds": 5, "min_train": 500, "reps": 1000, "lz": false, "start": "<first test time or null>"}
Writes results/<name>.json and results/<name>_preds.parquet.
"""
import json, sys, pathlib, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from kn import core, walk


def cutoffs(df, refit, start):
    t = df["t"]
    if np.issubdtype(t.dtype, np.datetime64):
        lo = pd.Timestamp(start) if start else t.min()
        return list(pd.date_range(lo.normalize(), t.max() + pd.Timedelta(days=1), freq=refit))
    step = float(refit)
    lo = float(start) if start is not None else float(t.min())
    return list(np.arange(lo, float(t.max()) + step, step))


def main(name):
    t0 = time.time()
    meta = json.loads((DATA / f"{name}.meta.json").read_text())
    df = pd.read_parquet(DATA / f"{name}.parquet")
    need = {"t", "avail", "entity", "y", "base_key"}
    assert need <= set(df.columns), f"missing {need - set(df.columns)}"
    assert (df["avail"] > df["t"]).all(), "label must become available after decision time"
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    xcols = meta.get("xcols", [])
    df = walk.entity_rates(df)
    cut = cutoffs(df, meta.get("refit", "YS"), meta.get("start"))
    df = walk.feature_models(df, cut, xcols, min_train=meta.get("min_train", 500),
                             bench=meta.get("bench", False), leo_folds=meta.get("leo_folds", 5))
    scored = df[df["p_M4"].notna()].copy()           # common evaluation set: rows every model can score
    models = ["p_M1", "p_M2", "p_M4", "p_M5", "p_M4x", "p_M4x_leo"] + (["bench", "p_M4b"] if meta.get("bench") else [])
    reps = meta.get("reps", 1000)
    summ = walk.summarize(scored, models, base="p_M0", reps=reps)
    ceil = walk.ceilings(scored)
    H = float(core.h(scored["y"].mean()))
    # information ladder (entropy shares from out-of-sample log losses; monotone by construction of nesting)
    ll0 = core.log_loss(scored["y"], scored["p_M0"])
    ladder = {"LL_M0": ll0, "LL_F1": core.log_loss(scored["y"], scored["p_M2"]),
              "LL_F2": core.log_loss(scored["y"], scored["p_M4"])}
    if meta.get("bench"):
        ladder["LL_F3"] = core.log_loss(scored["y"], scored["p_M4b"])
    # transfer ratio for feature-only model
    tau = None
    if scored["p_M4x_leo"].notna().any():
        ok = scored["p_M4x_leo"].notna()
        e_wf = core.accuracy(scored.loc[ok, "y"], scored.loc[ok, "p_M4x"]) - core.accuracy(scored.loc[ok, "y"], scored.loc[ok, "p_M0"])
        e_leo = core.accuracy(scored.loc[ok, "y"], scored.loc[ok, "p_M4x_leo"]) - core.accuracy(scored.loc[ok, "y"], scored.loc[ok, "p_M0"])
        tau = {"edge_wf": e_wf, "edge_leo": e_leo, "tau": e_leo / e_wf if abs(e_wf) > 1e-9 else None}
    lz = None
    if meta.get("lz"):
        rows = []
        for e, g in scored.sort_values("t").groupby("entity"):
            if len(g) >= 500:
                pi, S = core.lz_max_predictability(g["y"].to_numpy()[:3000])
                rows.append({"entity": str(e), "Pi_max": pi, "S": S,
                             "acc_M4": core.accuracy(g["y"], g["p_M4"])})
            if len(rows) >= 25:
                break
        lz = {"mean_Pi_max": float(np.mean([r["Pi_max"] for r in rows])) if rows else None,
              "mean_acc_M4": float(np.mean([r["acc_M4"] for r in rows])) if rows else None, "per_entity": rows}
    cov = {m: core.coverage_curve(scored["y"].to_numpy(), scored[m].to_numpy()) for m in ["p_M2", "p_M4"]}
    out = {"name": name, "n": int(len(scored)), "n_all": int(len(df)), "entities": int(scored["entity"].nunique()),
           "period": [str(scored["t"].min()), str(scored["t"].max())], "base_rate": float(scored["y"].mean()),
           "H_Y": H, "summary": summ, "ceilings": ceil, "ladder": ladder, "tau": tau, "lz": lz,
           "coverage": cov, "secs": time.time() - t0, "meta": meta}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    keep = ["t", "entity", "y"] + [c for c in df.columns if c.startswith("p_") or c in ("bench", "n_hist")]
    scored[keep].to_parquet(ROOT / "results" / f"{name}_preds.parquet")
    s = summ.get("p_M4", {})
    print(f"{name}: n={out['n']} ents={out['entities']} base={out['base_rate']:.3f} "
          f"acc0={s.get('acc0', np.nan):.4f} accM4={s.get('acc', np.nan):.4f} LSS_M4={s.get('LSS', np.nan):.4f} "
          f"LSS_M2={summ.get('p_M2', {}).get('LSS', np.nan):.4f} secs={out['secs']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
