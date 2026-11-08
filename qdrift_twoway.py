"""Dependence-robust inference for the IBM calibration-drift forecasts (reviewer item: shared device-time shocks).

Uses the row-level predictions saved by domains/qdrift.py (results/qdrift_rows_<src>/). For each metric and horizon:
  * the pre-specified model is HGB (the earlier analysis reported the better of ridge and HGB on the test data, a mild
    selection; ridge is kept as a sensitivity);
  * MSE skill against each qubit's climatology, 1 - sum (y - yhat)^2 / sum (y - clim)^2;
  * 95% intervals from the qubit-cluster bootstrap (as before) and from the two-way (qubit x calendar block) pigeonhole
    bootstrap with calendar blocks of 1, 7, 14 and 28 days pooled over the three processors, so a recalibration or fault
    that moves many qubits on the same days is resampled as one unit.
For the out-of-tolerance task (taskB) the same is done for log-loss skill of HGB against the device base rate.
Writes results/qdrift_twoway.json.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent; RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from kn.core import pigeonhole_ratio_bootstrap

BLOCKS = (1, 7, 14, 28); REPS = 1000


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def intervals(cl, o, num, den):
    rc = pd.factorize(pd.Series(cl))[0]
    out = {"n_rows": int(len(num)), "n_qubits": int(rc.max() + 1), "point": float(1 - num.sum() / den.sum())}
    one = pigeonhole_ratio_bootstrap(rc, np.zeros_like(rc), num, den, reps=REPS, seed=1)     # columns collapsed: qubit-only
    out["ci_qubit"] = (1 - np.percentile(one, [97.5, 2.5])).tolist()
    for L in BLOCKS:
        cc = pd.factorize(pd.Series(o // L))[0]
        bs = pigeonhole_ratio_bootstrap(rc, cc, num, den, reps=REPS, seed=L)
        out[f"ci_two_way_{L}d"] = (1 - np.percentile(bs, [97.5, 2.5])).tolist(); out[f"n_blocks_{L}d"] = int(cc.max() + 1)
    return out


R = {}
for src in ("primary", "secondary"):
    d = RES / f"qdrift_rows_{src}"
    if not d.exists():
        continue
    R[src] = {"taskA": {}}
    for f in sorted(d.glob("taskA_*.npz")):
        z = np.load(f)   # plain numeric and unicode arrays only; no pickled objects
        key = f.stem.replace("taskA_", "")
        se_c = (z["clim"] - z["y"]) ** 2
        rec = {"hgb": intervals(z["cl"], z["o"], (z["hgb"] - z["y"]) ** 2, se_c),
               "ridge_point": float(1 - ((z["ridge"] - z["y"]) ** 2).sum() / se_c.sum()),
               "persistence_point": float(1 - ((z["pers"] - z["y"]) ** 2).sum() / se_c.sum())}
        rec["best_of_two_point"] = max(rec["hgb"]["point"], rec["ridge_point"])
        R[src]["taskA"][key] = rec
    fb = d / "taskB.npz"
    if fb.exists():
        z = np.load(fb)
        R[src]["taskB_hgb"] = intervals(z["cl"], z["o"], ll(z["y"], z["p_h"]), ll(z["y"], z["base"]))
(RES / "qdrift_twoway.json").write_text(json.dumps(R, indent=1))
for src, v in R.items():
    for k, r in v["taskA"].items():
        if k.endswith("_h1") or k.endswith("_h14"):
            h = r["hgb"]
            print(src, k, "hgb", round(h["point"], 3), "qubit", [round(x, 3) for x in h["ci_qubit"]], "2w-1d", [round(x, 3) for x in h["ci_two_way_1d"]],
                  "2w-7d", [round(x, 3) for x in h["ci_two_way_7d"]], "2w-28d", [round(x, 3) for x in h["ci_two_way_28d"]], "ridge", round(r["ridge_point"], 3))
    if "taskB_hgb" in v:
        b = v["taskB_hgb"]; print(src, "taskB", round(b["point"], 3), b["ci_qubit"], b["ci_two_way_7d"], b["ci_two_way_28d"])
