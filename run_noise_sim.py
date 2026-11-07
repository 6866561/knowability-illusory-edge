"""Section 7.4 simulation: measured vs true edge of the empirical-Bayes entity model over the base rate
under independent vs entity-persistent label noise, plus disagreement-audit coverage.

Panel: E entities, T periods each, decisions at t = 0..T-1, label available at t + 1.
Writes results/noise_sim.json.
"""
import json, sys, pathlib, itertools, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kn import core, walk

E = 300
REPS = 40


def one(m, conc, pi, r, T, mode, rep):
    rng = np.random.default_rng(hash((m, conc, pi, r, T, mode, rep)) % 2**32)
    if np.isinf(conc):
        p = np.full(E, m)
    else:
        p = rng.beta(m * conc, (1 - m) * conc, E)
    ent = np.repeat(np.arange(E), T)
    t = np.tile(np.arange(T), E).astype(float)
    y = (rng.random(E * T) < p[ent]).astype(int)
    if mode == "persistent":
        bad = rng.random(E) < pi
        flip = bad[ent] & (y == 1) & (rng.random(E * T) < r)       # 1 -> 0 at rate r on corrupted entities
    else:  # independent noise with the same overall flip count
        rate = pi * r * m
        flip = rng.random(E * T) < rate
    yt = np.where(flip, 1 - y, y)
    df = pd.DataFrame({"t": t, "avail": t + 1, "entity": ent, "y": yt, "base_key": 0, "ytrue": y})
    out = walk.entity_rates(df)
    burn = out["t"] >= 2
    o = out[burn]
    pe, pb = o["p_M2"].to_numpy(), o["p_M0"].to_numpy()
    meas = core.accuracy(o["y"], pe) - core.accuracy(o["y"], pb)
    true = core.accuracy(o["ytrue"], pe) - core.accuracy(o["ytrue"], pb)
    d = core.disagreement_decomposition(o["y"].to_numpy(), pe, pb)
    D = np.flatnonzero(d["mask"])
    cover = np.nan
    if len(D) >= 10:
        k = min(50, len(D))
        s = rng.choice(D, k, replace=False)
        e = int(np.sum(o["y"].to_numpy()[s] != o["ytrue"].to_numpy()[s]))
        lb = core.audit_lower_bound(meas, d["P_D"], e, k)["delta_lower"]
        cover = float(true >= lb - 1e-12)
    return dict(m=m, conc=conc, pi=pi, r=r, T=T, mode=mode, rep=rep, meas=meas, true=true, P_D=d["P_D"], cover=cover)


if __name__ == "__main__":
    t0 = time.time()
    grid = list(itertools.product([0.55, 0.65, 0.75], [2.0, 10.0, np.inf], [0.0, 0.05, 0.1, 0.2],
                                  [0.5, 1.0], [8, 32], ["persistent", "independent"], range(REPS)))
    grid = [g for g in grid if not (g[2] == 0 and (g[3] != 1.0 or g[5] != "persistent"))]
    res = Parallel(n_jobs=6, batch_size=16)(delayed(one)(*g) for g in grid)
    df = pd.DataFrame(res)
    agg = df.groupby(["m", "conc", "pi", "r", "T", "mode"]).agg(
        meas=("meas", "mean"), true=("true", "mean"), meas_sd=("meas", "std"),
        P_D=("P_D", "mean"), cover=("cover", "mean"), sign_flip=("meas", lambda s: np.nan)).reset_index()
    sf = df.assign(flip=(df.meas > 0) & (df.true < 0)).groupby(["m", "conc", "pi", "r", "T", "mode"])["flip"].mean()
    agg["sign_flip"] = sf.values
    agg["confirmatory"] = (agg.r == 1.0) & ((1 - agg.pi) * agg.m > 0.5) & (agg["mode"] == "persistent")
    (ROOT / "results").mkdir(exist_ok=True)
    agg.replace([np.inf], 1e9).to_json(ROOT / "results" / "noise_sim.json", orient="records", indent=1)
    c = agg[agg.confirmatory & (agg.pi > 0)]
    print("rows", len(df), "secs", round(time.time() - t0))
    print("confirmatory cells", len(c), "mean meas", c.meas.mean().round(4), "mean true", c.true.mean().round(4),
          "share meas>true", (c.meas > c.true).mean().round(3))
    ind = agg[(agg["mode"] == "independent") & (agg.pi > 0)]
    print("independent: mean meas", ind.meas.mean().round(4), "mean true", ind.true.mean().round(4))
    print("audit coverage overall", agg.cover.mean().round(3), "min", agg.cover.min().round(3))
    print(agg[(agg.conc == 1e9) | (agg.conc == np.inf)].query("T==32 and r==1").to_string(max_rows=40))
