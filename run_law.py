"""Learnability-horizon law: samples needed to approach the physical ceiling grow exponentially with horizon.

Grid h in {0.5, ..., 4}, n in {1e3 ... 3e6}; learner = histogram gradient boosting (fixed hyper-parameters) and 1-NN.
For each h: physical ceiling from a 1,000-particle ensemble on 20k test states. Then fit gap(n) = c * n^(-b) per h and
solve n*(h) for gap = eps (eps in {0.05, 0.02}); fit log n*(h) = a + gamma * h and compare gamma with the Lyapunov exponent.
Writes results/law.json.
"""
import json, sys, pathlib, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kn import core
import run_learnability as L

SEED = 20261106


def main():
    t0 = time.time()
    sigma, ntest, npart = 1e-3, 20000, 1000
    hs = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]
    gen = torch.Generator(device=L.DEV).manual_seed(SEED + 1)
    Xall = L.attractor(max(ns) + ntest)
    obs = Xall + sigma * torch.randn(Xall.shape, device=L.DEV, dtype=torch.float64, generator=gen)
    rows = []
    Xte = obs[:ntest].cpu().numpy(); Xo = obs[ntest:].cpu().numpy()
    test_true, tr_true = Xall[:ntest].clone(), Xall[ntest:].clone()
    parts = obs[:ntest].repeat_interleave(npart, 0) + sigma * torch.randn(ntest * npart, 3, device=L.DEV, dtype=torch.float64, generator=gen)
    k = 0
    for h in hs:
        steps = int(round(h / 0.01)) - k
        test_true = L.advance(test_true, steps); tr_true = L.advance(tr_true, steps); parts = L.advance(parts, steps); k += steps
        y_te = (test_true[:, 0] > 0).cpu().numpy().astype(int)
        y_tr = (tr_true[:, 0] > 0).cpu().numpy().astype(int)
        eta = (parts[:, 0] > 0).double().view(ntest, npart).mean(1).cpu().numpy()
        A_phys = core.accuracy(y_te, eta)
        for n in ns:
            hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED).fit(Xo[:n], y_tr[:n])
            a = core.accuracy(y_te, hgb.predict_proba(Xte)[:, 1])
            r_nn = 1 - float(np.mean(KNeighborsClassifier(1).fit(Xo[:n], y_tr[:n]).predict(Xte) == y_te)) if n <= 1000000 else None
            rows.append({"h": h, "n": n, "A_phys": A_phys, "A_hgb": a, "gap": A_phys - a, "r_nn": r_nn})
            print(f"h={h} n={n:>8} phys={A_phys:.4f} hgb={a:.4f} gap={A_phys - a:.4f} t={time.time() - t0:.0f}s", flush=True)
    # fits
    fits = {}
    for h in hs:
        r = [x for x in rows if x["h"] == h and x["gap"] > 0.002]
        if len(r) >= 3:
            b, logc = np.polyfit(np.log([x["n"] for x in r]), np.log([x["gap"] for x in r]), 1)
            fits[str(h)] = {"exponent": -b, "c": float(np.exp(logc)),
                            "n_star_0.05": float(np.exp((np.log(0.05) - logc) / b)) if b < 0 else None,
                            "n_star_0.02": float(np.exp((np.log(0.02) - logc) / b)) if b < 0 else None}
    law = {}
    for eps in ("0.05", "0.02"):
        pts = [(float(h), f[f"n_star_{eps}"]) for h, f in fits.items() if f[f"n_star_{eps}"]]
        if len(pts) >= 3:
            g, a = np.polyfit([p[0] for p in pts], np.log([p[1] for p in pts]), 1)
            law[eps] = {"gamma": float(g), "a": float(a), "points": pts}
    lyap = 0.906
    out = {"sigma": sigma, "rows": rows, "fits": fits, "law": law, "lyapunov_ref": lyap,
           "gamma_over_lambda": {k: v["gamma"] / lyap for k, v in law.items()}, "secs": time.time() - t0}
    (ROOT / "results" / "law.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"law": law, "ratio": out["gamma_over_lambda"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
