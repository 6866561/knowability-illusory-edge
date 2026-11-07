"""Learnability-horizon scaling across chaotic systems (GPU).

Systems: Lorenz-63, Roessler, Lorenz-96 (N = 5, 6, 8; F = 8). For each system:
  1. largest Lyapunov exponent lambda (Benettin, float64) and correlation dimension D2 (Grassberger-Procaccia);
  2. horizons in Lyapunov times L = lambda*h in {0.25, ..., 3};
  3. physical ceiling: accuracy of a 500-particle ensemble decision rule on 20k test states;
  4. learnable ceiling: GPU 1-NN trained on n in {1e3, ..., 3e6} noisy observations;
  5. fits: gap ~ c n^-beta per horizon; n*(L) at gap = 0.05; ln n* = a + gamma_L * L.
Heuristic prediction (Section IV): gamma_L = D2.  Usage: python run_law2.py [system ...]
Writes results/law2_<system>.json.
"""
import json, math, sys, time, pathlib
import numpy as np
import torch
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent
DEV = torch.device("cuda")
T64 = torch.float64
SEED = 20261106


def l63(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return torch.stack([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z], 1)


def ross(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return torch.stack([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)], 1)


def l96(X, F=8.0):
    return (torch.roll(X, -1, 1) - torch.roll(X, 2, 1)) * torch.roll(X, 1, 1) - X + F


SYS = {
    "lorenz63": dict(f=l63, dim=3, dt=0.01, init=lambda n, g: torch.randn(n, 3, device=DEV, dtype=T64, generator=g) + torch.tensor([0, 0, 25.0], device=DEV, dtype=T64), burn=3000),
    "rossler": dict(f=ross, dim=3, dt=0.02, init=lambda n, g: torch.cat([torch.rand(n, 2, device=DEV, dtype=T64, generator=g) * 10 - 5, torch.rand(n, 1, device=DEV, dtype=T64, generator=g)], 1), burn=8000),
}
for N in (5, 6, 8):
    SYS[f"lorenz96_{N}"] = dict(f=l96, dim=N, dt=0.01, init=lambda n, g, N=N: 8.0 + 0.5 * torch.randn(n, N, device=DEV, dtype=T64, generator=g), burn=3000)


def rk4(f, X, dt):
    k1 = f(X); k2 = f(X + dt / 2 * k1); k3 = f(X + dt / 2 * k2); k4 = f(X + dt * k3)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def advance(f, X, steps, dt, chunk=1_500_000):
    out = []
    for i in range(0, len(X), chunk):
        Y = X[i:i + chunk].clone()
        for _ in range(steps):
            Y = rk4(f, Y, dt)
        out.append(Y)
    return torch.cat(out)


def attractor(cfg, n, g):
    X = cfg["init"](n, g)
    X = advance(cfg["f"], X, cfg["burn"], cfg["dt"])
    extra = torch.randint(0, 1000, (n,), device=DEV, generator=g)
    for s in range(1000):                       # decorrelate along the attractor
        m = extra > s
        X[m] = rk4(cfg["f"], X[m], cfg["dt"])
    ok = torch.isfinite(X).all(1)
    return X[ok]


def lyapunov(cfg, g, steps=60000, ens=64):
    """Benettin on an ensemble of 64 reference trajectories (float64); returns mean and sd."""
    f, dt = cfg["f"], cfg["dt"]
    X = attractor(cfg, ens, g)
    d0 = 1e-8
    V = torch.randn(X.shape, device=DEV, dtype=T64, generator=g); V = V / V.norm(dim=1, keepdim=True) * d0
    Y = X + V; s = torch.zeros(ens, device=DEV, dtype=T64)
    for _ in range(steps):
        X = rk4(f, X, dt); Y = rk4(f, Y, dt)
        d = (Y - X).norm(dim=1); s += torch.log(d / d0)
        Y = X + (Y - X) / d[:, None] * d0
    lam = (s / (steps * dt)).cpu().numpy()
    return float(lam.mean()), float(lam.std() / math.sqrt(ens))


def corr_dim(X, npts=20000, g=None):
    """Grassberger-Procaccia correlation dimension: slope of log C(r) over the central scaling decade."""
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))[:npts]
    P = X[idx].float()
    P = (P - P.mean(0)) / P.std(0)
    dmax = 2 * math.sqrt(P.shape[1])
    rs = torch.logspace(math.log10(dmax * 1e-3), math.log10(dmax * 0.5), 40, device=DEV)
    counts = torch.zeros(len(rs), device=DEV, dtype=T64)
    for i in range(0, npts, 2000):
        D = torch.cdist(P[i:i + 2000], P)
        for k, r in enumerate(rs):
            counts[k] += (D < r).sum() - min(2000, npts - i)     # exclude self-pairs
    C = (counts / (npts * (npts - 1))).cpu().numpy(); r = rs.cpu().numpy()
    ok = (C > 1e-6) & (C < 0.05)
    lr, lc = np.log(r[ok]), np.log(C[ok])
    # central window of the scaling region: use the middle 60% of valid points
    a, b = int(0.2 * len(lr)), int(0.8 * len(lr))
    fit = stats.linregress(lr[a:b], lc[a:b])
    return float(fit.slope), float(fit.stderr), int(b - a)


def nn1_predict(Xtr, ytr, Xte, chunk=512, trchunk=500_000):  # distance block <= 512 x 5e5 floats (~1 GB)
    """Exact 1-NN on GPU (float32)."""
    best_d = torch.full((len(Xte),), float("inf"), device=DEV); best_y = torch.zeros(len(Xte), device=DEV)
    for j in range(0, len(Xtr), trchunk):
        T = Xtr[j:j + trchunk]; yT = ytr[j:j + trchunk]
        for i in range(0, len(Xte), chunk):
            d = torch.cdist(Xte[i:i + chunk], T)
            m, a = d.min(1)
            upd = m < best_d[i:i + chunk]
            best_d[i:i + chunk] = torch.where(upd, m, best_d[i:i + chunk])
            best_y[i:i + chunk] = torch.where(upd, yT[a], best_y[i:i + chunk])
    return best_y


def run(name):
    t0 = time.time()
    cfg = SYS[name]; f, dt = cfg["f"], cfg["dt"]
    g = torch.Generator(device=DEV).manual_seed(SEED)
    lam, lam_se = lyapunov(cfg, g)
    print(f"[{name}] lambda = {lam:.4f} +- {lam_se:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]
    ntest, npart = 20000, 500
    X = attractor(cfg, max(ns) + ntest + 5000, g)
    D2, D2_se, D2_pts = corr_dim(X[-20000:])
    X = X[: max(ns) + ntest]
    scale = X.std(0)
    sig = 1e-4 * scale                                   # observation noise relative to the attractor's spread
    med0 = float(X[:, 0].median())
    print(f"[{name}] D2 = {D2:.3f} +- {D2_se:.3f}; n points {len(X)}  ({time.time() - t0:.0f}s)", flush=True)
    obs = X + sig * torch.randn(X.shape, device=DEV, dtype=T64, generator=g)
    te_true, tr_true = X[:ntest].clone(), X[ntest:].clone()
    Ote = ((obs[:ntest] - X.mean(0)) / scale).float(); Otr = ((obs[ntest:] - X.mean(0)) / scale).float()
    parts = obs[:ntest].repeat_interleave(npart, 0) + sig * torch.randn(ntest * npart, cfg["dim"], device=DEV, dtype=T64, generator=g)
    Ls = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    rows, k = [], 0
    for L in Ls:
        steps = int(round(L / lam / dt)) - k
        te_true = advance(f, te_true, steps, dt); tr_true = advance(f, tr_true, steps, dt); parts = advance(f, parts, steps, dt); k += steps
        y_te = (te_true[:, 0] > med0).float(); y_tr = (tr_true[:, 0] > med0).float()
        eta = (parts[:, 0] > med0).double().view(ntest, npart).mean(1)
        A_phys = float(((eta >= 0.5).float() == y_te).float().mean())
        for n in ns:
            yhat = nn1_predict(Otr[:n], y_tr[:n], Ote)
            A_nn = float((yhat == y_te).float().mean())
            rows.append({"L": L, "h": L / lam, "n": n, "A_phys": A_phys, "A_1nn": A_nn, "gap": A_phys - A_nn})
        print(f"[{name}] L={L} phys={A_phys:.4f} gaps={[round(r['gap'], 3) for r in rows if r['L'] == L]} ({time.time() - t0:.0f}s)", flush=True)
    # fits
    fits, pts = {}, []
    for L in Ls:
        rr = [r for r in rows if r["L"] == L]
        n = np.array([r["n"] for r in rr], float); gp = np.array([r["gap"] for r in rr])
        keep = gp > 0.003
        if keep.sum() >= 3:
            fit = stats.linregress(np.log(n[keep]), np.log(gp[keep]))
            fits[str(L)] = {"beta": -fit.slope, "logc": fit.intercept, "r2": fit.rvalue ** 2}
            # n* at gap 0.05: interpolate in log-log between measured points when bracketed, else extrapolate with the fit
            above = gp >= 0.05
            if above.any() and (~above).any() and np.argmax(~above) > 0:
                j = int(np.argmax(~above))
                x1, x2, y1, y2 = np.log(n[j - 1]), np.log(n[j]), np.log(gp[j - 1]), np.log(max(gp[j], 1e-6))
                ln_star = x1 + (np.log(0.05) - y1) * (x2 - x1) / (y2 - y1); how = "interpolated"
            elif fit.slope < 0:
                ln_star = (np.log(0.05) - fit.intercept) / fit.slope; how = "extrapolated"
            else:
                continue
            pts.append((L, float(ln_star), how))
    law = None
    meas = [p for p in pts if p[2] == "interpolated"]
    use = meas if len(meas) >= 3 else pts
    if len(use) >= 3:
        fit = stats.linregress([p[0] for p in use], [p[1] for p in use])
        tc = stats.t.ppf(0.975, len(use) - 2)
        law = {"gamma_L": fit.slope, "gamma_L_lo": fit.slope - tc * fit.stderr, "gamma_L_hi": fit.slope + tc * fit.stderr,
               "r2": fit.rvalue ** 2, "a": fit.intercept, "n_points": len(use), "points": use, "only_interpolated": len(meas) >= 3}
    betas = [v["beta"] for v in fits.values()]
    out = {"system": name, "lambda": lam, "lambda_se": lam_se, "D2": D2, "D2_se": D2_se, "dim": cfg["dim"], "sigma_rel": 1e-4,
           "rows": rows, "fits": fits, "law": law, "beta_mean": float(np.mean(betas)) if betas else None,
           "pred_gamma_L_D2": D2, "pred_gamma_L_inv_beta": 1 / float(np.mean(betas)) if betas else None, "secs": time.time() - t0}
    (ROOT / "results" / f"law2_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[{name}] DONE gamma_L={law and round(law['gamma_L'], 3)} CI=({law and round(law['gamma_L_lo'], 2)}, {law and round(law['gamma_L_hi'], 2)}) "
          f"vs D2={D2:.3f}, 1/beta={out['pred_gamma_L_inv_beta']}  secs={out['secs']:.0f}", flush=True)
    del parts, X, obs; torch.cuda.empty_cache()


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["lorenz63", "rossler", "lorenz96_5", "lorenz96_6", "lorenz96_8"]):
        run(s)
