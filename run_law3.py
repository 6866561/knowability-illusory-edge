"""Revision-2 chaos learnability study (GPU). Replaces the single-seed, 1-NN-only analysis of run_law2.py
(whose outputs are preserved) with:
  * 3 independent seeds per system (independent attractor samples, observation noise, particles);
  * saved ensemble posteriors eta_hat for every test point (500 particles centred on the OBSERVATION z, i.e. a
    local flat-prior approximation to p(s | z); validated separately in validate_posterior.py);
  * two Bayes-accuracy estimators: realized accuracy of the ensemble decision rule on the test outcomes, and the
    plug-in E[max(eta_hat, 1-eta_hat)];
  * the asymptotic 1-NN floor E[r(1-2r)], r = min(eta, 1-eta) (Cover & Hart 1967: R_NN = E[2r(1-r)]);
  * two learners: 1-NN (not Bayes-consistent when 0 < r < 1/2) and k-NN with k_n = ceil(sqrt(n)) (consistent: k -> inf,
    k/n -> 0; Stone 1977), majority vote, ties -> 1;
  * learning-curve fits with and without an asymptotic floor; threshold-crossing classification
    (interpolated / extrapolated / unattainable / not identifiable) for eps in {0.02, 0.05, 0.10};
  * horizon regressions per seed on interpolated crossings only.
Usage: python run_law3.py [system ...]   -> results/law3_<system>.json
"""
import json, math, sys, time, pathlib
import numpy as np
import torch
from scipy import stats, optimize

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_law2 as L2

DEV = L2.DEV
T64 = torch.float64
NS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
LS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NTEST, NPART, SIG_REL = 10000, 500, 1e-4
EPSS = [0.02, 0.05, 0.10]


def knn_indices(Xtr, Xte, k, trchunk=400_000, techunk=512):
    """Exact k nearest neighbours (sorted) of each test row among training rows, on GPU (float32)."""
    nt = len(Xte)
    best_d = torch.full((nt, k), float("inf"), device=DEV)
    best_i = torch.zeros((nt, k), dtype=torch.long, device=DEV)
    for j in range(0, len(Xtr), trchunk):
        T = Xtr[j:j + trchunk]
        kk = min(k, len(T))
        for i in range(0, nt, techunk):
            d = torch.cdist(Xte[i:i + techunk], T)
            dv, di = torch.topk(d, kk, dim=1, largest=False)
            cd = torch.cat([best_d[i:i + techunk], dv], 1); ci = torch.cat([best_i[i:i + techunk], di + j], 1)
            v, o = torch.topk(cd, k, dim=1, largest=False)
            best_d[i:i + techunk] = v; best_i[i:i + techunk] = torch.gather(ci, 1, o)
    return best_i


def fit_curves(n, gap):
    """Zero-floor power law on positive gaps, and floor model g = g_inf + c n^-beta (g_inf >= 0). Returns dicts."""
    out = {}
    ok = gap > 0.002
    if ok.sum() >= 3:
        r = stats.linregress(np.log(n[ok]), np.log(gap[ok]))
        rss = float(np.sum((np.log(gap[ok]) - (r.intercept + r.slope * np.log(n[ok]))) ** 2))
        out["zero_floor"] = {"beta": -r.slope, "logc": r.intercept, "r2": r.rvalue ** 2, "rss_log": rss, "k": int(ok.sum())}
    try:
        f = lambda x, g0, lc, b: g0 + np.exp(lc) * x ** (-b)
        p, _ = optimize.curve_fit(f, n, gap, p0=[max(gap.min(), 1e-4), 0.0, 0.5],
                                  bounds=([0, -20, 0.01], [0.5, 20, 3.0]), maxfev=20000)
        res = gap - f(n, *p)
        out["floor"] = {"g_inf": float(p[0]), "logc": float(p[1]), "beta": float(p[2]), "rss": float(np.sum(res ** 2))}
        rss0 = float(np.sum((gap - np.exp(out["zero_floor"]["logc"]) * n ** (-out["zero_floor"]["beta"])) ** 2)) if "zero_floor" in out else None
        if rss0 is not None:
            m = len(n)
            aic = lambda rss, kpar: m * math.log(max(rss, 1e-30) / m) + 2 * kpar
            out["aic_zero_floor"] = aic(rss0, 2); out["aic_floor"] = aic(out["floor"]["rss"], 3)
    except Exception as e:
        out["floor_error"] = str(e)
    return out


def crossing(n, gap, fits, eps, floor_theory):
    """Classify how n*(eps) is obtained."""
    if floor_theory is not None and floor_theory >= eps:
        return {"status": "unattainable", "n_star": None}
    below = gap <= eps
    if below.any():
        j = int(np.argmax(below))
        if j == 0:
            return {"status": "below_first_n", "n_star": None}
        x1, x2 = math.log(n[j - 1]), math.log(n[j]); y1, y2 = math.log(gap[j - 1]), math.log(max(gap[j], 1e-6))
        return {"status": "interpolated", "n_star": float(math.exp(x1 + (math.log(eps) - y1) * (x2 - x1) / (y2 - y1)))}
    fl = fits.get("floor")
    if fl and fl["g_inf"] < eps:
        ns = math.exp((math.log(eps - fl["g_inf"]) - fl["logc"]) / (-fl["beta"]))
        if ns <= 30 * n[-1]:
            return {"status": "extrapolated", "n_star": float(ns)}
    return {"status": "not_identifiable", "n_star": None}


def run_seed(name, seed, lam):
    cfg = L2.SYS[name]; f, dt = cfg["f"], cfg["dt"]
    g = torch.Generator(device=DEV).manual_seed(1000 * seed + 7)
    X = L2.attractor(cfg, max(NS) + NTEST, g)
    scale = X.std(0); mu = X.mean(0); sig = SIG_REL * scale; med0 = float(X[:, 0].median())
    obs = X + sig * torch.randn(X.shape, device=DEV, dtype=T64, generator=g)
    Ote = ((obs[:NTEST] - mu) / scale).float(); Otr = ((obs[NTEST:] - mu) / scale).float()
    te, tr = X[:NTEST].clone(), X[NTEST:].clone()
    parts = obs[:NTEST].repeat_interleave(NPART, 0) + sig * torch.randn(NTEST * NPART, cfg["dim"], device=DEV, dtype=T64, generator=g)
    del X, obs
    ytr, yte, eta = {}, {}, {}
    k = 0
    for Lh in LS:
        steps = int(round(Lh / lam / dt)) - k
        te = L2.advance(f, te, steps, dt); tr = L2.advance(f, tr, steps, dt); parts = L2.advance(f, parts, steps, dt); k += steps
        ytr[Lh] = (tr[:, 0] > med0).to(torch.uint8); yte[Lh] = (te[:, 0] > med0).cpu().numpy().astype(int)
        eta[Lh] = (parts[:, 0] > med0).double().view(NTEST, NPART).mean(1).cpu().numpy()
    del parts, te, tr; torch.cuda.empty_cache()
    res = {"seed": seed, "per_L": {}}
    for Lh in LS:
        e = eta[Lh]; y = yte[Lh]; r = np.minimum(e, 1 - e)
        res["per_L"][str(Lh)] = {"A_rule": float(np.mean((e >= 0.5).astype(int) == y)), "A_plugin": float(np.mean(1 - r)),
                                 "R_star_plugin": float(np.mean(r)), "floor_1nn": float(np.mean(r * (1 - 2 * r))),
                                 "share_uncertain": float(np.mean((r > 0.02))), "acc_1nn": [], "acc_knn": []}
    for n in NS:
        kn = int(math.ceil(math.sqrt(n)))
        idx = knn_indices(Otr[:n], Ote, kn)
        for Lh in LS:
            lab = ytr[Lh][:n][idx].float()
            p1 = lab[:, 0].cpu().numpy().astype(int)
            pk = (lab.mean(1) >= 0.5).cpu().numpy().astype(int)
            y = yte[Lh]
            res["per_L"][str(Lh)]["acc_1nn"].append(float(np.mean(p1 == y)))
            res["per_L"][str(Lh)]["acc_knn"].append(float(np.mean(pk == y)))
        del idx; torch.cuda.empty_cache()
    for Lh in LS:
        d = res["per_L"][str(Lh)]; nn_ = np.array(NS, float)
        for lname in ("1nn", "knn"):
            gap = d["A_rule"] - np.array(d[f"acc_{lname}"])
            d[f"gap_{lname}"] = gap.tolist()
            fits = fit_curves(nn_, gap)
            d[f"fits_{lname}"] = fits
            d[f"cross_{lname}"] = {str(eps): crossing(nn_, gap, fits, eps, d["floor_1nn"] if lname == "1nn" else None) for eps in EPSS}
    return res


def horizon_fit(seeds, lname, eps, lam):
    """Per-seed OLS of ln n* on L using interpolated crossings only; plus exponential-vs-power comparison."""
    per = []
    for s in seeds:
        pts = [(float(Lh), s["per_L"][Lh][f"cross_{lname}"][str(eps)]) for Lh in s["per_L"]]
        use = [(Lh, c["n_star"]) for Lh, c in pts if c["status"] == "interpolated"]
        status = {Lh: c["status"] for Lh, c in pts}
        if len(use) >= 3:
            x = np.array([u[0] for u in use]); yv = np.log([u[1] for u in use])
            r = stats.linregress(x, yv); rp = stats.linregress(np.log(x), yv)
            per.append({"seed": s["seed"], "gamma_L": r.slope, "se": r.stderr, "r2_exp": r.rvalue ** 2, "r2_power": rp.rvalue ** 2,
                        "n_points": len(use), "L_range": [float(x.min()), float(x.max())], "status": status})
        else:
            per.append({"seed": s["seed"], "gamma_L": None, "n_points": len(use), "status": status})
    g = [p["gamma_L"] for p in per if p["gamma_L"] is not None]
    return {"per_seed": per, "gamma_mean": float(np.mean(g)) if g else None, "gamma_sd": float(np.std(g, ddof=1)) if len(g) > 1 else None,
            "n_seeds_identified": len(g)}


def corr_dim_sensitivity(name, g):
    cfg = L2.SYS[name]
    out = []
    for rep in range(2):
        X = L2.attractor(cfg, 40000, g)
        for (lo, hi) in [(0.2, 0.8), (0.1, 0.6), (0.3, 0.9)]:
            P = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(rep))[:20000]].float()
            P = (P - P.mean(0)) / P.std(0)
            dmax = 2 * math.sqrt(P.shape[1])
            rs = torch.logspace(math.log10(dmax * 1e-3), math.log10(dmax * 0.5), 40, device=DEV)
            cnt = torch.zeros(len(rs), device=DEV, dtype=T64)
            for i in range(0, 20000, 2000):
                D = torch.cdist(P[i:i + 2000], P)
                for kk, rr in enumerate(rs):
                    cnt[kk] += (D < rr).sum() - min(2000, 20000 - i)
            C = (cnt / (20000 * 19999)).cpu().numpy(); r = rs.cpu().numpy()
            ok = (C > 1e-6) & (C < 0.05); lr, lc = np.log(r[ok]), np.log(C[ok])
            a, b = int(lo * len(lr)), int(hi * len(lr))
            out.append({"rep": rep, "window": [lo, hi], "D2": float(stats.linregress(lr[a:b], lc[a:b]).slope)})
    v = [o["D2"] for o in out]
    return {"runs": out, "min": min(v), "max": max(v), "mean": float(np.mean(v))}


def run(name, n_seeds=3):
    t0 = time.time()
    old = json.loads((ROOT / "results" / f"law2_{name}.json").read_text())
    lam, lam_se = old["lambda"], old["lambda_se"]
    seeds = []
    for s in range(1, n_seeds + 1):
        seeds.append(run_seed(name, s, lam))
        print(f"[{name}] seed {s} done {time.time() - t0:.0f}s floors={[round(seeds[-1]['per_L'][str(l)]['floor_1nn'], 4) for l in LS]}", flush=True)
    d2s = corr_dim_sensitivity(name, torch.Generator(device=DEV).manual_seed(99))
    summary = {lname: {str(eps): horizon_fit(seeds, lname, eps, lam) for eps in EPSS} for lname in ("1nn", "knn")}
    out = {"system": name, "lambda": lam, "lambda_se": lam_se, "D2_law2": old["D2"], "D2_sensitivity": d2s,
           "settings": {"NS": NS, "LS": LS, "NTEST": NTEST, "NPART": NPART, "SIG_REL": SIG_REL, "knn_k": "ceil(sqrt(n))",
                        "posterior": "Gaussian cloud centred on observation (local flat prior)", "dt": L2.SYS[name]["dt"]},
           "seeds": seeds, "summary": summary, "secs": time.time() - t0}
    (ROOT / "results" / f"law3_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    s5 = summary["knn"]["0.05"]; s1 = summary["1nn"]["0.05"]
    print(f"[{name}] DONE knn gamma_L={s5['gamma_mean']} sd={s5['gamma_sd']} (seeds id {s5['n_seeds_identified']}); "
          f"1nn gamma_L={s1['gamma_mean']} (id {s1['n_seeds_identified']}); D2 range {d2s['min']:.2f}-{d2s['max']:.2f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["lorenz63", "rossler", "lorenz96_5", "lorenz96_6", "lorenz96_8"]):
        run(s)
