"""Uncertainty and sensitivity of the learnability-horizon rates from the many-seed chaos run (results/law3s20_<system>.json).

For each seed the threshold crossings n* are recomputed from the saved gap curves (interpolated crossings only, between
measured sample sizes) under: the full sample-size range; dropping the smallest n; dropping the largest n; and
interpolation linear in n instead of log-linear. The horizon slope gamma_L (OLS of ln n* on L, >= 3 crossings) is
computed per seed, and a 95% bootstrap interval over seeds is reported for each variant, learner and tolerance.
Writes results/law3s20_sens_<system>.json (flat macro-ready strings) and results/law3s20_sens_full.json.
"""
import json, math, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent; RES = ROOT / "results"


def crossings(NS, gap, eps, mode="loglog"):
    NS = np.asarray(NS, float); gap = np.asarray(gap, float)
    if gap[0] <= eps:
        return None, "below_first_n"
    for i in range(1, len(NS)):
        if gap[i] <= eps:
            g0, g1 = gap[i - 1], gap[i]
            if mode == "loglog" and g0 > 0 and g1 > 0:
                t = (math.log(eps) - math.log(g0)) / (math.log(g1) - math.log(g0))
                return math.exp(math.log(NS[i - 1]) + t * (math.log(NS[i]) - math.log(NS[i - 1]))), "interpolated"
            t = (eps - g0) / (g1 - g0)
            return NS[i - 1] + t * (NS[i] - NS[i - 1]), "interpolated"
    return None, "none"


def slope(Ls, ns):
    pts = [(L, math.log(n)) for L, n in zip(Ls, ns) if n is not None]
    if len(pts) < 3:
        return None
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


out_flat, full = {}, {}
for sysname, tag in (("lorenz63", "Lor"), ("rossler", "Ros")):
    p = RES / f"law3s20_{sysname}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text()); NS = d["settings"]["NS"]; Ls = [float(x) for x in d["settings"]["LS"]]
    full[sysname] = {}
    variants = {"full": (slice(None), "loglog"), "dropmin": (slice(1, None), "loglog"), "dropmax": (slice(None, -1), "loglog"), "linear": (slice(None), "linear")}
    for ln in ("1nn", "knn"):
        for eps in (0.02, 0.05, 0.10):
            for vname, (sl, mode) in variants.items():
                g = []
                for s in d["seeds"]:
                    ns = []
                    for L in Ls:
                        key = str(L) if str(L) in s["per_L"] else f"{L:g}"
                        gap = s["per_L"][key][f"gap_{ln}"]
                        n_star, st = crossings(np.asarray(NS)[sl], np.asarray(gap)[sl], eps, mode)
                        ns.append(n_star if st == "interpolated" else None)
                    sl_ = slope(Ls, ns)
                    if sl_ is not None:
                        g.append(sl_)
                rec = {"n_seeds_identified": len(g), "n_seeds": len(d["seeds"])}
                if len(g) >= 3:
                    rng = np.random.default_rng(11)
                    bs = [np.mean(rng.choice(g, len(g))) for _ in range(5000)]
                    rec.update(mean=float(np.mean(g)), sd=float(np.std(g, ddof=1)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
                full[sysname][f"{ln}|{eps}|{vname}"] = rec
                lt = {"1nn": "onenn", "knn": "knn"}[ln]; et = {0.02: "Two", 0.05: "Five", 0.10: "Ten"}[eps]
                vt = {"full": "Full", "dropmin": "Dmin", "dropmax": "Dmax", "linear": "Lin"}[vname]
                if "mean" in rec:
                    out_flat.setdefault(sysname, {})[f"S{lt}{et}{vt}"] = f"{rec['mean']:.2f}"
                    out_flat[sysname][f"S{lt}{et}{vt}lo"] = f"{rec['lo']:.2f}"; out_flat[sysname][f"S{lt}{et}{vt}hi"] = f"{rec['hi']:.2f}"
                out_flat.setdefault(sysname, {})[f"S{lt}{et}{vt}n"] = str(rec["n_seeds_identified"])
    (RES / f"law3s20_sens_{sysname}.json").write_text(json.dumps(out_flat.get(sysname, {}), indent=1))
(RES / "law3s20_sens_full.json").write_text(json.dumps(full, indent=1))
for sname, v in full.items():
    for k, r in v.items():
        if k.split("|")[1] == "0.05":
            print(sname, k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in r.items()})
