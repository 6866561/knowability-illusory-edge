"""Post-hoc (exploratory): learnability-horizon law with the 1-NN learner (no capacity limit)."""
import json, pathlib
import numpy as np
R = pathlib.Path(__file__).resolve().parent / "results" / "law.json"
d = json.loads(R.read_text())
rows = [r for r in d["rows"] if r["r_nn"] is not None]
hs = sorted({r["h"] for r in rows})
fits, pts = {}, {}
for eps in (0.05, 0.10):
    pts[eps] = []
for h in hs:
    rr = sorted([r for r in rows if r["h"] == h], key=lambda r: r["n"])
    n = np.array([r["n"] for r in rr], float); gap = np.array([r["A_phys"] - (1 - r["r_nn"]) for r in rr])
    keep = gap > 0.003
    if keep.sum() < 3:
        continue
    b, logc = np.polyfit(np.log(n[keep]), np.log(gap[keep]), 1)
    resid = np.log(gap[keep]) - (b * np.log(n[keep]) + logc)
    r2 = 1 - resid.var() / np.log(gap[keep]).var()
    fits[h] = {"exponent": float(-b), "c": float(np.exp(logc)), "r2_powerlaw": float(r2), "gaps": gap.tolist(), "n": n.tolist()}
    for eps in pts:
        pts[eps].append((h, float(np.exp((np.log(eps) - logc) / b))))
law = {}
for eps, P in pts.items():
    hh = np.array([p[0] for p in P]); ln = np.log([p[1] for p in P])
    g, a = np.polyfit(hh, ln, 1)
    res = ln - (g * hh + a); r2 = 1 - res.var() / ln.var()
    law[str(eps)] = {"gamma": float(g), "a": float(a), "r2": float(r2), "points": P}
lam = 0.906
d["nn_fits"] = {str(k): v for k, v in fits.items()}
d["nn_law_all"] = law
d["nn_law"] = {"eps": 0.05, "gamma": law["0.05"]["gamma"], "r2": law["0.05"]["r2"], "gamma_over_lambda": law["0.05"]["gamma"] / lam}
R.write_text(json.dumps(d, indent=1, default=float))
for h, f in fits.items():
    print(f"h={h}: gap ~ n^-{f['exponent']:.3f} (R2 {f['r2_powerlaw']:.3f}) gaps={np.round(f['gaps'], 3).tolist()}")
for e, l in law.items():
    print(f"eps={e}: log n* = {l['a']:.2f} + {l['gamma']:.3f} h  R2={l['r2']:.3f}  gamma/lambda={l['gamma'] / lam:.2f}  n*={[f'{p[0]}:{p[1]:.2e}' for p in l['points']]}")
