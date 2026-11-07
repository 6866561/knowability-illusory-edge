"""Validate the ensemble posterior used for the chaos reference (Lorenz-63).

(1) Prior check at moderate noise: compare the Gaussian-cloud scheme (particles ~ N(z, sigma^2), i.e. a local flat prior)
    with an importance-sampling posterior p(s|z) ∝ N(z; s, sigma^2) mu(ds), where mu is represented by a library of
    2e7 independent samples from the invariant measure (each library point evolved to the horizon).
(2) Convergence at the scaling-study noise (sigma = 1e-4 x std): particle count 250/500/1000/2000 and step size dt vs dt/2.
Writes results/posterior_validation.json.
"""
import json, math, sys, time, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_law2 as L2

DEV, T64 = L2.DEV, torch.float64
cfg = L2.SYS["lorenz63"]; f, dt = cfg["f"], cfg["dt"]
LAM = 0.9046
LS = [0.5, 1.0, 2.0]


def evolve_labels(X, med, dtv=dt):
    labs, k = {}, 0
    for Lh in LS:
        steps = int(round(Lh / LAM / dtv)) - k
        X = L2.advance(f, X, steps, dtv); k += steps
        labs[Lh] = (X[:, 0] > med)
    return labs


def main():
    t0 = time.time(); g = torch.Generator(device=DEV).manual_seed(4242)
    out = {"prior_check": [], "convergence": {}}
    lib = L2.attractor(cfg, 20_000_000, g)
    med = float(lib[:20000, 0].median())
    lib0 = lib.float().clone()
    lab_lib = {k: v.to(torch.uint8) for k, v in evolve_labels(lib, med).items()}
    del lib; torch.cuda.empty_cache()
    print("library ready", time.time() - t0, flush=True)
    test = L2.attractor(cfg, 1000, g)
    for sig in (0.2, 0.5):
        z = test + sig * torch.randn(test.shape, device=DEV, dtype=T64, generator=g)
        zf = z.float()
        sw = torch.zeros(len(z), device=DEV, dtype=T64); sw2 = torch.zeros_like(sw)
        swl = {Lh: torch.zeros_like(sw) for Lh in LS}
        for i in range(0, len(z), 100):
            for j in range(0, len(lib0), 2_000_000):
                d2 = torch.cdist(zf[i:i + 100], lib0[j:j + 2_000_000]) ** 2
                w = torch.exp(-d2.double() / (2 * sig ** 2)) * (d2 < (4 * sig) ** 2)
                sw[i:i + 100] += w.sum(1); sw2[i:i + 100] += (w ** 2).sum(1)
                for Lh in LS:
                    swl[Lh][i:i + 100] += (w * lab_lib[Lh][j:j + 2_000_000].double()[None, :]).sum(1)
        ess = (sw ** 2 / sw2.clamp_min(1e-300)).cpu().numpy()
        parts = z.repeat_interleave(2000, 0) + sig * torch.randn(len(z) * 2000, 3, device=DEV, dtype=T64, generator=g)
        lab_p = evolve_labels(parts, med); y = evolve_labels(test.clone(), med)
        for Lh in LS:
            e_lib = (swl[Lh] / sw.clamp_min(1e-300)).cpu().numpy(); e_g = lab_p[Lh].double().view(len(z), 2000).mean(1).cpu().numpy()
            yy = y[Lh].cpu().numpy().astype(int); okm = ess >= 30
            r_l, r_g = np.minimum(e_lib, 1 - e_lib), np.minimum(e_g, 1 - e_g)
            out["prior_check"].append({"sigma": sig, "L": Lh, "median_ess": float(np.median(ess)), "share_ess_ge_30": float(okm.mean()),
                "mean_abs_eta_diff": float(np.mean(np.abs(e_lib - e_g)[okm])), "decision_agreement": float(np.mean(((e_lib >= .5) == (e_g >= .5))[okm])),
                "A_rule_library": float(np.mean(((e_lib >= .5).astype(int) == yy)[okm])), "A_rule_gauss": float(np.mean(((e_g >= .5).astype(int) == yy)[okm])),
                "floor_library": float(np.mean((r_l * (1 - 2 * r_l))[okm])), "floor_gauss": float(np.mean((r_g * (1 - 2 * r_g))[okm]))})
            print(out["prior_check"][-1], flush=True)
        del parts; torch.cuda.empty_cache()
    # convergence at sigma_rel = 1e-4
    scale = test.std(0); sig = 1e-4 * scale
    z = test + sig * torch.randn(test.shape, device=DEV, dtype=T64, generator=g)
    yv = {k: v.cpu().numpy().astype(int) for k, v in evolve_labels(test.clone(), med).items()}
    ref = None
    res = {}
    for npart in (250, 500, 1000, 2000):
        parts = z.repeat_interleave(npart, 0) + sig * torch.randn(len(z) * npart, 3, device=DEV, dtype=T64, generator=g)
        lp = evolve_labels(parts, med)
        res[npart] = {Lh: lp[Lh].double().view(len(z), npart).mean(1).cpu().numpy() for Lh in LS}
        del parts
    parts = z.repeat_interleave(1000, 0) + sig * torch.randn(len(z) * 1000, 3, device=DEV, dtype=T64, generator=g)
    lp_half = evolve_labels(parts, med, dtv=dt / 2)
    eta_half = {Lh: lp_half[Lh].double().view(len(z), 1000).mean(1).cpu().numpy() for Lh in LS}
    for Lh in LS:
        e_ref = res[2000][Lh]
        row = {}
        for npart in (250, 500, 1000):
            e = res[npart][Lh]; r = np.minimum(e, 1 - e); rr = np.minimum(e_ref, 1 - e_ref)
            row[str(npart)] = {"mean_abs_eta_diff_vs_2000": float(np.mean(np.abs(e - e_ref))),
                               "A_rule": float(np.mean((e >= .5).astype(int) == yv[Lh])), "floor": float(np.mean(r * (1 - 2 * r)))}
        rr = np.minimum(e_ref, 1 - e_ref)
        row["2000"] = {"A_rule": float(np.mean((e_ref >= .5).astype(int) == yv[Lh])), "floor": float(np.mean(rr * (1 - 2 * rr)))}
        eh = eta_half[Lh]; e1 = res[1000][Lh]
        row["dt_half_vs_dt"] = {"mean_abs_eta_diff": float(np.mean(np.abs(eh - e1))), "decision_agreement": float(np.mean((eh >= .5) == (e1 >= .5)))}
        out["convergence"][str(Lh)] = row
        print(Lh, row, flush=True)
    out["secs"] = time.time() - t0
    (ROOT / "results" / "posterior_validation.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
