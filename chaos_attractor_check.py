"""Attractor-aware check of the chaos ensemble reference in the regime used for the headline results (GPU).

Reference A (used in the study): particles drawn from N(z, sigma^2) around the observation z (local flat prior); they lie
off the attractor in the stable directions.
Reference B (attractor-aware): for each test state s0 we keep its own past state s(-tau), tau = 2 Lyapunov times, perturb it
by delta0 = 3 sigma exp(-lambda tau) and integrate forward to time 0. Stable directions contract, so these particles lie on
the attractor near s0, spread along the unstable direction; they are weighted by the observation likelihood N(z; x, sigma^2)
(self-normalised importance weights; effective sample size reported). B is not the exact posterior either (the proposal
density along the attractor is not the invariant density), but it removes the off-attractor mass that A contains.
Both are integrated to each horizon and compared: decision agreement, mean |eta_A - eta_B|, and the accuracy of each
decision rule against the true outcome. Observation noise is 1e-4 times each coordinate's standard deviation (as in the study).
Writes results/chaos_attractor_check.json.
"""
import json, math, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import run_law2 as L2
DEV, T64 = L2.DEV, torch.float64
N, P = 600, 2000
SETS = {"lorenz63": [0.5, 1.0, 2.0, 3.0, 5.0, 7.2], "rossler": [0.5, 1.0, 2.0, 3.0]}
out = {}
for name, Ls in SETS.items():
    cfg = L2.SYS[name]; f, dt = cfg["f"], cfg["dt"]
    lam = json.loads((ROOT / "results" / f"law2_{name}.json").read_text())["lambda"]
    g = torch.Generator(device=DEV).manual_seed(777)
    ref = L2.attractor(cfg, 200000, g)
    sd = ref.std(0); med = float(ref[:, 0].median()); sig = 1e-4 * sd
    tau_steps = int(round(2.0 / lam / dt))
    past = L2.attractor(cfg, N, g)
    s0 = L2.advance(f, past, tau_steps, dt)
    z = s0 + sig * torch.randn(s0.shape, device=DEV, dtype=T64, generator=g)
    A = (z[:, None, :] + sig * torch.randn((N, P, 3), device=DEV, dtype=T64, generator=g)).reshape(-1, 3)
    d0 = 3 * sig * math.exp(-2.0)
    B0 = (past[:, None, :] + d0 * torch.randn((N, P, 3), device=DEV, dtype=T64, generator=g)).reshape(-1, 3)
    B = L2.advance(f, B0, tau_steps, dt)
    lw = -0.5 * (((z[:, None, :] - B.reshape(N, P, 3)) / sig) ** 2).sum(-1)
    w = torch.softmax(lw, dim=1)
    ess = (1.0 / (w ** 2).sum(1)).cpu().numpy()
    truth, k = s0.clone(), 0
    res = {"sigma_rel": 1e-4, "n_test": N, "particles": P, "tau_lyapunov_times": 2.0, "median_ess": float(np.median(ess)),
           "share_ess_below_50": float((ess < 50).mean()), "per_L": {}}
    for L in Ls:
        steps = int(round(L / lam / dt)) - k
        A = L2.advance(f, A, steps, dt); B = L2.advance(f, B, steps, dt); truth = L2.advance(f, truth, steps, dt); k += steps
        y = (truth[:, 0] > med).cpu().numpy()
        eA = (A[:, 0] > med).double().reshape(N, P).mean(1).cpu().numpy()
        eB = ((B[:, 0] > med).double().reshape(N, P) * w).sum(1).cpu().numpy()
        ok = np.isfinite(eA) & np.isfinite(eB)
        dA, dB = eA >= 0.5, eB >= 0.5
        res["per_L"][str(L)] = {"decision_agreement": float((dA == dB)[ok].mean()), "mean_abs_eta_diff": float(np.abs(eA - eB)[ok].mean()),
                                "acc_A": float((dA == y)[ok].mean()), "acc_B": float((dB == y)[ok].mean()),
                                "acc_diff_A_minus_B": float((dA == y)[ok].mean() - (dB == y)[ok].mean())}
        print(name, L, res["per_L"][str(L)], flush=True)
    out[name] = res
    print(name, "median ESS", res["median_ess"], "share ESS<50", res["share_ess_below_50"], flush=True)
(ROOT / "results" / "chaos_attractor_check.json").write_text(json.dumps(out, indent=1))
