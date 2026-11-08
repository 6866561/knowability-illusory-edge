"""Coverage suite for the two inference procedures used in both papers (reviewer items: dependence-method validation and
audit-implementation validation). Writes results/coverage_suite.json.

Part A  Two-way (entity x time block) pigeonhole bootstrap for the log-loss skill ratio LSS = 1 - E l(p_hat) / E l(p0),
        against entity-only and time-only cluster bootstraps. Data: binary outcomes on an entity x block panel with
        entity and calendar shocks; the target is the population (superpopulation) LSS, computed by large Monte Carlo.
        Scenarios: independent shocks; AR(1) calendar shocks (rho = 0.7); highly unequal weights; 30% missing cells;
        few time clusters (8 blocks). 1,000 simulated panels per scenario, 299 bootstrap replicates, 95% percentile
        intervals; Monte Carlo standard errors reported.
Part B  Disagreement-audit lower bound. (i) Exact coverage of the hypergeometric upper bound K_U (no simulation):
        min over K of P_K(K <= K_U) for N_D = 82, n = 40. (ii) Monte Carlo coverage of the implemented weighted bound,
        plain and tightened, with the actual audit weights or highly unequal synthetic weights, errors concentrated on
        heavy rows or on one issuer, all errors adverse, unresolved sampled rows counted as wrong or (incorrectly)
        excluded with informative missingness, and adjudication that misses a share of errors. 20,000 draws each.
"""
import json, math, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent; K = ROOT.parent
import sys as _sys; _sys.path.insert(0, str(ROOT)); from kn.paths import DATA, RAW
sys.path.insert(0, str(ROOT))
from kn.core import pigeonhole_ratio_bootstrap

rng0 = np.random.default_rng(20261108)
sig = lambda z: 1 / (1 + np.exp(-z))


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------ Part A
BETA, SA, SB, C0 = 1.0, 0.6, 0.6, -0.3


def draw_panel(rng, E, T, m, rho=0.0, missing=0.0, wsd=0.0):
    a = rng.normal(0, SA, E)
    if rho > 0:
        b = np.empty(T); b[0] = rng.normal(0, SB)
        for t in range(1, T):
            b[t] = rho * b[t - 1] + rng.normal(0, SB * math.sqrt(1 - rho ** 2))
    else:
        b = rng.normal(0, SB, T)
    ei, ti = np.meshgrid(np.arange(E), np.arange(T), indexing="ij"); ei, ti = ei.ravel(), ti.ravel()
    keep = rng.random(E * T) >= missing
    ei, ti = np.repeat(ei[keep], m), np.repeat(ti[keep], m)
    x = rng.normal(0, 1, len(ei))
    y = (rng.random(len(ei)) < sig(C0 + BETA * x + a[ei] + b[ti])).astype(float)
    ph, p0 = sig(C0 + BETA * x), np.full(len(ei), sig(C0))       # model ignores shocks; base rate fixed
    w = rng.lognormal(0, wsd, E)[ei] if wsd > 0 else np.ones(len(ei))   # entity-level weights
    return ei, ti, y, ph, p0, w


def true_lss(wsd=0.0, n=8_000_000):
    rng = np.random.default_rng(7)
    a = rng.normal(0, SA, n); b = rng.normal(0, SB, n); x = rng.normal(0, 1, n)
    y = (rng.random(n) < sig(C0 + BETA * x + a + b)).astype(float)
    w = rng.lognormal(0, wsd, n) if wsd > 0 else np.ones(n)
    return 1 - np.sum(w * ll(y, sig(C0 + BETA * x))) / np.sum(w * ll(y, np.full(n, sig(C0))))


def oneway(codes, num, den, reps, seed):
    G = int(codes.max()) + 1
    Nn = np.bincount(codes, num, G); Dd = np.bincount(codes, den, G)
    rng = np.random.default_rng(seed); out = np.empty(reps)
    for k in range(reps):
        c = rng.multinomial(G, np.full(G, 1.0 / G)); out[k] = (c @ Nn) / (c @ Dd)
    return out


SCEN_A = {"independent": dict(E=60, T=60, m=3), "ar1_calendar_rho0.7": dict(E=60, T=60, m=3, rho=0.7),
          "unequal_weights_lognormal_sd1": dict(E=60, T=60, m=3, wsd=1.0), "missing_30pct_cells": dict(E=60, T=60, m=3, missing=0.3),
          "few_time_clusters_T8": dict(E=60, T=8, m=20)}
SIMS, B = 1000, 299
resA = {}
for name, kw in SCEN_A.items():
    tgt = true_lss(kw.get("wsd", 0.0))
    hit = {"two_way": 0, "entity_only": 0, "time_only": 0}; width = {k: [] for k in hit}
    for s in range(SIMS):
        rng = np.random.default_rng(1_000_000 + s)
        ei, ti, y, ph, p0, w = draw_panel(rng, **kw)
        num, den = w * ll(y, ph), w * ll(y, p0)
        reps = {"two_way": pigeonhole_ratio_bootstrap(ei, ti, num, den, reps=B, seed=s),
                "entity_only": oneway(ei, num, den, B, s), "time_only": oneway(ti, num, den, B, s)}
        for k, r in reps.items():
            lo, hi = np.percentile(1 - r, [2.5, 97.5]); hit[k] += lo <= tgt <= hi; width[k].append(hi - lo)
    resA[name] = {"true_LSS": float(tgt), **{k: {"coverage": hit[k] / SIMS, "mc_se": math.sqrt(hit[k] / SIMS * (1 - hit[k] / SIMS) / SIMS),
                                                 "median_width": float(np.median(width[k]))} for k in hit}}
    print(name, {k: round(v["coverage"], 3) for k, v in resA[name].items() if isinstance(v, dict)}, flush=True)

# ------------------------------------------------------------------ Part B
ND, NS = 82, 40


def k_upper(N, n, e, alpha=0.05):
    Ku = e
    for Kk in range(e, N + 1):
        if stats.hypergeom.cdf(e, N, Kk, n) > alpha:
            Ku = Kk
        else:
            break
    return Ku


KU = {e: k_upper(ND, NS, e) for e in range(NS + 1)}
exact = []
for Ktrue in range(ND + 1):
    pe = stats.hypergeom.pmf(np.arange(NS + 1), ND, Ktrue, NS)
    exact.append(float(sum(p for e, p in enumerate(pe) if KU[e] >= Ktrue)))
exact_min = min(exact)

samp = pd.read_csv(DATA / "d7_harvested_audit_sample.csv")
w_real = samp["weight"].to_numpy(); W_TOTAL = 383.0
issuer = pd.factorize(samp["entity"])[0]


def audit_mc(weights, Ktrue, placement, unresolved, miss_rate=0.0, p_unres_wrong=0.0, p_unres_right=0.0, draws=20000, seed=0):
    rng = np.random.default_rng(seed)
    order_w = np.argsort(-weights)
    cov_plain = cov_tight = 0
    for _ in range(draws):
        if placement == "heaviest":
            wrong = np.zeros(ND, bool); wrong[order_w[:Ktrue]] = True
        elif placement == "one_issuer":
            big = np.bincount(issuer).argmax(); idx = np.where(issuer == big)[0]
            rest = np.setdiff1d(np.arange(ND), idx)
            pick = np.concatenate([idx, rng.permutation(rest)])[:Ktrue]
            wrong = np.zeros(ND, bool); wrong[pick] = True
        else:
            wrong = np.zeros(ND, bool); wrong[rng.choice(ND, Ktrue, replace=False)] = True
        true_shift = 2 * weights[wrong].sum() / W_TOTAL          # all errors adverse: true edge = measured - shift
        S = rng.choice(ND, NS, replace=False)
        found = wrong[S] & (rng.random(NS) >= miss_rate)          # adjudication may miss errors
        unres = np.where(wrong[S], rng.random(NS) < p_unres_wrong, rng.random(NS) < p_unres_right)
        if unresolved == "as_wrong":
            bad = found | unres; n_eff = NS
        else:                                                     # excluded: sample shrinks, errors among resolved only
            bad = found & ~unres; n_eff = NS - int(unres.sum())
        e = int(bad.sum())
        Ku = k_upper(ND, n_eff, e) if n_eff != NS else KU[e]
        plain = 2 * np.sort(weights)[::-1][:Ku].sum() / W_TOTAL
        unaud = np.setdiff1d(np.arange(ND), S)
        sbad = S[bad]
        tight = 2 * (weights[sbad].sum() + np.sort(weights[unaud])[::-1][:max(Ku - e, 0)].sum()) / W_TOTAL
        cov_plain += plain >= true_shift - 1e-12; cov_tight += tight >= true_shift - 1e-12
    cp, ct = cov_plain / draws, cov_tight / draws
    return {"coverage_plain": cp, "coverage_tight": ct, "mc_se": math.sqrt(max(min(cp, ct) * (1 - min(cp, ct)), 1e-12) / draws)}


w_unequal = np.random.default_rng(3).lognormal(0, 1.5, ND)
SCEN_B = {
    "actual_weights_random_errors": dict(weights=w_real, placement="random", unresolved="as_wrong"),
    "actual_weights_errors_on_heaviest": dict(weights=w_real, placement="heaviest", unresolved="as_wrong"),
    "actual_weights_errors_one_issuer": dict(weights=w_real, placement="one_issuer", unresolved="as_wrong"),
    "unequal_lognormal1.5_errors_on_heaviest": dict(weights=w_unequal, placement="heaviest", unresolved="as_wrong"),
    "unresolved_counted_wrong_informative": dict(weights=w_real, placement="random", unresolved="as_wrong", p_unres_wrong=0.5, p_unres_right=0.02),
    "unresolved_excluded_informative": dict(weights=w_real, placement="random", unresolved="exclude", p_unres_wrong=0.5, p_unres_right=0.02),
    "adjudicator_misses_20pct": dict(weights=w_real, placement="random", unresolved="as_wrong", miss_rate=0.2),
    "adjudicator_misses_50pct": dict(weights=w_real, placement="random", unresolved="as_wrong", miss_rate=0.5),
}
resB = {"exact_hypergeometric_min_coverage": exact_min, "exact_coverage_by_K": exact, "scenarios": {}}
for name, kw in SCEN_B.items():
    worst = None
    for Ktrue in (3, 8, 15, 25):
        r = audit_mc(Ktrue=Ktrue, seed=Ktrue, **kw)
        if worst is None or min(r["coverage_plain"], r["coverage_tight"]) < min(worst["coverage_plain"], worst["coverage_tight"]):
            worst = dict(r, K_true=Ktrue)
    resB["scenarios"][name] = worst
    print(name, worst, flush=True)

(ROOT / "results" / "coverage_suite.json").write_text(json.dumps({"part_A_two_way_LSS": resA, "part_B_audit": resB,
                                                                     "settings": {"sims_A": SIMS, "bootstrap_reps": B, "draws_B": 20000}}, indent=1))
print("exact min coverage", exact_min)
