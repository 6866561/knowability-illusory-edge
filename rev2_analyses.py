"""Revision-2 analyses (no GPU): audit bounds, power, fixed-cohort earnings, asymmetric readout ceilings, time-block CIs.
Writes results/rev2_analyses.json. Original results files are not modified."""
import json, math, pathlib, sys
import numpy as np
import pandas as pd
from scipy import stats, integrate

ROOT = pathlib.Path(__file__).resolve().parent
RES, DATA = ROOT / "results", ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from kn import core
OUT = {}
rng = np.random.default_rng(20261106)


def hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ---------------------------------------------------------------- 1. disagreement audit, finite population
def hyper_upper(N, n, e, alpha=0.05):
    """Largest K (wrong labels in a population of N) with P(X <= e | N, K, n) > alpha (one-sided exact upper bound)."""
    K_U = e
    for K in range(e, N + 1):
        if stats.hypergeom.cdf(e, N, K, n) > alpha:
            K_U = K
        else:
            break
    return K_U


d7 = pd.read_parquet(DATA / "d7_combined.parquet").reset_index(drop=True)
hv = d7[d7["src"] != "audited20"].reset_index(drop=True)
y, w = hv["y"].to_numpy().astype(int), hv["weight"].to_numpy()
de, df_ = (hv["eb_beta"].to_numpy() >= .5).astype(int), (hv["freq"].to_numpy() >= .5).astype(int)
D = de != df_
N_D = int(D.sum()); PD_w = float(w[D].sum() / w.sum()); PD_u = float(D.mean())
delta = float(np.sum(w * ((de == y).astype(float) - (df_ == y).astype(float))) / w.sum())
aud = pd.read_csv(DATA / "d7_audit_results.csv")
scen = {}
for name, e, n in [("exclude_unverifiable", int((aud.verdict == "wrong").sum()), int((aud.verdict != "unverifiable").sum())),
                   ("unverifiable_as_wrong", int((aud.verdict != "correct").sum()), len(aud))]:
    K_U = hyper_upper(N_D, n, e)
    cp = core.clopper_pearson_upper(e, n)
    wD = np.sort(w[D])[::-1]
    worst_w = float(wD[:K_U].sum() / w.sum())          # heaviest K_U rows wrong (weighted worst case)
    scen[name] = {"e": e, "n": n, "N_D": N_D, "K_U_hypergeom": K_U, "rho_U_hypergeom": K_U / N_D, "rho_U_clopper_pearson": cp,
                  "W_top_KU": float(wD[:K_U].sum()),
                  "delta_lower_unweighted_hyper": delta - 2 * PD_u * K_U / N_D if False else None,
                  "delta_lower_weighted_hyper": delta - 2 * worst_w, "delta_lower_cp_PDw": delta - 2 * PD_w * cp}
OUT["audit"] = {"N_D_rows": N_D, "P_D_weighted": PD_w, "P_D_unweighted": PD_u, "delta_measured_weighted": delta,
                "accounting": {"n_combined_rows": len(d7), "n_audited20_rows": int((d7["src"] == "audited20").sum()),
                               "n_population_rows": len(hv), "n_population_issuers": int(hv["entity"].nunique()),
                               "W_total": float(w.sum()), "W_D": float(w[D].sum()), "w_min": float(w.min()), "w_max": float(w.max()),
                               "weight_definition": "1 / (number of rows forecasting the same outcome group); each outcome group carries total weight 1",
                               "population": "harvested (src=new) rows of the cleaned combined universe; the 340 previously hand-audited rows are excluded"},
                "sampling": "40 rows drawn uniformly without replacement from the 82 disagreement rows (numpy RandomState(20261106).choice(82, 40)); one unverifiable",
                "scenarios": scen,
                "note": "bound concerns the fixed finite population of harvested disagreement rows (labels as evaluated), not future predictive edge"}

# ---------------------------------------------------------------- 2. power
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
OUT["power"] = {"formula": "n = (z_{1-a/2}+z_{1-b})^2 * P(D) / delta^2 (McNemar, independent forecasts)",
                "alpha_two_sided": 0.05, "power": 0.8, "delta": 0.01, "P_D": PD_w, "n_required": z ** 2 * PD_w / 0.01 ** 2,
                "n_required_design_effect_2": 2 * z ** 2 * PD_w / 0.01 ** 2, "n_available": len(d7),
                "note": "independence across forecasts assumed; clustering by issuer and fiscal period raises the requirement by the design effect"}

# ---------------------------------------------------------------- 3. earnings fixed cohort
ba = pd.read_parquet(DATA / "d7_before_after.parquet")
coh = ba[(ba.scored_before == 1) & (ba.scored_after == 1) & (ba.removed_by == "none")].reset_index(drop=True)
rem = ba[(ba.scored_before == 1) & (ba.removed_by != "none")].reset_index(drop=True)


def edge(df, sfx, wcol):
    yy = df["y"].astype(int).to_numpy(); ww = df[wcol].to_numpy()
    ce = ((df[f"eb_beta_{sfx}"] >= .5).astype(int).to_numpy() == yy); cf = ((df[f"freq_{sfx}"] >= .5).astype(int).to_numpy() == yy)
    return float(np.sum(ww * (ce.astype(float) - cf)) / ww.sum())


def boot(df, fn, reps=2000):
    g = {k: np.asarray(v) for k, v in df.groupby("entity").indices.items()}
    b = core.cluster_bootstrap(g, lambda idx: fn(df.iloc[idx]), reps=reps)
    return np.percentile(b, [2.5, 97.5]).tolist()


OUT["earnings_cohort"] = {
    "fixed_cohort_rows": len(coh), "fixed_cohort_tickers": int(coh.entity.nunique()),
    "A_archived_before_sweep_on_cohort": edge(coh, "before", "weight_before"),
    "A_ci": boot(coh, lambda d: edge(d, "before", "weight_before")),
    "B_after_sweep_retrained_on_cohort": edge(coh, "after", "weight_before"),
    "B_ci": boot(coh, lambda d: edge(d, "after", "weight_before")),
    "B_minus_A_paired": edge(coh, "after", "weight_before") - edge(coh, "before", "weight_before"),
    "B_minus_A_ci": boot(coh, lambda d: edge(d, "after", "weight_before") - edge(d, "before", "weight_before")),
    "removed_rows": len(rem), "removed_rows_edge_archived_original_labels": edge(rem, "before", "weight_before"),
    "removed_by_counts": rem.removed_by.value_counts().to_dict(),
    "note": ("Cohort = rows scored in both sweeps and not removed: labels identical, so A vs B isolates the effect of "
             "correcting the training history (retraining). Removed rows were excluded, not relabelled: corrected labels "
             "(e.g. total-EPS guidance for continuing-operations rows) are not available, so 'scored against corrected labels' "
             "cannot be computed for them.")}

# ---------------------------------------------------------------- 4. asymmetric readout ceiling
def ceiling_asym(e0, e1):
    """theta ~ U(0, pi), p' = e0 + beta*sin^2(theta/2): optimal-threshold accuracy, base rate, kappa."""
    beta = 1 - e0 - e1; a = (e0 - e1) / 2; b = beta / 2
    if b <= abs(a):
        Aopt = 0.5 + abs(a)
    else:
        t0 = math.acos(a / b); Aopt = 0.5 + (beta * math.sin(t0) + a * (math.pi - 2 * t0)) / math.pi
    th = np.linspace(0, math.pi, 200001); pp = e0 + beta * np.sin(th / 2) ** 2
    A_num = float(np.mean(np.maximum(pp, 1 - pp)))
    A_fixed = float(np.mean(np.where(np.sin(th / 2) ** 2 >= 0.5, pp, 1 - pp)))   # threshold at the noiseless boundary theta = pi/2
    m = float(pp.mean())
    return {"A_opt": Aopt, "A_opt_numeric": A_num, "A_fixed_threshold": A_fixed, "symmetric_formula": 0.5 + beta / math.pi,
            "base_rate": m, "kappa": float(1 - hb(pp).mean() / hb(m))}


live = json.loads((RES / "ibm_real_ibm_kingston_20261107.json").read_text())
lv = {}
for qk in [k for k in live if k.startswith("qubit_")]:
    q = int(qk.split("_")[1])
    runs = [r for r in live["runs"] if r["kind"] == "ry" and r["qubit"] == q]
    th = np.array([r["theta"] for r in runs]); k1 = np.array([r["counts"].get("1", 0) for r in runs]); S = live["shots"]
    pid = np.sin(th / 2) ** 2
    beta_, alpha_ = np.polyfit(pid, k1 / S, 1); e0, e1 = alpha_, 1 - alpha_ - beta_
    base = ceiling_asym(e0, e1)
    bs = []
    for _ in range(2000):
        kb = rng.binomial(S, k1 / S); bb, aa = np.polyfit(pid, kb / S, 1); bs.append(ceiling_asym(aa, 1 - aa - bb)["A_opt"])
    lv[str(q)] = {"e0": e0, "e1": e1, **base, "A_opt_ci": np.percentile(bs, [2.5, 97.5]).tolist(),
                  "max_resid": float(np.max(np.abs(k1 / S - (alpha_ + beta_ * pid))))}
ib = json.loads((RES / "d3_ibm_noise.json").read_text())
nm = []
for b in ib["backends"]:
    for q in b["qubits"]:
        c = ceiling_asym(q["alpha_e0"], q["e1"])
        nm.append({"backend": b["backend"], "qubit": q["qubit"], "A_opt": c["A_opt"], "A_reported": q["A_dev"], "symmetric_formula": c["symmetric_formula"],
                   "abs_diff_formula": abs(c["A_opt"] - c["symmetric_formula"])})
OUT["readout"] = {"derivation": "A* = 1/2 + [beta sin(t0) + a (pi - 2 t0)]/pi, a = (e0-e1)/2, beta = 1-e0-e1, cos t0 = 2a/beta (|2a| < beta)",
                  "live": lv, "noise_models_max_abs_diff_formula_vs_opt": max(x["abs_diff_formula"] for x in nm),
                  "noise_models_max_abs_diff_reported_vs_opt": max(abs(x["A_opt"] - x["A_reported"]) for x in nm)}

# ---------------------------------------------------------------- 5. time-block bootstrap for atlas rows
blocks = {"d4_weather": ("M", "p_M4n"), "d4_weather_nwp": ("M", "p_M4bn"), "d5_football": ("M", "p_M4bn"), "d6_house": ("Y", "p_M4n"),
          "d6_president": ("Y", "p_M4n"), "d8_stocks": ("M", "p_M4n"), "d9_quakes": ("Q", "p_M4n")}
tb = {}
for name, (freq, col) in blocks.items():
    p = RES / f"{name}_preds.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    if col not in d:
        col = col.replace("n", "") if col != "p_M4bn" else "p_M4b"
    d = d.dropna(subset=[col, "p_M0"]).reset_index(drop=True)
    d["blk"] = pd.to_datetime(d["t"]).dt.to_period(freq).astype(str)
    yy, pm, p0 = d["y"].to_numpy(), d[col].to_numpy(), d["p_M0"].to_numpy()
    g = {k: np.asarray(v) for k, v in d.groupby("blk").indices.items()}
    bsv = core.cluster_bootstrap(g, lambda idx: 1 - core.log_loss(yy[idx], pm[idx]) / core.log_loss(yy[idx], p0[idx]), reps=1000)
    tb[name] = {"model": col, "block": freq, "n_blocks": len(g), "LSS": 1 - core.log_loss(yy, pm) / core.log_loss(yy, p0),
                "LSS_ci_time_block": np.percentile(bsv, [2.5, 97.5]).tolist()}
# earnings (combined universe; combo model vs point-in-time frequency base rate; outcome-group weights), calendar-quarter blocks
_d = d7.copy()
_d["blk"] = pd.to_datetime(_d["decision_time"]).dt.to_period("Q").astype(str)
_y, _w, _pm, _p0 = _d["y"].to_numpy().astype(int), _d["weight"].to_numpy(), _d["combo"].to_numpy(), _d["freq"].to_numpy()
_g = {k: np.asarray(v) for k, v in _d.groupby("blk").indices.items()}
_bs = core.cluster_bootstrap(_g, lambda idx: 1 - core.log_loss(_y[idx], _pm[idx], _w[idx]) / core.log_loss(_y[idx], _p0[idx], _w[idx]), reps=1000)
tb["d7_earnings"] = {"model": "combo", "block": "Q", "n_blocks": len(_g), "LSS": 1 - core.log_loss(_y, _pm, _w) / core.log_loss(_y, _p0, _w),
                     "LSS_ci_time_block": np.percentile(_bs, [2.5, 97.5]).tolist()}
OUT["time_block"] = tb
(RES / "rev2_analyses.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps({"audit": OUT["audit"]["scenarios"], "power": round(OUT["power"]["n_required"]), "cohort": {k: v for k, v in OUT["earnings_cohort"].items() if k != "note"},
                  "live": {q: (round(v["A_opt"], 4), [round(x, 4) for x in v["A_opt_ci"]], round(v["e0"], 4), round(v["e1"], 4)) for q, v in lv.items()},
                  "nm_diff": (OUT["readout"]["noise_models_max_abs_diff_formula_vs_opt"], OUT["readout"]["noise_models_max_abs_diff_reported_vs_opt"]),
                  "time_block": {k: (round(v["LSS"], 4), [round(x, 4) for x in v["LSS_ci_time_block"]], v["n_blocks"]) for k, v in tb.items()}}, indent=1, default=float))
