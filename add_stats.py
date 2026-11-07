"""Extra statistics requested in review -> ../paper/gen/macros_extra.tex"""
import json, math, pathlib, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent
RES, DATA, GEN = ROOT / "results", ROOT.parent / "data", ROOT.parent / "paper" / "gen"
sys.path.insert(0, str(ROOT))
from kn import core

M = {}
def mac(k, v, f="{:.3f}"):
    M[k] = f.format(v) if not isinstance(v, str) else v

# 1. law slope with OLS standard error and 95% CI (8 horizons); epsilon dependence
law = json.loads((RES / "law.json").read_text())
for eps, tag in (("0.05", "Five"), ("0.1", "Ten")):
    P = np.array(law["nn_law_all"][eps]["points"]); x, y = P[:, 0], np.log(P[:, 1])
    r = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    mac("lawG" + tag, r.slope, "{:.2f}"); mac("lawGlo" + tag, r.slope - tcrit * r.stderr, "{:.2f}"); mac("lawGhi" + tag, r.slope + tcrit * r.stderr, "{:.2f}")
lam = json.loads((RES / "controls.json").read_text())["D2"]["lyapunov"]
betas = [f["exponent"] for h, f in law["nn_fits"].items() if float(h) <= 3]
bhat = float(np.mean(betas))
mac("lamD", lam * 2.06, "{:.2f}"); mac("lamOverBeta", lam / bhat, "{:.2f}")
mac("lawMissPct", 100 * (float(M["lawGFive"]) / (lam * 2.06) - 1), "{:.0f}")
mac("lawMissPctB", 100 * (float(M["lawGFive"]) / (lam / bhat) - 1), "{:.0f}")

# 2. IBM per-qubit range and qubit choice
ib = json.loads((RES / "d3_ibm_noise.json").read_text())
qa = [q["A_dev"] for b in ib["backends"] for q in b["qubits"]]
mac("ibmQmin", min(qa)); mac("ibmQmax", max(qa)); mac("ibmQn", len(qa), "{:d}")
ee = np.array([q["alpha_e0"] + q["e1"] for b in ib["backends"] for q in b["qubits"]])
pred = 0.5 + (1 - ee) / math.pi
mac("ibmPredMAE", float(np.mean(np.abs(np.array(qa) - pred))), "{:.3f}")

# 3. bootstrap CI on the share of the pre-cleaning edge from later-removed rows (resample tickers)
ba = pd.read_parquet(DATA / "d7_before_after.parquet")
sub = ba[ba["scored_before"] == 1].reset_index(drop=True)
y = sub["y"].astype(int).to_numpy(); w = sub["weight_before"].to_numpy()
ce = ((sub["eb_beta_before"] >= .5).astype(int).to_numpy() == y).astype(float)
cf = ((sub["freq_before"] >= .5).astype(int).to_numpy() == y).astype(float)
rem = (sub["removed_by"] != "none").to_numpy()
g = {k: np.asarray(v) for k, v in sub.groupby("entity").indices.items()}
def share(idx):
    tot = np.sum(w[idx] * (ce[idx] - cf[idx])); rr = np.sum((w[idx] * (ce[idx] - cf[idx]))[rem[idx]])
    return rr / tot if abs(tot) > 1e-9 else np.nan
bs = core.cluster_bootstrap(g, share, reps=2000)
bs = bs[np.isfinite(bs)]
lo, hi = np.percentile(bs, [2.5, 97.5])
mac("eShareLo", 100 * lo, "{:.0f}"); mac("eShareHi", 100 * hi, "{:.0f}")
def edge_rem(idx):
    return np.sum((w[idx] * (ce[idx] - cf[idx]))[rem[idx]]) / np.sum(w[idx])
b2 = core.cluster_bootstrap(g, edge_rem, reps=2000)
mac("eRemLo", 100 * np.percentile(b2, 2.5), "{:+.2f}"); mac("eRemHi", 100 * np.percentile(b2, 97.5), "{:+.2f}")

# 4. control sampling errors
c = json.loads((RES / "controls.json").read_text())
ft = {r["n"]: r for r in c["D1"]["full_theta"]}
a = ft[100000]["E1_acc"]; mac("qSE", math.sqrt(a * (1 - a) / 100000), "{:.4f}")
ch = c["D1"]["chsh"]["quantum"]; mac("chshSE", math.sqrt(ch * (1 - ch) / 200000), "{:.4f}")
# chaos: which settings fail the Cover-Hart check
fails = [(r["sigma"], r["h"]) for r in c["D2"]["grid"] if r["E4_A_upper"] < r["A_ensemble"] - 0.01]
mac("chFailMinH", min(h for _, h in fails), "{:g}"); mac("chFailN", len(fails), "{:d}")
mac("chFailHgeOne", sum(1 for _, h in fails if h >= 1), "{:d}")

# 5. transfer ratios across all domains
for n, tag in [("d4_weather", "wx"), ("d5_football", "fb"), ("d6_house", "hs"), ("d6_president", "pr"), ("d8_stocks", "st"), ("d9_quakes", "eq")]:
    d = json.loads((RES / f"{n}.json").read_text())
    t = d.get("tau") or {}
    if t.get("edge_wf") is not None:
        mac(tag + "EdgeWF", 100 * t["edge_wf"], "{:+.2f}")
d7 = json.loads((RES / "d7_analysis.json").read_text())
mac("eTau", d7["transfer"]["tau"], "{:.2f}")
mac("eAuditN", 82, "{:d}")
# nwp: ratio of benchmark to best obs-only model on the same rows
nw = json.loads((RES / "d4_weather_nwp.json").read_text())["summary"]
best_obs = max(nw["p_M4n"]["LSS"], nw["p_M5"]["LSS"], nw["p_M4"]["LSS"])
mac("nwpBestObs", best_obs); mac("nwpRatio", nw["bench"]["LSS"] / best_obs, "{:.1f}")

lines = ["\\newcommand{\\" + k + "}{" + v + "}" for k, v in M.items()]
(GEN / "macros_extra.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
for k, v in M.items():
    print(k, v)
