"""Delft loophole-free Bell test, 2nd run (Hensen et al. arXiv:1603.05705), 4TU dataset.
Selection logic follows the dataset's own bell_open_data_2_analysis_example.py exactly."""
import json, sys, os
import numpy as np
from scipy.stats import binom, beta

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw/bell/x"
OUT = sys.argv[2] if len(sys.argv) > 2 else "repo/results/bell_delft.json"


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def load(fn, w0):
    d = np.loadtxt(fn, delimiter=',', usecols=range(1, 17), dtype=np.int64)
    c1t, c1c, c2t, c2c = d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    a, b = d[:, 6], d[:, 7]
    roA, roB = d[:, 10], d[:, 11]
    exA, exB = d[:, 12], d[:, 13]
    invA, invB = d[:, 14], d[:, 15]
    w1 = 5425100
    L = 50000
    sep = 250000
    f1 = ((w0 <= c1t) & (c1t < w0 + L) & (c1c == 0)) | ((w1 <= c1t) & (c1t < w1 + L) & (c1c == 1))
    f2min = ((w0 + sep <= c2t) & (c2t < w0 + sep + L) & (c2c == 0)) | ((w1 + sep <= c2t) & (c2t < w1 + sep + L) & (c2c == 1))
    f2plus = ((w0 + sep <= c2t) & (c2t < w0 + sep + 4000) & (c2c == 0)) | ((w1 + sep <= c2t) & (c2t < w1 + sep + 2500) & (c2c == 1))
    minus = c1c != c2c
    ev = (f1 & f2min & minus) | (f1 & f2plus & ~minus)
    okA = (invA == 0) | (invA > 250)
    okB = (invB == 0) | (invB > 250)
    valid = ev & okA & okB & (exA == 0) & (exB == 0)
    rA = (roA > 10620) & (roA <= 10620 + 3700)
    rB = (roB > 10620) & (roB <= 10620 + 3700)
    t = valid.astype(np.int64) * (np.where(minus, -1, 1))  # +1 psi+, -1 psi-
    return a, b, rA * 2 - 1, rB * 2 - 1, t, len(d)


w_old = 5426000
parts = [load(os.path.join(RAW, "bell_open_data_2_old_detector.txt"), w_old),
         load(os.path.join(RAW, "bell_open_data_2_new_detector.txt"), w_old - 700)]
a_all = np.concatenate([p[0] for p in parts])
b_all = np.concatenate([p[1] for p in parts])
x_all = np.concatenate([p[2] for p in parts])
y_all = np.concatenate([p[3] for p in parts])
t_all = np.concatenate([p[4] for p in parts])
n_rows = [p[5] for p in parts]
m = t_all != 0
a, b, x, y, t = a_all[m], b_all[m], x_all[m], y_all[m], t_all[m]
n = int(m.sum())
res = {"source": {"dataset": "4TU.ResearchData doi:10.4121/uuid:53644d31-d862-4f9f-9ad2-0b571874b829 (Hensen et al., arXiv:1603.05705)",
                  "licence": "4TU General Terms of Use (https://doi.org/10.4121/resource:terms_of_use)",
                  "file": "data.zip md5 de565fc8e550cdd6684b4182d4235d1f",
                  "rows_old_detector": n_rows[0], "rows_new_detector": n_rows[1]},
       "paper_reported": {"n": 300, "S2": 2.35, "S2_err": 0.18, "k": 237, "p_complete": 0.061, "p_conventional": 0.029}}
res["n_trials"] = n
res["n_psi_plus"] = int((t == 1).sum())
res["n_psi_minus"] = int((t == -1).sum())
res["n_trials_per_run"] = {"old": int((parts[0][4] != 0).sum()), "new": int((parts[1][4] != 0).sum())}
res["settings_counts_ab"] = {f"{i}{j}": int(((a == i) & (b == j)).sum()) for i in (0, 1) for j in (0, 1)}


def S_for(psi):
    s = 0
    var = 0
    E = {}
    for i in (0, 1):
        for j in (0, 1):
            mm = (t == psi) & (a == i) & (b == j)
            nn = int(mm.sum())
            e = float((x[mm] * y[mm]).sum()) / nn
            E[f"{i}{j}"] = {"n": nn, "E": e}
            if psi == -1:
                sign = -1 if (i == 1 and j == 1) else 1
            else:
                sign = -1 if (i == 1 and j == 0) else 1
            s += sign * e
            var += (1 - e * e) / nn
    return s, float(np.sqrt(var)), E


Sp, Sp_e, Ep = S_for(1)
Sm, Sm_e, Em = S_for(-1)
npl, nmi = res["n_psi_plus"], res["n_psi_minus"]
Sc = (npl * Sp + nmi * Sm) / n
Sc_e = float(np.sqrt((npl / n) ** 2 * Sp_e ** 2 + (nmi / n) ** 2 * Sm_e ** 2))
res["S"] = {"psi_plus": Sp, "psi_plus_err": Sp_e, "psi_minus": Sm, "psi_minus_err": Sm_e, "combined": Sc, "combined_err": Sc_e,
            "E_psi_plus": Ep, "E_psi_minus": Em}

# win rate with the paper's CHSH games: psi-: (-1)^{ab} xy=1 ; psi+: (-1)^{a(b+1)} xy=1
win = np.where(t == -1, ((-1.0) ** (a * b)) * x * y, ((-1.0) ** (a * (b + 1))) * x * y) == 1
k = int(win.sum())
w = k / n
lit = (((1 - x) // 2) ^ ((1 - y) // 2)) == (a & b)  # literal x xor y == a AND b, all trials
kl = int(lit.sum())
tau = 5.4e-6 * 2
ksi = 0.75 + 3 * (tau + tau ** 2)
res["omega"] = {"k": k, "n": n, "omega": w, "CP95": cp(k, n), "S_equiv_8w_minus_4": 8 * w - 4,
                "classical_bound": 0.75, "tsirelson": float(np.cos(np.pi / 8) ** 2),
                "p_binom_gt_0.75": float(binom.sf(k - 1, n, 0.75)),
                "paper_threshold_xi": ksi, "p_binom_gt_xi_dataset_script": float(binom.sf(k - 1, n, ksi)),
                "omega_literal_xor_eq_a_and_b_all_trials": kl / n, "literal_k": kl, "literal_CP95": cp(kl, n)}
res["omega_by_state"] = {}
for nm, mk in (("psi_minus", t == -1), ("psi_plus", t == 1)):
    kk = int(win[mk].sum())
    nn = int(mk.sum())
    res["omega_by_state"][nm] = {"k": kk, "n": nn, "omega": kk / nn, "CP95": cp(kk, nn), "S_equiv": 8 * kk / nn - 4}

# Bob predicts y from (a,b,x)
yb = (y == 1).astype(int)


def pack(keys):
    ks = np.asarray(keys)
    N = len(yb)
    ci = 0
    loo = np.zeros(N, bool)
    for u in np.unique(ks):
        idx = np.where(ks == u)[0]
        n1 = int(yb[idx].sum())
        n0 = len(idx) - n1
        ci += max(n0, n1)
        for ii in idx:
            c1 = n1 - yb[ii]
            c0 = n0 - (1 - yb[ii])
            pred = 1 if c1 >= c0 else 0  # ties -> 1 (fixed, arbitrary)
            loo[ii] = (pred == yb[ii])
    kl_ = int(loo.sum())
    return {"in_sample_acc": ci / N, "in_sample_CP95": cp(int(ci), N), "loo_acc": kl_ / N, "loo_CP95": cp(kl_, N)}, loo


r_none, l_none = pack(np.zeros(n, int))
r_b, l_b = pack(b)
r_abx, l_abx = pack(a * 4 + b * 2 + (x == 1))
r_abxt, l_abxt = pack(a * 8 + b * 4 + (x == 1) * 2 + (t == 1))
rng = np.random.default_rng(0)


def bdiff(l1, l0):
    d = l1.astype(float) - l0.astype(float)
    N = len(d)
    bs = [d[rng.integers(0, N, N)].mean() for _ in range(5000)]
    return {"mean_diff": float(d.mean()), "boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


per = {}
for i in (0, 1):
    for j in (0, 1):
        mk = (a == i) & (b == j)
        sub = {}
        for xv in (-1, 1):
            mm = mk & (x == xv)
            nn = int(mm.sum())
            n1 = int((yb[mm] == 1).sum())
            sub[f"x={xv}"] = {"n": nn, "n_y_plus1": n1, "best_pred_y": 1 if n1 * 2 >= nn else -1, "acc": max(n1, nn - n1) / nn}
        kk = sum(max(s["n_y_plus1"], s["n"] - s["n_y_plus1"]) for s in sub.values())
        per[f"ab={i}{j}"] = {"n": int(mk.sum()), "acc_from_abx_in_sample": kk / int(mk.sum()), "CP95": cp(int(kk), int(mk.sum())), "cells": sub}
res["prediction_Bob_y"] = {
    "note": "in-sample best-rule accuracy is optimistically biased; LOO is the honest estimate. y=+1 means readout photon detected at B.",
    "no_side_info_majority": r_none, "from_b_only": r_b, "from_a_b_x": r_abx, "from_a_b_x_state": r_abxt,
    "loo_gain_abx_over_b": bdiff(l_abx, l_b), "loo_gain_abx_over_none": bdiff(l_abx, l_none),
    "per_setting_pair": per, "frac_y_plus1": float(yb.mean())}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k_: res[k_] for k_ in ("n_trials", "n_psi_plus", "n_psi_minus", "n_trials_per_run", "settings_counts_ab", "omega", "omega_by_state")}, indent=1))
print({k_: v for k_, v in res["S"].items() if not k_.startswith("E_")})
print(json.dumps({k_: v for k_, v in res["prediction_Bob_y"].items() if k_ != "per_setting_pair"}, indent=1))
