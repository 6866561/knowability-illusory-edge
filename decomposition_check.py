"""Separate numerical verification of each component of the four-term decomposition (knowability paper, Proposition 2),
on the qubit control where every quantity is known exactly. All expectations are integrals over theta ~ U(0, pi),
evaluated by Gauss-Legendre quadrature (no sampling) unless stated. Writes results/decomposition_check.json.

(1) H(Y | theta) = 2 ln 2 - 1 (closed form) vs quadrature.
(2) For a trained gradient-boosting forecast (conditional on its training sample), E l(p_hat, Y) = H(Y|theta) + E KL(eta || p_hat),
    each side by quadrature; plus a Monte Carlo check with sampled labels.
(3) Epistemic term: with X = theta rounded to one of 8 bins and Z = theta, H(Y | X) - H(Y | X, Z) = I(Y; Z | X) >= 0.
(4) Signed measurement term: with label-flip probabilities f01(theta), f10(theta), E l(p_hat, Y~) - E l(p_hat, Y) equals
    M = E[(Y~ - Y) ln((1 - p_hat)/p_hat)] exactly; one corruption pattern that agrees with the forecast gives M < 0, one that
    opposes it gives M > 0.
"""
import json, math, pathlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parent
xg, wg = np.polynomial.legendre.leggauss(4000)
TH = (xg + 1) * math.pi / 2; W = wg / 2            # weights integrate to 1 under U(0, pi)
ETA = np.cos(TH / 2) ** 2


def hb(p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def kl(e, p):
    e = np.clip(e, 1e-15, 1 - 1e-15); p = np.clip(p, 1e-15, 1 - 1e-15)
    return e * np.log(e / p) + (1 - e) * np.log((1 - e) / (1 - p))


def exp_loss(q, p):            # E over theta of the log loss of p against a label with P(label = 1 | theta) = q
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(np.sum(W * (-(q * np.log(p) + (1 - q) * np.log(1 - p)))))


out = {}
H_exact = 2 * math.log(2) - 1
H_quad = float(np.sum(W * hb(ETA)))
out["H_cond"] = {"closed_form": H_exact, "quadrature": H_quad, "abs_diff": abs(H_exact - H_quad)}

rng = np.random.default_rng(5)
n = 20000; th = rng.uniform(0, math.pi, n); y = (rng.random(n) < np.cos(th / 2) ** 2).astype(int)
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(th[:, None], y)
P = m.predict_proba(TH[:, None])[:, 1]
lhs = exp_loss(ETA, P); KL = float(np.sum(W * kl(ETA, P)))
out["model_term"] = {"expected_loss": lhs, "H_plus_KL": H_quad + KL, "KL": KL, "abs_diff": abs(lhs - (H_quad + KL))}
th2 = rng.uniform(0, math.pi, 2_000_000); y2 = (rng.random(len(th2)) < np.cos(th2 / 2) ** 2)
p2 = np.clip(m.predict_proba(th2[:, None])[:, 1], 1e-15, 1 - 1e-15)
l2 = -(y2 * np.log(p2) + (~y2) * np.log(1 - p2))
out["model_term"]["monte_carlo_loss"] = float(l2.mean()); out["model_term"]["monte_carlo_se"] = float(l2.std() / math.sqrt(len(l2)))

bins = np.minimum((TH / math.pi * 8).astype(int), 7)
H_X = 0.0
for b in range(8):
    sel = bins == b; wb = W[sel].sum(); eb = np.sum(W[sel] * ETA[sel]) / wb
    H_X += wb * hb(eb)
out["epistemic"] = {"H_Y_given_X_8bins": float(H_X), "H_Y_given_X_Z": H_quad, "I_Y_Z_given_X": float(H_X - H_quad)}

res = {}
for name, (f01, f10) in {"agrees_with_forecast": (np.where(P > 0.5, 0.15, 0.0), np.where(P < 0.5, 0.15, 0.0)),
                         "opposes_forecast": (np.where(P < 0.5, 0.15, 0.0), np.where(P > 0.5, 0.15, 0.0))}.items():
    q = ETA * (1 - f10) + (1 - ETA) * f01                   # P(recorded label = 1 | theta)
    d = exp_loss(q, P) - exp_loss(ETA, P)
    M = float(np.sum(W * (q - ETA) * np.log((1 - np.clip(P, 1e-15, 1)) / np.clip(P, 1e-15, 1))))
    res[name] = {"measured_minus_true_loss": d, "M": M, "abs_diff": abs(d - M)}
out["measurement_term"] = res
(ROOT / "results" / "decomposition_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
