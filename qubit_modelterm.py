"""Model term of the qubit control in log-loss terms (knowability paper, review item 1).

The learner of run_controls.d1 (HistGradientBoosting on the exactly known angle) is trained on n draws; on an
independent test set we report (i) held-out log loss against the exact conditional entropy H(Y|theta) = 2 ln 2 - 1,
and (ii) the model term E KL(eta || p_hat), computed exactly from the known eta(theta) = cos^2(theta/2) at the
test angles (no label noise), with a bootstrap interval over test angles. Three training seeds per n.
Writes results/qubit_modelterm.json.
"""
import json, math, pathlib, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parent
H_COND = 2 * math.log(2) - 1
NTEST = 200_000


def kl(e, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); e = np.clip(e, 1e-12, 1 - 1e-12)
    return e * np.log(e / p) + (1 - e) * np.log((1 - e) / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


out = {"H_cond_exact": H_COND, "n_test": NTEST, "rows": []}
for n in (1000, 10000, 100000):
    for seed in (0, 1, 2):
        rng = np.random.default_rng(20261108 + 100 * seed + n)
        th = rng.uniform(0, math.pi, n + NTEST); eta = np.cos(th / 2) ** 2
        y = (rng.random(n + NTEST) < eta).astype(int)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(th[:n, None], y[:n])
        p = m.predict_proba(th[n:, None])[:, 1]
        e_te, y_te = eta[n:], y[n:]
        k = kl(e_te, p)
        b = np.random.default_rng(seed)
        bs = [k[b.integers(0, NTEST, NTEST)].mean() for _ in range(500)]
        out["rows"].append({"n": n, "seed": seed, "test_logloss": float(ll(y_te, p).mean()), "oracle_logloss": float(ll(y_te, e_te).mean()),
                            "model_term_KL": float(k.mean()), "model_term_KL_ci": np.percentile(bs, [2.5, 97.5]).tolist(),
                            "accuracy": float(np.mean((p >= .5) == y_te)), "max_abs_p_minus_eta": float(np.max(np.abs(p - e_te)))})
        print(out["rows"][-1])
summ = {}
for n in (1000, 10000, 100000):
    r = [x for x in out["rows"] if x["n"] == n]
    summ[str(n)] = {"model_term_KL_mean": float(np.mean([x["model_term_KL"] for x in r])),
                    "model_term_KL_max_ci_hi": float(max(x["model_term_KL_ci"][1] for x in r)),
                    "test_logloss_mean": float(np.mean([x["test_logloss"] for x in r])),
                    "share_of_H_cond": float(np.mean([x["model_term_KL"] for x in r]) / H_COND)}
out["summary"] = summ
(ROOT / "results" / "qubit_modelterm.json").write_text(json.dumps(out, indent=1))
print(json.dumps(summ, indent=1))
