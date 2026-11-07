"""Phase 1 controls: D1 quantum measurement (exact ceilings, side information, CHSH) and D2 Lorenz-63.

Writes results/controls.json. Seeded; deterministic.
"""
import json, math, sys, pathlib, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kn import core

SEED = 20261106
rng = np.random.default_rng(SEED)
OUT = {}


def learned_estimates(Xtr, ytr, Xte, yte, eta_te=None):
    """E1 (HGB lower bound), E4 (1-NN Cover-Hart upper), E5 interval from E1 log loss."""
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    nn = KNeighborsClassifier(1).fit(Xtr, ytr)
    r_nn = 1 - np.mean(nn.predict(Xte) == yte)
    ll = core.log_loss(yte, p)
    H = float(core.h(yte.mean()))
    ch = core.cover_hart_upper(r_nn)
    r = {"E1_acc": core.accuracy(yte, p), "E1_LSS": 1 - ll / H, "r_nn": float(r_nn),
         "E4_A_upper": ch["A_upper"], "kappa_upper_E7": core.kappa_upper_from_rstar(ch["R_star_lower"], H),
         "E5_R_interval_from_E1_LL": core.fano_interval(ll / math.log(2))}
    if eta_te is not None:
        r["true_A_on_sample"] = float(np.mean(np.where(eta_te >= .5, yte, 1 - yte)))
        r["oracle_acc"] = core.accuracy(yte, eta_te)
    return r


# ----------------------------------------------------------------------------- D1 quantum
def d1():
    res = {"closed_form": {"A_full": 0.5 + 1 / math.pi, "H_cond_nats": 2 * math.log(2) - 1,
                           "kappa_full": 1 / math.log(2) - 1}}
    # (i) full theta, estimator convergence over n
    conv = []
    for n in (1000, 10000, 100000):
        th = rng.uniform(0, math.pi, 2 * n)
        p = np.cos(th / 2) ** 2
        y = (rng.random(2 * n) < p).astype(int)
        r = learned_estimates(th[:n, None], y[:n], th[n:, None], y[n:], p[n:])
        r["n"] = n
        conv.append(r)
    res["full_theta"] = conv
    # (ii) noisy knowledge of theta: exact eta via grid integration
    grid = np.linspace(0, math.pi, 2001)
    pg = np.cos(grid / 2) ** 2
    sig = []
    for s in (0.0, 0.1, 0.3, 0.6, 1.0, 2.0):
        n = 20000
        th = rng.uniform(0, math.pi, 2 * n)
        that = th + (rng.normal(0, s, 2 * n) if s > 0 else 0)
        p = np.cos(th / 2) ** 2
        y = (rng.random(2 * n) < p).astype(int)
        if s > 0:
            wts = np.exp(-0.5 * ((that[:, None] - grid[None, :]) / s) ** 2)
            eta = (wts * pg).sum(1) / wts.sum(1)
        else:
            eta = p
        A_true = float(np.mean(np.maximum(eta, 1 - eta)))
        kap = float(1 - core.h(eta).mean() / math.log(2))
        r = learned_estimates(that[:n, None], y[:n], that[n:, None], y[n:], eta[n:])
        r.update({"sigma": s, "A_star": A_true, "kappa_true": kap})
        sig.append(r)
    res["noisy_theta"] = sig
    # (iii) depolarizing
    res["depolarizing"] = [{"lambda": l, "A_star": 0.5 + (1 - l) / math.pi} for l in (0, .05, .1, .2, .5, 1)]
    # (iv) side information with a Bell pair |Phi+>, A measured in Z or X at random
    N = 200000
    basis = rng.integers(0, 2, N)               # 0 = Z, 1 = X
    a = rng.integers(0, 2, N)                    # A's outcome is uniform in either basis
    # classical side info: B measured in Z beforehand -> equals A when A measured in Z; independent otherwise
    b_cls = np.where(basis == 0, a, rng.integers(0, 2, N))
    # quantum side info: holder of B measures in the announced basis -> perfectly correlated for Phi+
    b_q = a.copy()
    res["side_info"] = {"none": float(np.mean(rng.integers(0, 2, N) == a)),
                        "classical": float(np.mean(b_cls == a)),
                        "quantum": float(np.mean(b_q == a)),
                        "theory": {"none": 0.5, "classical": 0.75, "quantum": 1.0}}
    # (v) CHSH game, statevector simulation of |Phi+> with optimal angles
    x, yb = rng.integers(0, 2, N), rng.integers(0, 2, N)
    ta = np.where(x == 0, 0.0, math.pi / 4)
    tb = np.where(yb == 0, math.pi / 8, -math.pi / 8)
    p_same = np.cos(ta - tb) ** 2               # P(a == b) for real-plane measurements on Phi+
    same = rng.random(N) < p_same
    win_q = np.where((x & yb) == 1, ~same, same)
    res["chsh"] = {"quantum": float(win_q.mean()), "classical_best": 0.75,
                   "tsirelson": math.cos(math.pi / 8) ** 2}
    return res


# ----------------------------------------------------------------------------- D2 Lorenz-63
S_, R_, B_ = 10.0, 28.0, 8 / 3


def lorenz_rhs(X):
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([S_ * (y - x), x * (R_ - z) - y, x * y - B_ * z], -1)


def rk4(X, dt):
    k1 = lorenz_rhs(X); k2 = lorenz_rhs(X + dt / 2 * k1)
    k3 = lorenz_rhs(X + dt / 2 * k2); k4 = lorenz_rhs(X + dt * k3)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lyapunov(steps=40000, dt=0.01):
    X = np.array([1.0, 1.0, 1.0]); d0 = 1e-8
    Y = X + np.array([d0, 0, 0]); s = 0.0
    for _ in range(1000):
        X = rk4(X, dt); Y = rk4(Y, dt)
    Y = X + (Y - X) / np.linalg.norm(Y - X) * d0
    for _ in range(steps):
        X = rk4(X, dt); Y = rk4(Y, dt)
        d = np.linalg.norm(Y - X); s += math.log(d / d0)
        Y = X + (Y - X) / d * d0
    return s / (steps * dt)


def d2():
    dt = 0.01
    hs = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    lam = lyapunov()
    # attractor samples
    X = rng.normal(0, 1, (6000, 3)) + np.array([0, 0, 25])
    for _ in range(3000):
        X = rk4(X, dt)
    ntest, npart = 1500, 400
    res = {"lyapunov": lam, "grid": []}
    hsteps = {h: int(round(h / dt)) for h in hs}
    for s in sigmas:
        true0 = X[:ntest]
        obs = true0 + rng.normal(0, s, true0.shape)
        parts = obs[:, None, :] + rng.normal(0, s, (ntest, npart, 3))
        T, P = true0.copy(), parts.copy()
        k = 0
        # training set for learned estimator: separate states
        tr0 = X[ntest:ntest + 4000]
        tro = tr0 + rng.normal(0, s, tr0.shape)
        TR = tr0.copy()
        for h in hs:
            while k < hsteps[h]:
                T = rk4(T, dt); P = rk4(P, dt); TR = rk4(TR, dt); k += 1
            y = (T[:, 0] > 0).astype(int)
            eta = (P[..., 0] > 0).mean(1)
            ytr = (TR[:, 0] > 0).astype(int)
            A_ens = core.accuracy(y, eta)
            le = learned_estimates(tro, ytr, obs, y)
            res["grid"].append({"sigma": s, "h": h, "A_ensemble": A_ens,
                                "kappa_ensemble_LSS": 1 - core.log_loss(y, np.clip(eta, .002, .998)) / float(core.h(y.mean())),
                                "Tp_pred_tol1": math.log(1.0 / s) / lam, **le})
    # LZ critique: sign sequence sampled every 0.25
    Z = np.array([1.0, 1.0, 1.0]); seq = []
    for i in range(4000 * 25):
        Z = rk4(Z, dt)
        if i % 25 == 0:
            seq.append(int(Z[0] > 0))
    seq = np.array(seq[200:])
    pi_lz, S = core.lz_max_predictability(seq[:3000])
    # actual predictability of next symbol from last 4 symbols (empirical, out of sample)
    k = 4
    feats = np.array([seq[i - k:i] for i in range(k, len(seq))]); tgt = seq[k:]
    half = len(tgt) // 2
    m = HistGradientBoostingClassifier(random_state=SEED).fit(feats[:half], tgt[:half])
    res["lz"] = {"Pi_max_LZ": pi_lz, "S_bits": S, "learned_next_symbol_acc": float(np.mean(m.predict(feats[half:]) == tgt[half:])),
                 "base": float(max(tgt[half:].mean(), 1 - tgt[half:].mean()))}
    return res


if __name__ == "__main__":
    t0 = time.time()
    OUT["D1"] = d1(); print("D1 done", time.time() - t0, flush=True)
    OUT["D2"] = d2(); print("D2 done", time.time() - t0, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "controls.json").write_text(json.dumps(OUT, indent=1, default=float))
    print(json.dumps({"D1_side": OUT["D1"]["side_info"], "chsh": OUT["D1"]["chsh"], "lyap": OUT["D2"]["lyapunov"],
                      "lz": OUT["D2"]["lz"]}, indent=1))
