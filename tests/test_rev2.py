"""Numerical checks for the revision-2 mathematics (supporting validation, not proofs)."""
import math, sys, pathlib
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kn import core

rng = np.random.default_rng(7)


def ceiling_asym(e0, e1):
    beta = 1 - e0 - e1; a = (e0 - e1) / 2; b = beta / 2
    if b <= abs(a):
        return 0.5 + abs(a)
    t0 = math.acos(a / b)
    return 0.5 + (beta * math.sin(t0) + a * (math.pi - 2 * t0)) / math.pi


def test_asymmetric_readout_closed_form():
    for e0, e1 in [(0.01, 0.01), (0.02, 0.15), (0.2, 0.01), (0.3, 0.3), (0.05, 0.4)]:
        th = rng.uniform(0, math.pi, 1_000_000)
        pp = e0 + (1 - e0 - e1) * np.sin(th / 2) ** 2
        assert abs(np.mean(np.maximum(pp, 1 - pp)) - ceiling_asym(e0, e1)) < 2e-3
    assert abs(ceiling_asym(0.03, 0.03) - (0.5 + 0.94 / math.pi)) < 1e-12   # symmetric case reduces to 1/2 + beta/pi


def test_inflation_identity_row_varying():
    """Delta_meas - Delta_true = 2 E[(d_f - d_g)(q_tilde - q)] for fixed rules, q = P(Y=1|X), q_tilde = P(Y_obs=1|X)."""
    n = 2_000_000
    x = rng.random(n); q = 0.2 + 0.6 * x
    r = np.where(x > 0.7, 0.4, 0.0); s = np.where(x < 0.2, 0.3, 0.0)        # covariate-dependent flip rates
    qt = q * (1 - r) + (1 - q) * s
    y = (rng.random(n) < q).astype(int)
    flip = np.where(y == 1, rng.random(n) < r, rng.random(n) < s)
    yt = np.where(flip, 1 - y, y)
    df_, dg = (x > 0.35).astype(int), np.ones(n, int)
    meas = np.mean(df_ == yt) - np.mean(dg == yt); true = np.mean(df_ == y) - np.mean(dg == y)
    pred = 2 * np.mean((df_ - dg) * (qt - q))
    assert abs((meas - true) - pred) < 3e-3


def test_one_nn_asymptotic_error():
    """Cover-Hart: R_NN -> E[2 eta (1 - eta)] (1-D, continuous eta); k-NN with k = sqrt(n) -> Bayes error."""
    eta = lambda x: 0.5 + 0.4 * np.sin(6 * x)
    n, m = 200_000, 20_000
    xtr = rng.random(n); ytr = (rng.random(n) < eta(xtr)).astype(int)
    xte = rng.random(m); yte = (rng.random(m) < eta(xte)).astype(int)
    o = np.argsort(xtr); xs, ys = xtr[o], ytr[o]
    pos = np.clip(np.searchsorted(xs, xte), 1, n - 1)
    nn = np.where(np.abs(xs[pos] - xte) < np.abs(xs[pos - 1] - xte), pos, pos - 1)
    err_nn = np.mean(ys[nn] != yte)
    e = eta(xte); r = np.minimum(e, 1 - e)
    assert abs(err_nn - np.mean(2 * r * (1 - r))) < 0.012
    assert np.mean(2 * r * (1 - r)) - np.mean(r) > 0.03            # 1-NN floor above Bayes error here


def test_hypergeometric_audit_bound_coverage():
    """Exact finite-population upper bound has coverage >= 95% for every true K (N = 82, n = 40)."""
    N, n = 82, 40
    def upper(e):
        K_U = e
        for K in range(e, N + 1):
            if stats.hypergeom.cdf(e, N, K, n) > 0.05:
                K_U = K
            else:
                break
        return K_U
    for K in (0, 3, 8, 15, 30):
        cov = sum(stats.hypergeom.pmf(e, N, K, n) for e in range(0, min(K, n) + 1) if upper(e) >= K)
        assert cov >= 0.95 - 1e-9


def test_beta_binomial_finite_history_accuracy():
    """Closed form A_1^*(n) (Appendix A) vs Monte Carlo; monotone in n."""
    import numpy as np
    from scipy.special import betaln, comb
    a, b = 0.8, 1.3
    def closed(n):
        k = np.arange(n + 1); m = (a + k) / (a + b + n)
        w = comb(n, k) * np.exp(betaln(a + k, b + n - k) - betaln(a, b))
        return float(np.sum(w * np.maximum(m, 1 - m)))
    rng = np.random.default_rng(0); N = 400_000; n = 6
    p = rng.beta(a, b, N); k = rng.binomial(n, p); y = rng.random(N) < p
    pred = (a + k) / (a + b + n) > 0.5
    assert abs(np.mean(pred == y) - closed(n)) < 4e-3
    vals = [closed(j) for j in range(0, 30)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:]))
    assert vals[-1] < np.mean(np.maximum(p, 1 - p)) + 4e-3
