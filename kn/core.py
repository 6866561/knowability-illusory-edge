"""Core math for the knowability framework: metrics, ceilings, statistics, label-noise bounds.

Every function is pure and deterministic given its inputs (and seed where relevant).
Logs are natural unless a name ends in _bits.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import special as sp, stats, optimize

EPS = 1e-12


# ----------------------------------------------------------------------------- metrics
def _w(w, n):
    return np.ones(n) if w is None else np.asarray(w, float)


def accuracy(y, p, w=None, thr=0.5):
    y, p = np.asarray(y), np.asarray(p, float)
    w = _w(w, len(y))
    return float(np.sum(w * ((p >= thr).astype(int) == y)) / w.sum())


def log_loss(y, p, w=None):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    w = _w(w, len(y))
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum())


def brier(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = _w(w, len(y))
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def murphy(y, p, bins=10, w=None):
    """Murphy (1973) decomposition BS ~= REL - RES + UNC on equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = _w(w, len(y))
    W = w.sum()
    ybar = np.sum(w * y) / W
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        if not m.any():
            continue
        wk = w[m].sum()
        pk, yk = np.sum(w[m] * p[m]) / wk, np.sum(w[m] * y[m]) / wk
        rel += wk * (pk - yk) ** 2
        res += wk * (yk - ybar) ** 2
    return {"REL": rel / W, "RES": res / W, "UNC": ybar * (1 - ybar)}


def ece(y, p, bins=10, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = _w(w, len(y))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            wk = w[m].sum()
            tot += wk * abs(np.sum(w[m] * p[m]) / wk - np.sum(w[m] * y[m]) / wk)
    return float(tot / w.sum())


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def skill(y, p, p_base, w=None):
    """Skill scores relative to the base-rate model's own out-of-sample forecasts."""
    a, a0 = accuracy(y, p, w), accuracy(y, p_base, w)
    ll, ll0 = log_loss(y, p, w), log_loss(y, p_base, w)
    bs, bs0 = brier(y, p, w), brier(y, p_base, w)
    return {
        "acc": a, "acc0": a0, "S_acc": (a - a0) / (1 - a0) if a0 < 1 else float("nan"),
        "ll": ll, "ll0": ll0, "LSS": 1 - ll / ll0,
        "bs": bs, "bs0": bs0, "BSS": 1 - bs / bs0,
    }


def coverage_curve(y, p, w=None, grid=None):
    """Accuracy of forecasts with |p-0.5| >= c, as a function of coverage."""
    y, p = np.asarray(y), np.asarray(p, float)
    w = _w(w, len(y))
    conf = np.abs(p - 0.5)
    grid = np.linspace(0, 0.5, 51) if grid is None else grid
    out = []
    for c in grid:
        m = conf >= c
        if w[m].sum() < 1e-9:
            break
        out.append((float(w[m].sum() / w.sum()), accuracy(y[m], p[m], w[m]), float(c)))
    return out


# ----------------------------------------------------------------------------- entropy helpers
def h(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def h_bits(p):
    return h(p) / math.log(2)


def h_bits_inv(H):
    """Inverse of binary entropy (bits) on [0, 1/2]."""
    if H <= 0:
        return 0.0
    if H >= 1:
        return 0.5
    return optimize.brentq(lambda q: float(h_bits(q)) - H, 1e-12, 0.5)


# ----------------------------------------------------------------------------- closed-form ceilings
def qubit_ceiling(lam=0.0):
    """theta ~ U(0, pi), full knowledge of theta, depolarizing strength lam."""
    return {"A": 0.5 + (1 - lam) / math.pi,
            "H_cond_nats": None if lam else 2 * math.log(2) - 1,
            "kappa": None if lam else 1 / math.log(2) - 1}


def beta_ceiling(a, b):
    """Oracle entity-rate ceiling for p ~ Beta(a, b): accuracy and E[h(p)] (nats)."""
    A = a / (a + b) * (1 - sp.betainc(a + 1, b, 0.5)) + b / (a + b) * sp.betainc(a, b + 1, 0.5)
    Eh = sp.digamma(a + b + 1) - (a * sp.digamma(a + 1) + b * sp.digamma(b + 1)) / (a + b)
    m = a / (a + b)
    return {"A": float(A), "Eh": float(Eh), "H": float(h(m)), "kappa": float(1 - Eh / h(m)),
            "A0": max(m, 1 - m)}


def beta_learning_curve(a, b, n_max=40):
    """Bayes accuracy with n past outcomes per entity, n = 0..n_max (exact beta-binomial)."""
    out = []
    for n in range(n_max + 1):
        k = np.arange(n + 1)
        logpmf = (sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n - k + 1)
                  + sp.betaln(a + k, b + n - k) - sp.betaln(a, b))
        post = (a + k) / (a + b + n)
        out.append(float(np.sum(np.exp(logpmf) * np.maximum(post, 1 - post))))
    return np.array(out)


def mom_beta(k, n):
    """Method-of-moments beta prior from per-entity successes k and trials n (net of binomial noise)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    keep = n >= 2
    k, n = k[keep], n[keep]
    m = k.sum() / n.sum()
    rates = k / n
    v = max(np.var(rates) - m * (1 - m) * np.mean(1 / n), 1e-3)
    s = max(m * (1 - m) / v - 1, 0.5)
    return m * s, (1 - m) * s


def tie_mass_bound(frac_near_tie):
    return 1 - 0.5 * frac_near_tie


# ----------------------------------------------------------------------------- nonparametric ceilings
def cover_hart_upper(r_nn):
    """Upper bound on Bayes accuracy from asymptotic 1-NN error (Cover & Hart 1967)."""
    r_nn = min(max(r_nn, 0.0), 0.5)
    r_star_lo = (1 - math.sqrt(1 - 2 * r_nn)) / 2
    return {"A_upper": 1 - r_star_lo, "R_star_lower": r_star_lo}


def fano_interval(H_cond_bits):
    """Bayes error interval from conditional entropy (binary): Fano lower, Hellman-Raviv upper."""
    return {"R_lower": h_bits_inv(H_cond_bits), "R_upper": H_cond_bits / 2}


def kappa_upper_from_rstar(r_star_lower, H_nats):
    """Hellman-Raviv: H_bits(Y|F) >= 2 R*, so kappa <= 1 - 2 R* ln2 / H(Y)."""
    return 1 - 2 * r_star_lower * math.log(2) / H_nats


def lz_entropy_rate(seq):
    """Kontoyiannis et al. (1998) estimator, bits/symbol. seq: 1-D array of symbols."""
    s = "".join(chr(65 + int(x)) for x in seq)
    n = len(s)
    if n < 10:
        return float("nan")
    lam = []
    for i in range(n):
        L = 1  # shortest substring starting at i that never appeared in s[:i]
        while i + L <= n and s[i:i + L] in s[:i]:
            L += 1
        lam.append(L)
    return (np.mean(lam) / math.log2(n)) ** -1


def lz_max_predictability(seq):
    S = lz_entropy_rate(seq)
    if not np.isfinite(S):
        return float("nan"), S
    S = min(S, 1.0)
    pi = 1 - h_bits_inv(S)  # solve S = h(Pi) on Pi in [1/2, 1]
    return pi, S


# ----------------------------------------------------------------------------- statistics
def cluster_bootstrap(df_groups, stat_fn, reps=2000, seed=20261106):
    """Resample clusters with replacement. df_groups: dict cluster -> index array. stat_fn(idx)->float|tuple."""
    rng = np.random.default_rng(seed)
    keys = list(df_groups)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([df_groups[keys[i]] for i in pick])
        out.append(stat_fn(idx))
    return np.array(out)


def half_split_win(groups, fn_diff, reps=1000, seed=20261106):
    """Share of random 50/50 cluster splits (scored on one half) in which fn_diff(idx) > 0."""
    rng = np.random.default_rng(seed)
    keys = np.array(list(groups))
    wins = []
    for _ in range(reps):
        sel = rng.permutation(len(keys))[: len(keys) // 2]
        idx = np.concatenate([groups[k] for k in keys[sel]])
        wins.append(fn_diff(idx) > 0)
    return float(np.mean(wins))


def mcnemar(y, pa, pb):
    y = np.asarray(y)
    ca, cb = (np.asarray(pa) >= .5).astype(int) == y, (np.asarray(pb) >= .5).astype(int) == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": chi2, "p": float(stats.binomtest(b, b + c).pvalue)}


def diebold_mariano(loss_a, loss_b):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    L = int(T ** (1 / 3))
    dc = d - d.mean()
    g = [np.dot(dc[k:], dc[:T - k]) / T for k in range(L + 1)]
    var = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    dm = d.mean() / math.sqrt(max(var, 1e-18) / T)
    return {"dm": dm, "p": float(2 * stats.norm.sf(abs(dm))), "mean_diff": float(d.mean())}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def clopper_pearson_upper(e, k, alpha=0.05):
    return 1.0 if e >= k else float(stats.beta.ppf(1 - alpha, e + 1, k - e))


def mcnemar_n(p_disagree, delta, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z ** 2 * p_disagree / delta ** 2


# ----------------------------------------------------------------------------- label noise (Section 7)
def disagreement_decomposition(y, pa, pb, w=None):
    y = np.asarray(y)
    w = _w(w, len(y))
    da, db = (np.asarray(pa) >= .5).astype(int), (np.asarray(pb) >= .5).astype(int)
    D = da != db
    PD = w[D].sum() / w.sum()
    ca = np.sum(w[D] * (da[D] == y[D])) / max(w[D].sum(), EPS)
    return {"P_D": float(PD), "P_a_correct_given_D": float(ca), "delta": float(PD * (2 * ca - 1)), "mask": D}


def audit_lower_bound(delta_measured, P_D, e, k, alpha=0.05):
    rho_u = clopper_pearson_upper(e, k, alpha)
    return {"rho_D_upper": rho_u, "delta_lower": delta_measured - 2 * P_D * rho_u}


def prop72(m, pi):
    """Persistent-noise example: measured and true edge of entity model over base-rate model."""
    assert (1 - pi) * m > 0.5
    return {"measured_edge": pi, "true_edge": -pi * (2 * m - 1),
            "measured_base": (1 - pi) * m, "true_base": m}
