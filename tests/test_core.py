import math
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kn import core, walk

rng = np.random.default_rng(1)


def test_beta_ceiling():
    c = core.beta_ceiling(3, 1.5)
    assert abs(c["A"] - 0.72372) < 2e-4 and abs(c["Eh"] - 0.53930) < 2e-4
    x = rng.beta(3, 1.5, 400000)
    assert abs(np.maximum(x, 1 - x).mean() - c["A"]) < 3e-3


def test_learning_curve_limits():
    lc = core.beta_learning_curve(3, 1.5, 200)
    c = core.beta_ceiling(3, 1.5)
    assert abs(lc[0] - c["A0"]) < 1e-12
    assert lc[200] < c["A"] + 1e-9 and c["A"] - lc[200] < 0.01
    assert np.all(np.diff(lc[::2]) > -1e-9)


def test_qubit():
    th = rng.uniform(0, math.pi, 400000)
    p = np.cos(th / 2) ** 2
    assert abs(np.maximum(p, 1 - p).mean() - core.qubit_ceiling()["A"]) < 2e-3
    assert abs(core.h(p).mean() - (2 * math.log(2) - 1)) < 2e-3
    for lam in (0.05, 0.2):
        q = (1 - lam) * p + lam / 2
        assert abs(np.maximum(q, 1 - q).mean() - core.qubit_ceiling(lam)["A"]) < 2e-3


def test_prop71_independent_noise_shrinks():
    n, rho = 400000, 0.1
    y = (rng.random(n) < 0.7).astype(int)
    pa = np.where(rng.random(n) < 0.8, y, 1 - y).astype(float)  # 80% accurate
    pb = np.ones(n)
    flip = rng.random(n) < rho
    yt = np.where(flip, 1 - y, y)
    d_true = core.accuracy(y, pa) - core.accuracy(y, pb)
    d_meas = core.accuracy(yt, pa) - core.accuracy(yt, pb)
    assert abs(d_meas - (1 - 2 * rho) * d_true) < 5e-3


def test_prop72_persistent_noise_manufactures_edge():
    m, pi, E, T = 0.7, 0.2, 2000, 60
    ent = np.repeat(np.arange(E), T)
    y = (rng.random(E * T) < m).astype(int)
    corrupt = rng.random(E) < pi
    yt = np.where(corrupt[ent], 0, y)
    ent_pred = np.where(corrupt[ent], 0.0, 1.0)  # entity model at convergence
    base = np.ones(E * T)
    pi_hat = corrupt.mean()
    meas = core.accuracy(yt, ent_pred) - core.accuracy(yt, base)
    true = core.accuracy(y, ent_pred) - core.accuracy(y, base)
    r = core.prop72(m, pi_hat)
    assert abs(meas - r["measured_edge"]) < 0.01 and abs(true - r["true_edge"]) < 0.01
    assert meas > 0 > true


def test_disagreement_identity_and_bound():
    n = 100000
    y = (rng.random(n) < 0.6).astype(int)
    pa, pb = rng.random(n), rng.random(n)
    d = core.disagreement_decomposition(y, pa, pb)
    assert abs(d["delta"] - (core.accuracy(y, pa) - core.accuracy(y, pb))) < 1e-9
    # adversarial noise on D: flip labels where a is wrong (helps a)
    D = d["mask"]
    wrong_a = D & (((pa >= .5).astype(int)) != y)
    idx = np.flatnonzero(wrong_a)[: int(0.05 * D.sum())]
    yt = y.copy(); yt[idx] = 1 - yt[idx]
    meas = core.accuracy(yt, pa) - core.accuracy(yt, pb)
    assert abs(meas - d["delta"]) <= 2 * d["P_D"] * 0.05 + 1e-9


def test_walk_forward_no_future_labels():
    n = 3000
    df = pd.DataFrame({"t": np.arange(n), "entity": rng.integers(0, 30, n), "y": rng.integers(0, 2, n)})
    df["avail"] = df["t"] + 5
    df["base_key"] = 0
    out = walk.entity_rates(df)
    # recompute M1 by brute force for some rows
    for i in rng.integers(0, n, 50):
        r = out.iloc[i]
        past = out[(out["avail"] < r["t"]) & (out["entity"] == r["entity"])]
        assert abs(r["p_M1"] - (1 + past["y"].sum()) / (2 + len(past))) < 1e-12


def test_lz_iid_coin_near_one_bit():
    s = rng.integers(0, 2, 4000)
    S = core.lz_entropy_rate(s)
    assert 0.8 < S < 1.2
