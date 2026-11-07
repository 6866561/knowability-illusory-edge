"""Quantum vs classical on identical inputs (Section 8.3). Usage: python run_quantum.py <name> [<name> ...]

Names: d1_qubit (simulated, noisy theta), d2_lorenz (simulated, h=1, sigma=0.01), d7_earnings, or any
data/<name>.parquet with meta. Protocol: time split at the 70% quantile of t; train rows have avail < cut;
validation = last 15% of train; inputs = [logit EB entity rate] + domain features -> standardise -> PCA(d) -> [0, pi].
Writes results/quantum_<name>.json.
"""
import json, sys, pathlib, time, math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from kn import core, walk, quantum as Q

SEED = 20261106
D_QUBITS = 6
N_TRAIN, N_KERNEL, N_TEST = 20000, 3000, 30000


def load(name):
    rng = np.random.default_rng(SEED)
    if name == "d1_qubit":
        n = 40000
        th = rng.uniform(0, math.pi, n); that = th + rng.normal(0, 0.3, n)
        y = (rng.random(n) < np.cos(th / 2) ** 2).astype(int)
        df = pd.DataFrame({"t": np.arange(n, dtype=float), "entity": rng.integers(0, 50, n).astype(str), "y": y,
                           "x_th": that, "x_c": np.cos(that), "x_s": np.sin(that)})
        df["avail"] = df["t"] + 0.5; df["base_key"] = "0"
        return df, ["x_th", "x_c", "x_s"]
    if name == "d2_lorenz":
        import run_controls as rc
        X = rng.normal(0, 1, (40000, 3)) + np.array([0, 0, 25])
        for _ in range(3000):
            X = rc.rk4(X, 0.01)
        obs = X + rng.normal(0, 0.01, X.shape)
        T = X.copy()
        for _ in range(100):
            T = rc.rk4(T, 0.01)
        df = pd.DataFrame({"t": np.arange(len(X), dtype=float), "entity": rng.integers(0, 50, len(X)).astype(str),
                           "y": (T[:, 0] > 0).astype(int), "x_x": obs[:, 0], "x_y": obs[:, 1], "x_z": obs[:, 2]})
        df["avail"] = df["t"] + 0.5; df["base_key"] = "0"
        return df, ["x_x", "x_y", "x_z"]
    if name == "d7_earnings":
        d = pd.read_parquet(DATA / "d7_combined.parquet")
        df = pd.DataFrame({"t": pd.to_datetime(d["decision_time"]), "avail": pd.to_datetime(d["label_available"]),
                           "entity": d["entity"].astype(str), "y": d["y"].astype(int), "base_key": "all"})
        xc = []
        for c in ["z", "fund_ok", "k_known", "J", "period_kind", "h", "nhist"]:
            if c in d:
                df["x_" + c] = pd.to_numeric(d[c], errors="coerce").fillna(0).to_numpy(); xc.append("x_" + c)
        df["x_eb"] = walk.logit(d["eb_beta"].to_numpy()); xc.append("x_eb")
        return df, xc
    meta = json.loads((DATA / f"{name}.meta.json").read_text())
    return pd.read_parquet(DATA / f"{name}.parquet"), meta["xcols"]


def main(name):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df, xcols = load(name)
    df = walk.entity_rates(df)
    cols = (["x_eb"] if "x_eb" in df else []) + [c for c in xcols if c != "x_eb"]
    if "x_eb" not in df:
        df["lM2"] = walk.logit(df["p_M2"]); cols = ["lM2"] + list(xcols)
    cut = df["t"].quantile(0.7)
    tr = df[df["avail"] < cut].sort_values("t"); te = df[df["t"] >= cut]
    if len(tr) > N_TRAIN:
        tr = tr.iloc[np.sort(rng.choice(len(tr), N_TRAIN, replace=False))]
    if len(te) > N_TEST:
        te = te.iloc[np.sort(rng.choice(len(te), N_TEST, replace=False))]
    nv = int(0.15 * len(tr)); fit, val = tr.iloc[:-nv], tr.iloc[-nv:]
    Xf, Xv, Xt = (np.nan_to_num(a[cols].to_numpy(float)) for a in (fit, val, te))
    lo, hi = np.percentile(Xf, 1, 0), np.percentile(Xf, 99, 0)
    Xf, Xv, Xt = (np.clip(a, lo, hi) for a in (Xf, Xv, Xt))
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xf, Xv, Xt = ((a - mu) / sd for a in (Xf, Xv, Xt))
    d = min(D_QUBITS, Xf.shape[1])
    pca = PCA(d, random_state=SEED).fit(Xf)
    Pf, Pv, Pt = (pca.transform(a) for a in (Xf, Xv, Xt))
    qlo, qhi = np.percentile(Pf, 1, 0), np.percentile(Pf, 99, 0)
    sc = lambda a: np.clip((a - qlo) / (qhi - qlo + 1e-9), 0, 1) * math.pi
    Af, Av, At = sc(Pf), sc(Pv), sc(Pt)
    yf, yv, yt = fit["y"].to_numpy(), val["y"].to_numpy(), te["y"].to_numpy()
    preds, vpred = {"M0": te["p_M0"].to_numpy()}, {}
    def put(k, model, Xv_, Xt_):
        vpred[k] = model.predict_proba(Xv_)[:, 1]; preds[k] = model.predict_proba(Xt_)[:, 1]
    put("LR_full", LogisticRegression(C=0.5, max_iter=3000).fit(Xf, yf), Xv, Xt)
    put("LR_d", LogisticRegression(C=0.5, max_iter=3000).fit(Af, yf), Av, At)
    put("LR_d_strong", LogisticRegression(C=0.01, max_iter=3000).fit(Af, yf), Av, At)
    put("HGB_d", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED).fit(Af, yf), Av, At)
    fm = Q.fit_fourier(Af, yf); vpred["CQ1_fourier"] = fm.predict_proba(Q.fourier_features(Av))[:, 1]; preds["CQ1_fourier"] = fm.predict_proba(Q.fourier_features(At))[:, 1]
    timing = {}
    for L in (2, 4):
        s0 = time.time()
        m = Q.fit_vqc(Af, yf, Av, yv, d, layers=L, seed=SEED)   # Av is also used for early stopping of the VQC
        vpred[f"Q1_vqc_L{L}"] = Q.predict_vqc(m, Av); preds[f"Q1_vqc_L{L}"] = Q.predict_vqc(m, At); timing[f"Q1_L{L}"] = time.time() - s0
    ks = np.sort(rng.choice(len(Af), min(N_KERNEL, len(Af)), replace=False))
    Ak, yk = Af[ks], yf[ks]
    if len(np.unique(yk)) == 2:
        Sk, Sv, St = Q.zz_states(Ak), Q.zz_states(Av), Q.zz_states(At)
        svc = Q.fit_kernel_svc(Q.fidelity_kernel(Sk, Sk), yk, SEED)
        vpred["Q2_fidelity_kernel"] = svc.predict_proba(Q.fidelity_kernel(Sv, Sk))[:, 1]; preds["Q2_fidelity_kernel"] = svc.predict_proba(Q.fidelity_kernel(St, Sk))[:, 1]
        Bk, Bv, Bt = Q.bloch_vectors(Sk, d), Q.bloch_vectors(Sv, d), Q.bloch_vectors(St, d)
        g = 1.0 / (Bk.var() * Bk.shape[1] + 1e-9)
        svc = SVC(kernel="precomputed", probability=True, random_state=SEED).fit(Q.rbf(Bk, Bk, g), yk)
        vpred["Q3_projected_kernel"] = svc.predict_proba(Q.rbf(Bv, Bk, g))[:, 1]; preds["Q3_projected_kernel"] = svc.predict_proba(Q.rbf(Bt, Bk, g))[:, 1]
        g2 = 1.0 / (Ak.var() * Ak.shape[1] + 1e-9)
        svc = SVC(kernel="precomputed", probability=True, random_state=SEED).fit(Q.rbf(Ak, Ak, g2), yk)
        vpred["CQ2_rbf_svm"] = svc.predict_proba(Q.rbf(Av, Ak, g2))[:, 1]; preds["CQ2_rbf_svm"] = svc.predict_proba(Q.rbf(At, Ak, g2))[:, 1]
    groups = {k: np.asarray(v) for k, v in te.reset_index(drop=True).groupby("entity").indices.items()}
    res = {}
    for k, p in preds.items():
        res[k] = core.skill(yt, p, preds["M0"]); res[k]["auc"] = core.auc(yt, p)
        if k in vpred:
            res[k]["val_ll"] = core.log_loss(yv, vpred[k])
    qn = [k for k in vpred if k.startswith("Q")]
    cn_prereg = [k for k in ["LR_d", "HGB_d", "CQ1_fourier", "CQ2_rbf_svm"] if k in vpred]
    bq = min(qn, key=lambda k: res[k]["val_ll"]); bcp = min(cn_prereg, key=lambda k: res[k]["val_ll"])   # selected on validation
    bc = bcp
    ll = lambda p: -(yt * np.log(np.clip(p, 1e-6, 1)) + (1 - yt) * np.log(np.clip(1 - p, 1e-6, 1)))
    lq, lc = ll(preds[bq]), ll(preds[bcp]); lcp = lc
    dm = core.diebold_mariano(lq, lc); dm_prereg = dm
    bs = core.cluster_bootstrap(groups, lambda idx: lq[idx].mean() - lc[idx].mean(), reps=2000)
    from scipy import stats
    p_q_better = float(stats.norm.cdf(dm["dm"]))
    # one-sided p for H1 "quantum better" (ll diff Q-C < 0): share of bootstrap replicates that do NOT favour quantum.
    # (rev2 fix: an earlier version used bs <= 0, i.e. the opposite tail.)
    p_boot_q_better = float(np.mean(bs >= 0))
    tt = te["t"].reset_index(drop=True)
    if np.issubdtype(tt.dtype, np.datetime64):
        blk = tt.dt.to_period("M").astype(str)
    else:
        blk = pd.qcut(tt.rank(method="first"), 20, labels=False).astype(str)
    tgroups = {k: np.asarray(v) for k, v in pd.Series(blk).groupby(blk).indices.items()}
    bst = core.cluster_bootstrap(tgroups, lambda idx: lq[idx].mean() - lc[idx].mean(), reps=2000)
    p_boot_time = float(np.mean(bst >= 0))
    out = {"name": name, "n_fit": len(fit), "n_val": len(val), "n_test": len(te), "n_kernel": int(len(ks)), "d": d,
           "pca_var": pca.explained_variance_ratio_.tolist(), "results": res, "best_quantum": bq, "best_classical_same_input": bc,
           "ll_diff_q_minus_c": float(lq.mean() - lc.mean()), "ll_diff_ci": np.percentile(bs, [2.5, 97.5]).tolist(),
           "dm": dm, "p_quantum_better_one_sided": p_q_better, "p_boot_quantum_better": p_boot_q_better, "ll_diff_ci_time_block": np.percentile(bst, [2.5, 97.5]).tolist(),
           "p_boot_time_quantum_better": p_boot_time, "n_time_blocks": len(tgroups), "selection": "validation log loss",
           "prereg_classical_set": {"best": bcp, "ll_diff": float(lq.mean() - lcp.mean()), "dm": dm_prereg,
                                    "p_quantum_better_one_sided": float(stats.norm.cdf(dm_prereg["dm"]))}, "timing_s": timing, "secs": time.time() - t0}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"quantum2_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{name}: d={d} test={len(te)} " + " ".join(f"{k}={v['LSS']:.4f}" for k, v in res.items() if k != 'M0')
          + f" | best Q {bq} vs C {bc}: dLL={out['ll_diff_q_minus_c']:+.4f} CI={np.round(out['ll_diff_ci'], 4).tolist()} secs={out['secs']:.0f}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
