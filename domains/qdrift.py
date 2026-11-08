"""Quantum weather forecast: predictability of next-calibration values of IBM Heron processors.

Usage: python qdrift.py primary|secondary   (set OMP_NUM_THREADS=12)
Everything printed is computed; results go to results/qdrift_<src>.json (merged into results/qdrift.json).
"""
import sys, os, json, time, pathlib
K = str(pathlib.Path(__file__).resolve().parents[2])   # project root: repo/, data/, raw/ side by side
sys.path.insert(0, K + "/repo")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from kn import core

DEVS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
START = pd.Timestamp("2026-02-01", tz="UTC")
WARM = 30
REFIT = 14
HORIZONS = [1, 2, 4, 7, 14]
NB = 500
COVS = ["temperature_c", "pressure_hpa", "kp_index", "dst_nt", "neutron_flux"]
rng0 = np.random.default_rng(20261107)
ROWDIR = None


# ------------------------------------------------------------------ data
def load_primary():
    d = pd.read_parquet(K + "/data/qdrift_hf_subset.parquet")
    d = d[d.backend.isin(DEVS)]
    n0 = len(d)
    d = d[~d.is_failure_ceiling.fillna(False).astype(bool)]
    nfail = n0 - len(d)
    f = d.property_family
    bad = ((f == "readout_error") & (d.value >= 0.4)) | (f.isin(["T1", "T2"]) & (d.value <= 1.0))
    nbad = int(bad.sum())
    d = d[~bad]
    d = d.drop_duplicates(["backend", "property_family", "qubit_a", "qubit_b", "calibrated_time"])
    d["day"] = ((d.calibrated_time - START).dt.floor("D") / pd.Timedelta("1D")).astype(int)
    cov = d.groupby(["backend", "day"])[COVS].mean()
    d = d.sort_values("calibrated_time")
    meta = dict(rows_subset=n0, dropped_failure_ceiling=int(nfail), dropped_extra_failure=nbad)
    out = {}
    for fam, m in [("readout_error", "readout_error"), ("T1", "T1"), ("T2", "T2"), ("sx_error", "sx_error"), ("cz_error", "cz_error")]:
        x = d[d.property_family == fam]
        ent = x.qubit_a.astype(int).astype(str) if fam != "cz_error" else x.qubit_a.astype(int).astype(str) + "-" + x.qubit_b.astype(int).astype(str)
        out[m] = pd.DataFrame(dict(dev=x.backend.values, ent=ent.values, qa=x.qubit_a.astype(int).values, day=x.day.values, value=x.value.values))
    return out, cov, meta


def load_secondary():
    s = pd.read_csv(K + "/raw/qdrift/ibm_calibration_history.csv", usecols=["backend", "snapshot_time", "qubit", "T1", "T2", "readout_error"])
    s["ts"] = pd.to_datetime(s.snapshot_time, utc=True, format="mixed")
    n0 = len(s)
    s["day"] = ((s.ts - START).dt.floor("D") / pd.Timedelta("1D")).astype(int)
    s = s.sort_values("ts")
    meta = dict(rows=n0, snapshots=int(s.ts.nunique()), t0=str(s.ts.min()), t1=str(s.ts.max()))
    out = {}
    for m in ["readout_error", "T1", "T2"]:
        x = s[["backend", "qubit", "ts", "day", m]].rename(columns={m: "value"}).dropna()
        bad = (x.value >= 0.4) if m == "readout_error" else (x.value <= 1.0)
        meta["dropped_" + m] = int(bad.sum())
        x = x[~bad]
        # distinct calibrations: drop consecutive repeats of an identical value per (device, qubit)
        x = x.sort_values(["backend", "qubit", "ts"])
        ch = (x.groupby(["backend", "qubit"]).value.diff() != 0)
        meta["distinct_" + m] = int(ch.sum()); meta["polls_" + m] = len(x)
        x = x[ch]
        out[m] = pd.DataFrame(dict(dev=x.backend.values, ent=x.qubit.astype(str).values, qa=x.qubit.values, day=x.day.values, value=x.value.values))
    return out, None, meta


def to_panel(df, is_log=True):
    """-> dict with V (S x D log values, NaN missing), dev idx, cluster ids"""
    df = df[df.dev.isin(DEVS)]
    df = df.copy()
    df["v"] = np.log(np.maximum(df.value.values, 1e-6)) if is_log else df.value.values
    df = df.groupby(["dev", "ent", "qa", "day"], as_index=False).v.last()  # last calibration of the day (df pre-sorted by time)
    df = df[(df.day >= 0)]
    D = int(df.day.max()) + 1
    keys = df[["dev", "ent", "qa"]].drop_duplicates().reset_index(drop=True)
    keys["sid"] = np.arange(len(keys))
    df = df.merge(keys, on=["dev", "ent", "qa"])
    V = np.full((len(keys), D), np.nan)
    V[df.sid.values, df.day.values] = df.v.values
    keys["devi"] = keys.dev.map({d: i for i, d in enumerate(DEVS)})
    keys["cl"] = keys.dev + "#" + keys.qa.astype(str)
    return dict(V=V, keys=keys, D=D)


def ffill(V):
    S, D = V.shape
    idx = np.where(np.isnan(V), -1, np.arange(D)[None, :])
    L = np.maximum.accumulate(idx, axis=1)
    F = np.take_along_axis(V, np.maximum(L, 0), axis=1)
    F[L < 0] = np.nan
    return F, L


def roll(V, w):
    ok = ~np.isnan(V); x = np.where(ok, V, 0.0)
    cs = np.cumsum(np.pad(x, ((0, 0), (1, 0))), 1); cn = np.cumsum(np.pad(ok.astype(float), ((0, 0), (1, 0))), 1)
    c2 = np.cumsum(np.pad(x * x, ((0, 0), (1, 0))), 1)
    D = V.shape[1]; hi = np.arange(1, D + 1); lo = np.maximum(hi - w, 0)
    n = cn[:, hi] - cn[:, lo]; s = cs[:, hi] - cs[:, lo]; s2 = c2[:, hi] - c2[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n; var = s2 / n - m * m
    m[n == 0] = np.nan
    sd = np.sqrt(np.maximum(var, 0)); sd[n < 2] = np.nan
    return m, sd, n


# ------------------------------------------------------------------ Task A
def featA(P, covarr):
    V = P["V"]; keys = P["keys"]; S, D = V.shape
    F, L = ffill(V)
    ok = ~np.isnan(V); x = np.where(ok, V, 0.0)
    cnt = np.cumsum(ok, 1); clim = np.cumsum(x, 1) / np.maximum(cnt, 1); clim[cnt == 0] = np.nan
    feats = {"last": F, "age": np.where(L >= 0, np.arange(D)[None, :] - L, np.nan), "clim_gap": clim - F, "nobs": cnt.astype(float)}
    for k in range(1, 7):
        lag = np.full_like(F, np.nan); lag[:, k:] = F[:, :-k]; feats[f"lag{k}"] = lag - F
    for w in (7, 14):
        m, sd, n = roll(V, w); feats[f"rm{w}"] = m - F; feats[f"rs{w}"] = sd
    # device anomaly (cross-sectional) at day d: mean over series of device of (ffilled last - own climatology)
    A = F - clim
    dm = np.full_like(F, np.nan)
    for i in range(len(DEVS)):
        sel = (keys.devi.values == i)
        if sel.any():
            with np.errstate(all="ignore"):
                dm[sel] = np.nanmean(A[sel], axis=0)[None, :]
    feats["dev_anom"] = dm; feats["own_anom"] = A - dm
    for i in range(len(DEVS)):
        feats[f"dev{i}"] = np.where(keys.devi.values[:, None] == i, 1.0, 0.0) * np.ones((S, D))
    if covarr is not None:
        for j, c in enumerate(COVS):
            feats["cov_" + c] = covarr[keys.devi.values, :, j][:, :D]
    names = list(feats)
    X = np.stack([feats[n] for n in names], axis=-1)  # S x D x nf
    return X, names, F, clim, cnt


def make_cov(cov, D):
    arr = np.full((len(DEVS), D, len(COVS)), np.nan)
    for i, dv in enumerate(DEVS):
        if dv in cov.index.get_level_values(0):
            c = cov.loc[dv]; c = c[(c.index >= 0) & (c.index < D)]
            arr[i, c.index.values] = c[COVS].values
    return arr


def run_taskA(P, covarr, h, model_cap=100000):
    V = P["V"]; S, D = V.shape
    X, names, F, clim, cnt = featA(P, covarr)
    last = F
    org = np.arange(WARM, D - h)
    Y = V[:, org + h]
    valid = (~np.isnan(Y)) & (~np.isnan(last[:, org])) & (cnt[:, org] >= 5)
    si, oi = np.where(valid)
    o = org[oi]
    Xr = X[si, o, :]; lastr = last[si, o]; yr = V[si, o + h]; clr = clim[si, o]
    tgt = yr - lastr
    res = dict(o=o, si=si, y=yr, pers=lastr, clim=clr)
    pr = np.full(len(o), np.nan); ph = np.full(len(o), np.nan)
    refits = list(range(WARM, D - h, REFIT))
    med_all = None
    for T in refits:
        tr = (o + h <= T); te = (o >= T) & (o < T + REFIT)
        if te.sum() == 0 or tr.sum() < 500: continue
        itr = np.where(tr)[0]
        if len(itr) > model_cap: itr = rng0.choice(itr, model_cap, replace=False)
        Xt, yt = Xr[itr], tgt[itr]
        med = np.nanmedian(Xt, axis=0); med = np.where(np.isnan(med), 0, med)
        imp = lambda Z: np.where(np.isnan(Z), med[None, :], Z)
        sc = StandardScaler().fit(imp(Xt))
        zc = lambda Z: np.clip(sc.transform(imp(Z)), -5, 5)  # clip standardized features (guards near-constant covariate columns)
        rg = Ridge(alpha=10.0).fit(zc(Xt), yt)
        pr[te] = lastr[te] + rg.predict(zc(Xr[te]))
        ck = np.where((~np.isnan(Xt)).sum(0) >= 20)[0]
        hg = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(Xt[:, ck], yt)
        ph[te] = lastr[te] + hg.predict(Xr[te][:, ck])
    res["ridge"] = pr; res["hgb"] = ph
    m = ~np.isnan(ph)
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in res.items()}


def clus_boot(cl, se_dict, pairs, reps=NB):
    """cluster bootstrap via core.cluster_bootstrap on a cluster-level table of sums."""
    u, inv = np.unique(cl, return_inverse=True)
    names = list(se_dict)
    tab = np.stack([np.bincount(inv, weights=se_dict[n], minlength=len(u)) for n in names] + [np.bincount(inv, minlength=len(u)).astype(float)], 1)
    groups = {i: np.array([i]) for i in range(len(u))}
    ix = {n: i for i, n in enumerate(names)}
    def stat(idx):
        s = tab[idx].sum(0)
        return [1 - s[ix[a]] / s[ix[b]] for a, b in pairs]
    arr = core.cluster_bootstrap(groups, stat, reps=reps)
    pt = tab.sum(0)
    point = [1 - pt[ix[a]] / pt[ix[b]] for a, b in pairs]
    return point, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)


def summarize_A(res, P, h):
    keys = P["keys"]; cl = keys.cl.values[res["si"]]
    se = {m: (res[m] - res["y"]) ** 2 for m in ["pers", "clim", "ridge", "hgb"]}
    mse = {m: float(v.mean()) for m, v in se.items()}
    pairs = [(m, "clim") for m in ["pers", "ridge", "hgb"]] + [(m, "pers") for m in ["ridge", "hgb"]]
    pt, lo, hi = clus_boot(cl, se, pairs)
    out = dict(n=int(len(cl)), n_series=int(len(np.unique(res["si"]))), n_clusters=int(len(np.unique(cl))), mse=mse)
    for (a, b), p, l, u in zip(pairs, pt, lo, hi):
        out[f"skill_{a}_vs_{b}"] = dict(skill=float(p), lo=float(l), hi=float(u))
    return out


# ------------------------------------------------------------------ Task B
def expanding_thr(Vraw, devi, q, below):
    """per device, threshold at day t from pooled daily values of days <= t-1 (strictly past). returns S-less array dev x D"""
    D = Vraw.shape[1]
    thr = np.full((len(DEVS), D), np.nan)
    for i in range(len(DEVS)):
        sel = devi == i
        if not sel.any(): continue
        for t in range(WARM, D):
            past = Vraw[sel, :t].ravel(); past = past[~np.isnan(past)]
            if len(past) > 200: thr[i, t] = np.percentile(past, q)
    return thr


def run_taskB(Pro, Pt1, covarr, h=1, with_cov=True):
    # align panels on common (dev, qa) qubit set
    kr, kt = Pro["keys"], Pt1["keys"]
    k = kr[["dev", "qa", "cl", "devi"]].merge(kt[["dev", "qa"]].assign(j=np.arange(len(kt))), on=["dev", "qa"])
    # order by k
    ir = np.array([kr.index[(kr.dev == a) & (kr.qa == b)][0] for a, b in zip(k.dev, k.qa)])
    it = k.j.values
    D = min(Pro["D"], Pt1["D"])
    RO = Pro["V"][ir, :D]; T1 = Pt1["V"][it, :D]; devi = k.devi.values.astype(int); cl = np.asarray(k.cl.values, dtype=object).astype(str)
    thr_ro = expanding_thr(RO, devi, 90, False); thr_t1 = expanding_thr(T1, devi, 10, True)
    S = len(k)
    TRo = thr_ro[devi]; TT1 = thr_t1[devi]
    bad_ro = RO > TRo; bad_t1 = T1 < TT1
    obs = ~np.isnan(RO) & ~np.isnan(T1) & ~np.isnan(TRo) & ~np.isnan(TT1)
    Y = np.where(obs, (bad_ro | bad_t1).astype(float), np.nan)
    bad_ro_only = np.where(obs, bad_ro, np.nan)
    # persistence of badness for several lags
    pers = {}
    for lag in [1, 2, 4, 7, 14]:
        a = Y[:, WARM:D - lag]; b = Y[:, WARM + lag:D]
        m = ~np.isnan(a) & ~np.isnan(b)
        cl_u, inv = np.unique(np.repeat(cl[:, None], a.shape[1], 1)[m], return_inverse=True)
        a_, b_ = a[m], b[m]
        tab = np.stack([np.bincount(inv, weights=(a_ == 1) * (b_ == 1), minlength=len(cl_u)), np.bincount(inv, weights=(a_ == 1), minlength=len(cl_u)),
                        np.bincount(inv, weights=(a_ == 0) * (b_ == 1), minlength=len(cl_u)), np.bincount(inv, weights=(a_ == 0), minlength=len(cl_u)),
                        np.bincount(inv, weights=b_, minlength=len(cl_u)), np.bincount(inv, minlength=len(cl_u)).astype(float)], 1)
        groups = {i: np.array([i]) for i in range(len(cl_u))}
        def st(idx):
            s = tab[idx].sum(0); return [s[0] / s[1], s[2] / s[3], s[4] / s[5]]
        bs = core.cluster_bootstrap(groups, st, reps=NB)
        s = tab.sum(0)
        pers[lag] = dict(p_bad_given_bad=float(s[0] / s[1]), ci=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                         p_bad_given_good=float(s[2] / s[3]), ci_good=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                         base_rate=float(s[4] / s[5]), n_pairs=int(s[5]), n_bad_origin=int(s[1]))
    # features at origin d for target d+h (h=1)
    Fro, _ = ffill(RO); Ft1, _ = ffill(T1)
    # threshold at target day t+h is known at day t+h using data <= t+h-1 = origin data (h=1)
    zr = np.where(np.isnan(Fro), np.nan, Fro - np.roll(TRo, -1, axis=1)); zt = np.where(np.isnan(Ft1), np.nan, Ft1 - np.roll(TT1, -1, axis=1))
    Yf = np.where(np.isnan(Y), 0.0, Y); Yo = (~np.isnan(Y)).astype(float)
    cb = np.cumsum(Yf, 1); cn = np.cumsum(Yo, 1)
    rate_q = (cb + 0.15 * 20) / (cn + 20)   # shrunk past bad rate of the qubit (all labelled days <= d)
    f = {"zr": zr, "zt": zt}
    for kk in range(1, 7):
        for nm, z in (("r", zr), ("t", zt)):
            lag = np.full_like(z, np.nan); lag[:, kk:] = z[:, :-kk]; f[f"{nm}lag{kk}"] = lag
    Yff, _ = ffill(Y); f["bad_now"] = Yff
    for w in (7, 14):
        m, _, n = roll(Y, w); f[f"badrate{w}"] = m
    f["rate_q"] = rate_q; f["n_lab"] = cn
    fr = np.full_like(Yf, np.nan)
    for i in range(len(DEVS)):
        sel = devi == i
        with np.errstate(all="ignore"): fr[sel] = np.nanmean(Y[sel], axis=0)[None, :]
    f["dev_bad_frac"] = fr
    for i in range(len(DEVS)): f[f"dev{i}"] = (devi[:, None] == i) * np.ones((S, D))
    if with_cov and covarr is not None:
        for j, c in enumerate(COVS): f["cov_" + c] = covarr[devi, :, j][:, :D]
    names = list(f); X = np.stack([f[n] for n in names], -1)
    org = np.arange(WARM, D - h)
    yt = Y[:, org + h]
    valid = ~np.isnan(yt) & (cn[:, org] >= 5)
    si, oi = np.where(valid); o = org[oi]
    Xr = X[si, o]; y = Y[si, o + h].astype(int)
    # expanding device base rate with labels known at origin (target day <= o -> label day <= o)
    base = np.full(len(o), np.nan)
    for i in range(len(DEVS)):
        sel = devi == i
        cbd = np.cumsum(np.nansum(Y[sel], 0)); cnd = np.cumsum((~np.isnan(Y[sel])).sum(0))
        mm = devi[si] == i
        base[mm] = (cbd[o[mm]] + 1) / (cnd[o[mm]] + 2)
    p_q = rate_q[si, o]
    p_h = np.full(len(o), np.nan); p_l = np.full(len(o), np.nan)
    for T in range(WARM, D - h, REFIT):
        tr = (o + h <= T); te = (o >= T) & (o < T + REFIT)
        if te.sum() == 0 or tr.sum() < 500: continue
        itr = np.where(tr)[0]
        if len(itr) > 150000: itr = rng0.choice(itr, 150000, replace=False)
        if len(np.unique(y[itr])) < 2: continue
        ck = np.where((~np.isnan(Xr[itr])).sum(0) >= 20)[0]
        hg = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(Xr[itr][:, ck], y[itr])
        p_h[te] = hg.predict_proba(Xr[te][:, ck])[:, 1]
        med = np.nanmedian(Xr[itr], axis=0); med = np.where(np.isnan(med), 0, med)
        imp = lambda Z: np.where(np.isnan(Z), med[None, :], Z)
        sc = StandardScaler().fit(imp(Xr[itr]))
        zc = lambda Z: np.clip(sc.transform(imp(Z)), -5, 5)
        lr = LogisticRegression(C=0.1, max_iter=300).fit(zc(Xr[itr]), y[itr])
        p_l[te] = lr.predict_proba(zc(Xr[te]))[:, 1]
    m = ~np.isnan(p_h)
    y, base, p_q, p_h, p_l, si, o = y[m], base[m], p_q[m], p_h[m], p_l[m], si[m], o[m]
    clr = cl[si]
    uc, inv = np.unique(clr, return_inverse=True)
    groups = {i: np.where(inv == i)[0] for i in range(len(uc))}
    out = dict(n_rows=int(len(y)), n_qubits=int(len(uc)), n_days_eval=int(len(np.unique(o))), base_rate=float(y.mean()),
               persistence=pers)
    for nm, p in [("qubit_rate", p_q), ("logit", p_l), ("hgb", p_h)]:
        sk = core.skill(y, p, base)
        def stat(idx, p=p):
            r = core.skill(y[idx], p[idx], base[idx]); return [r["LSS"], r["BSS"], r["acc"], r["acc0"], r["S_acc"]]
        bs = core.cluster_bootstrap(groups, stat, reps=NB)
        sk["ci"] = {n: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))] for j, n in enumerate(["LSS", "BSS", "acc", "acc0", "S_acc"])}
        out[nm] = {k_: (float(v) if not isinstance(v, dict) else v) for k_, v in sk.items()}
    out["auc_hgb"] = core.auc(y, p_h); out["auc_qubit_rate"] = core.auc(y, p_q)
    # entity ceiling on per-qubit bad counts over evaluation rows
    kq = np.bincount(inv, weights=y); nq = np.bincount(inv).astype(float)
    a, b = core.mom_beta(kq, nq)
    bc = core.beta_ceiling(a, b)
    out["ceiling"] = dict(a=float(a), b=float(b), qubit_rate_spread=float(np.std(kq / nq)), mean_n=float(nq.mean()), **{k_: float(v) for k_, v in bc.items()},
                          LSS_ceiling=float(bc["kappa"]))
    out["label_def"] = "bad = readout_error > device expanding P90 (strictly past days) OR T1 < device expanding P10; both observed on target day"
    out["bad_readout_share_among_eval_bad"] = float(np.nanmean(bad_ro_only[si, o + h][y == 1]))
    if ROWDIR is not None:   # row-level predictions for dependence-robust inference (qdrift_twoway.py)
        np.savez_compressed(os.path.join(ROWDIR, "taskB.npz"), y=y, base=base, p_q=p_q, p_l=p_l, p_h=p_h, o=o, cl=np.asarray(clr, dtype="U"), devi=devi[si])
    return out


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    src = sys.argv[1]
    t0 = time.time()
    ROWDIR = os.path.join(K, "repo", "results", f"qdrift_rows_{src}"); os.makedirs(ROWDIR, exist_ok=True)
    if src == "primary":
        raw, cov, meta = load_primary(); metrics = ["readout_error", "T1", "T2", "sx_error", "cz_error"]
    else:
        raw, cov, meta = load_secondary(); metrics = ["readout_error", "T1", "T2"]
    if len(sys.argv) > 2: metrics = sys.argv[2].split(",")
    panels = {m: to_panel(raw[m]) for m in raw}
    D = max(p["D"] for p in panels.values())
    covarr = make_cov(cov, D) if cov is not None else None
    R = dict(source=src, meta=meta, window_days=D, start=str(START.date()), warmup=WARM, refit_days=REFIT)
    R["data"] = {m: dict(series=int(len(p["keys"])), by_device={d: int((p["keys"].dev == d).sum()) for d in DEVS},
                         qubits=int(p["keys"].cl.nunique()), days=int(p["D"]),
                         obs=int(np.sum(~np.isnan(p["V"]))), first_obs_day=str((START + pd.Timedelta(days=int(np.where((~np.isnan(p["V"])).any(0))[0].min()))).date())) for m, p in panels.items()}
    print(json.dumps(R["data"], indent=1), flush=True)
    R["taskA"] = {}
    for m in metrics:
        R["taskA"][m] = {}
        for h in HORIZONS:
            res = run_taskA(panels[m], covarr, h)
            np.savez_compressed(os.path.join(ROWDIR, f"taskA_{m}_h{h}.npz"), o=res["o"], y=res["y"], pers=res["pers"], clim=res["clim"],
                                ridge=res["ridge"], hgb=res["hgb"], cl=np.asarray(panels[m]["keys"].cl.to_numpy(dtype=object)[res["si"]], dtype="U"))
            R["taskA"][m][f"h{h}"] = summarize_A(res, panels[m], h)
            s = R["taskA"][m][f"h{h}"]
            print(m, h, "n=%d" % s["n"], {k: (round(v["skill"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in s.items() if k.startswith("skill")}, "mse", {k: round(v, 5) for k, v in s["mse"].items()}, "t=%.0fs" % (time.time() - t0), flush=True)
    if "readout_error" in panels and "T1" in panels:
        R["taskB"] = run_taskB(panels["readout_error"], panels["T1"], covarr)
        print(json.dumps(R["taskB"], indent=1), flush=True)
    os.makedirs(K + "/repo/results", exist_ok=True)
    json.dump(R, open(K + f"/repo/results/qdrift_{src}.json", "w"), indent=1, default=float)
    print("done %.0fs" % (time.time() - t0))
