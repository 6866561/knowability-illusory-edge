"""Point-in-time walk-forward runner shared by every real-data domain.

Domain frame schema (one row per forecast):
    t        decision time (sortable: datetime64 or float)
    avail    time the label becomes known (> t)
    entity   entity id (station, team, district, ticker, cell)
    y        0/1 outcome
    base_key key for the base-rate model M0 (pooled = constant; weather = station-month)
    x_*      feature columns, known at t
    bench    optional benchmark probability (odds, NWP), known at t
Models:
    M0 expanding frequency within base_key          (F0)
    M1 Laplace entity frequency (1+k)/(2+n)          (F1)
    M2 empirical-Bayes beta-binomial (MoM prior)     (F1)
    M4 L2 logistic on [logit M2, x_*]                (F2)
    M5 histogram gradient boosting on same inputs    (F2)
    M6 1-NN on standardized x_* (feeds Cover-Hart)   (F2)
    M4b logistic on [logit M2, x_*, logit bench]     (F3) if bench present
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from . import core

SEED = 20261106


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def entity_rates(df: pd.DataFrame) -> pd.DataFrame:
    """M0, M1, M2 forecasts using only labels with avail < t. Returns df with columns p_M0, p_M1, p_M2, n_hist."""
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    lab = df[["avail", "entity", "base_key", "y"]].sort_values("avail", kind="mergesort").to_numpy()
    j, nlab = 0, len(lab)
    ek, en = defaultdict(float), defaultdict(float)
    bk, bn = defaultdict(float), defaultdict(float)
    K = N = 0.0
    out = np.zeros((len(df), 4))
    ts = df["t"].to_numpy()
    ents = df["entity"].to_numpy()
    bkeys = df["base_key"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        t = ts[s]
        while j < nlab and lab[j][0] < t:
            _, en_, bk_, y = lab[j]
            ek[en_] += y; en[en_] += 1; bk[bk_] += y; bn[bk_] += 1; K += y; N += 1
            j += 1
        # MoM prior from entities with >= 2 matured outcomes
        ks = np.array([ek[x] for x in en if en[x] >= 2]); ns = np.array([en[x] for x in en if en[x] >= 2])
        if len(ks) >= 3:
            a, b = core.mom_beta(ks, ns)
        else:
            a = b = 1.0
        pooled = (1 + K) / (2 + N)
        for i in range(s, e):
            x, q = ents[i], bkeys[i]
            out[i, 0] = (1 + bk[q]) / (2 + bn[q]) if bn[q] > 0 else pooled
            out[i, 1] = (1 + ek[x]) / (2 + en[x])
            out[i, 2] = np.clip((a + ek[x]) / (a + b + en[x]), 0.01, 0.99)
            out[i, 3] = en[x]
    df["p_M0"], df["p_M1"], df["p_M2"], df["n_hist"] = out.T
    return df


def _prep(train, test, cols):
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    lo, hi = np.nanpercentile(Xtr, 1, axis=0), np.nanpercentile(Xtr, 99, axis=0)
    Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return np.nan_to_num(Xtr), np.nan_to_num(Xte)


def feature_models(df: pd.DataFrame, cutoffs, xcols, min_train=200, bench=False, knn=True, leo_folds=0):
    """Refit M4/M5/M6 at each cutoff on rows with avail < cutoff; predict rows with cutoff <= t < next."""
    df = df.copy()
    df["lM2"] = logit(df["p_M2"])
    cols = ["lM2"] + list(xcols)
    for c in ["p_M4", "p_M5", "p_M6", "p_M4b", "p_M4x_leo", "p_M4x"]:
        df[c] = np.nan
    if bench:
        df["lbench"] = logit(df["bench"])
    cut = list(cutoffs) + [None]
    rng = np.random.default_rng(SEED)
    ents = df["entity"].unique()
    fold_of = dict(zip(ents, rng.integers(0, max(leo_folds, 1), len(ents))))
    for c0, c1 in zip(cut[:-1], cut[1:]):
        te = (df["t"] >= c0) if c1 is None else (df["t"] >= c0) & (df["t"] < c1)
        tr = df["avail"] < c0
        if te.sum() == 0 or tr.sum() < min_train or df.loc[tr, "y"].nunique() < 2:
            continue
        train, test = df[tr], df[te]
        Xtr, Xte = _prep(train, test, cols)
        ytr = train["y"].to_numpy()
        lr = LogisticRegression(C=0.5, max_iter=2000).fit(Xtr, ytr)
        df.loc[te, "p_M4"] = lr.predict_proba(Xte)[:, 1]
        hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=SEED).fit(Xtr, ytr)
        df.loc[te, "p_M5"] = hgb.predict_proba(Xte)[:, 1]
        if xcols:
            Xtr2, Xte2 = _prep(train, test, list(xcols))
            df.loc[te, "p_M4x"] = LogisticRegression(C=0.5, max_iter=2000).fit(Xtr2, ytr).predict_proba(Xte2)[:, 1]
            if knn:  # 1-NN on the full F2 inputs (entity rate + features) so Cover-Hart bounds the same information set as M4
                sub = rng.permutation(len(Xtr))[:50000]
                nn = KNeighborsClassifier(n_neighbors=1).fit(Xtr[sub], ytr[sub])
                df.loc[te, "p_M6"] = nn.predict(Xte).astype(float)
            if leo_folds:
                for f in range(leo_folds):
                    tef = te & df["entity"].map(fold_of).eq(f)
                    trf = tr & ~df["entity"].map(fold_of).eq(f)
                    if tef.sum() == 0 or df.loc[trf, "y"].nunique() < 2:
                        continue
                    a, b = _prep(df[trf], df[tef], list(xcols))
                    m = LogisticRegression(C=0.5, max_iter=2000).fit(a, df.loc[trf, "y"].to_numpy())
                    df.loc[tef, "p_M4x_leo"] = m.predict_proba(b)[:, 1]
        if bench:
            c2 = cols + ["lbench"]
            a, b = _prep(train, test, c2)
            df.loc[te, "p_M4b"] = LogisticRegression(C=0.5, max_iter=2000).fit(a, ytr).predict_proba(b)[:, 1]
    return df


def summarize(df, models, base="p_M0", groups_col="entity", reps=1000):
    """Skill table with entity-clustered bootstrap CIs for accuracy and LSS differences vs base."""
    rows = {}
    groups = {k: v.to_numpy() for k, v in df.groupby(groups_col).indices.items()} if False else \
        {k: np.asarray(v) for k, v in df.groupby(groups_col).indices.items()}
    y = df["y"].to_numpy()
    for m in models:
        if m not in df or df[m].isna().all():
            continue
        ok = df[m].notna().to_numpy() & df[base].notna().to_numpy()
        p, p0 = df[m].to_numpy(), df[base].to_numpy()
        s = core.skill(y[ok], p[ok], p0[ok])
        s["n"] = int(ok.sum())
        s.update(core.murphy(y[ok], p[ok]))
        s["ece"] = core.ece(y[ok], p[ok])
        s["auc"] = core.auc(y[ok], p[ok])
        g2 = {k: v[ok[v]] for k, v in groups.items()}
        g2 = {k: v for k, v in g2.items() if len(v)}

        def stat(idx):
            return (core.accuracy(y[idx], p[idx]) - core.accuracy(y[idx], p0[idx]),
                    1 - core.log_loss(y[idx], p[idx]) / core.log_loss(y[idx], p0[idx]))
        bs = core.cluster_bootstrap(g2, stat, reps=reps)
        s["dacc_ci"] = np.percentile(bs[:, 0], [2.5, 97.5]).tolist()
        s["LSS_ci"] = np.percentile(bs[:, 1], [2.5, 97.5]).tolist()
        s["half_split_win"] = core.half_split_win(
            g2, lambda idx: core.accuracy(y[idx], p[idx]) - core.accuracy(y[idx], p0[idx]), reps=reps)
        rows[m] = s
    return rows


def ceilings(df, ent_col="entity"):
    """Entity beta-binomial ceiling, hindsight majority, Cover-Hart from p_M6."""
    g = df.groupby(ent_col)["y"].agg(["sum", "count"])
    a, b = core.mom_beta(g["sum"], g["count"])
    bc = core.beta_ceiling(a, b)
    hind = float(np.sum(np.maximum(g["sum"], g["count"] - g["sum"])) / g["count"].sum())
    out = {"alpha": a, "beta": b, "beta_ceiling": bc, "hindsight_entity_majority": hind,
           "learning_curve": core.beta_learning_curve(a, b, 40).tolist(),
           "median_hist": float(df["n_hist"].median())}
    if "p_M6" in df and df["p_M6"].notna().any():
        ok = df["p_M6"].notna()
        r_nn = 1 - core.accuracy(df.loc[ok, "y"], df.loc[ok, "p_M6"])
        ch = core.cover_hart_upper(r_nn)
        out["r_nn"] = r_nn
        out["cover_hart"] = ch
        H = float(core.h(df["y"].mean()))
        out["kappa_upper"] = core.kappa_upper_from_rstar(ch["R_star_lower"], H)
    return out
