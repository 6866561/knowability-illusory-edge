"""D7 earnings guidance: knowability analysis on the reproduced package outputs.

Uses outcome-group weights exactly as the package (1 / versions per fiscal outcome). Writes results/d7_analysis.json
and results/d7_earnings_entity.json (entity-rung ceilings for the shared figures).
"""
import json, sys, pathlib, math
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
sys.path.insert(0, str(ROOT))
from kn import core, walk

SEED = 20261106


def wacc(y, p, w):
    return core.accuracy(y, p, w)


def groups_of(df, col="entity"):
    return {k: np.asarray(v) for k, v in df.reset_index(drop=True).groupby(col).indices.items()}


def boot_ci(df, fn, reps=2000, col="entity"):
    g = groups_of(df, col)
    b = core.cluster_bootstrap(g, fn, reps=reps)
    return np.percentile(b, [2.5, 97.5]).tolist()


def pit_1nn_error(X, y, t, avail, w):
    """Point-in-time 1-NN: each row's neighbour drawn only from rows whose label was known before its decision time."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    err, wt = 0.0, 0.0
    for i in range(len(X)):
        m = avail < t[i]
        if m.sum() < 20:
            continue
        d = ((X[m] - X[i]) ** 2).sum(1)
        j = np.argmin(d)
        err += w[i] * (y[m][j] != y[i]); wt += w[i]
    return err / wt


def main():
    out = {}
    d = pd.read_parquet(DATA / "d7_combined.parquet").reset_index(drop=True)
    y, w = d["y"].to_numpy().astype(int), d["weight"].to_numpy()
    P = {k: d[k].to_numpy() for k in ["freq", "eb_beta", "combo", "ctrl_lite", "struct_t"]}
    base = P["freq"]
    table = {}
    for k, p in P.items():
        s = core.skill(y, p, base, w)
        s["auc"] = core.auc(y, p)
        s["dacc_ci"] = boot_ci(d, lambda idx, p=p: wacc(y[idx], p[idx], w[idx]) - wacc(y[idx], base[idx], w[idx]))
        s["LSS_ci"] = boot_ci(d, lambda idx, p=p: 1 - core.log_loss(y[idx], p[idx], w[idx]) / core.log_loss(y[idx], base[idx], w[idx]))
        s["half_split_win"] = core.half_split_win(groups_of(d), lambda idx, p=p: wacc(y[idx], p[idx], w[idx]) - wacc(y[idx], base[idx], w[idx]))
        table[k] = s
    out["combined"] = {"n": len(d), "groups": int(d["outcome_group"].nunique()), "issuers": int(d["entity"].nunique()),
                       "base_rate_w": float(np.sum(w * y) / w.sum()), "table": table}
    # by slice
    sl = {}
    for name, m in [("annual", d["period_kind"] == 1), ("quarterly", d["period_kind"] == 0),
                    ("audited20", d["src"] == "audited20"), ("harvested", d["src"] != "audited20")]:
        m = m.to_numpy()
        sl[name] = {k: {"acc": wacc(y[m], P[k][m], w[m]), "ll": core.log_loss(y[m], P[k][m], w[m])} for k in ["freq", "eb_beta", "combo"]}
        sl[name]["n"] = int(m.sum())
    out["slices"] = sl
    # information ladder
    lad = {"LL_M0": core.log_loss(y, base, w), "LL_F1": core.log_loss(y, P["eb_beta"], w), "LL_F2": core.log_loss(y, P["combo"], w)}
    ll0 = lad["LL_M0"]
    shares = [max(ll0 - lad["LL_F1"], 0) / ll0, max(lad["LL_F1"] - lad["LL_F2"], 0) / ll0, 0.0]
    shares.append(1 - sum(shares))
    out["ladder"] = lad
    # ceilings: entity beta-binomial (weighted counts), hindsight majority, point-in-time 1-NN Cover-Hart on F2 inputs
    g = d.assign(wy=w * y).groupby("entity").agg(k=("wy", "sum"), n=("weight", "sum"))
    a, b = core.mom_beta(g["k"], g["n"])
    bc = core.beta_ceiling(a, b)
    hind = float(np.sum(np.maximum(g["k"], g["n"] - g["k"])) / g["n"].sum())
    X = np.column_stack([walk.logit(P["eb_beta"]), d["z"].fillna(0), d["fund_ok"].fillna(0), d["period_kind"], d["k_known"].fillna(0)])
    t = pd.to_datetime(d["decision_time"]).to_numpy(); av = pd.to_datetime(d["label_available"]).to_numpy()
    r_nn = pit_1nn_error(X, y, t, av, w)
    ch = core.cover_hart_upper(r_nn)
    H = float(core.h(out["combined"]["base_rate_w"]))
    kU = core.kappa_upper_from_rstar(ch["R_star_lower"], H)
    ent = {"alpha": a, "beta": b, "beta_ceiling": bc, "hindsight_entity_majority": hind,
           "learning_curve": core.beta_learning_curve(a, b, 40).tolist(), "r_nn_pit": r_nn, "cover_hart": ch, "kappa_upper": kU}
    out["ceilings"] = ent
    (ROOT / "results" / "d7_earnings_entity.json").write_text(json.dumps({"ceilings": ent}, indent=1, default=float))
    # tie mass from actuals (before/after file, kept rows)
    ba = pd.read_parquet(DATA / "d7_before_after.parquet")
    kept = ba[ba["scored_after"] == 1]
    near = (kept["actual"] - kept["M"]).abs() < 0.05
    out["tie_mass"] = {"share_within_5c": float(near.mean()), "bound": core.tie_mass_bound(float(near.mean())), "n": int(len(kept))}
    # before / after label cleaning on new tickers
    res = {}
    for st in ("before", "after"):
        m = ba[f"scored_{st}"] == 1
        sub = ba[m]
        yy, ww = sub["y"].astype(int).to_numpy(), sub[f"weight_{st}"].to_numpy()
        fr, eb, cl = sub[f"freq_{st}"].to_numpy(), sub[f"eb_beta_{st}"].to_numpy(), sub[f"ctrl_lite_{st}"].to_numpy()
        res[st] = {"n": int(m.sum()), "tickers": int(sub["entity"].nunique()), "acc_freq": wacc(yy, fr, ww),
                   "acc_eb": wacc(yy, eb, ww), "acc_ctrl": wacc(yy, cl, ww),
                   "edge_eb": wacc(yy, eb, ww) - wacc(yy, fr, ww), "edge_ctrl": wacc(yy, cl, ww) - wacc(yy, fr, ww),
                   "edge_eb_ci": boot_ci(sub, lambda idx: wacc(yy[idx], eb[idx], ww[idx]) - wacc(yy[idx], fr[idx], ww[idx])),
                   "ll_freq": core.log_loss(yy, fr, ww), "ll_eb": core.log_loss(yy, eb, ww)}
    sig = {}
    for kind, gdf in ba[ba["scored_before"] == 1].groupby("removed_by"):
        sig[kind] = {"n": int(len(gdf)), "beat_rate": float(gdf["y"].mean()), "eb_pred_beat_share": float((gdf["eb_beta_before"] >= .5).mean())}
    res["removed_signature"] = sig
    # how much of the measured before-edge comes from rows later removed?
    sub = ba[ba["scored_before"] == 1]
    rem = (sub["removed_by"] != "none").to_numpy()
    yy, ww = sub["y"].astype(int).to_numpy(), sub["weight_before"].to_numpy()
    ce = ((sub["eb_beta_before"] >= .5).astype(int) == yy).astype(float).to_numpy()
    cf = ((sub["freq_before"] >= .5).astype(int) == yy).astype(float).to_numpy()
    res["edge_from_removed_rows"] = float(np.sum(ww[rem] * (ce[rem] - cf[rem])) / ww.sum())
    res["edge_from_kept_rows"] = float(np.sum(ww[~rem] * (ce[~rem] - cf[~rem])) / ww.sum())
    out["before_after"] = res
    # disagreement decomposition (harvested rows of the combined universe) + audit bound
    hv = d[d["src"] != "audited20"].reset_index(drop=True)
    dd = core.disagreement_decomposition(hv["y"].to_numpy(), hv["eb_beta"].to_numpy(), hv["freq"].to_numpy(), hv["weight"].to_numpy())
    audit = {"P_D": dd["P_D"], "delta_measured": dd["delta"], "P_eb_correct_given_D": dd["P_a_correct_given_D"]}
    ap = DATA / "d7_audit_results.csv"
    if ap.exists():
        a_ = pd.read_csv(ap)
        k = int((a_["verdict"] != "unverifiable").sum()); e = int((a_["verdict"] == "wrong").sum())
        audit.update({"k": k, "e": e, **core.audit_lower_bound(dd["delta"], dd["P_D"], e, k)})
        audit["failure_types"] = a_.loc[a_["verdict"] == "wrong", "failed_check"].value_counts().to_dict()
    out["audit"] = audit
    # transfer: audited 20 walk-forward vs unseen-ticker refit
    tr = {}
    for nm in ("audited_wf", "unseen"):
        u = pd.read_parquet(DATA / f"d7_{nm}.parquet")
        u = u[u["scored"] == 1]
        yy, ww = u["label"].astype(int).to_numpy(), u["weight"].to_numpy()
        tr[nm] = {"edge_eb": wacc(yy, u["eb_beta"].to_numpy(), ww) - wacc(yy, u["freq"].to_numpy(), ww), "n": int(len(u))}
    tr["tau"] = tr["unseen"]["edge_eb"] / tr["audited_wf"]["edge_eb"] if abs(tr["audited_wf"]["edge_eb"]) > 1e-9 else None
    out["transfer"] = tr
    # power: n needed to detect a 1-point edge given observed disagreement
    out["power_n_for_1pt"] = core.mcnemar_n(max(dd["P_D"], 1e-3), 0.01)
    s = table["combo"]
    out["atlas_row"] = {"shares": shares, "LSS": s["LSS"], "LSS_ci": s["LSS_ci"], "acc": s["acc"], "acc0": s["acc0"],
                        "S_acc": s["S_acc"], "BSS": s["BSS"], "kU": kU, "n": len(d), "entities": int(d["entity"].nunique()),
                        "A_CH": ch["A_upper"], "A1_beta": bc["A"], "tau": tr["tau"]}
    (ROOT / "results" / "d7_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"table": {k: {kk: round(v[kk], 4) for kk in ("acc", "acc0", "S_acc", "LSS", "BSS")} for k, v in table.items()},
                      "ladder": lad, "ceil": {"A1": bc["A"], "hind": hind, "A_CH": ch["A_upper"], "kU": kU}, "tie": out["tie_mass"],
                      "before_after": {k: res[k] for k in ("before", "after")}, "sig": sig,
                      "edge_split": [res["edge_from_removed_rows"], res["edge_from_kept_rows"]], "audit": audit, "transfer": tr,
                      "power_n": out["power_n_for_1pt"]}, indent=1, default=lambda o: round(float(o), 4)))


if __name__ == "__main__":
    main()
