"""Point-in-time audit of every real-data domain (reviewer item 4). Writes results/pit_audit.json.

(1) Label timing: every row's label becomes available after its decision time (avail > t); minimum gap reported.
(2) Entity-rate leakage test on the evaluator actually used (kn.walk.entity_rates): labels of all rows that become available
    after a cutoff tau are randomly permuted; every forecast for a decision time t <= tau must be unchanged.
(3) Model-fit leakage test (kn.walk.feature_models, one refit cutoff c): labels available at or after c are permuted with the
    entity-rate features held fixed; predictions for rows with t >= c must be unchanged.
(4) Source recomputation on a random sample of rows, from the cached raw source files: earthquakes (event counts in the
    7 and 30 days before t, and the label for [t, t+7) days, from USGS ComCat monthly files) and stocks (one-day log return
    to the close at t, and the label close(t+1) > close(t), from the cached daily price files). Earnings: the guidance
    release date precedes the decision time for the 82 audited disagreement rows.
Features that cannot be recomputed from cached raw files here (weather station/NWP inputs; football ratings and closing odds;
election histories) are covered by (1)-(3) and by code inspection only; this is stated in the output.
"""
import json, pathlib, sys, math
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from kn import walk
from kn.paths import DATA, RAW
rng = np.random.default_rng(20261109)
DOMS = ["d4_weather", "d4_weather_nwp", "d5_football", "d6_house", "d6_president", "d8_stocks", "d9_quakes"]
out = {}


def load(name):
    df = pd.read_parquet(DATA / f"{name}.parquet"); meta = json.loads((DATA / f"{name}.meta.json").read_text())
    return df, meta


for name in DOMS:
    if not (DATA / f"{name}.parquet").exists():
        out[name] = {"status": "data file not available"}; continue
    df, meta = load(name)
    r = {"rows": int(len(df))}
    gap = (df["avail"] - df["t"])
    gap_days = gap.dt.total_seconds() / 86400 if hasattr(gap, "dt") else gap.astype(float)
    r["label_after_decision_violations"] = int((gap_days <= 0).sum()); r["min_gap_days"] = float(gap_days.min())
    # (2) entity-rate test on a subsample of entities (keeps runtime modest on the largest panels)
    ents = df["entity"].unique()
    keep = rng.choice(ents, min(len(ents), 40), replace=False) if len(df) > 60000 else ents
    sub = df[df["entity"].isin(keep)].reset_index(drop=True)
    tau = sub["t"].sort_values().iloc[len(sub) // 2]
    a = walk.entity_rates(sub.copy())
    sc = sub.copy(); m = sc["avail"] >= tau
    sc.loc[m, "y"] = rng.permutation(sc.loc[m, "y"].to_numpy())
    b = walk.entity_rates(sc)
    early = a["t"] <= tau
    r["entity_rates_rows_checked"] = int(early.sum())
    r["entity_rates_max_abs_change_before_cutoff"] = float(max((a.loc[early, c].to_numpy() - b.loc[early, c].to_numpy()).__abs__().max() for c in ("p_M0", "p_M1", "p_M2")))
    # (3) model-fit test at one cutoff: the first decision time on or after the median
    c0 = a["t"].sort_values().iloc[int(len(a) * 0.6)]
    xcols = meta["xcols"]; bench = bool(meta.get("bench")) and "bench" in a
    p1 = walk.feature_models(a.copy(), [c0], xcols, min_train=50, bench=bench, knn=False)
    a2 = a.copy(); m2 = a2["avail"] >= c0
    a2.loc[m2, "y"] = rng.permutation(a2.loc[m2, "y"].to_numpy())
    p2 = walk.feature_models(a2, [c0], xcols, min_train=50, bench=bench, knn=False)
    te = (a["t"] >= c0).to_numpy()
    diffs = [np.nanmax(np.abs(p1.loc[te, c].to_numpy() - p2.loc[te, c].to_numpy())) for c in ("p_M4", "p_M4n", "p_M5n") if p1.loc[te, c].notna().any()]
    r["feature_models_rows_checked"] = int(te.sum()); r["feature_models_max_abs_change"] = float(max(diffs)) if diffs else None
    out[name] = r
    print(name, r, flush=True)

# (4) source recomputation
src = {}
q, _ = load("d9_quakes")
files = sorted((RAW / "d9_quakes").glob("*.csv"))
if files:
    ev = pd.concat([pd.read_csv(f, usecols=["time", "latitude", "longitude", "mag"]) for f in files], ignore_index=True).dropna()
    ev["time"] = pd.to_datetime(ev["time"], utc=True).dt.tz_localize(None)
    ev = ev[(ev["mag"] >= 4.0) & (ev["time"] >= "1990-01-01") & (ev["time"] < "2025-01-01")].drop_duplicates()
    ev["cell"] = np.floor(ev["latitude"] / 2).astype(int).astype(str) + "_" + np.floor(ev["longitude"] / 2).astype(int).astype(str)
    g = {k: np.sort(v["time"].to_numpy()) for k, v in ev.groupby("cell")}
    smp = q.sample(300, random_state=1); bad = 0
    for _, row in smp.iterrows():
        T = g.get(row["entity"], np.array([], dtype="datetime64[ns]")); t = np.datetime64(row["t"])
        cnt = lambda lo, hi: int(np.searchsorted(T, hi, "left") - np.searchsorted(T, lo, "left"))
        c7 = cnt(t - np.timedelta64(7, "D"), t); c30 = cnt(t - np.timedelta64(30, "D"), t); y = int(cnt(t, t + np.timedelta64(7, "D")) >= 1)
        ok = abs(math.log1p(c7) - row["x_c7"]) < 1e-9 and abs(math.log1p(c30) - row["x_c30"]) < 1e-9 and y == row["y"]
        bad += (not ok)
    src["d9_quakes"] = {"rows_sampled": 300, "mismatches": bad, "checked": "x_c7, x_c30 (events strictly before t) and label (events in [t, t+7d))"}
st, _ = load("d8_stocks")
pxdir = RAW / "d8_stocks"
if pxdir.exists():
    smp = st.sample(300, random_state=2); bad = 0; checked = 0
    cache = {}
    for _, row in smp.iterrows():
        f = pxdir / f"{row['entity']}.csv"
        if not f.exists():
            continue
        if row["entity"] not in cache:
            cache[row["entity"]] = pd.read_csv(f, index_col=0, parse_dates=True)["Close"].astype(float)
        c = cache[row["entity"]]; t = pd.Timestamp(row["t"])
        if c.index.tz is not None:
            c = c.tz_localize(None); cache[row["entity"]] = c
        i = c.index.get_indexer([t])[0]
        if i <= 0 or i + 1 >= len(c):
            continue
        checked += 1
        r1 = math.log(c.iloc[i] / c.iloc[i - 1]); y = int(c.iloc[i + 1] > c.iloc[i])
        bad += not (abs(r1 - row["x_r1"]) < 1e-9 and y == row["y"])
    src["d8_stocks"] = {"rows_sampled": checked, "mismatches": bad, "checked": "x_r1 (return to the close at t) and label (close at t+1 above close at t)"}
h = pd.read_csv(DATA / "d7_harvested_audit_sample.csv")
fd, dt = pd.to_datetime(h["file_date"]), pd.to_datetime(h["decision_time"])
src["d7_earnings_audit_rows"] = {"rows": int(len(h)), "decision_before_release": int((dt < fd.dt.normalize()).sum()),
                                  "max_days_release_to_decision": float((dt - fd).dt.days.max()), "min_days_release_to_decision": float((dt - fd).dt.days.min())}
d7 = pd.read_parquet(DATA / "d7_combined.parquet")
g7 = (pd.to_datetime(d7["label_available"]) - pd.to_datetime(d7["decision_time"])).dt.days
src["d7_earnings_label_timing"] = {"rows": int(len(d7)), "label_after_decision_violations": int((g7 <= 0).sum()), "min_gap_days": float(g7.min())}
out["source_recomputation"] = src
out["not_recomputed_from_source"] = ["weather station and NWP inputs (raw station files not cached)", "football ratings, rest, form and closing odds",
                                     "House and presidential histories"]
(ROOT / "results" / "pit_audit.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(src, indent=1, default=str))
