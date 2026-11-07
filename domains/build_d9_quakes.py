"""D9: weekly M>=4 earthquake occurrence in 2x2 deg cells. Source: USGS ComCat FDSN event API (public domain)."""
import pathlib, time, json, io
import numpy as np, pandas as pd, requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "d9_quakes"; RAW.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
HDR = {"User-Agent": "knowability-research/1.0"}

def fetch(a, b, tag):
    f = RAW / f"{tag}.csv"
    if f.exists(): return pd.read_csv(f, encoding="utf-8")
    for k in range(6):
        try:
            r = requests.get(URL, headers=HDR, timeout=120, params=dict(
                format="csv", starttime=a, endtime=b, minmagnitude=4.0, orderby="time-asc", eventtype="earthquake"))
            if r.status_code == 200:
                f.write_bytes(r.content); time.sleep(0.6); return pd.read_csv(io.BytesIO(r.content))
            print(tag, r.status_code)
        except Exception as e: print(tag, e)
        time.sleep(2 ** k)
    raise RuntimeError(tag)

parts = []
for y in range(1990, 2025):
    for m in range(1, 13):
        a = pd.Timestamp(y, m, 1); b = a + pd.offsets.MonthBegin(1)
        d = fetch(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), f"{y}-{m:02d}")
        assert len(d) < 20000, (y, m)
        parts.append(d[["time", "latitude", "longitude", "mag"]])
ev = pd.concat(parts, ignore_index=True).dropna()
ev["time"] = pd.to_datetime(ev["time"], utc=True).dt.tz_localize(None)
ev = ev[(ev["mag"] >= 4.0) & (ev["time"] >= "1990-01-01") & (ev["time"] < "2025-01-01")].drop_duplicates()
ev = ev.sort_values("time").reset_index(drop=True)
ev["ci"] = np.floor(ev["latitude"] / 2).astype(int); ev["cj"] = np.floor(ev["longitude"] / 2).astype(int)
ev["ts"] = (ev["time"] - pd.Timestamp("1990-01-01")) / pd.Timedelta(days=1)
print("events", len(ev))

sel = ev[ev["time"] < "2005-01-01"].groupby(["ci", "cj"]).size().sort_values(ascending=False).head(300)
cells = list(sel.index)
print("min events in selected cell (1990-2004):", sel.min())

bycell = {k: g for k, g in ev.groupby(["ci", "cj"])}
E = np.empty(0)
def arrs(ci, cj):
    """times (sorted) and mags of events in cell (ci,cj), with lon wrap."""
    cj = (cj + 90) % 180 - 90
    g = bycell.get((ci, cj))
    return (g["ts"].to_numpy(), g["mag"].to_numpy()) if g is not None else (E, E)
def merged(ci, cj, ds):
    T, M = [], []
    for di, dj in ds:
        t_, m_ = arrs(ci + di, cj + dj); T.append(t_); M.append(m_)
    T = np.concatenate(T); M = np.concatenate(M); o = np.argsort(T, kind="stable")
    return T[o], M[o]
NB = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
ALL9 = NB + [(0, 0)]

t0 = pd.Timestamp("1990-01-01")
first = pd.Timestamp("1991-01-07")  # Monday; leaves 1y history for x_c365
weeks = pd.date_range(first, "2024-12-23", freq="7D")  # last week [12-23,12-30) lies fully inside the 1990..2024 download
assert weeks[0].dayofweek == 0
tw = ((weeks - t0) / pd.Timedelta(days=1)).to_numpy()
rows = []
for (ci, cj) in cells:
    T, M = arrs(ci, cj)
    TN, _ = merged(ci, cj, NB)
    T9, M9 = merged(ci, cj, ALL9)
    big9 = T9[M9 >= 6.0]
    cnt = lambda A, lo, hi: np.searchsorted(A, hi, "left") - np.searchsorted(A, lo, "left")  # events in [lo,hi)
    c7 = cnt(T, tw - 7, tw); c30 = cnt(T, tw - 30, tw); c365 = cnt(T, tw - 365, tw)
    y = (cnt(T, tw, tw + 7) >= 1).astype(int)
    nb7 = cnt(TN, tw - 7, tw)
    bs = (cnt(big9, tw - 30, tw) > 0).astype(float)
    ix = np.searchsorted(T, tw, "left") - 1
    ds = np.where(ix >= 0, tw - T[np.clip(ix, 0, None)] if len(T) else 3650, 3650)
    ds = np.minimum(ds, 3650)
    # max magnitude over past 30 days
    lo_i = np.searchsorted(T, tw - 30, "left"); hi_i = np.searchsorted(T, tw, "left")
    mm = np.array([M[a:b].max() if b > a else 0.0 for a, b in zip(lo_i, hi_i)])
    rows.append(pd.DataFrame({"t": weeks, "entity": f"{ci}_{cj}", "y": y,
        "x_c7": np.log1p(c7), "x_c30": np.log1p(c30), "x_c365": np.log1p(c365), "x_maxmag30": mm,
        "x_days_since": np.log1p(ds), "x_nb7": np.log1p(nb7), "x_bigshock": bs}))
df = pd.concat(rows, ignore_index=True)
df["avail"] = df["t"] + pd.Timedelta(days=8)
df["base_key"] = df["entity"]
xcols = ["x_c7", "x_c30", "x_c365", "x_maxmag30", "x_days_since", "x_nb7", "x_bigshock"]
df = df[["t", "avail", "entity", "y", "base_key"] + xcols]
df.to_parquet(DATA / "d9_quakes.parquet")
meta = {"refit": "YS", "xcols": xcols, "bench": False, "leo_folds": 5, "min_train": 500, "reps": 300, "lz": False, "start": "2005-01-01"}
(DATA / "d9_quakes.meta.json").write_text(json.dumps(meta))
print("rows", len(df), "entities", df.entity.nunique(), "base", df.y.mean(), df.t.min(), df.t.max())
