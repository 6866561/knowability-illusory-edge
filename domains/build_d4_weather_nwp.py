"""D4 NWP benchmark: same rows as d4_weather but restricted to period with genuine >=24h-ahead forecasts.
Forecast: Open-Meteo Previous Runs API, hourly precipitation_previous_day1 (forecast issued ~1 day before valid time).
bench = sigmoid(a + b*log1p(fc_mm_sum_day_{d+1})), (a,b) fit by logistic regression ONLY on the first 25% (by time) of the
NWP-period rows; those calibration rows are dropped from the dataset."""
import json, time, pathlib, datetime as dt, io
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression
import build_d4_weather as b

b.END = dt.date(2026, 10, 31)           # extend observations beyond 2024 (forecast archive only exists from 2024-01)
b.RAW = b.ROOT / "raw" / "d4ext"; b.RAW.mkdir(exist_ok=True)
NWP_FIRST = pd.Timestamp("2024-01-20")   # first day with full previous_day1 coverage (verified by probe: starts 2024-01-19T12:00 UTC)
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def get_fc(sid, lat, lon):
    cf = b.ROOT / "raw" / "d4ext" / f"fc_{sid}.parquet"
    if cf.exists(): return pd.read_parquet(cf)
    q = dict(latitude=lat, longitude=lon, hourly="precipitation_previous_day1", start_date="2024-01-01",
             end_date="2026-10-31", timezone="auto")
    for k in range(6):
        try:
            r = requests.get(URL, params=q, headers=b.HDR, timeout=120)
            if r.status_code == 429: time.sleep(30 * (k + 1)); continue
            r.raise_for_status(); break
        except Exception as e:
            print("retry", sid, e); time.sleep(5 * 2 ** k); r = None
    time.sleep(1.0)
    if r is None: return None
    h = r.json()["hourly"]
    d = pd.DataFrame({"time": pd.to_datetime(h["time"]), "fc": h["precipitation_previous_day1"]})
    d = d.dropna(subset=["fc"])
    d["day"] = d["time"].dt.normalize()
    n = d.groupby("day")["fc"].agg(["sum", "count"])
    out = n[n["count"] == 24]["sum"].rename("fc_mm").to_frame()   # only full 24h days
    out.to_parquet(cf); return out

def main():
    picks = json.loads((b.ROOT / "raw" / "d4" / "station_pick.json").read_text())
    seen, uniq = set(), []
    for p in picks:
        if p["id"] not in seen: seen.add(p["id"]); uniq.append(p)
    parts = []; nfc = 0
    for p in uniq:
        s = b.build_station(p["id"], p["city"])       # uses extended END
        fc = get_fc(p["id"], p["lat"], p["lon"])
        if fc is None or len(fc) == 0: continue
        s = s[s["t"] >= NWP_FIRST].copy()
        s["fc_mm"] = (s["t"] + pd.Timedelta(days=1)).map(fc["fc_mm"])  # forecast total for day d+1 (issued ~day d)
        s = s.dropna(subset=["fc_mm"]); 
        if len(s) > 0: parts.append(s); nfc += 1
    df = pd.concat(parts, ignore_index=True)
    print("stations with NWP", nfc, "rows", len(df), df.t.min(), df.t.max())
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    tcut = df["t"].quantile(0.25)
    cal = df["t"] <= tcut
    z = np.log1p(df["fc_mm"].clip(lower=0)).to_numpy()[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[cal], df.loc[cal, "y"])
    a, bb = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    print("calibration rows", int(cal.sum()), "up to", tcut, "a,b =", a, bb)
    df["bench"] = np.clip(1 / (1 + np.exp(-(a + bb * z[:, 0]))), 1e-4, 1 - 1e-4)
    df = df[~cal].reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    out = df[["t", "avail", "entity", "y", "base_key"] + b.XC + ["bench"]]
    out.to_parquet(b.DATA / "d4_weather_nwp.parquet")
    start = (tcut.normalize() + pd.offsets.MonthBegin(1)) + pd.DateOffset(months=3)
    meta = dict(refit="MS", xcols=b.XC, bench=True, leo_folds=5, min_train=300, reps=300, lz=True,
                start=str(start.date()), calib=dict(a=a, b=bb, until=str(tcut.date()), rows=int(cal.sum())))
    (b.DATA / "d4_weather_nwp.meta.json").write_text(json.dumps(meta))
    print("rows", len(out), "stations", out.entity.nunique(), "base rate", out.y.mean(), out.t.min(), out.t.max(), "start", start)
    print("bench-vs-y: mean bench", out.bench.mean())

if __name__ == "__main__":
    main()
