"""D8: next-day stock direction. Source: yfinance (Stooq blocked by JS challenge), auto_adjust=True.
Universe: current S&P 100 constituents (hard-coded; SURVIVORSHIP BIAS) + 10 ETFs."""
import pathlib, time, json
import numpy as np, pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "d8_stocks"; RAW.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
SP100 = """AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK-B C CAT CHTR CL CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR FDX GD GE GILD GM GOOG GOOGL GS HD HON IBM INTC INTU ISRG JNJ JPM KO LIN LLY LMT LOW MA MCD MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NVDA ORCL PEP PFE PG PLTR PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS TSLA TXN UNH UNP UPS USB V VZ WFC WMT XOM""".split()
ETF = "SPY QQQ IWM DIA EFA EEM TLT GLD XLF XLE".split()
TICK = SP100 + ETF

def fetch(tk):
    f = RAW / f"{tk}.csv"
    if not f.exists():
        for a in range(4):
            try:
                d = yf.download(tk, start="2004-11-01", end="2025-11-04", auto_adjust=True, progress=False, multi_level_index=False)
                if len(d): d.to_csv(f); break
            except Exception as e: print(tk, e)
            time.sleep(2 ** a)
        time.sleep(0.6)
    return pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None

px = {tk: fetch(tk) for tk in TICK}
miss = [t for t, d in px.items() if d is None or len(d) == 0]
print("missing:", miss)
spy = px["SPY"]; mkt = np.log(spy["Close"]).diff().rename("x_mkt_r1")

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

out = []
for tk, d in px.items():
    if d is None or len(d) < 300: continue
    d = d.dropna(subset=["Close"]); d = d[d["Close"] > 0]
    c = d["Close"]; lr = np.log(c).diff(); v = d["Volume"].replace(0, np.nan)
    f = pd.DataFrame({"x_r1": lr, "x_r5": np.log(c).diff(5), "x_r20": np.log(c).diff(20), "x_vol20": lr.rolling(20).std(),
                      "x_vchg": np.log(v / v.shift(1).rolling(20).mean()), "x_dow": d.index.dayofweek.astype(float),
                      "x_rsi14": rsi(c)}, index=d.index)
    f["x_mkt_r1"] = mkt.reindex(d.index)
    f["y"] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())
    f["t"] = d.index; f["entity"] = tk
    out.append(f)
df = pd.concat(out).replace([np.inf, -np.inf], np.nan)
df = df[(df.t >= "2005-01-01") & (df.t <= "2025-10-31")]
df = df.dropna(subset=["y", "x_r1", "x_r5", "x_r20", "x_vol20", "x_vchg", "x_mkt_r1", "x_rsi14"]).copy()
df["y"] = df.y.astype(int); df["t"] = pd.to_datetime(df.t).astype("datetime64[ns]")
df["avail"] = df.t + pd.Timedelta(days=3); df["base_key"] = "all"
xc = ["x_r1", "x_r5", "x_r20", "x_vol20", "x_vchg", "x_dow", "x_mkt_r1", "x_rsi14"]
df = df[["t", "avail", "entity", "y", "base_key"] + xc].sort_values(["t", "entity"]).reset_index(drop=True)
df.to_parquet(DATA / "d8_stocks.parquet")
json.dump({"refit": "YS", "xcols": xc, "bench": False, "leo_folds": 5, "min_train": 500, "reps": 300, "lz": True, "start": "2008-01-01"},
          open(DATA / "d8_stocks.meta.json", "w"))
print(len(df), df.entity.nunique(), df.t.min(), df.t.max(), "up share", df.y.mean())
