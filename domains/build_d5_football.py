import pathlib, time, json, io, collections
import numpy as np, pandas as pd, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"raw"/"d5_football"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"data"; OUT.mkdir(exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "knowability-research/1.0"
def get(url, path):
    if path.exists(): return path.read_bytes()
    for i in range(5):
        try:
            r = S.get(url, timeout=60); time.sleep(0.6)
            if r.status_code == 404: path.write_bytes(b""); return b""
            r.raise_for_status(); path.write_bytes(r.content); return r.content
        except Exception as e:
            time.sleep(2**i)
    raise RuntimeError(url)
DIVS = ["E0","D1","I1","SP1","F1"]
SEAS = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2005, 2025)]
SRC = [("PSCH","PSCD","PSCA","pin_close"),("PSH","PSD","PSA","pin"),("B365H","B365D","B365A","b365"),
       ("AvgCH","AvgCD","AvgCA","avg_close"),("AvgH","AvgD","AvgA","avg"),("BbAvH","BbAvD","BbAvA","bbavg")]
rows = []; srcs = collections.Counter()
for div in DIVS:
    frames = []
    for s in SEAS:
        b = get(f"https://www.football-data.co.uk/mmz4281/{s}/{div}.csv", RAW/f"{div}_{s}.csv")
        if not b: print("missing", div, s); continue
        d = pd.read_csv(io.BytesIO(b), encoding="latin-1", on_bad_lines="skip")
        d = d.dropna(subset=["HomeTeam","AwayTeam","FTR"]).copy()
        d["season"] = s
        d["date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed", errors="coerce")
        d = d.dropna(subset=["date"])
        # odds
        pH = np.full(len(d), np.nan); sc = np.array([None]*len(d), dtype=object)
        for h,dr,a,name in SRC:
            if h not in d: continue
            o = d[[h,dr,a]].apply(pd.to_numeric, errors="coerce")
            ok = (o>1).all(axis=1).to_numpy() & np.isnan(pH)
            p = (1/o[h])/((1/o).sum(axis=1))
            pH[ok] = p.to_numpy()[ok]; sc[ok] = name
        d["bench"] = pH; d["src"] = sc
        frames.append(d[["date","HomeTeam","AwayTeam","FTR","season","bench","src"]])
    d = pd.concat(frames).sort_values(["date"], kind="stable").reset_index(drop=True)
    R = collections.defaultdict(lambda: 1500.0); last = {}; hist = collections.defaultdict(list)
    cur = None
    for r in d.itertuples():
        if r.season != cur:
            if cur is not None:
                for k in list(R): R[k] = 1500 + (R[k]-1500)*(2/3)
            cur = r.season
        h, a = r.HomeTeam, r.AwayTeam
        rh, ra = R[h], R[a]
        fh = hist[h][-5:]; fa = hist[a][-5:]
        rows.append(dict(t=r.date, avail=r.date+pd.Timedelta(days=1), entity=f"{div}:{h}", y=int(r.FTR=="H"),
            base_key=div, x_elo_diff=(rh+60-ra)/400, x_elo_home=rh, x_elo_away=ra,
            x_rest_home=min(30,(r.date-last[h]).days) if h in last else 30,
            x_rest_away=min(30,(r.date-last[a]).days) if a in last else 30,
            x_form_home=np.mean(fh) if fh else 1.0, x_form_away=np.mean(fa) if fa else 1.0,
            bench=r.bench, src=r.src))
        e = 1/(1+10**(-(rh+60-ra)/400)); sh = {"H":1,"D":.5,"A":0}[r.FTR]
        R[h] = rh+20*(sh-e); R[a] = ra-20*(sh-e)
        ph, pa = {"H":3,"D":1,"A":0}[r.FTR], {"H":0,"D":1,"A":3}[r.FTR]
        hist[h].append(ph); hist[a].append(pa); last[h] = last[a] = r.date
df = pd.DataFrame(rows)
print("rows before odds filter", len(df))
df = df.dropna(subset=["bench"]); df = df[(df.bench>0)&(df.bench<1)]
print("source counts", df.src.value_counts().to_dict())
print(df.groupby("base_key").size().to_dict(), "base rate", df.y.mean())
df.drop(columns="src").to_parquet(OUT/"d5_football.parquet", index=False)
xc = [c for c in df if c.startswith("x_")]
json.dump({"refit":"YS","xcols":xc,"bench":True,"leo_folds":5,"min_train":500,"reps":500,"lz":False,"start":"2009-01-01"}, open(OUT/"d5_football.meta.json","w"))
