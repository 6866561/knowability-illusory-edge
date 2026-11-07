import pathlib, time, json, io, collections
import numpy as np, pandas as pd, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"raw"/"d5_tennis"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"data"; OUT.mkdir(exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "knowability-research/1.0"
def get(url, path):
    if path.exists(): return path.read_bytes()
    for i in range(5):
        try:
            r = S.get(url, timeout=60); time.sleep(0.6); r.raise_for_status()
            path.write_bytes(r.content); return r.content
        except Exception: time.sleep(2**i)
    raise RuntimeError(url)
fr = []
for y in range(2000, 2025):
    b = get(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv", RAW/f"atp_{y}.csv")
    fr.append(pd.read_csv(io.BytesIO(b), low_memory=False))
d = pd.concat(fr, ignore_index=True)
print("raw", len(d))
d = d.dropna(subset=["winner_rank","loser_rank","tourney_date","winner_id","loser_id"]).copy()
d = d[d.winner_rank != d.loser_rank]
off = {"R128":0,"R64":1,"R32":2,"R16":3,"QF":4,"SF":5,"F":6,"RR":3}
d["t"] = pd.to_datetime(d.tourney_date.astype(int).astype(str), format="%Y%m%d") + pd.to_timedelta(d["round"].map(off).fillna(0).astype(int), unit="D")
d = d.sort_values(["t","tourney_id","match_num"], kind="stable").reset_index(drop=True)
print("with ranks", len(d))
R = collections.defaultdict(lambda: 1500.0); h2h = collections.defaultdict(int)  # key (a,b)->wins of a over b
rows = []
for r in d.itertuples():
    w, l = int(r.winner_id), int(r.loser_id)
    surf = r.surface if isinstance(r.surface, str) else "NA"
    wf = r.winner_rank < r.loser_rank   # winner is fav
    f, dg = (w, l) if wf else (l, w)
    rf, rd = (r.winner_rank, r.loser_rank) if wf else (r.loser_rank, r.winner_rank)
    pf, pd_ = (r.winner_rank_points, r.loser_rank_points) if wf else (r.loser_rank_points, r.winner_rank_points)
    af, ad = (r.winner_age, r.loser_age) if wf else (r.loser_age, r.winner_age)
    ok = pd.notna(pf) and pd.notna(pd_) and pf > 0 and pd_ > 0
    Ef, Ed = R[(f,surf)], R[(dg,surf)]
    rows.append(dict(t=r.t, avail=r.t+pd.Timedelta(days=1), entity=str(f), y=int(wf), base_key=str(r.tourney_level),
        x_logrank_ratio=np.log(rd/rf), x_pts_diff=np.log(pf/pd_) if ok else 0.0, x_pts_missing=0.0 if ok else 1.0,
        x_surface_elo_diff=(Ef-Ed)/400, x_age_diff=(af-ad) if pd.notna(af) and pd.notna(ad) else 0.0,
        x_age_missing=0.0 if pd.notna(af) and pd.notna(ad) else 1.0,
        x_h2h=h2h[(f,dg)]-h2h[(dg,f)]))
    e = 1/(1+10**(-(R[(w,surf)]-R[(l,surf)])/400))
    R[(w,surf)] += 32*(1-e); R[(l,surf)] -= 32*(1-e)
    h2h[(w,l)] += 1
df = pd.DataFrame(rows)
print("rows", len(df), "base rate", df.y.mean(), df.base_key.value_counts().to_dict(), "pts missing", df.x_pts_missing.sum())
df.to_parquet(OUT/"d5_tennis.parquet", index=False)
xc = [c for c in df if c.startswith("x_")]
json.dump({"refit":"YS","xcols":xc,"bench":False,"leo_folds":5,"min_train":500,"reps":500,"lz":False,"start":"2004-01-01"}, open(OUT/"d5_tennis.meta.json","w"))
