"""d6_president: does the Democratic candidate win the state? Source: MEDSL "U.S. President 1976-2024"
(Harvard Dataverse doi:10.7910/DVN/42MVDX, CC0), file 1976-2024-president.csv (public, no guestbook)."""
import pathlib, time, json
import numpy as np, pandas as pd, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"raw"/"d6"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"data"; OUT.mkdir(exist_ok=True)
URL = "https://dataverse.harvard.edu/api/access/datafile/13887042"
f = RAW/"president.csv"
if not f.exists():
    for i in range(5):
        try:
            r = requests.get(URL, headers={"User-Agent": "knowability-research/1.0"}, timeout=120); r.raise_for_status()
            f.write_bytes(r.content); break
        except Exception: time.sleep(2**i)
d = pd.read_csv(f)
d = d[d.candidate.notna()].copy()
# nominees = top Democrat / top Republican candidate nationally per year (by total votes); fusion lines summed by candidate name
nom = {}
for y, g in d.groupby("year"):
    for p in ("DEMOCRAT", "REPUBLICAN"):
        nom[(y, p)] = g[g.party_simplified == p].groupby("candidate").candidatevotes.sum().idxmax()
tot = d.groupby(["year", "state_po"]).totalvotes.max()
cv = d.groupby(["year", "state_po", "candidate"]).candidatevotes.sum().reset_index()
rows = []
for (y, s), g in cv.groupby(["year", "state_po"]):
    dv = g.loc[g.candidate == nom[(y, "DEMOCRAT")], "candidatevotes"].sum()
    rv = g.loc[g.candidate == nom[(y, "REPUBLICAN")], "candidatevotes"].sum()
    winner = g.loc[g.candidatevotes.idxmax(), "candidate"]
    rows.append(dict(year=y, state=s, y=int(winner == nom[(y, "DEMOCRAT")]), dv=dv, rv=rv, tot=tot[(y, s)],
                     won_rep=int(winner == nom[(y, "REPUBLICAN")])))
S = pd.DataFrame(rows)
S["margin"] = (S.dv-S.rv)/S.tot
natm = S.groupby("year").apply(lambda g: (g.dv.sum()-g.rv.sum())/g.tot.sum(), include_groups=False).to_dict()
# nominee home states (Dem, Rep); judgment calls: Bush 41/43 = TX, Romney = MA, Trump 2016 = NY, Trump 2020/24 = FL, Clinton 2016 = NY
HOME = {1976: ("GA", "MI"), 1980: ("GA", "CA"), 1984: ("MN", "CA"), 1988: ("MA", "TX"), 1992: ("AR", "TX"),
        1996: ("AR", "KS"), 2000: ("TN", "TX"), 2004: ("MA", "TX"), 2008: ("IL", "AZ"), 2012: ("IL", "MA"),
        2016: ("NY", "NY"), 2020: ("DE", "FL"), 2024: ("CA", "FL")}
M = S.set_index(["year", "state"]).margin
out = []
for r in S.itertuples():
    y = r.year
    if y-4 < 1976 or y-8 < 1976: continue
    hd, hr = HOME[y]
    hs = (1 if r.state == hd else 0) - (1 if r.state == hr else 0)
    out.append(dict(t=pd.Timestamp(y, 11, 1), avail=pd.Timestamp(y, 11, 1)+pd.Timedelta(days=30), entity=r.state, y=r.y,
                    base_key="all", x_prev_dem_margin=float(M[(y-4, r.state)]), x_prev2_dem_margin=float(M[(y-8, r.state)]),
                    x_prev_nat_margin=float(natm[y-4]), x_home_state=float(hs)))
df = pd.DataFrame(out).sort_values(["t", "entity"]).reset_index(drop=True)
xcols = [c for c in df if c.startswith("x_")]
df.to_parquet(OUT/"d6_president.parquet", index=False)
(OUT/"d6_president.meta.json").write_text(json.dumps(dict(refit="YS", xcols=xcols, bench=False, leo_folds=5, min_train=100, reps=500, lz=False, start="1992-01-01")))
print("nominees", {k: v for k, v in nom.items() if k[0] in (2016, 2024)})
print("natl margin", {k: round(v, 4) for k, v in natm.items()})
print("rows", len(df), "states", df.entity.nunique(), "years", df.t.dt.year.min(), df.t.dt.year.max(), "base", round(df.y.mean(), 4))
print("per-year dem wins", df.groupby(df.t.dt.year).y.sum().to_dict(), "per-year n", df.groupby(df.t.dt.year).size().unique())
print("home-state rows", df[df.x_home_state != 0].groupby(df.t.dt.year).apply(lambda g: dict(zip(g.entity, g.x_home_state)), include_groups=False).to_dict())
print("states with winner neither D nor R", int(((S.y == 0) & (S.won_rep == 0)).sum()))
