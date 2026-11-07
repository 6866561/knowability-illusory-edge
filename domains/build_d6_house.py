"""d6_house: does the incumbent party hold a US House seat (entity = state_po-district)?
Source: MEDSL "U.S. House 1976-2018" official GitHub release (MEDSL/constituency-returns, CC0 data
mirror of Harvard Dataverse doi:10.7910/DVN/IG0UN2). The Dataverse 2020-2024 files require a
guestbook form (name/email/institution), so they were NOT downloaded."""
import pathlib, time, json, re, sys
import numpy as np, pandas as pd, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"raw"/"d6"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT/"data"; OUT.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/MEDSL/constituency-returns/master/1976-2018-house.csv"
f = RAW/"house_1976_2018.csv"
if not f.exists():
    for i in range(5):
        try:
            r = requests.get(URL, headers={"User-Agent": "knowability-research/1.0"}, timeout=60); r.raise_for_status()
            f.write_bytes(r.content); break
        except Exception: time.sleep(2**i)
raw = pd.read_csv(f, encoding="latin-1")
N_RAW = len(raw)
d = raw[(raw.stage == "gen") & (~raw.special) & (raw.runoff != True)].copy()
d = d[d.candidate.notna()]
d = d[~d.candidate.str.contains(r"scatter|blank|write-?in|void|over ?vote|under ?vote|none of", case=False, regex=True)]
D_ = {"democrat", "democratic-farmer-labor", "democratic-npl", "democratic-nonpartisan league", "foglietta (democrat)"}
R_ = {"republican", "independent-republican"}
pl = d.party.fillna("").str.lower()
d["pp"] = np.where(pl.isin(D_), "D", np.where(pl.isin(R_), "R", "O"))
d["entity"] = d.state_po + "-" + d.district.astype(int).astype(str)

def norm_key(name):
    s = name.lower()
    s = re.sub(r"\(.*?\)|\".*?\"", " ", s)
    s = re.sub(r"[^a-z, ]", " ", s)
    if "," in s:
        a, b = s.split(",", 1); toks_l = a.split(); toks_f = b.split()
    else:
        toks = s.split(); toks_l = toks; toks_f = toks
    suf = {"jr", "sr", "ii", "iii", "iv", "v", "md", "dr"}
    if "," in s:
        last = [t for t in toks_l if t not in suf]; first = [t for t in toks_f if t not in suf]
        last = last[-1] if last else ""; first = first[0] if first else ""
    else:
        toks = [t for t in toks if t not in suf]
        last = toks[-1] if toks else ""; first = toks[0] if toks else ""
    return last + "|" + first[:1]

# candidate-level (sum fusion lines)
d["votes"] = d.candidatevotes
g = d.groupby(["year", "entity", "candidate", "pp"], as_index=False).votes.sum()
# candidate party = major-party line with most votes if any, else O
g["major"] = g.pp != "O"
g = g.sort_values(["year", "entity", "candidate", "major", "votes"], ascending=[True, True, True, False, False])
cand = g.groupby(["year", "entity", "candidate"], as_index=False).agg(party=("pp", "first"), votes=("votes", "sum"))
cand["key"] = cand.candidate.map(norm_key)

rows = []
for (yr, ent), c in cand.groupby(["year", "entity"]):
    c = c.sort_values("votes", ascending=False)
    w = c.iloc[0]
    dv = c.loc[c.party == "D", "votes"].max() if (c.party == "D").any() else 0
    rv = c.loc[c.party == "R", "votes"].max() if (c.party == "R").any() else 0
    nD = int((c.party == "D").sum()); nR = int((c.party == "R").sum())
    tot = c.votes.sum()
    # candidate-level D/R votes use the top D and top R candidate (fusion already summed)
    two = dv + rv
    dshare = dv/two if two > 0 else np.nan
    if w.party == "D": sign = 1
    elif w.party == "R": sign = -1
    else: sign = 0
    if w.party in ("D", "R") and two > 0:
        marg = sign*(2*dshare-1)
    else:  # independent/other winner: top-two margin over all votes
        marg = (c.votes.iloc[0]-(c.votes.iloc[1] if len(c) > 1 else 0))/tot if tot > 0 else np.nan
    rows.append(dict(year=yr, entity=ent, winner=w.candidate, wparty=w.party, wkey=w.key, margin=marg,
                     dv=dv, rv=rv, contested=int(nD >= 1 and nR >= 1), keys=set(c.key), total=tot))
E = pd.DataFrame(rows)
# national two-party D share, contested races only
con = E[E.contested == 1]
nat = (con.groupby("year").dv.sum()/(con.groupby("year").dv.sum()+con.groupby("year").rv.sum())).to_dict()
PRES = {}
for yrs, p in [((1976,), "R"), ((1978, 1980), "D"), ((1982, 1984, 1986, 1988), "R"), ((1990, 1992), "R"),
               ((1994, 1996, 1998, 2000), "D"), ((2002, 2004, 2006, 2008), "R"), ((2010, 2012, 2014, 2016), "D"),
               ((2018,), "R")]:
    for y in yrs: PRES[y] = p
Ei = E.set_index(["year", "entity"])
out = []; stats = dict(no_prev=0, unc_cur=0, unc_cur_hold=0)
audit = []
for r in E.itertuples():
    if (r.year-2, r.entity) not in Ei.index: stats["no_prev"] += 1; continue
    p = Ei.loc[(r.year-2, r.entity)]
    hold = int(r.wparty == p.wparty)
    if not r.contested:
        stats["unc_cur"] += 1; stats["unc_cur_hold"] += hold; continue
    if r.year-4 not in nat or r.year-2 not in nat: continue
    sg = {"D": 1, "R": -1, "O": 0}[p.wparty]
    open_seat = int(p.wkey not in r.keys)
    mid = int(r.year % 4 != 0)
    pinc = int(p.wparty == PRES[r.year])
    out.append(dict(t=pd.Timestamp(r.year, 11, 1), avail=pd.Timestamp(r.year, 11, 1)+pd.Timedelta(days=30),
                    entity=r.entity, y=hold, base_key="all", year=r.year,
                    x_prev_margin=float(p.margin), x_prev_uncontested=float(1-p.contested), x_open_seat=float(open_seat),
                    x_midterm=float(mid), x_pres_party_inc=float(pinc), x_midterm_x_pres=float(mid*pinc),
                    x_prev_nat_swing=float(sg*(nat[r.year-2]-nat[r.year-4])),
                    _prevwinner=p.winner, _prevparty=p.wparty, _curcands="; ".join(sorted(cand[(cand.year == r.year) & (cand.entity == r.entity)].candidate))))
df = pd.DataFrame(out).dropna(subset=[c for c in out[0] if c.startswith("x_")])
# open-seat audit: 30 random rows printed for eyeballing
aud = df.sample(30, random_state=7)[["year", "entity", "_prevwinner", "x_open_seat", "_curcands"]]
aud.to_csv(RAW/"open_seat_audit.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 110)
print(aud.to_string(index=False))
xcols = [c for c in df if c.startswith("x_")]
df = df.sort_values(["t", "entity"]).reset_index(drop=True)
df[["t", "avail", "entity", "y", "base_key"]+xcols].to_parquet(OUT/"d6_house.parquet", index=False)
(OUT/"d6_house.meta.json").write_text(json.dumps(dict(refit="YS", xcols=xcols, bench=False, leo_folds=5, min_train=300, reps=500, lz=False, start="1986-01-01")))
print(f"raw rows {N_RAW}; district-years {len(E)}; no previous election {stats['no_prev']}")
print(f"contested-with-prev rows (dataset) {len(df)}; years {df.year.min()}-{df.year.max()}; entities {df.entity.nunique()}; base rate {df.y.mean():.4f}")
print(f"uncontested current (with prev, excluded): {stats['unc_cur']}, hold rate {stats['unc_cur_hold']/max(stats['unc_cur'],1):.4f}")
print("open seat share", df.x_open_seat.mean(), "prev uncontested share", df.x_prev_uncontested.mean())
print("hold rate by open seat", df.groupby("x_open_seat").y.agg(["mean", "size"]).to_dict())
