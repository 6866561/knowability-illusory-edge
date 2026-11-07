"""D4 weather: will it rain tomorrow (>=1mm on d+1). Builds d4_weather and (via --nwp) d4_weather_nwp.
Source: Meteostat (bulk daily CSV, observations only, no model-filled data; station search via the meteostat python lib). Run: python build_d4_weather.py [--nwp]"""
import sys, json, time, pathlib, datetime as dt
import numpy as np, pandas as pd
import meteostat as ms

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "d4"; DATA = ROOT / "data"
RAW.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
START, END = dt.date(2000, 1, 1), dt.date(2024, 12, 31)
CITIES = [  # name, lat, lon  (spread over climate zones / continents)
("London",51.5,-0.1),("Paris",48.9,2.4),("Berlin",52.5,13.4),("Madrid",40.4,-3.7),("Rome",41.9,12.5),("Athens",37.98,23.7),
("Stockholm",59.3,18.1),("Helsinki",60.2,24.9),("Reykjavik",64.1,-21.9),("Moscow",55.8,37.6),("Istanbul",41.0,29.0),("Vienna",48.2,16.4),
("Lisbon",38.7,-9.1),("Warsaw",52.2,21.0),("Oslo",59.9,10.7),
("New York",40.7,-74.0),("Los Angeles",34.0,-118.2),("Chicago",41.9,-87.6),("Miami",25.8,-80.3),("Seattle",47.6,-122.3),("Denver",39.7,-105.0),
("Phoenix",33.4,-112.0),("Houston",29.8,-95.4),("Anchorage",61.2,-149.9),("Honolulu",21.3,-157.9),("Toronto",43.7,-79.4),("Vancouver",49.3,-123.1),
("Winnipeg",49.9,-97.2),("Mexico City",19.4,-99.1),("Atlanta",33.7,-84.4),
("Sao Paulo",-23.5,-46.6),("Buenos Aires",-34.6,-58.4),("Santiago",-33.4,-70.7),("Lima",-12.0,-77.1),("Bogota",4.7,-74.1),("Manaus",-3.1,-60.0),
("Tokyo",35.7,139.7),("Seoul",37.6,127.0),("Beijing",39.9,116.4),("Shanghai",31.2,121.5),("Hong Kong",22.3,114.2),("Singapore",1.35,103.8),
("Bangkok",13.8,100.5),("Delhi",28.6,77.2),("Mumbai",19.1,72.9),("Dubai",25.2,55.3),("Tel Aviv",32.1,34.8),("Tehran",35.7,51.4),("Manila",14.6,121.0),("Jakarta",-6.2,106.8),
("Sydney",-33.9,151.2),("Perth",-31.9,115.9),("Darwin",-12.5,130.9),("Auckland",-36.9,174.8),("Melbourne",-37.8,145.0),("Brisbane",-27.5,153.0),
("Cairo",30.0,31.2),("Nairobi",-1.3,36.8),("Johannesburg",-26.2,28.0),("Cape Town",-33.9,18.4),("Lagos",6.5,3.4),("Casablanca",33.6,-7.6),("Dakar",14.7,-17.5),("Addis Ababa",9.0,38.8),
("Taipei",25.0,121.5),("Kuala Lumpur",3.1,101.7),("Hanoi",21.0,105.8),("Karachi",24.9,67.0),("Baghdad",33.3,44.4),("Riyadh",24.7,46.7),("Kyiv",50.45,30.5),("Bucharest",44.4,26.1),("Budapest",47.5,19.0),("Prague",50.1,14.4),
("Dublin",53.3,-6.3),("Edinburgh",55.95,-3.2),("Zurich",47.4,8.5),("Munich",48.1,11.6),("Barcelona",41.4,2.2),("Tromso",69.7,19.0),("Murmansk",68.97,33.1),("Yakutsk",62.0,129.7),("Irkutsk",52.3,104.3),
("Dallas",32.8,-96.8),("Minneapolis",44.98,-93.3),("Boston",42.4,-71.0),("Salt Lake City",40.8,-111.9),("Las Vegas",36.1,-115.2),("Fairbanks",64.8,-147.7),("Calgary",51.1,-114.1),("Montreal",45.5,-73.6),("Halifax",44.6,-63.6),
("San Juan",18.4,-66.0),("Havana",23.1,-82.4),("Guatemala City",14.6,-90.5),("Quito",-0.2,-78.5),("Caracas",10.5,-66.9),("Montevideo",-34.9,-56.2),("Brasilia",-15.8,-47.9),
("Adelaide",-34.9,138.6),("Hobart",-42.9,147.3),("Alice Springs",-23.7,133.9),("Christchurch",-43.5,172.6),("Wellington",-41.3,174.8),("Noumea",-22.3,166.5),("Port Moresby",-9.4,147.2),
("Tunis",36.8,10.2),("Algiers",36.7,3.1),("Accra",5.6,-0.2),("Windhoek",-22.6,17.1),("Harare",-17.8,31.0),("Maputo",-25.97,32.6),("Antananarivo",-18.9,47.5),("Mauritius",-20.4,57.7),
("Ulaanbaatar",47.9,106.9),("Almaty",43.2,76.9),("Tashkent",41.3,69.3),("Novosibirsk",55.0,82.9),("Colombo",6.9,79.9),("Kathmandu",27.7,85.3),
]
HDR = {"User-Agent": "knowability-research/1.0"}

COLS = ["date","tavg","tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
def load_station(sid):
    """Meteostat bulk daily CSV (observations only; identical prcp to ms.daily(provider=DAILY), but much faster)."""
    cf = RAW / f"st_{sid}.parquet"
    if cf.exists(): return pd.read_parquet(cf)
    import requests
    for k in range(4):
        try:
            r = requests.get(f"https://bulk.meteostat.net/v2/daily/{sid}.csv.gz", headers=HDR, timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status(); break
        except Exception as e:
            time.sleep(2 ** k); r = None
    time.sleep(0.5)
    if r is None: return None
    import io
    df = pd.read_csv(io.BytesIO(r.content), compression="gzip", header=None, names=COLS, parse_dates=["date"]).set_index("date")
    df = df.loc[str(START):str(END)].rename(columns={"tavg": "temp"})
    df.to_parquet(cf); return df

def stations_cov():
    f = RAW / "station_pick.json"
    if f.exists(): return json.loads(f.read_text())
    picks = []
    for name, la, lo in CITIES:
        try:
            near = ms.stations.nearby(ms.Point(la, lo), radius=120000, limit=15)
        except Exception as e:
            print("nearby fail", name, e); continue
        best = None
        for sid in near.index:
            df = load_station(sid)
            if df is None: continue
            if "prcp" not in df or len(df) == 0: continue
            n = (END - START).days + 1
            cov = df["prcp"].notna().sum() / n
            if cov >= 0.90 and (best is None or cov > best[1]): best = (sid, cov)
            if best and best[1] >= 0.97: break
        if best:
            r = near.loc[best[0]]
            picks.append(dict(city=name, id=str(best[0]), cov=float(best[1]), lat=float(r["latitude"]), lon=float(r["longitude"])))
            print("ok", name, best, flush=True)
        else: print("no station >=90%", name, flush=True)
    f.write_text(json.dumps(picks, indent=1)); return picks

def build_station(sid, name):
    df = load_station(sid)
    df.index = pd.to_datetime(df.index)
    df = df.reindex(pd.date_range(START, END)).astype(float)
    for c in ["prcp","tmin","tmax","temp","wspd","pres"]:
        if c not in df: df[c] = np.nan
    p = df["prcp"]
    o = pd.DataFrame(index=df.index)
    o["t"] = df.index
    o["avail"] = df.index + pd.Timedelta(days=2)
    o["y_raw"] = p.shift(-1)
    o["y"] = np.where(o["y_raw"].isna(), np.nan, (o["y_raw"] >= 1.0).astype(float))
    o["x_prcp"] = np.log1p(p.clip(lower=0))
    rain = (p >= 1.0).astype(float).where(p.notna())
    o["x_rain"] = rain
    o["x_rain3"] = rain.fillna(0).rolling(3, min_periods=1).sum()
    d = df["pres"] - df["pres"].shift(1)
    o["x_pres_missing"] = d.isna().astype(float); o["x_dpres"] = d.fillna(0.0)
    o["x_trange"] = df["tmax"] - df["tmin"]
    o["x_wspd"] = df["wspd"]
    doy = df.index.day_of_year
    o["x_doy_sin"] = np.sin(2*np.pi*doy/365.25); o["x_doy_cos"] = np.cos(2*np.pi*doy/365.25)
    tavg = df["temp"].fillna((df["tmin"] + df["tmax"]) / 2)
    mon = df.index.month
    # past-only expanding monthly mean: mean of same-calendar-month days strictly before day d
    g = tavg.groupby(mon)
    past = g.transform(lambda s: s.shift(1).expanding().mean())
    o["x_tavg"] = (tavg - past).fillna(0.0)
    o["entity"] = str(sid)
    o["base_key"] = str(sid) + "-" + (df.index + pd.Timedelta(days=1)).month.astype(str)
    o["fc_dm"] = np.nan
    return o.dropna(subset=["y"]).reset_index(drop=True)

XC = ["x_prcp","x_rain","x_rain3","x_dpres","x_pres_missing","x_trange","x_wspd","x_doy_sin","x_doy_cos","x_tavg"]

def main():
    picks = stations_cov()
    print(len(picks), "stations")
    # drop duplicate station ids
    seen, uniq = set(), []
    for p in picks:
        if p["id"] not in seen: seen.add(p["id"]); uniq.append(p)
    picks = uniq
    parts = [build_station(p["id"], p["city"]) for p in picks]
    df = pd.concat(parts, ignore_index=True)
    df["y"] = df["y"].astype(int)
    out = df[["t","avail","entity","y","base_key"] + XC]
    out.to_parquet(DATA / "d4_weather.parquet")
    meta = dict(refit="YS", xcols=XC, bench=False, leo_folds=5, min_train=500, reps=300, lz=True, start="2004-01-01")
    (DATA / "d4_weather.meta.json").write_text(json.dumps(meta))
    print("rows", len(out), "stations", out.entity.nunique(), "base rate", out.y.mean(), out.t.min(), out.t.max())
    print("per-station missing target dropped; features NaN share:\n", out[XC].isna().mean().round(3).to_dict())

if __name__ == "__main__":
    main()
