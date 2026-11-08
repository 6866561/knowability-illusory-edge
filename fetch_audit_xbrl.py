"""Download SEC XBRL company facts (public data.sec.gov API) for the issuers in the earnings audit sample.
Files go to ../raw/sec_audit/<TICKER>.json (raw downloads kept apart from code). Generic research User-Agent; no personal data."""
import json, pathlib, time, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "raw" / "sec_audit"; OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "KnowabilityResearch research-bot@example.org"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


aud = pd.read_csv(ROOT.parent / "data" / "d7_audit_results.csv")
tick = sorted(aud.issuer.unique())
mp = {v["ticker"]: int(v["cik_str"]) for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")).values()}
for t in tick:
    p = OUT / f"{t}.json"
    if p.exists() and p.stat().st_size > 1000:
        continue
    if t not in mp:
        print("no CIK", t); continue
    p.write_bytes(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{mp[t]:010d}.json"))
    print(t, mp[t], p.stat().st_size); time.sleep(0.25)
