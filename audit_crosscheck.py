"""Independent cross-checks of the 40-row earnings disagreement audit (no AI judgement involved).

(1) Reported-EPS side: first-reported diluted EPS from SEC XBRL company facts (us-gaap:EarningsPerShareDiluted, falling back to
    EarningsPerShareBasicAndDiluted) for the row's fiscal period end and period length (annual 330-380 days, quarter 80-100 days),
    compared with the recorded actual (tolerance 0.005), and the label recomputed as 1{EPS > M}.
(2) Guidance side: a rule-based screen of the press-release text for a numeric range whose endpoints match the recorded
    guidance range (tolerance 0.005), with flags for nearby "continuing operations", "adjusted"/"non-GAAP" wording.
Writes ../data/d7_audit_crosscheck.csv and results/audit_crosscheck.json.
"""
import json, pathlib, re
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent; K = ROOT.parent
import sys as _sys; _sys.path.insert(0, str(ROOT)); from kn.paths import DATA, RAW
aud = pd.read_csv(DATA / "d7_audit_results.csv")
smp = pd.read_csv(DATA / "d7_harvested_audit_sample.csv").reset_index().rename(columns={"index": "row_id"})
d = aud.drop(columns=["fiscal_end", "period"], errors="ignore").merge(smp, on="row_id", how="left")
assert len(d) == 40 and (d.issuer == d.entity).all()
TEXT = K / "pkg_rerun" / "data" / "edgar" / "text"
TOL = 0.005


def eps_facts(ticker):
    f = json.loads((RAW / "sec_audit" / f"{ticker}.json").read_text())
    g = f.get("facts", {}).get("us-gaap", {})
    out = []
    for tag in ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"):
        for unit, rows in g.get(tag, {}).get("units", {}).items():
            if unit.replace(" ", "").lower() in ("usd/shares", "usd/share"):
                for r in rows:
                    if r.get("start"):
                        out.append(dict(tag=tag, start=r["start"], end=r["end"], val=r["val"], filed=r["filed"], form=r.get("form")))
        if out:
            break
    return pd.DataFrame(out)


cache, rows = {}, []
for _, r in d.iterrows():
    t = r.issuer
    if t not in cache:
        cache[t] = eps_facts(t)
    f = cache[t]
    rec = dict(row_id=int(r.row_id), issuer=t, fiscal_end=r.fiscal_end, period=r.period_kind_name, M=r.M, low=r["low"], high=r["high"],
               actual_recorded=r.actual, y_recorded=int(r.y), ai_verdict=r.verdict)
    # ---- (1) reported EPS
    xb = None
    if len(f):
        dur = (pd.to_datetime(f.end) - pd.to_datetime(f.start)).dt.days
        want = dur.between(330, 380) if r.period_kind_name == "annual" else dur.between(80, 100)
        c = f[(f.end == str(r.fiscal_end)[:10]) & want].sort_values("filed")
        if len(c):
            xb = c.iloc[0]
    if xb is None:
        rec.update(eps_status="not_in_xbrl")
    else:
        rec.update(eps_xbrl=float(xb.val), eps_filed=xb.filed, eps_form=xb.form, eps_tag=xb.tag)
        rec["eps_match"] = bool(abs(float(xb.val) - float(r.actual)) <= TOL)
        rec["y_xbrl"] = int(float(xb.val) > float(r.M) + 1e-9)
        rec["label_match"] = bool(rec["y_xbrl"] == int(r.y))
        rec["eps_status"] = "match" if rec["eps_match"] else ("label_same_value_differs" if rec["label_match"] else "label_differs")
    # ---- (2) rule-based guidance screen
    tf = TEXT / str(r.text_file)
    if not tf.exists():
        rec.update(guid_status="text_missing")
    else:
        txt = tf.read_text(encoding="utf-8", errors="ignore")
        num = r"\(?\$?\s?(-?\d{1,3}\.\d{2})\)?"
        hits = []
        for m in re.finditer(num + r"\s*(?:to|-|–|—|and|through)\s*" + num, txt):
            lo, hi = float(m.group(1)), float(m.group(2))
            if abs(lo - r["low"]) <= TOL and abs(hi - r["high"]) <= TOL:
                w = txt[max(0, m.start() - 400): m.end() + 200].lower()
                hits.append(dict(cont=("continuing operations" in w), adj=bool(re.search(r"non-gaap|adjusted", w)),
                                 gaap=bool(re.search(r"(?<!non-)gaap", w)), dil=("diluted" in w)))
        if not hits:
            rec.update(guid_status="range_not_found")
        else:
            clean = [h for h in hits if h["gaap"] and h["dil"] and not h["cont"]]
            rec.update(guid_status="range_found_gaap_diluted" if clean else "range_found_flagged",
                       guid_flags=";".join(sorted({k for h in hits for k, v in h.items() if v and k in ("cont", "adj")})))
    rows.append(rec)

o = pd.DataFrame(rows)
if "guid_flags" not in o:
    o["guid_flags"] = ""   # press-release texts absent (e.g. clean clone): guidance screen not run
o.to_csv(DATA / "d7_audit_crosscheck.csv", index=False)
summ = {"n": len(o), "eps_status": o.eps_status.value_counts().to_dict(), "guid_status": o.guid_status.value_counts().to_dict(),
        "eps_by_ai_verdict": {f"{a}|{b}": int(v) for (a, b), v in o.groupby(["ai_verdict", "eps_status"]).size().items()},
        "guid_by_ai_verdict": {f"{a}|{b}": int(v) for (a, b), v in o.groupby(["ai_verdict", "guid_status"]).size().items()}}
(ROOT / "results" / "audit_crosscheck.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=str))
print(json.dumps(summ, indent=1, default=str))
print(o[o.eps_status != "match"][["row_id", "issuer", "fiscal_end", "period", "M", "actual_recorded", "eps_xbrl", "y_recorded", "eps_status", "ai_verdict"]].to_string())
print(o[o.guid_status != "range_found_gaap_diluted"][["row_id", "issuer", "period", "low", "high", "guid_status", "guid_flags", "ai_verdict"]].to_string())

# ---- sensitivity of the audit bound to stricter treatments of rows not confirmed by both checks
from scipy import stats
d7 = pd.read_parquet(DATA / "d7_combined.parquet").reset_index(drop=True)
hv = d7[d7["src"] != "audited20"].reset_index(drop=True)
y, w = hv["y"].to_numpy().astype(int), hv["weight"].to_numpy()
de, df_ = (hv["eb_beta"].to_numpy() >= .5).astype(int), (hv["freq"].to_numpy() >= .5).astype(int)
D = de != df_; N_D = int(D.sum()); W = float(w.sum())
delta = float(np.sum(w * ((de == y).astype(float) - (df_ == y).astype(float))) / W)


def k_upper(N, n, e, alpha=0.05):
    K_ = e
    for Kk in range(e, N + 1):
        if stats.hypergeom.cdf(e, N, Kk, n) > alpha:
            K_ = Kk
        else:
            break
    return K_


wD = np.sort(w[D])[::-1]
smp_w = smp.set_index("row_id")["weight"]
unaud = np.sort(smp.loc[~smp.row_id.isin(o.row_id), "weight"].to_numpy())[::-1]
flag_guid = set(o.loc[(o.ai_verdict == "correct") & (o.guid_status == "range_found_flagged"), "row_id"])   # only rows the screen actually flagged
eps_unchk = set(o.loc[o.eps_status != "match", "row_id"])
bad_base = set(o.loc[o.ai_verdict != "correct", "row_id"])
sens = {}
for name, bad in [("ai_primary", bad_base), ("plus_screen_flagged", bad_base | flag_guid), ("plus_screen_and_eps_unchecked", bad_base | flag_guid | eps_unchk)]:
    e = len(bad); Ku = k_upper(N_D, 40, e)
    plain = delta - 2 * wD[:Ku].sum() / W
    tight = delta - 2 * (float(smp_w.loc[list(bad)].sum()) + unaud[:max(Ku - e, 0)].sum()) / W
    sens[name] = {"e": e, "n": 40, "K_U": Ku, "bound_plain_pts": 100 * plain, "bound_tight_pts": 100 * tight}
out = json.loads((ROOT / "results" / "audit_crosscheck.json").read_text())
out["bound_sensitivity"] = sens; out["delta_measured_pts"] = 100 * delta
(ROOT / "results" / "audit_crosscheck.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(sens, indent=1))
