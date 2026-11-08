"""Build a worksheet for an independent human re-check of the 40 audited earnings rows (4 screen-flagged rows first)."""
import json, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent; K = ROOT.parent
import sys as _sys; _sys.path.insert(0, str(ROOT)); from kn.paths import DATA, RAW
x = pd.read_csv(DATA / "d7_audit_crosscheck.csv")
a = pd.read_csv(DATA / "d7_audit_results.csv")[["row_id", "evidence", "failed_check"]]
s = pd.read_csv(DATA / "d7_harvested_audit_sample.csv").reset_index().rename(columns={"index": "row_id"})[["row_id", "decision_time", "file_date", "adsh", "doc"]]
d = x.merge(a, on="row_id").merge(s, on="row_id")
cik = {t: json.loads((RAW / "sec_audit" / f"{t}.json").read_text())["cik"] for t in d.issuer.unique()}
d["sec_document_url"] = [f"https://www.sec.gov/Archives/edgar/data/{cik[t]}/{str(ad).replace('-', '')}/{doc}" for t, ad, doc in zip(d.issuer, d.adsh, d.doc)]
d["priority"] = (d.ai_verdict == "correct") & (d.guid_status != "range_found_gaap_diluted")
d = d.sort_values(["priority", "row_id"], ascending=[False, True])
cols = ["priority", "row_id", "issuer", "period", "fiscal_end", "file_date", "low", "high", "M", "actual_recorded", "eps_xbrl", "y_recorded",
        "ai_verdict", "failed_check", "evidence", "guid_status", "guid_flags", "eps_status", "sec_document_url"]
out = d[cols].copy()
for c in ["your_verdict (correct/wrong/unverifiable)", "your_failed_check (a/b/c/d)", "your_note"]:
    out[c] = ""
p = DATA / "d7_human_audit_sheet.csv"
out.to_csv(p, index=False)
print(p, len(out), int(out.priority.sum()))
