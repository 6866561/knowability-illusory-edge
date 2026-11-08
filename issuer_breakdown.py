"""Issuer-level breakdown of the earnings edge before cleaning (Illusory Edge, reviewer item: make the mechanism inspectable).

For the rows scored before cleaning (archived predictions, original labels): per issuer, the number of rows and of rows later
removed as invalid (by reason), the number of distinct fiscal periods with invalid rows (recurrence), the recorded-outcome rate
on invalid vs valid rows, how often the entity (EB) rule and the base-rate rule disagree, and each issuer's contribution to the
weighted EB edge over the base rate (points), split into invalid and valid rows. Writes results/issuer_breakdown.json,
data/d7_issuer_breakdown.csv and ../paper_rev/figs/fig_issuer.{pdf,png}.
"""
import json, pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent; K = ROOT.parent
import sys as _sys; _sys.path.insert(0, str(ROOT)); from kn.paths import DATA, RAW
sys.path.insert(0, str(ROOT))
import figs as F
_fig_dir = K / "paper_rev" / "figs" if (K / "paper_rev").exists() else ROOT / "results" / "figs"   # clean clone: results/figs
_fig_dir.mkdir(parents=True, exist_ok=True)
F.OUT = _fig_dir
from figs import save, COL1

b = pd.read_parquet(DATA / "d7_before_after.parquet")
d = b[b.scored_before == 1].copy()
d["inv"] = d.removed_by != "none"
w = d.weight_before.to_numpy(); W = w.sum()
ce = ((d.eb_beta_before >= .5).astype(int) == d.y).astype(float); cf = ((d.freq_before >= .5).astype(int) == d.y).astype(float)
d["contrib"] = 100 * w * (ce - cf) / W                       # points of the pooled weighted edge
d["disagree"] = (d.eb_beta_before >= .5) != (d.freq_before >= .5)
d["eb_follows_record"] = ((d.eb_beta_before >= .5).astype(int) == d.y)

g = d.groupby("entity")
tab = pd.DataFrame({
    "rows": g.size(), "invalid_rows": g.inv.sum(),
    "invalid_periods": d[d.inv].groupby("entity").fiscal_end.nunique(),
    "reasons": d[d.inv].groupby("entity").removed_by.agg(lambda s: ";".join(sorted(set(s)))),
    "beat_rate_invalid": d[d.inv].groupby("entity").y.mean(), "beat_rate_valid": d[~d.inv].groupby("entity").y.mean(),
    "disagreements_invalid": d[d.inv].groupby("entity").disagree.sum(),
    "edge_pts_invalid": d[d.inv].groupby("entity").contrib.sum(), "edge_pts_valid": d[~d.inv].groupby("entity").contrib.sum(),
}).fillna({"invalid_rows": 0, "invalid_periods": 0, "reasons": "", "disagreements_invalid": 0, "edge_pts_invalid": 0.0, "edge_pts_valid": 0.0})
tab = tab.sort_values("edge_pts_invalid", ascending=False)
tab.to_csv(DATA / "d7_issuer_breakdown.csv")

inv = d[d.inv]
top2 = tab.head(2)
summary = {
    "rows_scored": int(len(d)), "invalid_rows": int(d.inv.sum()), "issuers_with_invalid": int((tab.invalid_rows > 0).sum()),
    "issuers_total": int(len(tab)),
    "edge_total_pts": float(d.contrib.sum()), "edge_invalid_pts": float(inv.contrib.sum()), "edge_valid_pts": float(d[~d.inv].contrib.sum()),
    "top2_issuers": list(top2.index), "top2_invalid_rows": int(top2.invalid_rows.sum()), "top2_edge_invalid_pts": float(top2.edge_pts_invalid.sum()),
    "issuers_invalid_ge2_periods": int((tab.invalid_periods >= 2).sum()),
    "invalid_rows_in_issuers_ge2_periods": int(tab.loc[tab.invalid_periods >= 2, "invalid_rows"].sum()),
    "beat_rate_invalid": float(inv.y.mean()), "beat_rate_valid": float(d[~d.inv].y.mean()),
    "eb_matches_record_invalid": float(inv.eb_follows_record.mean()), "base_matches_record_invalid": float(((inv.freq_before >= .5).astype(int) == inv.y).mean()),
    "eb_matches_record_valid": float(d[~d.inv].eb_follows_record.mean()), "base_matches_record_valid": float(((d[~d.inv].freq_before >= .5).astype(int) == d[~d.inv].y).mean()),
}
# within-issuer persistence: for issuers with >= 3 invalid rows, share of their invalid rows that share the issuer's majority recorded outcome
pers = []
for e, s in inv.groupby("entity"):
    if len(s) >= 3:
        m = s.y.mean(); pers.append({"issuer": e, "n": int(len(s)), "majority_share": float(max(m, 1 - m))})
summary["persistence_issuers_ge3_invalid"] = pers
(ROOT / "results" / "issuer_breakdown.json").write_text(json.dumps({"summary": summary, "top": tab.head(12).reset_index().to_dict("records")}, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float))
print(tab.head(10).to_string())

# figure: per-issuer contribution of invalid vs valid rows (issuers with invalid rows, sorted)
t = tab[tab.invalid_rows > 0].head(14)[::-1]
fig, ax = plt.subplots(figsize=(COL1, 2.9))
yy = np.arange(len(t))
ax.barh(yy - 0.18, t.edge_pts_invalid, height=0.36, color="#e67e22", label="rows later removed as invalid")
ax.barh(yy + 0.18, t.edge_pts_valid, height=0.36, color="#1b3a6b", label="valid rows")
for yi, (n_inv, n_per) in zip(yy, zip(t.invalid_rows, t.invalid_periods)):
    ax.text(ax.get_xlim()[1] if False else 0, yi, "", fontsize=5)
ax.set_yticks(yy); ax.set_yticklabels([f"{e} ({int(r)} invalid, {int(p)} period{'s' if int(p) != 1 else ''})" for e, r, p in zip(t.index, t.invalid_rows, t.invalid_periods)], fontsize=5.6)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("contribution to pooled EB edge (points)")
ax.legend(fontsize=5.6, loc="lower right")
fig.tight_layout()
save(fig, "fig_issuer")
