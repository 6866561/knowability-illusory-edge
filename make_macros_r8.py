"""Macros for review round 8: issuer breakdown, coverage suite, many-seed chaos scaling, clean-room reproduction.
Writes ../paper_rev/gen/macros_r8.tex (copied to paper_know/gen by the build). Missing inputs are skipped."""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent; RES = ROOT / "results"
GEN = ROOT.parent / "paper_rev" / "gen"
M = {}
J = lambda n: json.loads((RES / f"{n}.json").read_text()) if (RES / f"{n}.json").exists() else None

ib = J("issuer_breakdown")
if ib:
    s = ib["summary"]
    M.update(ibInv=str(s["invalid_rows"]), ibIss=str(s["issuers_with_invalid"]), ibIssTot=str(s["issuers_total"]),
             ibRecur=str(s["issuers_invalid_ge2_periods"]), ibRecurRows=str(s["invalid_rows_in_issuers_ge2_periods"]),
             ibBeatInv=f"{100 * s['beat_rate_invalid']:.0f}", ibBeatVal=f"{100 * s['beat_rate_valid']:.0f}",
             ibEbInv=f"{100 * s['eb_matches_record_invalid']:.0f}", ibBaseInv=f"{100 * s['base_matches_record_invalid']:.0f}",
             ibEbVal=f"{100 * s['eb_matches_record_valid']:.0f}", ibBaseVal=f"{100 * s['base_matches_record_valid']:.0f}",
             ibEdgeAll=f"{s['edge_total_pts']:+.2f}", ibEdgeInv=f"{s['edge_invalid_pts']:+.2f}", ibEdgeVal=f"{s['edge_valid_pts']:+.2f}")
    tf = [p for p in s["persistence_issuers_ge3_invalid"] if p["issuer"] == "TFX"]
    if tf:
        M["ibTfxN"] = str(tf[0]["n"])
    top = {r["entity"]: r for r in ib["top"]}
    if "PNR" in top:
        M["ibPnrN"] = str(int(top["PNR"]["invalid_rows"])); M["ibPnrPer"] = str(int(top["PNR"]["invalid_periods"]))
    if "TFX" in top:
        M["ibTfxPer"] = str(int(top["TFX"]["invalid_periods"]))

cs = J("coverage_suite")
if cs:
    A = cs["part_A_two_way_LSS"]
    tag = {"independent": "Ind", "ar1_calendar_rho0.7": "Ar", "unequal_weights_lognormal_sd1": "Uw", "missing_30pct_cells": "Mis", "few_time_clusters_T8": "Few"}
    for k, t in tag.items():
        if k in A:
            for meth, mt in (("two_way", "Tw"), ("entity_only", "En"), ("time_only", "Ti")):
                M[f"cov{t}{mt}"] = f"{100 * A[k][meth]['coverage']:.1f}"
    M["covSE"] = f"{100 * max(v['two_way']['mc_se'] for v in A.values()):.1f}"
    M["covSEall"] = f"{100 * max(v[m]['mc_se'] for v in A.values() for m in ('two_way', 'entity_only', 'time_only')):.1f}"
    M["covTwMin"] = f"{100 * min(v['two_way']['coverage'] for v in A.values()):.1f}"
    M["covOneMin"] = f"{100 * min(min(v['entity_only']['coverage'], v['time_only']['coverage']) for v in A.values()):.1f}"
    M["covOneMax"] = f"{100 * max(max(v['entity_only']['coverage'], v['time_only']['coverage']) for v in A.values()):.1f}"
    rat = [v["two_way"]["median_width"] / v["entity_only"]["median_width"] for v in A.values()]
    M["covWlo"] = f"{min(rat):.1f}"; M["covWhi"] = f"{max(rat):.1f}"
    M["covSims"] = f"{cs['settings']['sims_A']:,}"; M["covReps"] = str(cs["settings"]["bootstrap_reps"])
    B = cs["part_B_audit"]
    M["covExactMin"] = f"{100 * B['exact_hypergeometric_min_coverage']:.1f}"
    sc = B["scenarios"]
    tb = {"actual_weights_random_errors": "AuRand", "actual_weights_errors_on_heaviest": "AuHeavy", "actual_weights_errors_one_issuer": "AuIss",
          "unequal_lognormal1.5_errors_on_heaviest": "AuUneq", "unresolved_counted_wrong_informative": "AuUnresW",
          "unresolved_excluded_informative": "AuUnresX", "adjudicator_misses_20pct": "AuMissTwenty", "adjudicator_misses_50pct": "AuMissFifty"}
    for k, t in tb.items():
        if k in sc:
            M[t + "P"] = f"{100 * sc[k]['coverage_plain']:.1f}"; M[t + "T"] = f"{100 * sc[k]['coverage_tight']:.1f}"
    M["covDrawsB"] = f"{cs['settings']['draws_B']:,}"

for sysname, t in (("lorenz63", "Lor"), ("rossler", "Ros")):
    d = J(f"law3s20_{sysname}")
    if not d:
        continue
    for ln, lt in (("1nn", "onenn"), ("knn", "knn")):
        for eps, et in (("0.02", "Two"), ("0.05", "Five"), ("0.1", "Ten")):
            g = [x["gamma_L"] for x in d["summary"][ln][eps]["per_seed"] if x["gamma_L"] is not None]
            if len(g) >= 3:
                rng = np.random.default_rng(1)
                bs = [np.mean(rng.choice(g, len(g))) for _ in range(5000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                M[f"ms{t}{lt}{et}"] = f"{np.mean(g):.2f}"; M[f"ms{t}{lt}{et}lo"] = f"{lo:.2f}"; M[f"ms{t}{lt}{et}hi"] = f"{hi:.2f}"
                M[f"ms{t}{lt}{et}sd"] = f"{np.std(g, ddof=1):.2f}"; M[f"ms{t}{lt}{et}n"] = str(len(g))
    M[f"ms{t}Seeds"] = str(len(d["seeds"]))
    sens = J(f"law3s20_sens_{sysname}")
    if sens:
        for k, v in sens.items():
            M[f"ms{t}{k}"] = v

rp = J("reproduction")
if rp:
    M["rpN"] = str(rp["n_compared"]); M["rpPass"] = str(rp["n_pass"]); M["rpMaxRel"] = f"{rp['max_rel_diff']:.1e}"; M["rpTol"] = rp["tolerance_text"]

(GEN / "macros_r8.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in M.items()) + "\n", encoding="utf-8")
print(len(M), "macros")
