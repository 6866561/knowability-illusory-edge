"""Macros and tables for revision 2 -> ../paper_rev/gen/{macros_rev2.tex, tab_law3.tex, tab_quantum_rev.tex, tab_atlas_rev.tex}."""
import json, math, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"; GEN = ROOT.parent / "paper_rev" / "gen"; GEN.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((RES / f"{n}.json").read_text())
M = {}
def mac(k, v, f="{:.3f}"):
    M[k] = f.format(v) if not isinstance(v, str) else v
def tag(s):
    return {"lorenz63": "Lor", "rossler": "Ros", "lorenz96_5": "Lfive", "lorenz96_6": "Lsix", "lorenz96_8": "Leight"}[s]
names = {"lorenz63": "Lorenz-63", "rossler": "R\\\"ossler", "lorenz96_5": "Lorenz-96 ($N{=}5$)", "lorenz96_6": "Lorenz-96 ($N{=}6$)", "lorenz96_8": "Lorenz-96 ($N{=}8$)$^\\ddagger$"}

# ---------------- chaos (law3)
rows, maxfloor = [], 0.0
for s in names:
    p = RES / f"law3_{s}.json"
    if not p.exists():
        continue
    d = J(f"law3_{s}"); t = tag(s)
    floors = [pl["floor_1nn"] for sd in d["seeds"] for pl in sd["per_L"].values()]
    maxfloor = max(maxfloor, max(floors))
    mac(t + "FloorMax", max(floors), "{:.4f}")
    st = {}
    for ln in ("1nn", "knn"):
        lt = {"1nn": "onenn", "knn": "knn"}[ln]
        sm = d["summary"][ln]["0.05"]
        g = [x["gamma_L"] for x in sm["per_seed"] if x["gamma_L"] is not None]
        statuses = [c for sd in d["seeds"] for pl in sd["per_L"].values() for c in [pl[f"cross_{ln}"]["0.05"]["status"]]]
        st[ln] = {k: statuses.count(k) for k in set(statuses)}
        if g:
            mac(f"{t}G{lt}", float(np.mean(g)), "{:.2f}"); mac(f"{t}Gsd{lt}", float(np.std(g, ddof=1)) if len(g) > 1 else 0.0, "{:.2f}")
        mac(f"{t}Nid{lt}", len(g), "{:d}")
        r2e = [x["r2_exp"] for x in sm["per_seed"] if x.get("r2_exp") is not None]; r2p = [x["r2_power"] for x in sm["per_seed"] if x.get("r2_power") is not None]
        if r2e:
            mac(f"{t}Rexp{lt}", float(np.mean(r2e)), "{:.3f}"); mac(f"{t}Rpow{lt}", float(np.mean(r2p)), "{:.3f}")
    for ln in ("1nn", "knn"):
        lt = {"1nn": "onenn", "knn": "knn"}[ln]
        for e, en in (("0.02", "Two"), ("0.1", "Ten")):
            g = [x["gamma_L"] for x in d["summary"][ln][e]["per_seed"] if x["gamma_L"] is not None]
            mac(f"{t}G{lt}{en}", float(np.mean(g)) if g else "--", "{:.2f}")
    ds = d["D2_sensitivity"]
    mac(t + "Dtwolo", ds["min"], "{:.2f}"); mac(t + "Dtwohi", ds["max"], "{:.2f}"); mac(t + "LamR", d["lambda"], "{:.4f}"); mac(t + "LamSE", d["lambda_se"], "{:.4f}")
    LT = {"1nn": "onenn", "knn": "knn"}
    fmt = lambda ln: (f"{M.get(f'{t}G{LT[ln]}', '--')} $\\pm$ {M.get(f'{t}Gsd{LT[ln]}', '--')} ({M[f'{t}Nid{LT[ln]}']}/3)"
                      if f"{t}G{LT[ln]}" in M else f"-- ({M[f'{t}Nid{LT[ln]}']}/3)")
    cnt = lambda ln: "/".join(str(st[ln].get(k, 0)) for k in ("interpolated", "extrapolated", "unattainable", "not_identifiable"))
    rows.append(f"{names[s]} & {d['lambda']:.4f} & {ds['min']:.2f}--{ds['max']:.2f} & {max(floors):.4f} & {fmt('1nn')} & {fmt('knn')} & {cnt('1nn')} & {cnt('knn')} \\\\")
(GEN / "tab_law3.tex").write_text("\n".join(rows), encoding="utf-8")
mac("floorMaxAll", maxfloor, "{:.4f}")

# posterior validation
pv = RES / "posterior_validation.json"
if pv.exists():
    v = J("posterior_validation")
    pc = v["prior_check"]
    mac("pvAgreeMin", min(x["decision_agreement"] for x in pc)); mac("pvEtaDiffMax", max(x["mean_abs_eta_diff"] for x in pc))
    mac("pvAdiffMax", max(abs(x["A_rule_library"] - x["A_rule_gauss"]) for x in pc), "{:.4f}")
    mac("pvFloorDiffMax", max(abs(x["floor_library"] - x["floor_gauss"]) for x in pc), "{:.4f}")
    mac("pvEssMed", min(x["median_ess"] for x in pc), "{:.0f}")
    cv = v["convergence"]
    mac("pvPartDiff", max(cv[L]["500"]["mean_abs_eta_diff_vs_2000"] for L in cv), "{:.4f}")
    mac("pvPartAdiff", max(abs(cv[L]["500"]["A_rule"] - cv[L]["2000"]["A_rule"]) for L in cv), "{:.3f}")
    mac("pvDtAgree", min(cv[L]["dt_half_vs_dt"]["decision_agreement"] for L in cv), "{:.3f}")

# ---------------- revision analyses
r = J("rev2_analyses")
au = r["audit"]; s1, s2 = au["scenarios"]["exclude_unverifiable"], au["scenarios"]["unverifiable_as_wrong"]
mac("auND", au["N_D_rows"], "{:d}"); mac("auPDw", au["P_D_weighted"]); mac("auDelta", 100 * au["delta_measured_weighted"], "{:+.2f}")
mac("auKUa", s1["K_U_hypergeom"], "{:d}"); mac("auRhoHa", s1["rho_U_hypergeom"]); mac("auRhoCPa", s1["rho_U_clopper_pearson"])
mac("auLBwHa", 100 * s1["delta_lower_weighted_hyper"], "{:+.2f}"); mac("auLBcpa", 100 * s1["delta_lower_cp_PDw"], "{:+.2f}")
mac("auKUb", s2["K_U_hypergeom"], "{:d}"); mac("auRhoHb", s2["rho_U_hypergeom"]); mac("auLBwHb", 100 * s2["delta_lower_weighted_hyper"], "{:+.2f}")
mac("auLBcpb", 100 * s2["delta_lower_cp_PDw"], "{:+.2f}")
mac("pwN", r["power"]["n_required"], "{:,.0f}"); mac("pwNde", r["power"]["n_required_design_effect_2"], "{:,.0f}")
mac("pwGroups", r["power"]["n_outcome_groups_available"], "{:d}"); mac("pwIss", r["power"]["n_issuers_available"], "{:d}")
ec = r["earnings_cohort"]
mac("ecRows", ec["fixed_cohort_rows"], "{:d}"); mac("ecTick", ec["fixed_cohort_tickers"], "{:d}")
mac("ecA", 100 * ec["A_archived_before_sweep_on_cohort"], "{:+.2f}"); mac("ecAlo", 100 * ec["A_ci"][0], "{:+.2f}"); mac("ecAhi", 100 * ec["A_ci"][1], "{:+.2f}")
mac("ecB", 100 * ec["B_after_sweep_retrained_on_cohort"], "{:+.2f}"); mac("ecBlo", 100 * ec["B_ci"][0], "{:+.2f}"); mac("ecBhi", 100 * ec["B_ci"][1], "{:+.2f}")
mac("ecRem", ec["removed_rows"], "{:d}"); mac("ecRemEdge", 100 * ec["removed_rows_edge_archived_original_labels"], "{:+.1f}")
mac("ecDiff", 100 * ec["B_minus_A_paired"], "{:+.2f}"); mac("ecDifflo", 100 * ec["B_minus_A_ci"][0], "{:+.2f}"); mac("ecDiffhi", 100 * ec["B_minus_A_ci"][1], "{:+.2f}")
acc_ = au["accounting"]
mac("auComb", acc_["n_combined_rows"], "{:d}"); mac("auAudOld", acc_["n_audited20_rows"], "{:d}"); mac("auPop", acc_["n_population_rows"], "{:d}")
mac("auIss", acc_["n_population_issuers"], "{:d}"); mac("auW", acc_["W_total"], "{:.0f}"); mac("auWD", acc_["W_D"], "{:.1f}")
mac("auWmin", acc_["w_min"], "{:.3f}"); mac("auPDu", au["P_D_unweighted"], "{:.3f}")
mac("auWtopA", s1["W_top_KU"], "{:.1f}"); mac("auWtopB", s2["W_top_KU"], "{:.1f}")
mac("auEa", s1["e"], "{:d}"); mac("auNa", s1["n"], "{:d}"); mac("auEb", s2["e"], "{:d}"); mac("auNb", s2["n"], "{:d}")
lv = r["readout"]["live"]
for q, v in lv.items():
    mac(f"ro{'A' if q == '5' else 'B'}", v["A_opt"], "{:.4f}"); mac(f"ro{'A' if q == '5' else 'B'}lo", v["A_opt_ci"][0], "{:.4f}"); mac(f"ro{'A' if q == '5' else 'B'}hi", v["A_opt_ci"][1], "{:.4f}")
    mac(f"ro{'A' if q == '5' else 'B'}ez", v["e0"], "{:.4f}"); mac(f"ro{'A' if q == '5' else 'B'}eo", v["e1"], "{:.4f}"); mac(f"ro{'A' if q == '5' else 'B'}k", v["kappa"], "{:.3f}")
mac("roFormulaDiff", r["readout"]["noise_models_max_abs_diff_formula_vs_opt"], "{:.1e}")
for k, v in r["time_block"].items():
    t = {"d4_weather": "wx", "d4_weather_nwp": "nwp", "d5_football": "fb", "d6_house": "hs", "d6_president": "pr", "d8_stocks": "st", "d9_quakes": "eq", "d7_earnings": "ed"}[k]
    mac(t + "TBlo", v["LSS_ci_time_block"][0], "{:.4f}" if abs(v["LSS"]) < 0.01 else "{:.3f}"); mac(t + "TBhi", v["LSS_ci_time_block"][1], "{:.4f}" if abs(v["LSS"]) < 0.01 else "{:.3f}")
    mac(t + "TBn", v["n_blocks"], "{:d}")

# ---------------- quantum (validation-selected)
qn = [("d1_qubit", "Qubit ($\\sigma_\\theta{=}0.3$)"), ("d2_lorenz", "Lorenz-63 ($h{=}1$)"), ("d4_weather", "Rain"), ("d5_football", "Football"),
      ("d6_house", "US House"), ("d7_earnings", "Earnings"), ("d8_stocks", "Stocks"), ("d9_quakes", "Earthquakes")]
qrows, pv_list = [], []
for n, lab in qn:
    p = RES / f"quantum2_{n}.json"
    if not p.exists():
        continue
    d = J(f"quantum2_{n}"); res = d["results"]
    pcons = max(d["p_boot_quantum_better"], d["p_boot_time_quantum_better"])   # conservative over both dependence structures
    pv_list.append(pcons)
    qrows.append(dict(lab=lab, n=d["n_test"], lc=res[d["best_classical_same_input"]]["LSS"], c=d["best_classical_same_input"].replace("_", "\_"),
                      lq=res[d["best_quantum"]]["LSS"], q=d["best_quantum"].replace("_", "\_"), dll=d["ll_diff_q_minus_c"], ci=d["ll_diff_ci"],
                      cit=d["ll_diff_ci_time_block"], dm=d["p_quantum_better_one_sided"], pe=d["p_boot_quantum_better"], pt=d["p_boot_time_quantum_better"], name=n))
if qrows:
    import sys; sys.path.insert(0, str(ROOT)); from kn.core import holm
    adj = holm(pv_list)
    out = []
    for r_, pa in zip(qrows, adj):
        out.append(f"{r_['lab']} & {r_['n']:,} & {r_['lc']:.3f} ({r_['c']}) & {r_['lq']:.3f} ({r_['q']}) & {r_['dll']:+.4f} & [{r_['ci'][0]:+.4f}, {r_['ci'][1]:+.4f}] & [{r_['cit'][0]:+.4f}, {r_['cit'][1]:+.4f}] & {r_['dm']:.3f} & {r_['pe']:.3f} / {r_['pt']:.3f} & {pa:.3f} \\\\")
    (GEN / "tab_quantum_rev.tex").write_text("\n".join(out), encoding="utf-8")
    mac("qrevMinHolm", float(min(adj))); mac("qrevN", len(qrows), "{:d}")
    mac("qrevNsigWorse", sum(1 for r_ in qrows if r_["ci"][0] > 0 and r_["cit"][0] > 0), "{:d}")
    st_ = [r_ for r_ in qrows if r_["name"] == "d8_stocks"]
    if st_:
        r_ = st_[0]
        mac("qrevStPe", r_["pe"], "{:.4f}"); mac("qrevStPt", r_["pt"], "{:.3f}"); mac("qrevStDll", r_["dll"], "{:+.4f}")
        mac("qrevStLq", r_["lq"], "{:+.4f}"); mac("qrevStLc", r_["lc"], "{:+.4f}")
        mac("qrevStCilo", r_["cit"][0], "{:+.4f}"); mac("qrevStCihi", r_["cit"][1], "{:+.4f}")

(GEN / "macros_rev2.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in M.items()) + "\n", encoding="utf-8")
print(len(M), "macros"); print({k: M[k] for k in list(M) if k.startswith(("Lor", "Ros", "Lfive", "Lsix", "Leight", "floor", "au", "ec", "pw", "qrev", "pv"))})

# normalise negative zeros produced by rounding in generated tables (display only)
import re as _re
for _t in ("tab_atlas.tex", "tab_controls.tex", "tab_law3.tex", "tab_quantum_rev.tex"):
    _p = GEN / _t
    if _p.exists():
        _s = _p.read_text(encoding="utf-8")
        _s2 = _re.sub(r"(?<=& )-(0\.0+)(?= &)", r"\1", _s)   # standalone cells only; CI endpoints keep their sign
        if _s2 != _s:
            _p.write_text(_s2, encoding="utf-8"); print("normalised -0 in", _t)

# atlas table with a time-block interval column (derived file; tab_atlas.tex itself is left unchanged)
_tbmap = {"Rain tomorrow (+NWP)": "d4_weather_nwp", "President by state": "d6_president", "US House seats": "d6_house",
          "Football (+odds)": "d5_football", "Earnings guidance": "d7_earnings", "Rain tomorrow": "d4_weather",
          "Earthquakes (week, cell)": "d9_quakes", "Stock direction": "d8_stocks"}
_tb = r["time_block"]
_rows = []
for _line in (GEN / "tab_atlas.tex").read_text(encoding="utf-8").splitlines():
    _cells = _line.split(" & ")
    _name = _cells[0].strip()
    if _name in _tbmap and _tbmap[_name] in _tb:
        _v = _tb[_tbmap[_name]]; _f = "{:.4f}" if abs(_v["LSS"]) < 0.01 else "{:.3f}"
        _ins = "[" + _f.format(_v["LSS_ci_time_block"][0]) + ", " + _f.format(_v["LSS_ci_time_block"][1]) + "]"
    elif _name.startswith("Bad qubit"):
        _ins = "qubit only$^\S$"
    else:
        _ins = "--"
    _cells.insert(6, _ins)
    _rows.append(" & ".join(_cells))
(GEN / "tab_atlas_tb.tex").write_text("\n".join(_rows), encoding="utf-8")
print("tab_atlas_tb written")
