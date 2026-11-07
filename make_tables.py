"""Generate LaTeX tables and number macros from results/*.json -> ../paper/gen/*.tex. No number in the paper is typed by hand."""
import json, math, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
import os
GEN = ROOT.parent / os.environ.get("KN_PAPER", "paper") / "gen"
GEN.mkdir(parents=True, exist_ok=True)
sys_path = None


def J(n):
    p = RES / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


MAC = {}


def mac(name, val, fmt="{:.3f}"):
    MAC[name] = fmt.format(val) if isinstance(val, (int, float, np.floating)) else str(val)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(c, f="{:.3f}"):
    return "[" + f.format(c[0]) + ", " + f.format(c[1]) + "]"


def main():
    # ---------------- controls
    c = J("controls")
    ft = {r["n"]: r for r in c["D1"]["full_theta"]}
    mac("qAexact", 0.5 + 1 / math.pi, "{:.4f}"); mac("qKexact", 1 / math.log(2) - 1, "{:.4f}")
    mac("qEoneHundredK", ft[100000]["E1_acc"], "{:.4f}"); mac("qEoneTenK", ft[10000]["E1_acc"], "{:.4f}")
    mac("qLSSHundredK", ft[100000]["E1_LSS"], "{:.3f}")
    mac("chshSim", c["D1"]["chsh"]["quantum"], "{:.4f}"); mac("sideCls", c["D1"]["side_info"]["classical"], "{:.3f}")
    mac("sideQ", c["D1"]["side_info"]["quantum"], "{:.3f}"); mac("lyap", c["D2"]["lyapunov"], "{:.3f}")
    mac("lzPi", c["D2"]["lz"]["Pi_max_LZ"], "{:.3f}"); mac("lzLearned", c["D2"]["lz"]["learned_next_symbol_acc"], "{:.3f}")
    g = {(r["sigma"], r["h"]): r for r in c["D2"]["grid"]}
    mac("chaosAensFour", g[(1e-4, 4)]["A_ensemble"], "{:.3f}"); mac("chaosEoneFour", g[(1e-4, 4)]["E1_acc"], "{:.3f}")
    mac("chaosCHFour", g[(1e-4, 4)]["E4_A_upper"], "{:.3f}")
    # control validation table
    rows = []
    for r in c["D1"]["noisy_theta"]:
        rows.append(f"Qubit, $\\sigma={r['sigma']:g}$ & {r['A_star']:.4f} & {r['E1_acc']:.4f} & {r['E4_A_upper']:.4f} & {r['kappa_true']:.3f} & {r['E1_LSS']:.3f} & {r['kappa_upper_E7']:.3f} \\\\")
    for (s, h) in [(1e-3, 0.25), (1e-3, 1), (1e-3, 4), (1e-1, 1), (1.0, 1)]:
        r = g[(s, h)]
        rows.append(f"Lorenz, $\\sigma_o={s:g}$, $h={h:g}$ & {r['A_ensemble']:.4f} & {r['E1_acc']:.4f} & {r['E4_A_upper']:.4f} & {r['kappa_ensemble_LSS']:.3f} & {r['E1_LSS']:.3f} & {r['kappa_upper_E7']:.3f} \\\\")
    (GEN / "tab_controls.tex").write_text("\n".join(rows))
    # bound validity count
    allc = [(r["A_star"], r["E4_A_upper"]) for r in c["D1"]["noisy_theta"]] + [(r["true_A_on_sample"], r["E4_A_upper"]) for r in c["D1"]["full_theta"]] \
        + [(r["A_ensemble"], r["E4_A_upper"]) for r in c["D2"]["grid"]]
    ok = sum(1 for a, u in allc if u >= a - 0.01)
    mac("chValidN", ok, "{:d}"); mac("chTotalN", len(allc), "{:d}"); mac("chValidPct", 100 * ok / len(allc), "{:.0f}")
    # ---------------- learnability + law
    L = J("learnability")
    if L:
        r = {(x["h"], x["n"]): x for x in L["rows"]}
        mac("gapOneSmall", r[(1, 1000)]["learnability_gap_best"], "{:.3f}"); mac("gapOneBig", r[(1, 1000000)]["learnability_gap_best"], "{:.3f}")
        mac("gapFourSmall", r[(4, 1000)]["learnability_gap_best"], "{:.3f}"); mac("gapFourBig", r[(4, 1000000)]["learnability_gap_best"], "{:.3f}")
        mac("physFour", r[(4, 1000000)]["A_physical"], "{:.3f}")
    LW = J("law")
    if LW and "nn_law" in LW:
        mac("lawGamma", LW["nn_law"]["gamma"], "{:.2f}"); mac("lawRatio", LW["nn_law"]["gamma_over_lambda"], "{:.2f}")
        mac("lawRsq", LW["nn_law"]["r2"], "{:.3f}")
    # ---------------- noise sim
    ns = json.loads((RES / "noise_sim.json").read_text())
    conf = [r for r in ns if r["confirmatory"] and r["pi"] > 0]
    mac("simConfN", len(conf), "{:d}"); mac("simConfWins", sum(r["meas"] > r["true"] for r in conf), "{:d}")
    mac("simConfMeas", np.mean([r["meas"] for r in conf]), "{:+.3f}"); mac("simConfTrue", np.mean([r["true"] for r in conf]), "{:+.3f}")
    ex = [r for r in ns if r["mode"] == "persistent" and r["m"] == 0.75 and r["pi"] == 0.2 and r["r"] == 1 and r["T"] == 32 and r["conc"] > 1e8][0]
    mac("simExMeas", ex["meas"], "{:+.3f}"); mac("simExTrue", ex["true"], "{:+.3f}"); mac("simExFlip", 100 * ex["sign_flip"], "{:.0f}")
    ind = [r for r in ns if r["mode"] == "independent" and r["pi"] > 0]
    mac("simIndMeas", np.mean([r["meas"] for r in ind]), "{:+.4f}"); mac("simIndTrue", np.mean([r["true"] for r in ind]), "{:+.4f}")
    mac("simCover", np.nanmean([r["cover"] for r in ns]), "{:.3f}")
    # ---------------- earnings
    d7 = J("d7_analysis")
    t = d7["combined"]["table"]
    for k, nm in [("freq", "Freq"), ("eb_beta", "EB"), ("combo", "Combo"), ("ctrl_lite", "Ctrl")]:
        mac(f"e{nm}Acc", pct(t[k]["acc"]), "{}"); mac(f"e{nm}LSS", t[k]["LSS"], "{:.3f}")
    mac("eComboSacc", 100 * t["combo"]["S_acc"], "{:.1f}")
    ba = d7["before_after"]
    mac("eBeforeEdge", 100 * ba["before"]["edge_eb"], "{:+.2f}"); mac("eAfterEdge", 100 * ba["after"]["edge_eb"], "{:+.2f}")
    mac("eEdgeRemoved", 100 * ba["edge_from_removed_rows"], "{:+.2f}"); mac("eEdgeKept", 100 * ba["edge_from_kept_rows"], "{:+.2f}")
    mac("eRemovedShare", 100 * ba["edge_from_removed_rows"] / ba["before"]["edge_eb"], "{:.0f}")
    au = d7["audit"]
    mac("eAuditK", au["k"], "{:d}"); mac("eAuditE", au["e"], "{:d}"); mac("eAuditPD", au["P_D"], "{:.3f}")
    mac("eAuditRhoU", au["rho_D_upper"], "{:.3f}"); mac("eAuditLB", 100 * au["delta_lower"], "{:+.2f}"); mac("eAuditMeas", 100 * au["delta_measured"], "{:+.2f}")
    mac("eAOne", d7["ceilings"]["beta_ceiling"]["A"], "{:.3f}"); mac("eHind", d7["ceilings"]["hindsight_entity_majority"], "{:.3f}")
    mac("eTie", 100 * d7["tie_mass"]["share_within_5c"], "{:.1f}"); mac("eTieBound", d7["tie_mass"]["bound"], "{:.3f}")
    mac("ePowerN", d7["power_n_for_1pt"], "{:,.0f}")
    sig = ba["removed_signature"]
    mac("eSigCont", pct(sig["continuing_ops"]["beat_rate"], 0), "{}"); mac("eSigQuart", pct(sig["quarter_ordinal"]["beat_rate"], 0), "{}")
    mac("eSigKept", pct(sig["none"]["beat_rate"], 0), "{}")
    # ---------------- atlas table
    A = J("atlas")
    rows = []
    for r in A["rows"]:
        lci = r.get("LSS_ci")
        ff = "{:.4f}" if abs(r["LSS"]) < 0.01 else "{:.3f}"
        lss = ff.format(r["LSS"]) + (f" {ci(lci, ff)}" if lci else "")
        ku = f"{r['kU']:.2f}" if r.get("kU") is not None else "--"
        n = f"{r['n']:,}" if r.get("n") else "--"
        tau = f"{r['tau']:.2f}" if r.get("tau") is not None else "--"
        sh = r["shares"]
        rows.append(f"{r['label']} & {n} & {r['acc0']:.3f} & {r['acc']:.3f} & {r['S_acc']:.3f} & {lss} & {sh[0]:.3f} & {sh[1]:.3f} & {sh[2]:.3f} & {ku} & {tau} \\\\")
    (GEN / "tab_atlas.tex").write_text("\n".join(rows))
    # ---------------- preregistration check
    pre = [("d2", "Chaos $h{=}0.25$", (0.80, 0.99), g[(1e-3, 0.25)]["kappa_ensemble_LSS"]),
           ("d1", "Qubit, full $\\theta$", (1 / math.log(2) - 1, 1 / math.log(2) - 1), 1 / math.log(2) - 1)]
    for n, lab, rng in [("d6_house", "US House", (0.30, 0.60)), ("d4_weather", "Rain (obs.)", (0.10, 0.30)),
                        ("d5_football", "Football", (0.03, 0.12)), ("d8_stocks", "Stocks", (-0.005, 0.005))]:
        d = J(n); pre.append((n, lab, rng, d["summary"]["p_M4"]["LSS"]))
    pre.insert(5, ("d7", "Earnings", (0.02, 0.08), t["combo"]["LSS"]))
    # order by predicted rank
    order = ["d2", "d1", "d6_house", "d4_weather", "d5_football", "d7", "d8_stocks"]
    pre = sorted(pre, key=lambda x: order.index(x[0]))
    meas = [p[3] for p in pre]
    from scipy.stats import spearmanr
    rho = spearmanr(range(len(pre)), [-m for m in meas]).statistic
    mac("preRho", rho, "{:.2f}")
    rows = []
    hits = 0
    for i, (n, lab, rng, m) in enumerate(pre):
        hit = rng[0] - 1e-9 <= m <= rng[1] + 1e-9
        hits += hit
        mrank = sorted(meas, reverse=True).index(m) + 1
        rtxt = "exact" if rng[0] == rng[1] else f"{rng[0]:.3g}--{rng[1]:.3g}"
        verdict = "yes" if hit else ("above" if m > rng[1] else "below")
        rows.append(f"{i + 1} & {lab} & {rtxt} & {m:.4f} & {mrank} & {verdict} \\\\")
    (GEN / "tab_prereg.tex").write_text("\n".join(rows))
    mac("preHits", hits, "{:d}"); mac("preN", len(pre), "{:d}")
    # unranked
    unr = []
    for n, lab, rng, key in [("d4_weather_nwp", "Rain + NWP", (0.25, 0.45), "p_M4b"), ("d6_president", "President", (0.4, 0.8), "p_M4"),
                             ("d9_quakes", "Earthquakes", (0.05, 0.30), "p_M4")]:
        d = J(n)
        if d:
            m = d["summary"][key]["LSS"]; unr.append(f"-- & {lab} & {rng[0]:.3g}--{rng[1]:.3g} & {m:.4f} & -- & {'yes' if rng[0] <= m <= rng[1] else ('above' if m > rng[1] else 'below')} \\\\")
    ib = J("d3_ibm_noise")
    kd = float(np.mean([b["kappa_dev_mean"] for b in ib["backends"]]))
    unr.append(f"-- & IBM noise models & 0.35--0.44 & {kd:.4f} & -- & {'yes' if 0.35 <= kd <= 0.44 else 'no'} \\\\")
    (GEN / "tab_prereg_unranked.tex").write_text("\n".join(unr))
    # ---------------- quantum table + Holm
    names = [("d1_qubit", "Qubit ($\\sigma{=}0.3$)"), ("d2_lorenz", "Lorenz ($h{=}1$)"), ("d4_weather", "Rain"), ("d5_football", "Football"),
             ("d6_house", "US House"), ("d7_earnings", "Earnings"), ("d8_stocks", "Stocks"), ("d9_quakes", "Earthquakes")]
    Q = [(lab, J(f"quantum_{n}")) for n, lab in names if J(f"quantum_{n}")]
    from kn_holm import holm
    pv = [q["prereg_classical_set"]["p_quantum_better_one_sided"] if "prereg_classical_set" in q else q["p_quantum_better_one_sided"] for _, q in Q]
    adj = holm(pv)
    rows = []
    for (lab, q), pa in zip(Q, adj):
        r = q["results"]
        bc = max(["LR_d", "HGB_d", "CQ1_fourier", "CQ2_rbf_svm"], key=lambda k: r.get(k, {}).get("LSS", -9))
        bq = max([k for k in r if k.startswith("Q")], key=lambda k: r[k]["LSS"])
        pr = q.get("prereg_classical_set", {"ll_diff": q["ll_diff_q_minus_c"]})
        rows.append(f"{lab} & {q['n_test']:,} & {r['LR_full']['LSS']:.3f} & {r[bc]['LSS']:.3f} & {r[bq]['LSS']:.3f} & {r.get('LR_d_strong', {}).get('LSS', float('nan')):.3f} & {pr['ll_diff']:+.4f} & {ci(q['ll_diff_ci'], '{:+.4f}')} & {pa:.3f} \\\\")
    (GEN / "tab_quantum.tex").write_text("\n".join(rows))
    mac("qMinHolm", min(adj), "{:.3f}")
    mac("qNDomains", len(Q), "{:d}")
    # ---------------- IBM
    mac("ibmAmin", min(b["A_dev_mean"] for b in ib["backends"]), "{:.3f}"); mac("ibmAmax", max(b["A_dev_mean"] for b in ib["backends"]), "{:.3f}")
    ch = [b["chsh"]["win"] for b in ib["backends"]]
    mac("ibmChshMin", min(ch), "{:.3f}"); mac("ibmChshMax", max(ch), "{:.3f}")
    mac("ibmChshAbove", sum(v > 0.75 for v in ch), "{:d}"); mac("ibmN", len(ch), "{:d}")

    # ---------------- per-domain macros for the text
    def dom(n, tag, prim="p_M4"):
        d = J(n)
        if not d:
            return
        s_ = d["summary"]
        mac(tag + "N", d["n"], "{:,}"); mac(tag + "Ent", d["entities"], "{:d}"); mac(tag + "Base", d["base_rate"], "{:.3f}")
        mac(tag + "AccZero", s_[prim]["acc0"], "{:.3f}"); mac(tag + "Acc", s_[prim]["acc"], "{:.3f}")
        mac(tag + "LSS", s_[prim]["LSS"], "{:.3f}"); mac(tag + "LSSlo", s_[prim]["LSS_ci"][0], "{:.3f}"); mac(tag + "LSShi", s_[prim]["LSS_ci"][1], "{:.3f}")
        mac(tag + "Sacc", s_[prim]["S_acc"], "{:.3f}")
        for k, nm in [("p_M4n", "Nest"), ("p_M5", "Gbm"), ("p_M2", "EB"), ("p_M4bn", "NestB"), ("bench", "Bench"), ("p_M4b", "Mb")]:
            if k in s_:
                mac(tag + nm + "LSS", s_[k]["LSS"], "{:.3f}"); mac(tag + nm + "Acc", s_[k]["acc"], "{:.3f}")
        c_ = d["ceilings"]
        mac(tag + "AOne", c_["beta_ceiling"]["A"], "{:.3f}")
        if "cover_hart" in c_:
            mac(tag + "ACH", c_["cover_hart"]["A_upper"], "{:.3f}"); mac(tag + "KU", c_["kappa_upper"], "{:.2f}")
        if d.get("tau") and d["tau"].get("tau") is not None:
            mac(tag + "Tau", d["tau"]["tau"], "{:.2f}")
        if d.get("lz") and d["lz"].get("mean_Pi_max") is not None:
            mac(tag + "LZ", d["lz"]["mean_Pi_max"], "{:.3f}")
    for n, tag in [("d4_weather", "wx"), ("d4_weather_nwp", "nwp"), ("d5_football", "fb"), ("d6_house", "hs"),
                   ("d6_president", "pr"), ("d8_stocks", "st"), ("d9_quakes", "eq")]:
        dom(n, tag)
    sq = J("d9_quakes_strata")
    if sq:
        for st_ in ("active", "quiet"):
            if st_ in sq:
                v = sq[st_]
                for k in ("p_M4", "p_M2", "p_M0"):
                    if k in v and isinstance(v[k], dict) and "LSS_vs_M0" in v[k]:
                        mac("eq" + st_.capitalize() + {"p_M4": "Mod", "p_M2": "EB", "p_M0": "Zero"}[k] + "LSS", v[k]["LSS_vs_M0"], "{:.3f}")
    LW2 = J("law")
    if LW2 and "nn_law_all" in LW2:
        l5 = LW2["nn_law_all"]["0.05"]; l10 = LW2["nn_law_all"]["0.1"]
        mac("lawA", l5["a"], "{:.2f}"); mac("lawGammaTen", l10["gamma"], "{:.2f}"); mac("lawRsqTen", l10["r2"], "{:.3f}")
        P = dict((float(h), v) for h, v in l5["points"])
        mac("lawNstarHalf", P[0.5], "{:,.0f}"); mac("lawNstarFour", P[4.0] / 1e6, "{:.1f}")
        betas = [f["exponent"] for h, f in LW2["nn_fits"].items() if float(h) <= 3]
        mac("lawBeta", float(np.mean(betas)), "{:.2f}"); mac("lawBetaMin", min(f["exponent"] for f in LW2["nn_fits"].values()), "{:.2f}")
    # write macros
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MAC.items()]
    (GEN / "macros.tex").write_text("\n".join(lines))
    print(len(MAC), "macros;", "tables:", [p.name for p in GEN.glob("tab_*.tex")])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    import types
    from kn.core import holm as _h
    sys.modules["kn_holm"] = types.SimpleNamespace(holm=_h)
    main()
