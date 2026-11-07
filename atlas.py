"""Aggregate all domain results into results/atlas.json (shares of outcome entropy per information rung)."""
import json, math, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"


def J(n):
    p = RES / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def shares_from_ladder(lad):
    ll0 = lad["LL_M0"]
    f1 = max(ll0 - lad["LL_F1"], 0) / ll0
    f2 = max(lad["LL_F1"] - lad["LL_F2"], 0) / ll0
    f3 = max(lad["LL_F2"] - lad.get("LL_F3", lad["LL_F2"]), 0) / ll0 if "LL_F3" in lad else 0.0
    return [f1, f2, f3, max(1 - f1 - f2 - f3, 0)]


def domain_row(name, label, short, primary="p_M4"):
    d = J(name)
    if not d:
        return None
    if primary not in d["summary"]:  # results file predates the nested models
        primary = {"p_M4n": "p_M4", "p_M4bn": "p_M4b"}[primary]
    s = d["summary"][primary]
    return {"name": name, "label": label, "short": short, "shares": shares_from_ladder(d["ladder"]),
            "LSS": s["LSS"], "LSS_ci": s["LSS_ci"], "acc": s["acc"], "acc0": s["acc0"], "S_acc": s["S_acc"],
            "BSS": s["BSS"], "kU": d["ceilings"].get("kappa_upper"), "n": d["n"], "entities": d["entities"],
            "base_rate": d["base_rate"], "tau": (d.get("tau") or {}).get("tau"),
            "bench_LSS": d["summary"].get("bench", {}).get("LSS"), "A_CH": d["ceilings"].get("cover_hart", {}).get("A_upper"),
            "A1_beta": d["ceilings"]["beta_ceiling"]["A"], "period": d["period"]}


def main():
    rows = []
    c = J("controls")
    if c:
        g = {(r["sigma"], r["h"]): r for r in c["D2"]["grid"]}
        r = g[(0.001, 0.25)]
        rows.append({"name": "d2_lorenz_h025", "label": r"Chaos ($h{=}0.25$, $\sigma_o{=}10^{-3}$)", "short": "chaos",
                     "shares": [0, r["kappa_ensemble_LSS"], 0, 1 - r["kappa_ensemble_LSS"]], "LSS": r["kappa_ensemble_LSS"],
                     "acc": r["A_ensemble"], "acc0": 0.5, "S_acc": (r["A_ensemble"] - 0.5) / 0.5, "kU": r["kappa_upper_E7"], "control": True})
        k = 1 / math.log(2) - 1
        rows.append({"name": "d1_qubit", "label": r"Qubit (full $\theta$, exact)", "short": "qubit",
                     "shares": [0, k, 0, 1 - k], "LSS": k, "acc": 0.5 + 1 / math.pi, "acc0": 0.5,
                     "S_acc": 2 / math.pi, "kU": None, "control": True})
    ib = J("d3_ibm_noise")
    if ib:
        kd = float(np.mean([b["kappa_dev_mean"] for b in ib["backends"]])); ad = float(np.mean([b["A_dev_mean"] for b in ib["backends"]]))
        rows.append({"name": "d3_ibm", "label": "Qubit (IBM device noise)", "short": "IBM", "shares": [0, kd, 0, 1 - kd],
                     "LSS": kd, "acc": ad, "acc0": 0.5, "S_acc": (ad - 0.5) / 0.5, "kU": None, "control": True})
    for n, lab, sh in [("d6_president", "President by state", "president"), ("d6_house", "US House seats", "House"),
                       ("d4_weather", "Rain tomorrow", "rain"), ("d4_weather_nwp", "Rain tomorrow (+NWP)", "rain+NWP"),
                       ("d9_quakes", "Earthquakes (week, cell)", "quakes"), ("d5_football", "Football (+odds)", "football"),
                       ("d7_earnings", "Earnings guidance", "earnings"), ("d8_stocks", "Stock direction", "stocks")]:
        # atlas uses the nested top rung so that LSS equals the sum of the entropy shares (Deviation 4);
        # the preregistration scorecard (make_tables.py) still uses the preregistered p_M4.
        p = "p_M4bn" if n in ("d5_football", "d4_weather_nwp") else ("combo" if n == "d7_earnings" else "p_M4n")
        r = domain_row(n, lab, sh, primary=p) if n != "d7_earnings" else None
        if n == "d7_earnings":
            d = J("d7_analysis")
            if d:
                r = {"name": n, "label": lab, "short": sh, **d["atlas_row"]}
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -r["LSS"])
    (RES / "atlas.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    for r in rows:
        print(f"{r['short']:>10} LSS={r['LSS']:.4f} acc={r['acc']:.3f} acc0={r['acc0']:.3f} S_acc={r['S_acc']:.3f} "
              f"shares={np.round(r['shares'], 3).tolist()} kU={r.get('kU')}")


if __name__ == "__main__":
    main()
