"""Macros for the extensions (five chaotic systems, drift forecasting, Delft Bell test, live IBM run) -> ../paper/gen/macros_ext.tex
Also adds two rows to results/atlas.json (drift task B, live ibm_kingston qubit)."""
import json, math, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RES, GEN = ROOT / "results", ROOT.parent / "paper" / "gen"
J = lambda n: json.loads((RES / f"{n}.json").read_text())
M = {}
def mac(k, v, f="{:.3f}"):
    M[k] = f.format(v) if not isinstance(v, str) else v
def h(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log(p) + (1 - p) * np.log(1 - p))

# five chaotic systems
names = {"lorenz63": "Lor", "rossler": "Ros", "lorenz96_5": "Lfive", "lorenz96_6": "Lsix", "lorenz96_8": "Leight"}
rows = []
for key, tag in names.items():
    d = J(f"law2_{key}"); l = d["law"]
    mac(tag + "Lam", d["lambda"], "{:.4f}"); mac(tag + "Dtwo", d["D2"], "{:.2f}"); mac(tag + "Beta", d["beta_mean"], "{:.2f}")
    mac(tag + "InvBeta", 1 / d["beta_mean"], "{:.2f}")
    mac(tag + "G", l["gamma_L"], "{:.2f}"); mac(tag + "Glo", l["gamma_L_lo"], "{:.2f}"); mac(tag + "Ghi", l["gamma_L_hi"], "{:.2f}")
    mac(tag + "Rsq", l["r2"], "{:.3f}"); mac(tag + "Ratio", l["gamma_L"] / d["D2"], "{:.1f}")
    rows.append(f"{ {'lorenz63': 'Lorenz-63', 'rossler': 'R\\\"ossler', 'lorenz96_5': 'Lorenz-96 ($N{=}5$)', 'lorenz96_6': 'Lorenz-96 ($N{=}6$)', 'lorenz96_8': 'Lorenz-96 ($N{=}8$)'}[key] } & "
                f"{d['lambda']:.4f} & {d['D2']:.2f} & {d['beta_mean']:.2f} & {1 / d['beta_mean']:.2f} & {l['gamma_L']:.2f} [{l['gamma_L_lo']:.2f}, {l['gamma_L_hi']:.2f}]"
                f"{'$^\\dagger$' if not l['only_interpolated'] else ''} & {l['r2']:.3f} \\\\")
(GEN / "tab_scaling.tex").write_text("\n".join(rows), encoding="utf-8")
r2s = [J(f"law2_{k}")["law"]["r2"] for k in names]
mac("scaleRsqMin", min(r2s)); mac("scaleRsqMax", max(r2s))

# drift
P, S = J("qdrift_primary"), J("qdrift_secondary")
best = lambda A, hh: max(A[f"h{hh}"]["skill_ridge_vs_clim"], A[f"h{hh}"]["skill_hgb_vs_clim"], key=lambda d: d["skill"])
for m, tag in [("readout_error", "Ro"), ("cz_error", "Cz"), ("sx_error", "Sx"), ("T1", "Tone"), ("T2", "Ttwo")]:
    A = P["taskA"][m]
    for hh, ht in [(1, "One"), (14, "Ft")]:
        b = best(A, hh); mac(f"dr{tag}{ht}", b["skill"]); mac(f"dr{tag}{ht}lo", b["lo"]); mac(f"dr{tag}{ht}hi", b["hi"])
    mac(f"dr{tag}Pers", A["h1"]["skill_pers_vs_clim"]["skill"])
    mac(f"dr{tag}VsPers", max(A["h1"]["skill_ridge_vs_pers"]["skill"], A["h1"]["skill_hgb_vs_pers"]["skill"]))
pd_ = P["data"]
mac("drQubits", pd_["readout_error"]["qubits"], "{:d}"); mac("drDays", pd_["readout_error"]["days"], "{:d}")
mac("drObs", sum(v["obs"] for v in pd_.values()), "{:,}"); mac("drEdges", pd_["cz_error"]["series"], "{:,}")
mac("drTOneObs", pd_["T1"]["obs"], "{:,}")
mac("drRoSecOne", best(S["taskA"]["readout_error"], 1)["skill"]); mac("drToneSecOne", best(S["taskA"]["T1"], 1)["skill"])
mac("drSecDays", S["data"]["readout_error"]["days"], "{:d}")
for D, t in [(P, "P"), (S, "S")]:
    tb = D["taskB"]
    mac(f"db{t}Base", tb["base_rate"]); mac(f"db{t}Rows", tb["n_rows"], "{:,}")
    mac(f"db{t}BadBad", tb["persistence"]["1"]["p_bad_given_bad"]); mac(f"db{t}BadGood", tb["persistence"]["1"]["p_bad_given_good"])
    mac(f"db{t}BadBadFt", tb["persistence"]["14"]["p_bad_given_bad"])
    for mod, mt in [("qubit_rate", "Hist"), ("logit", "Logit"), ("hgb", "Hgb")]:
        mac(f"db{t}{mt}LSS", tb[mod]["LSS"]); mac(f"db{t}{mt}LSSlo", tb[mod]["ci"]["LSS"][0]); mac(f"db{t}{mt}LSShi", tb[mod]["ci"]["LSS"][1])
    mac(f"db{t}HgbAcc", tb["hgb"]["acc"]); mac(f"db{t}AccZero", tb["hgb"]["acc0"]); mac(f"db{t}Kappa", tb["ceiling"]["kappa"])
mac("dbPAuc", P["taskB"]["auc_hgb"])

# Bell
b = J("bell_delft")
mac("bellN", b["n_trials"], "{:d}"); mac("bellK", b["omega"]["k"], "{:d}"); mac("bellS", b["S"]["combined"], "{:.2f}"); mac("bellSerr", b["S"]["combined_err"], "{:.2f}")
mac("bellW", b["omega"]["omega"]); mac("bellWlo", b["omega"]["CP95"][0]); mac("bellWhi", b["omega"]["CP95"][1]); mac("bellP", b["omega"]["p_binom_gt_0.75"])
pb = b["prediction_Bob_y"]
mac("bellOwn", pb["from_b_only"]["loo_acc"]); mac("bellOwnlo", pb["from_b_only"]["loo_CP95"][0]); mac("bellOwnhi", pb["from_b_only"]["loo_CP95"][1])
mac("bellSide", pb["from_a_b_x"]["loo_acc"]); mac("bellSidelo", pb["from_a_b_x"]["loo_CP95"][0]); mac("bellSidehi", pb["from_a_b_x"]["loo_CP95"][1])
mac("bellState", pb["from_a_b_x_state"]["loo_acc"])

# live IBM
L = J("ibm_real_ibm_kingston_20261107")
w = L["chsh_win"]; nl = 4 * L["shots"]
mac("liveChsh", w, "{:.4f}"); mac("liveChshSE", math.sqrt(w * (1 - w) / nl), "{:.4f}"); mac("liveTrials", nl, "{:,}")
qs = sorted(k for k in L if k.startswith("qubit_"))
kd = []
for k in qs:
    a, bb = L[k]["alpha"], L[k]["beta"]
    tt = np.linspace(0, math.pi, 20001); pp = np.clip(a + bb * np.sin(tt / 2) ** 2, 0, 1)
    kd.append(float(1 - h(pp).mean() / h(pp.mean())))
mac("liveAqA", L[qs[0]]["A_dev"], "{:.4f}"); mac("liveAqB", L[qs[1]]["A_dev"], "{:.4f}")
mac("liveQA", qs[0].split("_")[1]); mac("liveQB", qs[1].split("_")[1])
mac("liveKappa", float(np.mean(kd))); mac("liveLoss", (0.5 + 1 / math.pi) - float(np.mean([L[k]["A_dev"] for k in qs])), "{:.4f}")
ro = [v["readout_error"] for v in L["calibration_snapshot"].values()]; t1 = [v["T1"] * 1e6 for v in L["calibration_snapshot"].values()]
mac("liveRoMin", 100 * min(ro), "{:.2f}"); mac("liveRoMax", 100 * max(ro), "{:.2f}"); mac("liveTOneMin", min(t1), "{:.0f}"); mac("liveTOneMax", max(t1), "{:.0f}")
mac("liveDate", L["utc"][:10])
d3m = json.loads((RES / "drift3d_meta.json").read_text())
mac("landDays", d3m["readout_error"]["n_days"], "{:d}"); mac("landQubits", d3m["readout_error"]["n_qubits"], "{:d}")

# ---- review-round additions
import pandas as pd
sec = pd.read_csv(ROOT.parent / "raw" / "qdrift" / "ibm_calibration_history.csv", usecols=["snapshot_time"])
t0, t1 = pd.to_datetime(sec.snapshot_time.min(), utc=True), pd.to_datetime(sec.snapshot_time.max(), utc=True)
mac("drSecDays", (t1 - t0).days + 1, "{:d}"); mac("drSecStart", t0.strftime("%-d %B %Y") if False else f"{t0.day} {t0:%B %Y}"); mac("drSecEnd", f"{t1.day} {t1:%B %Y}")
for m, tag in [("T1", "Tone"), ("T2", "Ttwo"), ("readout_error", "Ro")]:
    b2 = best(S["taskA"][m], 1); mac(f"dr{tag}SecOne", b2["skill"]); mac(f"dr{tag}SecOnelo", b2["lo"]); mac(f"dr{tag}SecOnehi", b2["hi"])
lawj = J("law"); lam63 = J("law2_lorenz63")["lambda"]
import scipy.stats as st
Pp = np.array(lawj["nn_law_all"]["0.05"]["points"]); rr = st.linregress(Pp[:, 0], np.log(Pp[:, 1])); tc = st.t.ppf(0.975, len(Pp) - 2)
mac("lawGL", rr.slope / lam63, "{:.2f}"); mac("lawGLlo", (rr.slope - tc * rr.stderr) / lam63, "{:.2f}"); mac("lawGLhi", (rr.slope + tc * rr.stderr) / lam63, "{:.2f}")
mac("lamCtl", J("controls")["D2"]["lyapunov"], "{:.3f}")
live_mean = float(np.mean([L[k]["A_dev"] for k in qs]))
mac("liveAboveModels", sum(1 for bk in ib_ if bk["A_dev_mean"] < live_mean) if (ib_ := J("d3_ibm_noise")["backends"]) else 0, "{:d}")
mac("liveGapTs", (math.cos(math.pi / 8) ** 2) - w, "{:.3f}"); mac("liveGapTsSE", ((math.cos(math.pi / 8) ** 2) - w) / math.sqrt(w * (1 - w) / nl), "{:.1f}")
mac("bellWlit", b["omega"]["omega_literal_xor_eq_a_and_b_all_trials"]); mac("bellWlitlo", b["omega"]["literal_CP95"][0]); mac("bellWlithi", b["omega"]["literal_CP95"][1])
mac("bellPconv", b["paper_reported"]["p_conventional"])
mac("drRoHalf", best(P["taskA"]["readout_error"], 14)["skill"] / best(P["taskA"]["readout_error"], 1)["skill"], "{:.2f}")

(GEN / "macros_ext.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in M.items()) + "\n", encoding="utf-8")

# atlas rows
A = J("atlas")
A["rows"] = [r for r in A["rows"] if r["name"] not in ("qdrift_bad", "live_ibm")]
tb = P["taskB"]
A["rows"].append({"name": "qdrift_bad", "label": "Bad qubit at next calibration (IBM)", "short": "bad qubit",
                  "shares": [tb["qubit_rate"]["LSS"], max(tb["hgb"]["LSS"] - tb["qubit_rate"]["LSS"], 0), 0.0, 1 - tb["hgb"]["LSS"]],
                  "LSS": tb["hgb"]["LSS"], "LSS_ci": tb["hgb"]["ci"]["LSS"], "acc": tb["hgb"]["acc"], "acc0": tb["hgb"]["acc0"],
                  "S_acc": tb["hgb"]["S_acc"], "kU": None, "n": tb["n_rows"], "tau": None})
A["rows"].append({"name": "live_ibm", "label": "Qubit (live ibm\\_kingston)", "short": "live IBM", "shares": [0, float(np.mean(kd)), 0, 1 - float(np.mean(kd))],
                  "LSS": float(np.mean(kd)), "acc": float(np.mean([L[k]["A_dev"] for k in qs])), "acc0": 0.5,
                  "S_acc": (float(np.mean([L[k]["A_dev"] for k in qs])) - 0.5) / 0.5, "kU": None, "control": True})
A["rows"].sort(key=lambda r: -r["LSS"])
(RES / "atlas.json").write_text(json.dumps(A, indent=1, default=float))
print(len(M), "macros"); print({k: M[k] for k in list(M)[:12]}); print("live kappa", M["liveKappa"], "atlas rows", len(A["rows"]))
