"""Figures for the extensions: live hardware + Bell (fig_hardware), drift forecasting (fig_drift),
multi-system scaling (fig_scaling), and dark 3D calibration landscapes (comp_drift3d_*.png)."""
import json, math, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import figs as F
from figs import save, C, COL1, COL2

RES = ROOT / "results"
DATA = ROOT.parent / "data"
J = lambda n: json.loads((RES / f"{n}.json").read_text())
TS = math.cos(math.pi / 8) ** 2


def fig_hardware():
    live = J("ibm_real_ibm_kingston_20261107"); bell = J("bell_delft"); ctl = J("controls"); ib = J("d3_ibm_noise")
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.35), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax = axs[0]
    th = np.linspace(0, math.pi, 200)
    ax.plot(th, np.sin(th / 2) ** 2, "k-", lw=0.8, label=r"ideal $\sin^2(\theta/2)$")
    for q, col, mk in [(5, "#1b3a6b", "o"), (42, "#c0392b", "s")]:
        r = [x for x in live["runs"] if x["kind"] == "ry" and x["qubit"] == q]
        ax.plot([x["theta"] for x in r], [x["counts"].get("1", 0) / live["shots"] for x in r], mk, ms=2.8, color=col,
                label=f"qubit {q}: $A^*_{{\\rm dev}}={live[f'qubit_{q}']['A_dev']:.4f}$")
    ax.set_xlabel(r"rotation $\theta$"); ax.set_ylabel(r"$P(\mathrm{read}\ 1)$")
    ax.set_title("(a) live ibm_kingston, 4,000 shots"); ax.legend(fontsize=5.6, loc="upper left")
    ax = axs[1]
    rows = []
    rows.append(("statevector simulation", ctl["D1"]["chsh"]["quantum"], 1.96 * math.sqrt(TS * (1 - TS) / 200000), C["phys"]))
    for b in ib["backends"]:
        rows.append((b["backend"].replace("Fake", "") + " noise model", b["chsh"]["win"], 1.96 * math.sqrt(0.84 * 0.16 / 32768), "#8e7cc3"))
    w = live["chsh_win"]; n_live = 4 * live["shots"]
    rows.append(("live ibm_kingston", w, 1.96 * math.sqrt(w * (1 - w) / n_live), "#e67e22"))
    om = bell["omega"]
    rows.append(("Delft Bell test (300 trials)", om["omega"], None, "#117a65"))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, e, col) in zip(y, rows):
        if lab.startswith("Delft"):
            lo, hi = om["CP95"]; ax.plot([lo, hi], [yi, yi], color=col, lw=1.4)
        elif e:
            ax.plot([v - e, v + e], [yi, yi], color=col, lw=1.4)
        ax.plot(v, yi, "o", color=col, ms=3.5)
    ax.axvline(0.75, color="k", ls="--", lw=0.7); ax.axvline(TS, color=C["phys"], ls=":", lw=0.8)
    tr = ax.get_xaxis_transform()
    ax.text(0.75, 1.01, "classical 0.75", fontsize=5.5, ha="center", va="bottom", transform=tr)
    ax.text(TS, 1.01, r"$\cos^2\frac{\pi}{8}$", fontsize=6, ha="center", va="bottom", color=C["phys"], transform=tr)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlim(0.68, 0.875); ax.set_xlabel("CHSH win rate"); ax.set_title("(b) entanglement as side information")
    ax = axs[2]
    p = bell["prediction_Bob_y"]
    sim = ctl["D1"]["side_info"]
    groups = [("simulated qubits", [("none", sim["none"], None), ("classical", sim["classical"], None), ("quantum", sim["quantum"], None)]),
              ("Delft electrons", [("own setting", p["from_b_only"]["loo_acc"], p["from_b_only"]["loo_CP95"]),
                                   ("+ partner", p["from_a_b_x"]["loo_acc"], p["from_a_b_x"]["loo_CP95"]),
                                   ("+ state", p["from_a_b_x_state"]["loo_acc"], p["from_a_b_x_state"]["loo_CP95"])])]
    x0, ticks, labs = 0, [], []
    cols = ["#b0b7c3", "#5d6d7e", "#e67e22"]
    for gname, items in groups:
        for k, (lab, v, ci) in enumerate(items):
            ax.bar(x0, v, color=cols[k], width=0.8)
            if ci:
                ax.plot([x0, x0], ci, color="k", lw=0.8)
            ticks.append(x0); labs.append(lab); x0 += 1
        x0 += 0.7
    ax.axhline(0.5, color="k", lw=0.4)
    ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6.6)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("accuracy predicting an outcome")
    ax.text(1, 1.03, "simulation", ha="center", fontsize=6); ax.text(4.7, 1.03, "Delft data", ha="center", fontsize=6)
    ax.set_title("(c) side information", pad=9)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_hardware")


def fig_drift():
    P, S = J("qdrift_primary"), J("qdrift_secondary")
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.3))
    ax = axs[0]
    hs = [1, 2, 4, 7, 14]
    names = {"readout_error": ("readout", "#c0392b"), "cz_error": ("two-qubit (CZ)", "#e67e22"), "sx_error": ("one-qubit (SX)", "#8e44ad"),
             "T1": ("$T_1$", "#1b3a6b"), "T2": ("$T_2$", "#5dade2")}
    for m, (lab, col) in names.items():
        A = P["taskA"].get(m)
        if not A:
            continue
        best = [max(A[f"h{h}"]["skill_ridge_vs_clim"], A[f"h{h}"]["skill_hgb_vs_clim"], key=lambda d: d["skill"]) for h in hs]
        ax.plot(hs, [b["skill"] for b in best], "o-", color=col, ms=2.6, label=lab)
        ax.fill_between(hs, [b["lo"] for b in best], [b["hi"] for b in best], color=col, alpha=0.15, lw=0)
        if m in S["taskA"]:
            B = S["taskA"][m]
            sb = [max(B[f"h{h}"]["skill_ridge_vs_clim"]["skill"], B[f"h{h}"]["skill_hgb_vs_clim"]["skill"]) for h in hs]
            ax.plot(hs, sb, "--", color=col, lw=0.8)
    pers = [P["taskA"]["readout_error"][f"h{h}"]["skill_pers_vs_clim"]["skill"] for h in hs]
    ax.plot(hs, pers, ":", color="k", lw=0.9, label="persistence (readout)")
    ax.axhline(0, color="k", lw=0.4); ax.set_xscale("log", base=2); ax.set_xticks(hs); ax.set_xticklabels(hs)
    ax.set_ylim(-0.62, 0.42); ax.set_xlabel("forecast horizon (calibrations)"); ax.set_ylabel("MSE skill vs qubit climatology")
    ax.legend(fontsize=5.0, ncol=3, loc="lower center", columnspacing=0.6, handlelength=1.4); ax.set_title("(a) forecasting calibration drift")
    ax = axs[1]
    for D, ls, lab in [(P, "-", "primary"), (S, "--", "replication")]:
        pr = D["taskB"]["persistence"]; ks = sorted(pr, key=int); kk = [int(k) for k in ks]
        ax.plot(kk, [pr[k]["p_bad_given_bad"] for k in ks], "o" + ls, color="#c0392b", ms=2.6, label=f"after a bad day ({lab})")
        if ls == "-":
            ax.fill_between(kk, [pr[k]["ci"][0] for k in ks], [pr[k]["ci"][1] for k in ks], color="#c0392b", alpha=0.15, lw=0)
        ax.plot(kk, [pr[k]["p_bad_given_good"] for k in ks], "s" + ls, color="#1b3a6b", ms=2.4, label=f"after a good day ({lab})")
        ax.plot(kk, [pr[k]["base_rate"] for k in ks], ls, color="#7f8c8d", lw=0.7)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 7, 14]); ax.set_xticklabels([1, 2, 4, 7, 14])
    ax.set_ylim(0, 0.7); ax.set_xlabel("calibrations ahead"); ax.set_ylabel("P(qubit bad)")
    ax.legend(fontsize=5.2, loc="center right"); ax.set_title("(b) badness persists (grey: base rate)")
    ax = axs[2]
    mods = [("qubit_rate", "qubit history"), ("logit", "logistic"), ("hgb", "boosting")]
    for j, (D, off, lab, col) in enumerate([(P, -0.18, "primary", "#2e4057"), (S, 0.18, "replication", "#048ba8")]):
        tb = D["taskB"]
        vals = [tb[m]["LSS"] for m, _ in mods]; cis = [tb[m]["ci"]["LSS"] for m, _ in mods]
        xs = np.arange(len(mods)) + off
        ax.bar(xs, vals, width=0.34, color=col, label=lab)
        for x, c in zip(xs, cis):
            ax.plot([x, x], c, color="k", lw=0.8)
        ax.axhline(tb["ceiling"]["kappa"], color=col, ls=":", lw=0.9)
    ax.text(-0.45, 0.41, "entity-only ceilings (dotted)", fontsize=5.5, ha="left")
    ax.set_xticks(range(len(mods))); ax.set_xticklabels([m[1] for m in mods], fontsize=6.5)
    ax.set_ylabel("log-loss skill (LSS)"); ax.legend(fontsize=5.6, loc="upper left"); ax.set_ylim(0, 0.5)
    ax.set_title("(c) will a qubit be bad next?")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_drift")


def fig_scaling():
    syst = [("lorenz63", "Lorenz-63", "#1b3a6b", "o"), ("rossler", "Rössler", "#117a65", "s"),
            ("lorenz96_5", "Lorenz-96, N=5", "#e67e22", "^"), ("lorenz96_6", "Lorenz-96, N=6", "#c0392b", "v"),
            ("lorenz96_8", "Lorenz-96, N=8", "#8e44ad", "D")]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.3))
    for key, lab, col, mk in syst:
        d = J(f"law2_{key}"); l = d["law"]
        if not l:
            continue
        P = np.array([[p[0], p[1]] for p in l["points"]]); interp = [p[2] == "interpolated" for p in l["points"]]
        for (L, ln), ok in zip(P, interp):
            axs[0].plot(L, ln / math.log(10), mk, color=col, ms=3.2, mfc=col if ok else "white")
        LL = np.linspace(P[:, 0].min(), P[:, 0].max(), 20)
        ext = not l["only_interpolated"]
        axs[0].plot(LL, (l["a"] + l["gamma_L"] * LL) / math.log(10), ":" if ext else "-", color=col, lw=0.9,
                    label=lab + (" (extrapolated)" if ext else ""))
        ext = not l["only_interpolated"]
        axs[1].errorbar(d["D2"], l["gamma_L"], yerr=[[l["gamma_L"] - l["gamma_L_lo"]], [l["gamma_L_hi"] - l["gamma_L"]]],
                        fmt=mk, color=col, ms=3.6, capsize=2, lw=0.9, alpha=0.45 if ext else 1.0)
        axs[1].plot(d["D2"], 1 / d["beta_mean"], mk, color=col, ms=3.6, mfc="white")
        axs[2].plot(d["D2"], d["beta_mean"], mk, color=col, ms=4)
        axs[2].annotate(lab.replace("Lorenz-96, ", "L96 "), (d["D2"], d["beta_mean"]),
                        xytext=(4, -10) if key == "rossler" else (3, 2), textcoords="offset points", fontsize=5.5)
    axs[0].set_ylim(2.5, 9.5)
    axs[0].set_xlabel(r"horizon in Lyapunov times $L=\lambda h$"); axs[0].set_ylabel(r"$\log_{10} n^*$ (gap $\leq 0.05$)")
    axs[0].legend(fontsize=5.0, loc="upper right"); axs[0].set_title("(a) exponential in every system")
    dd = np.linspace(1.6, 5.1, 20)
    axs[1].plot(dd, dd, "k--", lw=0.8, label=r"heuristic $\gamma_L=D_2$")
    axs[1].plot([], [], "ko", ms=3.5, label=r"measured $\gamma_L$ (95% CI)"); axs[1].plot([], [], "ko", ms=3.5, mfc="white", label=r"$1/\hat\beta$")
    axs[1].set_xlabel(r"correlation dimension $D_2$"); axs[1].set_ylabel(r"growth rate $\gamma_L$")
    axs[1].legend(fontsize=5.6, loc="upper left"); axs[1].set_title("(b) rate vs attractor dimension")
    axs[2].plot(dd, 1 / dd, "k--", lw=0.8, label=r"$1/D_2$")
    axs[2].set_xlabel(r"correlation dimension $D_2$"); axs[2].set_ylabel(r"1-NN exponent $\hat\beta$")
    axs[2].legend(fontsize=6); axs[2].set_title(r"(c) learning slows faster than $1/D_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_scaling")


def comp_drift3d():
    import plotly.graph_objects as go
    from PIL import Image, ImageChops
    cols = ["backend", "property_family", "qubit_a", "value", "is_failure_ceiling", "calibrated_time", "is_new_measurement"]
    d = pd.read_parquet(DATA / "qdrift_hf_subset.parquet", columns=cols)
    d = d[(d.backend == "ibm_kingston") & (~d.is_failure_ceiling.fillna(False))]
    bg = "#141b24"; muted = "#9aa5b4"; line = "#263241"
    out = {}
    for prop, fname, tf, title, cs in [("readout_error", "comp_drift3d_readout", lambda v: np.log10(v), "(a) ibm_kingston readout error", "Plasma"),
                                       ("T1", "comp_drift3d_t1", lambda v: np.log10(v), "(b) ibm_kingston T1", "Viridis")]:
        x = d[d.property_family == prop].copy()
        if prop == "readout_error":
            x = x[(x.value > 0) & (x.value < 0.35)]
        else:  # the source switched units in May 2026 (seconds -> microseconds); harmonise to microseconds for display
            x = x.assign(value=np.where(x.value < 0.01, x.value * 1e6, x.value))
            x = x[(x.value > 2.0)]
        x["day"] = x.calibrated_time.dt.tz_convert(None).dt.normalize()
        g = x.sort_values("calibrated_time").groupby(["qubit_a", "day"]).value.last().unstack()
        g = g.loc[:, g.notna().mean() > 0.5]
        g = g.ffill(axis=1, limit=2)
        order = g.median(axis=1).sort_values().index
        g = g.loc[order]
        Z = tf(g.to_numpy(float))
        days = np.arange(Z.shape[1]); qs = np.arange(Z.shape[0])
        out[prop] = dict(n_qubits=int(Z.shape[0]), n_days=int(Z.shape[1]), first=str(g.columns[0].date()), last=str(g.columns[-1].date()))
        fig = go.Figure(go.Surface(x=days, y=qs, z=Z, colorscale=cs, showscale=True, connectgaps=False,
                                   colorbar=dict(title=dict(text="log10 " + ("error" if prop == "readout_error" else "T1 (µs)"), font=dict(color=muted, size=20)),
                                                 tickfont=dict(color=muted, size=16), len=0.6, thickness=18)))
        ax3 = lambda: dict(title=dict(text=""), color=muted, gridcolor=line, zerolinecolor=line, backgroundcolor=bg, showbackground=True, tickfont=dict(size=16))
        fig.update_layout(paper_bgcolor=bg, font=dict(color="#e6e9ee", size=20), margin=dict(l=0, r=0, t=50, b=0),
                          title=dict(text=title, x=0.03, y=0.97, font=dict(size=26, color="#e6e9ee")),
                          scene=dict(xaxis=ax3(), yaxis=ax3(), zaxis=ax3(), aspectratio=dict(x=1.6, y=1.0, z=0.55),
                                     camera=dict(eye=dict(x=-1.25, y=-1.55, z=0.85))))
        path = F.OUT / f"{fname}.png"
        fig.write_image(path, width=1100, height=760, scale=2.2)
        im = Image.open(path).convert("RGB"); box = ImageChops.difference(im, Image.new("RGB", im.size, (0x14, 0x1b, 0x24))).getbbox()
        if box:
            im = im.crop((max(box[0] - 25, 0), max(box[1] - 25, 0), min(box[2] + 25, im.width), min(box[3] + 25, im.height)))
        im.thumbnail((1600, 1600)); im.save(path, optimize=True)
        print("saved", fname, out[prop])
    (RES / "drift3d_meta.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for f in (fig_hardware, fig_drift, fig_scaling, comp_drift3d):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAILED", f.__name__, type(e).__name__, e)
