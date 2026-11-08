"""Revision-2 figures: fig_law3 (chaos learnability with floors, 1-NN vs k-NN, seeds, crossing status),
fig_atlas_rev (achieved skill, references distinct, no ceiling whiskers), fig_drift_heat (2D calibration heatmaps),
fig_qvc_rev (validation-selected quantum comparison). Output: ../paper_rev/figs/."""
import json, math, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import figs as F
RES = ROOT / "results"; DATA = ROOT.parent / "data"
OUT = ROOT.parent / "paper_rev" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
F.OUT = OUT
from figs import save, COL1, COL2
J = lambda n: json.loads((RES / f"{n}.json").read_text())
SYS = [("lorenz63", "Lorenz-63", "#1b3a6b"), ("rossler", "Rössler", "#117a65"), ("lorenz96_5", "Lorenz-96, N=5", "#e67e22"),
       ("lorenz96_6", "Lorenz-96, N=6", "#c0392b"), ("lorenz96_8", "Lorenz-96, N=8", "#8e44ad")]
MK = {"interpolated": "o", "extrapolated": "^", "unattainable": "x", "not_identifiable": "s", "below_first_n": "v"}


def fig_law3():
    avail = [(k, l, c) for k, l, c in SYS if (RES / f"law3_{k}.json").exists()]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.75))
    # (a) Lorenz-63 gap curves: 1-NN vs k-NN, all seeds, L = 1 and 3, with the 1-NN floor
    d = J("law3_lorenz63"); NS = d["settings"]["NS"]
    for Lh, ls in (("1.0", "-"), ("3.0", "--")):
        for s in d["seeds"]:
            pl = s["per_L"][Lh]
            axs[0].loglog(NS, np.clip(pl["gap_1nn"], 1e-4, None), ls, color="#1b3a6b", lw=0.8, alpha=0.8)
            axs[0].loglog(NS, np.clip(pl["gap_knn"], 1e-4, None), ls, color="#e67e22", lw=0.8, alpha=0.8)
            axs[0].axhline(max(pl["floor_1nn"], 1e-4), color="#1b3a6b", ls=":", lw=0.6)
    axs[0].axhline(0.05, color="k", lw=0.5)
    axs[0].plot([], [], color="#1b3a6b", label="1-NN"); axs[0].plot([], [], color="#e67e22", label=r"$k$-NN, $k=\lceil\sqrt{n}\rceil$")
    axs[0].plot([], [], "k-", lw=0.8, label="$L=1$"); axs[0].plot([], [], "k--", lw=0.8, label="$L=3$"); axs[0].plot([], [], "k:", lw=0.6, label="1-NN floor")
    axs[0].set_xlabel("training samples $n$"); axs[0].set_ylabel("gap to ensemble rule (accuracy)")
    axs[0].set_ylim(6e-5, 1.0); axs[0].legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False); axs[0].set_title("(a) Lorenz-63, 3 seeds")
    # (b) n*(L) per system and learner, marker = crossing status
    for si, (k, lab, col) in enumerate(avail):
        d = J(f"law3_{k}")
        for lname, off, fill in (("1nn", -0.04, True), ("knn", 0.04, False)):
            for s in d["seeds"]:
                for Lh, pl in s["per_L"].items():
                    c = pl[f"cross_{lname}"]["0.05"]
                    if c["n_star"]:
                        axs[1].plot(float(Lh) + off, min(math.log10(c["n_star"]), 6.9), MK[c["status"]], color=col, ms=2.8,
                                    mfc=col if fill else "white", mew=0.6, alpha=0.9)
                    elif c["status"] in ("unattainable", "not_identifiable"):
                        axs[1].plot(float(Lh) + off * 2.5, 7.25 + 0.13 * si, MK[c["status"]], color=col, ms=3, mew=0.7, mfc="white" if c["status"] == "not_identifiable" else col)
    axs[1].axhline(6.0, color="grey", lw=0.5, ls=":"); axs[1].axhline(7.12, color="grey", lw=0.4)
    axs[1].set_yticks([3, 4, 5, 6, 7.5]); axs[1].set_yticklabels(["3", "4", "5", "6", "none"])
    for k, lab, col in avail:
        axs[1].plot([], [], "o", color=col, ms=3, label=lab)
    axs[1].plot([], [], "ko", ms=3, label="1-NN (filled)"); axs[1].plot([], [], "ko", ms=3, mfc="white", label="$k$-NN (open)")
    axs[1].plot([], [], "k^", ms=3, label="extrapolated")
    axs[1].set_xlabel(r"horizon in Lyapunov times $L$"); axs[1].set_ylabel(r"$\log_{10} n^*$ (gap $\leq 0.05$)")
    axs[1].plot([], [], "ks", ms=3, mfc="white", label="not identifiable")
    axs[1].set_ylim(2.5, 7.95); axs[1].legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False); axs[1].set_title("(b) threshold crossings")
    # (c) growth rate per seed vs D2 (interpolated crossings only)
    dd = np.linspace(1.6, 5.2, 10); axs[2].plot(dd, dd, "k--", lw=0.7, label=r"$\gamma_L=D_2$")
    for k, lab, col in avail:
        d = J(f"law3_{k}"); D2 = d["D2_sensitivity"]["mean"]; d2lo, d2hi = d["D2_sensitivity"]["min"], d["D2_sensitivity"]["max"]
        for lname, mk in (("1nn", "o"), ("knn", "s")):
            ps = d["summary"][lname]["0.05"]["per_seed"]
            g = [p["gamma_L"] for p in ps if p["gamma_L"] is not None]
            for gi in g:
                axs[2].plot(D2, gi, mk, color=col, ms=3, mfc=col if lname == "1nn" else "white", mew=0.6)
            if g:
                axs[2].plot([d2lo, d2hi], [np.mean(g)] * 2, "-", color=col, lw=0.8)
    axs[2].plot([], [], "ko", ms=3, label="1-NN, per seed"); axs[2].plot([], [], "ks", ms=3, mfc="white", label="$k$-NN, per seed")
    axs[2].set_xlabel(r"correlation dimension $D_2$ (range over estimators)"); axs[2].set_ylabel(r"$\gamma_L$ (interpolated crossings)")
    axs[2].legend(fontsize=5.4, loc="upper left"); axs[2].set_title("(c) rate depends on the learner")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_law3")


def fig_atlas_rev():
    A = J("atlas")["rows"]; tb = J("rev2_analyses")["time_block"]
    rows = sorted(A, key=lambda r: -r["LSS"])
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 3.0))
    y = np.arange(len(rows))[::-1]
    cols = ["#2e4057", "#048ba8", "#f18f01"]
    labs = ["entity history", "features", "external benchmark"]
    for yi, r in zip(y, rows):
        if r.get("control"):
            ax.plot(r["LSS"], yi, "D", color="#7f8c8d", ms=4)
            continue
        left = 0
        for k in range(3):
            v = r["shares"][k]
            ax.barh(yi, v, left=left, color=cols[k], height=0.6, edgecolor="white", linewidth=0.3); left += v
        ci = tb.get(r["name"], {}).get("LSS_ci_time_block") or r.get("LSS_ci")
        if ci:
            ax.plot(ci, [yi, yi], color="k", lw=0.8)
    for k in range(3):
        ax.barh([], [], color=cols[k], label=labs[k])
    ax.plot([], [], "D", color="#7f8c8d", ms=4, label="control: exact / reference value")
    ax.plot([], [], "k-", lw=0.8, label="95% interval (time-block where available)")
    ax.set_yticks(y); ax.set_yticklabels([r["label"].replace("\\_", "_") for r in rows], fontsize=6.3)
    ax.set_xlabel("achieved log-loss skill relative to the point-in-time base rate (LSS)")
    ax.axvline(0, color="k", lw=0.4); ax.set_xlim(-0.02, 1.04)
    ax.legend(fontsize=5.6, loc="lower right")
    save(fig, "fig_atlas_rev")


def fig_drift_heat():
    cols = ["backend", "property_family", "qubit_a", "value", "is_failure_ceiling", "calibrated_time"]
    d = pd.read_parquet(DATA / "qdrift_hf_subset.parquet", columns=cols)
    d = d[(d.backend == "ibm_kingston") & (~d.is_failure_ceiling.fillna(False))]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.1))
    for ax, prop, title, cmap in ((axs[0], "readout_error", r"(a) $\log_{10}$ readout error", "plasma"), (axs[1], "T1", r"(b) $\log_{10} T_1$ ($\mu$s)", "viridis")):
        x = d[d.property_family == prop].copy()
        if prop == "readout_error":
            x = x[(x.value > 0) & (x.value < 0.35)]
        else:
            x = x.assign(value=np.where(x.value < 0.01, x.value * 1e6, x.value)); x = x[x.value > 2.0]
        x["day"] = x.calibrated_time.dt.tz_convert(None).dt.normalize()
        g = x.sort_values("calibrated_time").groupby(["qubit_a", "day"]).value.last().unstack()
        g = g.loc[g.median(axis=1).sort_values().index]
        im = ax.imshow(np.log10(g.to_numpy(float)), aspect="auto", cmap=cmap, interpolation="nearest",
                       extent=[0, g.shape[1], g.shape[0], 0])
        ticks = np.linspace(0, g.shape[1] - 1, 5).astype(int)
        ax.set_xticks(ticks); ax.set_xticklabels([str(g.columns[t].date())[5:] for t in ticks], fontsize=6)
        ax.set_xlabel("calibration day (2026, month-day)"); ax.set_ylabel("qubit (sorted by median)")
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_drift_heat")


def fig_qvc_rev():
    names = [("d1_qubit", "Qubit"), ("d2_lorenz", "Lorenz"), ("d4_weather", "Rain"), ("d5_football", "Football"),
             ("d6_house", "House"), ("d7_earnings", "Earnings"), ("d8_stocks", "Stocks"), ("d9_quakes", "Quakes")]
    rows = [(lab, J(f"quantum2_{n}")) for n, lab in names if (RES / f"quantum2_{n}.json").exists()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d) in zip(y, rows):
        lo, hi = d["ll_diff_ci"]; ax.plot([lo, hi], [yi, yi], color="#6c3483", lw=1.3); ax.plot(d["ll_diff_q_minus_c"], yi, "o", color="#6c3483", ms=3.2)
        ax.text(0.62, yi, f"{d['best_quantum'].split('_')[0]} vs {d['best_classical_same_input']}", fontsize=4.8, va="center")
    ax.axvline(0, color="k", lw=0.6); ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(-0.03, 0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel(r"$\Delta$LL, validation-selected quantum $-$ classical (test split)")
    save(fig, "fig_qvc_rev")


def fig_illusory_rev():
    """Illusory edge: (a) Proposition surfaces + simulation, (b) simulated panels, (c) earnings forest plot in which every
    estimate is labelled by its population (removed rows, fixed cohort, paired change, audit population)."""
    rows = J("noise_sim")
    fig = plt.figure(figsize=(COL2, 2.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    P, M = np.meshgrid(np.linspace(0, 0.3, 40), np.linspace(0.52, 0.95, 40), indexing="ij")
    valid = (1 - P) * M > 0.5
    meas = np.where(valid, P, np.nan); true = np.where(valid, -P * (2 * M - 1), np.nan)
    ax.plot_surface(P, M, meas, color="#f0a35e", alpha=0.8, linewidth=0, shade=False)
    ax.plot_surface(P, M, true, color="#5dade2", alpha=0.85, linewidth=0, shade=False)
    ax.plot_surface(P, M, 0 * P, color="k", alpha=0.07, linewidth=0)
    sim = [r for r in rows if r["mode"] == "persistent" and r["r"] == 1.0 and r["T"] == 32 and r["conc"] > 1e8 and (1 - r["pi"]) * r["m"] > 0.5]
    ax.scatter([r["pi"] for r in sim], [r["m"] for r in sim], [r["meas"] for r in sim], color="#a04000", s=6, depthshade=False)
    ax.scatter([r["pi"] for r in sim], [r["m"] for r in sim], [r["true"] for r in sim], color="#1b3a6b", s=6, depthshade=False)
    ax.set_xlabel(r"corrupted share $\omega$", labelpad=-4); ax.set_ylabel(r"true rate $m$", labelpad=-4)
    ax.set_zlabel("edge over base rate", labelpad=-4); ax.view_init(20, -120); F.pane3d(ax)
    ax.set_title("(a) measured (orange) vs true (blue)", pad=-4)
    ax2 = fig.add_subplot(1, 3, 2)
    for mode, col, lab in [("persistent", "#e67e22", "persistent noise"), ("independent", "#2e86c1", "independent noise")]:
        rr = [r for r in rows if r["mode"] == mode and r["pi"] > 0]
        ax2.scatter([r["true"] for r in rr], [r["meas"] for r in rr], s=4, color=col, alpha=0.6, label=lab, linewidths=0)
    ax2.plot([-0.15, 0.22], [-0.15, 0.22], "k-", lw=0.5); ax2.axhline(0, color="k", lw=0.3); ax2.axvline(0, color="k", lw=0.3)
    ax2.fill_between([-0.15, 0], 0, 0.22, color="#e74c3c", alpha=0.07)
    ax2.text(-0.145, 0.19, "measured > 0 > true", fontsize=6.5, color="#922b21")
    ax2.set_xlabel("true edge"); ax2.set_ylabel("measured edge"); ax2.legend(loc="lower right"); ax2.set_title("(b) 9,360 simulated panels")
    ax3 = fig.add_subplot(1, 3, 3)
    r = J("rev2_analyses"); ec, au = r["earnings_cohort"], r["audit"]; s1 = au["scenarios"]["unverifiable_as_wrong"]   # primary, conservative: unverifiable row counted as wrong
    acc = au["accounting"]
    items = [
        (f"removed rows (n={ec['removed_rows']})\noriginal labels", 100 * ec["removed_rows_edge_archived_original_labels"], None, "#e67e22"),
        (f"fixed cohort (n={ec['fixed_cohort_rows']})\narchived model", 100 * ec["A_archived_before_sweep_on_cohort"], [100 * x for x in ec["A_ci"]], "#1b3a6b"),
        ("fixed cohort\nretrained on cleaned", 100 * ec["B_after_sweep_retrained_on_cohort"], [100 * x for x in ec["B_ci"]], "#1b3a6b"),
        ("paired change\n(retrained $-$ archived)", 100 * ec["B_minus_A_paired"], [100 * x for x in ec["B_minus_A_ci"]], "#6c3483"),
        (f"audit population (n={acc['n_population_rows']})\nmeasured; 95% lower bound\n(unverifiable counted wrong)", 100 * au["delta_measured_weighted"], None, "#117a65"),
    ]
    ys = np.arange(len(items))[::-1]
    for yi, (lab, v, ci, col) in zip(ys, items):
        if ci:
            ax3.plot(ci, [yi, yi], color=col, lw=1.3)
        ax3.plot(v, yi, "o", color=col, ms=3.5)
    lb = 100 * s1["delta_lower_weighted_hyper"]
    ax3.annotate("", xy=(lb, ys[-1]), xytext=(100 * au["delta_measured_weighted"], ys[-1]),
                 arrowprops=dict(arrowstyle="-[", color="#117a65", lw=1.1))
    ax3.axhline(ys[-1] + 0.5, color="grey", lw=0.4, ls=":")
    ax3.axvline(0, color="k", lw=0.5)
    ax3.set_yticks(ys); ax3.set_yticklabels([it[0] for it in items], fontsize=5.4)
    ax3.set_xlabel("EB edge over base rate (points)"); ax3.set_xlim(-9, 23)
    ax3.set_title("(c) earnings: estimates by population")
    fig.subplots_adjust(left=0.02, right=0.99, wspace=0.62, bottom=0.17, top=0.9)
    save(fig, "fig_illusory")


if __name__ == "__main__":
    for f in (fig_law3, fig_atlas_rev, fig_drift_heat, fig_qvc_rev, fig_illusory_rev):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__, e)
