"""Revised figures after referee review (overrides selected functions of figs.py)."""
import sys, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import figs as F
from figs import J, save, pane3d, C, COL1, COL2, RES
from kn import core


def fig_chaos():
    d = J("controls")
    Z = np.load(RES / "fig_lorenz_local.npz"); X = Z["X"]
    fig = plt.figure(figsize=(COL2, 2.55))
    for i, h in enumerate(["h1", "h2"]):
        ax = fig.add_axes([0.0 + 0.30 * i, 0.13, 0.32, 0.8], projection="3d")
        o = np.argsort(-Z[h])
        sc = ax.scatter(X[o, 0], X[o, 1], X[o, 2], c=Z[h][o], s=0.35, cmap="inferno", vmin=0, vmax=1, linewidths=0, rasterized=True)
        ax.view_init(16, -60); pane3d(ax)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel("$x$", labelpad=-12); ax.set_ylabel("$y$", labelpad=-12); ax.set_zlabel("$z$", labelpad=-12)
        ax.set_title(f"({'ab'[i]}) Bayes certainty, $h={h[1:]}$", pad=-6, y=0.98)
    cax = fig.add_axes([0.12, 0.08, 0.38, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"certainty $2|\eta-1/2|$ given a noisy observation ($\sigma_o=0.1$)", labelpad=1)
    ax = fig.add_axes([0.68, 0.2, 0.31, 0.7])
    g = {(r["sigma"], r["h"]): r for r in d["D2"]["grid"]}
    hs = sorted({r["h"] for r in d["D2"]["grid"]})
    for s, col in [(1e-3, "#1b3a6b"), (1e-1, "#2e86c1"), (1.0, "#85c1e9")]:
        ax.plot(hs, [g[(s, h)]["A_ensemble"] for h in hs], "-", color=col, label=fr"ensemble, $\sigma_o={s:g}$")
        ax.plot(hs, [g[(s, h)]["E1_acc"] for h in hs], "--", color=col)
    ax.plot([], [], "k--", label="learner ($n=4000$)")
    ax.set_xscale("log", base=2); ax.set_xlabel("horizon $h$"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.48, 1.01); ax.legend(fontsize=5.6, loc="lower left"); ax.set_title("(c) physical vs learned")
    save(fig, "fig_chaos")


def fig_law():
    d = J("law")
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax, ax2, ax3 = axs
    cmap = matplotlib.colormaps["plasma"]
    keys = sorted(d["nn_fits"], key=float); hs = [float(k) for k in keys]
    for i, k in enumerate(keys):
        f = d["nn_fits"][k]; col = cmap(i / (len(keys) - 0.5))
        n = np.array(f["n"]); g = np.array(f["gaps"])
        ax.loglog(n, g, "o", ms=2.4, color=col)
        nn = np.logspace(3, 6.1, 30); ax.loglog(nn, f["c"] * nn ** (-f["exponent"]), "-", lw=0.8, color=col, label=f"$h={float(k):g}$")
    ax.axhline(0.05, color="k", ls=":", lw=0.6)
    ax.set_xlabel("training samples $n$"); ax.set_ylabel(r"gap $A^*_{\mathrm{phys}}-A_{\mathrm{1NN}}$")
    ax.legend(fontsize=5.2, ncol=2, loc="lower left"); ax.set_title(r"(a) gap $\approx c\,n^{-\beta}$")
    law = d["nn_law_all"]["0.05"]; P = np.array(law["points"])
    ax2.semilogy(P[:, 0], P[:, 1], "o", color="#1b3a6b", ms=3.2, label=r"$n^*(h)$, $\varepsilon=0.05$")
    hh = np.linspace(0.3, 4.3, 50)
    ax2.semilogy(hh, np.exp(law["a"] + law["gamma"] * hh), "-", color="#1b3a6b", lw=0.9, label=fr"fit $\gamma={law['gamma']:.2f}$")
    lamD = J("controls")["D2"]["lyapunov"] * 2.06
    a0 = np.mean(np.log(P[:, 1]) - lamD * P[:, 0])
    ax2.semilogy(hh, np.exp(a0 + lamD * hh), "--", color="#c0392b", lw=0.9, label=fr"heuristic $\lambda D_2={lamD:.2f}$")
    ax2.set_xlabel("horizon $h$"); ax2.set_ylabel("samples needed"); ax2.legend(fontsize=5.8); ax2.set_title("(b) exponential in horizon")
    G = np.array([d["nn_fits"][k]["gaps"] for k in keys]); ln = np.log10(d["nn_fits"][keys[0]]["n"])
    im = ax3.contourf(ln, hs, np.log10(np.clip(G, 1e-3, None)), levels=14, cmap="plasma_r")
    ax3.contour(ln, hs, np.log10(np.clip(G, 1e-3, None)), levels=[math.log10(0.05)], colors="w", linewidths=1.3)
    ax3.plot(np.log10(P[:, 1]), P[:, 0], "o", ms=2.2, color="w")
    cb = fig.colorbar(im, ax=ax3, fraction=0.05, pad=0.02); cb.set_label(r"$\log_{10}$ gap")
    ax3.set_xlabel(r"$\log_{10} n$"); ax3.set_ylabel("horizon $h$"); ax3.set_title("(c) gap landscape (white: 0.05)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_law")


def fig_law3d():
    d = J("law")
    keys = sorted(d["nn_fits"], key=float); hs = [float(k) for k in keys]
    n = np.array(d["nn_fits"][keys[0]]["n"]); G = np.array([d["nn_fits"][k]["gaps"] for k in keys])
    Ln, H = np.meshgrid(np.log10(n), hs)
    fig = plt.figure(figsize=(COL1, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    Zz = np.log10(np.clip(G, 1e-3, None))
    ax.plot_surface(Ln, H, Zz, cmap="plasma_r", linewidth=0.25, edgecolor="k", alpha=0.95)
    P = np.array(d["nn_law_all"]["0.05"]["points"])
    ax.plot(np.log10(P[:, 1]), P[:, 0], np.full(len(P), math.log10(0.05)), "o-", color="#00e5ff", lw=1.4, ms=2.5, zorder=10)
    ax.contour(Ln, H, Zz, levels=8, zdir="z", offset=-3.2, cmap="plasma_r", linewidths=0.6)
    ax.set_zlim(-3.2, 0); ax.view_init(27, -125); pane3d(ax)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$\log_{10} n$", labelpad=-3, fontsize=8); ax.set_ylabel("horizon $h$", labelpad=-3, fontsize=8)
    ax.set_zlabel(r"$\log_{10}$ gap", labelpad=-3, fontsize=8)
    save(fig, "fig_law3d")


def fig_beta():
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    m = np.linspace(0.5, 0.95, 60); ls = np.linspace(np.log10(0.6), np.log10(200), 60)
    Zz = np.array([[(lambda c: c["A"] - c["A0"])(core.beta_ceiling(mm * 10 ** l, (1 - mm) * 10 ** l)) for l in ls] for mm in m])
    im = ax.contourf(m, ls, Zz.T, levels=16, cmap="viridis")
    cs = ax.contour(m, ls, Zz.T, levels=[0.01, 0.05, 0.1, 0.2], colors="w", linewidths=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(r"$A_1^*-A_0$")
    for name, lab in [("d4_weather", "rain"), ("d5_football", "football"), ("d6_house", "House"), ("d7_earnings_entity", "earnings"),
                      ("d8_stocks", "stocks"), ("d9_quakes", "quakes"), ("d6_president", "president")]:
        dd = J(name)
        if dd:
            a, b = dd["ceilings"]["alpha"], dd["ceilings"]["beta"]
            mm, l = max(a, b) / (a + b), math.log10(a + b)
            if 0.5 <= mm <= 0.95 and ls[0] <= l <= ls[-1]:
                ax.plot(mm, l, "o", color="#e74c3c", ms=3.5, mec="w", mew=0.4)
                ax.annotate(lab, (mm, l), xytext=(3, 3), textcoords="offset points", fontsize=6, color="w")
    ax.set_xlabel(r"mean entity rate $\max(m,1-m)$"); ax.set_ylabel(r"$\log_{10}(\alpha+\beta)$")
    save(fig, "fig_beta")


def fig_qvc():
    names = [("d1_qubit", "Qubit"), ("d2_lorenz", "Lorenz"), ("d4_weather", "Rain"), ("d5_football", "Football"),
             ("d6_house", "House"), ("d7_earnings", "Earnings"), ("d8_stocks", "Stocks"), ("d9_quakes", "Quakes")]
    rows = [(lab, J(f"quantum_{n}")) for n, lab in names if J(f"quantum_{n}")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.35), gridspec_kw={"width_ratios": [1, 1.35]})
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d) in zip(y, rows):
        lo, hi = d["ll_diff_ci"]; dv = d.get("prereg_classical_set", {}).get("ll_diff", d["ll_diff_q_minus_c"])
        ax.plot([lo, hi], [yi, yi], color=C["q"], lw=1.4); ax.plot(dv, yi, "o", color=C["q"], ms=3.5)
    ax.axvline(0, color="k", lw=0.6); ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(-0.03, 0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"$\Delta$LL (best quantum $-$ best classical)"); ax.set_title("(a) quantum minus classical (right: quantum worse)")
    models = ["LR_d", "HGB_d", "CQ1_fourier", "CQ2_rbf_svm", "Q1_vqc_L2", "Q1_vqc_L4", "Q2_fidelity_kernel", "Q3_projected_kernel"]
    labs = ["LR", "GBM", "Fourier", "RBF-SVM", "VQC-2", "VQC-4", "Q-kernel", "PQ-kernel"]
    Mx = np.array([[d["results"].get(m, {}).get("LSS", np.nan) for m in models] for _, d in rows])
    rel = (Mx - np.nanmax(Mx, 1, keepdims=True)) / np.maximum(np.nanmax(np.abs(Mx), 1, keepdims=True), 1e-3)
    im = ax2.imshow(np.clip(rel, -1, 0), cmap="Reds_r", vmin=-1, vmax=0, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            ax2.text(j, i, f"{Mx[i, j]:.3f}", ha="center", va="center", fontsize=5, color="k" if rel[i, j] > -0.5 else "w")
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(labs, rotation=40, ha="right")
    ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([r[0] for r in rows]); ax2.axvline(3.5, color="k", lw=0.9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02); cb.set_label("LSS relative to row best")
    ax2.set_title("(b) LSS per model: classical | quantum")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_qvc")


def fig_ibm():
    d = J("d3_ibm_noise")
    fig, ax = plt.subplots(figsize=(COL1, 1.8))
    b = d["backends"]; x = np.arange(len(b)); w = [r["chsh"]["win"] for r in b]
    ax.bar(x, w, color=[C["q"] if v > 0.75 else "#c0392b" for v in w], width=0.6)
    ax.axhline(0.75, color="k", ls="--", lw=0.7); ax.text(-0.45, 0.737, "classical bound 0.75", fontsize=6)
    ax.axhline(math.cos(math.pi / 8) ** 2, color=C["phys"], ls=":", lw=0.7)
    ax.text(-0.45, 0.858, r"Tsirelson $\cos^2(\pi/8)$", fontsize=6, color=C["phys"])
    ax.set_xticks(x); ax.set_xticklabels([r["backend"].replace("Fake", "") for r in b], rotation=40, ha="right")
    ax.set_ylim(0.6, 0.88); ax.set_ylabel("CHSH win rate")
    save(fig, "fig_ibm")


if __name__ == "__main__":
    for f in (fig_chaos, fig_law, fig_law3d, fig_beta, fig_qvc, fig_ibm):
        try:
            f()
        except Exception as e:
            print("FAILED", f.__name__, type(e).__name__, e)
