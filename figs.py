"""All paper figures. Usage: python figs.py [name ...]  (no args = all available). Output: ../paper/figs/*.pdf (+ .png)."""
import json, math, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT.parent / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from kn import core

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.dpi": 400, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
C = {"phys": "#1b3a6b", "learn": "#c0392b", "upper": "#7f8c8d", "q": "#6c3483", "c": "#117a65", "base": "#b9770e"}


def J(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("saved", name)


def pane3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((0.8, 0.8, 0.8, 0.6))
        a._axinfo["grid"]["linewidth"] = 0.3
    ax.tick_params(pad=0)


# ----------------------------------------------------------------------------- Fig: quantum control
def fig_quantum():
    d = J("controls")
    fig = plt.figure(figsize=(COL2, 2.55))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:160j, 0:np.pi:80j]
    x, y, z = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    A = (1 + np.abs(z)) / 2
    ax.plot_surface(x, y, z, facecolors=cm.viridis(colors.Normalize(0.5, 1)(A)), rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=0.95, rasterized=True)
    lam = 0.2
    ax.plot_wireframe((1 - lam) * x, (1 - lam) * y, (1 - lam) * z, rstride=10, cstride=10, color="k", linewidth=0.25, alpha=0.35)
    th = np.linspace(0, np.pi, 100)
    ax.plot(np.sin(th), 0 * th, np.cos(th), color="w", lw=1.2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(r"(a) Bloch sphere: ceiling $(1+|z|)/2$", pad=-2)
    m = cm.ScalarMappable(norm=colors.Normalize(0.5, 1), cmap="viridis")
    cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.0, fraction=0.04); cb.set_label(r"$A^*$ of the state", labelpad=1)
    ax2 = fig.add_subplot(1, 3, 2)
    s = d["D1"]["noisy_theta"]
    sig = [r["sigma"] for r in s]
    ax2.plot(sig, [r["A_star"] for r in s], "-", color=C["phys"], label=r"exact $A^*(\sigma)$")
    ax2.plot(sig, [r["E1_acc"] for r in s], "o", ms=3, color=C["learn"], label="best learner (E1)")
    ax2.plot(sig, [r["E4_A_upper"] for r in s], "^", ms=3, color=C["upper"], label="Cover–Hart bound (E4)")
    ax2.axhline(0.5 + 1 / math.pi, ls=":", color="k", lw=0.6)
    ax2.text(1.15, 0.5 + 1 / math.pi + 0.006, r"$\frac{1}{2}+\frac{1}{\pi}$", fontsize=7)
    ax2.set_xlabel(r"noise on knowledge of $\theta$, $\sigma$"); ax2.set_ylabel("accuracy")
    ax2.legend(loc="lower left"); ax2.set_title("(b) known-ceiling dial")
    ax3 = fig.add_subplot(1, 3, 3)
    ib = J("d3_ibm_noise")
    if ib:
        mk = "osD^v<>p"
        for i, bk in enumerate(ib["backends"]):
            ee = [q["alpha_e0"] + q["e1"] for q in bk["qubits"]]
            ax3.scatter(ee, [q["A_dev"] for q in bk["qubits"]], s=10, marker=mk[i % 8], label=bk["backend"].replace("Fake", ""), alpha=0.85)
        xx = np.linspace(0, 0.45, 50)
        ax3.plot(xx, 0.5 + (1 - xx) / math.pi, "k-", lw=0.8, label=r"$1/2+(1-e_0-e_1)/\pi$")
        ax3.set_xlabel(r"readout error $e_0+e_1$"); ax3.set_ylabel(r"device ceiling $A^*_{\mathrm{dev}}$")
        ax3.legend(fontsize=5.2, ncol=2, loc="lower left"); ax3.set_title("(c) IBM device noise models")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_quantum")


# ----------------------------------------------------------------------------- Fig: chaos
def fig_chaos():
    d = J("controls")
    npz = RES / "fig_lorenz_local.npz"
    fig = plt.figure(figsize=(COL2, 2.6))
    sc = None
    if npz.exists():
        Z = np.load(npz)
        X = Z["X"]
        for i, h in enumerate(["h1", "h2"]):
            ax = fig.add_axes([0.0 + 0.315 * i, 0.12, 0.33, 0.8], projection="3d")
            o = np.argsort(-Z[h])
            sc = ax.scatter(X[o, 0], X[o, 1], X[o, 2], c=Z[h][o], s=0.35, cmap="inferno", vmin=0, vmax=1, linewidths=0, rasterized=True)
            ax.view_init(16, -60); pane3d(ax)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.set_xlabel("$x$", labelpad=-12); ax.set_ylabel("$y$", labelpad=-12); ax.set_zlabel("$z$", labelpad=-12)
            ax.set_title(f"({'ab'[i]}) local Bayes certainty, $h={h[1:]}$", pad=-6, y=0.98)
        cax = fig.add_axes([0.12, 0.08, 0.4, 0.025])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(r"certainty $2|\eta-1/2|$ of the sign of $x_{t+h}$ given a noisy observation ($\sigma_o=0.1$)", labelpad=1)
    ax = fig.add_axes([0.64, 0.1, 0.36, 0.84], projection="3d")
    g = d["D2"]["grid"]
    sig = sorted({r["sigma"] for r in g}); hs = sorted({r["h"] for r in g})
    S, H = np.meshgrid(np.log10(sig), np.log2(hs), indexing="ij")
    look = {(r["sigma"], r["h"]): r for r in g}
    Ap = np.array([[look[(s, h)]["A_ensemble"] for h in hs] for s in sig])
    Al = np.array([[look[(s, h)]["E1_acc"] for h in hs] for s in sig])
    ax.plot_surface(S, H, Ap, cmap="Blues", vmin=0.2, vmax=1.0, alpha=0.8, linewidth=0.25, edgecolor="#1b3a6b")
    ax.plot_wireframe(S, H, Al, color=C["learn"], linewidth=0.8)
    ax.set_xlabel(r"$\log_{10}\sigma_o$", labelpad=-5); ax.set_ylabel(r"$\log_2 h$", labelpad=-5)
    ax.set_zlabel("accuracy", labelpad=-6); ax.zaxis.set_rotate_label(False)
    ax.view_init(22, 40); pane3d(ax); ax.set_zlim(0.5, 1.0)
    ax.set_title("(c) physical ceiling (blue) vs learned (red)", pad=-6, y=0.98)
    save(fig, "fig_chaos")


# ----------------------------------------------------------------------------- Fig: learnability gap
def fig_learnability():
    d = J("learnability")
    if not d:
        return
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.0), sharey=True)
    for ax, h in zip(axs, [1, 2, 4]):
        r = [x for x in d["rows"] if x["h"] == h]
        n = [x["n"] for x in r]
        ax.axhline(r[0]["A_physical"], color=C["phys"], lw=1.2, label="physical ceiling (ensemble)")
        ax.plot(n, [x["A_cover_hart_upper"] for x in r], "^--", ms=3, color=C["upper"], label="Cover–Hart 'upper' bound")
        ax.plot(n, [x["A_mlp"] for x in r], "o-", ms=3, color=C["learn"], label="neural net (GPU)")
        ax.plot(n, [x["A_hgb"] for x in r], "s-", ms=3, color=C["c"], label="gradient boosting")
        ax.set_xscale("log"); ax.set_title(f"horizon $h={h}$"); ax.set_xlabel("training samples $n$")
    axs[0].set_ylabel("accuracy"); axs[0].legend(loc="lower right", fontsize=6)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_learnability")


# ----------------------------------------------------------------------------- Fig: illusory edge
def fig_illusory():
    rows = json.loads((RES / "noise_sim.json").read_text())
    fig = plt.figure(figsize=(COL2, 2.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    # analytic Prop. 7.2 surfaces over (pi, m)
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
    ax.set_zlabel("edge over base rate", labelpad=-4); ax.view_init(20, -120); pane3d(ax)
    ax.set_title("(a) measured (orange) vs true (blue)", pad=-4)
    ax2 = fig.add_subplot(1, 3, 2)
    for mode, col, lab in [("persistent", "#e67e22", "persistent noise"), ("independent", "#2e86c1", "independent noise")]:
        rr = [r for r in rows if r["mode"] == mode and r["pi"] > 0]
        ax2.scatter([r["true"] for r in rr], [r["meas"] for r in rr], s=4, color=col, alpha=0.6, label=lab, linewidths=0)
    lim = [-0.15, 0.22]
    ax2.plot(lim, lim, "k-", lw=0.5); ax2.axhline(0, color="k", lw=0.3); ax2.axvline(0, color="k", lw=0.3)
    ax2.fill_between([-0.15, 0], 0, 0.22, color="#e74c3c", alpha=0.07)
    ax2.text(-0.145, 0.19, "measured > 0 > true", fontsize=6.5, color="#922b21")
    ax2.set_xlabel("true edge"); ax2.set_ylabel("measured edge"); ax2.legend(loc="lower right"); ax2.set_title("(b) 9,360 simulated panels")
    ax3 = fig.add_subplot(1, 3, 3)
    ba = J("d7_analysis")
    if ba and "before_after" in ba:
        b = ba["before_after"]
        rem, kept, aft = b["edge_from_removed_rows"], b["edge_from_kept_rows"], b["after"]["edge_eb"]
        ax3.bar([0], [kept], color="#2e86c1", width=0.55, label="from correctly labelled rows")
        ax3.bar([0], [rem], bottom=[kept], color="#e67e22", width=0.55, label="from rows later found mislabelled")
        ax3.bar([1], [aft], color="#2e86c1", width=0.55)
        ax3.text(0, kept + rem + 0.002, f"{100 * (kept + rem):+.2f} pts", ha="center", fontsize=7)
        ax3.text(0.33, kept + rem / 2, f"{100 * rem / (kept + rem):.0f}%", fontsize=7, color="#a04000", va="center")
        ax3.text(1, aft - 0.005, f"{100 * aft:+.2f} pts", ha="center", fontsize=7)
        if "audit" in ba and ba["audit"].get("delta_lower") is not None:
            ax3.axhline(ba["audit"]["delta_lower"], ls="--", color="k", lw=0.6)
            ax3.text(1.45, ba["audit"]["delta_lower"] + 0.002, "audit 95% lower bound", ha="right", fontsize=6)
        ax3.axhline(0, color="k", lw=0.5)
        ax3.set_xticks([0, 1]); ax3.set_xticklabels(["before\ncleaning", "after\ncleaning"])
        ax3.set_ylim(-0.05, 0.045); ax3.legend(loc="upper right", fontsize=5.5)
        ax3.set_ylabel("EB edge over base rate"); ax3.set_title("(c) earnings, new tickers")
    fig.subplots_adjust(left=0.02, right=0.99, wspace=0.42, bottom=0.17, top=0.9)
    save(fig, "fig_illusory")


# ----------------------------------------------------------------------------- Fig: beta ceiling landscape
def fig_beta():
    fig = plt.figure(figsize=(COL1, 2.5))
    ax = fig.add_subplot(111, projection="3d")
    m = np.linspace(0.5, 0.95, 46); ls = np.linspace(np.log10(0.6), np.log10(200), 46)
    Mm, Ls = np.meshgrid(m, ls, indexing="ij")
    Z = np.zeros_like(Mm)
    for i in range(len(m)):
        for j in range(len(ls)):
            s = 10 ** ls[j]; c = core.beta_ceiling(m[i] * s, (1 - m[i]) * s); Z[i, j] = c["A"] - c["A0"]
    ax.plot_surface(Mm, Ls, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.contour(Mm, Ls, Z, levels=8, zdir="z", offset=0, cmap="viridis", linewidths=0.6)
    pts = []
    for name in ["d4_weather", "d5_football", "d6_house", "d7_earnings_entity", "d8_stocks", "d9_quakes"]:
        dd = J(name)
        if dd and "ceilings" in dd:
            a, b = dd["ceilings"]["alpha"], dd["ceilings"]["beta"]
            mm = max(a, b) / (a + b); s = a + b
            gain = dd["ceilings"]["beta_ceiling"]["A"] - dd["ceilings"]["beta_ceiling"]["A0"]
            pts.append((mm, math.log10(s), gain, name.split("_")[1]))
    for mm, l, g, nm in pts:
        ax.scatter(mm, l, g, color="r", s=8, depthshade=False); ax.text(mm, l, g + 0.01, nm, fontsize=5.5)
    ax.set_xlabel(r"mean rate $\max(m,1-m)$", labelpad=-3); ax.set_ylabel(r"$\log_{10}(\alpha+\beta)$", labelpad=-3)
    ax.set_zlabel(r"$A_1^*-A_0$", labelpad=-3); ax.view_init(26, -132); pane3d(ax)
    save(fig, "fig_beta")


# ----------------------------------------------------------------------------- Fig: atlas
ATLAS = [("d2_lorenz_h025", "Chaos ($h{=}0.25$)"), ("d1_qubit", "Qubit (full $\\theta$)"), ("d3_ibm", "Qubit (IBM noise)"),
         ("d6_president", "President by state"), ("d6_house", "US House seats"), ("d4_weather", "Rain tomorrow"),
         ("d4_weather_nwp", "Rain (+NWP)"), ("d9_quakes", "Earthquakes (week)"), ("d5_football", "Football"),
         ("d7_earnings", "Earnings guidance"), ("d8_stocks", "Stock direction")]


def atlas_rows():
    """Returns list of dicts: name, label, shares (entity, features, benchmark, unexplained), LSS, kU, acc, acc0."""
    rows = []
    a = J("atlas")
    return a["rows"] if a else rows


def fig_atlas():
    rows = atlas_rows()
    if not rows:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(rows))[::-1]
    cols = ["#2e4057", "#048ba8", "#f18f01", "#e8e8e8"]
    labs = ["entity history", "features", "benchmark (odds / NWP)", "not explained"]
    left = np.zeros(len(rows))
    for k in range(4):
        v = np.array([r["shares"][k] for r in rows])
        ax.barh(y, v, left=left, color=cols[k], height=0.62, label=labs[k], edgecolor="white", linewidth=0.3)
        left += v
    for yi, r in zip(y, rows):
        if r.get("kU") is not None:
            ax.plot([r["LSS"], min(r["kU"], 1)], [yi, yi], color="k", lw=0.6)
            ax.plot(min(r["kU"], 1), yi, "|", color="k", ms=5)
        ax.text(1.01, yi, f"{r['LSS']:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([r["label"].replace("\\_", "_") for r in rows]); ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"share of outcome entropy $H(Y)$ (knowable fraction $\kappa$ = coloured part)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6.5)
    for r in rows:
        ax2.scatter(r["acc"], r["S_acc"], s=14, color="#2e4057")
        off = {"IBM": (-18, 3), "president": (4, -9), "quakes": (-34, 3), "rain": (4, -9), "qubit": (-12, 7),
               "live IBM": (5, -6), "bad qubit": (4, -8)}.get(r["short"], (2, 2))
        ax2.annotate(r["short"], (r["acc"], r["S_acc"]), fontsize=6, xytext=off, textcoords="offset points")
    ax2.set_xlabel("raw accuracy"); ax2.set_ylabel(r"skill over base rate $S_{acc}$")
    ax2.set_title("high accuracy $\\neq$ high knowability")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_atlas")


def fig_atlas3d():
    rows = atlas_rows()
    if not rows:
        return
    fig = plt.figure(figsize=(COL1, 2.7))
    ax = fig.add_subplot(111, projection="3d")
    rungs = ["base", "+entity", "+features", "+benchmark"]
    cmap = matplotlib.colormaps["viridis"]
    for i, r in enumerate(rows):
        cum = np.cumsum([0] + list(r["shares"][:3]))
        for k in range(4):
            if k == 3 and r["shares"][2] == 0:
                continue
            ax.bar3d(i, k, 0, 0.6, 0.6, max(cum[k], 1e-4), color=cmap(0.15 + 0.8 * cum[k]), shade=True, alpha=0.95)
    ax.set_xticks(np.arange(len(rows)) + 0.3); ax.set_xticklabels([r["short"] for r in rows], rotation=60, fontsize=5.5, ha="right")
    ax.set_yticks(np.arange(4) + 0.3); ax.set_yticklabels(rungs, fontsize=5.5, rotation=-15)
    ax.set_zlabel(r"$\hat\kappa$ (LSS)", labelpad=-3); ax.view_init(28, -60); pane3d(ax)
    save(fig, "fig_atlas3d")


# ----------------------------------------------------------------------------- Fig: quantum vs classical
def fig_qvc():
    names = [("d1_qubit", "Qubit"), ("d2_lorenz", "Chaos"), ("d4_weather", "Rain"), ("d5_football", "Football"),
             ("d6_house", "House"), ("d7_earnings", "Earnings"), ("d8_stocks", "Stocks"), ("d9_quakes", "Quakes")]
    rows = [(lab, J(f"quantum_{n}")) for n, lab in names if J(f"quantum_{n}")]
    if not rows:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d) in zip(y, rows):
        lo, hi = d["ll_diff_ci"]
        ax.plot([lo, hi], [yi, yi], color=C["q"], lw=1.4); ax.plot(d["ll_diff_q_minus_c"], yi, "o", color=C["q"], ms=3.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"$\Delta$ log loss: best quantum $-$ best classical"); ax.set_title("(a) quantum is never better (right = worse)")
    models = ["LR_d", "HGB_d", "CQ1_fourier", "CQ2_rbf_svm", "Q1_vqc_L2", "Q1_vqc_L4", "Q2_fidelity_kernel", "Q3_projected_kernel"]
    labs = ["LR", "GBM", "Fourier", "RBF-SVM", "VQC-2", "VQC-4", "QK-fid", "QK-proj"]
    M = np.array([[d["results"].get(m, {}).get("LSS", np.nan) for m in models] for _, d in rows])
    Mn = M - np.nanmax(M, axis=1, keepdims=True)
    im = ax2.imshow(Mn, cmap="RdBu", vmin=-0.1, vmax=0.1, aspect="auto")
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(labs, rotation=45, ha="right")
    ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([r[0] for r in rows])
    ax2.axvline(3.5, color="k", lw=0.8)
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02); cb.set_label("LSS minus best in row")
    ax2.set_title("(b) classical | quantum")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_qvc")


# ----------------------------------------------------------------------------- Fig: IBM CHSH
def fig_ibm():
    d = J("d3_ibm_noise")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(COL1, 1.8))
    b = d["backends"]
    x = np.arange(len(b))
    w = [r["chsh"]["win"] for r in b]
    ax.bar(x, w, color=[C["q"] if v > 0.75 else "#c0392b" for v in w], width=0.6)
    ax.axhline(0.75, color="k", ls="--", lw=0.7); ax.text(len(b) - 0.5, 0.752, "classical bound 0.75", ha="right", fontsize=6.5)
    ax.axhline(math.cos(math.pi / 8) ** 2, color=C["phys"], ls=":", lw=0.7)
    ax.text(len(b) - 0.5, 0.857, r"Tsirelson $\cos^2(\pi/8)$", ha="right", fontsize=6.5, color=C["phys"])
    ax.set_xticks(x); ax.set_xticklabels([r["backend"].replace("Fake", "") for r in b], rotation=40, ha="right")
    ax.set_ylim(0.6, 0.88); ax.set_ylabel("CHSH win rate")
    save(fig, "fig_ibm")


# ----------------------------------------------------------------------------- Fig: learning curves (entity history)
def fig_learning():
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for name, lab, col in [("d7_earnings_entity", "earnings", "#c0392b"), ("d5_football", "football", "#117a65"),
                           ("d4_weather", "rain", "#1b3a6b"), ("d8_stocks", "stocks", "#7f8c8d"), ("d9_quakes", "quakes", "#b9770e")]:
        d = J(name)
        if not d:
            continue
        lc = np.array(d["ceilings"]["learning_curve"]); A0 = d["ceilings"]["beta_ceiling"]["A0"]
        ax.plot(np.arange(len(lc)), lc - A0, color=col, label=lab)
        ax.plot(len(lc) + 1.5, d["ceilings"]["beta_ceiling"]["A"] - A0, ">", color=col, ms=3.5)
    ax.set_xlabel("past outcomes per entity $n$"); ax.set_ylabel(r"Bayes gain $A_1^*(n)-A_0$")
    ax.legend(fontsize=6.5); ax.set_title(r"entity-history ceilings (arrows: $n\to\infty$)")
    save(fig, "fig_learning")


ALL = {"quantum": fig_quantum, "chaos": fig_chaos, "learnability": fig_learnability, "illusory": fig_illusory,
       "beta": fig_beta, "atlas": fig_atlas, "atlas3d": fig_atlas3d, "qvc": fig_qvc, "ibm": fig_ibm, "learning": fig_learning}



def fig_law():
    d = J("law")
    if not d or "nn_fits" not in d:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2 * 0.78, 2.1))
    cmap = matplotlib.colormaps["plasma"]
    hs = sorted(float(h) for h in d["nn_fits"])
    for i, h in enumerate(hs):
        f = d["nn_fits"][str(h) if str(h) in d["nn_fits"] else str(int(h))]
        col = cmap(i / (len(hs) - 0.5))
        n = np.array(f["n"]); g = np.array(f["gaps"])
        ax.loglog(n, g, "o", ms=2.6, color=col)
        nn = np.logspace(3, 6.2, 30)
        ax.loglog(nn, f["c"] * nn ** (-f["exponent"]), "-", lw=0.8, color=col, label=f"$h={h:g}$")
    ax.axhline(0.05, color="k", ls=":", lw=0.6)
    ax.set_xlabel("training samples $n$"); ax.set_ylabel("learnability gap $A^*_{\mathrm{phys}}-A_{\mathrm{1NN}}$")
    ax.legend(fontsize=5.5, ncol=2, loc="lower left"); ax.set_title(r"(a) gap $\propto n^{-\beta}$, $\beta\approx 1/D$")
    law = d["nn_law_all"]["0.05"]
    P = np.array(law["points"])
    ax2.semilogy(P[:, 0], P[:, 1], "o", color="#1b3a6b", ms=3.5, label=r"measured $n^*(h)$ ($\varepsilon=0.05$)")
    hh = np.linspace(0.3, 4.3, 50)
    ax2.semilogy(hh, np.exp(law["a"] + law["gamma"] * hh), "-", color="#1b3a6b", lw=0.9,
                 label=fr"fit: $\gamma={law['gamma']:.2f}$ ($R^2={law['r2']:.3f}$)")
    lamD = 0.906 * 2.06
    a0 = np.mean(np.log(P[:, 1]) - lamD * P[:, 0])
    ax2.semilogy(hh, np.exp(a0 + lamD * hh), "--", color="#c0392b", lw=0.9, label=fr"theory slope $\lambda D_2={lamD:.2f}$")
    ax2.set_xlabel("forecast horizon $h$"); ax2.set_ylabel("samples to reach the ceiling")
    ax2.legend(fontsize=6); ax2.set_title("(b) learnability–horizon law")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_law")


ALL["law"] = fig_law


def fig_law3d():
    d = J("law")
    if not d or "nn_fits" not in d:
        return
    hs = sorted(float(h) for h in d["nn_fits"])
    fits = [d["nn_fits"][k] for k in sorted(d["nn_fits"], key=float)]
    n = np.array(fits[0]["n"])
    G = np.array([f["gaps"] for f in fits])
    Ln, H = np.meshgrid(np.log10(n), hs)
    fig = plt.figure(figsize=(COL1, 2.7))
    ax = fig.add_subplot(111, projection="3d")
    Z = np.log10(np.clip(G, 1e-3, None))
    ax.plot_surface(Ln, H, Z, cmap="plasma_r", linewidth=0.2, edgecolor="k", alpha=0.95, antialiased=True)
    ax.contour(Ln, H, Z, levels=[np.log10(0.05)], colors="w", linewidths=1.6, offset=None)
    ax.contour(Ln, H, Z, levels=[np.log10(0.05)], zdir="z", offset=-3.2, colors="k", linewidths=0.9)
    ax.set_zlim(-3.2, 0); ax.view_init(28, -128); pane3d(ax)
    ax.set_xlabel(r"$\log_{10} n$", labelpad=-4); ax.set_ylabel(r"horizon $h$", labelpad=-4)
    ax.set_zlabel(r"$\log_{10}$ gap", labelpad=-4)
    ax.set_title(r"learnability landscape (white: gap $=0.05$)", pad=-2)
    save(fig, "fig_law3d")


ALL["law3d"] = fig_law3d


if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        try:
            ALL[k]()
        except Exception as e:
            print("FAILED", k, type(e).__name__, e)
