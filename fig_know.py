"""Figures and macros for the knowability paper (paper_know/).

fig_demon: epistemic vs ontic limits. (a) Lorenz-63 ensemble-reference accuracy against horizon for each
observation-noise level, with the qubit's exact ceiling (independent of horizon); (b) ceiling against
observation noise: Lorenz-63 at fixed horizons rises towards 1 as noise falls, the qubit saturates at 1/2+1/pi.
All values from results/controls.json and results/rev2_analyses.json.
Writes ../paper_know/figs/fig_demon.{pdf,png} and ../paper_know/gen/macros_know.tex.
"""
import json, math, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import figs as F
RES = ROOT / "results"
PK = ROOT.parent / "paper_know"
OUT = PK / "figs"; OUT.mkdir(parents=True, exist_ok=True)
GEN = PK / "gen"; GEN.mkdir(parents=True, exist_ok=True)
F.OUT = OUT
from figs import save, COL1, COL2
J = lambda n: json.loads((RES / f"{n}.json").read_text())

c = J("controls"); lam = c["D2"]["lyapunov"]; grid = c["D2"]["grid"]
qa = 0.5 + 1 / math.pi
noisy = c["D1"]["noisy_theta"]
live = J("rev2_analyses")["readout"]["live"]
SIG = sorted({g["sigma"] for g in grid})
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(SIG)))


def fig_demon():
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.45))
    ax = axs[0]
    for s, col in zip(SIG, cols):
        rows = sorted([g for g in grid if g["sigma"] == s], key=lambda g: g["h"])
        L = [lam * g["h"] for g in rows]; A = [g["A_ensemble"] for g in rows]
        ax.plot(L, A, "o-", color=col, ms=2.6, lw=1.0, label=rf"$\sigma_o=10^{{{int(round(math.log10(s)))}}}$")
    ax.axhline(qa, color="#c0392b", ls="--", lw=1.0)
    ax.text(0.09, qa - 0.045, r"qubit: $1/2+1/\pi$ at every horizon and precision", color="#c0392b", fontsize=5.6)
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(r"horizon in Lyapunov times $\lambda h$"); ax.set_ylabel("ensemble-rule accuracy")
    ax.set_title("(a) chaos: precision buys horizon"); ax.legend(fontsize=5.2, loc="lower left", title="Lorenz-63", title_fontsize=5.4)
    ax = axs[1]
    for h, mk in ((2, "o"), (4, "s"), (8, "^")):
        rows = sorted([g for g in grid if g["h"] == h], key=lambda g: g["sigma"])
        ax.plot([g["sigma"] for g in rows], [g["A_ensemble"] for g in rows], mk + "-", color="#1b3a6b", ms=2.8, lw=0.9,
                mfc="white" if h == 8 else "#1b3a6b", label=rf"Lorenz-63, $\lambda h={lam * h:.1f}$")
    sq = [r["sigma"] for r in noisy if r["sigma"] > 0]; aq = [r["A_star"] for r in noisy if r["sigma"] > 0]
    ax.plot(sq, aq, "D-", color="#c0392b", ms=2.8, lw=0.9, label=r"qubit, angle noise $\sigma_\theta$ (rad)")
    ax.axhline(qa, color="#c0392b", ls="--", lw=0.8)
    for q, mk in (("5", "*"), ("42", "P")):
        ax.plot(1.2e-4 if q == "5" else 1.8e-4, live[q]["A_opt"], mk, color="#e67e22", ms=4.5, label=f"live ibm_kingston qubit {q} (readout)")
    ax.set_xscale("log"); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("observation noise (system units)"); ax.set_ylabel("accuracy: exact ceiling (qubit),\nensemble rule (Lorenz-63)")
    ax.set_title("(b) the demon's limit: noise $\\to0$"); ax.legend(fontsize=5.0, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_demon")


def macros():
    M = {}
    def g(s, h):
        return [x for x in grid if x["sigma"] == s and x["h"] == h][0]["A_ensemble"]
    M["kLam"] = f"{lam:.3f}"
    M["kQa"] = f"{qa:.4f}"
    M["kLzTinyEight"] = f"{g(1e-4, 8):.3f}"; M["kLzOneEight"] = f"{g(1.0, 8):.3f}"
    M["kLzTinyFour"] = f"{g(1e-4, 4):.3f}"; M["kLzOneFour"] = f"{g(1.0, 4):.3f}"
    M["kLzOneOne"] = f"{g(1.0, 1):.3f}"
    M["kLTinyEight"] = f"{lam * 8:.1f}"; M["kLFour"] = f"{lam * 4:.1f}"
    # finite-sample uncertainty of ensemble-rule accuracies (1,500 test states per cell; Wilson 95% interval)
    from statsmodels.stats.proportion import proportion_confint
    NT = 1500
    M["kNtest"] = f"{NT:,}"
    for key, (s_, h_) in {"kLzOneEight": (1.0, 8), "kLzTinyEight": (1e-4, 8)}.items():
        lo, hi = proportion_confint(round(g(s_, h_) * NT), NT, alpha=0.05, method="wilson")
        M[key + "lo"] = f"{lo:.3f}"; M[key + "hi"] = f"{hi:.3f}"
    M["kQsigOne"] = f"{[r for r in noisy if r['sigma'] == 1.0][0]['A_star']:.3f}"
    M["kQsigTenth"] = f"{[r for r in noisy if r['sigma'] == 0.1][0]['A_star']:.3f}"
    lz = c["D2"]["lz"]
    M["kLZpi"] = f"{lz['Pi_max_LZ']:.3f}"; M["kLZlearn"] = f"{lz['learned_next_symbol_acc']:.3f}"; M["kLZbase"] = f"{lz['base']:.3f}"
    dep = c["D1"]["depolarizing"]
    M["kDepTwo"] = f"{[d for d in dep if d['lambda'] == 0.2][0]['A_star']:.3f}"
    (GEN / "macros_know.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in M.items()) + "\n", encoding="utf-8")
    print(M)


if __name__ == "__main__":
    fig_demon(); macros()
