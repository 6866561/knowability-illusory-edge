"""Write gen/macros_default.tex: \\providecommand fallbacks (bold ??) for every result macro used in main.tex."""
import re, pathlib

HERE = pathlib.Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text(encoding="utf-8") + (HERE / "appendix_math.tex").read_text(encoding="utf-8")
names = set(re.findall(r"\\([a-zA-Z]+)", tex))
defined = set()
for g in (HERE / "gen").glob("macros*.tex"):
    if g.name != "macros_default.tex":
        defined |= set(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}", g.read_text(encoding="utf-8")))
pref = ("q", "chsh", "side", "lyap", "lz", "chaos", "chV", "chT", "gap", "phys", "law", "sim", "pre", "ibm", "wx", "nwp",
        "fb", "hs", "pr", "st", "eq", "e", "pv", "ro", "au", "ec", "pw", "floor", "Lor", "Ros", "Lfive", "Lsix", "Leight", "qrev", "dr", "db", "live", "bell", "land")
std = set("""emph eqref end eta epsilon exp equation enumerate parbox propto quad qquad setlength section scriptsize sim subsection
label ldots lambda left right ln log lvert rvert langle rangle hat hfill includegraphics input item kappa cite caption centering
chi cdot cos sin rho ref sigma textwidth texttt theta thanks title times bottomrule midrule toprule footnotesize tabcolsep
columnwidth mathbb mathcal mathrm max min mid markboth maketitle newtheorem newcommand alpha beta begin bibliography
bibliographystyle frac gamma ge infty int documentclass usepackage theoremstyle vspace varepsilon appendices psi phi pi
providecommand textbf tfrac sqrt sum leq le prime pmod pm author endcsname equiv""".split())
out = []
for n in sorted(names):
    if n.startswith(pref) and n not in std and len(n) > 3 and n not in defined:
        out.append("\\providecommand{\\" + n + "}{\\textbf{??}}")
(HERE / "gen" / "macros_default.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print("fallbacks:", len(out), out[:40])
