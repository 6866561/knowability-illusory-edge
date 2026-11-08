"""Clean-room reproduction of the principal CPU results from a fresh clone of the public repository.

Run from the repository root in a fresh environment (see manuscript_rev2/release/README.md):
    python reproduce.py            # about 30-40 minutes on a laptop CPU
It records the committed result files, re-runs the analyses, and compares every numeric value in the regenerated files
with the committed ones (relative tolerance 1e-9, absolute 1e-12). Writes results/reproduction.json and reproduction_log.txt.

Not covered here (documented in the output): the GPU chaos studies, the quantum-model comparison, task builders that
download raw data, the stock-direction rows (prediction file not redistributed), and the rule-based guidance screen of the
audit cross-check, which needs the press-release texts (their SEC URLs are in data/d7_human_audit_sheet.csv).
"""
import json, math, pathlib, platform, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent
RTOL, ATOL = 1e-9, 1e-12
STEPS = [
    ("unit tests", [sys.executable, "-m", "pytest", "-q", "tests/test_core.py", "tests/test_rev2.py"], []),
    ("audit bounds, earnings cohort, readout ceilings, time-block and two-way intervals", [sys.executable, "rev2_analyses.py"], ["results/rev2_analyses.json"]),
    ("issuer-level breakdown", [sys.executable, "issuer_breakdown.py"], ["results/issuer_breakdown.json"]),
    ("qubit model term", [sys.executable, "qubit_modelterm.py"], ["results/qubit_modelterm.json"]),
    ("SEC XBRL download", [sys.executable, "fetch_audit_xbrl.py"], []),
    ("audit cross-check (reported-EPS side)", [sys.executable, "audit_crosscheck.py"], ["results/audit_crosscheck.json"]),
    ("coverage suite", [sys.executable, "coverage_suite.py"], ["results/coverage_suite.json"]),
]
# parts of regenerated files that legitimately differ in a clean clone (inputs not redistributed)
EXCLUDE = {"results/rev2_analyses.json": ["time_block.d8_stocks", "two_way.d8_stocks"],
           "results/audit_crosscheck.json": ["guid", "plus_screen", "rows"]}


def committed(path):
    r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


log = []
before = {f: committed(f) for _, _, fs in STEPS for f in fs}
steps_out = []
for name, cmd, files in STEPS:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    steps_out.append({"step": name, "command": " ".join(cmd[1:]), "returncode": r.returncode, "seconds": round(time.time() - t0, 1)})
    log.append(f"=== {name}: {' '.join(cmd[1:])} (exit {r.returncode}, {time.time() - t0:.0f}s)\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    print(name, "exit", r.returncode, flush=True)

comp, fails, maxrel, n_skip = 0, [], 0.0, 0
for f, old in before.items():
    if old is None:
        continue
    new = json.loads((ROOT / f).read_text())
    newl = dict(leaves(new)); ex = EXCLUDE.get(f, [])
    for k, v in leaves(old):
        if any(e in k for e in ex):
            n_skip += 1; continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        comp += 1
        w = newl.get(k)
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            fails.append({"file": f, "key": k, "committed": v, "reproduced": w}); continue
        if math.isnan(v) and math.isnan(w):
            continue
        d = abs(v - w); rel = d / max(abs(v), 1e-300)
        if not (d <= ATOL or rel <= RTOL):
            fails.append({"file": f, "key": k, "committed": v, "reproduced": w, "rel_diff": rel})
        if d > ATOL:
            maxrel = max(maxrel, rel)

out = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
       "python": sys.version.split()[0], "platform": platform.platform(), "steps": steps_out,
       "n_compared": comp, "n_pass": comp - len(fails), "n_excluded": n_skip, "max_rel_diff": maxrel,
       "tolerance_text": "relative 1e-9 or absolute 1e-12", "discrepancies": fails[:50],
       "not_covered": ["GPU chaos studies (run_law3.py, validate_posterior.py)", "quantum-model comparison (run_quantum2.py)",
                       "raw-data task builders", "stock-direction rows (prediction file not redistributed)",
                       "rule-based guidance screen (press-release texts not redistributed)"]}
(ROOT / "results" / "reproduction.json").write_text(json.dumps(out, indent=1, default=str))
(ROOT / "reproduction_log.txt").write_text("\n".join(log), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("commit", "n_compared", "n_pass", "max_rel_diff")}, indent=1))
