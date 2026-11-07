"""D3: quantum measurement on IBM device noise models (calibration snapshots bundled with qiskit-ibm-runtime).

For each backend and 4 physical qubits: prepare R_y(theta)|0> on a 32-point theta grid, 8192 shots, measure.
Fit p' = alpha + beta p (readout/depolarizing), compute device ceiling A*_dev = E_theta[max(p', 1-p')] and kappa_dev.
Also run the CHSH game with a Bell pair on the best-connected qubit pair: does device noise keep the quantum
win rate above the classical bound 0.75?
Writes results/d3_ibm_noise.json.
"""
import json, sys, pathlib, math, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as fp

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kn import core

SEED = 20261106
BACKENDS = ["FakeSherbrooke", "FakeBrisbane", "FakeKyiv", "FakeTorino", "FakeFez", "FakeMarrakesh", "FakeQuebec", "FakeKawasaki"]
THETAS = np.linspace(0, math.pi, 32)
SHOTS = 8192


def ry_circuits(q, n):
    cs = []
    for th in THETAS:
        qc = QuantumCircuit(1, 1)
        qc.ry(th, 0); qc.measure(0, 0)
        cs.append(qc)
    return cs


def chsh_circuits(a, b, n):
    cs, settings = [], []
    for x in (0, 1):
        for y in (0, 1):
            qc = QuantumCircuit(2, 2)
            a_, b_ = a, b
            a, b = 0, 1
            qc.h(a); qc.cx(a, b)
            ta = 0.0 if x == 0 else math.pi / 4
            tb = math.pi / 8 if y == 0 else -math.pi / 8
            qc.ry(-2 * ta, a); qc.ry(-2 * tb, b)
            qc.measure(a, 0); qc.measure(b, 1)
            a, b = a_, b_
            cs.append(qc); settings.append((x, y))
    return cs, settings


def main():
    out = {"ideal": {"A": 0.5 + 1 / math.pi, "kappa": 1 / math.log(2) - 1, "chsh": math.cos(math.pi / 8) ** 2},
           "backends": []}
    t0 = time.time()
    for name in BACKENDS:
        try:
            be = getattr(fp, name)()
        except Exception as e:
            print(name, "unavailable", e); continue
        sim = AerSimulator.from_backend(be, seed_simulator=SEED)
        n = be.num_qubits
        props = []
        cmap = be.coupling_map
        edges = list(cmap.get_edges()) if cmap is not None else []
        qubits = [0, n // 4, n // 2, 3 * n // 4]
        qres = []
        for q in qubits:
            cs = transpile(ry_circuits(q, n), be, initial_layout=[q], optimization_level=1, seed_transpiler=SEED)
            counts = sim.run(cs, shots=SHOTS).result().get_counts()
            p_obs = np.array([c.get("1", 0) / SHOTS for c in counts])
            p_ideal = np.sin(THETAS / 2) ** 2
            beta, alpha = np.polyfit(p_ideal, p_obs, 1)
            # device ceiling: theta ~ U(0, pi); p' from the fitted line (smooths shot noise)
            tt = np.linspace(0, math.pi, 20001); pp = np.clip(alpha + beta * np.sin(tt / 2) ** 2, 0, 1)
            A_dev = float(np.mean(np.maximum(pp, 1 - pp)))
            kap = float(1 - core.h(pp).mean() / core.h(pp.mean()))
            lam_equiv = float(1 - math.pi * (A_dev - 0.5))   # depolarizing strength giving the same ceiling
            qres.append({"qubit": q, "alpha_e0": float(alpha), "beta": float(beta), "e1": float(1 - alpha - beta),
                         "A_dev": A_dev, "kappa_dev": kap, "lambda_equiv": lam_equiv,
                         "max_abs_resid": float(np.max(np.abs(p_obs - (alpha + beta * p_ideal))))})
        # CHSH on an edge near the middle of the device
        chsh = None
        if edges:
            a, b = edges[len(edges) // 2]
            cs, st = chsh_circuits(a, b, n)
            cs = transpile(cs, be, initial_layout=[a, b], optimization_level=1, seed_transpiler=SEED)
            res = sim.run(cs, shots=SHOTS).result().get_counts()
            wins = []
            for (x, y), c in zip(st, res):
                same = sum(v for k, v in c.items() if k[0] == k[1]) / SHOTS   # bitstring 'ba'
                wins.append(1 - same if (x & y) else same)
            chsh = {"pair": [int(a), int(b)], "win": float(np.mean(wins)), "per_setting": [float(w) for w in wins]}
        rec = {"backend": name, "num_qubits": n, "qubits": qres, "chsh": chsh,
               "A_dev_mean": float(np.mean([r["A_dev"] for r in qres])),
               "kappa_dev_mean": float(np.mean([r["kappa_dev"] for r in qres]))}
        out["backends"].append(rec)
        print(f"{name}: n={n} A_dev={rec['A_dev_mean']:.4f} kappa={rec['kappa_dev_mean']:.4f} "
              f"CHSH={chsh['win'] if chsh else None} t={time.time() - t0:.0f}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "d3_ibm_noise.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
