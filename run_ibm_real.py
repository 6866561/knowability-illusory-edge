"""Live IBM Quantum run (free Open Plan): qubit prediction ceiling and CHSH on real hardware.

Prerequisite (done once by the user, in their own terminal; the token is never shared):
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<YOUR API KEY>', overwrite=True)"

What it runs (about 10-20 s of QPU time, well inside the 10 min / 28 days free allowance):
  * 16 angles theta in [0, pi] x 2 physical qubits, R_y(theta) then measure, 4,000 shots each   (32 circuits)
  * CHSH with a Bell pair on one coupled pair, 4 setting combinations, 4,000 shots each        (4 circuits)
Results (counts + the backend calibration snapshot at run time) -> results/ibm_real_<backend>_<UTC date>.json
Usage: python run_ibm_real.py [--backend NAME] [--shots 4000] [--dry-run]
"""
import argparse, datetime as dt, json, math, pathlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

ROOT = pathlib.Path(__file__).resolve().parent
THETAS = np.linspace(0, math.pi, 16)


def ry_circ(th):
    qc = QuantumCircuit(1, 1); qc.ry(th, 0); qc.measure(0, 0); return qc


def chsh_circs():
    out = []
    for x in (0, 1):
        for y in (0, 1):
            qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1)
            qc.ry(-2 * (0.0 if x == 0 else math.pi / 4), 0); qc.ry(-2 * (math.pi / 8 if y == 0 else -math.pi / 8), 1)
            qc.measure([0, 1], [0, 1]); out.append(((x, y), qc))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend"); ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    svc = QiskitRuntimeService()
    be = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False)
    props = be.properties()
    # choose the two qubits with the lowest readout error, and the coupled pair with the lowest two-qubit error
    n = be.num_qubits
    ro = [(props.readout_error(q), q) for q in range(n) if props.readout_error(q) is not None]
    q_good = [q for _, q in sorted(ro)[:2]]
    edges = list(be.coupling_map.get_edges())
    pair = min(edges, key=lambda e: sum(props.readout_error(q) or 1 for q in e))
    circs, meta = [], []
    for q in q_good:
        for th in THETAS:
            circs.append(transpile(ry_circ(th), be, initial_layout=[q], optimization_level=1)); meta.append({"kind": "ry", "qubit": q, "theta": float(th)})
    for (x, y), qc in chsh_circs():
        circs.append(transpile(qc, be, initial_layout=list(pair), optimization_level=1)); meta.append({"kind": "chsh", "pair": list(pair), "x": x, "y": y})
    print(f"backend {be.name}: {len(circs)} circuits x {a.shots} shots; qubits {q_good}; CHSH pair {pair}")
    if a.dry_run:
        return
    job = SamplerV2(mode=be).run(circs, shots=a.shots)
    print("job", job.job_id(), "submitted; waiting ...")
    res = job.result()
    counts = [r.data.c.get_counts() for r in res]
    snap = {}
    for q in set(q_good) | set(pair):
        snap[str(q)] = {"readout_error": props.readout_error(q), "T1": props.t1(q), "T2": props.t2(q)}
    # analysis
    out = {"backend": be.name, "job_id": job.job_id(), "utc": dt.datetime.utcnow().isoformat(), "shots": a.shots,
           "calibration_snapshot": snap, "last_update": str(props.last_update_date), "runs": []}
    for m, c in zip(meta, counts):
        out["runs"].append({**m, "counts": c})
    for q in q_good:
        p_obs = np.array([r["counts"].get("1", 0) / a.shots for r in out["runs"] if r["kind"] == "ry" and r["qubit"] == q])
        p_id = np.sin(THETAS / 2) ** 2
        beta, alpha = np.polyfit(p_id, p_obs, 1)
        tt = np.linspace(0, math.pi, 20001); pp = np.clip(alpha + beta * np.sin(tt / 2) ** 2, 0, 1)
        out[f"qubit_{q}"] = {"alpha": float(alpha), "beta": float(beta), "A_dev": float(np.mean(np.maximum(pp, 1 - pp)))}
    wins = []
    for r in out["runs"]:
        if r["kind"] == "chsh":
            same = sum(v for k, v in r["counts"].items() if k[0] == k[1]) / a.shots
            wins.append(1 - same if (r["x"] & r["y"]) else same)
    out["chsh_win"] = float(np.mean(wins)) if wins else None
    path = ROOT / "results" / f"ibm_real_{be.name}_{dt.datetime.utcnow():%Y%m%d}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("saved", path, "| CHSH win rate", out["chsh_win"], "| qubit ceilings", {q: out[f'qubit_{q}']['A_dev'] for q in q_good})


if __name__ == "__main__":
    main()
