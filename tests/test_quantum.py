import sys, pathlib, math
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kn import quantum as Q
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def test_vqc_matches_qiskit():
    d, L = 3, 2
    m = Q.VQC(d, L, seed=3).to(Q.DEV)
    x = np.array([[0.3, 1.1, 2.0], [2.5, 0.2, 0.9]])
    ours = m.expval(torch.as_tensor(x, dtype=torch.float32, device=Q.DEV)).detach().cpu().numpy()
    w = m.w.detach().cpu().numpy()
    for n in range(len(x)):
        qc = QuantumCircuit(d)
        for q in range(d):
            qc.ry(x[n, q], d - 1 - q)          # qiskit is little-endian: our qubit q (MSB) = qiskit qubit d-1-q
        for l in range(L):
            for q in range(d):
                phi, th, om = w[l, q]
                qc.rz(phi, d - 1 - q); qc.ry(th, d - 1 - q); qc.rz(om, d - 1 - q)
            for c in range(d):
                qc.cx(d - 1 - c, d - 1 - ((c + 1) % d))
        z0 = SparsePauliOp("Z" + "I" * (d - 1))   # Z on qiskit qubit d-1 = our qubit 0
        ref = Statevector(qc).expectation_value(z0).real
        assert abs(ref - ours[n]) < 1e-4, (ref, ours[n])


def test_zz_states_normalised_and_kernel_diag_one():
    X = np.random.default_rng(0).uniform(0, math.pi, (20, 4))
    S = Q.zz_states(X)
    assert torch.allclose((S.abs() ** 2).sum(1), torch.ones(20, device=Q.DEV), atol=1e-5)
    K = Q.fidelity_kernel(S, S)
    assert np.allclose(np.diag(K), 1, atol=1e-5) and np.all(K <= 1 + 1e-5)
    B = Q.bloch_vectors(S, 4)
    norms = np.sqrt((B.reshape(20, 4, 3) ** 2).sum(-1))
    assert np.all(norms <= 1 + 1e-4)
