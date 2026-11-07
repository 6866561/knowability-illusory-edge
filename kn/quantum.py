"""Batched statevector quantum models on GPU (PyTorch), plus matched classical surrogates.

Q1  Variational quantum classifier: RY angle embedding, L strongly-entangling layers (Rot + CNOT ring), readout <Z_0>.
Q2  Fidelity quantum kernel with the ZZ feature map of Havlicek et al. (2019); SVC on the precomputed kernel.
Q3  Projected quantum kernel (Huang et al. 2021): RBF kernel on single-qubit Bloch vectors of the ZZ-map state.
CQ1 Classical Fourier surrogate of Q1 (Schuld, Sweke & Meyer 2021): logistic regression on the trigonometric
    monomials that span Q1's function class, prod_j {1, cos x_j, sin x_j} truncated to degree <= 2.
"""
from __future__ import annotations

import itertools, math
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CT = torch.complex64


def _apply_1q(state, U, q, d):
    """state (N, 2^d); U (2,2) or (N,2,2); acts on qubit q (q = 0 is the most significant bit)."""
    N = state.shape[0]
    s = state.reshape(N, 2 ** q, 2, 2 ** (d - q - 1))
    s = torch.einsum("ij,najb->naib", U, s) if U.dim() == 2 else torch.einsum("nij,najb->naib", U, s)
    return s.reshape(N, 2 ** d)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CT)


def _rz(phi):
    e = torch.exp(-0.5j * phi.to(CT))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def _bits(d):
    b = np.array(list(itertools.product([0, 1], repeat=d)))  # row k = bits of basis state k, MSB first
    return b


def _cnot_ring_perm(d):
    b = _bits(d).copy()
    for c in range(d):
        t = (c + 1) % d
        if d == 1:
            break
        b[:, t] = b[:, t] ^ b[:, c]
    idx = (b * (2 ** np.arange(d - 1, -1, -1))).sum(1)
    # new_state[idx[k]] = old_state[k]  ->  new = old[inv]
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))
    return torch.as_tensor(inv, device=DEV)


class VQC(torch.nn.Module):
    def __init__(self, d, layers=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.L = d, layers
        self.w = torch.nn.Parameter(torch.rand(layers, d, 3, generator=g) * 2 * math.pi)
        self.a = torch.nn.Parameter(torch.tensor(1.0)); self.b = torch.nn.Parameter(torch.tensor(0.0))
        self.perm = _cnot_ring_perm(d)
        self.zsign = torch.as_tensor(1 - 2 * _bits(d)[:, 0], dtype=torch.float32, device=DEV)

    def expval(self, x):
        N, d = x.shape[0], self.d
        st = torch.zeros(N, 2 ** d, dtype=CT, device=x.device); st[:, 0] = 1
        for q in range(d):
            st = _apply_1q(st, _ry(x[:, q]), q, d)
        for l in range(self.L):
            for q in range(d):
                phi, th, om = self.w[l, q]
                U = _rz(om) @ _ry(th) @ _rz(phi)
                st = _apply_1q(st, U, q, d)
            st = st[:, self.perm]
        return (st.abs() ** 2 * self.zsign).sum(1)

    def forward(self, x):
        return torch.sigmoid(self.a * self.expval(x) + self.b)


def fit_vqc(Xtr, ytr, Xva, yva, d, layers=2, epochs=80, lr=0.05, batch=4096, seed=0, patience=10):
    torch.manual_seed(seed)
    m = VQC(d, layers, seed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    Xv = torch.as_tensor(Xva, dtype=torch.float32, device=DEV); yv = torch.as_tensor(yva, dtype=torch.float32, device=DEV)
    bce = torch.nn.BCELoss()
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch].to(DEV)
            opt.zero_grad()
            loss = bce(m(Xt[idx]).clamp(1e-6, 1 - 1e-6), yt[idx]); loss.backward(); opt.step()
        with torch.no_grad():
            vl = bce(m(Xv).clamp(1e-6, 1 - 1e-6), yv).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m


def predict_vqc(m, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            out.append(m(torch.as_tensor(X[i:i + 16384], dtype=torch.float32, device=DEV)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- ZZ feature map states
def _wht(st, d):
    H = torch.tensor([[1, 1], [1, -1]], dtype=CT, device=st.device) / math.sqrt(2)
    for q in range(d):
        st = _apply_1q(st, H, q, d)
    return st


def zz_states(X, reps=2):
    """|phi(x)> = (U_Z(x) H^d)^reps |0>, U_Z = exp(i sum_j x_j Z_j + i sum_{j} (pi-x_j)(pi-x_{j+1}) Z_j Z_{j+1})."""
    X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    N, d = X.shape
    z = torch.as_tensor(1 - 2 * _bits(d), dtype=torch.float32, device=DEV)  # (2^d, d), +-1
    phase = X @ z.T
    for j in range(d - 1):
        phase = phase + ((math.pi - X[:, j]) * (math.pi - X[:, j + 1]))[:, None] * (z[:, j] * z[:, j + 1])[None, :]
    st = torch.zeros(N, 2 ** d, dtype=CT, device=DEV); st[:, 0] = 1
    for _ in range(reps):
        st = _wht(st, d)
        st = st * torch.exp(1j * phase.to(CT))
    return st


def fidelity_kernel(S1, S2):
    out = []
    for i in range(0, len(S1), 4096):
        out.append((S1[i:i + 4096].conj() @ S2.T).abs().pow(2).cpu().numpy())
    return np.concatenate(out)


def bloch_vectors(S, d):
    """Single-qubit reduced density matrices -> (<X_k>, <Y_k>, <Z_k>) for each qubit k."""
    H = torch.tensor([[1, 1], [1, -1]], dtype=CT, device=S.device) / math.sqrt(2)
    Sdg = torch.tensor([[1, 0], [0, -1j]], dtype=CT, device=S.device)
    zs = torch.as_tensor(1 - 2 * _bits(d), dtype=torch.float32, device=S.device)
    feats = []
    for k in range(d):
        p = S.abs() ** 2
        ez = (p * zs[:, k]).sum(1)
        ex = ((_apply_1q(S, H, k, d).abs() ** 2) * zs[:, k]).sum(1)
        ey = ((_apply_1q(S, H @ Sdg, k, d).abs() ** 2) * zs[:, k]).sum(1)
        feats += [ex, ey, ez]
    return torch.stack(feats, 1).cpu().numpy()


def fit_kernel_svc(Ktr, ytr, seed=0):
    return SVC(kernel="precomputed", C=1.0, probability=True, random_state=seed).fit(Ktr, ytr)


def rbf(A, B, gamma):
    A2 = (A ** 2).sum(1)[:, None]; B2 = (B ** 2).sum(1)[None, :]
    return np.exp(-gamma * np.maximum(A2 + B2 - 2 * A @ B.T, 0))


# ----------------------------------------------------------------------------- classical Fourier surrogate
def fourier_features(X, degree=2):
    d = X.shape[1]
    base = [np.ones(len(X))]
    uni = [(np.cos(X[:, j]), np.sin(X[:, j])) for j in range(d)]
    cols = []
    for j in range(d):
        cols += list(uni[j])
    if degree >= 2:
        for j, k in itertools.combinations(range(d), 2):
            for a in uni[j]:
                for b in uni[k]:
                    cols.append(a * b)
    return np.column_stack(cols)


def fit_fourier(Xtr, ytr, C=0.5):
    return LogisticRegression(C=C, max_iter=4000).fit(fourier_features(Xtr), ytr)
