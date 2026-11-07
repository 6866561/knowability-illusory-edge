"""Learnability gap in deterministic chaos: physical ceiling (ensemble truth) vs data-driven ceilings as n grows.

Lorenz-63 integrated on GPU (torch, RK4, dt = 0.01). Observation: full state + N(0, sigma^2) per coordinate.
Target: sign of x at horizon h. For each h, train learners on n in {1e3, 1e4, 1e5, 1e6} and evaluate on 20k test states
whose Bayes probability is estimated with a 1,000-particle ensemble.
Writes results/learnability.json.
"""
import json, sys, pathlib, time, math
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kn import core

DEV = torch.device("cuda")
SEED = 20261106
S_, R_, B_ = 10.0, 28.0, 8 / 3
torch.manual_seed(SEED)


def rhs(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return torch.stack([S_ * (y - x), x * (R_ - z) - y, x * y - B_ * z], 1)


def rk4(X, dt=0.01):
    k1 = rhs(X); k2 = rhs(X + dt / 2 * k1); k3 = rhs(X + dt / 2 * k2); k4 = rhs(X + dt * k3)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def attractor(n):
    X = torch.randn(n, 3, device=DEV, dtype=torch.float64) + torch.tensor([0, 0, 25.0], device=DEV, dtype=torch.float64)
    for _ in range(2000):
        X = rk4(X)
    # decorrelate: random extra integration time per sample
    extra = torch.randint(0, 500, (n,), device=DEV)
    for s in range(500):
        m = extra > s
        X[m] = rk4(X[m])
    return X


def advance(X, steps, chunk=2_000_000):
    out = []
    for i in range(0, len(X), chunk):
        Y = X[i:i + chunk].clone()
        for _ in range(steps):
            Y = rk4(Y)
        out.append(Y)
    return torch.cat(out)


def mlp_fit_predict(Xtr, ytr, Xte, epochs=40):
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=DEV)
    net = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = max(200, epochs * len(Xt) // 4096)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    for s in range(steps):
        idx = torch.randint(0, len(Xt), (min(4096, len(Xt)),), device=DEV)
        opt.zero_grad(); loss = lossf(net(Xt[idx]).squeeze(1), yt[idx]); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)).squeeze(1)).cpu().numpy()
    return p


def main():
    t0 = time.time()
    sigma, hs, ns, ntest, npart = 1e-3, [1, 2, 4], [1000, 10000, 100000, 1000000], 20000, 1000
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    Xall = attractor(max(ns) + ntest)
    obs_all = Xall + sigma * torch.randn(Xall.shape, device=DEV, dtype=torch.float64, generator=gen)
    test_true, test_obs = Xall[:ntest], obs_all[:ntest]
    res = {"sigma": sigma, "ntest": ntest, "npart": npart, "rows": []}
    for h in hs:
        steps = int(round(h / 0.01))
        y_te = (advance(test_true, steps)[:, 0] > 0).cpu().numpy().astype(int)
        parts = test_obs.repeat_interleave(npart, 0) + sigma * torch.randn(ntest * npart, 3, device=DEV, dtype=torch.float64, generator=gen)
        eta = (advance(parts, steps)[:, 0] > 0).double().view(ntest, npart).mean(1).cpu().numpy()
        del parts; torch.cuda.empty_cache()
        A_phys = core.accuracy(y_te, eta)
        tr_true, tr_obs = Xall[ntest:], obs_all[ntest:]
        y_tr_all = (advance(tr_true, steps)[:, 0] > 0).cpu().numpy().astype(int)
        Xo_all, Xte = tr_obs.cpu().numpy(), test_obs.cpu().numpy()
        for n in ns:
            Xtr, ytr = Xo_all[:n], y_tr_all[:n]
            hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED).fit(Xtr, ytr)
            a_hgb = core.accuracy(y_te, hgb.predict_proba(Xte)[:, 1])
            nn = KNeighborsClassifier(1).fit(Xtr, ytr)
            r_nn = 1 - float(np.mean(nn.predict(Xte) == y_te))
            a_mlp = core.accuracy(y_te, mlp_fit_predict(Xtr, ytr, Xte))
            row = {"h": h, "n": n, "A_physical": A_phys, "A_hgb": a_hgb, "A_mlp": a_mlp, "r_nn": r_nn,
                   "A_cover_hart_upper": core.cover_hart_upper(r_nn)["A_upper"],
                   "learnability_gap_best": A_phys - max(a_hgb, a_mlp)}
            res["rows"].append(row)
            print(f"h={h} n={n:>7} phys={A_phys:.4f} hgb={a_hgb:.4f} mlp={a_mlp:.4f} CH={row['A_cover_hart_upper']:.4f} "
                  f"gap={row['learnability_gap_best']:.4f} t={time.time() - t0:.0f}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "learnability.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
