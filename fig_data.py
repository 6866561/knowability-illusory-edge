"""Precompute data for 3D figures (GPU): local knowability on the Lorenz attractor.

For 40,000 attractor points and horizons h in {0.5, 1, 2, 3}, the Bayes certainty 2|eta - 1/2| of sign(x_{t+h})
given a noisy observation (sigma = 0.1), from a 256-particle ensemble. Saves results/fig_lorenz_local.npz.
"""
import pathlib, sys, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_learnability as L

DEV = L.DEV


def main():
    t0 = time.time()
    g = torch.Generator(device=DEV).manual_seed(7)
    X = L.attractor(40000)
    sigma, npart = 0.1, 256
    obs = X + sigma * torch.randn(X.shape, device=DEV, dtype=torch.float64, generator=g)
    parts = obs.repeat_interleave(npart, 0) + sigma * torch.randn(len(X) * npart, 3, device=DEV, dtype=torch.float64, generator=g)
    hs, cert, k = [0.5, 1, 2, 3], {}, 0
    for h in hs:
        steps = int(round(h / 0.01)) - k
        for _ in range(steps):
            parts = L.rk4(parts)
        k += steps
        eta = (parts[:, 0] > 0).double().view(len(X), npart).mean(1)
        cert[f"h{h}"] = (2 * (eta - 0.5).abs()).cpu().numpy()
        print(h, float(cert[f"h{h}"].mean()), time.time() - t0, flush=True)
    np.savez_compressed(ROOT / "results" / "fig_lorenz_local.npz", X=X.cpu().numpy(), **cert)


if __name__ == "__main__":
    main()
