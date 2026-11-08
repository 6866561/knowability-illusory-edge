"""Exact checks of each component of the four-term decomposition on the qubit control (quadrature, no sampling)."""
import math
import numpy as np

xg, wg = np.polynomial.legendre.leggauss(2000)
TH = (xg + 1) * math.pi / 2; W = wg / 2; ETA = np.cos(TH / 2) ** 2


def _hb(p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _loss(q, p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(np.sum(W * (-(q * np.log(p) + (1 - q) * np.log(1 - p)))))


def test_conditional_entropy_closed_form():
    assert abs(float(np.sum(W * _hb(ETA))) - (2 * math.log(2) - 1)) < 1e-10


def test_model_term_identity_for_fixed_forecast():
    P = np.clip(ETA + 0.08 * np.sin(2 * TH), 0.01, 0.99)          # any fixed forecast of theta
    e = np.clip(ETA, 1e-15, 1 - 1e-15)
    kl = e * np.log(e / P) + (1 - e) * np.log((1 - e) / (1 - P))
    assert abs(_loss(ETA, P) - (np.sum(W * _hb(ETA)) + np.sum(W * kl))) < 1e-12
    assert np.sum(W * kl) > 0


def test_epistemic_term_nonnegative():
    bins = np.minimum((TH / math.pi * 8).astype(int), 7); HX = 0.0
    for b in range(8):
        s = bins == b; wb = W[s].sum(); HX += wb * _hb(np.sum(W[s] * ETA[s]) / wb)
    assert HX - np.sum(W * _hb(ETA)) > 0


def test_signed_measurement_term_both_signs():
    P = np.clip(ETA + 0.05 * np.sin(TH), 0.01, 0.99)
    for f01, f10, sign in ((np.where(P > .5, .15, 0), np.where(P < .5, .15, 0), -1), (np.where(P < .5, .15, 0), np.where(P > .5, .15, 0), +1)):
        q = ETA * (1 - f10) + (1 - ETA) * f01
        M = float(np.sum(W * (q - ETA) * np.log((1 - P) / P)))
        assert abs((_loss(q, P) - _loss(ETA, P)) - M) < 1e-12
        assert np.sign(M) == sign
