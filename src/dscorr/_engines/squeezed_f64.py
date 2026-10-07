"""Float64 route to the squeezed coefficients."""
from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.special import loggamma

from .bessel import besseli_imag, besselk_imag
from .grid import EXP_CLIP, LogGrid, cumulative_simpson

GRID_M = (-60.0, 12.0, 0.002)


def cgamma(z):
    return np.exp(loggamma(z))


def phi_hat(mu):
    return np.pi * np.sqrt(np.pi / 2.0) / np.cosh(np.pi * mu)


def lambdas(mu):
    d = 2.0 * np.sinh(np.pi * mu)
    return (np.exp(np.pi * mu) - 1j) / d, -(np.exp(-np.pi * mu) + 1j) / d


def H_moment(a, mu1):
    """int_0^inf dt t^(a-1) e^-t K_{i mu1}(t)."""
    return (np.sqrt(np.pi) * cgamma(a + 1j * mu1) * cgamma(a - 1j * mu1)
            / (np.exp(a * np.log(2.0)) * cgamma(a + 0.5)))


def T_KK(p, muA, muB):
    """int_0^inf dt t^(1/2+p) K_{i muA}(t) K_{i muB}(t)."""
    z = 1.5 + p
    prod = 1.0
    for s in (1, -1):
        for sp in (1, -1):
            prod = prod * cgamma((z + s * 1j * muA + sp * 1j * muB) / 2.0)
    return np.exp((z - 3.0) * np.log(2.0)) / cgamma(z) * prod


def M1(mu):
    nu = 1j * mu
    ph = np.pi * np.sqrt(np.pi / 2.0) / np.cosh(np.pi * mu)
    c = 1.0 - 1j / np.sinh(np.pi * mu)
    ln2 = np.log(2.0)
    val = (-(ph / 4.0) * c * np.exp(-nu * ln2) * cgamma(2.5 + nu)
           * np.exp((-2.5 - nu) * ln2) / cgamma(1.0 + nu))
    return complex(val)


class HardLine:
    """Leg functions of a hard line at unit momentum on the uniform ln t grid (4.3)."""

    def __init__(self, mu, grid):
        g = grid
        t = g.t
        K = besselk_imag(mu, t)
        I = besseli_imag(mu, t)
        J = I - (1j / np.pi) * np.exp(-np.pi * mu) * K
        w = np.exp(-np.minimum(t, EXP_CLIP))
        eh = np.exp(g.L / 2.0)
        CJ = cumulative_simpson(eh * w * J, g.hL)
        CI = cumulative_simpson(eh * w * I, g.hL)
        CK = cumulative_simpson(eh * w * K, g.hL)
        self.Phi_num = float(CK[-1])
        BK = self.Phi_num - CK
        self.K, self.I, self.J, self.w = K, I, J, w
        self.CJ, self.CI, self.CK, self.BK = CJ, CI, CK, BK
        self.FF = K * CJ + J * BK
        self.g = K * CI + I * BK


def _grid():
    return LogGrid(*GRID_M)


def M2_parts(mu1, mu2):
    g = _grid()
    hl = HardLine(mu1, g)
    ap, am = 2.0 + 1j * mu2, 2.0 - 1j * mu2
    HFp = g.simpson(np.exp(ap * g.L) * hl.w * hl.FF)
    HFm = g.simpson(np.exp(am * g.L) * hl.w * hl.FF)
    psp = g.simpson(np.exp(ap * g.L) * hl.w * hl.g)
    psm = g.simpson(np.exp(am * g.L) * hl.w * hl.g)
    H = H_moment(ap, mu1)
    ph1, ph2 = phi_hat(mu1), phi_hat(mu2)
    lp, lm = lambdas(mu2)
    csch = 1.0 / np.sinh(np.pi * mu2)
    pref = ph2 * np.exp(-1j * mu2 * np.log(2.0)) / (8.0 * cgamma(1.0 + 1j * mu2))
    A = pref * (lp * HFp + lm * np.conj(HFm) - ph1 / np.pi * (1.0 - 1j * csch) * H)
    coth = np.cosh(np.pi * mu2) / np.sinh(np.pi * mu2)
    A_alt = pref * (lp * psp + lm * np.conj(psm)
                    - ph1 * H / np.pi * ((1.0 - 1j * csch) + 1j * np.exp(-np.pi * mu1) * coth))
    e = np.exp(-np.pi * mu2)
    Ap = np.exp(-1j * mu2 * np.log(2.0)) * (1.0 / cgamma(1.0 + 1j * mu2)
                                            - 1j / (2.0 * np.pi) * e * cgamma(-1j * mu2))
    Am = -1j / (2.0 * np.pi) * e * np.exp(1j * mu2 * np.log(2.0)) * cgamma(1j * mu2)
    return {"A": A, "A_alt": A_alt, "HFp": HFp, "HFm": HFm, "psip": psp, "psim": np.conj(psm),
            "H": H, "Apsi_p": ph2 * Ap * HFp, "Apsi_m": ph2 * Am * HFm}


def M3_parts(muA, muB, muC):
    g = _grid()
    la = HardLine(muA, g)
    lb = HardLine(muB, g)
    r = {}
    for tag, p in (("p", 1j * muC), ("m", -1j * muC)):
        wgt = np.exp((1.5 + p) * g.L)
        r["TFF" + tag] = g.simpson(wgt * la.FF * lb.FF)
        r["U" + tag] = g.simpson(wgt * la.K * lb.FF)
        r["V" + tag] = g.simpson(wgt * la.FF * lb.K)
    TKK = T_KK(1j * muC, muA, muB)
    phA, phB, phC = phi_hat(muA), phi_hat(muB), phi_hat(muC)
    Tp = r["TFFp"] - phA / np.pi * r["Up"] - phB / np.pi * r["Vp"]
    Tm = np.conj(r["TFFm"] - phA / np.pi * r["Um"] - phB / np.pi * r["Vm"])
    lp, lm = lambdas(muC)
    csch = 1.0 / np.sinh(np.pi * muC)
    pref = -phC * np.exp(-1j * muC * np.log(2.0)) / (8.0 * cgamma(1.0 + 1j * muC))
    A = pref * (lp * Tp + lm * Tm + phA * phB / np.pi ** 2 * (1.0 - 1j * csch) * TKK)
    r["A"] = A
    r["TKK"] = TKK
    return {k: r[k] for k in ("A", "TFFp", "TFFm", "Up", "Um", "Vp", "Vm", "TKK")}
