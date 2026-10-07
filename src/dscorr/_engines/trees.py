"""Numerical core of the tree engines.

Nothing here validates input; the public layer (``dscorr.tree``) does.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import loggamma

from .contour import InnerContour, leg_function, thermal_weights, trap_grid

SIGNS2 = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SIGNS3 = tuple(itertools.product((1, -1), repeat=3))

R_DOUBLE = 2.0      # power of the pi-dot sigma1 sigma2 vertex
R_TRIPLE = 1.5      # power of the sigma_A sigma_B sigma_C vertex
R_SINGLE = 2.5      # power of the pi-dot pi-dot sigma vertex


# --------------------------------------------------------------------------
# cut blocks (time-domain integrals on the ln t grid)
# --------------------------------------------------------------------------

def single_channel(grid, blk, x):
    """Bracket of one single-exchange channel: -2 Re psi2F + (2/pi) Phi_A Phi_B."""
    meas = grid.pow(R_SINGLE) * grid.damp(x)
    psi2F = grid.simpson(meas * blk.FF)
    PhiA = float(grid.simpson(meas * blk.K))
    PhiB = blk.Phi
    bracket = -2.0 * psi2F.real + (2.0 / math.pi) * PhiA * PhiB
    return {"bracket": float(bracket), "psi2F": complex(psi2F), "PhiA": PhiA, "PhiB": PhiB}


def double_cut_blocks(grid, b1, b2, x_B):
    """Cut blocks and time-domain uncut block of the double exchange.

    b1: line sigma_1 (on k2), b2: line sigma_2 (on k3), x_B = k1.
    """
    meas = grid.pow(R_DOUBLE) * grid.damp(x_B)
    sw = grid.simpson
    return {
        "Phi_1": b1.Phi,
        "Phi_2": b2.Phi,
        "Psi_2": complex(sw(meas * b1.K * b2.FF)),
        "Psi_1": complex(sw(meas * b2.K * b1.FF)),
        "Xi": float(sw(meas * b1.K * b2.K)),
        "psiF_time": complex(sw(meas * b1.FF * b2.FF)),
    }


def double_assemble(psiF, cb):
    uncut = 2.0 * psiF.real
    cut1 = -(2.0 / math.pi) * (cb["Phi_1"] * cb["Psi_2"].real + cb["Phi_2"] * cb["Psi_1"].real)
    cut2 = (2.0 / math.pi ** 2) * cb["Phi_1"] * cb["Phi_2"] * cb["Xi"]
    return uncut, cut1, cut2, uncut + cut1 + cut2


def triple_cut_blocks(grid, bA, bB, bC):
    """Cut blocks and time-domain uncut block of the triple exchange."""
    meas = grid.pow(R_TRIPLE)
    sw = grid.simpson
    b = (bA, bB, bC)
    Psi, Xi = [], []
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        Psi.append(complex(sw(meas * b[i].K * b[j].FF * b[k].FF)))
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        Xi.append(complex(sw(meas * b[i].K * b[j].K * b[k].FF)))
    return {
        "Phi": [bA.Phi, bB.Phi, bC.Phi],
        "Phic": float(sw(meas * bA.K * bB.K * bC.K)),
        "Psi": Psi,
        "Xi": Xi,
        "psiF_time": complex(sw(meas * bA.FF * bB.FF * bC.FF)),
    }


def triple_assemble(psiF, cb):
    P = cb["Phi"]
    uncut = -2.0 * psiF.real
    cut1 = (2.0 / math.pi) * sum(P[i] * cb["Psi"][i].real for i in range(3))
    cut2 = 0.0
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        cut2 += P[i] * P[j] * cb["Xi"][k].real
    cut2 = -(2.0 / math.pi ** 2) * cut2
    cut3 = (2.0 / math.pi ** 3) * P[0] * P[1] * P[2] * cb["Phic"]
    return uncut, cut1, cut2, cut3, uncut + cut1 + cut2 + cut3


# --------------------------------------------------------------------------
# contour (Mellin-Barnes) uncut blocks
# --------------------------------------------------------------------------

class _ContourBase:
    def __init__(self, c, T, h, c_s, T_s, h_s):
        self.c, self.T, self.h = float(c), float(T), float(h)
        self.inner = InnerContour(c_s, T_s, h_s)
        u, wt = trap_grid(self.T, self.h)
        self.u, self.wt = u, wt
        self.N = u.size - 1
        self.sp = self.c + 1j * u                      # outer contour s'
        self.w = -2.0 * self.T + np.arange(2 * self.N + 1) * self.h   # sum grid
        self.pref = (self.h / (2.0 * math.pi)) ** 2
        self._prepared = False

    def _twist(self, arr, nu, k, sp=None):
        """arr * k^(nu + 2 s' - 1/2) (power law of the leg function)."""
        sp = self.sp if sp is None else sp
        return arr * np.exp((nu + 2.0 * sp - 0.5) * math.log(k))


class DoubleContour(_ContourBase):
    """psi^F of the chain block on straight contours."""

    def __init__(self, mu1, mu2, c=-0.25, T=12.0, h=0.02, c_s=-0.125, T_s=20.0, h_s=0.02):
        super().__init__(c, T, h, c_s, T_s, h_s)
        self.mu1, self.mu2 = float(mu1), float(mu2)
        self.w1 = thermal_weights(self.mu1)
        self.w2 = thermal_weights(self.mu2)

    def nus(self, s1, s2):
        return s1 * 1j * self.mu1, s2 * 1j * self.mu2

    def weight(self, s1, s2):
        return self.w1[0 if s1 > 0 else 1] * self.w2[0 if s2 > 0 else 1]

    def prepare(self):
        if self._prepared:
            return
        self.wtM1 = {}
        self.wtM2 = {}
        for s in (1, -1):
            self.wtM1[s] = self.wt * leg_function(self.sp, 1.0, 1.0, s * 1j * self.mu1, self.inner)
            self.wtM2[s] = self.wt * leg_function(self.sp, 1.0, 1.0, s * 1j * self.mu2, self.inner)
        self._prepared = True

    def G(self, nu1, nu2, xB):
        z = R_DOUBLE + nu1 + nu2 + 4.0 * self.c + 2j * self.w
        return np.exp(loggamma(z) - z * math.log(xB))

    def chain(self, k1, k2, k3, s1, s2, method="twist"):
        nu1, nu2 = self.nus(s1, s2)
        if method == "engine":
            a = self.wt * leg_function(self.sp, k2, k2, nu1, self.inner)
            c = self.wt * leg_function(self.sp, k3, k3, nu2, self.inner)
        else:
            self.prepare()
            a = self._twist(self.wtM1[s1], nu1, k2)
            c = self._twist(self.wtM2[s2], nu2, k3)
        W = np.convolve(a, c)
        return 0.25 * self.pref * np.sum(W * self.G(nu1, nu2, k1))

    def psiF(self, k1, k2, k3, method="twist"):
        tot = 0j
        for s1, s2 in SIGNS2:
            tot += self.weight(s1, s2) * self.chain(k1, k2, k3, s1, s2, method)
        return complex(tot)

    # ---- family scans ----
    def scan_soft_k3(self, kappas):
        """psi^F on (1, 1, kappa) for all kappas."""
        self.prepare()
        lk = np.log(kappas)
        out = np.zeros(lk.size, dtype=complex)
        N = self.N
        for s1, s2 in SIGNS2:
            nu1, nu2 = self.nus(s1, s2)
            G = self.G(nu1, nu2, 1.0)
            a = self.wtM1[s1]
            # V_j = sum_a a_a G_{a+j}, j = 0..N
            V = np.array([np.dot(a, G[j:j + N + 1]) for j in range(N + 1)])
            cV = self.wtM2[s2] * V
            ph = np.exp(2j * np.outer(lk, self.u))
            pw = np.exp((nu2 + 2.0 * self.c - 0.5) * lk)
            out += self.weight(s1, s2) * 0.25 * self.pref * pw * (ph @ cV)
        return out

    def scan_soft_k1(self, kappas):
        """psi^F on (kappa, 1, 1) for all kappas."""
        self.prepare()
        lk = np.log(kappas)
        out = np.zeros(lk.size, dtype=complex)
        for s1, s2 in SIGNS2:
            nu1, nu2 = self.nus(s1, s2)
            WG = np.convolve(self.wtM1[s1], self.wtM2[s2]) * self.G(nu1, nu2, 1.0)
            ph = np.exp(-2j * np.outer(lk, self.w))
            pw = np.exp(-(R_DOUBLE + nu1 + nu2 + 4.0 * self.c) * lk)
            out += self.weight(s1, s2) * 0.25 * self.pref * pw * (ph @ WG)
        return out


class TripleContour(_ContourBase):
    """psi^F of the star block on straight contours."""

    def __init__(self, muA, muB, muC, c=-0.25, T=12.0, h=0.02, c_s=-0.125, T_s=20.0, h_s=0.02):
        super().__init__(c, T, h, c_s, T_s, h_s)
        self.mus = (float(muA), float(muB), float(muC))
        self.tw = [thermal_weights(m) for m in self.mus]

    def nus(self, sg):
        return tuple(s * 1j * m for s, m in zip(sg, self.mus))

    def weight(self, sg):
        r = 1.0
        for s, tw in zip(sg, self.tw):
            r *= tw[0 if s > 0 else 1]
        return r

    def spC(self, sg):
        nuA, nuB, nuC = self.nus(sg)
        S0 = -0.5 * (1.5 + nuA + nuB + nuC) - 2.0 * self.c
        return S0 - 1j * self.w

    def prepare(self):
        if self._prepared:
            return
        muA, muB, muC = self.mus
        self.wtMA = {s: self.wt * leg_function(self.sp, 1.0, 1.0, s * 1j * muA, self.inner)
                     for s in (1, -1)}
        self.wtMB = {s: self.wt * leg_function(self.sp, 1.0, 1.0, s * 1j * muB, self.inner)
                     for s in (1, -1)}
        self.MC = {sg: leg_function(self.spC(sg), 1.0, 1.0, sg[2] * 1j * muC, self.inner)
                   for sg in SIGNS3}
        self._prepared = True

    def star(self, k1, k2, k3, sg, method="twist"):
        nuA, nuB, nuC = self.nus(sg)
        spC = self.spC(sg)
        if method == "engine":
            a = self.wt * leg_function(self.sp, k1, k1, nuA, self.inner)
            b = self.wt * leg_function(self.sp, k2, k2, nuB, self.inner)
            c = leg_function(spC, k3, k3, nuC, self.inner)
        else:
            self.prepare()
            a = self._twist(self.wtMA[sg[0]], nuA, k1)
            b = self._twist(self.wtMB[sg[1]], nuB, k2)
            c = self._twist(self.MC[sg], nuC, k3, sp=spC)
        W = np.convolve(a, b)
        return (1.0 / 16.0) * self.pref * np.sum(W * c)

    def psiF(self, k1, k2, k3, method="twist"):
        tot = 0j
        for sg in SIGNS3:
            tot += self.weight(sg) * self.star(k1, k2, k3, sg, method)
        return complex(tot)

    def scan(self, kappas):
        """psi^F on (1, 1, kappa) for all kappas."""
        self.prepare()
        lk = np.log(kappas)
        out = np.zeros(lk.size, dtype=complex)
        ph = np.exp(-2j * np.outer(lk, self.w))
        for sg in SIGNS3:
            nuA, nuB, nuC = self.nus(sg)
            V = np.convolve(self.wtMA[sg[0]], self.wtMB[sg[1]]) * self.MC[sg]
            pw = np.exp((-2.0 - 4.0 * self.c - nuA - nuB) * lk)
            out += self.weight(sg) * (1.0 / 16.0) * self.pref * pw * (ph @ V)
        return out
