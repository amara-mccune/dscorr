"""Mellin-Barnes contour machinery of the tree engines.

Leg function of a line that ends on a quadratic mixing (power 1/2, energy x):

    M(s'; x, y, nu) = Gamma(-s') / Gamma(nu + s' + 1)
        * int ds/(2 pi i) Gamma(-s) Gamma(b) Gamma(1-b) / Gamma(nu + s + 1)
                          (y/2)^(2b-2) Gamma(1/2 + nu + 2s) x^-(1/2 + nu + 2s),
    b = 1 + nu + s + s',

computed on the straight inner contour s = c_s + i u by the trapezoidal rule.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import loggamma

LN_PI = math.log(math.pi)
LN_I_HALF = complex(math.log(0.5), math.pi / 2.0)  # ln(i/2)


def thermal_weights(mu):
    """(w_+, w_-) = (e^{pi mu}, -e^{-pi mu}) / (2 sinh pi mu)."""
    mu = float(mu)
    d = 2.0 * math.sinh(math.pi * mu)
    return math.exp(math.pi * mu) / d, -math.exp(-math.pi * mu) / d


def log_pi_over_sin(b):
    """ln(pi / sin(pi b)), stable for large |Im b| (branch immaterial)."""
    b = np.asarray(b, dtype=complex)
    pos = b.imag >= 0
    bb = np.where(pos, b, np.conj(b))
    lns = np.log1p(-np.exp(2j * np.pi * bb)) - 1j * np.pi * bb + LN_I_HALF
    lns = np.where(pos, lns, np.conj(lns))
    return LN_PI - lns


def trap_grid(T, h):
    """Nodes u_j = -T + j h, j = 0..N (N = 2T/h) and trapezoidal weights."""
    N = int(round(2.0 * T / h))
    u = -T + np.arange(N + 1) * h
    wt = np.ones(N + 1)
    wt[0] = wt[-1] = 0.5
    return u, wt


class InnerContour:
    """The inner contour s_j = c_s + i u_j with trapezoidal weights."""

    def __init__(self, c_s=-0.125, T_s=20.0, h_s=0.02):
        self.c_s, self.T_s, self.h_s = float(c_s), float(T_s), float(h_s)
        u, wt = trap_grid(self.T_s, self.h_s)
        self.s = self.c_s + 1j * u
        self.wt = wt


def leg_function(sp, x, y, nu, inner, chunk=256):
    """M(s'; x, y, nu) for an array of s' on the inner contour ``inner``."""
    sp = np.atleast_1d(np.asarray(sp, dtype=complex))
    nu = complex(nu)
    s = inner.s
    lnx = math.log(float(x))
    A = (loggamma(-s) - loggamma(nu + s + 1.0) + loggamma(0.5 + nu + 2.0 * s)
         - (0.5 + nu + 2.0 * s) * lnx)
    lny2 = math.log(float(y) / 2.0)
    out = np.empty(sp.shape, dtype=complex)
    for i in range(0, sp.size, chunk):
        spc = sp[i:i + chunk]
        b = 1.0 + nu + s[None, :] + spc[:, None]
        lnf = A[None, :] + (2.0 * b - 2.0) * lny2 + log_pi_over_sin(b)
        out[i:i + chunk] = np.exp(lnf) @ inner.wt
    pre = np.exp(loggamma(-sp) - loggamma(nu + sp + 1.0))
    return pre * (inner.h_s / (2.0 * math.pi)) * out
