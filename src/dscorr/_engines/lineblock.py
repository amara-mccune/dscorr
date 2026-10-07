"""Line blocks on the logarithmic time grid.

For a line of mass mu and (normalised) momentum k, on the grid t_j = exp(L_j):

    z = k t,  K = K_{i mu}(z),  I = I_{i mu}(z),  J = I - (i/pi) e^{-pi mu} K,
    w = exp(-min(z, 745)),
    g^J = e^{L/2} w J,  g^K = e^{L/2} w K,
    C^J, C^K = cumulative Simpson integrals of g^J, g^K from L_0,
    Phi = C^K_n,  B = Phi - C^K,  FF = K C^J + J B.

Phi is the cut mixing int_0^inf du u^{-1/2} e^{-ku} K_{i mu}(ku) and FF the
Feynman leg function at every node.
"""
from __future__ import annotations

import math

import numpy as np

from .bessel import besseli_imag, besselk_imag
from .grid import EXP_CLIP, cumulative_simpson


class LineBlock:
    __slots__ = ("mu", "k", "K", "I", "J", "w", "CJ", "CK", "Phi", "B", "FF")

    def __init__(self, grid, mu, k):
        mu = float(mu)
        k = float(k)
        self.mu, self.k = mu, k
        z = k * grid.t
        K = besselk_imag(mu, z)
        I = besseli_imag(mu, z)
        J = I - (1j / math.pi) * math.exp(-math.pi * mu) * K
        w = np.exp(-np.minimum(z, EXP_CLIP))
        eh = np.exp(grid.L / 2.0)
        gJ = eh * w * J
        gK = eh * w * K
        CJ = cumulative_simpson(gJ, grid.hL)
        CK = cumulative_simpson(gK, grid.hL)
        Phi = float(CK[-1])
        B = Phi - CK
        self.K, self.I, self.J, self.w = K, I, J, w
        self.CJ, self.CK, self.Phi, self.B = CJ, CK, Phi, B
        self.FF = K * CJ + J * B


class LineBlockCache:
    """Per-routine cache keyed by (mu, k rounded to 14 decimals); emptied above 64 entries."""

    MAX = 64

    def __init__(self, grid):
        self.grid = grid
        self._d = {}

    def get(self, mu, k):
        key = (float(mu), round(float(k), 14))
        blk = self._d.get(key)
        if blk is None:
            if len(self._d) > self.MAX:
                self._d.clear()
            blk = LineBlock(self.grid, mu, k)
            self._d[key] = blk
        return blk

    def __len__(self):
        return len(self._d)
