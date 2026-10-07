"""Modified Bessel functions of imaginary order in double precision.

No library Bessel routine is called.  For real z > 0 and nu = i mu:

* I_{i mu}(z): ascending series with a common number of terms per array,
  m_max = floor(z_m/2 + 16 sqrt(z_m) + 50), z_m the largest argument <= 400;
  entries with z > 400 are set to zero.
* K_{i mu}(z): -pi Im I / sinh(pi mu) for z < 2; the integral
  int_0^7.2 exp(-z cosh u) cos(mu u) du by composite Gauss-Legendre
  (36 panels of width 0.2, 10 nodes each) for 2 <= z < 700; zero for z >= 700.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import loggamma

I_CUT = 400.0
K_SERIES_BELOW = 2.0
K_CUT = 700.0
_K_UMAX = 7.2
_K_PANEL = 0.2
_K_NPANEL = 36
_K_NODES = 10

_kquad = None


def _k_quadrature():
    """Nodes and weights of the composite Gauss-Legendre rule on [0, 7.2] (cached)."""
    global _kquad
    if _kquad is None:
        x, w = np.polynomial.legendre.leggauss(_K_NODES)
        us, ws = [], []
        for p in range(_K_NPANEL):
            a = p * _K_PANEL
            us.append(a + 0.5 * _K_PANEL * (x + 1.0))
            ws.append(0.5 * _K_PANEL * w)
        _kquad = (np.concatenate(us), np.concatenate(ws))
    return _kquad


def besseli_imag(mu, z):
    """I_{i mu}(z) for real z > 0 (complex array)."""
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    out = np.zeros(z.shape, dtype=complex)
    ok = z <= I_CUT
    if ok.any():
        zz = z[ok]
        zm = float(zz.max())
        mmax = int(math.floor(zm / 2.0 + 16.0 * math.sqrt(zm) + 50.0))
        nu = 1j * float(mu)
        half = zz / 2.0
        term = np.exp(nu * np.log(half) - loggamma(nu + 1.0))
        q = half * half
        s = term.copy()
        for m in range(1, mmax + 1):
            term = term * q / (m * (nu + m))
            s = s + term
        out[ok] = s
    return out[0] if scalar else out


def besselk_imag(mu, z):
    """K_{i mu}(z) for real z > 0 (real array)."""
    mu = float(mu)
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    out = np.zeros(z.shape, dtype=float)
    small = z < K_SERIES_BELOW
    if small.any():
        I = besseli_imag(mu, z[small])
        out[small] = -np.pi * I.imag / np.sinh(np.pi * mu)
    mid = (z >= K_SERIES_BELOW) & (z < K_CUT)
    if mid.any():
        u, w = _k_quadrature()
        cw = w * np.cos(mu * u)
        ch = np.cosh(u)
        zm = z[mid]
        # chunk to bound memory
        res = np.empty(zm.shape)
        step = 4096
        for i in range(0, zm.size, step):
            zc = zm[i:i + step]
            res[i:i + step] = np.exp(-np.outer(zc, ch)) @ cw
        out[mid] = res
    return out[0] if scalar else out
