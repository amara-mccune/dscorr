"""The one-line block psi_2(x1, x2, y, nu) in three representations.

Auxiliary research module; not called by the tree engines.

    psi_2 = int dt1 dt2 (t1 t2)^(-1/2) e^(-x1 t1 - x2 t2) K_nu(y t_>) I_nu(y t_<).
"""
from __future__ import annotations

import mpmath as mp


def _H(nu, w):
    eps = mp.eps
    s = 0
    for m in range(4000):
        term = mp.gamma(mp.mpf(1) / 2 + nu + 2 * m) / (mp.factorial(m) * mp.gamma(1 + nu + m)) * w ** m
        s += term
        if m >= 5 and abs(term) < eps * abs(s):
            break
    return s


def psi2_series(x1, x2, y, nu, dps=30):
    """Residue series, valid for y < x1 < x2."""
    with mp.workdps(dps):
        x1, x2, y = mp.mpf(x1), mp.mpf(x2), mp.mpf(y)
        nu = mp.mpmathify(nu)
        if not (y < x1 < x2):
            raise ValueError("psi2_series: needs y < x1 < x2")
        eps = mp.eps
        half = mp.mpf(1) / 2
        w1, w2 = y ** 2 / (4 * x1 ** 2), y ** 2 / (4 * x2 ** 2)
        s = mp.pi / (2 * mp.sin(mp.pi * nu))
        TAA = -s * (y / 2) ** (2 * nu) * x1 ** (-half - nu) * x2 ** (-half - nu) * _H(nu, w1) * _H(nu, w2)
        TB1 = s * x1 ** (-half + nu) * x2 ** (-half - nu) * _H(-nu, w1) * _H(nu, w2)
        TB3 = 0
        for n in range(200):
            inner = 0
            for k in range(400):
                term = ((-1) ** k / mp.factorial(k)
                        * mp.gamma(mp.mpf(1) / 4 + nu / 2 + mp.mpf(k) / 2)
                        * mp.gamma(mp.mpf(1) / 4 - nu / 2 + mp.mpf(k) / 2)
                        * mp.gamma(1 + 2 * n + k)
                        / (mp.gamma(mp.mpf(5) / 4 + n + mp.mpf(k) / 2 - nu / 2)
                           * mp.gamma(mp.mpf(5) / 4 + n + mp.mpf(k) / 2 + nu / 2))
                        * x1 ** k * x2 ** (-1 - 2 * n - k))
                inner += term
                if k >= 10 and abs(term) < eps * abs(inner):
                    break
            tn = (y / 2) ** (2 * n) * inner
            TB3 += tn
            if n >= 5 and abs(tn) < eps * abs(TB3):
                break
        TB3 = -TB3 / 4
        return +(TAA + TB1 + TB3)


def psi2_time(x1, x2, y, nu, dps=15):
    """Nested adaptive quadrature over the two times (slow: ~1 min at 15 digits)."""
    with mp.workdps(dps):
        x1, x2, y = mp.mpf(x1), mp.mpf(x2), mp.mpf(y)
        nu = mp.mpmathify(nu)

        def outer(t2):
            a = mp.besselk(nu, y * t2) * mp.quad(
                lambda t1: t1 ** -0.5 * mp.exp(-x1 * t1) * mp.besseli(nu, y * t1), [0, t2])
            b = mp.besseli(nu, y * t2) * mp.quad(
                lambda t1: t1 ** -0.5 * mp.exp(-x1 * t1) * mp.besselk(nu, y * t1), [t2, mp.inf])
            return t2 ** -0.5 * mp.exp(-x2 * t2) * (a + b)
        return mp.quad(outer, [0, 1, mp.inf])


def psi2_contour(x1, x2, y, nu, c=None, dps=15):
    """The double Mellin-Barnes integral on Re s = Re s' = c (slow two-dimensional quadrature)."""
    with mp.workdps(dps):
        x1, x2, y = mp.mpf(x1), mp.mpf(x2), mp.mpf(y)
        nu = mp.mpmathify(nu)
        if c is None:
            c = -mp.mpf(1) / 4 - mp.re(nu) / 2 + mp.mpf("0.05")

        def f(u, v):
            s = c + 1j * u
            sp = c + 1j * v
            b = 1 + nu + s + sp
            return (mp.gamma(-s) * mp.gamma(-sp) * mp.gamma(b) * mp.gamma(1 - b)
                    / (mp.gamma(nu + s + 1) * mp.gamma(nu + sp + 1)) * (y / 2) ** (2 * b - 2)
                    * mp.gamma(mp.mpf(1) / 2 + nu + 2 * s) * mp.gamma(mp.mpf(1) / 2 + nu + 2 * sp)
                    * x1 ** (-(mp.mpf(1) / 2 + nu + 2 * s)) * x2 ** (-(mp.mpf(1) / 2 + nu + 2 * sp)))
        val = mp.quad(f, [-mp.inf, 0, mp.inf], [-mp.inf, 0, mp.inf])
        return val / (2 * (2 * mp.pi) ** 2)
