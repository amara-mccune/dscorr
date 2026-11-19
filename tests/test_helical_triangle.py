"""Checks of examples/Loops/helical_triangle.py (the helical-vector triangle beyond the leading squeezed term).

    pytest tests/test_helical_triangle.py            # fast tier: the polynomial algebra (seconds)
    pytest -m slow tests/test_helical_triangle.py    # also the leading order at mu = kappa = 1 (about 3 min)
"""
import math
import os
import random
import sys

import mpmath as mp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops"))
from ext import vectri as V  # noqa: E402


def _ev(p, r, q, S, c, y1, y2):
    return sum(complex(v) * r ** n * q[0] ** mx * q[1] ** my * q[2] ** mz * S ** ms * c ** mc * y1 ** da * y2 ** db
               for (n, mx, my, mz, ms, mc, da, db), v in p.items())


def test_reduction_against_angular_integration():
    """Azimuthal average and reduction to monomials in y1, y2, against a direct sum over the azimuth."""
    mp.mp.dps = 30
    random.seed(1)
    nmax = 3
    X = [[V.padd(*[(1, V.pmul(V.proj_T(1, V.N1, nmax)[i][j], V.proj_T(-1, V.N2, nmax)[i][k], nmax)) for i in range(3)])
          for k in range(3)] for j in range(3)]
    for comp in V._hard_components("s", "s", nmax)[:3]:
        P = V.pmul(V.pmul(V.Triangle._contract(X, comp[1], nmax), V.USER, nmax), V.USER, nmax)
        red = V.reduce_poly(P)
        for _ in range(3):
            y1 = random.uniform(0.3, 2.0)
            y2 = random.uniform(abs(1 - y1) + 0.05, 1 + y1 - 0.05)
            c = random.uniform(-0.9, 0.9)
            S, r = math.sqrt(1 - c * c), 0.37
            qz = (1 + y1 * y1 - y2 * y2) / 2
            rho = math.sqrt(y1 * y1 - qz * qz)
            direct = sum(_ev(P, r, (rho * math.cos(2 * math.pi * k / 64), rho * math.sin(2 * math.pi * k / 64), qz), S, c, y1, y2)
                         for k in range(64)) / 64
            reduced = sum(complex(v) * r ** n * c ** mc * y1 ** da * y2 ** db for (n, mc, da, db), v in red.items())
            assert abs(direct - reduced) <= 1e-9 * max(1e-12, abs(direct))


def test_same_helicity_overlap():
    """Trace of the soft tensor: |eps^h1 . eps^h2|^2 = (1 - h1 h2 c)^2/4."""
    mp.mp.dps = 30
    y1, y2 = 0.8, 0.9
    c = (1 - y1 * y1 - y2 * y2) / (2 * y1 * y2)
    for h1, h2 in ((1, 1), (1, -1)):
        X = [[V.padd(*[(1, V.pmul(V.proj_T(h1, V.N1, 0)[i][j], V.proj_T(h2, V.N2, 0)[i][k], 0)) for i in range(3)])
              for k in range(3)] for j in range(3)]
        tr = V.reduce_poly(V.padd(*[(1, X[j][j]) for j in range(3)]))
        val = sum(complex(v) * y1 ** da * y2 ** db for (n, mc, da, db), v in tr.items())
        assert abs(val - (1 - h1 * h2 * c) ** 2 / 4) < 1e-13


@pytest.mark.slow
def test_leading_order():
    """M_A and the monopole of the triangle at mu = kappa = 1 (Secs. 7.2 and 7.3)."""
    mp.mp.dps = 30
    T = V.Triangle(1, 1, nmax=0)
    Mb = T.coefficients("bubble")[0]
    res = T.coefficients("triangle", isosceles=False)
    mono = res[(0, 0)] + res[(0, 2)] / 3
    R = -T.hardline.m2 * mono / Mb
    assert abs(Mb - mp.mpc("0.00266561645095246342", "0.000402918297821133207")) < 1e-12 * abs(Mb)
    assert abs(abs(R) - mp.mpf("0.79163004345639")) < 1e-11
    assert abs(res.get((0, 1), 0)) < 1e-20 * abs(mono)


@pytest.mark.slow
def test_general_shape_exchange():
    """General shape at mu = kappa = 1 through first order: the symmetry k1 <-> k2 (which relates M_1 to M_0), and
    the isosceles limit against the same code with k1 = k2."""
    from ext import shapevars
    mp.mp.dps = 30
    T = V.Triangle(1, 1, nmax=1, general=True)
    C = T.coefficients("triangle", isosceles=False)
    assert max(shapevars.exchange_error(C, T.nu, 1, dim=2).values()) < 1e-22
    D = shapevars.symmetric(C, T.nu, 1, dim=2)
    iso = V.Triangle(1, 1, nmax=1).coefficients("triangle")
    assert max(abs(shapevars.at_delta(D, 0, 1)[n] / iso[n] - 1) for n in range(2)) < 1e-22
    assert max((abs(v) for (n, md), v in D.items() if md % 2), default=0) < 1e-22 * abs(iso[0])
