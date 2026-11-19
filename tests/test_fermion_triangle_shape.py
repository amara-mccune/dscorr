"""Checks of the fermion-triangle series (examples/Loops/ext/fermitri.py, ext/shapevars.py).

Fast: the closed form of the ordered hard-line moment against quadrature; the exchange map is an involution;
the symmetric variables reproduce a function that is symmetric by construction.
Slow (about 2 min, `-m slow` or RUN_SLOW=1): the leading coefficient against the value of record (twice the
closed form of Song et al.), its independence of the angle, and the exchange identity at first order.
"""
import os
import sys

import mpmath as mp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops"))
from ext import shapevars  # noqa: E402
from ext.fermitri import FermiTriangle, ordered_moment  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="slow; set RUN_SLOW=1")


def test_ordered_moment_against_quadrature():
    with mp.workdps(20):
        nu, ka = mp.mpc(0, 1.64), mp.mpf(1.3)
        a, b, Km, Kw = nu, nu + 1, 0.5 + 1j * ka, -0.5 + 1j * ka
        closed = ordered_moment(a, b, Km, nu, Kw, nu)
    with mp.workdps(15):
        inner = lambda t2: mp.quad(lambda t1: t1 ** a * mp.exp(-t1) * mp.whitm(Km, nu, 2 * t1), [0, t2])
        num = mp.quad(lambda t2: t2 ** b * mp.exp(-t2) * mp.whitw(Kw, nu, 2 * t2) * inner(t2), [0, 1, 4, 12, 40, 100])
    assert abs(closed / num - 1) < 1e-10


def test_exchange_is_an_involution():
    with mp.workdps(30):
        nu, N = mp.mpc(0, 1.3), 4
        coef = {(n, mc): mp.mpc(mp.rand() - 0.5, mp.rand() - 0.5) for n in range(N + 1) for mc in range(n + 2)}
        twice = shapevars.exchanged(shapevars.exchanged(coef, nu, N), nu, N)
        assert max(abs(twice.get(k, 0) - v) for k, v in coef.items()) < 1e-24


def test_symmetric_variables_on_a_symmetric_function():
    """F = K^-6 R^(2 nu) (1 + a R + b R^2 delta^2) written in (r, c) must come back with even powers of delta only."""
    with mp.workdps(30):
        nu, N, a, b = mp.mpc(0, 0.9), 3, mp.mpf("0.7"), mp.mpf("-0.4")
        # inverse map to first orders: R = 2 r/(1 + x2), delta = (1 - x2)/r, K = (1 + x2)/2, x2 = sqrt(1 + 2 r c + r^2)
        f = lambda r, c: ((1 + mp.sqrt(1 + 2 * r * c + r * r)) / 2) ** -6 * (2 / (1 + mp.sqrt(1 + 2 * r * c + r * r))) ** (2 * nu) \
            * (1 + a * 2 * r / (1 + mp.sqrt(1 + 2 * r * c + r * r)) + b * (2 * (1 - mp.sqrt(1 + 2 * r * c + r * r)) / (1 + mp.sqrt(1 + 2 * r * c + r * r))) ** 2)
        # Taylor coefficients in r at several c, then a polynomial fit in c of degree n + 1
        cs = [mp.mpf(k) / 7 - mp.mpf(1) / 2 for k in range(N + 3)]
        with mp.workdps(60):
            tay = {c: mp.taylor(lambda r: f(r, c), 0, N) for c in cs}
        coef = {}
        for n in range(N + 1):
            deg = n + 1
            pts = cs[:deg + 1]
            sol = mp.lu_solve(mp.matrix([[c ** k for k in range(deg + 1)] for c in pts]), mp.matrix([tay[c][n] for c in pts]))
            for k in range(deg + 1):
                coef[(n, k)] = sol[k]
        D = shapevars.symmetric(coef, nu, N)
        assert abs(D[(0, 0)] - 1) < 1e-20 and abs(D[(1, 0)] - a) < 1e-18 and abs(D[(2, 2)] - b) < 1e-16
        assert max(abs(v) for (n, md), v in D.items() if md % 2) < 1e-16
        assert max(shapevars.exchange_error(coef, nu, N).values()) < 1e-16


@slow
def test_leading_coefficient_and_first_order():
    with mp.workdps(30):
        T = FermiTriangle(1, 1.3, nmax=1)
        C = T.coefficients(T.accumulate())
        song = mp.mpc("-0.000610433622423581018", "0.000151140525456825077")        # 2 kappa^3 Lambda^3 C_+ of Song et al.
        assert abs(C[(0, 0)] / song - 1) < 1e-13
        assert abs(C.get((0, 1), 0) / C[(0, 0)]) < 1e-25                              # no dependence on the angle
        assert abs(C[(1, 1)] / (-(3 + T.nu) * C[(0, 0)]) - 1) < 1e-22                # exchange identity at first order
        assert max(shapevars.exchange_error(C, T.nu, 1).values()) < 1e-22


def test_spin_traces_against_direct_azimuthal_average():
    """The reduced polynomials of tr[S1 Pi_A S2 Pi_B S3 Pi_C] against 2x2 matrices averaged over the azimuth.
    Through r^2 the polynomials are exact (the hard projector and sigma.k2 are linear in r)."""
    import cmath
    import math
    import random
    from itertools import product
    with mp.workdps(20):
        T = FermiTriangle(1, 1.3, nmax=2)
        Tr, _ = T._traces()
    sig = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
    one = [[1, 0], [0, 1]]

    def mat(s, v):
        return [[s * one[i][j] + sum(v[k] * sig[k][i][j] for k in range(3)) for j in range(2)] for i in range(2)]

    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]

    rng = random.Random(3)
    for _ in range(2):
        r, c = rng.uniform(0.05, 0.4), rng.uniform(-0.9, 0.9)
        y1 = rng.uniform(0.4, 1.5)
        y2 = rng.uniform(abs(1 - y1) + 0.05, 1 + y1 - 0.05)
        S = math.sqrt(1 - c * c)
        qz = (1 + y1 * y1 - y2 * y2) / 2
        rho = math.sqrt(y1 * y1 - qz * qz)
        k1 = (S, 0.0, c)
        for s1, s2, s3, pC, pB, pA in product("1k", "1k", "1k", (0, 1), (0, 1), (0, 1)):
            tot, nphi = 0, 16
            for i in range(nphi):
                ph = 2 * math.pi * (i + 0.5) / nphi
                q = (rho * math.cos(ph), rho * math.sin(ph), qz)
                P = tuple(k1[a] + r * q[a] for a in range(3))
                k2 = tuple(-(k1[a] + (r if a == 2 else 0)) for a in range(3))
                S1 = one if s1 == "1" else mat(0, k1)
                S2 = one if s2 == "1" else mat(0, k2)
                S3 = one if s3 == "1" else mat(0, (0, 0, 1))
                PA = mat(0.5, (0, 0, 0)) if pA == 0 else mat(0, tuple(x / 2 for x in P))
                PB = mat(0.5, (0, 0, 0)) if pB == 0 else mat(0, tuple((q[a] - (1 if a == 2 else 0)) / (2 * y2) for a in range(3)))
                PC = mat(0.5, (0, 0, 0)) if pC == 0 else mat(0, tuple(x / (2 * y1) for x in q))
                M = mul(mul(mul(mul(mul(S1, PA), S2), PB), S3), PC)
                tot += M[0][0] + M[1][1]
            direct = tot / nphi
            red = sum(complex(v) * r ** n * c ** mc * y1 ** da * y2 ** db for (n, mc, da, db), v in Tr[(s1, s2, s3, pC, pB, pA)][0].items())
            assert abs(direct - red) < 1e-12, (s1, s2, s3, pC, pB, pA, direct, red)
