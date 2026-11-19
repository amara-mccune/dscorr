"""Checks of the scalar-triangle series at general shape (examples/Loops/ext/scalartri.py).

Fast (about 20 s): the recurrence for the hard moments against Eq. (B.4); the coefficients of Eq. (5.6) on
k1 = k2; the exchange of the hard legs in (R, delta) and, as an identity that the expansion does not respect term
by term, in (r, c); the in-in assembly against M_tri and D_tri; the stored triangle integral of Ref. [26] at
delta = 0.5 and 1 against the series.
With scipy: the parameter integral over three-site chains by quadrature, at real orders.
"""
import json
import os
import sys

import mpmath as mp
import pytest

LOOPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops")
if not os.path.isdir(LOOPS):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, LOOPS)

from ext.blocks import H_moment  # noqa: E402
from ext.record import compare  # noqa: E402
from ext.scalartri import ScalarTriangle, correlator_family, exchanged_angle, to_symmetric  # noqa: E402

I = mp.mpc(0, 1)
NU = (I, I * mp.mpf("1.3"), I * mp.mpf("0.7"))
with open(os.path.join(LOOPS, "scalar_triangle_shape_record.json")) as _f:
    TABLE = json.load(_f)


def ok(key, value):
    return compare(key, value, TABLE, quiet=True)[0]


def test_recurrence_for_the_hard_moments():
    with mp.workdps(30):
        T = ScalarTriangle(*NU, Nmax=4)
        for m0, m1 in ((1, 0), (0, 1), (2, 1), (3, 3), (5, 0)):
            ref = H_moment(NU[0], 1 + NU[2] + m0, 1 + NU[1] + m1)
            assert abs(T.H(m0, m1) / ref - 1) < 1e-25
        # other vertex powers and real orders
        T = ScalarTriangle(mp.mpf("0.3"), mp.mpf("0.2"), mp.mpf("0.35"), r1=3, r2=mp.mpf("2.5"), Nmax=2)
        assert abs(T.H(2, 1) / H_moment(mp.mpf("0.3"), 2 + mp.mpf("0.35") + 2, mp.mpf("1.5") + mp.mpf("0.2") + 1) - 1) < 1e-25


def test_isosceles_coefficients():
    with mp.workdps(30):
        T = ScalarTriangle(*NU, Nmax=6)
        co, iso = T.coefficients(), T.isosceles(3)
        for n in range(3):
            assert ok(f"scalar_shape/c{n}", iso[n])
            assert abs(co[(2 * n, 0)] / iso[n] - 1) < 1e-25
        # odd orders vanish on k1 = k2, and the degree in delta is at most N with parity (-1)^N
        assert all((n - m) % 2 == 0 and m <= n for (n, m) in co)


def test_exchange_of_the_hard_legs():
    with mp.workdps(30):
        N = 8
        T, T2 = ScalarTriangle(*NU, Nmax=N), ScalarTriangle(NU[0], NU[2], NU[1], Nmax=N)
        co, co2 = T.coefficients(), T2.coefficients()
        scale = {n: max(abs(v) for (nn, m), v in co.items() if nn == n) for n in range(N + 1)}
        assert max(abs(v - (-1) ** k[1] * co2.get(k, 0)) / scale[k[0]] for k, v in co.items()) < 1e-23
        # the expansion in r = k3/k1 at a fixed angle treats the two hard legs differently
        ca, ca2 = T.coefficients_angle(), T2.coefficients_angle()
        expo = T.r1 + T.r2 + T.nuBC
        ex = exchanged_angle(ca, expo, N)
        sca = {n: max(abs(v) for (nn, m), v in ca2.items() if nn == n) for n in range(N + 1)}
        assert max(abs(ex.get(k, 0) - ca2.get(k, 0)) / sca[k[0]] for k in set(ex) | set(ca2)) < 1e-23
        back = to_symmetric(ca, expo, N)
        assert max(abs(back.get(k, 0) - co.get(k, 0)) / scale[k[0]] for k in set(back) | set(co)) < 1e-20


def test_exchange_with_unequal_vertex_powers():
    with mp.workdps(30):
        N = 5
        a = ScalarTriangle(NU[0], NU[1], NU[2], r1=2, r2=3, Nmax=N).coefficients()
        b = ScalarTriangle(NU[0], NU[2], NU[1], r1=3, r2=2, Nmax=N).coefficients()
        scale = max(abs(v) for v in a.values())
        assert max(abs(v - (-1) ** k[1] * b.get(k, 0)) for k, v in a.items()) < 1e-22 * scale


def test_in_in_assembly():
    with mp.workdps(30):
        mus = (mp.mpf(1), mp.mpf("1.3"), mp.mpf("0.7"))
        assert ok("scalar_shape/M_tri", correlator_family(mus, 1, 1, Nmax=0)[(0, 0)])
        assert ok("scalar_shape/D_tri", correlator_family(mus, 1, -1, Nmax=0)[(0, 0)])


def test_triangle_integral_at_general_shape():
    """The stored values of the Mellin-Barnes integral of Ref. [26] at delta = 0.5 and 1 against the series."""
    import scalar_triangle_shape as S
    with mp.workdps(30):
        N = 20
        T = ScalarTriangle(*NU, Nmax=N)
        co = T.coefficients()
        data = S.direct_data()
        deg = data["degrees"][-1]
        for tag, pts in data["shapes"].items():
            ser = S.at_delta(co, mp.mpf(tag), N)
            none = S.direct_fit(pts, [], 1 + T.nuBC, [deg])[deg][0]
            full = S.direct_fit(pts, ser, 1 + T.nuBC, [deg])[deg][0]
            assert full < none / 100
            assert ok(f"scalar_shape/direct residual delta={tag}", full)
            for Nf in (0, 1, 2):
                fit = S.direct_fit(pts, ser, 1 + T.nuBC, [deg], free=(Nf,))[deg][1][Nf]
                assert ok(f"scalar_shape/direct D{Nf} delta={tag}", abs(fit - 1))


def test_parameter_integral_by_quadrature():
    """Eq. (pairshape) of the write-up, at real orders: Gauss-Jacobi in lambda and in the ratio of the two times,
    generalized Gauss-Laguerre in the later time."""
    np = pytest.importorskip("numpy")
    sp = pytest.importorskip("scipy.special")
    nuA, nuB, nuC, R, dl = 0.3, 0.2, 0.35, 0.4, 0.5
    k1, k2, k3 = 1 + R * dl / 2, 1 - R * dl / 2, R
    nuBC = nuB + nuC
    x, w = sp.roots_jacobi(30, 0.5 + nuB, 0.5 + nuC)
    tot = 0.0
    for lam, wl in zip((1 + x) / 2, w / 2 ** (2 + nuBC)):
        a, b = lam * k3, (1 - lam) * k3
        y = np.sqrt((1 - lam) * k1 ** 2 + lam * k2 ** 2 - lam * (1 - lam) * k3 ** 2)
        val = 0.0
        for kT, kt, cT, ct, nuT, nut in ((k2, k1, b, a, nuB, nuC), (k1, k2, a, b, nuC, nuB)):
            rate, al = min(kT + y - cT, k1 + k2 - k3), 3 + nuBC
            u, wu = sp.roots_genlaguerre(80, al)
            s, ws = sp.roots_jacobi(40, 0.0, 1 + nut + nuA)
            s, ws = (1 + s) / 2, ws / 2 ** (2 + nut + nuA)
            TT, SS = np.meshgrid(u / rate, s, indexing="ij")
            t = TT * SS
            f = (TT * np.exp(-kT * TT) * sp.iv(nuT, cT * TT) * sp.kv(nuA, y * TT)
                 * TT * t * np.exp(-kt * t) * sp.iv(nut, ct * t) * sp.iv(nuA, y * t))
            f = f / SS ** (1 + nut + nuA) * np.exp(rate * TT) / (rate * TT) ** al / rate
            val += np.einsum("i,j,ij->", wu, ws, f)
        tot += wl * val / (lam ** nuC * (1 - lam) ** nuB)
    quad = np.pi ** 2 * sp.gamma(2 + nuBC) * sp.gamma(-1 - nuBC) * k3 * tot
    with mp.workdps(20):
        ser = ScalarTriangle(nuA, nuB, nuC, Nmax=14).family(k1, k2, k3)
        assert abs(quad / ser - 1) < 1e-7
