"""Checks of the total-energy phase of the fermion loops at kappa >> m (examples/Loops/fermion_phase.py,
ext/fermicontact.py).

Fast: the closed forms of the model loop against the series built from its soft moments; the limits of the derived
forms; the series of the contact term at general shape.
Slow (about 3 min, RUN_SLOW=1): the identity triangle = contact term + triangle of two mass vertices at
kappa = 5, m = 1 through first order, and the first coefficient of the contact term against 1 + eps.
"""
import os
import sys

import mpmath as mp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops"))
import fermion_phase  # noqa: E402
from ext import fermicontact as fc  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="slow; set RUN_SLOW=1")
I = mp.mpc(0, 1)


def test_closed_forms_against_the_moments():
    worst_series, worst_elementary = fermion_phase.closed_form_checks()
    assert worst_series < 1e-30 and worst_elementary < 1e-30


def test_model_loop_by_quadrature():
    """The 2F1 form is the part of the Euler integral that is analytic at R = 0. With a damped exponent the integral
    converges, and at small R the other part, of order R^(r2 - 3 - r1 - 2 nu), is negligible."""
    with mp.workdps(25):
        nu, r1, r2, R = mp.mpc(-7, 3), 4, 2, mp.mpf("0.02")
        f = lambda R_: mp.quad(lambda xi: (xi * xi - 1) * (1 + xi) ** -(r2 - 2 * nu) * (1 + R_ * xi / 2) ** -(r1 + 2 * nu), [1, 2, 5, 20, mp.inf])
        ratio = f(R) / mp.mpf(2) ** (2 - (r2 - 2 * nu - 1)) * ((r2 - 2 * nu - 2) * (r2 - 2 * nu - 3))
        assert abs(ratio / fc.pair_closed(R, nu, r1, r2) - 1) < 1e-12


def test_limits_of_the_derived_forms():
    with mp.workdps(30):
        R = mp.mpf("0.3")
        big = mp.mpc(0, 10 ** 7)
        # kappa -> oo: the bubble is (1 + R/2)^(-2 nu) (1 - R^2/4)^-2, and the average of xi is 1 + i/kappa
        assert abs(fc.bubble_closed(R, big) * (1 + R / 2) ** (2 * big) * (1 - R * R / 4) ** 2 - 1) < 1e-6
        assert abs((fc.segment_average(1, 10 ** 5) - 1) * 10 ** 5 - I) < 1e-4
        # eps = 0: no correction
        assert fc.eps_factor(R, mp.mpc(0, 5), 5) == 1
        # first coefficients of the contact term: L1 = 1 + eps, L2 = 1/4 - 1/(2 nu) - eps/6
        ka = mp.mpf(10)
        nu = I * mp.sqrt(1 + ka ** 2)
        eps = nu - I * ka
        L = mp.taylor(lambda r: mp.log(fc.contact_first_order(r, nu, ka) * (1 + r / 2) ** (2 * nu)), 0, 2)
        assert abs(L[1] - (1 + eps)) < 1e-15 and abs(L[2] - (mp.mpf(1) / 4 - 1 / (2 * nu) - eps / 6)) < 1e-15


def test_series_tools():
    with mp.workdps(30):
        nu = mp.mpc(0, 3)
        c = mp.taylor(lambda r: 2 * (1 + r / 2) ** (-2 * nu) * mp.exp(mp.mpf("0.3") * r - r * r), 0, 5)
        L = fc.log_series(fc.strip_total_energy(c, nu))
        assert abs(L[1] - mp.mpf("0.3")) < 1e-25 and abs(L[2] + 1) < 1e-25 and max(abs(x) for x in L[3:]) < 1e-24


def test_contact_term_at_general_shape():
    """contact_symmetric against contact_coefficients, with made-up soft sums."""
    with mp.workdps(30):
        nu, m, ka = mp.mpc(0, 5.1), mp.mpf(1), mp.mpf(5)
        Sc = [mp.mpc(1, 0.5), mp.mpc(-0.3, 0.2), mp.mpc(0.1, -0.7), mp.mpc(0.4, 0.1), mp.mpc(-0.2, 0.3)]
        D = fc.contact_symmetric(Sc, nu, m, ka)
        assert all(md % 2 == 0 for (_, md) in D)
        on_axis = fc.contact_coefficients(Sc, nu, m, ka)
        assert max(abs(D[(N, 0)] - on_axis[N]) / abs(on_axis[N]) for N in range(5)) < 1e-25
        # at fixed x = R^2 delta^2/4 the two forms agree up to the truncation in R
        R, delta = mp.mpf("0.01"), mp.mpf("0.8")
        full = mp.fsum(cN * R ** N for N, cN in enumerate(fc.contact_coefficients(Sc, nu, m, ka, (R * delta / 2) ** 2)))
        series = mp.fsum(v * R ** N * delta ** md for (N, md), v in D.items())
        assert abs(series / full - 1) < 1e-8


@slow
def test_identity_and_contact_term():
    with mp.workdps(30):
        ka, m = mp.mpf(5), mp.mpf(1)
        T = fc.Contact("1", "5", nmax=1)
        nu = T.nu
        tri = T.isosceles(T.coefficients(T.accumulate()))
        con = fc.contact_coefficients(T.Sc(), nu, m, ka)
        t3, spread = T.t3_isosceles()
        assert spread < 1e-6
        assert max(abs(a - b - c) / abs(a) for a, b, c in zip(tri, con, t3)) < 1e-12
        record = mp.mpc("-1.063758421328667", "0.10831543805235125")              # C_0, twice the closed form of Song et al.
        assert abs(tri[0] / record - 1) < 1e-13
        L = fc.log_series(fc.strip_total_energy(con, nu))
        assert abs(L[1] - (1 + nu - I * ka)) < 0.05
        assert abs(t3[0] / con[0] + mp.mpf("0.347")) < 0.03
