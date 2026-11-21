"""Checks of the closed-form soft vertex (examples/Loops/ext/softclosed.py), about two minutes.

Scalar lines against Eq. (3.17); the simplest moment as one Barnes integral; single moments of the four kinds of
Dirac pair, of a helical pair, and of a longitudinal pair against soft_J; and the squeezed coefficient C of the
fermion triangle at kappa = 1.3 against its value of record (which equals twice the closed form of Song et al.).
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

import soft_closed as sc  # noqa: E402
from ext.record import compare  # noqa: E402
from ext.softclosed import ClosedSoftTable, SoftClosed, soft_J_closed_limit  # noqa: E402
from ext.softvertex import Grid, Leg, soft_J  # noqa: E402

I = mp.mpc(0, 1)
with open(os.path.join(LOOPS, "soft_closed_record.json")) as _f:
    TABLE = json.load(_f)


def ok(key, value):
    return compare(key, value, TABLE, quiet=True)[0]


def _soft(ia, ib, q, nu, K1, K2):
    return soft_J(Leg(nu - mp.mpf(1) / 2 + ia, K1, nu), Leg(nu - mp.mpf(1) / 2 + ib, K2, nu), q, Grid())


def test_scalar_lines_reduce_to_gamma_functions():
    with mp.workdps(30):
        assert ok("soft_closed/scalar vs Eq. (3.17)", sc.scalar_check())


def test_simplest_moment_is_one_barnes_integral():
    with mp.workdps(30):
        ka = mp.mpf("1.3")
        nu = I * mp.sqrt(1 + ka ** 2)
        assert ok("soft_closed/one Barnes integral, Dirac", sc.simplest_moment(nu, -0.5 + I * ka, SoftClosed(nu, kmax=ka)))
        nu = I * mp.mpf(1)
        assert ok("soft_closed/one Barnes integral, helical", sc.simplest_moment(nu, 0.75 * I, SoftClosed(nu, kmax=1)))


def test_dirac_moments_against_position_space():
    with mp.workdps(30):
        worst = sc.moments_check("1.3", 1, moments=[(-1, -1, 1), (0, 0, 1), (-1, 2, 0)])
        assert ok("soft_closed/moments vs soft_J kappa=1.3,m=1", worst)


def test_helical_and_longitudinal_moments():
    with mp.workdps(30):
        nu = I * mp.mpf(1)
        K = {"p": 0.75 * I, "m": -0.75 * I, "z": mp.mpf(0), "o": mp.mpf(1)}
        S = SoftClosed(nu, kmax=1)
        for K1, K2, ia, ib, q in (("p", "p", 0, 0, 2), ("p", "m", -1, -1, 1), ("z", "o", 1, -1, 3)):
            assert abs(S.J(ia, ib, q, K[K1], K[K2]) / _soft(ia, ib, q, nu, K[K1], K[K2]) - 1) < 1e-22


def test_coincident_poles_at_kappa_equal_mu():
    with mp.workdps(32):
        nu = I * mp.mpf(1)
        K = -I * mp.mpf(1)                         # c = 1/2 - nu - K = 1/2 is real
        with pytest.raises((ValueError, ZeroDivisionError)):
            SoftClosed(nu, kmax=1).J(-1, -1, 1, K, K)
        val = soft_J_closed_limit(-1, -1, 1, nu, K, K)
        assert abs(val / _soft(-1, -1, 1, nu, K, K) - 1) < 1e-12


def test_unsupported_pairs_are_refused():
    with mp.workdps(20):
        nu = I * mp.mpf(1)
        with pytest.raises(ValueError):
            SoftClosed(nu, kmax=1).J(0, 0, 1, 0.75 * I, 0)          # one transverse and one longitudinal line
        with pytest.raises(ValueError):
            SoftClosed(nu, kmax=1).J(-2, 0, 1, 0.75 * I, 0.75 * I)  # a monomial with y^(-2)


def test_fermion_triangle_coefficient():
    with mp.workdps(30):
        ka = mp.mpf("1.3")
        nu = I * mp.sqrt(1 + ka ** 2)
        C = sc.triangle_C(ka, mp.mpf(1), ClosedSoftTable(nu, sc.dirac_K(ka)))
        assert ok("soft_closed/C kappa=1.3,m=1", C)
