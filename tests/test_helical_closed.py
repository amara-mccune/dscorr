"""Checks of the soft vertex of the helical-vector loops (examples/Loops/ext/helclosed.py), about two minutes.

The explicit Kampe de Feriet formula for the simplest moment; single moments with one transverse and one longitudinal
line and with y^-2 against soft_J; two enhanced lines at kappa = mu; the Meijer form of ext/softclosed.py where it
applies; and the leading coefficient M_A of the gauge-boson bubble against its value of record.
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

import helical_closed as hc  # noqa: E402
from ext.helclosed import HelClosed  # noqa: E402
from ext.record import compare  # noqa: E402
from ext.softclosed import SoftClosed  # noqa: E402
from ext.vectri import SoftTable  # noqa: E402

I = mp.mpc(0, 1)
with open(os.path.join(LOOPS, "helical_closed_record.json")) as _f:
    TABLE = json.load(_f)
with open(os.path.join(LOOPS, "helical_triangle_record.json")) as _f:
    TRIANGLE = json.load(_f)


def ok(key, value, table=TABLE):
    return compare(key, value, table, quiet=True)[0]


def test_simplest_moment_is_a_kampe_de_feriet_series():
    assert ok("helical_closed/simplest moment, explicit formula", hc.simplest_check(dps=24))


def test_mixed_pairs_against_position_space():
    with mp.workdps(30):
        nu, K = I * mp.mpf(1), hc.labels("0.75")
        tab, H = SoftTable(nu, "0.75"), HelClosed(nu)
        for ia, ib, q, K1, K2 in ((0, 0, 1, "p", "z"), (0, -1, 2, "m", "z"), (0, 0, 1, "o", "p")):
            assert abs(H.J(ia, ib, q, K[K1], K[K2]) / tab(ia, ib, q, K1, K2) - 1) < 1e-24


def test_inverse_squares_against_position_space():
    # the reference averages two shifted powers and is good to 1e-20 at 30 digits
    with mp.workdps(30):
        nu, K = I * mp.mpf(1), hc.labels("0.75")
        tab, H = SoftTable(nu, "0.75"), HelClosed(nu)
        for ia, ib, q, K1, K2 in ((-2, 0, 1, "p", "p"), (-2, -2, 1, "m", "z"), (-2, 3, 2, "o", "z")):
            assert abs(H.J(ia, ib, q, K[K1], K[K2]) / tab(ia, ib, q, K1, K2) - 1) < 1e-17


def test_enhanced_pair_at_kappa_equal_mu():
    # c1 = c2 = 1/2: the exponents of the two lines coincide, and no limit is taken
    with mp.workdps(30):
        nu, K = I * mp.mpf(1), -I * mp.mpf(1)
        tab, H = SoftTable(nu, "1"), HelClosed(nu)
        d = max(abs(H.J(ia, ib, q, K, K) / tab(ia, ib, q, "m", "m") - 1) for ia, ib, q in ((0, 0, 1), (1, -1, 1)))
        assert ok("helical_closed/enhanced pair at kappa=mu vs soft_J", d)


def test_agrees_with_the_meijer_form():
    with mp.workdps(30):
        nu, K = I * mp.mpf(1), hc.labels("0.75")
        S, H = SoftClosed(nu, kmax=1), HelClosed(nu)
        for ia, ib, q, K1, K2 in ((0, 0, 1, "p", "p"), (-1, -1, 1, "p", "m"), (1, -1, 2, "z", "o")):
            assert abs(H.J(ia, ib, q, K[K1], K[K2]) / S.J(ia, ib, q, K[K1], K[K2]) - 1) < 1e-24


def test_unsupported_powers_are_refused():
    with mp.workdps(20):
        with pytest.raises(ValueError):
            HelClosed(I * mp.mpf(1)).J(-3, 0, 1, 0, 0)


def test_bubble_coefficient():
    with mp.workdps(30):
        L = hc.leading("1", "1")
        assert ok("helical_closed/M_A mu=1 kappa=1", L["M_A"])
        assert ok("leading/exact over late mu=1 kappa=1", L["exact_over_late"], TRIANGLE)
