"""The collapsed series of the helical loop (examples/Loops/parity_box_series.py): minute-scale checkpoints.

Fast tier (under a minute): orders N <= 1 at (mu, kappa) = (1, 1) against parity_box.py and against the exchange
identities, and the vanishing of the single-frequency families at kappa = 0. The slow tier repeats the checks of
the script at N = 2 and compares with the values of record at N = 4.
"""
import os
import sys

import mpmath as mp
import pytest

LOOPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops")
if not os.path.isdir(LOOPS):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, LOOPS)

import parity_box_series as PBS  # noqa: E402
from ext.boxseries import harmonics  # noqa: E402
from ext.record import compare, load  # noqa: E402

TABLE = load()


@pytest.fixture(scope="module")
def first_order():
    with mp.workdps(25):
        return PBS.Collapsed(1, 1, 1, families=PBS.OSC)


def test_leading_order_is_parity_box(first_order):
    with mp.workdps(25):
        assert PBS.check_leading(first_order) < 1e-18


def test_exchange_identities(first_order):
    """Exchange of the sides, and of the legs within a side (which mixes the orders 0 and 1)."""
    with mp.workdps(25):
        assert PBS.check_sides(first_order) < 1e-18
        for fam in PBS.OSC:
            assert PBS.check_exchange(first_order, fam) < 1e-18


def test_single_frequency_families():
    """They vanish for a Proca field (kappa = 0) and not for a helical one; the box has them at leading order,
    the bubble only from the first order on."""
    with mp.workdps(25):
        C0 = PBS.Collapsed(1, 0, 1, families=PBS.OSC)
        odd, single = PBS.check_parity(C0)
        assert odd < 1e-18 and single < 1e-18
        C = PBS.Collapsed(1, 1, 1, families=("singleA",), diagrams=("CC", "PP"))
        pp = harmonics(C.coefs[("singleA", "PP")], 0, 0)
        cc = harmonics(C.coefs[("singleA", "CC")], 0, 0)
        assert abs(pp[(0, 0)][0]) > 1e-3
        assert abs(cc[(0, 0)].get(0, 0)) < 1e-18 and abs(cc[(0, 1)][0]) > 1e-3


def test_crossed_box_cancels_odd_harmonics(first_order):
    """In the configuration k1 = k2 = k3 = k4 the crossed box is the planar one with beta -> beta + pi."""
    with mp.workdps(25):
        planar = first_order.equal_k("pair", "PP", crossed=False)
        both = first_order.equal_k("pair", "PP")
        assert abs(planar[-1][0]) > 1e-4
        assert all(abs(x) < 1e-20 for n in (-1, 1) for x in both[n])
        assert abs(both[-2][0] / planar[-2][0] - 2) < 1e-18


@pytest.mark.slow
def test_script_checks_second_order():
    with mp.workdps(25):
        C = PBS.Collapsed(1, 1, 2, families=PBS.OSC)
        Cm = PBS.Collapsed(1, -1, 2, families=PBS.OSC)
        assert PBS.check_reflection(C, Cm) < 1e-18
        hv = PBS.check_heavy(1, 1, 2, families=("pair",))
        assert hv[40][1] < 0.02 and 3.5 < hv[20][1] / hv[40][1] < 4.5
        assert PBS.check_precision(1, 1, 2, C) < 1e-18


@pytest.mark.slow
def test_values_of_record_fourth_order():
    with mp.workdps(25):
        C = PBS.Collapsed(1, 1, 4, families=PBS.OSC)
        full = C.equal_k_full("pair")
        for n in (-2, 0, 2):
            for k in range(5):
                assert compare(f"parity_box_series/mu=1 kappa=1 pair c_{n:+d} N={k}", full[n][k], TABLE, quiet=True)[0]
        single = PBS.single_full(C)
        for k in range(5):
            assert compare(f"parity_box_series/mu=1 kappa=1 single c_+0 N={k}", single[0][k], TABLE, quiet=True)[0]
