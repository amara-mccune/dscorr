"""Scripts of the loops paper (examples/Loops): minute-scale checkpoints.

The scripts are imported from examples/Loops (they are not library calls). Reference values are the
values of record in examples/Loops/values_of_record.json, i.e. the numbers quoted in the paper.
"""
import os
import sys

import mpmath as mp
import pytest

LOOPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops")
if not os.path.isdir(LOOPS):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, LOOPS)

from ext.blocks import two_moment  # noqa: E402
from ext.record import compare, load  # noqa: E402
from ext.softvertex import Grid, Leg, soft_J  # noqa: E402

I = mp.mpc(0, 1)
TABLE = load()


def ok(key, value):
    return compare(key, value, TABLE, quiet=True)[0]


# ------------------------------------------------------------------------------------- Sec. 6.2
def test_universal_factor():
    import universal as U
    with mp.workdps(30):
        for mu in (0.5, 2):
            nu = I * mu
            assert abs(U.ratio_from_moments(nu, 1, 2) / U.U(3 + 2 * nu, 2) - 1) < 1e-25
        assert abs(U.U_position(3 + 2j, 3.5) / U.U(3 + 2j, 3.5) - 1) < 1e-12
        for mu in ("0.5", "1", "2", "4"):
            assert ok(f"universal/|U| heavy mu={mu}", abs(U.U_heavy(mu)))
        assert ok("universal/U light nu=0.3", U.U_light("0.3"))


# ------------------------------------------------------------------------------------- soft vertex
def test_position_space_soft_vertex_K0():
    """At K = 0, W_{0,nu}(2x) = sqrt(2x/pi) K_nu(x), so J reduces to the closed form [eq:twomoment]."""
    with mp.workdps(25):
        nu = I
        for s1, s2, q in ((1, 1, 1), (-1, -1, 1), (1, 1, 3.5)):
            J = soft_J(Leg(s1 * nu - 0.5, 0, nu), Leg(s2 * nu - 0.5, 0, nu), q, Grid())
            ref = (2 / mp.pi) / (2 * mp.pi) ** 3 * mp.mpf(2) ** ((s1 + s2) * nu) * two_moment(nu, nu, s1, s2, q + 1)
            assert abs(J / ref - 1) < 1e-20


# ------------------------------------------------------------------------------------- Sec. 7.5
def test_charged_scalars():
    import charged_scalars as CS
    with mp.workdps(50):
        assert CS.check_late(2.5) < 1e-12
        B = CS.ChargedBubble(1, 3, 10)
        assert abs(B.family(-1, -1, -1) - mp.conj(B.family(1, 1, 1))) < mp.mpf(10) ** -40
        q = CS.ChargedBubble(4, 4, 40).f() / CS.f_bbsx(4, 4, 40)
        assert ok("charged/|ratio| (40,4)", abs(q))
        assert ok("charged/|ratio| (10,5.1)", abs(CS.ChargedBubble(5.1, 5.1, 10).f() / CS.f_bbsx(5.1, 5.1, 10)))
        assert ok("charged/|R| mu=2", abs(CS.ChargedBubble(2, 2, 4).residue()))


# ------------------------------------------------------------------------------------- Sec. 7.4
def test_parity_box_chirality():
    import parity_box as PB
    with mp.workdps(25):
        B = PB.CollapsedBox(1, 1)
        c = B.harmonics()
        assert ok("parity_box/|c_-2/c_+2| mu=1 kappa=1", abs(c[-2] / c[2]))
        assert abs(c[3]) / abs(c[0]) < 1e-15
        full = B.harmonics(weights=(B.cc, B.cp))
        assert ok("parity_box/odd fraction mu=1 kappa=1", PB.odd_fraction(full))


@pytest.mark.slow
def test_parity_box_hard_moments():
    import parity_box as PB
    assert PB.check_hard_moments(points=((1, 0.75),), dps=18) < 1e-11


# ------------------------------------------------------------------------------------- Sec. 8.2
def test_dirac_pair_S0():
    import dirac_pair as DP
    with mp.workdps(30):
        wn, wa, z = DP.line_checks()
        assert wn < 1e-20 and wa < 1e-20 and z < 1e-25
        P = DP.DiracPair(1, 1)
        assert ok("dirac_pair/S(0) m=k=1", P.S(0))
        assert ok("dirac_pair/|exact/late pair| (1,1)", abs(P.S(0) / P.S(0, mode='late')))


def test_han_fig8_point():
    """Han, Lei, Qin and Yin (2611.20919), Fig. 8 at M = 5, lam = 1 (about 40 s)."""
    import json
    import han_check as HC
    fig = json.load(open(os.path.join(LOOPS, "han_fig8.json")))
    with mp.workdps(30):
        amp, ph = HC.HanBubble(5, 1).amplitude_phase()
    assert abs(float(mp.log10(amp)) - fig["right"]["5"]["1"]) < 2e-3


# ------------------------------------------------------------------------------------- Sec. 8.2 (triangle)
@pytest.mark.slow
def test_fermion_triangle_kappa13():
    """kappa = 1.3, m = 1 on a coarse grid (about 3 min)."""
    import fermion_triangle as FT
    out = FT.run_point(1.3, 1, theta=0.6, dps_hard=25, dps_soft=30,
                       segments=((-32, 1.5, 0.5), (1.5, 5.0, 0.15)), n=24, verbose=False)
    assert ok("fermion_triangle/C exact kappa=1.3,m=1", out["exact"])
    assert ok("fermion_triangle/C nl kappa=1.3,m=1", out["nl"])
