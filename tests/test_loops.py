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


# ------------------------------------------------------------------- two-vertex loops at every shape
def test_bubble_single_sums():
    """ext/bubble.py: the tree, the closed form of the cut, and the lattice values of S_0."""
    from ext.bubble import PATTERNS, Ihat, Khat, Trees, tree_remainder
    with mp.workdps(30):
        # ordered remainder of one tree against the double series of Eq. (3.9), at complex vertex powers
        nu, rho1, rho2 = mp.mpc(0, '0.8'), mp.mpc('1.5', '-2'), mp.mpc('0.5', '2')
        k, x1, x2 = mp.mpf(1), mp.mpf('2.7'), mp.mpf('1.4')
        R, dbl = rho1 + rho2, 0
        for l in range(90):
            prod = 1
            for N in range(60):
                prod *= nu ** 2 - (rho2 + l + 2 * N) ** 2
                dbl += (-1) ** (N + l) * mp.gamma(R + l + 2 * N) / mp.factorial(l) * k ** (2 * N) * x1 ** (-(R + l + 2 * N)) * x2 ** l / prod
        assert abs(tree_remainder(nu, rho1, rho2, k, x1, x2) / dbl - 1) < mp.mpf(10) ** -18
        # cut combination of the tree sums: closed form (also at a double pole)
        tr = Trees(mp.mpc(0, '1.3'), mp.mpc(0, '0.7'))
        p0, p0p = mp.mpc(-1.5, -3), mp.mpc(-1.5, 3)
        for z in (mp.mpc(0.5, 3), mp.mpc(1.5, 3), mp.mpc(7.5, 3)):
            num = sum(e[0] * e[1] * tr.tsum(e, z, (p0, p0p)) for e in PATTERNS)
            assert abs(num / complex(tr.cut_sum(z, (p0, p0p))) - 1) < 1e-12
        # S_0 at equal masses on the lattice
        import higgs_portal as HP
        lat = HP.S0_lattice_equal(mp.mpc(0, 2), 8)
        tr2 = Trees(mp.mpc(0, 2), mp.mpc(0, 2))
        for m in (0, 1, 2, 5, 8):
            assert abs(tr2.S0((1, 1), 0.5 + m) / complex(lat[m]) - 1) < 1e-10


def test_charged_scalars_shape():
    import charged_scalars as CS
    half = mp.mpf(1) / 2
    S, B = CS.ChargedShape('1', '3', '10'), CS.ChargedBubble('1', '3', '10')
    with mp.workdps(S.dps):
        for e in CS.PATTERNS:
            assert abs(S.family_leading(e) / B.family(1, e[0], e[1]) - 1) < mp.mpf(10) ** -30
    S = CS.ChargedShape('0.7', '1.1', '2')
    with mp.workdps(S.dps):
        Xt, Xs = S.X(half, half, '0.3'), CS.spectral_X(S, half, half, '0.3')
        assert abs(Xt - Xs) / abs(Xt) < 1e-10
    S = CS.ChargedShape('1', '1', '10')
    assert ok("charged_shape/F channel (10,1) u=0.3", S.channel(half, half, '0.3')['F'])
    assert ok("charged_shape/|sum line/template| (10,1) u=0.3",
              abs(CS.sumline_ratio(S, '0.3') * mp.conj(S.family_leading((1, 1))) / CS.f_bbsx(1, 1, 10) / (1 + 4 * mp.mpf('0.3') ** 3)))


def test_higgs_portal():
    import higgs_portal as HP
    from universal import U
    with mp.workdps(20):
        H = HP.HiggsPortal(I * 1)
        a, b = HP.block_trees(H, 1, 2, mp.mpf(2)), HP.zform_block(I * 1, 2, mp.mpf(2))
        assert abs(a / b - 1) < 1e-12
    with mp.workdps(30):
        for nu in (I * 2, mp.mpf('1.4')):
            H = HP.HiggsPortal(nu, mmax=4)
            for eps in (1, -1):
                ours = H.leading(eps, 1, 1, mp.mpf(10) ** -6) / U(3 + 2 * eps * H.nu, 2) / (2 * mp.mpf(10) ** -18)
                cwx = -1 / (8 * mp.pi ** 4) * HP.C_H(-eps * H.nu) * (mp.mpf(10) ** -6 / 2) ** (2 * eps * H.nu)
                assert abs(ours / cwx - 1) < mp.mpf(10) ** -20
        H = HP.HiggsPortal(I * 1, mmax=4)
        rows = H.order_table(1, Nmax=1)
        assert abs(rows[0][2] / rows[0][1] - 1) < mp.mpf(10) ** -20
        assert ok("higgs/|E_1/L_1| mu=1", abs(rows[1][2]))
        H = HP.HiggsPortal(I * 1)
        half = mp.mpf(1) / 2
        assert ok("higgs/I mu=1 u=0.5", H.calI(half, half, half))


def test_sm_templates_densities_and_closed_forms():
    """W/Z and top-quark loops at every shape: pair densities, tree weights, and the closed squeezed coefficients."""
    import sm_templates as SM
    with mp.workdps(30):
        w = SM.check_densities()
        assert w['F2'] < mp.mpf(10) ** -25 and w['psi'] < mp.mpf(10) ** -25
        # top quark: the series of dirac_pair.py from the closed tree coefficients
        for N in range(4):
            ref = TABLE[f"dirac_ref/S({2 * N + 1}) kappa=0 m=1"]['value']
            assert abs(SM.top_series('1', N) / mp.mpc(mp.mpf(ref[0]), mp.mpf(ref[1])) - 1) < mp.mpf(10) ** -20
        assert ok("dirac_orders/|S(3)/S(1)| kappa=0 m=1", abs(SM.top_series('1', 1) / SM.top_series('1', 0)))
        # the Higgs portal through the same class reproduces higgs_portal.py
        H = SM.SMLoop('higgs', '1')
        half = mp.mpf(1) / 2
        assert ok("higgs/I mu=1 u=0.5", H.calI(half, half, half))
        assert ok("sm/|signal/leading| wz mu=1 R=0.5", abs(SM.SMLoop('wz', '1', mmax=4).ratio('0.5')))
        assert ok("sm/|signal/leading| top mu=1 R=0.5", abs(SM.SMLoop('top', '1', mmax=4).ratio('0.5')))
    with mp.workdps(45):
        # W/Z: nine-pair soft moments of gauge_loop.py (stored at 45 digits), orders 0 and 1 of the hard vertex
        for order, dq in ((0, 0), (0, 2), (1, 0), (1, 3)):
            ref = TABLE[f"gauge_ref/order {order} mu=1 r2={2 + dq}"]['value']
            assert abs(SM.wz_soft_moment('1', order, dq) / mp.mpc(mp.mpf(ref[0]), mp.mpf(ref[1])) - 1) < mp.mpf(10) ** -20
        assert ok("gauge_orders/|E_1/L_1| mu=2", SM.wz_E1_over_L1('2'))


def test_sm_templates_two_routes():
    """Single sums against the invariant-distance integral (third difference in k^2), one shape and one power."""
    import sm_templates as SM
    with mp.workdps(25):
        assert SM.two_routes('wz', '1', ('0.2',), mmax=300, r1s=(2,)) < 1e-12
        assert SM.two_routes('top', '1', ('0.2',), mmax=300, r1s=(4,)) < 1e-12


def test_spin_bilinears_closed_forms():
    """Soft vertex of a tensor bilinear: the factors for the structures S, V, T2, and the structure of F_mr F_n^r."""
    import spin_bilinears as SB
    from universal import U
    with mp.workdps(30):
        for D in (3 + 2 * I, mp.mpf('2.4')):
            for r2 in (2, 3):
                assert abs(SB.V_S(D, r2) / SB.V_S(D, r2, False) / U(D, r2) - 1) < mp.mpf(10) ** -25
                assert abs(SB.V_T2(D, r2) / SB.V_T2(D, r2, False) / SB.U_T2(D, r2) - 1) < mp.mpf(10) ** -25
                assert abs(SB.V_V(D, r2) / SB.V_V(D, r2, False) / SB.U_V(D, r2) - 1) < mp.mpf(10) ** -25
        D0 = 3 + 2 * I
        assert abs(SB.V_V(D0, 2, False) + SB.V_S(D0, 2, False) / 4) < mp.mpf(10) ** -25 * abs(SB.V_S(D0, 2, False))
        r, res = SB.structure_FF(I * 1)
        assert abs(r - mp.mpf(4) / 3) < 1e-15 and res < 1e-15
        assert ok("spin/|U_O86| mu=1", abs(SB.U_O86(D0, 2)))


@pytest.mark.slow
def test_wz_nine_pairs_against_closed_form():
    """The nine-pair soft moment of gauge_loop.py itself (not the stored value) against the closed tree coefficient."""
    import gauge_loop as GL
    import sm_templates as SM
    with mp.workdps(45):
        G = GL.GaugeLoop('1')
        assert abs(G.order(0, 0) / SM.wz_soft_moment('1', 0, 0) - 1) < mp.mpf(10) ** -20


@pytest.mark.slow
def test_proca_tensor_vertex():
    """O_8,6 with Proca lines and all polarizations: isotropic and P_2 parts of the soft moment against the closed forms."""
    import spin_bilinears as SB
    with mp.workdps(30):
        assert max(SB.proca_tensor('1', r2s=(2, 3))) < mp.mpf(10) ** -16


@pytest.mark.slow
def test_tensor_vertex_momentum_space():
    """O_8,5 with scalar lines: the momentum-space soft moment follows the closed form V + S/4 in the soft power."""
    import spin_bilinears as SB
    with mp.workdps(30):
        nu = I * 1
        O = SB.TensorVertex(nu)
        D0 = 3 + 2 * nu
        rat = [O.moment(r2) / (SB.V_V(D0, r2) + SB.V_S(D0, r2) / 4) for r2 in (2, 3)]
        assert abs(rat[1] / rat[0] - 1) < mp.mpf(10) ** -20


def test_dirac_top_quark():
    """kappa = 0: the leading pair operator and the even orders are absent; the series starts at n = 1."""
    import dirac_pair as DP
    with mp.workdps(30):
        P = DP.DiracPair(1, 0)
        S = [P.S(n) for n in range(3)]
        assert abs(S[0]) < mp.mpf(10) ** -25 and abs(S[2]) < mp.mpf(10) ** -25
        assert ok("dirac_orders/|S(1)| kappa=0 m=1", abs(S[1]))


@pytest.mark.slow
def test_gauge_loop_soft_factor():
    """W/Z loop: the nine-pair soft moment depends on the soft power as the soft factor of dimension 3 + 2 nu."""
    import gauge_loop as GL
    with mp.workdps(45):
        G = GL.GaugeLoop('1')
        c = [G.order(0, dq) / GL.soft_tree(G.nu, 0, dq) for dq in (0, 1)]
        assert abs(c[1] / c[0] - 1) < mp.mpf(10) ** -15


def test_shape_ranges_record(tmp_path):
    """The 10% ranges of Table 5 (Sec. 10.1) reproduce their record."""
    import json
    import subprocess
    d = os.path.join(LOOPS, "shape_ranges")
    out = tmp_path / "ranges.json"
    subprocess.run([sys.executable, "shape_ranges.py", "--json", str(out)], cwd=d, check=True,
                   capture_output=True)
    new = json.load(open(out))
    rec = json.load(open(os.path.join(d, "data", "shape_ranges_record.json")))
    assert len(new) == len(rec)
    for a, b in zip(new, rec):
        assert (a["label"], a["lead"], a["factor"]) == (b["label"], b["lead"], b["factor"])
        assert abs(a["Rmax"] - b["Rmax"]) < 1e-12
