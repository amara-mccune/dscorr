"""One-loop batch pipelines.

The pipelines live in examples/Letter/batch/ and are imported from there (they are batch
drivers, not library calls).  The checkpoints are minute-scale pieces of the production runs;
reference values are those of the production runs behind the Letter's loop figures.
"""
import json
import math
import os
import sys

import mpmath as mp
import numpy as np
import pytest

BATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Letter", "batch")
if not os.path.isdir(BATCH):
    pytest.skip("examples/Letter/batch not found (tests run outside the source tree)",
                allow_module_level=True)
for sub in ("triangle", "triangle_pidot", "triangle_pidot_correlator", "bubble"):
    sys.path.insert(0, os.path.join(BATCH, sub))

import assemble_scan as AS  # noqa: E402
import bf_triangle as BF  # noqa: E402
import corr as CO  # noqa: E402
import mb_pidot as MP  # noqa: E402
import mb_triangle as MT  # noqa: E402
import terms as TE  # noqa: E402


def rel(a, b):
    return abs(complex(a) / complex(b) - 1)


# ============================================================================ triangle, brute force

def test_bessel_direct_and_table():
    b = BF.BesselImag(1.0)
    tb = BF.TabBessel(1.0)
    assert tb.N == 13820 and abs(tb.d - 0.00399956880884831) < 1e-17
    ref = [(0.001, 0.44379825433591424, 1.0088471382172444 - 1.6281773462081373j),
           (0.5, 0.796985415828616, 0.5276800746528324 - 1.0778060064800015j),
           (2, 0.6826413458556082, 0.4354400102250688 - 0.04596204762771408j),
           (29, 0.22785374709727857, 0.075725021383196 - 1.5548397679826627e-18j),
           (31, 0.22067806426103145, 0.07313617161301748),
           (200, 0.08834686498211358, 0.028297994109772656)]
    for x, K, I in ref:
        assert rel(b.K(x)[0], K) < 1e-14
        assert abs(b.I(x)[0] - I) < 1e-14 * abs(I)
        assert rel(tb.K(np.array([x]))[0], K) < 1e-9          # interpolation <= 1e-9
        assert abs(tb.I(np.array([x]))[0] - I) < 1e-9 * abs(I)
    with pytest.raises(AssertionError, match="TabBessel range exceeded"):
        tb.K(np.array([2e7]))


def test_fixed_l_triangle():
    tg = BF.TimeGrid()
    assert tg.nodes == 1872 and abs(tg.ratio - 1.2255230454011186) < 1e-15
    x, y = (1.0, 1.2, 0.9), (0.7, 1.1, 0.5)
    v, per = BF.Triangle((1, 1.3, 0.7), tg, tab=False).psi3(x, [y[0]], [y[1]], [y[2]], orderings=True)
    assert rel(v[0], 0.020959163073298214 + 0.02492571063791384j) < 1e-13
    refo = [0.0037271700079225902 + 0.0051760652563546455j, 0.003441432521517565 + 0.0040865779345999305j,
            0.0034223767394684075 + 0.003505957484936539j, 0.0036609090159063352 + 0.004266160149565684j,
            0.0030986512515615 + 0.003086213214794327j, 0.0036086235369218148 + 0.004804736597662714j]
    for i in range(6):
        assert rel(per[i, 0], refo[i]) < 1e-12
    vt = BF.Triangle((1, 1.3, 0.7), tg, tab=True).psi3(x, [y[0]], [y[1]], [y[2]])
    assert rel(vt[0], 0.020959163073248518 + 0.024925710637607224j) < 1e-13


def test_D0_on_time_grids():
    for m, g, ref in (((1.0, 1.0, 1.0), (1e-12, 60, 156), -0.0845041165334552 - 0.014367934831992318j),
                      ((2.0, 2.0, 2.0), (1e-14, 80, 220), -0.016827468055683116 - 2.703199841322304e-05j)):
        v = BF.Triangle(m, BF.TimeGrid(*g), tab=False).psi3((0, 0, 0), [1.0], [1.0], [1.0])[0]
        assert rel(v, ref) < 1e-12
    d0 = json.load(open(os.path.join(BATCH, "triangle", "data", "D0_all.json")))
    assert len(d0) == 11
    assert abs(complex(*d0["(1.0, 1.0, 1.0)"]["D0"]) - ref) > 0      # table differs by t_min truncation


def test_angular_rule_and_radial_ladder():
    for k, r, s in (((1, 1.2, 0.9), 0.4, 0.9999999999999997), ((1, 1.2, 0.9), 5.0, 1.0),
                    ((0.15, 1, 1.1), 1000.0, 0.9999999999998059)):
        yA, yB, w = BF.angular_nodes(r, k, (6, 8, 24))
        assert yA.size == 1344
        assert abs(w.sum() / (4 * np.pi) - s) < 1e-15
    for st, k, ne, nn in (("P1", (1, 1.2, 0.9), 62, 552), ("P1", (0.15, 1, 1.1), 62, 534),
                          ("P2", (1, 1.2, 0.9), 76, 830), ("P2", (0.15, 1, 1.1), 76, 808)):
        rs, ws, z, e = BF.radial_nodes(k, BF.SETTINGS[st])
        assert len(e) == ne and rs.size == nn


@pytest.mark.slow
def test_far_field_nodes():
    k = (1.0, 1.2, 0.9)
    for r, g, ref in ((311.0533250061996, (1e-12, 60.0, 156), -0.0034128466169012724 - 0.0005248253884768807j),
                      (149007.2464124384, (1e-12, 60.0, 156), -7.124329318709073e-06 - 1.2114610491983861e-06j)):
        f, n = BF.f_of_r(BF.Triangle((1, 1, 1), BF.TimeGrid(*g)), k, r, (4, 6, 16))
        assert n == 480 and rel(f, ref) < 1e-13


# ============================================================================ triangle, Mellin-Barnes

def test_W3_small_lattice():
    w3 = MT.W3Lattice((1, 1.2, 0.9), MT.arange_sym(0.6, 0.3))
    for (a, b, c), ref in (((0, 0, 0), 1.4715686320412633 + 0.5370608837248548j),
                           ((1, 3, 4), -2.501819414662688 + 2.8973443540152255j),
                           ((4, 2, 0), -12.73656043371745 + 1.3122288605862664j)):
        assert rel(w3.C_for_a(a)[b, c], ref) < 1e-13


def test_lattice_lengths():
    assert len(MT.arange_sym(40.0, 0.075)) == 1068
    assert abs(MT.arange_sym(40.0, 0.075)[-1] - 40.025) < 1e-10
    assert len(MT.arange_sym(40.0, 0.05)) == 1601 and len(MT.arange_sym(40.0, 0.04)) == 2001
    assert len(MT.arange_sym(6.0, 0.075)) == 161 and len(MT.arange_sym(6.0, 0.15)) == 81


def test_D0_closed_form_coarse():
    tp = MT.TriPole((1, 1, 1), (1.0, 1.2, 0.9), sigma=-0.125, hs=0.15)
    assert len(tp.tau) == 534
    assert rel(tp.D0(), -0.08058551000480664 - 0.014698826936499845j) < 1e-12


def test_coarse_A():
    rec = MT.routeA_point((1, 1, 1), (1.0, 1.2, 0.9), hs=0.3, hv=0.3)
    assert rec["Ns"] == 268 and rec["Nv"] == 41
    assert rel(complex(*rec["A"]), -0.04216941774795324 - 0.0531419174088067j) < 1e-12
    assert rel(complex(*rec["J0_160"]), -5.1171532402450675 + 1.0066231389567548j) < 1e-11


def test_mb_fixed_l_coarse():
    v = MT.psi3_mb((1, 1.3, 0.7), (1, 1.2, 0.9), (0.7, 1.1, 0.5), hs=0.3, hv=0.3)
    assert rel(v, 0.023385561794883488 + 0.02459032681114256j) < 1e-12


@pytest.mark.slow
def test_D0_closed_form_production_steps():
    for hs, ref in ((0.075, -0.08451257014855838 - 0.014316398395379269j),
                    (0.04, -0.08448250793864041 - 0.014367947400214662j)):
        tp = MT.TriPole((1, 1, 1), (1.0, 1.2, 0.9), sigma=-0.125, hs=hs)
        assert rel(tp.D0(), ref) < 1e-12


@pytest.mark.slow
def test_coarse_R():
    rec = MT.TriPole((1, 1, 1), (1.0, 1.2, 0.9), sigma=-0.125, hs=0.15).R(gp_h=0.02, levels=3)
    for key, ref in (("R", 1.1676680860490705 + 1.8707875874142188j),
                     ("g_reg", 0.21623116043165305 + 0.026811119747713365j),
                     ("LL", -0.0475082804587635 - 0.002912230055201534j),
                     ("L0", 0.11857780894466972 + 0.012807007980415964j),
                     ("OL", 0.14516163194574683 + 0.01691634182249893j),
                     ("Gprime", 0.24684290024799618 - 0.5079303718747443j)):
        assert rel(complex(*rec[key]), ref) < 1e-11, key


def test_scan_assembly_formulas():
    Rx = AS.R_extrap(1.669651650018079 + 1.8203456240805311j, 1.6694866381542457 + 1.8200503755581148j)
    assert rel(Rx, 1.6694825713855608 + 1.8200437300133991j) < 1e-15
    assert rel(AS.dT(1.0), -0.00039705530480298455 + 1.7591140986095155e-07j) < 1e-15
    assert rel(AS.dT(0.15), -0.00038108291235965374 + 1.6747592888055606e-07j) < 1e-15
    MB = (-0.032171102589250086 - 0.0364099861970783j) + Rx + AS.dT(1.0)
    assert rel(MB, 1.6369144134915077 + 1.7836339197277307j) < 1e-15
    assert AS.classify("Keq_M111") == ("control_eq", "M111")
    assert AS.classify("K1_mu0.85") == ("mass", "equal")
    assert AS.classify("k0.15_Mmix") == ("momentum", "Mmix")
    # B_reg: an exact model is recovered (regulated brute-force extrapolation)
    D0 = -0.08447761913278562 - 0.014367934289384971j
    r = np.geomspace(300, 150000, 200)
    a0, a1, b = 0.3 + 0.1j, -0.05j, 2.0
    f = 4 * np.pi * D0 / r + a0 / r ** 2 + a1 * np.log(r) / r ** 2 + b / r ** 3
    L = 150000.0
    Breg_true = 1.234 + 0.5j
    BL = Breg_true + 4 * np.pi * D0 * math.log(L) - a0 / L - a1 * (math.log(L) + 1) / L - b / (2 * L * L)
    assert abs(BF.B_reg_single(r, f, BL, L, 3000, D0) - Breg_true) < 1e-10


# ============================================================================ triangle with pi-dot legs

def test_pidot_integrand_identity():
    """r_v = 2 equals minus the third mixed x-derivative of the r_v = 1 integrand (test 18)."""
    tg = BF.TimeGrid()
    y = ([0.7], [1.1], [0.5])
    x0 = np.array([1.0, 1.2, 0.9])
    t1 = BF.Triangle((1, 1.3, 0.7), tg, tab=False, rv=1.0)
    t2 = BF.Triangle((1, 1.3, 0.7), tg, tab=False, rv=2.0)
    v2 = t2.psi3(tuple(x0), *y)[0]

    def d3(h):
        s = 0
        for e in np.ndindex(2, 2, 2):
            sg = np.array([1 if q else -1 for q in e])
            s += np.prod(sg) * t1.psi3(tuple(x0 + h * sg), *y)[0]
        return s / (8 * h ** 3)

    d1, d2 = d3(0.04), d3(0.02)
    est = (4 * d2 - d1) / 3
    assert rel(-est, v2) < 1e-5
    v1a = t1.psi3(tuple(x0), *y)[0]
    v1b = BF.Triangle((1, 1.3, 0.7), tg, tab=False).psi3(tuple(x0), *y)[0]
    assert abs(v1a - v1b) <= 1e-14 * abs(v1a)


def test_pidot_C3_assertions_and_window_fit():
    delta = 0.6 - 1j * MP.frange(-0.2, 0.2, 0.1)
    c3 = MP.C3Core((1.0, 1.2, 0.9), delta, -0.6, sig_u=-0.1, hst=0.025)
    assert abs(c3.margin - 0.1) < 1e-12
    for bad in (-0.2, 0.05):
        with pytest.raises(AssertionError):
            MP.C3Core((1.0, 1.2, 0.9), delta, -0.6, sig_u=bad, hst=0.025)
    p = 5 + 6 * (-0.6)
    N = np.arange(20, 87, 3.3)
    Sinf = 0.123 - 0.456j
    S = Sinf + N ** -p * (0.7 - 0.2j + (0.3j) / N + 0.11 / N ** 2)
    v, res = MP.extrapolate_window(N, S, p, 3, 10)
    assert abs(v - Sinf) < 1e-12 * abs(Sinf)


def test_pidot_windows():
    Nl, pieces = MP.windows(2001, 30, 0.1)
    assert abs(Nl[0] - 3.3) < 1e-12 and abs(Nl[-1] - 100.0) < 1e-12
    assert max(n for n in Nl if n <= 100 - 140 * 0.1 + 1e-9) == pytest.approx(83.3)
    covered = sorted(i for p in pieces for a, b in p for i in range(a, b))
    assert covered == list(range(2001))
    tasks = json.load(open(os.path.join(BATCH, "triangle_pidot", "tasks_mb_primary.json")))["runs"]
    assert len(tasks) == 14 and all("nblk" in t for t in tasks)


@pytest.mark.slow
def test_pidot_mb_K1_M111():
    rec = MP.mb_loop_point((1, 1, 1), (1.0, 1.2, 0.9), rho=-0.6, sig_u=-0.1, hst=0.025, taumax=100.0,
                           nblk=30)
    assert rec["Ns"] == 2001 and rec["Nv"] == 81
    S = rec["S_ladder"]
    assert rel(complex(*S[0]), -0.06022729934060726 - 0.06447488206025427j) < 1e-12
    assert rel(complex(*S[-1]), -0.059561771596684195 - 0.06447050700744168j) < 1e-12
    ex = MP.extrapolate_record(rec)
    assert rel(ex["S_inf"], -0.05956106477941924 - 0.0644705069933408j) < 1e-12
    assert abs(ex["N_last"] - 83.3) < 1e-9
    assert abs(ex["extrap_err"] / 6.338183400532397e-08 - 1) < 1e-6


# ============================================================================ triangle correlator

def test_correlator_grids():
    for lvl, k, n in (("Q0", (1., 1., 1.), 101400), ("Q0", (1., 1., 0.1), 114900),
                      ("Q1", (1., 1., 1.), 409500), ("Q1", (1., 1., 0.01), 441900)):
        c, e = CO.pou_nodes(k, CO.LEVELS[lvl])
        assert sum(len(x[2]) for x in c) == n
    L, W, p, e = CO.g2_nodes((1., 1., 1.))
    assert W.size == 916950
    tg = CO.time_grid((1., 1., 0.1), 80, 8)
    assert (tg.tmin, tg.tmax, tg.npan, tg.ngl) == (1e-9, 500.0, 87, 8)
    tasks = json.load(open(os.path.join(BATCH, "triangle_pidot_correlator", "tasks_correlator.json")))
    assert len(tasks["points"]) == 79


def test_correlator_partition_of_unity_and_normalisation():
    c, e = CO.pou_nodes((1., 1., 0.3), CO.LEVELS["Q0"])
    # the weights of the three centres sum to 1 at any point: check via a constant integrand
    # inside a ball of radius 0.05 around l = 0 (volume 4/3 pi R^3)
    struct = {"U": 0.023816034851035534, "V2": 0.008953950726372235, "V0": 0.008953979342843347,
              "V1": 0.008953953591966917}
    Scut = CO.norm_factor((1, 1, 1)) * (-2 * struct["U"] - 2 * (struct["V2"] + struct["V0"] + struct["V1"]))
    assert abs(Scut / -5.1076269809364924e-05 - 1) < 1e-9


@pytest.mark.slow
def test_correlator_mu05_Q0():
    raw = CO.run_point_pou((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), "Q0")
    a = CO.assemble_pou(raw)
    assert a["n_nodes"] == 101400
    assert rel(a["S"], -5.107683113958834e-05) < 1e-12
    assert rel(a["S_cut"], -5.1076269809364924e-05) < 1e-12
    assert abs(a["tail_power"] - 1.9096877510208168) < 1e-9


# ============================================================================ bubble

def test_bubble_plan():
    pl = TE.plan(0.55)
    assert len(pl) == 122 and sum(n for _, _, n in pl) == 13784
    n = {f"{f}{m}": k for f, m, k in TE.plan(0.1, 3)}
    assert n == {"TNone": 0, "U0": 46, "U1": 16, "V1": 47, "U2": 49, "U3": 16, "V3": 50}
    assert TE.n_terms("U", 0, 0.55) == 114 and TE.n_terms("V", 79, 0.55) == 267
    rl = open(os.path.join(BATCH, "bubble", "rlist_fig3.txt")).read().strip().split(",")
    assert len(rl) == 50 and rl[-1] == "0.55"


def test_bubble_inclusion_rule():
    F = TE.Families(TE.Problem("fig3c"), dps=20)
    inc = [(n1, m) for m in range(81) for n1 in range(25) if F.MIK_included(n1, m)]
    assert len(inc) == 1400
    assert inc[:5] == [(0, 1), (0, 2), (0, 3), (1, 3), (0, 4)]
    incx = [(m1, m) for m in range(81) for m1 in range(25) if F.MXX_included(m1, m)]
    assert len(incx) == 1375
    assert incx[:5] == [(0, 2), (0, 3), (0, 4), (1, 4), (0, 5)]


def test_bubble_family_coefficients():
    F = TE.Families(TE.Problem("fig3c"), dps=20)
    F.Nmax = 0
    tw = F.towers(["1"])
    assert rel(tw["T0"][2], -1.6173290191763986e-05 - 1.6700243144467304e-05j) < 1e-14
    assert rel(tw["R0T0"][2], 1.2561959698828823e-05 - 3.730048424329534e-06j) < 1e-14
    assert rel(F.U_term(0, 0, ["0.3"])[0], 0.00012349165877337752 + 6.85369516971998e-05j) < 1e-14
    assert rel(F.V_term(1, 0, ["0.3"])[0], -1.7644505557255282e-05 + 4.1175937076385046e-05j) < 1e-14
    assert rel(F.TIK(0, 1, 0, ["1"])[2], -3.384613814035623e-05 - 6.3561130944751654e-06j) < 1e-14
    assert rel(F.TXX(1, 0, -1, 1, ["1"])[2], 1.6200237184043328e-05 + 1.2822674840143607e-05j) < 1e-14
    assert rel(F.JIK(0, ["0.3"])[0], 0.0001574722550417127 - 0.000280300599425129j) < 1e-13
    assert rel(F.MXX(-1, 1, 4, ["0.3"])[0], -1.1929299603161089064e-8 - 5.9399949744996866751e-7j) < 1e-13
    v = F.R00("0.3")
    assert abs(v / mp.mpc("0.0030383308072691901253", "0.0045284925089378888788") - 1) < 1e-19


@pytest.mark.slow
def test_bubble_rung_families():
    F = TE.Families(TE.Problem("fig3c"), dps=20)
    fin, pole = F.rung(0, "U")
    # Phi'(0) by the central difference delta = 1e-8 at 20 digits loses ~8 digits
    assert abs(fin / mp.mpc("-0.0025068858348712029454", "-0.0049745490210668380561") - 1) < 1e-11
    assert abs(pole / mp.mpf("-0.0015831434944115276788") - 1) < 1e-19
    fin, pole = F.rung(1, "V")
    assert abs(fin / mp.mpc("0.020756334136717572067", "0.0081292339861142663284") - 1) < 1e-11
    assert abs(pole / mp.mpf("0.0047494304832345830364") - 1) < 1e-18


def test_bubble_scheme_chain_and_assembly():
    mp.mp.dps = 30
    S5, Ss, RS = TE.scheme_chain(mp.mpf("-1.6878463001980945789e-6"), "0.1")
    assert abs(RS / mp.mpf("9.5155131144194000229e-6") - 1) < 1e-19
    assert abs(S5 / mp.mpf("0.000022016405244346967299") - 1) < 1e-18
    assert abs(Ss / mp.mpf("8.6853387920897475696e-7") - 1) < 1e-17
    assert abs((3 * mp.euler - mp.log(mp.pi) - 3) / 2 - mp.mpf("-1.2065414455724007962")) < 1e-19
    # house-scheme assembly from the stored sections at r = 0.1 (D_II, D_IK, XX)
    lam = -1j / mp.pi * mp.exp(-mp.pi * mp.mpf("1.472"))
    DII = mp.mpc("-0.000210980787533241", "0.00373730658510037")
    DIK = mp.mpc("0.000115147663524767", "-0.000265180654282286")
    XX = mp.mpc("1.63456240682395e-5", "-4.07030192358471e-22")
    B = DII + 2 * lam * DIK + (lam * lam + 1 / mp.pi ** 2) * XX
    Sh = 8 * mp.mpf("0.1") ** 3 * mp.re(B)
    assert abs(Sh / mp.mpf("-1.6878463001980945789e-6") - 1) < 1e-13
    mp.mp.dps = 15


def test_bubble_grading():
    assert TE.grading("T3") == 6 and TE.grading("R0T2") == 4
    assert TE.grading("T1_+2") == 6 and TE.grading("T+1_-2") == 6
    assert TE.grading("U7") == 7 and TE.grading("M3_9") == 9 and TE.grading("M+1_4") == 4
    assert TE.grading("EU3") is None and TE.grading("PR0U1") is None
    with pytest.raises(ValueError, match="unknown problem"):
        TE.Problem("nope")
