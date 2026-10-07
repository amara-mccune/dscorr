"""Direct Schwinger-Keldysh route."""
import warnings

import mpmath as mp
import numpy as np
import pytest

import dscorr
from dscorr import (ValidityWarning, double_exchange, double_exchange_direct,
                    single_exchange_direct, triple_exchange_direct)
from dscorr._engines import direct as D
from dscorr._engines.direct import SETTING_A, SETTING_B, direct_refined


def rel(a, b):
    return abs(a / b - 1)


@pytest.fixture(autouse=True)
def _restore_dps():
    old = mp.mp.dps
    yield
    mp.mp.dps = old


# ---------------------------------------------------------------- 
@pytest.mark.parametrize("fn,args", [
    (double_exchange_direct, (1, 1, 1, None, 1.0)),
    (triple_exchange_direct, (1, 1, 1, None, 1.0, 1.0)),
    (single_exchange_direct, (1, 1, 1, None)),
])
@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), "x"])
def test_bad_masses(fn, args, bad):
    a = list(args)
    a[3] = bad
    with pytest.raises(ValueError) as e:
        fn(*a)
    msg = str(e.value)
    assert msg.startswith(fn.__name__ + ":") and "mu" in msg


@pytest.mark.parametrize("ks", [(1, 1, 0), (1, -1, 1), (1, 1, float("nan")), (1, 1, 2.5),
                                (3, 1, 1), (1, 0.3, 0.5)])
def test_bad_momenta(ks):
    for fn, m in ((double_exchange_direct, (1.0, 1.5)), (triple_exchange_direct, (1.0, 1.5, 0.7))):
        with pytest.raises(ValueError, match="moment"):
            fn(*ks, *m)
    with pytest.raises(ValueError, match=r"^double_exchange_direct: .*\(k1, k2, k3\)"):
        double_exchange_direct(1, 1, 0, 1.0, 1.5)
    with pytest.raises(ValueError, match=r"triangle.*\(1.0, 0.3, 0.5\)"):
        triple_exchange_direct(1, 0.3, 0.5, 1.0, 1.5, 0.7)


def test_check_order():
    # double checks the momenta first, triple and single the masses first
    with pytest.raises(ValueError, match="moment"):
        double_exchange_direct(1, 1, 0, -1.0, 1.0)
    with pytest.raises(ValueError, match="mass"):
        triple_exchange_direct(1, 1, 0, -1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="mass"):
        single_exchange_direct(1, 1, 0, -1.0)


def test_nproc_and_family_errors():
    with pytest.raises(ValueError, match="single_exchange_direct: nproc must be >= 1, got 0"):
        single_exchange_direct(1, 1, 1, 1.0, nproc=0)
    with pytest.raises(ValueError, match=r"engine_family=True needs \(k1,k2,k3\) = \(1,1,kappa\)"):
        triple_exchange_direct(1, 0.8, 0.5, 1.0, 1.5, 0.7, engine_family=True, mpanels=1, nn=2)


def test_warnings_direct():
    with pytest.warns(ValidityWarning, match="tested range"):
        double_exchange_direct(1, 1, 1, 5.0, 1.0, mpanels=1, nn=2)
    with pytest.warns(ValidityWarning, match=r"1.9950e-03 < tested minimum 2.0000e-03"):
        triple_exchange_direct(1, 1, 1.995e-3, 1.0, 1.0, 1.0, mpanels=1, nn=2)


# ---------------------------------------------------------------- output selection
def test_output_selection():
    q = (1.0, 1.0, 0.5)
    assert D is not None
    from dscorr.direct_sk import _output
    assert _output("f", 0.25, 4.0, q, 2.0, "S", False, {}) == 0.25
    assert _output("f", 0.25, 4.0, q, 2.0, "F", False, {}) == 0.5
    d = _output("f", 0.25, 4.0, q, 2.0, "X", True, {})
    assert d["B"] == 0.5 / (8 * 1 * 1 * 0.5 * 8)
    with pytest.raises(ValueError, match="out must be"):
        _output("f", 0.25, 4.0, q, 2.0, "X", False, {})


# ---------------------------------------------------------------- checkpoints
def test_gauss_legendre_mp():
    with mp.workdps(25):
        x, w = D.gauss_legendre_mp(4)
        assert mp.nstr(x[3], 25) == "0.8611363115940525752239465"
        assert mp.nstr(x[2], 25) == "0.3399810435848562648026658"
        assert mp.nstr(w[3], 25) == "0.3478548451374538573730639"
        assert mp.nstr(w[2], 24) == "0.652145154862546142626936"
        x, w = D.gauss_legendre_mp(24)
        assert mp.nstr(x[0], 25) == "-0.9951872199970213601799974"
        assert mp.nstr(w[0], 25) == "0.01234122979998719954680568"
        assert abs(sum(w) - 2) < mp.mpf(10) ** -23


def test_outer_grid_and_tail():
    with mp.workdps(25):
        q = (mp.mpf("0.8"), mp.mpf(1), mp.mpf("0.5"))
        nodes, wts, tmax = D.outer_grid(q, D.DirectSettings())
        assert len(nodes) == 528
        assert mp.nstr(tmax, 25) == "21.73913043478260869565217"
        assert mp.nstr(nodes[0], 25) == "1.007312223136343911694133e-12"
        assert mp.nstr(nodes[-1], 25) == "21.69977059484945884985084"
        q2 = (mp.mpf(1), mp.mpf(1), mp.mpf("0.002"))
        _, _, tm2 = D.outer_grid(q2, D.DirectSettings())
        assert D._tail_panels(q2[2], tm2, D.DirectSettings()) == (375, 22)
        A = D.DirectSettings(window=70, tail_width="0.25", tail_extent=22, soft_below="0.1",
                             min_soft_panels=7, mpanels=36)
        _, _, tmA = D.outer_grid(q2, A)
        assert D._tail_panels(q2[2], tmA, A) == (125, 90)
        B = D.DirectSettings(window=60, tail_width="0.35", tail_extent=18, soft_below="1.5",
                             min_soft_panels=7, mpanels=28)
        _, _, tmB = D.outer_grid(q2, B)
        assert D._tail_panels(q2[2], tmB, B) == (175, 54)


@pytest.mark.slow
def test_constant_C():
    with mp.workdps(25):
        q = (mp.mpf("0.8"), mp.mpf(1), mp.mpf("0.5"))
        st = D.DirectSettings()
        nodes, wts, tmax = D.outer_grid(q, st)
        lt = D.LineTables(mp.mpf(1), mp.mpf(1), nodes, tmax, st)
        assert lt.npoints == 7020
        assert abs(lt.C - mp.mpf("0.3396668700127642342310193")) < mp.mpf(10) ** -24
        lt = D.LineTables(mp.mpf("0.5"), mp.mpf("1.5"), nodes, tmax, st)
        assert abs(lt.C - mp.mpf("0.1000357489566032069719875")) < mp.mpf(10) ** -24


def test_cumulative_matrix():
    S = D.cumulative_matrix(3)
    ref = np.array([[0.2777777777777777, -0.07195333504987775, 0.019578888030616637],
                    [0.600526389961729, 0.44444444444444453, -0.044970834406173665],
                    [0.5359766675249388, 0.9608422239387668, 0.27777777777777773]])
    assert np.abs(S - ref).max() < 1e-15
    S = D.cumulative_matrix(8)
    x, _ = np.polynomial.legendre.leggauss(8)
    f = 3 * x ** 5 - x ** 2 + 0.5
    exact = 0.5 * x ** 6 - x ** 3 / 3 + 0.5 * x - (0.5 + 1 / 3 - 0.5)
    assert np.abs(S @ f - exact).max() < 1e-14


def test_mode_functions():
    (u, us), = D.mode_functions(([0.5], 1.0, 1.0))
    assert rel(u, -1.1177375094006143 - 0.5451899973658634j) < 1e-15
    assert abs(us.imag / -0.09642356924962048 - 1) < 1e-15 and abs(us.real) < 1e-25


# ---------------------------------------------------------------- coarse values
def test_coarse_double():
    assert double_exchange_direct(0.8, 1, 0.5, 1.0, 1.5, mpanels=2, nn=4) == -0.012946906192620924
    assert double_exchange_direct(0.8, 1, 0.5, 1.0, 1.5, mpanels=2, nn=4, out="F") == -0.2589381238524185
    assert double_exchange_direct(0.8, 1, 0.5, 1.0, 1.5, mpanels=2, nn=4, out="B") == -0.08091816370388077
    d = double_exchange_direct(1.6, 2, 1.0, 1.0, 1.5, mpanels=2, nn=4, full=True)
    assert d["S"] == -0.012946906192620924
    assert d["F"] == -0.03236726548155231
    assert d["B"] == -0.001264346307873137
    assert set(d) >= {"S", "F", "B", "psiF_time", "k_unit", "scale", "mp_values"}
    assert d["scale"] == 2.0 and d["k_unit"] == (0.8, 1.0, 0.5)
    assert mp.mp.dps == 25                       # documented global side effect
    assert double_exchange_direct(1, 1, 0.01, 1.0, 1.5, mpanels=2, nn=4) == -0.0006588650034967453


def test_coarse_triple():
    assert triple_exchange_direct(1, 0.8, 0.5, 1.0, 1.5, 0.7, mpanels=2, nn=4) == -0.006130191308446524
    assert triple_exchange_direct(0.5, 1, 0.8, 0.7, 1.0, 1.5, mpanels=2, nn=4) == -0.006130191308446524
    assert triple_exchange_direct(1, 1, 0.01, 1.0, 1.5, 0.7, mpanels=2, nn=4) == -0.00425092677170578
    a = triple_exchange_direct(2, 2, 0.2, 1.0, 1.5, 0.7, mpanels=2, nn=4, full=True)
    b = triple_exchange_direct(2, 2, 0.2, 1.0, 1.5, 0.7, mpanels=2, nn=4, full=True,
                               engine_family=True)
    assert a["S"] == b["S"] == -0.0067404240681203115
    assert a["mp_values"]["S"] == b["mp_values"]["S"]


def test_coarse_refined():
    assert direct_refined("triple", (1.0, 1.5, 0.7), (1, 1, 0.01), mpanels=3, tmax_fac=70,
                          tail_width="0.25", tail_extent=22, soft_below="0.1",
                          nn=4) == -0.0012871126513028096
    assert direct_refined("double", (1.0, 1.5), (0.8, 1, 0.5), mpanels=3, tmax_fac=60,
                          tail_width="0.35", tail_extent=18, soft_below="1.5",
                          nn=4) == -0.010726983748806065


def test_coarse_single():
    v = single_exchange_direct(1, 1, 0.5, 1.0, h=2.0, n=4, nproc=1)
    assert rel(v, 0.022522925787205727) < 4e-16
    assert rel(single_exchange_direct(0.5, 1.0, 0.8, 1.5, h=1.0, n=6, nproc=1),
               0.013984865758234433) < 4e-16
    d = single_exchange_direct(2, 2, 1.0, 1.0, h=2.0, n=4, nproc=1, full=True)
    assert rel(d["S"], 0.022522925787205727) < 4e-16
    assert rel(d["F"], 0.045045851574411454) < 4e-16
    assert rel(d["B"], 0.001407682861700358) < 4e-16
    assert set(d) >= {"S", "F", "B", "imag_leak", "channels", "k_unit", "scale"}


def test_raw_mode():
    a = double_exchange_direct(1.6, 2.0, 1.0, 1.0, 1.5, mpanels=2, nn=4, full=True, raw=True)
    assert a["scale"] == 1.0 and a["k_unit"] == (1.6, 2.0, 1.0)
    # the outer grid starts at the fixed t_min = 1e-12, so a coarse grid is not scale
    # covariant: raw and normalised S agree only when converged (slow test below)
    assert abs(a["S"] / -0.012946906192620924 - 1) < 1e-2


# ---------------------------------------------------------------- slow: converged values
@pytest.mark.slow
def test_T1_T4_converged():
    d = double_exchange_direct(0.8, 1.0, 0.5, 1.0, 1.5, full=True)
    assert d["S"] == -0.002813660693529512
    assert abs(d["mp_values"]["S"] - mp.mpf("-0.002813660693529512034017313")) < mp.mpf("1e-26")
    assert rel(d["F"], -0.05627321387059024) < 1e-15
    assert rel(d["B"], -0.017585379334559448) < 1e-15
    assert rel(d["psiF_time"], -0.03764356942166244 + 0.10008984261655085j) < 1e-15
    # out='F' against the Mellin-Barnes F to 3e-10
    assert rel(d["F"], double_exchange(0.8, 1.0, 0.5, 1.0, 1.5, out="F")) < 3e-10
    t = triple_exchange_direct(1.0, 0.8, 0.5, 1.0, 1.5, 0.7, full=True)
    assert t["S"] == -0.019266977812344356
    assert abs(t["mp_values"]["S"] - mp.mpf("-0.01926697781234435735394759")) < mp.mpf("1e-26")


@pytest.mark.slow
def test_T2_T5_T7_family():
    ref = dscorr.data.reference_points()
    r = [x for x in ref["double"] if x["pair"] == "eq" and x["kappa"] == 0.002][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        assert rel(double_exchange_direct(1, 1, 0.002, 1.0, 1.0, mpanels=28), r["S_direct_SK"]) < 1e-13
        r = [x for x in ref["triple"] if x["triple"] == "neq2" and x["kappa"] == 0.002][0]
        assert rel(triple_exchange_direct(1, 1, 0.002, 0.5, 1.0, 1.5, mpanels=28),
                   r["S_direct_SK"]) < 1e-12
    a = triple_exchange_direct(1.0, 1.0, 0.1, 1.0, 1.5, 0.7, full=True, engine_family=True)
    b = triple_exchange_direct(1.0, 1.0, 0.1, 1.0, 1.5, 0.7, full=True)
    assert a["mp_values"]["S"] == b["mp_values"]["S"]
    assert a["S"] == -0.012392684374366587


@pytest.mark.slow
def test_T3_T6_refined():
    a = direct_refined("double", (1.0, 1.0), (1.0, 1.0, 0.002), **SETTING_A)
    assert rel(a, -0.0006323005760508468) < 1e-15
    b = direct_refined("double", (1.0, 1.0), (1.0, 1.0, 0.002), **SETTING_B)
    assert rel(a, b) < 1e-15
    a = direct_refined("triple", (0.5, 1.0, 1.5), (1.0, 1.0, 0.002), **SETTING_A)
    assert rel(a, -5.737836779939709e-06) < 1e-15


@pytest.mark.slow
def test_scale_and_raw_converged():
    d = double_exchange_direct(1.6, 2.0, 1.0, 1.0, 1.5, full=True)
    assert d["S"] == -0.002813660693529512
    assert rel(d["F"], -0.00703415173382378) < 1e-15
    assert rel(d["B"], -0.00027477155210249137) < 1e-15
    r = double_exchange_direct(1.6, 2.0, 1.0, 1.0, 1.5, full=True, raw=True)
    assert rel(r["S"], -0.002813660693529512) < 1e-15
    assert rel(r["psiF_time"], -0.004705446177707805 + 0.012511230327068856j) < 1e-14


@pytest.mark.slow
def test_T8_heavy():
    d = double_exchange_direct(1.0, 1.0, 0.05, 4.0, 2.0, mpanels=28, dps=40, full=True)
    assert rel(d["S"], 2.9096236639103804e-05) < 1e-15
    assert mp.mp.dps == 40


@pytest.mark.slow
def test_single_regeneration():
    d = single_exchange_direct(1.0, 1.0, 0.1, 1.0, sum_perms=True, nproc=2, full=True)
    assert len(d["channels"]) == 3
    assert rel(d["S"], 0.029826138118702297) < 1e-11
    assert rel(d["channels"][2]["F_sectors"].real, 1.0865677873625172) < 1e-13
    ref = [r for r in dscorr.data.reference_points()["mass_range"]
           if r["diagram"] == "single" and r["kinematics"] == "scalene" and r["masses"] == [3.0]][0]
    k1, k2, k3 = ref["k"]
    v = single_exchange_direct(k2, k3, k1, 3.0, nproc=2)       # channel with the line on k1
    assert rel(v, ref["channels"][0]["S_direct_SK"]) < 1e-11
