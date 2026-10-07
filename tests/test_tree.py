"""Tree routines."""
import itertools
import warnings

import numpy as np
import pytest

import dscorr
from dscorr import (DoubleExchange, SingleExchange, TripleExchange, ValidityWarning,
                    double_exchange, single_exchange, triple_exchange)

FAST = dict(L0=-54.0, L1=10.6, hL=0.005, T=8.0, h=0.05, T_s=8.0, h_s=0.025)
FAST_D = dict(L0=-54.0, L1=10.6, hL=0.005, T_sp=8.0, h=0.05, T_s=8.0, h_s=0.025)


def rel(a, b):
    return np.max(np.abs(np.asarray(a) / np.asarray(b) - 1))


@pytest.fixture(scope="module")
def T1157():
    return TripleExchange(1.0, 1.5, 0.7)


@pytest.fixture(scope="module")
def D115():
    return DoubleExchange(1.0, 1.5)


# ---------------------------------------------------------------- reference values of the production run
def test_frozen_single():
    assert rel(single_exchange(1, 1, 0.1, 1.0), 0.027164194684059412) < 1e-12          # 1
    assert rel(single_exchange(1, 1, 0.1, 1.0, sum_perms=True), 0.029826138118716518) < 1e-12
    assert rel(single_exchange(1, 0.8, 0.5, 2.0), 0.013207409267227757) < 1e-12        # 3


def test_single_blocks_point1():
    b = single_exchange(1, 1, 0.1, 1.0, out="blocks")
    c = b["channels"][0]
    assert c["exchanged_leg"] == 3
    assert rel(c["bracket"], 1.0865677873623765) < 1e-13
    assert rel(c["psi2F"], -0.527534111439929 - 0.18257073984031505j) < 1e-13
    assert rel(c["PhiA"], 0.04606501712603949) < 1e-13
    assert rel(c["PhiB"], 1.0741209549406476) < 1e-13
    assert rel(b["F"], 2.173135574724753) < 1e-13            # F = twice the bracket
    assert rel(single_exchange(1, 1, 0.1, 1.0, hL=0.004), 0.02716419468400193) < 1e-12
    assert rel(single_exchange(1, 1, 0.1, 1.0, hL=0.001), 0.027164194684063017) < 1e-12


def test_frozen_double(D115):
    assert rel(double_exchange(0.8, 1, 0.5, 1.0, 1.5), -0.0028136606935521957) < 1e-12  # 4
    assert rel(double_exchange(1, 1, 0.1, 1.0, 1.5), 0.0020564441711803513) < 1e-12     # 5
    assert rel(double_exchange(0.1, 1, 1, 1.0, 1.5), -0.0017749372702582878) < 1e-12    # 6
    assert rel(double_exchange(1, 0.8, 0.5, 1.0, 1.5, sum_perms=True),
               -0.023641963377518723) < 1e-12                                            # 7
    assert rel(double_exchange(0.8, 1, 0.5, 1.0, 1.5, uncut="time"),
               -0.0028136606936316247) < 1e-13


def test_frozen_triple(T1157):
    assert rel(T1157(1, 0.8, 0.5), -0.019266977812433563) < 1e-12                       # 8
    assert rel(T1157(1, 1, 0.1), -0.012392684374425658) < 1e-12                         # 9
    assert rel(T1157(1, 0.8, 0.5, uncut="time"), -0.019266977811807393) < 1e-13


def test_point12_heavy_masses():
    assert rel(double_exchange(0.8, 1, 0.5, 4.0, 2.0), 0.0004867217599214816) < 1e-12
    assert rel(double_exchange(0.8, 1, 0.5, 4.0, 2.0), 0.00048672175992159464) < 3e-10


def test_point11_settings():
    ref = 0.0002781625362265638                       # refined direct value
    vals = {"half": (DoubleExchange(0.7, 1.5, hL=0.002)(1, 1, 0.002), 0.0002781625362265827),
            "default": (DoubleExchange(0.7, 1.5)(1, 1, 0.002), 0.0002781625362264895),
            "fast": (DoubleExchange(0.7, 1.5, **FAST_D)(1, 1, 0.002), 0.00027816253622607514)}
    for k, (v, frozen) in vals.items():
        assert rel(v, frozen) < 1e-12, k
    assert rel(vals["half"][0], ref) < 2e-13
    assert rel(vals["default"][0], ref) < 1e-12
    assert rel(vals["fast"][0], ref) < 3e-12


def test_point4_settings():
    assert rel(DoubleExchange(1.0, 1.5, hL=0.002)(0.8, 1, 0.5), -0.0028136606935311053) < 1e-12
    assert rel(DoubleExchange(1.0, 1.5, **FAST_D)(0.8, 1, 0.5), -0.002813660693582721) < 1e-12


def test_block_values_point4():
    b = double_exchange(0.8, 1, 0.5, 1.0, 1.5, out="blocks")
    ref = {"psiF": -0.037643569421665926 + 0.10008984261655977j,
           "psiF_time": -0.037643569422460214 + 0.1000898426145269j,
           "uncut": -0.07528713884333185, "cut1": 0.01884430338535536,
           "cut2": 0.00016962158693257845, "Phi_1": 0.3396668700128083,
           "Phi_2": 0.10003574895661346, "Psi_1": 0.006506839827509731 - 0.030874674876528823j,
           "Psi_2": -0.0890622012475628 - 0.043663383944245876j, "Xi": 0.024634426403787286,
           "F_unit": -0.05627321387104391}
    for k, v in ref.items():
        tol = 1e-12 if k in ("psiF", "uncut", "F_unit") else 1e-13
        assert rel(b[k], v) < tol, k
    for k in ("k_input", "k_unit", "scale", "S", "F"):
        assert k in b


def test_block_values_point8(T1157):
    b = T1157(1, 0.8, 0.5, out="blocks")
    ref = {"psiF": 0.1674139790124973 + 0.054483107430750743j,
           "psiF_time": 0.1674139790062356 + 0.05448310743493803j,
           "uncut": -0.3348279580249946, "cut1": -0.058628875181215366,
           "cut2": 0.00801713154267226, "cut3": 0.00010014541486649619,
           "Phic": 0.047371785860384014, "F_unit": -0.3853395562486712}
    for k, v in ref.items():
        tol = 1e-12 if k in ("psiF", "uncut", "F_unit") else 1e-13
        assert rel(b[k], v) < tol, k
    for got, v in zip(b["Phi"], [0.3396668700128083, 0.07908520353593115, 1.2200637087178772]):
        assert rel(got, v) < 1e-13
    for got, v in zip(b["Psi"], [-0.09455566311041885 + 0.15717686184212784j,
                                 -0.11175805840771223 - 0.014917262833904937j,
                                 -0.041914357821406706 + 0.11881957391063243j]):
        assert rel(got, v) < 1e-13
    for got, v in zip(b["Xi"], [0.0014681138782737938 - 0.05937172892870949j,
                                -0.09644570475972183 - 0.06637961091314368j,
                                0.009826027146846853 - 0.08396430427251185j]):
        assert rel(got, v) < 1e-13


def test_other_outputs(T1157):
    assert rel(double_exchange(1.6, 2.0, 1.0, 1.0, 1.5, out="F"), -0.007034151733880489) < 1e-12
    assert rel(double_exchange(1.6, 2.0, 1.0, 1.0, 1.5, out="B"), -0.0002747715521047066) < 1e-12
    assert rel(T1157(2, 1.6, 1.0, out="F"), -0.0481674445310839) < 1e-12
    assert rel(T1157(2, 1.6, 1.0, out="B"), -0.001881540801995465) < 1e-12
    assert rel(single_exchange(2, 2, 0.2, 1.0, out="F"), 0.2716419468405941) < 1e-12
    assert rel(single_exchange(2, 2, 0.2, 1.0, out="B"), 0.042444054193842826) < 1e-12
    assert rel(double_exchange(0.8, 0.5, 1.0, 1.5, 1.0), -0.0028136606935521952) < 1e-12
    assert rel(SingleExchange(1.0).scan([1.0, 0.1]),
               [0.011358957175825631, 0.027164194684059412]) < 1e-12
    assert rel(SingleExchange(1.0).scan([1.0, 0.1], sum_perms=True),
               [0.03407687152747689, 0.029826138118716518]) < 1e-12
    assert rel(DoubleExchange(1.0, 1.5).scan([0.1], out="F"), 0.1645155336944281) < 1e-12


# ---------------------------------------------------------------- direct subset
def test_agreement_with_direct_subset(T1157):
    ref = dscorr.data.reference_points()
    for r in ref["single"]:
        if r["mu"] in (1.0, 2.0):
            v = single_exchange(r["k1"], r["k2"], r["k3"], r["mu"], sum_perms=True)
            assert rel(v, r["S_direct_SK"]) < 3e-10
    for r in ref["double"]:
        if r["pair"] == "neq":
            v = double_exchange(r["k1"], r["k2"], r["k3"], r["mu1"], r["mu2"])
            assert rel(v, r["S_direct_SK"]) < 3e-10
    for r in ref["scalene"]:
        if r["diagram"] == "triple" and r["masses"] == [1.0, 1.5, 0.7]:
            assert rel(T1157(*r["k"]), r["S_direct_SK"]) < 3e-10
        if r["diagram"] == "double" and r["masses"] == [1.0, 1.5]:
            assert rel(double_exchange(*r["k"], 1.0, 1.5), r["S_direct_SK"]) < 3e-10
    for r in ref["mass_range"]:
        if r["diagram"] == "double" and r["masses"][0] in (0.3, 4.0):
            assert rel(double_exchange(*r["k"], *r["masses"]), r["S_direct_SK"]) < 3e-10
        if r["diagram"] == "single" and r["masses"][0] in (0.3, 4.0):
            v = single_exchange(*r["k"], r["masses"][0], sum_perms=True)
            assert rel(v, r["S_direct_SK"]) < 3e-10


@pytest.mark.slow
def test_agreement_with_direct_all_reference_points():
    ref = dscorr.data.reference_points()
    for r in ref["triple"]:
        v = triple_exchange(r["k1"], r["k2"], r["k3"], r["muA"], r["muB"], r["muC"])
        assert rel(v, r["S_direct_SK"]) < 3e-10
    for r in ref["double"]:
        v = double_exchange(r["k1"], r["k2"], r["k3"], r["mu1"], r["mu2"])
        assert rel(v, r["S_direct_SK"]) < 3e-10
    for r in ref["single"]:
        v = single_exchange(r["k1"], r["k2"], r["k3"], r["mu"], sum_perms=True)
        assert rel(v, r["S_direct_SK"]) < 3e-10
    for r in ref["scalene"]:
        f = double_exchange if r["diagram"] == "double" else triple_exchange
        assert rel(f(*r["k"], *r["masses"]), r["S_direct_SK"]) < 3e-10
    for r in ref["mass_range"]:
        if r["diagram"] == "single":
            v = single_exchange(*r["k"], r["masses"][0], sum_perms=True)
        else:
            f = double_exchange if r["diagram"] == "double" else triple_exchange
            v = f(*r["k"], *r["masses"])
        assert rel(v, r["S_direct_SK"]) < 3e-10


# ---------------------------------------------------------------- channels
def test_single_channels():
    ref = [r for r in dscorr.data.reference_points()["mass_range"]
           if r["diagram"] == "single" and r["kinematics"] == "scalene" and r["masses"] == [3.0]][0]
    k1, k2, k3 = ref["k"]
    ch = {1: single_exchange(k2, k3, k1, 3.0), 2: single_exchange(k3, k1, k2, 3.0),
          3: single_exchange(k1, k2, k3, 3.0)}
    for c in ref["channels"]:
        assert rel(ch[c["exchanged_leg"]], c["S_direct_SK"]) < 3e-10
    tot = single_exchange(k1, k2, k3, 3.0, sum_perms=True)
    assert rel(ch[1] + ch[2] + ch[3], tot) < 1e-14


def test_double_sum_perms_explicit():
    k = (1.0, 0.8, 0.5)
    tot = sum(double_exchange(*(k[i] for i in p), 1.0, 1.5)
              for p in ((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1), (0, 2, 1)))
    assert rel(double_exchange(*k, 1.0, 1.5, sum_perms=True), tot) < 1e-14


# ---------------------------------------------------------------- symmetries
def test_symmetries_fast():
    assert rel(double_exchange(0.8, 1, 0.5, 1.0, 1.5), double_exchange(0.8, 0.5, 1, 1.5, 1.0)) < 1e-11
    assert rel(single_exchange(1, 0.8, 0.5, 1.0), single_exchange(0.8, 1, 0.5, 1.0)) < 1e-12
    T = TripleExchange(1.0, 1.5, 0.7, **FAST)
    T2 = TripleExchange(1.5, 1.0, 0.7, **FAST)
    T3 = TripleExchange(0.7, 1.5, 1.0, **FAST)
    assert rel(T(1, 0.8, 0.5), T2(0.8, 1, 0.5)) < 1e-11
    assert rel(T(1, 0.8, 0.5), T3(0.5, 0.8, 1)) < 1e-11


@pytest.mark.slow
def test_relabelled_triple_defaults():
    assert rel(triple_exchange(0.8, 1.0, 0.5, 1.5, 1.0, 0.7), -0.019266977812433563) < 1e-12
    assert rel(triple_exchange(0.5, 1.0, 0.8, 0.7, 1.0, 1.5), -0.019266977812433556) < 1e-12


# ---------------------------------------------------------------- scans
def test_scan_vs_call_double(D115):
    ks = [1.0, 0.1, 0.0107]
    assert rel(D115.scan(ks), [D115(1, 1, k) for k in ks]) < 1e-12
    assert rel(D115.scan(ks, soft="k1"), [D115(k, 1, 1) for k in ks]) < 1e-12
    for code in ("mixed", "C", "a"):
        np.testing.assert_array_equal(D115.scan(ks, soft=code), D115.scan(ks))
    for code in ("unmixed", "B", "b"):
        np.testing.assert_array_equal(D115.scan(ks, soft=code), D115.scan(ks, soft="k1"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        v = D115.scan([1.0, 0.1, 0.002])
        v1 = D115.scan([1.0, 0.1, 0.002], soft="k1")
    assert rel(v, [-0.005036567447771637, 0.0020564441711803513, 0.00023495290899786062]) < 1e-12
    assert rel(v1, [-0.005036567447771637, -0.0017749372702582878, -4.090080126365104e-05]) < 1e-12


def test_scan_vs_call_triple(T1157):
    ks = [1.0, 0.1, 0.0107]
    assert rel(T1157.scan(ks), [T1157(1, 1, k) for k in ks]) < 1e-12
    assert rel(T1157.scan([1.0, 0.1, 0.002]),
               [-0.018955142051970627, -0.012392684374425658, 0.0011579130024840113]) < 1e-12


def test_scan_misc(D115):
    assert D115.scan([]).size == 0
    with pytest.raises(ValueError):
        D115.scan([0.5], soft="k2")
    with pytest.raises(ValueError):
        D115.scan([2.5])
    with pytest.raises(ValueError):
        D115.scan([0.0])
    assert rel(D115.scan([1.5]), D115(1, 1, 1.5)) < 1e-13


@pytest.mark.slow
def test_scan_contour_abscissa():
    kp = [1.0, 0.3, 0.1, 0.02, 0.002]
    T = TripleExchange(1.0, 1.5, 0.7)
    base = T.scan([1.0, 0.1])
    for c in (-0.2, -0.3):
        Tc = TripleExchange(1.0, 1.5, 0.7, c=c)
        s = Tc.scan([1.0, 0.1])
        assert rel(s, [Tc(1, 1, 1.0), Tc(1, 1, 0.1)]) < 1e-12
        assert rel(s, base) < 1e-9
    T2 = TripleExchange(1.0, 1.5, 0.7, c=-0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        s13 = T2.scan(kp)
        g13 = [T2(1, 1, k) for k in kp]
    assert rel(s13, [-0.018955142051970585, -0.018814435832708462, -0.012392684374425636,
                     -0.002760843492642904, 0.0011579130024840126]) < 1e-12
    assert rel(g13, [-0.018955142051970585, -0.018814435832708462, -0.012392684374425636,
                     -0.002760843492642909, 0.0011579130024840095]) < 1e-12
    assert rel(T2.scan([0.1], out="F"), -0.9914147499540508) < 1e-12
    T3 = TripleExchange(1.0, 1.0, 1.0, c=-0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        s14 = T3.scan(kp)
    ref14 = [-0.025907391964191655, -0.019376748517621167, -0.008138268365467634,
             0.0010937172494644258, 2.6560324925403213e-05]
    assert rel(s14[:4], ref14[:4]) < 1e-12
    # kappa = 0.002: S is a small difference of O(1e-2) terms (absolute agreement 1e-16)
    assert abs(s14[4] - ref14[4]) < 1e-15


# ---------------------------------------------------------------- 
def test_methods_and_routes(D115, T1157):
    assert rel(D115.psi_F(0.8, 1, 0.5, method="engine"), D115.psi_F(0.8, 1, 0.5)) < 1e-12
    for b in (D115(0.8, 1, 0.5, out="blocks"), T1157(1, 0.8, 0.5, out="blocks")):
        assert rel(b["psiF"], b["psiF_time"]) < 1e-9
        tot = b["uncut"] + b["cut1"] + b["cut2"] + b.get("cut3", 0.0)
        assert abs(tot - b["F_unit"]) <= 1e-15 * abs(b["F_unit"])


@pytest.mark.slow
def test_engine_method_triple(T1157):
    assert rel(T1157.psi_F(1, 0.8, 0.5, method="engine"), T1157.psi_F(1, 0.8, 0.5)) < 1e-12


# ---------------------------------------------------------------- scale invariance
def test_scale_invariance_and_outputs(T1157):
    for f in (lambda *k, **kw: double_exchange(*k, 1.0, 1.5, **kw), T1157,
              lambda *k, **kw: single_exchange(*k, 1.0, **kw)):
        S = f(0.8, 1, 0.5)
        S2 = f(1.6, 2, 1.0)
        assert rel(S2, S) < 1e-13
        F = f(1.6, 2, 1.0, out="F")
        B = f(1.6, 2, 1.0, out="B")
        assert rel(1.6 * 2 * 1.0 * F / 8, S2) < 1e-14
        assert rel((1.6 * 2 * 1.0) ** 2 * B, S2) < 1e-14


# ---------------------------------------------------------------- arrays
def test_arrays(D115):
    k3 = np.array([[1.0, 0.1], [0.0107, 0.5]])
    v = double_exchange(1, 1, k3, 1.0, 1.5)
    assert v.shape == k3.shape
    for i in np.ndindex(k3.shape):
        assert rel(v[i], double_exchange(1, 1, float(k3[i]), 1.0, 1.5)) < 1e-12
    assert rel(v.ravel(), D115.scan(k3.ravel())) < 1e-12
    assert isinstance(double_exchange(1, 1, 0.5, 1.0, 1.5), float)
    bl = double_exchange(1, 1, [0.5, 0.4], 1.0, 1.5, out="blocks")
    assert isinstance(bl, list) and len(bl) == 2
    s = single_exchange([1.0, 1.0], 1.0, [0.1, 0.5], 1.0)
    assert rel(s[0], single_exchange(1, 1, 0.1, 1.0)) < 1e-14


# ---------------------------------------------------------------- invalid input
def test_invalid_input():
    with pytest.raises(ValueError, match="^double_exchange:"):
        double_exchange(1, 1, 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="^DoubleExchange:"):
        DoubleExchange(1.0, -1.0)
    with pytest.raises(ValueError, match=r"^single_exchange: .*\(k1, k2, k3\)"):
        single_exchange(1, 1, 0, 1.0)
    with pytest.raises(ValueError, match=r"triangle.*\(1.0, 0.3, 0.5\)"):
        single_exchange(1, 0.3, 0.5, 1.0)
    with pytest.raises(TypeError, match="double_exchange"):
        double_exchange(1, 1, 1, 1.0, 1.0, T=8.0)
    with pytest.raises(TypeError, match="T_sp"):
        DoubleExchange(1.0, 1.0, T=8.0)
    with pytest.raises(TypeError):
        TripleExchange(1.0, 1.0, 1.0, T_sp=8.0)
    with pytest.raises(TypeError):
        triple_exchange(1, 1, 1, 1.0, 1.0, 1.0, sum_perms=True)
    with pytest.raises(TypeError, match="L0, L1, hL"):
        SingleExchange(1.0, c=-0.2)
    with pytest.raises(TypeError):
        DoubleExchange(1.0)
    with pytest.raises(TypeError):
        TripleExchange((1.0, 1.0))
    with pytest.raises(ValueError):
        single_exchange(1, 1, 1, 1.0, out="X")
    with pytest.raises(ValueError):
        double_exchange(1, 1, 1, 1.0, 1.5, sum_perms=True, out="blocks")
    # masses as a tuple
    assert DoubleExchange((1.0, 1.5)).masses == (1.0, 1.5)
    # contour keywords are not validated
    DoubleExchange(1.0, 1.5, c=0.5)
    # folded / degenerate triangles accepted
    single_exchange(2, 1, 1, 1.0)
    single_exchange(1, 0.5, 0.5 + 5e-13, 1.0)


def test_warnings():
    with pytest.warns(ValidityWarning):
        SingleExchange(5.0)
    with pytest.warns(ValidityWarning):
        single_exchange(1, 1, 1.995e-3, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single_exchange(1, 1, 0.002, 0.3)


# ---------------------------------------------------------------- lines of Fig. S2(a),(b) of the Letter
KAP40 = [float(np.exp(np.log(0.05) * (1 - j / 39))) for j in range(40)]


def test_figS2_kappa_grid():
    assert KAP40[0] == 0.05000000000000001
    assert KAP40[20] == 0.23236187587325832
    assert KAP40[-1] == 1.0


def test_figS2_lines_double():
    k = KAP40[0]
    assert rel(double_exchange(1, 1, k, 1.0, 1.0, sum_perms=True), 0.008449530826778089) < 1e-11
    single = [double_exchange(1, 1, k, 1.0, 1.0), double_exchange(1, k, 1, 1.0, 1.0),
              double_exchange(k, 1, 1, 1.0, 1.0)]
    for v, r in zip(single, [0.0031674669190034995, 0.0031674669190034995, -0.0021101684246179545]):
        assert rel(v, r) < 1e-11
    for j, r in ((0, 0.005701423853193988), (20, -0.00556178439293373), (39, -0.030219404686629824)):
        assert rel(double_exchange(1, 1, KAP40[j], 1.0, 1.5, sum_perms=True), r) < 1e-11
    m = KAP40[20]
    single = [double_exchange(1, 1, m, 1.0, 1.5), double_exchange(1, m, 1, 1.0, 1.5),
              double_exchange(m, 1, 1, 1.0, 1.5)]
    for v, r in zip(single, [0.001391394029085558, -0.000755951628125621, -0.0034163345974268016]):
        assert rel(v, r) < 1e-11
    line = double_exchange(1, 1, m, 1.0, 1.5, sum_perms=True)
    assert rel(line, 2 * sum(single)) < 1e-12
    assert abs(line / single[0] - 1) > 0.1


def sum6(T, k):
    return sum(T(*p) for p in itertools.permutations((1.0, 1.0, k)))


def test_figS2_lines_triple(T1157):
    for j, r in ((0, -0.019088715478154807), (20, -0.07369074439515898), (39, -0.11373085231182377)):
        assert rel(sum6(T1157, KAP40[j]), r) < 1e-11
    m = KAP40[20]
    for v, r in zip([T1157(1, 1, m), T1157(1, m, 1), T1157(m, 1, 1)],
                    [-0.017654904471314653, -0.006636484162246077, -0.012553983564018762]):
        assert rel(v, r) < 1e-11


@pytest.mark.slow
def test_figS2_lines_triple_equal_masses():
    T = TripleExchange(1.0, 1.0, 1.0)
    assert rel(sum6(T, KAP40[20]), -0.10099693421306787) < 1e-11


# ---------------------------------------------------------------- series solution of Aoki et al. (arXiv:2404.09547) at one point
def test_exact_series_point():
    k = 0.19785178061376274
    # value of the series solution of Aoki et al., evaluated by the author from their formulas
    attributed = 0.0010084776191287925
    d = DoubleExchange(0.7, 1.5)(1, 1, k)
    h = DoubleExchange(0.7, 1.5, hL=0.002)(1, 1, k)
    assert rel(d, 0.0010084776190967197) < 1e-12
    assert rel(h, 0.0010084776191268605) < 1e-12
    assert rel(d, attributed) < 4e-11
    assert rel(h, attributed) < 3e-12


# ---------------------------------------------------------------- accuracy settings at the worst validation point
@pytest.mark.slow
def test_point8_and_10_settings():
    assert rel(TripleExchange(1.0, 1.5, 0.7, hL=0.002)(1, 0.8, 0.5), -0.019266977812352645) < 1e-12
    assert rel(TripleExchange(1.0, 1.5, 0.7, **FAST)(1, 0.8, 0.5), -0.01926697781253595) < 1e-12
    ref = -5.737836779939709e-06
    vals = {"half": (TripleExchange(0.5, 1.0, 1.5, hL=0.002), -5.737836779848365e-06, 1.59e-11),
            "default": (TripleExchange(0.5, 1.0, 1.5), -5.737836778873114e-06, 1.86e-10),
            "fast": (TripleExchange(0.5, 1.0, 1.5, **FAST), -5.737836776839801e-06, 5.40e-10)}
    # At this point S = -5.7e-6 is the sum of O(1) terms (uncut -0.774, cut1 +0.751):
    # last-bit differences of those terms give ~1e-11 in S (floating-point floor: this code vs
    # the production run 7.8e-12 at half step, 1.9e-11 fast).  The accuracy statement is
    # therefore checked with that floor added (half step: 2.37e-11 here vs 1.59e-11 recorded).
    floor = 2e-11
    for name, (T, frozen, acc) in vals.items():
        v = T(1, 1, 0.002)
        assert rel(v, frozen) < floor, name
        assert rel(v, ref) < acc + floor, name
