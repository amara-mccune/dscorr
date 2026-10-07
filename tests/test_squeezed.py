"""Squeezed coefficients, templates and the fit."""
import warnings

import mpmath as mp
import numpy as np
import pytest

import dscorr
from dscorr import M1, M2, M3, SqueezedFit, TripleExchange, ValidityWarning, extract, signal, template
from dscorr._engines import squeezed_f64 as F
from dscorr._engines.squeezed_mp import SeriesLine


def rel(a, b):
    return abs(complex(a) / complex(b) - 1)


FILE_30 = {  # 30-digit reference values of the closed forms (150-digit series route)
    ("double", (1.0, 1.5)): mp.mpc("-0.0026768313688852251219811396709115",
                                   "-0.00091245982920738259072513008713011"),
    ("double", (0.7, 1.5)): mp.mpc("-0.0033384444459235045994826368748616",
                                   "-0.0015332659794474194687125149022638"),
    ("double", (1.0, 1.0)): mp.mpc("-0.007759663854657429520566731995117",
                                   "-0.0081195874456850885165318203162089"),
    ("triple", (1.0, 1.5, 0.7)): mp.mpc("0.0019874702440619763055219793131571",
                                        "-0.015259095273482261336809092995378"),
    ("triple", (1.0, 1.0, 1.0)): mp.mpc("0.00043716137921192201440633484237563",
                                        "-0.0095922360232390260138333725234814"),
    ("triple", (0.5, 1.0, 1.5)): mp.mpc("-0.00089221336371390131187508644949264",
                                        "-0.0026141138731487681089908967073398"),
}


# ---------------------------------------------------------------- 
def test_closed_forms_against_reference():
    for r in dscorr.data.reference_points()["squeezed"]:
        m = r["masses"]
        f = {"single": M1, "double": M2, "triple": M3}[r["diagram"]]
        tol = 1e-14 if r["diagram"] == "single" else 5e-12
        assert rel(f(*m), complex(*r["M"])) < tol
    for (d, m), v in FILE_30.items():
        f = M2 if d == "double" else M3
        assert rel(f(*m), complex(v)) < 5e-12


def test_float64_values_bitlevel():
    assert M1(1.0) == -0.02754209997498152 + 0.012660884323437285j
    assert M1(0.3) == -0.158180674743796 + 0.15779824130424852j
    assert M1(4.0) == 9.175030587694046e-06 - 5.002696013082441e-06j
    for v, r in ((M2(1.0, 1.5), -0.0026768313688890115 - 0.0009124598292138959j),
                 (M2(1.0, 1.0), -0.007759663854665647 - 0.0081195874457017j),
                 (M2(0.7, 1.5), -0.003338444445927855 - 0.0015332659794530133j),
                 (M3(1.0, 1.5, 0.7), 0.0019874702441078657 - 0.015259095273480053j),
                 (M3(0.5, 1.0, 1.5), -0.0008922133637039666 - 0.0026141138731522076j),
                 (M3(1.0, 1.0, 1.0), 0.00043716137923882886 - 0.00959223602324353j),
                 (M2(0.3, 0.3), -0.0060134745176666435 - 0.20607148814621917j),
                 (M3(0.3, 0.3, 0.3), 0.12482815511571803 - 0.3975504117284231j),
                 (M3(3.0, 3.0, 3.0), 2.0525991611384227e-08 + 1.0461425411010308e-06j)):
        assert rel(v, r) < 1e-14
    # mu = 4: the oscillatory part is a strong cancellation (5e-15 level agreement)
    assert rel(M2(4.0, 4.0), 3.272443799173206e-07 - 7.176826343840518e-08j) < 1e-13
    assert rel(M3(4.0, 4.0, 4.0), 2.114855435280543e-08 + 1.5698966559353421e-09j) < 1e-13


def test_float64_intermediates():
    p = M2(1.0, 1.5, parts=True)
    ref = dict(HFp=-0.02275974906126143 - 0.08987707445931292j,
               HFm=0.046453261956834836 - 0.17184911150766838j,
               psip=-0.02263073864645757 - 0.08938922682315349j,
               psim=0.04632425154203095 + 0.17136126387150893j,
               H=0.10441364147042317 - 0.027611996449004975j,
               A=-0.0026768313688890115 - 0.0009124598292138959j,
               A_alt=-0.002676831368889009 - 0.000912459829213892j,
               Apsi_p=-0.02033497647634796 - 0.009722714548945039j,
               Apsi_m=-3.174549273251342e-06 + 1.4559558640746764e-06j)
    for k, v in ref.items():
        assert rel(p[k], v) < 1e-14, k
    assert rel(p["A"], p["A_alt"]) < 1e-14           # float64
    p = M3(1.0, 1.5, 0.7, parts=True)
    ref = dict(TFFp=-0.03512745546097485 + 0.06884160447576006j,
               TFFm=-0.12781411373661608 + 0.11521338552395657j,
               Up=-0.07326848829004942 - 0.04924735047943384j,
               Um=-0.06404705954157788 - 0.1185246526206229j,
               Vp=-0.00938025788816252 - 0.08154802580535382j,
               Vm=-0.0001588291396911115 - 0.08622397729470056j,
               TKK=0.06693134550424286 - 0.0036788178134740147j)
    for k, v in ref.items():
        assert rel(p[k], v) < 1e-14, k


# ---------------------------------------------------------------- 
def test_M1_dps():
    v = M1(1.0, dps=30)
    assert rel(v, M1(1.0)) < 1e-13
    with mp.workdps(30):
        assert mp.nstr(v.real, 30) == "-0.0275420999749814138964760816991"
        assert mp.nstr(v.imag, 30) == "0.0126608843234372423224267844681"
        w = M1(0.5, dps=30)
        assert mp.nstr(w.real, 30) == "-0.100337683269638343763317042404"
        assert mp.nstr(w.imag, 30) == "0.0551526777441539138097644770293"


# ---------------------------------------------------------------- 
def test_template_signal():
    M = 0.3 - 0.2j
    for k in (1e-3, 1e-2, 0.1):
        t = template(k, M, 1.3)
        assert abs(t - 2 * (M * k ** (0.5 + 1.3j)).real) < 1e-15 * abs(t)
        assert abs(signal(k, M, 1.3) * np.sqrt(k) - t) < 1e-15 * abs(t)
    M3v = M3(1.0, 1.5, 0.7)
    # M3v itself carries last-bit (BLAS-reduction) variations of ~1e-15 between processes
    assert abs(template(0.01, M3v, 0.7) / -0.0001461082192449669 - 1) < 1e-14
    assert abs(signal(0.01, M3v, 0.7) / -0.0014610821924496689 - 1) < 1e-14
    assert isinstance(template(0.01, M3v, 0.7), np.floating)
    assert template(np.array([0.1, 0.2]), M3v, 0.7).shape == (2,)


# ---------------------------------------------------------------- 
def test_extract_synthetic():
    k = np.logspace(-4, -2, 60)
    mu = 0.7
    c = np.array([-0.03, 0.01, -0.012, 0.002, -0.015, 0.002, -0.003, 1e-6, -2e-6, 3e-7, 1e-7])
    S = dscorr.squeezed.design_matrix(k, mu) @ c
    fit = extract(k, S, mu)
    assert isinstance(fit, SqueezedFit) and len(tuple(fit)) == 4
    assert rel(fit.M, complex(c[3], c[4])) < 1e-9
    assert abs(fit.S_an(1e-3) - (c[0] * 1e-3 + c[1] * 1e-6 + c[2] * 1e-9)) < 1e-15


@pytest.mark.slow
def test_extract_recovers_closed_form():
    kap = np.logspace(-4, -2, 80)
    T = TripleExchange(1.0, 1.5, 0.7, L1=16.0)
    with pytest.warns(ValidityWarning):
        S = T.scan(kap)
    M, res, coef, S_an = extract(kap, S, 0.7)
    assert rel(M, M3(1.0, 1.5, 0.7)) < 1e-8
    assert res < 1e-14
    assert rel(M, 0.001987470236874772 - 0.015259095276459115j) < 1e-10
    assert abs(S_an(1e-3) / -2.7705429302699692e-05 - 1) < 1e-7


@pytest.mark.slow
def test_deep_extraction_table_rows():
    file_extracted = {(1.0, 1.5, 0.7): 0.00198747023690737 - 0.015259095276434181j}
    kap = np.logspace(-4, -2, 320)
    T = TripleExchange(1.0, 1.5, 0.7, L1=16.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        S = T.scan(kap)
    assert rel(S[0], -1.3397535238855016e-05) < 1e-12
    fit = extract(kap, S, 0.7)
    assert rel(fit.M, file_extracted[(1.0, 1.5, 0.7)]) < 1e-11
    m30 = complex(FILE_30[("triple", (1.0, 1.5, 0.7))])
    assert abs(abs(fit.M) / abs(m30) - 1) < 1e-9
    assert abs(np.angle(fit.M / m30)) < 1e-9


# ---------------------------------------------------------------- (slow)
@pytest.mark.slow
def test_mp_routes_against_record():
    for (d, m), v in FILE_30.items():
        if m not in ((1.0, 1.5), (1.0, 1.5, 0.7)):
            continue
        p = (M2 if d == "double" else M3)(*m, dps=30, parts=True)
        val = p["A"]
        assert abs(val / v - 1) < 1e-14
        if d == "double":
            assert p["errH"] < 1e-35 and p["errAB"] < 1e-35
        else:
            assert p["errT"] < 1e-35
    with mp.workdps(33):
        v = M2(1.0, 1.5, dps=30)
        assert mp.nstr(v.real, 30) == "-0.00267683136888522512198113967091"
        v = M3(1.0, 1.5, 0.7, dps=30)
        assert mp.nstr(v.real, 30) == "0.00198747024406197564168915746269"


# ---------------------------------------------------------------- 
def test_hard_line_checkpoints():
    g = F._grid()
    hl = F.HardLine(1.0, g)
    with mp.workdps(150):
        sl = SeriesLine(mp.mpf(1))
    phi = float(F.phi_hat(1.0))
    assert abs(hl.Phi_num / phi - 1) < 1e-11
    assert rel(hl.Phi_num, 0.33966687001280743) < 1e-15
    for t_L, fK, fI, fFF, fg in (
            (-2.0, 0.35549056172321425, -1.4089718454600915 - 1.3068110012975005j,
             -0.758742309617688 - 0.44554134668998974j, -0.758742309617688 - 0.44388040250902483j),
            (0.0, 0.2894280370259932, 1.90079967581943 - 1.0639600135544445j,
             0.0005865134231184677 - 0.3627442505401971j, 0.0005865134231184677 - 0.3613919676228221j),
            (2.0, 0.0002631228955858078, 260.26623637155757 - 0.0009672602642247964j,
             0.00023089480563715008 - 0.0003297756448891254j,
             0.00023089480563715008 - 0.00032854626642072455j)):
        j = int(round((t_L - g.L0) / g.hL))
        assert abs(g.L[j] - t_L) < 1e-12
        assert rel(hl.K[j], fK) < 1e-13
        assert rel(hl.I[j], fI) < 1e-14
        assert rel(hl.FF[j], fFF) < 1e-12
        assert rel(hl.g[j], fg) < 1e-12
        # identity FF = g - (i/pi) e^{-pi mu} Phi_num K
        assert abs(hl.FF[j] - (hl.g[j] - 1j / np.pi * np.exp(-np.pi) * hl.Phi_num * hl.K[j])) < 1e-15
        # against the 150-digit series route (independent): 1e-11 rel at t <= 1, 1e-10 abs at e^2
        with mp.workdps(150):
            K, FF, gg = sl.at(mp.exp(t_L))
            if t_L <= 0:
                assert rel(hl.FF[j], complex(FF)) < 1e-11
            else:
                assert abs(hl.FF[j] - complex(FF)) < 1e-10
            assert rel(hl.K[j], float(mp.re(K))) < 1e-11


def test_phi_num_values():
    for mu, v in ((0.3, 2.664012923593625), (4.0, 2.746214092835629e-05)):
        hl = F.HardLine(mu, F._grid())
        assert rel(hl.Phi_num, v) < 1e-14


# ---------------------------------------------------------------- 
def test_input_validation():
    for f, args in ((M1, ()), (M2, (1.0,)), (M3, (1.0, 1.0))):
        for bad in (0.0, -1.0, float("nan"), "x"):
            with pytest.raises(ValueError, match=f"^{f.__name__}: "):
                f(*args, bad)
    with pytest.raises(ValueError, match="template: kappa must be finite and > 0"):
        template(0.0, 1j, 1.0)
    with pytest.raises(ValueError, match="signal: kappa must be finite and > 0"):
        signal(np.array([0.1, np.inf]), 1j, 1.0)
    k = np.logspace(-3, -1, 10)
    with pytest.raises(ValueError, match="extract: kappas and S must be 1-D arrays"):
        extract(k, k[:-1], 1.0)
    with pytest.raises(ValueError):
        extract(k[:7], k[:7], 1.0)
    with pytest.raises(ValueError):
        extract(k.reshape(2, 5), k.reshape(2, 5), 1.0)
    with pytest.raises(ValueError, match="extract: kappas must be finite and > 0 and S finite"):
        extract(-k, k, 1.0)
    with pytest.warns(ValidityWarning):
        M1(5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M1(0.3)
        M1(4.0)
    assert M2(1, 1) == M2(1.0, 1.0)


# ---------------------------------------------------------------- 
def test_symmetry_M3():
    assert rel(M3(1.0, 1.5, 0.7), M3(1.5, 1.0, 0.7)) < 1e-15
