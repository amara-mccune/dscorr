"""signal_sizes.py (examples/Loops): conventions, closed forms of the template papers, and the f_NL table.

Reference values are in examples/Loops/signal_sizes_record.json.
"""
import os
import sys

import mpmath as mp
import pytest

LOOPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops")
if not os.path.isdir(LOOPS):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, LOOPS)

import signal_sizes as ss  # noqa: E402

I = mp.mpc(0, 1)


def test_table_reproduces_the_record():
    """Every number of the f_NL table, and the checks made along the way."""
    with mp.workdps(30):
        rows, res = ss.build(verbose=False)
    assert len(rows) > 30
    bad = [err for ok, err in res if not ok]
    assert not bad, bad


def test_paper_values_of_section_8_3():
    """|f_NL^osc| = N_c 4 pi^2 P_zeta |C| gives the two numbers quoted in Sec. 8.3 (4e-6 and 6e-5)."""
    with mp.workdps(30):
        f10 = ss.as_fnl(ss.fermion_C(10, 1))
        f20 = ss.as_fnl(ss.fermion_C(20, 2))
        assert abs(f10 / mp.mpf("4.37e-6") - 1) < 5e-3
        assert abs(f20 / mp.mpf("5.74e-5") - 1) < 5e-3
        assert round(float(f10) * 1e6) == 4 and round(float(f20) * 1e5) == 6
        # late-time soft vertex of Aoki and Strumia: lower by the factors of Table 4
        assert abs(f10 / ss.as_fnl(ss.fermion_C(10, 1, "nl")) / 113 - 1) < 0.01


def test_closed_forms_agree_with_the_other_scripts():
    import higgs_portal as hp
    import universal as un
    with mp.workdps(30):
        for mu in (0.5j, 1j, 1.4):
            assert abs(ss.C_H(mu) / hp.C_H(mu) - 1) < 1e-25
        assert abs(ss.U(3 + 2j, 2) / un.U(3 + 2j, 2) - 1) < 1e-25
        # Wang and Xianyu: Eq. (45) tends to Eq. (47) at large mass for kappa = mu,
        # and to (1 + e^(-2 pi |kappa - mu|))/2 times Eq. (47) otherwise
        nu = mp.sqrt(mp.mpf(100) ** 2 - mp.mpf(1) / 4)
        assert abs(ss.wx_fnl(nu, nu, "b") / ss.wx_fnl(nu, nu, "b", asymptotic=True) - 1) < 0.03
        for ka in (mp.mpf("100.5"), mp.mpf("99.25")):
            limit = (1 + mp.exp(-2 * mp.pi * abs(ka - 100))) / 2
            assert abs(ss.wx_fnl(ka, 100, "b") / ss.wx_fnl(ka, 100, "b", asymptotic=True) / limit - 1) < 2e-3
        # their parametric estimates: (a) is (rho/m_A)^3 of (b) at u = 1
        m = mp.sqrt(mp.mpf(9) + mp.mpf(1) / 4)
        assert abs(ss.wx_estimate(4, 3, "a") / ss.wx_estimate(4, 3, "b") / (4 / m ** 2) ** 3 - 1) < 1e-25
        # diagram (c) is diagram (b) over 2 e^(2 pi Omega), their Eq. (37)
        assert abs(ss.wx_fnl(3.5, 3, "b") / ss.wx_fnl(3.5, 3, "c") / (2 * mp.exp(mp.pi)) - 1) < 1e-25


def test_wx_closed_form_in_the_normalization_of_the_paper():
    """|wx_M| is 0.35, 0.50, 0.61 of the exact M_A at kappa = mu = 1, 2, 3 (Sec. 7.2)."""
    with mp.workdps(30):
        inp = ss.helical_inputs()
        for mu, ref in (("1", 0.35), ("2", 0.50), ("3", 0.61)):
            ratio = abs(ss.wx_M(mu, mu)) / abs(inp[(mu, mu)]["M_A"])
            assert abs(ratio - ref) < 0.006


def test_top_quark_two_point_function():
    """Flat-space limit, de Sitter invariance, and the absence of the powers 2 + 2im and 3 + 2im on the pair branch."""
    with mp.workdps(40):
        T = ss.TopQuark(1)
        X = mp.mpf("1e-4")
        assert abs(T.two_point(mp.mpf(-1), mp.mpf(-1), X, pair=False) * mp.pi ** 4 * X ** 6 - 1) < 1e-6
        tau, taup, X = mp.mpf(-1), mp.mpf(-2), mp.mpf(5)
        Z = (tau ** 2 + taup ** 2 - X ** 2) / (2 * tau * taup)
        t2 = mp.mpf("-0.7")
        X2 = mp.sqrt(2 * t2 ** 2 * (1 - Z))
        a, b = T.two_point(tau, taup, X, pair=False), T.two_point(t2, t2, X2, pair=False)
        assert abs(a / b - 1) < 1e-25
        assert T.lower_powers() < 1e-12
        assert abs(T.cwx_map_check() + 4) < 1e-25
        late, exact, u = T.first_term()
        assert abs(u - 26.16) < 0.01


def test_conventions_of_the_neutrino_papers():
    with mp.workdps(30):
        # Song et al.: f_CC = (10/9) 2 pi^2 P |C_Dirac|, and their Eq. (20) is its large-lambda limit
        for ka, m, tol in ((10, 1, 0.06), (20, 2, 0.03)):
            assert abs(ss.song_eq20(m, ka) / ss.majorana_fcc(ss.fermion_C(ka, m)) - 1) < tol
        lam = ss.phidot0() / 30
        assert abs(ss.song_eq20(4, lam) / mp.mpf("1.38e-2") - 1) < 0.01          # their Table I
        assert abs(ss.song_eq21(4, lam) / mp.mpf("5.44e-5") - 1) < 0.01
        # You et al. (v2), Eq. (4.6), is lam^2/4 times Song et al.'s Eq. (21) at large lam
        assert abs(ss.you_eq46(4, lam) / ss.song_eq21(4, lam) * 4 / lam ** 2 - 1) < 0.01


@pytest.mark.slow
def test_helical_input_recomputed():
    """One input of the record from helical_triangle.py (about two minutes)."""
    import helical_triangle as ht
    with mp.workdps(30):
        L = ht.leading("2", "2.5")
        ref = ss.helical_inputs()[("2", "2.5")]
        assert abs(L["M_A"] / ref["M_A"] - 1) < 1e-12
        assert abs(L["R_hard"] / ref["R_hard"] - 1) < 1e-12
