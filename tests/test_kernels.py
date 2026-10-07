"""dscorr.kernels."""
import mpmath as mp
import pytest

from dscorr import IdentityCheck, NumericVsClosedForm, kernel_identity_check
from dscorr.kernels import (K_barnes, barnes_2F1_lemma, kernel_identity_lhs, kernel_identity_rhs,
                            mellin_transform_lhs, mellin_transform_rhs)


def test_identity_documented_point():
    r = kernel_identity_check("1.3j", 1.0, 0.7, 1.9, dps=20)
    assert isinstance(r, IdentityCheck)
    assert mp.nstr(r.lhs, 15) == "(0.0945812503519151 - 0.232173510073162j)"
    assert mp.nstr(r.rhs, 15) == "(0.0945812503519151 - 0.232173510073162j)"
    assert 1e-22 < r.rel < 1e-20                   # documented 1.88873987504497e-21
    assert abs(r.rel / mp.mpf("1.88873987504497e-21") - 1) < 1e-3


@pytest.mark.parametrize("nu", ["2/3", "1.3j", "1/2", "1", "0.3+0.7j"])
def test_identity_five_orders(nu):
    assert kernel_identity_check(nu, 0.7, 0.6, 1.1, dps=20).rel < 1e-15


def test_identity_equal_times():
    assert kernel_identity_check("1.3j", 1.0, 0.9, 0.9, dps=20).rel < 1e-15


def test_table_first_row():
    v = kernel_identity_lhs("2/3", "0.7", "0.6", "1.1", dps=45)
    assert mp.nstr(v, 35) == "0.28929095302715905065592015783088425"


def test_mellin_pair():
    v = mellin_transform_rhs("1.0j", 1.0, 0.3, 0.3, dps=20)
    assert mp.nstr(v, 15) == "(-1.16949828706756 - 1.87952550149032j)"
    with mp.workdps(30):
        lhs = mellin_transform_lhs("1.0j", 1.0, 0.3, 0.3)
        rhs = mellin_transform_rhs("1.0j", 1.0, 0.3, 0.3)
        assert abs(lhs / rhs - 1) < 1e-25
        lhs = mellin_transform_lhs("0.5j", 0.8, "0.4+0.2j", 0.5)
        rhs = mellin_transform_rhs("0.5j", 0.8, "0.4+0.2j", 0.5)
        assert abs(lhs / rhs - 1) < 1e-25


def test_K_barnes():
    r = K_barnes("1.3j", 0.9)
    assert isinstance(r, NumericVsClosedForm)
    assert mp.nstr(mp.re(r.numeric), 15) == "0.240452626377056"
    assert r.rel < 1e-25


def test_barnes_lemma():
    assert barnes_2F1_lemma(0.5, 0.7, 6, 8, 0.3).rel < 1e-25


def test_domain_errors():
    with pytest.raises(ValueError):
        kernel_identity_lhs("abc", 1, 1, 1)
    with pytest.raises(ValueError):
        kernel_identity_lhs(1, -1, 1, 1)
    with pytest.raises(ValueError):
        kernel_identity_lhs(1, 1, float("inf"), 1)
    with pytest.raises(ValueError):
        kernel_identity_rhs("1.3j", 1, 1, 2, b0=0.4)
    with pytest.raises(ValueError):
        kernel_identity_rhs("-0.5", 1, 1, 2, b0=0.7)
    with pytest.raises(ValueError):
        mellin_transform_rhs("1.0j", 1.0, 1.2, 1.0)
    with pytest.raises(ValueError):
        mellin_transform_lhs("1.0j", 1.0, -0.1, 0.5)
    with pytest.raises(ValueError):
        K_barnes("1.3j", 0.9, sigp=0.1)
    with pytest.raises(ValueError):
        K_barnes("1.3j", -0.9)
    with pytest.raises(ValueError):
        barnes_2F1_lemma(0.5, 0.7, 6, 8, 1.3)
    with pytest.raises(ValueError):
        barnes_2F1_lemma(-0.5, 0.7, 6, 8, 0.3)
    with pytest.raises(ValueError):
        barnes_2F1_lemma(0.5, 0.7, 1, 1, 0.3)
