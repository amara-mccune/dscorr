"""Input checking."""
import math
import warnings

import numpy as np
import pytest

import dscorr
from dscorr import ValidityWarning, TESTED
from dscorr._validate import check_family, check_momenta, check_mu


def test_tested_constant():
    assert TESTED == {"mu_min": 0.3, "mu_max": 4.0, "momentum_ratio_min": 0.002}
    assert dscorr.TESTED is dscorr._validate.TESTED
    assert issubclass(ValidityWarning, UserWarning)
    # equal to the tested_range of the packaged reference points
    ref = dscorr.data.reference_points()["tested_range"]
    for k in ref:
        assert TESTED[k] == ref[k]


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), "x", None, 1 + 2j])
def test_bad_mass(bad):
    with pytest.raises(ValueError) as e:
        check_mu(bad, where="double_exchange_direct")
    msg = str(e.value)
    assert msg.startswith("double_exchange_direct:") and "mu" in msg


def test_mass_messages():
    with pytest.raises(ValueError, match=r"mass parameter mu must be a real number, got 'x'"):
        check_mu("x")
    with pytest.raises(ValueError, match="mass parameter mu must be finite, got nan"):
        check_mu(float("nan"))
    with pytest.raises(ValueError, match=r"pass abs\(mu\)\), got mu = 0.0"):
        check_mu(0.0, where="f")
    assert check_mu("1.5", 1) == (1.5, 1.0)


def test_mass_warnings():
    for mu in (0.1, 5.0):
        with pytest.warns(ValidityWarning, match="tested range"):
            check_mu(mu, where="triple_exchange_direct")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_mu(0.3, 4.0)


@pytest.mark.parametrize("ks", [(1, 1, 0), (1, -1, 1), (1, 1, float("nan")), (1, 1, 2.5),
                                (3, 1, 1), (1, 0.3, 0.5)])
def test_bad_momenta(ks):
    with pytest.raises(ValueError, match="moment"):
        check_momenta(*ks, where="g")


def test_momentum_messages():
    with pytest.raises(ValueError, match=r"^double_exchange_direct: .*\(k1, k2, k3\)"):
        check_momenta(1, 1, 0, where="double_exchange_direct")
    with pytest.raises(ValueError, match=r"triangle .*\(1.0, 0.3, 0.5\)"):
        check_momenta(1, 0.3, 0.5, where="triple_exchange_direct")
    with pytest.raises(ValueError, match="momentum k2 must be a real number, got 'a'"):
        check_momenta(1, "a", 1)
    with pytest.raises(ValueError, match="momentum k1 must be a real number"):
        check_momenta([1, 2], 1, 1)


def test_folded_and_degenerate_accepted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_momenta(2, 1, 1) == (2.0, 1.0, 1.0)
        check_momenta(1, 0.5, 0.5 + 5e-13)
        check_momenta(1, 0.5, 0.5 - 5e-13)
    with pytest.raises(ValueError):
        check_momenta(1, 0.5, 0.5 - 5e-12)


def test_ratio_warning():
    with pytest.warns(ValidityWarning, match=r"1.9950e-03 < tested minimum 2.0000e-03"):
        check_momenta(1, 1, 1.995e-3, where="x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_momenta(1, 1, 0.002)


def test_family():
    assert check_family([]).size == 0
    with pytest.raises(ValueError, match=r"kappa = np.float64\(-1.0\)"):
        check_family([0.5, -1.0])
    with pytest.raises(ValueError, match=r"triangle inequality \(k3 > k1 \+ k2\)"):
        check_family([2.5])
    np.testing.assert_array_equal(check_family(2.0 + 5e-13), [2.0 + 5e-13])
    with pytest.warns(ValidityWarning):
        check_family([0.1, 0.001, 0.0005])
