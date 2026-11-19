"""The direct integration of the fermion triangle against the series in the symmetric variables (stored runs).

Each test fits an analytic background to a stored run of examples/Loops/direct/direct_fermi2.py and applies the
criteria of the verification of Nov 19, 2026: the series with no free parameter leaves at most 1% of its envelope,
a free factor on the series returns one within 1%, a free factor on the corrections returns one within 3%, and the
leading term alone leaves a residual at least ten times larger. Needs numpy.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = os.path.join(HERE, "..", "examples", "Loops", "direct")
sys.path.insert(0, DIRECT)
import fermi_shape_fit as F  # noqa: E402

DATA = os.path.join(DIRECT, "data")
RUNS = [("series_1_1.3_n10.json", "iso_pc1.8.jsonl", 0.29), ("series_1_1.3_n10.json", "collinear_pc1.0.jsonl", 0.28),
        ("series_1_1.3_n10.json", "collinear_pc1.2.jsonl", 0.22), ("series_1_1.3_n10.json", "delta-0.5_pc1.5.jsonl", 0.27),
        ("series_1_5.json", "kappa5_collinear_pc1.0.jsonl", 0.16)]


@pytest.mark.parametrize("series, run, reach", RUNS)
def test_direct_run_against_series(series, run, reach):
    S = F.load_series(os.path.join(DATA, series))
    out = F.report(S, [os.path.join(DATA, run)], quiet=True)
    assert out["R"][1] > reach and len(out["fits"]) == 3
    for a in out["fits"].values():
        assert a["res_full"] < 1e-2
        assert a["s_all_dev"] < 1e-2
        assert a["s_corr_dev"] < 3e-2
        assert a["power"] > 10


def test_symmetric_and_fixed_angle_forms_agree():
    S = F.load_series(os.path.join(DATA, "series_1_1.3_n10.json"))
    r = np.array([0.01, 0.05])
    for c in (np.array([-1.0, -1.0]), np.array([0.5, 0.5]), -r / 2):
        a, b = F.terms(S, r, c).sum(0), F.terms_C(S, r, c).sum(0)
        assert np.max(np.abs(a / b - 1)) < 1e-8
