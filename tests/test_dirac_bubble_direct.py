"""The direct integration of the Dirac-pair bubble against its squeezed series (stored runs).

Each test fits an analytic background to a stored run of examples/Loops/direct/direct_dirac_bubble.py and applies the
criteria of the check of Nov 20, 2026: with no free parameter the series leaves a residual at least 100 times smaller
than the leading term alone, a free factor on the corrections returns one within 1e-3, and the first two corrections,
fitted one at a time, are confirmed to 1% and 5%. The series is the sum of the two pair branches over the four
Schwinger-Keldysh sectors; the + branch alone does not fit. Needs numpy.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = os.path.join(HERE, "..", "examples", "Loops", "direct")
if not os.path.isdir(DIRECT):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, DIRECT)
import dirac_bubble_fit as F  # noqa: E402

# point, background sizes that meet the criteria (the slow oscillation at (1, 0.5) needs 14 terms or more)
CASES = [("(1,1)", (5, 6, 7)), ("(2,2)", (5, 6, 7)), ("(1,0.5)", (6, 7))]


@pytest.mark.parametrize("point, nbgs", CASES)
def test_direct_run_against_series(point, nbgs):
    ser, run, rmin, wpow = F.RUNS[point]
    S = F.load_series(os.path.join(F.DATA, ser))
    out = F.report(S, [os.path.join(F.DATA, run)], rmin=rmin, wpow=wpow, nbgs=nbgs, quiet=True)
    assert len(out["fits"]) == len(nbgs) and out["corr_over_lead_max"] > 1
    for a in out["fits"].values():
        assert a["power"] > 100
        assert a["s_corr_dev"] < 1e-3
        assert a["s_single"][1][0] < 1e-2
        assert a["s_single"][2][0] < 5e-2


@pytest.mark.parametrize("point", ["(1,1)", "(2,2)", "(1,0.5)"])
def test_plus_branch_alone_does_not_fit(point):
    ser, run, rmin, wpow = F.RUNS[point]
    S = F.load_series(os.path.join(F.DATA, ser), model="plus")
    out = F.report(S, [os.path.join(F.DATA, run)], rmin=rmin, wpow=wpow, nbgs=(7,), quiet=True)
    assert out["fits"][7]["s_corr_dev"] > 0.2 and out["fits"][7]["power"] < 20


@pytest.mark.parametrize("part", ["D", "T"])
def test_sectors_separately(part):
    """The doubly cut and the time-ordered terms each carry their own combination of the two branches."""
    for point in ("(1,1)", "(2,2)"):
        ser, run, rmin, wpow = F.RUNS[point]
        S = F.load_series(os.path.join(F.DATA, ser), part=part)
        out = F.report(S, [os.path.join(F.DATA, run)], rmin=rmin, wpow=wpow, nbgs=(7,), part=part, quiet=True)
        assert out["fits"][7]["s_corr_dev"] < 1e-3 and out["fits"][7]["power"] > 100


def test_loop_cut_changes_only_analytic_terms():
    res, npts = F.cut_difference(os.path.join(F.DATA, "m1_k1_pc1.5.jsonl"), os.path.join(F.DATA, "m1_k1_pc1.0.jsonl"))
    assert npts == 36 and res < 1e-11


def test_values_of_record():
    assert F.check() == 0
