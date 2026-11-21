"""The direct integration of the contact-pair diagram of the collapsed trispectrum against its series (stored runs).

The diagram with one quartic vertex and one hard pair is examples/Loops/direct/direct_vec.py with X3=2 PW3=2 at equal
external momenta. The stored run of Nov 20, 2026 (mu = kappa = 1, 0.01 <= s/k <= 0.35, loop cut 0.5) is fitted with an
analytic background together with the series of parity_box_series.py through N = 4. Needs numpy.
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
import cp_direct_fit as F  # noqa: E402


@pytest.mark.parametrize("rmax", [0.2, 0.25])
def test_series_against_direct_run(rmax):
    D = F.load()
    out = F.fits(D, rmax)
    assert len(out) == 5
    for e in out:
        if e["kind"] == "even":                       # all three kinds of term are predictions
            assert abs(e["c"] - 1) < 1e-4 and abs(e["lam_p"] - 1) < 5e-4
            assert abs(e["lam_s"] - 1) < 5e-4 and abs(e["lam_0"] - 1) < 3e-2
        elif e["terms"] >= 9:                         # powers and logarithms: a free factor on the pair orders N >= 1
            assert abs(e["c"] - 1) < 1e-4 and abs(e["lam_p"] - 1) < 5e-4
            assert e["residual_without_corrections"] > 1e3 * e["residual"]
        else:
            assert abs(e["lam_p"] - 1) < 5e-3


def test_loop_cut_changes_only_even_powers():
    res, size, n = F.cut_difference(None, os.path.join(F.DATA, "cp_1_1_pc0.8.jsonl"))
    assert n == 12 and size > 1e-3 and res < 1e-10
