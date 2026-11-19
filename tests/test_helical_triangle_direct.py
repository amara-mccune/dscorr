"""The series of the helical-vector triangle against the stored direct time-domain values (examples/Loops/direct).

    pytest tests/test_helical_triangle_direct.py     # seconds: the fits of direct/data_vec, and the stored coefficients

The direct values were computed with direct/direct_vec.py on Nov 19, 2026 (the integration itself needs python-flint
and the mode tables of loops_numcheck_code.zip, and is not repeated here). Each fit leaves one coefficient of the
series free, and the fitted factor must be one.
"""
import json
import os
import subprocess
import sys

import mpmath as mp
import pytest

LOOPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "Loops")
DATA = os.path.join(LOOPS, "direct", "data_vec")
FIT = os.path.join(LOOPS, "direct", "vec_direct_fit.py")
sys.path.insert(0, LOOPS)
from ext import shapevars  # noqa: E402


def fit(tmp_path, series, data, *opts):
    out = tmp_path / "fit.json"
    subprocess.run([sys.executable, FIT, os.path.join(DATA, series), os.path.join(DATA, data), f"--json={out}", *opts],
                   check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))


# (series, data, options, bounds on |a - 1| for the first three coefficients and for the sum of the corrections)
CASES = [
    ("series_1_1_n6.json", "c-1_pc0.5.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_1_1_n6.json", "c+1_pc0.5.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_1_1_n6.json", "c+0.5_pc0.5_nphi6.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_1_1_n6.json", "c-1_pc1.0.jsonl", ("--sym",), (1e-4, 3e-3, 5e-2, 3e-3)),
    ("series_1_1_n8_isosceles.json", "iso_pc1.0_nphi8.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_2_2.5_n8.json", "mu2_kappa2.5_c-1_pc0.5.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_2_2.5_n8.json", "mu2_kappa2.5_c+1_pc0.5.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
    ("series_2_2.5_n8.json", "mu2_kappa2.5_c+0.5_pc0.5_nphi6.jsonl", (), (1e-4, 1e-3, 3e-2, 1e-3)),
]


@pytest.mark.parametrize("series,data,opts,bounds", CASES)
@pytest.mark.parametrize("nbg", [5, 6])
def test_free_coefficients(tmp_path, series, data, opts, bounds, nbg):
    """Backgrounds of 12 and 14 parameters: each of the first three coefficients, left free, comes back as the series."""
    d = fit(tmp_path, series, data, f"--nbg={nbg}", *opts)
    assert d["npoints"] >= d["nparam"] + 4
    for n in range(3):
        assert d["free"][str(n)]["dev"] < bounds[n]
    assert d["free_corrections"]["dev"] < bounds[3]
    res = d["residual"]                      # none, then through n = 0, 1, 2, ...
    assert res[1] < res[0] and res[2] < res[1] and res[3] < res[2]


@pytest.mark.parametrize("nbg", [5, 6])
def test_contact_term(tmp_path, nbg):
    """Runs with and without the contact term of the hard line: their difference is the contact part of the series."""
    d = fit(tmp_path, "series_1_1_n6_contact.json", "c-1_pc0.5.jsonl", f"--nbg={nbg}",
            "--minus=" + os.path.join(DATA, "c-1_pc0.5_nocontact.jsonl"), "--norm=" + os.path.join(DATA, "series_1_1_n6.json"))
    assert d["free_corrections"]["orders"][0] == 2          # the contact term starts at order r^2
    assert d["free_corrections"]["dev"] < 5e-2
    assert d["free"]["2"]["dev"] < 5e-2


@pytest.mark.parametrize("name", ["series_1_1_n6.json", "series_2_2.5_n8.json", "series_3_3.75_n6.json"])
def test_stored_series_is_symmetric(name):
    """The stored coefficients satisfy the exchange identity k1 <-> k2 at every order, and D_N is even in delta."""
    mp.mp.dps = 30
    d = json.load(open(os.path.join(DATA, name)))
    cz = lambda p: mp.mpc(p[0], p[1])
    C = {tuple(map(int, k.split(","))): cz(v) for k, v in d["coef"].items()}
    nu, N = mp.mpc(0, mp.mpf(d["mu"])), int(d["nmax"])
    assert max(shapevars.exchange_error(C, nu, N, dim=2).values()) < 1e-20
    D = shapevars.symmetric(C, nu, N, dim=2)
    scale = abs(D[(0, 0)])
    assert max(abs(v) for (n, md), v in D.items() if md % 2) < 1e-20 * scale
    for k, v in d["sym"].items():
        assert abs(D[tuple(map(int, k.split(",")))] - cz(v)) < 1e-20 * scale
