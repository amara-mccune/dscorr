"""The direct integration of the box of the helical-vector loop against the collapsed series (stored runs).

examples/Loops/direct/direct_box.py integrates the box (a hard pair on both sides) over the four vertex times and the
loop momentum. The stored runs of Nov 22, 2026 (mu = kappa = 1, equal external momenta, 32 values of s/k from 0.01 to
0.35, loop cuts 0.8 and 1.2) are fitted with an analytic background together with the series of parity_box_series.py
(pair and single-frequency terms through N = 6, non-oscillating terms through N = 4). Needs numpy; the test of the
kernel also needs numba and python-flint.
"""
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
HERE = os.path.dirname(os.path.abspath(__file__))
LOOPS = os.path.join(HERE, "..", "examples", "Loops")
DIRECT = os.path.join(LOOPS, "direct")
if not os.path.isdir(DIRECT):
    pytest.skip("examples/Loops not found (tests run outside the source tree)", allow_module_level=True)
sys.path.insert(0, DIRECT)
import box_direct_fit as F  # noqa: E402
import box_template as T  # noqa: E402

RECORD = json.load(open(os.path.join(LOOPS, "box_direct_record.json")))


@pytest.fixture(scope="module")
def summary():
    return F.summary()


def test_series_against_direct_runs(summary):
    """All three kinds of term are predictions: free factors return one, and the orders N >= 1 are needed."""
    for key in ("A_0", "A_2", "D_2", "A_1", "D_1"):
        e = summary[key]
        assert e["c"] < 1e-4 and e["lam_p"] < 5e-4 and e["lam_s"] < 5e-4 and e["lam_0"] < 5e-4
        assert e["residual"] < 1e-5 and e["residual_without_corrections_min"] > 1e3 * e["residual"]
    for key in ("I_-2", "I_2"):                    # the harmonic itself: background of both parities
        e = summary[key]
        assert e["c"] < 1e-4 and e["lam_p"] < 1e-3 and e["lam_s"] < 5e-3 and e["lam_0"] < 2e-2


def test_harmonics_absent_at_leading_order(summary):
    """|n| = 3 and 4 have no term r^3 times an oscillation or a logarithm (they start at N = 2 and N = 4)."""
    assert max(summary["leading_absent"].values()) < 1e-3


def test_loop_cut_changes_only_powers_of_the_right_parity(summary):
    for e in summary["cut_difference"].values():
        assert e["points"] == 32 and e["size"] > 1e-4 and e["residual"] < 1e-10


def test_record(summary):
    for key, e in RECORD["fits_cut_0.8"].items():
        if key == "cut_difference":
            for k, v in e.items():
                assert abs(summary[key][k]["leading_coefficient"] / v["leading_coefficient"] - 1) < 1e-6
            continue
        for k, v in e.items():
            assert summary[key][k] == pytest.approx(v, rel=0.05, abs=1e-12)


def test_parity_odd_box_without_cut(capsys):
    """The regions of loop momentum add up to the stored harmonics, and the outer shells fall off quickly."""
    R = T.main(["-"])
    capsys.readouterr()
    rec = RECORD["template_odd"]
    assert [e["r"] for e in R["points"]] == rec["r"]
    for n in (1, 2, 3, 4):
        assert np.allclose([e["D_M20"][n] for e in R["points"]], rec["D_M20"][str(n)], rtol=1e-9)
        assert np.allclose([e["D"][n] for e in R["points"]], rec["D"][str(n)], rtol=1e-9)
    e = [x for x in R["points"] if abs(x["r"] - 0.3) < 1e-9][0]
    shells = [abs(v[2]) for k, v in e["regions"].items() if float(k.split("-")[0]) >= 2.6]
    assert len(shells) == 5 and all(b < 0.12 * a for a, b in zip(shells[:-1], shells[1:])) and shells[-1] < 1e-9
    # every region near the ring at M = 40 against the modeled correction of the M = 20 values
    assert e["D_M40"][2] == pytest.approx(e["D"][2], rel=1e-3) and abs(e["D_M40"][2] / e["D_M20"][2] - 1) < 0.12
    b1 = R["collapsed_region"][2]["analytic_coefficients_r1_r3_r5"][0]
    assert b1 == pytest.approx(2.4019e-3, rel=1e-3)


def test_kernel_against_reference():
    """The compiled kernel of the 4-cycle against the sum over all orderings and branches, sector by sector."""
    pytest.importorskip("numba")
    pytest.importorskip("flint")
    import direct_box as D
    rng = np.random.default_rng(1)
    t, h, wint = D.tgrid2(tlo=1e-3, tsplit=0.05, thi=30.0, w_early=1.5, w_late=0.45)
    M = 2
    shape = (4, M, 2, 4, 3, 3)
    T4 = np.zeros(shape + (t.size,), complex)
    for _ in range(3):
        co = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        a, b = rng.uniform(0.5, 2.0, size=shape), rng.uniform(0.8, 2.0, size=shape)
        T4 += co[..., None] * t ** (1 + a[..., None]) * np.exp(-b[..., None] * t)
    Z, Zs = D.box_kernel(T4, D.Q_N, h, wint)
    secs = [(0, 0, 0, 0), (0, 1, 1, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 0, 1)]
    for j in range(M):
        for jp in range(M):
            A = [T4[0, j], T4[1, j], T4[2, jp], T4[3, jp]]
            for i, sc in enumerate(secs):
                ref = D.cycle_sum(A, h, wint, only=sc)
                assert abs(ref - Zs[j, jp, i]) < 1e-11 * abs(ref)
            assert abs(D.cycle_sum(A, h, wint) - Z[j, jp]) < 1e-11 * abs(Z[j, jp])
