"""Bessel routines, time grid, line blocks, leg functions, chain block."""
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import cumulative_simpson as scipy_cumsimp

from dscorr._engines.bessel import besseli_imag, besselk_imag
from dscorr._engines.contour import InnerContour, leg_function, thermal_weights, trap_grid
from dscorr._engines.grid import LogGrid, cumulative_simpson
from dscorr._engines.lineblock import LineBlock
from dscorr._engines.trees import SIGNS2, DoubleContour


def rel(a, b):
    return abs(a / b - 1)


def test_bessel_checkpoints():
    assert rel(besselk_imag(1.0, 0.5), 0.48339609004387946) < 1e-15
    assert rel(besseli_imag(1.0, 0.5), 0.8699973632047564 - 1.7770016885719386j) < 1e-15
    assert rel(besselk_imag(1.0, 5.0), 0.003367099988561044) < 1e-15
    assert rel(besseli_imag(1.0, 5.0), 30.542593924781315 - 0.0123777218899806j) < 1e-15


def test_bessel_against_mpmath():
    with mp.workdps(30):
        for mu in (0.3, 1.0, 4.0):
            for z in (0.001, 0.5, 1.9, 2.0, 5.0, 30.0, 300.0):
                I = complex(mp.besseli(1j * mu, z))
                K = float(mp.re(mp.besselk(1j * mu, z)))
                assert rel(besseli_imag(mu, np.array([z]))[0], I) < 1e-14
                tol = 1e-13 if z <= 30 else 1e-11
                assert rel(besselk_imag(mu, np.array([z]))[0], K) < tol


def test_bessel_cuts():
    assert besseli_imag(1.0, np.array([401.0]))[0] == 0
    assert besselk_imag(1.0, np.array([700.0]))[0] == 0


def test_grid_node_counts():
    assert LogGrid(-60.0, 10.6, 0.004).nodes == 17651
    assert LogGrid(-60.0, 10.6, 0.002).nodes == 35301
    assert LogGrid(-54.0, 10.6, 0.005).nodes == 12921
    assert LogGrid(-60.0, 16, 0.004).nodes == 19001
    assert LogGrid(-60.0, 12.0, 0.002).nodes == 36001


def test_cumulative_simpson_equals_scipy():
    rng = np.random.default_rng(1)
    for n in (11, 12, 101):
        y = rng.random(n) + 1j * rng.random(n)
        ref = scipy_cumsimp(y, dx=0.3, initial=0)
        assert np.abs(cumulative_simpson(y, 0.3) - ref).max() < 1e-15 * np.abs(ref).max()


def test_line_block_checkpoints():
    g = LogGrid(-60.0, 10.6, 0.004)
    b = LineBlock(g, 1.0, 1.0)
    assert rel(b.Phi, 0.3396668700128083) < 1e-15
    j = int(np.argmin(np.abs(g.L)))
    assert g.L[j] == 0.0
    assert rel(b.K[j], 0.2894280370259932) < 1e-15
    assert rel(b.J[j], 1.90079967581943 - 1.0679412170121854j) < 1e-15
    assert rel(b.FF[j], 0.0005865134220175566 - 0.3627442505401982j) < 1e-14
    b4 = LineBlock(g, 4.0, 1.0)
    assert rel(b4.Phi, 2.7462140928356406e-05) < 1e-14
    g2 = LogGrid(-60.0, 10.6, 0.002)
    assert rel(LineBlock(g2, 1.0, 1.0).Phi, 0.33966687001280743) < 1e-15


@pytest.mark.parametrize("mu,k", [(0.3, 1.0), (1.0, 0.5), (2.0, 0.1), (4.0, 1.0)])
def test_phi_closed_form(mu, k):
    g = LogGrid(-60.0, 10.6, 0.004)
    phi_hat = math.pi * math.sqrt(math.pi / 2) / math.cosh(math.pi * mu)
    assert rel(LineBlock(g, mu, k).Phi * math.sqrt(k), phi_hat) < 1e-11


def test_thermal_weights():
    ref = {0.3: (1.179016990265822, -0.17901699026582202),
           1.0: (1.0018709365986607, -0.0018709365986606444),
           1.5: (1.0000807060305081, -8.070603050803164e-05),
           4.0: (1.0000000000121616, -1.2161556709557224e-11)}
    for mu, (wp, wm) in ref.items():
        a, b = thermal_weights(mu)
        assert rel(a, wp) < 1e-15 and rel(b, wm) < 1e-15
        assert abs(a + b - 1) < 1e-15


def test_leg_function_checkpoints():
    inner = InnerContour()
    sp = np.array([-0.25, -0.25 + 1j, -0.25 - 3j, -0.25 + 12j])
    refp = [3.294119956128516 - 4.517445028255857j, 0.07306087359504326 - 0.11687745328936708j,
            -0.0002722734302669733 + 0.002023285328945772j,
            3.218305322950871e-17 + 4.044328532044022e-17j]
    refm = [3.294119956128512 + 4.517445028255854j, -0.983231468606345 + 0.6859175726107218j,
            0.00011982003523877459 + 0.00011798022821502785j,
            -5.469956283094478e-16 + 5.657597022741627e-16j]
    # reference values from the production run of the Letter; this code agrees to a uniform
    # 2e-14 relative (floating-point rounding of the trapezoidal sum)
    for nu, ref in ((1j, refp), (-1j, refm)):
        M = leg_function(sp, 1.0, 1.0, nu, inner)
        for a, b in zip(M, ref):
            assert rel(a, b) < 5e-14


def test_leg_function_power_law_and_profile():
    inner = InnerContour()
    u, _ = trap_grid(12.0, 0.02)
    s = -0.25 + 1j * u
    nu = 1.5j
    Mh = leg_function(s, 0.5, 0.5, nu, inner)
    M1 = leg_function(s, 1.0, 1.0, nu, inner)
    assert np.abs(Mh - 0.5 ** (nu + 2 * s - 0.5) * M1).max() / np.abs(Mh).max() < 1e-14
    Mi = leg_function(s, 1.0, 1.0, 1j, inner)
    assert rel(np.abs(Mi).max(), 6.185639012559014) < 1e-13
    assert abs(Mi[0]) < 1e-15 and abs(Mi[-1]) < 1e-16


def test_chain_block_checkpoints():
    dc = DoubleContour(1.0, 1.5)
    refs = {(1, 1): -0.037430764003942414 + 0.09967558217066742j,
            (1, -1): 0.07159149792132427 + 0.12273800623768934j,
            (-1, 1): 0.07159149792132419 - 0.12273800623768921j,
            (-1, -1): -0.03743076400394238 - 0.09967558217066733j}
    wref = {(1, 1): 1.001951793625035, (1, -1): -8.085702637424174e-05,
            (-1, 1): -0.0018710875945268546, (-1, -1): 1.5099586621009893e-07}
    for sg in SIGNS2:
        assert rel(dc.chain(0.8, 1.0, 0.5, *sg), refs[sg]) < 1e-12
        assert rel(dc.weight(*sg), wref[sg]) < 1e-15
    psi = dc.psiF(0.8, 1.0, 0.5)
    assert rel(psi, -0.037643569421665926 + 0.10008984261655977j) < 1e-12
    assert rel(dc.psiF(0.8, 1.0, 0.5, method="engine"), psi) < 1e-12
