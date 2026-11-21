"""The soft vertex with a chemical potential in closed form (Secs. 6.1 and 8.2; ext/softclosed.py).

For first indices that are equal or opposite up to an integer, the soft moment

    J(ia, ib, q; K1, K2) = int d^3l/(2 pi)^3 y1^(nu - 1/2 + ia) y2^(nu - 1/2 + ib)
                           int dt t^(q-1) e^(-t) W_{K1,nu}(2 y1 t) W_{K2,nu}(2 y2 t)

(ia, ib >= -1, integer q >= 0) is a finite sum of Gamma functions, Gauss functions at argument -1, 3F2 at unit
argument, and one-fold Barnes integrals of Gamma functions (Meijer functions G^{3,4}_{4,4} at argument -1). This
covers every pair of a Dirac line. The script checks

  * scalar lines (K = 0) against the ratio of Gamma functions of Eq. (3.17);
  * the simplest moment as one Barnes integral,
        J(-1, -1, 0; K, K) = Gamma(-1 - 2 nu)/(2 pi Gamma(c)^2)
            (1/2 pi i) int ds tan(pi s/2) Gamma(-s) Gamma(2 nu + 1 + s) Gamma(c - 1/2 + s/2)/Gamma(c + 1/2 + 2 nu + s/2),
    c = 1/2 - nu - K, against the general routine;
  * single moments for the four kinds of Dirac pair against soft_J (position space);
  * the squeezed coefficient C of the fermion triangle with closed-form soft moments against its values of record
    (soft_closed_record.json; they equal twice the closed form of Song et al. to 3e-29 or better), and against C with
    soft_J.

The closed form loses about five digits at (kappa, m) = (20, 2), so every point is run at 30 digits. The values of
record were computed at 30, 45, 60, and 90 digits.

Usage:  python soft_closed.py          kappa = 1.3, m = 1 (about 2 min)
        python soft_closed.py --all    also C at (5, 1), (10, 1), and (20, 2) (about 5 min)
        python soft_closed.py --song   also compares C with song_check.Cplus at each point (which is evaluated at
                                       30, 45, 60, and 90 digits; with --all about 7 min)
"""
import json
import os
import sys
import time

import mpmath as mp

from ext.blocks import two_moment
from ext.fermitri import FermiSoftTable, FermiTriangle
from ext.record import compare, report
from ext.softclosed import ClosedSoftTable, SoftClosed
from ext.softvertex import Grid, Leg, soft_J

I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))


def dirac_K(kappa):
    k = mp.mpf(kappa)
    return {"a": 0.5 + I * k, "b": 0.5 - I * k, "c": -0.5 + I * k, "d": -0.5 - I * k}


def scalar_check(mu=1):
    """max over q = 1, 2, 3 of |J(0, 0, q; 0, 0)/[2^(2 nu + 1)/(pi (2 pi)^3) M(r2 = q + 1)] - 1|, M of Eq. (3.17)"""
    nu = I * mp.mpf(mu)
    S = SoftClosed(nu)
    pref = mp.mpf(2) ** (2 * nu + 1) / (mp.pi * (2 * mp.pi) ** 3)
    return max(abs(S.J(0, 0, q, 0, 0) / (pref * two_moment(nu, nu, 1, 1, q + 1)) - 1) for q in (1, 2, 3))


def simplest_moment(nu, K, closed):
    """Eq. (simplest) of the report on the contour of `closed`, relative to closed.J(-1, -1, 0, K, K)."""
    c = 0.5 - nu - K
    g = closed._grid_for(2 * c - 1)
    val = g.w * mp.fsum(mp.tan(mp.pi * s / 2) * mp.gamma(-s) * mp.gamma(2 * nu + 1 + s) * mp.gamma(c - 0.5 + s / 2)
                        * mp.rgamma(c + 0.5 + 2 * nu + s / 2) for s in g.s)
    m = 0
    while mp.re(1 - 2 * c - 2 * m) > g.sig:                     # poles of Gamma(c - 1/2 + s/2) to the right of the line
        s0 = 1 - 2 * c - 2 * m
        val += (mp.tan(mp.pi * s0 / 2) * mp.gamma(-s0) * mp.gamma(2 * nu + 1 + s0) * 2 * (-1) ** m / mp.factorial(m)
                * mp.rgamma(1 + 2 * nu - m))
        m += 1
    val *= mp.gamma(-1 - 2 * nu) / (2 * mp.pi * mp.gamma(c) ** 2)
    return abs(val / closed.J(-1, -1, 0, K, K) - 1)


MOMENTS = [(-1, -1, 1), (-1, -1, 0), (0, 0, 1), (1, 1, 1), (-1, 3, 0), (2, -1, 1)]
PAIRS = [("c", "a"), ("b", "d"), ("a", "b"), ("c", "d")]       # K1 - K2 = -1, +1; K1 + K2 = +1, -1


def moments_check(kappa, m, moments=MOMENTS, pairs=PAIRS, npts=24):
    nu = I * mp.sqrt(mp.mpf(m) ** 2 + mp.mpf(kappa) ** 2)
    K = dirac_K(kappa)
    S = SoftClosed(nu, kmax=kappa)
    grid, legs, worst = Grid(npts=npts), {}, mp.mpf(0)
    for K1, K2 in pairs:
        for ia, ib, q in moments:
            for key in ((ia, K1), (ib, K2)):
                if key not in legs:
                    legs[key] = Leg(nu - mp.mpf(1) / 2 + key[0], K[key[1]], nu)
            ref = soft_J(legs[(ia, K1)], legs[(ib, K2)], q, grid)
            worst = max(worst, abs(S.J(ia, ib, q, K[K1], K[K2]) / ref - 1))
    return worst


def triangle_C(kappa, m, table):
    T = FermiTriangle(m, kappa, nmax=0, table=table)
    return T.at_angle(T.coefficients(), 1)[0]


def main(full=False, song=False):
    with open(os.path.join(HERE, "soft_closed_record.json")) as f:
        table = json.load(f)
    res = []
    with mp.workdps(30):
        res.append(compare("soft_closed/scalar vs Eq. (3.17)", scalar_check(), table))
        ka = mp.mpf("1.3")
        nu = I * mp.sqrt(1 + ka ** 2)
        S = SoftClosed(nu, kmax=ka)
        res.append(compare("soft_closed/one Barnes integral, Dirac", simplest_moment(nu, -0.5 + I * ka, S), table))
        nuh = I * mp.mpf(1)
        res.append(compare("soft_closed/one Barnes integral, helical",
                           simplest_moment(nuh, I * mp.mpf("0.75"), SoftClosed(nuh, kmax=1)), table))
        t = time.time()
        res.append(compare("soft_closed/moments vs soft_J kappa=1.3,m=1", moments_check("1.3", 1), table))
        print(f"   ({len(MOMENTS) * len(PAIRS)} moments, {time.time() - t:.0f} s)")
    pts = [("1.3", 1, 30)] + ([("5", 1, 45), ("10", 1, 60), ("20", 2, 90)] if full else [])
    for ka, m, dps_song in pts:
        with mp.workdps(30):
            dps = 30
            kam, mm = mp.mpf(ka), mp.mpf(m)
            nu = I * mp.sqrt(mm ** 2 + kam ** 2)
            t = time.time()
            C = triangle_C(kam, mm, ClosedSoftTable(nu, dirac_K(kam)))
            print(f"   kappa = {ka}, m = {m}: C = {mp.nstr(C, 25)}   ({time.time() - t:.0f} s, {dps} digits)")
            res.append(compare(f"soft_closed/C kappa={ka},m={m}", C, table))
            if ka == "1.3":
                C2 = triangle_C(kam, mm, FermiSoftTable(nu, kam))
                res.append(compare("soft_closed/C vs soft_J kappa=1.3,m=1", C / C2 - 1, table))
            if song:
                import song_check
                with mp.workdps(dps_song):
                    Cs = 2 * song_check.Cplus(mp.mpf(m), mp.mpf(ka))[2] * mp.mpf(ka) ** 3
                res.append(compare(f"soft_closed/C vs 2 Song kappa={ka},m={m}", C / Cs - 1, table))
    return report(res)


if __name__ == "__main__":
    sys.exit(main(full="--all" in sys.argv, song="--song" in sys.argv))
