"""The helical-vector triangle beyond the leading squeezed term (follow-up to Sec. 7.3).

Model of Wang and Xianyu (2004.02887): the gauge-boson triangle with three cubic vertices, soft leg k3 and
k1 = k2 = k. The pair-frequency family of the bispectrum is a series in r = k3/k,

    S = (k1 k2 k3)^2 <phi phi phi>'  contains  2 Re sum_{n >= 0} M_n r^(2 + 2 i mu + n),

with exact soft vertices and the hard gauge-boson line kept in full (all three hard helicities, all orderings
of the three vertices, all nine soft helicity pairs with their A_0 components). The powers advance in unit
steps, as for the bubble of Sec. 7.2 [Eq. (7.12)], because the branch series of the Whittaker function has
e_1 = -K/(1 + 2 nu); at kappa = 0 the odd orders vanish. The method is in ext/vectri.py:

    hard ends of the soft lines   branch series, Eq. (D.6), with the end factors of Eq. (D.2)
    hard line                     moments of Eq. (7.14) at shifted exponents; energy Y = |k1 + q| by scaling, Eq. (B.6)
    soft vertex                   position-space moments of ext/softvertex.py, one per monomial y1^A y2^B

Coefficients are quoted at unit couplings, M_n (triangle: prefactor 1/8; contracted bubble: 1/16, so that
M_0(bubble) = M_A of Sec. 7.2), as ratios M_n/M_0, and in the normalization of Wang and Xianyu,
-m_h^2 e^(-2 pi Omega) M_n with Omega = max(kappa - mu, 0), in which R_hard of Sec. 7.3 is the monopole of
the n = 0 term over M_A. On the isosceles family khat1.khat3 = -r/2, so the n = 0 coefficient is the
monopole minus half the quadrupole.

Checks made here (default run): the nine soft pairs share one monopole ratio; no dipole; R_hard and the ratio
of M_A to its late-time estimate against Secs. 7.2 and 7.3; and the series through n = 2 at mu = kappa = 1.
With --all: the benchmark points through n = 8, kappa = 0 (odd orders vanish), and a heavy hard line (each
order tends to twice the contracted bubble).

General shape. With the energy of the second hard leg expanded as well (Triangle(general=True)), the coefficients
M_n(c) hold for k1 != k2. In R = 2 k3/(k1 + k2) and delta = (k1 - k2)/k3 the series is 2 Re sum_N D_N(delta) R^(2 + 2 nu + N)
with D_N even in delta (ext/shapevars.py), and the symmetry k1 <-> k2 is checked order by order. The leading
coefficient is D_0(delta) = monopole + quadrupole (3 delta^2 - 1)/2, since khat1.khat3 -> -delta in the squeezed
limit. template() sums the series over the three choices of the soft leg. The series is checked against a direct
time-domain integration of the triangle at fixed shape in direct/ (direct_vec.py, vec_direct_fit.py).

Leading order (--leading): the ratio of M_A to its late-time estimate at kappa = mu (6.07, 16.70, 29.99 for
mu = 1, 2, 3; Sec. 7.2), |R_hard| at the same points (0.8, 2.1, 3.9) and at the points of Fig. 10 with mu = 2, 3
(Sec. 7.3), and the heavy hard line. The exact values are compared with the values of record, and at the points of
Fig. 10 also with the values plotted there (which came from an earlier direct calculation, good to 2e-4).

Section, equation, and figure numbers are those of v5.12 of the paper.

Usage:  python helical_triangle.py                      (about 10 min)
        python helical_triangle.py --point 2 2.5 --nmax 6 [--hard mu_h kappa_h] [--dps 30] [--general] [--json out.json]
        python helical_triangle.py --leading [i/n]      (about 3 min per point; i/n runs every n-th point from the i-th)
        python helical_triangle.py --all                (several hours)
        python helical_triangle.py --deep [N]           (the series at (3, 3.75) through n = N, default 12; about 2 hours)
"""
import argparse
import json
import os
import sys

import mpmath as mp

from ext import shapevars
from ext.record import compare, report
from ext.vectri import Triangle

HERE = os.path.dirname(os.path.abspath(__file__))
# leading-order points: kappa = mu (Secs. 7.2 and 7.3), and the points of Fig. 10 with mu = 2, 3 and 0.5 <= kappa - mu <= 1.25
LEADING = [("1", "1"), ("2", "2"), ("3", "3"), ("2", "2.5"), ("2", "2.75"), ("2", "3"), ("2", "3.25"), ("3", "3.75"), ("3", "4"),
           ("3", "4.25")]
CACHE = os.path.join(HERE, "cache")
PAIRS = [(a, b) for a in (1, -1, 0) for b in (1, -1, 0)]


def record():
    with open(os.path.join(HERE, "helical_triangle_record.json")) as f:
        return json.load(f)


def wx_factor(T):
    """Triangle in the normalization of Wang and Xianyu: -m_h^2 e^(-2 pi Omega) M_n."""
    return -T.hardline.m2 * mp.exp(-2 * mp.pi * max(T.kappa - T.mu, 0))


def triangle(mu, kappa, nmax, mu_h=None, kappa_h=None):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"soft_{mu}_{kappa}_dps{mp.mp.dps}.json")
    return Triangle(mu, kappa, mu_h, kappa_h, nmax=nmax, cache=cache)


def leading(mu, kappa, mu_h=None, kappa_h=None, verbose=True):
    """Leading order, pair by pair: M_A, its late-time estimate, and the multipoles of the triangle."""
    T = triangle(mu, kappa, 0, mu_h, kappa_h)
    fac = wx_factor(T)
    tot = dict(bub=0, late=0, mono=0, quad=0, dip=0)
    ratio = {}
    for pr in PAIRS:
        T.soft_pairs = [pr]
        ab, at = T.accumulate("bubble"), T.accumulate("triangle")
        Mb = T.coefficients("bubble", acc=ab)[0]
        res = T.coefficients("triangle", isosceles=False, acc=at)
        m0, m1, m2 = res.get((0, 0), 0), res.get((0, 1), 0), res.get((0, 2), 0)      # M(c) = m0 + m1 c + m2 c^2
        tot["bub"] += Mb
        tot["late"] += T.coefficients("bubble", acc=ab, late=True)[0]
        tot["mono"] += m0 + m2 / 3
        tot["quad"] += 2 * m2 / 3
        tot["dip"] += m1
        ratio[pr] = fac * (m0 + m2 / 3) / Mb
    R = fac * tot["mono"] / tot["bub"]
    out = dict(M_A=tot["bub"], exact_over_late=abs(tot["bub"] / tot["late"]), R_hard=R,
               R_hard0=abs(R) * mp.exp(2 * mp.pi * (T.kappa - T.mu)),
               spread=max(abs(ratio[p] / R - 1) for p in PAIRS), dipole=abs(tot["dip"] / tot["mono"]),
               quadrupole=abs(tot["quad"] / tot["mono"]), M0_isosceles=tot["mono"] - tot["quad"] / 2)
    if verbose:
        print(f"  mu = {mu}, kappa = {kappa}" + (f", hard line ({mu_h}, {kappa_h})" if mu_h is not None else "")
              + f":  M_A = {mp.nstr(out['M_A'], 12)},  exact/late = {mp.nstr(out['exact_over_late'], 6)},  "
              f"|R_hard| = {mp.nstr(abs(R), 10)},  pairs agree to {mp.nstr(out['spread'], 2)},  "
              f"dipole {mp.nstr(out['dipole'], 2)},  quadrupole/monopole {mp.nstr(out['quadrupole'], 4)}")
    return out


def series(mu, kappa, nmax, mu_h=None, kappa_h=None, verbose=True):
    """[M_n(triangle)], [M_n(bubble)] for n = 0..nmax, and the Wang-Xianyu factor."""
    T = triangle(mu, kappa, nmax, mu_h, kappa_h)
    Mb = T.coefficients("bubble")
    Mt = T.coefficients("triangle")
    if verbose:
        print(f"  mu = {mu}, kappa = {kappa}" + (f", hard line ({mu_h}, {kappa_h})" if mu_h is not None else "")
              + f":  -m_h^2 e^(-2 pi Omega) = {mp.nstr(wx_factor(T), 8)}")
        print("     n    M_n (triangle, unit couplings)              M_n/M_0                          M_n (bubble)")
        for n in range(nmax + 1):
            print(f"    {n:2d}   {mp.nstr(Mt[n], 12):>40s}   {mp.nstr(Mt[n] / Mt[0], 8):>30s}   {mp.nstr(Mb[n], 12):>40s}")
    return Mt, Mb, wx_factor(T)


def general(mu, kappa, nmax, verbose=True, json_out=None):
    """General shape: {(n, mc): M_n(c)} with the energy of the second hard leg expanded, the same series in the
    symmetric variables R = 2 k3/(k1 + k2), delta = (k1 - k2)/k3 (S contains 2 Re sum_N D_N(delta) R^(2 + 2 nu + N)),
    and the largest violation of the symmetry k1 <-> k2 at each order. json_out: file for direct/vec_direct_fit.py."""
    os.makedirs(CACHE, exist_ok=True)
    T = Triangle(mu, kappa, nmax=nmax, cache=os.path.join(CACHE, f"soft_{mu}_{kappa}_dps{mp.mp.dps}.json"), general=True)
    C = T.coefficients("triangle", isosceles=False)
    D = shapevars.symmetric(C, T.nu, nmax, dim=2)
    ex = shapevars.exchange_error(C, T.nu, nmax, dim=2)
    D0 = shapevars.at_delta(D, 0, nmax)
    if verbose:
        print(f"  mu = {mu}, kappa = {kappa}, general shape, orders n <= {nmax}")
        for dl in (0, 0.5, 1):
            print(f"     |D_N(delta)/D_0(0)| at delta = {dl:g}: "
                  + ", ".join(mp.nstr(abs(z / D0[0]), 4) for z in shapevars.at_delta(D, dl, nmax)))
        print("     exchange k1 <-> k2, largest relative difference by order: "
              + ", ".join(f"{n}: {mp.nstr(v, 2)}" for n, v in ex.items()))
        m0, m2 = C.get((0, 0), 0), C.get((0, 2), 0)
        print(f"     leading term: D_0(delta) = mono + quad (3 delta^2 - 1)/2 with quad/mono = {mp.nstr(2 * m2 / 3 / (m0 + m2 / 3), 6)}")
        for name, R, dl in (("equilateral", 1, 0), ("folded, k1 = 2 k2 = 2 k3", mp.mpf(2) / 3, 1)):
            DN = shapevars.at_delta(D, dl, nmax)
            full = mp.fsum(DN[n] * R ** n for n in range(nmax + 1))
            part = mp.fsum(DN[n] * R ** n for n in range(nmax - 1))
            print(f"     series over D_0(0) at the {name} shape: modulus {mp.nstr(abs(full / D0[0]), 4)}"
                  f" (the last two orders change it by {mp.nstr(abs(abs(part / full) - 1), 2)})")
    if json_out:
        c = lambda z: [mp.nstr(mp.re(z), 25), mp.nstr(mp.im(z), 25)]
        with open(json_out, "w") as f:
            json.dump(dict(mu=str(mu), kappa=str(kappa), nmax=nmax, dps=mp.mp.dps,
                           coef={f"{n},{mc}": c(v) for (n, mc), v in sorted(C.items())},
                           isosceles=[c(z) for z in D0], sym={f"{n},{md}": c(v) for (n, md), v in sorted(D.items())},
                           exchange={str(n): mp.nstr(v, 5) for n, v in ex.items()}), f, indent=1)
    return C, D, ex


def template(k1, k2, k3, D, mu, nmax, Rmax=1.0):
    """Pair-frequency signal symmetric in the three legs: the sum over the choice i of the soft leg of
    2 Re sum_N D_N(delta_i) R_i^(2 + 2 i mu + N), with R_i = 2 k_i/(k_j + k_l) and delta_i = (k_j - k_l)/k_i.
    D: the coefficients {(N, power of delta): value} returned by general(). Only the legs with R_i <= Rmax are
    summed (the series converges for R_i < 2, slowly near R_i = 1; in a squeezed triangle only the soft leg counts)."""
    tot = mp.mpf(0)
    for ks, ka, kb in ((k3, k1, k2), (k1, k2, k3), (k2, k3, k1)):
        R, dl = 2 * mp.mpf(ks) / (ka + kb), (mp.mpf(ka) - kb) / ks
        if R > Rmax:
            continue
        DN = shapevars.at_delta(D, dl, nmax)
        tot += 2 * mp.re(mp.fsum(DN[n] * R ** (2 + n + 2j * mp.mpf(mu)) for n in range(nmax + 1)))
    return tot


def leading_checks(points, tab):
    """Leading-order values against the values of record (and against the points plotted in Fig. 10)."""
    res = []
    for mu, ka in points:
        L = leading(mu, ka)
        key = f"mu={mu} kappa={ka}"
        res.append(compare(f"leading/exact over late {key}", L["exact_over_late"], tab))
        res.append(compare(f"leading/|R_hard| {key}", abs(L["R_hard"]), tab))
        res.append(compare(f"leading/pair spread {key}", L["spread"], tab))
        res.append(compare(f"leading/dipole {key}", L["dipole"], tab))
        if f"figure/|R_hard| {key}" in tab:
            res.append(compare(f"figure/|R_hard| {key}", abs(L["R_hard"]), tab))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs=2)
    ap.add_argument("--general", action="store_true", help="with --point: coefficients at general shape")
    ap.add_argument("--hard", nargs=2)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--json", help="with --point and --general: write the coefficients for direct/vec_direct_fit.py")
    ap.add_argument("--leading", nargs="?", const="0/1", help="leading-order checks only; i/n runs every n-th point from the i-th")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--deep", nargs="?", const=12, type=int, help="the series at (3, 3.75) through this order (default 12)")
    args = ap.parse_args()
    mp.mp.dps = args.dps
    if args.point:
        mu_h, kappa_h = args.hard if args.hard else (None, None)
        leading(*args.point, mu_h, kappa_h)
        series(*args.point, args.nmax, mu_h, kappa_h)
        if args.general and not args.hard:
            general(*args.point, args.nmax, json_out=args.json)
        return 0
    tab, res = record(), []
    print("Leading order (Secs. 7.2 and 7.3)")
    if args.leading:
        i, n = (int(x) for x in args.leading.split("/"))
        res += leading_checks(LEADING[i::n], tab)
        if i == 0:
            L = leading("2", "3", "10", "0")
            res.append(compare("leading/|R_hard,0| heavy mu_h=10", L["R_hard0"], tab))
        return report(res)
    if args.deep is not None:
        print("Series in r = k3/k at mu = 3, kappa = 3.75")
        Mt, Mb, fac = series("3", "3.75", args.deep)
        for n in range(args.deep + 1):
            res.append(compare(f"series/M_{n} mu=3 kappa=3.75", Mt[n], tab))
        if args.deep >= 12:                                   # the equilateral point, r = 1
            eq = [abs(mp.fsum(Mt[:n + 1]) / Mt[0]) for n in (10, 12)]
            res.append(compare("series/equilateral value over leading term mu=3 kappa=3.75", eq[1], tab))
            res.append(compare("series/equilateral change of the last two orders mu=3 kappa=3.75", eq[1] / eq[0] - 1, tab))
        return report(res)
    res += leading_checks(LEADING if args.all else [("1", "1")], tab)
    if args.all:
        L = leading("2", "3", "10", "0")
        res.append(compare("leading/|R_hard,0| heavy mu_h=10", L["R_hard0"], tab))
    print("Series in r = k3/k")
    runs = [("1", "1", 8), ("2", "2.5", 8), ("3", "3.75", 8)] if args.all else [("1", "1", 2)]
    for mu, ka, nmax in runs:
        Mt, Mb, fac = series(mu, ka, nmax)
        key = f"mu={mu} kappa={ka}"
        for n in range(nmax + 1):
            res.append(compare(f"series/M_{n} {key}", Mt[n], tab))
        for n in (1, 2):
            res.append(compare(f"series/bubble M_{n} over M_A {key}", Mb[n] / Mb[0], tab))
    print("General shape (k1 != k2)")
    Mt, Mb, fac = series("1", "1", 2, verbose=False)
    C, D, ex = general("1", "1", 2)
    D0 = shapevars.at_delta(D, 0, 2)
    res.append(compare("general/exchange symmetry mu=1 kappa=1", max(ex.values()), tab))
    res.append(compare("general/isosceles limit mu=1 kappa=1", max(abs(D0[n] / Mt[n] - 1) for n in range(3)), tab))
    res.append(compare("general/|D_0(delta=1)/D_0(0)| mu=1 kappa=1", abs(shapevars.at_delta(D, 1, 2)[0] / D0[0]), tab))
    if args.all:
        Mt, Mb, fac = series("1", "0", 4)
        res.append(compare("series/odd orders at kappa=0", max(abs(Mt[1] / Mt[0]), abs(Mt[3] / Mt[0])), tab))
        Mt, Mb, fac = series("1", "1", 4, "20", "0")
        dev = max(abs(-(mp.mpf(20) ** 2 + mp.mpf(1) / 4) * Mt[n] / Mb[n] - 2) for n in range(5))
        res.append(compare("series/heavy hard line mu_h=20: max |R_0,n - 2|", dev, tab))
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
