"""The fermion triangle with an axial chemical potential beyond the leading squeezed term (follow-up to Sec. 8.2).

Model of fermion_triangle.py (Aoki and Strumia, arXiv:2604.05548; a Dirac fermion of mass m with axial chemical
potential kappa, two Majorana copies of the model of Song et al., arXiv:2611.23220). The pair-frequency part of
the bispectrum is computed order by order in the soft momentum and at general shape,

    k1^6 <dtau^3>'  contains  sum_n C_n(c) r^(2 nu + n) + c.c.,     r = k3/k1,  c = k1hat . k3hat,  nu = i sqrt(m^2 + kappa^2),

with exact soft vertices and the hard fermion line kept in full (both helicities, all orderings of the three
vertices, both orientations of the loop). C_0 is the coefficient C of fermion_triangle.py and does not depend
on c. The method is in ext/fermitri.py:

    hard ends of the soft lines   branch series of the Whittaker function, Eq. (D.6), at the Dirac indices
    hard line                     folded moments in closed form (Gamma functions and 4F3(-1)); the energy
                                  |k1 + q| and the energy of the second hard leg by scaling
    spin traces                   Pauli algebra with polynomial entries, azimuthal average around k3
    soft vertex                   position-space moments of ext/softvertex.py, one per monomial y1^A y2^B

At fixed c the series in r converges only for k3 < k1. In the variables that are symmetric under k1 <-> k2
(ext/shapevars.py),

    K^6 <dtau^3>'  contains  sum_N D_N(delta) R^(2 nu + N) + c.c.,     2 K = k1 + k2,  R = 2 k3/(k1 + k2),  delta = (k1 - k2)/k3,

every D_N is an even polynomial in delta and the coefficients fall off, so the series reaches the equilateral
shape (R = 1, delta = 0). The shape function is S = -4 pi^2 P_zeta R^2 (1 - R^2 delta^2/4)^2 [sum_N D_N R^(2 nu + N) + c.c.].

Checks made here: C_0 against twice the closed form of Song et al. (value of record, 18 digits); the exchange
symmetry k1 <-> k2 at every order; and the coefficients on k1 = k2 against the values of record.

Usage:  python fermion_triangle_shape.py                        kappa = 1.3, m = 1 through n = 2 (about 3 min)
        python fermion_triangle_shape.py --point 5 1 --nmax 6   one point (kappa, m) [--dps 30]
        python fermion_triangle_shape.py --all                  kappa = 1.3 and 5 (n <= 10), 10 (n <= 8); hours
        python fermion_triangle_shape.py --point 1.3 1 --nmax 10 --save series.json     coefficients for direct/fermi_shape_fit.py
"""
import argparse
import json
import os
import sys

import mpmath as mp

from ext import shapevars
from ext.fermitri import FermiTriangle
from ext.record import compare, report

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def record():
    with open(os.path.join(HERE, "fermion_triangle_shape_record.json")) as f:
        return json.load(f)


def series(kappa, m, nmax, structures=None, verbose=False):
    """C_n(c), D_N(delta), the coefficients on k1 = k2, and the exchange check, for one (kappa, m)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"fsoft_{m}_{kappa}_dps{mp.mp.dps}.json")
    # decimal strings, so that kappa = 1.3 is 13/10 and not its double-precision neighbour (a 3e-16 effect on C_0)
    T = FermiTriangle(str(m), str(kappa), nmax=nmax, cache=cache, structures=structures)
    C = T.coefficients(T.accumulate(verbose=verbose))
    D = shapevars.symmetric(C, T.nu, nmax)
    return dict(T=T, C=C, D=D, iso=T.isosceles(C), exchange=shapevars.exchange_error(C, T.nu, nmax))


def show(kappa, m, res, table, results):
    T, nmax = res["T"], res["T"].nmax
    tag = f"kappa={kappa:g},m={m:g}"
    iso = res["iso"]
    print(f"kappa = {kappa:g}, m = {m:g}: nu = {mp.nstr(T.nu, 12)}, orders n <= {nmax}")
    print(f"   C_0 = {mp.nstr(iso[0], 18)}     |S|/P_zeta at leading order = {mp.nstr(4 * mp.pi ** 2 * abs(iso[0]), 6)}")
    print("   on k1 = k2 (delta = 0):  D_N/D_0 = " + ", ".join(mp.nstr(z / iso[0], 6) for z in iso[1:]))
    for dl in (0.5, 1):
        print(f"   |D_N/D_0| at delta = {dl:g}: " + ", ".join(mp.nstr(abs(z / iso[0]), 4) for z in shapevars.at_delta(res["D"], dl, nmax)))
    print("   exchange k1 <-> k2, largest relative difference by order: "
          + ", ".join(f"{n}: {mp.nstr(v, 2)}" for n, v in res["exchange"].items()))
    key = f"fermion_shape/C0 {tag}"
    if key in table:
        results.append(compare(key, iso[0], table))
    for n in range(1, nmax + 1):
        key = f"fermion_shape/D{n}/D0 delta=0 {tag}"
        if key in table:
            results.append(compare(key, iso[n] / iso[0], table))
        key = f"fermion_shape/|D{n}/D0| delta=1 {tag}"
        if key in table:
            results.append(compare(key, abs(shapevars.at_delta(res["D"], 1, nmax)[n] / iso[0]), table))
    key = f"fermion_shape/exchange {tag}"
    if key in table:
        results.append(compare(key, max(res["exchange"].values()), table))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs=2, type=float, metavar=("KAPPA", "M"))
    ap.add_argument("--nmax", type=int, default=2)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--save", metavar="FILE", help="write the coefficients C[(n, mc)] of the last point as json (input of direct/fermi_shape_fit.py)")
    args = ap.parse_args()
    if args.all:
        runs = [(1.3, 1, 10, 30), (5, 1, 10, 30), (10, 1, 8, 40)]
    elif args.point:
        runs = [(args.point[0], args.point[1], args.nmax, args.dps)]
    else:
        runs = [(1.3, 1, args.nmax, args.dps)]
    table, results = record(), []
    for kappa, m, nmax, dps in runs:
        kappa = int(kappa) if float(kappa).is_integer() else kappa
        m = int(m) if float(m).is_integer() else m
        with mp.workdps(dps):
            res = series(kappa, m, nmax)
            show(kappa, m, res, table, results)
            if args.save:
                num = lambda z: [mp.nstr(mp.re(z), 25), mp.nstr(mp.im(z), 25)]
                with open(args.save, "w") as f:
                    json.dump(dict(m=str(m), kappa=str(kappa), nmax=nmax, dps=dps, nu=num(res["T"].nu),
                                   coef={f"{n},{mc}": num(v) for (n, mc), v in sorted(res["C"].items())}), f, indent=1)
    return report(results) if results else 0


if __name__ == "__main__":
    sys.exit(main())
