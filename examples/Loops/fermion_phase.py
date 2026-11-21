"""The total-energy phase of the fermion loops at kappa >> m: derived forms against the exact series (Secs. 6.4, 8.4).

For kappa >> m the pair signal of the Dirac loops oscillates in k3/k_T, k_T = k1 + k2 + k3. This script compares
the forms derived for that regime (ext/fermicontact.py) with the exact series of dirac_pair.py and ext/fermitri.py.

  1. Closed forms. The model loop of two lines with first index 1/2 + nu against the series built from its soft
     moments, for several vertex powers, and its elementary form at (r1, r2) = (4, 2). The tree-level counterpart:
     the saddle-point phase of Qin, Renaux-Petel, Tong, Werth, and Zhu (2506.01555, Eq. (3.49)) for kappa -> mubar
     against kappa ln(1 + k12/s) + (mubar - kappa) ln(k12/s).
  2. The segment. The soft-moment average of (y1 + y2)/k3 (ext/softvertex.soft_J) against the end-point form.
  3. The Dirac bubble (dirac_pair.py). Coefficients L_n of the logarithm of the pair series, with (1 + R/2)^(-2 nu)
     removed, against (1 - R^2/4)^-2 [1 + 4 rho/c + 6 rho^2/(c (c + 1))] (1 + R/2)^(2 eps) <(1 - R^2 eta^2/4)^-eps>.
  4. The triangle. The exact identity  triangle = B' + T3'  (contact term plus the triangle of two mass vertices)
     on k1 = k2 and at general shape, the size of T3', and the series of B' against
     (1 + R/2)^(-2 i kappa) (1 - R/2)^-2 (1 - R^2 delta^2/4)^-2  and its first corrections.

Usage:  python fermion_phase.py                        kappa = 5, m = 1 through N = 2 (about 2 min at 30 digits)
        python fermion_phase.py --quick                part 1 only (seconds)
        python fermion_phase.py --point 10 1 --dps 40  another point (kappa, m); nothing is compared with the record
        python fermion_phase.py --nmax 1               orders of the triangle (default 2) [--nbubble 2]
        python fermion_phase.py --dump values.json     also write the computed values to a file
"""
import argparse
import json
import os
import sys
import time

import mpmath as mp

from ext import shapevars
from ext.fermicontact import (Contact, bubble_closed, contact_coefficients, contact_first_order, contact_symmetric,
                              eps_factor, log_series, pair_closed, segment_average, strip_total_energy,
                              tree_saddle_phase)
from ext.record import compare, report
from ext.softvertex import Grid, Leg, soft_J

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
I = mp.mpc(0, 1)


def record():
    with open(os.path.join(HERE, "fermion_phase_record.json")) as f:
        return json.load(f)


def fmt(z, d=4):
    return f"{float(mp.re(z)):+.{d}f}{float(mp.im(z)):+.{d}f}i"


# ------------------------------------------------------------------------------ 1. closed forms
def moment_series(nu, r1, r2, N):
    """Taylor coefficients in R of <(1 + R xi/2)^-(r1 + 2 nu)> over the weight (xi^2 - 1)(1 + xi)^-(r2 - 2 nu) on
    xi >= 1, with the moments continued analytically: int_0^oo ds s (2 + s)^-G = 2^(2 - G)/((G - 1)(G - 2))."""
    g, beta = r2 - 2 * nu, r1 + 2 * nu

    def mom(n):
        return mp.fsum(mp.binomial(n, k) * (-1) ** (n - k) * mp.mpf(2) ** (2 - (g - 1 - k)) / ((g - 2 - k) * (g - 3 - k))
                       for k in range(n + 1))
    m0 = mom(0)
    return [mp.binomial(-beta, n) * mp.mpf(2) ** -n * mom(n) / m0 for n in range(N + 1)]


def closed_form_checks():
    worst_series = worst_elem = mp.mpf(0)
    with mp.workdps(40):
        for kap in (5, 10):
            nu = mp.mpc(0, kap)
            for r1, r2 in ((4, 2), (2, 1), (1, 1), (0, 1), (2, 0), (3, 2)):
                s = moment_series(nu, r1, r2, 6)
                t = mp.taylor(lambda R: pair_closed(R, nu, r1, r2), 0, 6)
                worst_series = max(worst_series, max(abs(a - b) / max(1, abs(b)) for a, b in zip(s, t)))
            for R in ("0.1", "0.5", "1"):
                worst_elem = max(worst_elem, abs(bubble_closed(R, nu) / pair_closed(R, nu, 4, 2) - 1))
    return worst_series, worst_elem


def tree_saddle_check():
    """Largest |d Phi - d[kappa ln(1 + X) + (mubar - kappa) ln X]| kappa^3/m^4 between X = k12/s = 1.2, 2, 5, 20."""
    worst = mp.mpf(0)
    with mp.workdps(30):
        for kap, m in ((5, 1), (10, 1), (20, 1), (20, 2), (100, 1)):
            mu = mp.sqrt(mp.mpf(kap) ** 2 + m ** 2)
            f = lambda X: tree_saddle_phase(X, mu, kap) - kap * mp.log(1 + mp.mpf(X)) - (mu - kap) * mp.log(mp.mpf(X))
            ref = f(20)
            worst = max(worst, max(abs(f(X) - ref) for X in ("1.2", 2, 5)) * mp.mpf(kap) ** 3 / mp.mpf(m) ** 4)
    return worst


# ------------------------------------------------------------------------------ 2. the segment
def segment_exact(nu, K, q, grid=None):
    """[J(1, 0) + J(0, 1)]/J(0, 0) for two lines with first index K: the average of (y1 + y2)/k3."""
    grid = grid or Grid()
    leg = lambda a: Leg(nu - mp.mpf(1) / 2 + a, K, nu)
    l0, l1 = leg(0), leg(1)
    return 2 * soft_J(l1, l0, q, grid) / soft_J(l0, l0, q, grid)


# ------------------------------------------------------------------------------ 3. the bubble
def bubble_L(m, kappa, nmax):
    from dirac_pair import DiracPair
    P = DiracPair(mp.mpf(m), mp.mpf(kappa))
    c = [mp.gamma(4 + 2 * P.nu + n) * P.S(n) / mp.mpf(2) ** n for n in range(nmax + 1)]
    return P.nu, log_series(strip_total_energy(c, P.nu))


def bubble_derived(nu, kappa, nmax):
    f = lambda R: mp.log(bubble_closed(R, nu) * (1 + R / 2) ** (2 * nu) * eps_factor(R, nu, kappa))
    g = lambda R: mp.log(bubble_closed(R, nu) * (1 + R / 2) ** (2 * nu))
    return [mp.mpc(x) for x in mp.taylor(g, 0, nmax)], [mp.mpc(x) for x in mp.taylor(f, 0, nmax)]


# ------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs=2, metavar=("KAPPA", "M"))
    ap.add_argument("--nmax", type=int, default=2)
    ap.add_argument("--nbubble", type=int, default=2)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dump", metavar="FILE")
    args = ap.parse_args(argv)
    kappa, m = args.point if args.point else ("5", "1")
    check = not args.point and args.nmax == 2 and args.nbubble == 2 and args.dps == 30
    table, results, vals, t0 = record(), [], {}, time.time()
    tag = f"kappa={kappa},m={m}"

    def note(key, value, quiet=False):
        vals[f"fermion_phase/{key} {tag}"] = (mp.mpmathify(value), quiet)

    def finish():
        if args.dump:
            with open(args.dump, "w") as f:
                json.dump({k: [mp.nstr(mp.re(v), 20), mp.nstr(mp.im(v), 20)] for k, (v, _) in vals.items()}, f, indent=1)
        if check:
            for k, (v, quiet) in vals.items():
                results.append(compare(k, v, table, quiet="max" in table[k] and quiet))
        return report(results)

    print("1. Closed forms of the model loop")
    ws, we = closed_form_checks()
    print(f"   series from the soft moments = Taylor series of the 2F1 form, six pairs (r1, r2), n <= 6: {mp.nstr(ws, 3)}")
    print(f"   elementary form at (r1, r2) = (4, 2) against the 2F1 form: {mp.nstr(we, 3)}")
    wt = tree_saddle_check()
    print(f"   tree-level saddle phase of 2506.01555 against kappa ln(1 + X) + (mubar - kappa) ln X, times kappa^3/m^4: {mp.nstr(wt, 3)}")
    results.append(compare("fermion_phase/closed form, series", ws, table, quiet=True))
    results.append(compare("fermion_phase/closed form, elementary", we, table, quiet=True))
    results.append(compare("fermion_phase/tree-level saddle phase", wt, table, quiet=True))
    if args.quick:
        return report(results)

    mp.mp.dps = args.dps
    ka, mm = mp.mpf(kappa), mp.mpf(m)
    nu = I * mp.sqrt(mm ** 2 + ka ** 2)
    eps = nu - I * ka

    print(f"2. The segment at {tag}: average of (y1 + y2)/k3 over the soft moment, first indices 1/2 - i kappa")
    for q in (1, 2):
        ex, dv = segment_exact(nu, mp.mpf(1) / 2 - I * ka, q), segment_average(q, ka)
        print(f"   q = {q}: exact {mp.nstr(ex, 10)}   end point {mp.nstr(dv, 10)}   difference {mp.nstr(abs(ex - dv), 3)}", flush=True)
        note(f"segment average q={q}", ex)
        note(f"segment, exact - end point q={q}", ex - dv)

    print(f"3. The Dirac bubble at {tag}: L_n with (1 + R/2)^(-2 nu) removed", flush=True)
    _, L = bubble_L(m, kappa, args.nbubble)
    d0, d1 = bubble_derived(nu, ka, args.nbubble)
    for n in range(1, args.nbubble + 1):
        print(f"   n = {n}: exact {fmt(L[n])}   closed form at eps = 0 {fmt(d0[n])}   with first order in eps {fmt(d1[n])}"
              f"   remainder {mp.nstr(abs(L[n] - d1[n]), 3)}", flush=True)
        note(f"bubble L{n}", L[n])
        note(f"bubble L{n} - derived", L[n] - d1[n])

    print(f"4. The triangle at {tag}, orders N <= {args.nmax}", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    T = Contact(str(m), str(kappa), nmax=args.nmax, cache=os.path.join(CACHE, f"fsoft_{m}_{kappa}_dps{mp.mp.dps}.json"))
    C = T.coefficients(T.accumulate())
    tri = T.isosceles(C)
    Sc = T.Sc()
    con = contact_coefficients(Sc, nu, mm, ka)
    C3 = []
    for dl in (T.delta_t, -T.delta_t):                       # the two regulated values of T3'
        T._q0, T._ser, T._ord = {}, {}, {}
        C3.append(T.coefficients(T.t3_accumulate(dl)))
    T._q0, T._ser, T._ord = {}, {}, {}
    C3m = {k: (C3[0].get(k, 0) + C3[1].get(k, 0)) / 2 for k in set(C3[0]) | set(C3[1])}
    t3 = T.isosceles(C3m)
    spread = max(abs(a - b) / abs(a + b) * 2 for a, b in zip(T.isosceles(C3[0]), T.isosceles(C3[1])))
    ident = max(abs(a - b - c) / abs(a) for a, b, c in zip(tri, con, t3))
    print(f"   D_0: triangle {mp.nstr(tri[0], 12)} = contact {mp.nstr(con[0], 12)} + T3' {mp.nstr(t3[0], 12)}")
    print(f"   identity on k1 = k2, N <= {args.nmax}: {mp.nstr(ident, 3)}   (pole of single assignments: the two regulated values differ by {mp.nstr(spread, 3)})")
    Dt, D3, Dc = shapevars.symmetric(C, nu, args.nmax), shapevars.symmetric(C3m, nu, args.nmax), contact_symmetric(Sc, nu, mm, ka)
    scale = max(abs(v) for v in Dt.values())
    gen = max(abs(Dt.get(k, 0) - Dc.get(k, 0) - D3.get(k, 0)) for k in set(Dt) | set(Dc) | set(D3)) / scale
    print(f"   identity at general shape (all coefficients of R^N delta^md): {mp.nstr(gen, 3)}")
    tau0 = t3[0] / con[0]
    print(f"   T3'/B' at R -> 0: {fmt(tau0)}   (times kappa/m^2: {fmt(tau0 * ka / mm ** 2, 3)});  |1 + T3'/B'| = {mp.nstr(abs(1 + tau0), 5)},"
          f" exp(-3 pi (mubar - kappa)/2) = {mp.nstr(mp.exp(-3 * mp.pi * abs(eps) / 2), 5)}")
    LT, LB = log_series(strip_total_energy(tri, nu)), log_series(strip_total_energy(con, nu))
    LD = [mp.mpc(x) for x in mp.taylor(lambda R: mp.log(contact_first_order(R, nu, ka) * (1 + R / 2) ** (2 * nu)), 0, args.nmax)]
    for n in range(1, args.nmax + 1):
        print(f"   L_{n}: triangle {fmt(LT[n])}   contact term {fmt(LB[n])}   derived for the contact term {fmt(LD[n])}")
    note("C0", tri[0])
    note("contact D0", con[0])
    note("T3'/B' at R -> 0", tau0)
    note("identity", ident)
    note("identity, general shape", gen)
    note("contact L1", LB[1])
    note("contact L1 - (1 + eps)", LB[1] - 1 - eps)
    note("triangle L1", LT[1])
    if args.nmax >= 2:
        note("contact L2", LB[2])
        note("contact L2 - derived", LB[2] - LD[2])
        note("triangle L2", LT[2])
    print(f"   [{time.time() - t0:.0f} s]")
    return finish()


if __name__ == "__main__":
    sys.exit(main())
