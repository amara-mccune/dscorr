"""The pair family of the scalar triangle with inflaton legs, at general shape (follow-up to Sec. 5.2).

Eqs. (5.4)-(5.6) of the paper give the pair family of the triangle wavefunction coefficient on k1 = k2, to all
orders in (k3/k1)^2. Here it is given at every shape. With

    K = (k1 + k2)/2,   R = k3/K = 2 k3/(k1 + k2),   delta = (k1 - k2)/k3,

    psi_3  contains  K^(1 - r1 - r2) R^(1 + nuB + nuC) sum_N D_N(delta) R^N          (r3 = 2),

where D_N is a polynomial of degree at most N and parity (-1)^N in delta, a finite sum of Gamma functions times
the hard moments of Eq. (B.4) (ext/scalartri.py). On k1 = k2, D_2N(0) = c_N of Eq. (5.6). The series is the
expansion of one parameter integral over three-site chains (the soft loop momentum is a point of the segment
from 0 to k3), which converges for k3 < k1 + k2, so at fixed delta the radius in R is 2.

The script

  1. reproduces c_0, c_1, c_2 of Eq. (5.6) at mu = (1, 1.3, 0.7) (values of Oct. 27, from the Mellin-space sums);
  2. checks the one-step recurrence for the hard moments against Eq. (B.4);
  3. checks the exchange of the hard legs, delta -> -delta with (nuB, r2) <-> (nuC, r1), in (R, delta), and
     order by order in r = k3/k1 at fixed angle, where the expansion does not respect it term by term;
  4. assembles the in-in correlator with the weights of Eq. (2.10) and reproduces M_tri and D_tri of Ref. [26];
  5. gives the size of the series at the shapes quoted in the write-up;
  6. compares with the triangle integral of Ref. [26] at delta = 0.5 and 1 (values in
     scalar_triangle_shape_direct.json; examples/Letter/batch/triangle_pidot with the block from Eq. (A.5)).

Values of record: scalar_triangle_shape_record.json. The script prints one line per value and exits with a
nonzero status if a value is not reproduced.

Usage:  python scalar_triangle_shape.py                     (about 1 min)
        python scalar_triangle_shape.py --mu 1 1.3 0.7 --nmax 30 [--dps 40] [--delta 0.5]
"""
import argparse
import json
import os
import sys

import mpmath as mp

from ext.record import compare, report
from ext.scalartri import ScalarTriangle, correlator_family, exchanged_angle, to_symmetric

HERE = os.path.dirname(os.path.abspath(__file__))
I = mp.mpc(0, 1)


def record():
    with open(os.path.join(HERE, "scalar_triangle_shape_record.json")) as f:
        return json.load(f)


def direct_data():
    with open(os.path.join(HERE, "scalar_triangle_shape_direct.json")) as f:
        return json.load(f)


def at_delta(coef, delta, Nmax):
    out = [mp.mpc(0)] * (Nmax + 1)
    for (n, m), v in coef.items():
        out[n] += v * mp.mpmathify(delta) ** m
    return out


def ratio(coef, R, delta, Nmax, drop=0):
    """sum_N D_N(delta) R^N / D_0(delta), without the last `drop` orders"""
    D = at_delta(coef, delta, Nmax)
    return mp.fsum(D[n] * mp.mpmathify(R) ** n for n in range(Nmax + 1 - drop)) / D[0]


def symmetry_checks(nu, N):
    """(exchange in (R, delta), exchange identity in (r, c), re-expansion of the (r, c) series)"""
    T, T2 = ScalarTriangle(*nu, Nmax=N), ScalarTriangle(nu[0], nu[2], nu[1], Nmax=N)
    co, co2 = T.coefficients(), T2.coefficients()
    scale = {n: max(abs(v) for (nn, m), v in co.items() if nn == n) for n in range(N + 1)}
    sym = max(abs(v - (-1) ** k[1] * co2.get(k, 0)) / scale[k[0]] for k, v in co.items())
    ca, ca2 = T.coefficients_angle(), T2.coefficients_angle()
    expo = T.r1 + T.r2 + T.nuBC                        # dim + e = (r1 + r2 - 1) + (1 + nuBC)
    ex = exchanged_angle(ca, expo, N)
    sca = {n: max(abs(v) for (nn, m), v in ca2.items() if nn == n) for n in range(N + 1)}
    ident = max(abs(ex.get(k, 0) - ca2.get(k, 0)) / sca[k[0]] for k in set(ex) | set(ca2))
    back = to_symmetric(ca, expo, N)
    reexp = max(abs(back.get(k, 0) - co.get(k, 0)) / scale[k[0]] for k in set(back) | set(co))
    return sym, ident, reexp


def direct_fit(points, series, expo, degrees, free=()):
    """Fit psi(R_j) = polynomial(R_j) + c R_j^expo sum_N s_N R_j^N on the points [(R, psi)].

    series: [s_0, s_1, ...] (the closed coefficients at this delta). The polynomial (the analytic part) has
    complex coefficients of degree `degree`. free: orders N whose coefficient is fitted instead of taken from
    the series (c = 1 for the others). Returns {degree: (rms residual, {N: fitted/closed})}.
    Least squares in mpmath, with the powers of R scaled to the largest point.
    """
    Rs = [mp.mpf(p[0]) for p in points]
    psi = [mp.mpc(p[1][0], p[1][1]) for p in points]
    Rmax = max(Rs)
    out = {}
    for deg in degrees:
        cols = [[(R / Rmax) ** n for R in Rs] for n in range(deg + 1)]
        rhs = list(psi)
        for N, sN in enumerate(series):
            term = [sN * R ** (expo + N) for R in Rs]
            if N in free:
                cols.append(term)
            else:
                rhs = [a - b for a, b in zip(rhs, term)]
        A = mp.matrix(len(Rs), len(cols))
        for j, col in enumerate(cols):
            for i, v in enumerate(col):
                A[i, j] = v
        x, _ = mp.qr_solve(A, mp.matrix(rhs))
        res = A * x - mp.matrix(rhs)
        rms = mp.sqrt(mp.fsum(abs(r) ** 2 for r in res) / len(Rs))
        out[deg] = (rms, {N: x[deg + 1 + i] for i, N in enumerate(sorted(free))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", nargs=3, type=str, default=["1", "1.3", "0.7"], metavar=("MUA", "MUB", "MUC"))
    ap.add_argument("--nmax", type=int, default=24)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--delta", type=float, nargs="*", default=[0, 0.5, 1])
    args = ap.parse_args()
    mp.mp.dps = args.dps
    mus = tuple(mp.mpf(m) for m in args.mu)
    nu = tuple(I * m for m in mus)
    N = args.nmax
    default = args.mu == ["1", "1.3", "0.7"]
    table, res = record(), []

    print(f"Scalar triangle, mu = ({', '.join(args.mu)}), r_v = 2: the pair family at general shape, orders N <= {N}")
    T = ScalarTriangle(*nu, Nmax=N)
    co = T.coefficients()

    print("1. k1 = k2: Eq. (5.6)")
    iso = T.isosceles(N // 2)
    for n, c in enumerate(iso[:3]):
        if default:
            res.append(compare(f"scalar_shape/c{n}", c, table))
        else:
            print(f"   c_{n} = {mp.nstr(c, 18)}")
    worst = max(abs(co[(2 * n, 0)] / c - 1) for n, c in enumerate(iso))
    print(f"   D_2N(0) against c_N from the single sum of Eq. (5.6), N <= {N // 2}: {mp.nstr(worst, 3)}")

    print("2. Hard moments: one seed and the recurrence, against Eq. (B.4) for every moment (N <= 4)")
    cn = ScalarTriangle(*nu, Nmax=4, recurrence=False).coefficients()
    rec = max(abs(co[k] / v - 1) for k, v in cn.items())
    res.append(compare("scalar_shape/recurrence", rec, table))

    print("3. Exchange of the hard legs (N <= 12)")
    sym, ident, reexp = symmetry_checks(nu, min(N, 12))
    res.append(compare("scalar_shape/exchange in (R, delta)", sym, table))
    res.append(compare("scalar_shape/exchange identity in (r, c)", ident, table))
    res.append(compare("scalar_shape/(r, c) series re-expanded", reexp, table))

    print("4. In-in assembly, Eq. (2.10)")
    A = {s: correlator_family(mus, 1, s, Nmax=N) for s in (1, -1)}
    if default:
        res.append(compare("scalar_shape/M_tri", A[1][(0, 0)], table))
        res.append(compare("scalar_shape/D_tri", A[-1][(0, 0)], table))
    else:
        print(f"   pair: A_0 = {mp.nstr(A[1][(0, 0)], 15)}   difference: A_0 = {mp.nstr(A[-1][(0, 0)], 15)}")

    print("5. Size of the series")
    print("   |D_N(delta)/D_0| of psi_3:")
    for dl in args.delta:
        D = at_delta(co, dl, N)
        print(f"     delta = {dl:g}: " + ", ".join(mp.nstr(abs(x / D[0]), 3) for x in D[1:11]))
    D = at_delta(co, 1, N)
    print(f"   |D_(N+2)/D_N| at N = {N - 2}: {mp.nstr(abs(at_delta(co, 0, N)[N] / at_delta(co, 0, N)[N - 2]), 5)} (delta = 0), "
          f"{mp.nstr(abs(D[N] / D[N - 2]), 5)} (delta = 1); the limit 1/4 is the collinear point R = 2")
    shapes = [("R=0.3 delta=0", 0.3, 0), ("R=0.5 delta=0", 0.5, 0), ("equilateral", 1, 0),
              ("folded k1=2k2=2k3", mp.mpf(2) / 3, 1), ("folded k2=2k1=2k3", mp.mpf(2) / 3, -1),
              ("R=0.5 delta=0.5", 0.5, 0.5), ("R=0.5 delta=1", 0.5, 1)]
    print("   series over its leading term, wavefunction coefficient and correlator (pair family), with the change from")
    print("   the last four orders:")
    for name, R, dl in shapes:
        w, c = ratio(co, R, dl, N), ratio(A[1], R, dl, N)
        tr = abs(1 - ratio(co, R, dl, N, 4) / w)
        print(f"     {name:20s} psi_3: {mp.nstr(abs(w), 7):>9s}   correlator: {mp.nstr(abs(c), 7):>9s}   ({mp.nstr(tr, 2)})")
        if default and tr < mp.mpf("1e-5"):
            res.append(compare(f"scalar_shape/|series/leading| psi {name}", abs(w), table, quiet=True))
            res.append(compare(f"scalar_shape/|series/leading| correlator {name}", abs(c), table, quiet=True))

    if default:
        print("6. The triangle integral of Ref. [26] at delta != 0 (K = 1, k1 = 1 + R delta/2, k2 = 1 - R delta/2, k3 = R)")
        data = direct_data()
        expo = 1 + T.nuBC
        for tag, pts in data["shapes"].items():
            dl = mp.mpf(tag)
            ser = at_delta(co, dl, N)
            pts = sorted(pts, key=lambda p: p[0])
            degs = data["degrees"]
            print(f"   delta = {tag}: {len(pts)} points, {pts[0][0]:g} <= R <= {pts[-1][0]:g}; rms residual of a polynomial fit of degree "
                  + ", ".join(str(d) for d in degs))
            for label, keep in (("no series", 0), ("N <= 0", 1), ("N <= 1", 2), ("N <= 2", 3), ("N <= 4", 5), ("all orders", N + 1)):
                fit = direct_fit(pts, ser[:keep], expo, degs)
                print(f"     {label:10s} " + "  ".join(mp.nstr(fit[d][0], 2) for d in degs))
                if label == "all orders":
                    res.append(compare(f"scalar_shape/direct residual delta={tag}", fit[degs[-1]][0], table, quiet=True))
            for Nf in (0, 1, 2):
                fit = direct_fit(pts, ser, expo, degs[-2:], free=(Nf,))
                vals = [fit[d][1][Nf] for d in degs[-2:]]
                print(f"     free D_{Nf}: fitted/closed - 1 = " + ", ".join(mp.nstr(abs(v - 1), 2) for v in vals))
                res.append(compare(f"scalar_shape/direct D{Nf} delta={tag}", abs(vals[-1] - 1), table, quiet=True))
    return report(res) if res else 0


if __name__ == "__main__":
    sys.exit(main())
