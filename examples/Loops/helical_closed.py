"""The soft vertex of the helical-vector loops as Kampe de Feriet series (Secs. 7.2 and 7.3; ext/helclosed.py).

For any pair of first indices and for line powers ia, ib >= -2, the soft moment

    J(ia, ib, q; K1, K2) = int d^3l/(2 pi)^3 y1^(nu - 1/2 + ia) y2^(nu - 1/2 + ib)
                           int dt t^(q-1) e^(-t) W_{K1,nu}(2 y1 t) W_{K2,nu}(2 y2 t)

is one Euler integral over u in (0, 1) of elementary factors and one Gauss function with a unit parameter
(ext/helclosed.py). This covers the two cases that ext/softclosed.py leaves open for the helical vector: pairs with
one transverse and one longitudinal line, and the monomials with y^-2 of the polarization sums. The simplest moment
is explicit,

    J(-1, -1, q; K1, K2) = 2^(1-q) Gamma(q - 1 - 2 nu)/(4 pi^2 Gamma(c1) Gamma(c2))
        { 2 pi cot(2 pi nu) g(q + 1 + 2 nu, c1 + c2 - 1)
          + sum over the two lines l (lb = the other line) of
              pi/sin(2 pi nu) g(q + 1 + 2 nu, c_lb - c_l - 2 nu)
              - B(c_l, 2 nu) [Xi(q + 1, c_lb; 1 - c_l - 2 nu, 1 - 2 nu) + Xi(q + 1, c_lb; c_l, 1 - 2 nu)] },

with c = 1/2 - nu - K,

    g(A, B)        = int_0^1 du u^(A-1) (1-u)^(B-1)/(2 - u) = B(A, B) 2F1(1, A; A + B; 1/2)/2,
    Xi(A, B; b, c) = int_0^1 du u^(A-1) (1-u)^(B-1)/(2 - u) 2F1(1, b; c; u) = sum_n T_n/2^(n+1),
    T_n = (c-1)/(c-1-b) B(A+n, B) 3F2(1, b, B; b-c+2, A+n+B; 1) + Gamma(c) Gamma(1+b-c)/Gamma(b) B(A+n+1-c, B+c-1-b).

Xi is a Kampe de Feriet function of arguments (1/2, 1). Its unit-argument sum has excess A + n - c + 1, so it
converges absolutely at the physical exponents, and the other sum converges as 2^-n. The T_n obey
(A+n+B-b) T_(n+1) = (A+n-c+1) T_n + (c-1) B(A+n, B). Other monomials are finite sums of functions of this kind
(for y^-2 with one more pair of parameters). The module evaluates all of them from the Euler integral, split at
u = 1/2, where every series converges as 2^-k.

The script checks
  * the explicit formula above against the module (mixed, equal, and opposite first indices);
  * single moments of every kind against soft_J (position space). The reference averages two shifted powers for
    y^-2 and loses a third of its digits there, so it runs at 60 digits and the series at 36;
  * two enhanced lines at kappa = mu, where the exponents of the two lines coincide (no limit is taken);
  * the leading coefficient M_A of the gauge-boson bubble from these moments, against its values of record
    (helical_closed_record.json), against helical_triangle_record.json (the ratio to the late-time coefficient,
    6.07 at kappa = mu = 1), and against M_A with soft_J;
  * with --triangle, the leading coefficient of the triangle (|R_hard|, Sec. 7.3).

Usage:  python helical_closed.py                 mu = kappa = 1 (about 3 min)
        python helical_closed.py --triangle      also |R_hard| at mu = kappa = 1 (about 1 min more)
        python helical_closed.py --all           the benchmark points (1, 1), (2, 2.5), (3, 3.75), (1, 0.75), (1, 0),
                                                 with the triangle (about 8 min)
"""
import json
import os
import sys
import time

import mpmath as mp

from ext.helclosed import HelClosed, HelClosedTable
from ext.record import compare, report
from ext.softvertex import Grid
from ext.vectri import SoftTable, Triangle

I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
HALF = mp.mpf(1) / 2

# 24 moments: closable (softclosed.py), mixed, and with y^-2 (transverse, longitudinal, mixed pairs)
MOMENTS = [(0, 0, 1, "p", "p"), (-1, -1, 1, "p", "m"), (1, -1, 1, "m", "m"), (0, 0, 1, "z", "z"), (0, 0, 1, "z", "o"),
           (0, 0, 1, "p", "z"), (0, -1, 2, "m", "z"), (2, -1, 2, "p", "z"), (1, 0, 2, "z", "m"), (0, 0, 1, "o", "p"),
           (0, 0, 1, "m", "o"),
           (-2, 0, 1, "p", "p"), (0, -2, 1, "p", "m"), (-2, 2, 1, "m", "m"), (-2, -2, 1, "p", "p"), (-2, -2, 1, "m", "p"),
           (-2, -2, 1, "z", "z"), (-2, 3, 2, "o", "z"), (-2, -2, 1, "o", "o"),
           (-2, 2, 1, "m", "z"), (-2, -1, 2, "p", "z"), (-2, -2, 1, "m", "z"), (2, -2, 1, "p", "o"), (-1, -2, 2, "z", "p")]
KINDS = ["closable"] * 5 + ["mixed"] * 6 + ["y^-2 transverse"] * 5 + ["y^-2 longitudinal"] * 3 + ["y^-2 mixed"] * 5


def labels(kappa):
    k = mp.mpf(kappa)
    return {"z": mp.mpf(0), "o": mp.mpf(1), "p": I * k, "m": -I * k}


# ---------------------------------------------------------------------------------------------------------------
# the simplest moment in closed form
# ---------------------------------------------------------------------------------------------------------------
def gfun(A, B):
    """int_0^1 u^(A-1) (1-u)^(B-1)/(2-u) du"""
    return mp.beta(A, B) * mp.hyp2f1(1, A, A + B, HALF) / 2


def T0(A, B, b, c):
    """int_0^1 u^(A-1) (1-u)^(B-1) 2F1(1, b; c; u) du, continued in B: a 3F2 at unit argument with excess A - c + 1"""
    C1 = (c - 1) / (c - 1 - b)
    C2 = mp.gamma(c) * mp.gamma(1 + b - c) / mp.gamma(b)
    return C1 * mp.beta(A, B) * mp.hyp3f2(1, b, B, b - c + 2, A + B, 1) + C2 * mp.beta(A + 1 - c, B + c - 1 - b)


def Xi(A, B, b, c, nmax=None):
    """sum_n T_n/2^(n+1). T_0 is the 3F2 above and the others follow from the recurrence (mpmath's hyp3f2 at unit
    argument is not reliable at large parameters)."""
    nmax = nmax or int((mp.mp.dps + 3) * 3.33)
    T = T0(A, B, b, c)
    tot = mp.mpc(0)
    for n in range(nmax + 1):
        tot += mp.mpf(2) ** (-n - 1) * T
        T = ((A + n - c + 1) * T + (c - 1) * mp.beta(A + n, B)) / (A + n + B - b)
    return tot


def J_simplest(q, nu, K1, K2):
    """J(-1, -1, q; K1, K2) from the formula of the module docstring"""
    c = [HALF - nu - K1, HALF - nu - K2]
    tot = 2 * mp.pi * mp.cot(2 * mp.pi * nu) * gfun(q + 1 + 2 * nu, c[0] + c[1] - 1)
    for l in (0, 1):
        cl, cb = c[l], c[1 - l]
        tot += mp.pi / mp.sin(2 * mp.pi * nu) * gfun(q + 1 + 2 * nu, cb - cl - 2 * nu)
        tot -= mp.beta(cl, 2 * nu) * (Xi(q + 1, cb, 1 - cl - 2 * nu, 1 - 2 * nu) + Xi(q + 1, cb, cl, 1 - 2 * nu))
    return mp.mpf(2) ** (1 - q) * mp.gamma(q - 1 - 2 * nu) / (4 * mp.pi ** 2 * mp.gamma(c[0]) * mp.gamma(c[1])) * tot


def simplest_check(mu="1", kappa="0.75", dps=28):
    """largest relative difference between the explicit formula and the module, over four pairs"""
    with mp.workdps(dps):
        nu = I * mp.mpf(mu)
        K = labels(kappa)
        with mp.workdps(dps + 8):
            H = HelClosed(I * mp.mpf(mu))
            ref = [H.J(-1, -1, q, labels(kappa)[a], labels(kappa)[b]) for q, a, b in ((1, "p", "z"), (2, "m", "z"), (1, "p", "p"), (1, "p", "m"))]
        val = [J_simplest(q, nu, K[a], K[b]) for q, a, b in ((1, "p", "z"), (2, "m", "z"), (1, "p", "p"), (1, "p", "m"))]
        return max(abs(v / r - 1) for v, r in zip(val, ref))


# ---------------------------------------------------------------------------------------------------------------
def moments_check(mu, kappa, dps=36, dps_ref=60, moments=MOMENTS, kinds=KINDS):
    """{kind: largest relative difference from soft_J}, the values, and the two run times"""
    with mp.workdps(dps_ref):
        tab = SoftTable(I * mp.mpf(mu), kappa, grid=Grid(npts=40) if dps_ref > 45 else None)
        t = time.time()
        ref = [tab(*m) for m in moments]
        t_ref = time.time() - t
    with mp.workdps(dps):
        H = HelClosed(I * mp.mpf(mu))
        K = labels(kappa)
        t = time.time()
        val = [H.J(m[0], m[1], m[2], K[m[3]], K[m[4]]) for m in moments]
        t_ser = time.time() - t
        worst, rows = {}, []
        for m, k, r, v in zip(moments, kinds, ref, val):
            d = abs(v / r - 1)
            worst[k] = max(worst.get(k, mp.mpf(0)), d)
            rows.append((m, k, v, d))
    return worst, rows, (t_ser, t_ref)


def leading(mu, kappa, triangle=False, soft=False):
    """M_A of the bubble from the series (and with soft_J if soft), its ratio to the late-time coefficient, and
    |R_hard| of the triangle."""
    out = {}
    t = time.time()
    T = Triangle(mu, kappa, nmax=0, table=HelClosedTable(I * mp.mpf(mu), kappa))
    acc = T.accumulate("bubble")
    out["M_A"] = T.coefficients("bubble", acc=acc)[0]
    out["exact_over_late"] = abs(out["M_A"] / T.coefficients("bubble", acc=acc, late=True)[0])
    out["t_bubble"] = time.time() - t
    out["moments"] = len(T.table.closed._j)
    if triangle:
        t = time.time()
        res = T.coefficients("triangle", isosceles=False)
        mono = res.get((0, 0), 0) + res.get((0, 2), 0) / 3
        fac = -T.hardline.m2 * mp.exp(-2 * mp.pi * max(T.kappa - T.mu, 0))
        out["R_hard"] = abs(fac * mono / out["M_A"])
        out["t_triangle"] = time.time() - t
    if soft:
        t = time.time()
        out["M_A_soft"] = Triangle(mu, kappa, nmax=0).coefficients("bubble")[0]
        out["t_soft"] = time.time() - t
    return out


def main(full=False, triangle=False):
    with open(os.path.join(HERE, "helical_closed_record.json")) as f:
        table = json.load(f)
    with open(os.path.join(HERE, "helical_triangle_record.json")) as f:
        tri = json.load(f)
    res = []
    res.append(compare("helical_closed/simplest moment, explicit formula", simplest_check(), table))
    worst, rows, (ts, tr) = moments_check("1", "0.75")
    print(f"   {len(rows)} moments at (mu, kappa) = (1, 0.75): series {ts:.0f} s (36 digits), soft_J {tr:.0f} s (60 digits)")
    res.append(compare("helical_closed/moments vs soft_J mu=1,kappa=0.75", max(worst.values()), table))
    with mp.workdps(30):
        nu, K = I * mp.mpf(1), -I * mp.mpf(1)
        tab = SoftTable(nu, "1")
        H = HelClosed(nu)
        d = max(abs(H.J(ia, ib, q, K, K) / tab(ia, ib, q, "m", "m") - 1) for ia, ib, q in ((0, 0, 1), (1, -1, 1), (-1, 1, 1)))
        res.append(compare("helical_closed/enhanced pair at kappa=mu vs soft_J", d, table))
    pts = [("1", "1"), ("2", "2.5"), ("3", "3.75"), ("1", "0.75"), ("1", "0")] if full else [("1", "1")]
    for mu, ka in pts:
        with mp.workdps(30):
            L = leading(mu, ka, triangle=(triangle or full) and ka != "0.75", soft=(mu, ka) == ("1", "1"))
            key = f"mu={mu} kappa={ka}"
            print(f"   {key}: M_A = {mp.nstr(L['M_A'], 22)}   ({L['moments']} moments, {L['t_bubble']:.0f} s)")
            res.append(compare(f"helical_closed/M_A {key}", L["M_A"], table))
            if f"leading/exact over late {key}" in tri:
                res.append(compare(f"leading/exact over late {key}", L["exact_over_late"], tri))
            if "R_hard" in L and f"leading/|R_hard| {key}" in tri:
                res.append(compare(f"leading/|R_hard| {key}", L["R_hard"], tri))
            if "M_A_soft" in L:
                res.append(compare("helical_closed/M_A vs soft_J mu=1 kappa=1", L["M_A"] / L["M_A_soft"] - 1, table))
    return report(res)


if __name__ == "__main__":
    sys.exit(main(full="--all" in sys.argv, triangle="--triangle" in sys.argv))
