"""Dirac-pair bubble with an axial chemical potential: the squeezed coefficient, exact and late-time
(Sec. 9.2, App. D; Eqs. [eq:diracMD], [eq:diracfull]).

    M_D(n) = Gamma(r1 + 2 nu + n) S(n),
    S(n)   = sum_{h1 h2} e^(pi (h1 + h2) kappa) sum_{n1 + n2 = n} pbar^h_{n1 n2} sum_{Y = alpha, beta} eps_Y c_Y^(h1) c_Y^(h2) J^Y_{h1 h2}(n1, n2),
    J^Y    = int d^3l/(2 pi)^3 (1 - h1 h2 c)/2 (2 y1)^(nu + n1 - 1/2) (2 y2)^(nu + n2 - 1/2)
             int dt t^(r2 - 2) e^(-x2 t) W_{K_Y^(h1), nu}(2 y1 t) W_{K_Y^(h2), nu}(2 y2 t),

the coefficient of x1^-(r1 + 2 nu + n) in the doubly cut term D at x2 = k = 1. The hard vertex leaves the
late-time ends pbar (both lines on the + branch of the conjugate modes); the soft moment J is computed in
position space (ext/softvertex.py). The script

  * checks the Dirac line: normalization, the per-line amplitude of Eq. [eq:diracocc], and the vanishing of
    the leading pair coefficient at kappa = 0;
  * reproduces S(0), S(1) and the nu -> -nu family at m = kappa = 1 against the values obtained earlier from
    a fit of the directly integrated doubly cut term (Oct. 15 check), and M_full of Eq. [eq:diracfull];
  * compares the exact soft moment with the late-time pair term (both soft lines on their pair branch at the
    soft vertex) and with the late-time estimate that keeps all branch assignments at leading order.

Usage:  python dirac_pair.py           (about 4 min; --quick: the m = kappa = 1 checks only, about 1 min)
"""
import sys

import mpmath as mp

from ext.blocks import C2hat
from ext.dirac import DiracLine
from ext.record import compare, load, report
from ext.softvertex import Grid, Leg, soft_J

mp.mp.dps = 30
I = mp.mpc(0, 1)
HEL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def monomials(h1, h2):
    """(1 - h1 h2 c)/2 = 1/2 - (h1 h2/4)(k^2/(y1 y2) - y1/y2 - y2/y1), as (coef, d1, d2) at k = 1."""
    hh = h1 * h2
    return [(mp.mpf(1) / 2, 0, 0), (-mp.mpf(hh) / 4, -1, -1), (mp.mpf(hh) / 4, 1, -1), (mp.mpf(hh) / 4, -1, 1)]


class DiracPair:
    def __init__(self, m, kappa, r2=2, grid=None):
        self.D = DiracLine(m, kappa)
        self.nu = self.D.nu
        self.r2 = mp.mpf(r2)
        self.grid = grid or Grid()
        self._legs = {}

    def leg(self, A, K):
        key = (mp.nstr(A, 25), mp.nstr(K, 25))
        if key not in self._legs:
            self._legs[key] = Leg(A, K, self.nu)
        return self._legs[key]

    def pbar(self, h1, h2, n1, n2, s=1):
        """Hard-vertex late-time ends of the conjugate modes, both lines on branch s."""
        D = self.D
        return mp.fsum(D.eps(X) * D.E(X, h1, s, n1, conj=True)[n1] * D.E(X, h2, s, n2, conj=True)[n2] for X in 'ab')

    def J(self, Y, h1, h2, n1, n2, s=1, mode='exact', late_branches=(1,)):
        """Soft moment J^Y; mode 'exact' (position space) or 'late' (W -> its branches in late_branches)."""
        D, nu = self.D, self.nu
        K1, K2 = D.K(Y, h1), D.K(Y, h2)
        q = self.r2 - 1                       # t^(r2-2) = t^(q-1)
        tot = 0
        for coef, d1, d2 in monomials(h1, h2):
            A1, A2 = s * nu + n1 - 0.5 + d1, s * nu + n2 - 0.5 + d2
            pre = coef * mp.mpf(2) ** (2 * s * nu + n1 + n2 - 1)
            if mode == 'exact':
                tot += pre * soft_J(self.leg(A1, K1), self.leg(A2, K2), q, self.grid)
            else:
                for b1 in late_branches:
                    for b2 in late_branches:
                        Bb = D.B(b1, K1) * D.B(b2, K2) * mp.mpf(2) ** (1 + (b1 + b2) * nu)
                        a, b = A1 + 0.5 + b1 * nu, A2 + 0.5 + b2 * nu
                        loop = C2hat(a + 1, b + 1) / (2 * mp.pi) ** 2      # int d^3l/(2pi)^3 y1^a y2^b, k = 1
                        tot += pre * Bb * loop * mp.gamma(q + 1 + (b1 + b2) * nu)
        return tot

    def S(self, n, s=1, mode='exact', late_branches=(1,)):
        D = self.D
        tot = 0
        for h1, h2 in HEL:
            w = mp.exp(mp.pi * (h1 + h2) * D.kappa)
            for n1 in range(n + 1):
                n2 = n - n1
                pb = self.pbar(h1, h2, n1, n2, s)
                inner = mp.fsum(D.eps(Y) * D.c(Y, h1) * D.c(Y, h2) * self.J(Y, h1, h2, n1, n2, s, mode, late_branches)
                                for Y in 'ab')
                tot += w * pb * inner
        return tot

    def M(self, n, r1=4, **kw):
        s = kw.get('s', 1)
        return mp.gamma(r1 + 2 * s * self.nu + n) * self.S(n, **kw)


def line_checks():
    """Normalization |alpha|^2 + |beta|^2 = 1, per-line amplitude (m/2 mubar) sqrt(n(1-n)), kappa = 0 zero."""
    worst_norm = worst_amp = 0
    for (m, ka) in [(1, 1), (0.5, 0.25), (2, 1)]:
        D = DiracLine(m, ka)
        for h in (1, -1):
            for tau in (-mp.mpf('0.01'), -mp.mpf(1), -mp.mpf(10)):
                nn = abs(D.mode('a', h, 1, tau)) ** 2 + abs(D.mode('b', h, 1, tau)) ** 2
                worst_norm = max(worst_norm, abs(nn - 1))
            n = D.occupation(h)
            target = D.m / (2 * D.mubar) * mp.sqrt(n * (1 - n))
            for X in 'ab':
                K = D.K(X, h)
                lhs = mp.exp(mp.pi * h * D.kappa) * abs(D.c(X, h)) ** 2 * abs(D.B(1, K) * D.B(1, mp.conj(K)))
                worst_amp = max(worst_amp, abs(lhs / target - 1))
    # at kappa = 0 the leading pair coefficient pbar_00 vanishes for every helicity pair
    P = DiracPair(1.3, 0)
    zero = max(abs(P.pbar(h1, h2, 0, 0)) / abs(P.D.E('a', h1, 1, 0, conj=True)[0] ** 2) for h1, h2 in HEL)
    return worst_norm, worst_amp, zero


def main(quick=False):
    res = []
    print("1. Dirac line")
    wn, wa, z = line_checks()
    print(f"   |alpha|^2 + |beta|^2 - 1: {mp.nstr(wn, 3)};  per-line amplitude [eq:diracocc]: {mp.nstr(wa, 3)};"
          f"  pbar_00 at kappa = 0 (relative): {mp.nstr(z, 3)}")
    res += [(wn < 1e-20, wn), (wa < 1e-20, wa), (z < 1e-25, z)]

    print("2. Squeezed coefficient at m = kappa = 1, x2 = k = 1, r1 = 4, r2 = 2")
    P = DiracPair(1, 1)
    S0, S1 = P.S(0), P.S(1)
    Sm0 = P.S(0, s=-1)
    res.append(compare("dirac_pair/S(0) m=k=1", S0))
    res.append(compare("dirac_pair/S(1) m=k=1", S1))
    res.append(compare("dirac_pair/S(0) nu->-nu m=k=1", Sm0))
    M0, M1 = mp.gamma(4 + 2 * P.nu) * S0, mp.gamma(5 + 2 * P.nu) * S1
    res.append(compare("dirac_pair/M_D(0) m=k=1", M0))
    res.append(compare("dirac_pair/M_D(1) m=k=1", M1))
    gam = 1
    Mf0 = M0 * (1 + gam * (3 + P.nu) / (1 + P.nu))
    Mf1 = M1 * (1 + gam * (3 + P.nu + 0.5) / (1 + P.nu + 0.5))
    res.append(compare("dirac_pair/M_full(0) m=k=1 gamma=1", Mf0))
    res.append(compare("dirac_pair/M_full(1) m=k=1 gamma=1", Mf1))

    print("3. Exact soft moment against the late-time estimates (n = 0)")
    pts = [(1, 1)] if quick else [(0.5, 0.25), (0.5, 0.5), (1, 0.5), (1, 1), (1, 1.5), (2, 1), (2, 2), (1.5, 1.5)]
    print("    (m, kappa)   |exact/pair-only late|   |exact/all-branch late|   phase(exact/all-branch)")
    ratios = []
    for m, ka in pts:
        P = DiracPair(m, ka)
        ex = S0 if (m, ka) == (1, 1) else P.S(0)
        lp = P.S(0, mode='late', late_branches=(1,))
        la = P.S(0, mode='late', late_branches=(1, -1))
        rp, ra = ex / lp, ex / la
        ratios.append((m, ka, rp, ra))
        print(f"    ({m}, {ka})      {mp.nstr(abs(rp), 4):>12s}            {mp.nstr(abs(ra), 4):>8s}               {mp.nstr(abs(mp.arg(ra)), 3)}")
    table = load()
    for m, ka, rp, ra in ratios:
        key = f"dirac_pair/|exact/late pair| ({m},{ka})"
        if key in table:
            res.append(compare(key, abs(rp), table))
    if not quick:
        lo, hi = min(abs(r[3]) for r in ratios), max(abs(r[3]) for r in ratios)
        plo, phi = min(abs(mp.arg(r[3])) for r in ratios), max(abs(mp.arg(r[3])) for r in ratios)
        print(f"   all-branch estimate: |exact/estimate| from {mp.nstr(lo, 3)} to {mp.nstr(hi, 3)}, phase off by {mp.nstr(plo, 2)} to {mp.nstr(phi, 2)} rad")
        res.append(compare("dirac_pair/all-branch min |ratio|", lo))
        res.append(compare("dirac_pair/all-branch max |ratio|", hi))
        res.append(compare("dirac_pair/all-branch min phase", plo))
        res.append(compare("dirac_pair/all-branch max phase", phi))
    return report(res)


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
