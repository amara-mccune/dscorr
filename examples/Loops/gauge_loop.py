"""The gauge-boson loop of the Standard Model background: exact against late-time soft vertex (Sec. 6.3).

Vertex f_A(X) F_mn F^mn with a massive (Proca) vector, all three polarizations, no chemical potential.
In the squeezed bispectrum the soft vertex is f_A' phidot_0 a^-1 dphi' (B1.B2 - E1.E2) (from
delta X = phidot_0 a^-1 dphi'); its time weight a^-1 tau e^(i k tau) ~ t^2 makes the power r2 = 2. We compare the soft
moment of the pair, weighted by the late-time hard ends (E fields on the branch tau^(-1/2 + nu)), with its
late-time version, in which the soft vertex keeps only E on the same branch (B is subleading at late times).
The claim of Sec. 6.3 is that the sum over the nine helicity pairs has the exact/late ratio
U(3 + 2 nu, 2) of Eq. (6.1), whereas the individual pairs do not.

Modes (conformal time, y = |p|, nu = i mu, m^2 = mu^2 + 1/4): g = W_{0,nu}(2 i y tau)/sqrt(2y) for both
the transverse line and the neutral function of the longitudinal line [Eq. (D.4)]. Then
    transverse:   E = g' eps^h,       B = h y g eps^h          (i phat x eps^h = h eps^h),
    longitudinal: E = -i (m/tau) g phat,  B = 0                (from v_L = (i/m)(tau d - 1) g, v_0 = -(y/m) tau g
                                                                 and g'' + (y^2 + m^2/tau^2) g = 0).
At late times both E fields go as tau^(-1/2 + nu), with relative weights i(1/2 + nu) (T) and m (L). On the
rotated contour tau = i t of the soft vertex (conjugate modes), with z = 2 y t,
    Ebar_T = -2 i y W'(z)/sqrt(2y),   Bbar_T = h y W(z)/sqrt(2y),   Ebar_L = (m/t) W(z)/sqrt(2y),
and W'(z) = W_{0,nu}(z)/2 - W_{1,nu}(z)/z. Polarization overlaps, c = (1 - y1^2 - y2^2)/(2 y1 y2):
TT (1 - h1 h2 c)^2/4, TL (1 - c^2)/2, LL c^2. The exact soft moments use ext/softvertex.py.

Usage:  python gauge_loop.py           (about 10 min)
"""
import sys
from itertools import product

import mpmath as mp

from ext.blocks import C2hat
from ext.record import compare, report
from ext.softvertex import Grid, Leg, soft_J
from universal import U

mp.mp.dps = 45
I = mp.mpc(0, 1)
POLS = (1, -1, 0)


def overlap(l1, l2):
    """{(a, b): coef} for y1^a y2^b (k = 1)."""
    C = {(-1, -1): mp.mpf(1) / 2, (1, -1): -mp.mpf(1) / 2, (-1, 1): -mp.mpf(1) / 2}       # c
    def mul(p, q):
        out = {}
        for (a1, b1), c1 in p.items():
            for (a2, b2), c2 in q.items():
                k = (a1 + a2, b1 + b2)
                out[k] = out.get(k, 0) + c1 * c2
        return out
    one = {(0, 0): mp.mpf(1)}
    if l1 != 0 and l2 != 0:
        lin = {k: -l1 * l2 * v for k, v in C.items()}
        lin[(0, 0)] = lin.get((0, 0), 0) + 1
        return {k: v / 4 for k, v in mul(lin, lin).items()}
    if l1 == 0 and l2 == 0:
        return mul(C, C)
    c2 = mul(C, C)
    out = {k: -v / 2 for k, v in c2.items()}
    out[(0, 0)] = out.get((0, 0), 0) + mp.mpf(1) / 2
    return out


# soft functions as lists of (coef, y-power shift, K, t-power shift): W(z) and W'(z) with z = 2 y t
W0 = [(mp.mpf(1), 0, 0, 0)]
DW = [(mp.mpf(1) / 2, 0, 0, 0), (-mp.mpf(1) / 2, -1, 1, -1)]


class GaugeLoop:
    def __init__(self, mu, grid=None):
        self.mu = mp.mpf(mu)
        self.nu = I * self.mu
        self.m = mp.sqrt(self.mu ** 2 + mp.mpf(1) / 4)
        self.grid = grid or Grid()
        self.Bp = mp.gamma(-2 * self.nu) / mp.gamma(0.5 - self.nu)     # W_{0,nu} = B_+ M_{0,nu} + ...
        self.legs = {}

    def leg(self, A, K):
        key = (mp.nstr(A, 25), K)
        if key not in self.legs:
            self.legs[key] = Leg(A, K, self.nu)
        return self.legs[key]

    def hard_weight(self, l):
        return I * (0.5 + self.nu) if l != 0 else self.m

    def soft_terms(self, l1, l2):
        """Soft bilinear (Bbar1.Bbar2 - Ebar1.Ebar2) * sqrt(2y1) sqrt(2y2) as [(coef, f1, f2, (dy1, dy2, dt))]."""
        m = self.m
        if l1 != 0 and l2 != 0:
            return [(l1 * l2, W0, W0, (1, 1, 0)), (4, DW, DW, (1, 1, 0))]
        if l1 != 0 and l2 == 0:
            return [(2 * I * m, DW, W0, (1, 0, -1))]
        if l1 == 0 and l2 != 0:
            return [(2 * I * m, W0, DW, (0, 1, -1))]
        return [(-m ** 2, W0, W0, (0, 0, -2))]

    def J(self, l1, l2, mode, delta=0):
        """Soft moment of the pair (l1, l2), up to a common factor of all pairs.

        Single monomials of the polarization overlaps are infrared divergent (log) where the full pair is not;
        delta shifts the hard-end power y^nu -> y^(nu + delta) and J_sym averages delta and -delta.
        """
        nu = self.nu
        w = self.hard_weight(l1) * self.hard_weight(l2)
        tot = 0
        for (a, b), oc in overlap(l1, l2).items():
            base1, base2 = nu + delta - 0.5 + a, nu + delta - 0.5 + b   # hard end y^nu, soft normalization y^(-1/2)
            if mode == 'exact':
                for coef, f1, f2, (dy1, dy2, dt) in self.soft_terms(l1, l2):
                    for c1, s1, K1, t1 in f1:
                        for c2, s2, K2, t2 in f2:
                            A = base1 + dy1 + s1
                            B = base2 + dy2 + s2
                            q = 3 + dt + t1 + t2                    # a^-1 dphi' ~ tau^2: t^2 -> q = 3
                            pre = coef * c1 * c2
                            tot += w * oc * pre * soft_J(self.leg(A, K1), self.leg(B, K2), q, self.grid)
            else:
                # late: -Ebar1.Ebar2 with Ebar_T -> -2iy (1/2 + nu) B+ z^(-1/2 + nu), Ebar_L -> 2 m y B+ z^(-1/2 + nu)
                e = lambda l: -2 * I * (0.5 + nu) if l != 0 else 2 * self.m
                pre = -e(l1) * e(l2) * self.Bp ** 2 * mp.mpf(2) ** (-1 + 2 * nu)
                p1, p2 = base1 + 1 + (-0.5 + nu), base2 + 1 + (-0.5 + nu)
                loop = C2hat(p1 + 1, p2 + 1) / (2 * mp.pi) ** 2
                tot += w * oc * pre * loop * mp.gamma(2 + 2 * nu)  # int t^2 t^(-1 + 2 nu) e^-t
        return tot

    def J_sym(self, l1, l2, mode, delta=mp.mpf(10) ** -12):
        return (self.J(l1, l2, mode, delta) + self.J(l1, l2, mode, -delta)) / 2


def main():
    res = []
    print("Exact over late-time soft moment of the W/Z pair (r2 = 2), summed over the nine helicity pairs")
    allpr, mid = [], []
    print("    mu      ratio (sum)                       U(3 + 2 nu, 2)                 |ratio/U - 1|   pair ratios (min, max)")
    for mu in ('0.5', '1', '1.5', '2', '3', '4'):
        G = GaugeLoop(mu)
        ex = {p: G.J_sym(*p, 'exact') for p in product(POLS, POLS)}
        la = {p: G.J_sym(*p, 'late') for p in product(POLS, POLS)}
        r = mp.fsum(ex.values()) / mp.fsum(la.values())
        u = U(3 + 2 * G.nu, 2)
        pr = [abs(ex[p] / la[p]) for p in ex if la[p] != 0]
        dev = abs(r / u - 1)
        print(f"   {mu:>4s}  {mp.nstr(r, 10):>30s}  {mp.nstr(u, 10):>30s}   {mp.nstr(dev, 3):>9s}     {mp.nstr(min(pr), 3)}, {mp.nstr(max(pr), 3)}")
        res.append(compare(f"gauge_loop/|ratio/U - 1| mu={mu}", dev, quiet=True))
        allpr += pr
        if 1 <= float(mu) <= 3:
            mid += pr
    print(f"   individual helicity pairs: |exact/late| from {mp.nstr(min(allpr), 3)} to {mp.nstr(max(allpr), 3)} for "
          f"1/2 <= mu <= 4, and from {mp.nstr(min(mid), 3)} to {mp.nstr(max(mid), 3)} for 1 <= mu <= 3")
    res.append(compare("gauge_loop/pair ratio min (1 <= mu <= 3)", min(mid)))
    res.append(compare("gauge_loop/pair ratio max (1 <= mu <= 3)", max(mid)))
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
