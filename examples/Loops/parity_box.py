"""The collapsed parity-odd trispectrum of a helical vector loop (Sec. 8.4; Eqs. [eq:Pitensor], [eq:Mbox]).

Model of Wang and Xianyu (the gauge-boson bubble of Sec. 8.2 with its hard line restored, Sec. 8.3). In the
collapsed limit s = |k1 + k2| -> 0 both ends of each soft line are hard, so the soft lines enter only through
their late-time ends g(a, b) (ext/helical.py). On each side the two soft lines meet either at the quartic
contact vertex (no direction) or at two cubic vertices joined by a hard line, which contributes the tensor

    Pi_jk(a, a') = delta_jk Q_mono + (phat_j phat_k - delta_jk/3) Q_Delta - (i/2) eps_jkl phat_l Q_O,
    Q_mono = (Q_+ + Q_- + Q_L)/3,  Q_Delta = Q_L - (Q_+ + Q_-)/2,  Q_O = Q_+ - Q_-,

with phat = khat1 (side 12) or khat3 (side 34) and the hard moments Q_lambda of ext/helical.py, which are
closed forms (transverse: Eq. [eq:helmoment]; longitudinal: by parts plus the Wronskian). The box is

    M_box = sum_{l1 l2} sum_{a_i b_i} g_l1(a1, b3) g_l2(a2, b4) Pi12_jk(a1, a2) Pi34_mn(b3, b4)
            int d^3l/(2pi)^3 (2 y1)^(2 nu) (2 y2)^(2 nu) P^l1_jm(lhat) P^l2_kn(what),     w = s - l,

with P^h = (delta - nn - i h eps.n)/2 and P^L = nn. The loop integral is done exactly: after the azimuthal
average around shat the integrand is a Laurent polynomial in y1, y2, and each monomial is the block C2.
With shat = z, khat1 = (sin a1, 0, cos a1), khat3 = (sin a3 cos b, sin a3 sin b, cos a3), the box is a
trigonometric polynomial in the dihedral angle b with harmonics c_n e^(i n b), |n| <= 2.

For the full collapsed trispectrum each side is c_c delta_jk [a = a'] + c_p Pi_jk(a, a'), with the weights of
the bispectrum (Sec. 8.3): c_c = 2^(-8-2nu) Gamma(4 + 2 nu), c_p = -m_h^2 e^(-2 pi Omega)/8, Omega = max(kappa - mu, 0).
(CAVEAT, as in the paper: this product form may miscount the symmetry factors of bubble against box.)

Checks: the closed-form hard moments against direct quadrature; kappa = 0 has no parity-odd part;
kappa -> -kappa exchanges c_n and c_-n; c_(+-2) is proportional to sin^2 a1 sin^2 a3; no |n| = 3.

Usage:  python parity_box.py            (about 5 min)
"""
import sys
from itertools import product

import mpmath as mp

from ext.blocks import C2hat
from ext.helical import VectorLine
from ext.record import compare, report

mp.mp.dps = 25
I = mp.mpc(0, 1)
LAMS = (1, -1, 0)                 # soft helicities: transverse +-, longitudinal


# ------------------------------------------------------------------------- polynomial algebra
# monomials keyed by (e_y1, e_y2, e_rho, e_cos, e_sin) -> coefficient
def padd(*terms):
    out = {}
    for c, p in terms:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return {k: v for k, v in out.items() if v != 0}


def pmul(p, q):
    out = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            out[k] = out.get(k, 0) + v1 * v2
    return {k: v for k, v in out.items() if v != 0}


def mono(c, e1=0, e2=0, er=0, ec=0, es=0):
    return {(e1, e2, er, ec, es): mp.mpmathify(c)}


Z1 = padd((0.5, mono(1, 2)), (0.5, mono(1)), (-0.5, mono(1, 0, 2)))       # l.z = (y1^2 + 1 - y2^2)/2
LVEC = [mono(1, 0, 0, 1, 1, 0), mono(1, 0, 0, 1, 0, 1), Z1]               # l = (rho cos, rho sin, z1)
WVEC = [mono(-1, 0, 0, 1, 1, 0), mono(-1, 0, 0, 1, 0, 1), padd((1, mono(1)), (-1, Z1))]   # w = s - l
LHAT = [pmul(c, mono(1, -1)) for c in LVEC]
WHAT = [pmul(c, mono(1, 0, -1)) for c in WVEC]
RHO2 = padd((1, mono(1, 2)), (-1, pmul(Z1, Z1)))                          # rho^2 = y1^2 - z1^2


def levi(i, j, k):
    return (i - j) * (j - k) * (k - i) / 2


def projector(lam, n):
    """P^lam_ij(n) as a 3x3 array of polynomials; n is a list of 3 polynomials (a unit vector)."""
    P = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            nn = pmul(n[i], n[j])
            if lam == 0:
                P[i][j] = nn
                continue
            eps = {}
            for k in range(3):
                e = levi(i, j, k)
                if e:
                    eps = padd((1, eps), (e, n[k]))
            P[i][j] = padd((0.5 if i == j else 0, mono(1)), (-0.5, nn), (-0.5 * I * lam, eps))
    return P


def phi_average(p):
    """Average over the azimuth; rho^(2k) -> (y1^2 - z1^2)^k. Returns {(e1, e2): coef}."""
    out = {}
    for (e1, e2, er, ec, es), v in p.items():
        if ec % 2 or es % 2:
            continue
        avg = mp.gamma((ec + 1) / mp.mpf(2)) * mp.gamma((es + 1) / mp.mpf(2)) / (mp.pi * mp.gamma((ec + es) / mp.mpf(2) + 1))
        assert er % 2 == 0
        r = {(0, 0, 0, 0, 0): mp.mpf(1)}
        for _ in range(er // 2):
            r = pmul(r, RHO2)
        for (f1, f2, _, _, _), w in r.items():
            k = (e1 + f1, e2 + f2)
            out[k] = out.get(k, 0) + v * avg * w
    return out


def loop_tensor(nu):
    """L[(l1, l2)][j][m][k][n] = int d^3l/(2pi)^3 (2y1)^(2nu) (2y2)^(2nu) P^l1_jm(lhat) P^l2_kn(what), s = 1."""
    blocks = {}
    def blk(a, b):
        if (a, b) not in blocks:
            blocks[(a, b)] = mp.mpf(2) ** (4 * nu) * C2hat(a + 2 * nu + 1, b + 2 * nu + 1) / (2 * mp.pi) ** 2
        return blocks[(a, b)]
    L = {}
    for l1, l2 in product(LAMS, LAMS):
        P1, P2 = projector(l1, LHAT), projector(l2, WHAT)
        T = [[[[0] * 3 for _ in range(3)] for _ in range(3)] for _ in range(3)]
        for j, m, k, n in product(range(3), repeat=4):
            avg = phi_average(pmul(P1[j][m], P2[k][n]))
            T[j][m][k][n] = mp.fsum(v * blk(a, b) for (a, b), v in avg.items())
        L[(l1, l2)] = T
    return L


# ------------------------------------------------------------------------- the box
class CollapsedBox:
    def __init__(self, mu, kappa, mu_h=None, kappa_h=None):
        self.mu, self.kappa = mp.mpf(mu), mp.mpf(kappa)
        self.nu = I * self.mu
        mu_h = self.mu if mu_h is None else mp.mpf(mu_h)
        kappa_h = self.kappa if kappa_h is None else mp.mpf(kappa_h)
        self.soft = {l: VectorLine(mu, kappa, l) for l in LAMS}
        alpha = 0.5 + self.nu
        Q = {l: VectorLine(mu_h, kappa_h, l).Q(alpha) for l in LAMS}
        self.Qx = {}
        for ab in Q[1]:
            Qp, Qm, QL = Q[1][ab], Q[-1][ab], Q[0][ab]
            self.Qx[ab] = {'mono': (Qp + Qm + QL) / 3, 'D': QL - (Qp + Qm) / 2, 'O': Qp - Qm}
        self.L = loop_tensor(self.nu)
        mh2 = mu_h ** 2 + mp.mpf(1) / 4
        Om = max(self.kappa - self.mu, 0)
        self.cc = mp.mpf(2) ** (-8 - 2 * self.nu) * mp.gamma(4 + 2 * self.nu)
        self.cp = -mh2 * mp.exp(-2 * mp.pi * Om) / 8

    @staticmethod
    def structures(p):
        """The three tensors of Pi: delta, phat phat - delta/3, -(i/2) eps.phat."""
        d = [[mp.mpf(1) if i == j else 0 for j in range(3)] for i in range(3)]
        D = [[p[i] * p[j] - d[i][j] / 3 for j in range(3)] for i in range(3)]
        O = [[-I / 2 * mp.fsum(levi(i, j, k) * p[k] for k in range(3)) for j in range(3)] for i in range(3)]
        return {'mono': d, 'D': D, 'O': O}

    def side(self, a, a2, p, weights):
        """Side tensor for branches (a, a2) on its two soft-line ends: cc delta [a = a2] + cp Pi(a, a2)."""
        cc, cp = weights
        S = self.structures(p)
        T = [[0] * 3 for _ in range(3)]
        for X in ('mono', 'D', 'O'):
            q = cp * self.Qx[(a, a2)][X]
            for i in range(3):
                for j in range(3):
                    T[i][j] += q * S[X][i][j]
        if cc and a == a2:
            for i in range(3):
                T[i][i] += cc
        return T

    def M(self, a1, a3, beta, weights=(0, 1)):
        """Collapsed coefficient at angles alpha1, alpha3 (to shat) and dihedral beta.
        weights = (c_contact, c_pair) per side: (0, 1) is the box alone, eq:Mbox."""
        k1 = [mp.sin(a1), 0, mp.cos(a1)]
        k3 = [mp.sin(a3) * mp.cos(beta), mp.sin(a3) * mp.sin(beta), mp.cos(a3)]
        tot = 0
        sides12 = {(a, b): self.side(a, b, k1, weights) for a in (1, -1) for b in (1, -1)}
        sides34 = {(a, b): self.side(a, b, k3, weights) for a in (1, -1) for b in (1, -1)}
        for l1, l2 in product(LAMS, LAMS):
            Lt = self.L[(l1, l2)]
            g1, g2 = self.soft[l1].g, self.soft[l2].g
            for a1_, a2_, b3, b4 in product((1, -1), repeat=4):
                w = g1(a1_, b3) * g2(a2_, b4)
                A, B = sides12[(a1_, a2_)], sides34[(b3, b4)]
                tot += w * mp.fsum(A[j][k] * B[m][n] * Lt[j][m][k][n]
                                   for j, m, k, n in product(range(3), repeat=4) if A[j][k] != 0 and B[m][n] != 0)
        return tot

    def harmonics(self, a1=mp.pi / 2, a3=mp.pi / 2, weights=(0, 1), nb=12):
        """c_n, n = -3..3, of M(beta) = sum_n c_n e^(i n beta) (DFT on nb points; exact for |n| <= 2)."""
        vals = [self.M(a1, a3, 2 * mp.pi * j / nb, weights) for j in range(nb)]
        return {n: mp.fsum(v * mp.exp(-I * n * 2 * mp.pi * j / nb) for j, v in enumerate(vals)) / nb for n in range(-3, 4)}


def odd_fraction(c):
    """Parity-odd part over the monopole, as quoted in Sec. 8.4: max_n |c_n - c_-n| / (2 |c_0|), n = 1, 2.
    (The sin(n beta) term of M(beta) has amplitude |c_n - c_-n|, twice this.)"""
    return max(abs(c[n] - c[-n]) for n in (1, 2)) / (2 * abs(c[0]))


def main():
    res = []
    print("1. Closed-form hard moments against direct quadrature (both transverse helicities and the longitudinal line)")
    w = check_hard_moments()
    print(f"   max relative difference over (mu, kappa) = (1, 0.75), (2, 2.5) and the four SK moments: {mp.nstr(w, 3)}")
    res.append(compare("parity_box/hard moments closed vs quadrature", w))
    print("2. kappa = mu = 1, khat1 and khat3 perpendicular to s: harmonics c_n of the box (Eq. [eq:Mbox])")
    B = CollapsedBox(1, 1)
    c = B.harmonics()
    for n in range(-3, 4):
        print(f"   c_{n:+d} = {mp.nstr(c[n], 6):>34s}   |c| = {mp.nstr(abs(c[n]), 4)}")
    res.append(compare("parity_box/|c_-2/c_+2| mu=1 kappa=1", abs(c[-2] / c[2])))
    res.append(compare("parity_box/|c_-1/c_+1| mu=1 kappa=1", abs(c[-1] / c[1])))
    res.append(compare("parity_box/|c_3|/|c_0| mu=1 kappa=1", abs(c[3]) / abs(c[0])))
    # c_(+-2) proportional to sin^2 a1 sin^2 a3
    a1, a3 = mp.mpf('0.7'), mp.mpf('1.1')
    c2 = B.harmonics(a1, a3)
    dev = max(abs(c2[n] / c[n] / (mp.sin(a1) ** 2 * mp.sin(a3) ** 2) - 1) for n in (-2, 2))
    print(f"   c_(+-2)(a1 = 0.7, a3 = 1.1) / [c_(+-2)(pi/2, pi/2) sin^2 a1 sin^2 a3] - 1: {mp.nstr(dev, 3)}")
    res.append(compare("parity_box/sin^2 law deviation", dev))
    print("3. kappa = 0: parity-odd part; kappa -> -kappa exchanges c_n and c_-n")
    c0 = CollapsedBox(1, 0).harmonics()
    odd0 = max(abs(c0[n] - c0[-n]) for n in (1, 2)) / abs(c0[0])
    cm = CollapsedBox(1, -1).harmonics()
    sw = max(abs(cm[n] - c[-n]) / abs(c[-n]) for n in (-2, -1, 0, 1, 2))
    print(f"   kappa = 0: max |c_n - c_-n|/|c_0| = {mp.nstr(odd0, 3)};  kappa = -1 vs kappa = 1 swapped: {mp.nstr(sw, 3)}")
    res.append(compare("parity_box/kappa=0 odd part", odd0))
    res.append(compare("parity_box/kappa reflection", sw))
    print("4. Chirality of the dihedral quadrupole |c_-2/c_+2|, and the parity-odd fraction of the full")
    print("   collapsed trispectrum (contact + pair vertices on each side, weights of Sec. 8.3)")
    print("     mu  kappa   |c_-2/c_+2|   box |c_-2|/|c_0|   full: odd/monopole")
    scan = [(1, k) for k in ('0.1', '0.25', '0.5', '0.75', '1', '1.5', '2')] + \
           [(2, k) for k in ('0.25', '0.5', '1', '1.5', '2', '2.5', '3')]
    for mu, ka in scan:
        Bk = B if (mu, ka) == (1, '1') else CollapsedBox(mu, ka)
        ck = c if (mu, ka) == (1, '1') else Bk.harmonics()
        full = Bk.harmonics(weights=(Bk.cc, Bk.cp))
        r2, fr = abs(ck[-2] / ck[2]), odd_fraction(full)
        print(f"   {mu:3d}  {ka:>5s}   {mp.nstr(r2, 4):>10s}   {mp.nstr(abs(ck[-2]) / abs(ck[0]), 3):>10s}       {mp.nstr(fr, 3)}")
        for key, val in ((f"parity_box/|c_-2/c_+2| mu={mu} kappa={ka}", r2), (f"parity_box/odd fraction mu={mu} kappa={ka}", fr)):
            try:
                res.append(compare(key, val, quiet=True))
                if not res[-1][0]:
                    print(f"   FAIL {key}: {mp.nstr(val, 5)}")
            except KeyError:
                pass
    return report(res)


def check_hard_moments(points=((1, 0.75), (2, 2.5)), dps=20):
    """Direct quadrature (Chebyshev panels in ln t) of the hard moments against the closed forms."""
    from ext.cheb import Panels
    def DW(K, nu, x):
        return (x / 2 - K - 1) * mp.whitw(K, nu, x) - mp.whitw(K + 1, nu, x)
    def DM(K, m_, x):
        return (x / 2 - K - 1) * mp.whitm(K, m_, x) + (0.5 + m_ + K) * mp.whitm(K + 1, m_, x)
    worst = 0
    with mp.workdps(dps):
        P = Panels([(-30, 1.0, 0.5), (1.0, 3.8, 0.1)], 24)
        for mu, ka in points:
            alpha = 0.5 + I * mp.mpf(mu)
            for h in LAMS:
                L = VectorLine(mu, ka, h)
                nu, m = L.nu, L.m
                if h != 0:
                    fns = (L.e, L.estar, L.ebar, L.estarbar)
                else:
                    fns = (lambda t: I / m * mp.fsum(L.A[s] / mp.sqrt(2) * DM(0, s * nu, 2 * t) for s in (1, -1)),
                           lambda t: -I / m / mp.sqrt(2) * DW(0, nu, 2 * t),
                           lambda t: -I / m * mp.fsum(mp.conj(L.A[s]) / mp.sqrt(2) * DM(0, -s * nu, 2 * t) for s in (1, -1)),
                           lambda t: I / m / mp.sqrt(2) * DW(0, nu, 2 * t))
                tab = [[[t ** alpha * mp.exp(-t) * t * fn(t) for t in row] for row in P.t] for fn in fns]
                Ce, Ceb = P.cumulative(tab[0]), P.cumulative(tab[2])
                d = {(1, 1): 2 * P.integral([[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(Ce, tab[1])]),
                     (-1, -1): 2 * P.integral([[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(Ceb, tab[3])]),
                     (1, -1): P.integral(tab[1]) * P.integral(tab[3])}
                cf = L.Q(alpha)
                worst = max(worst, max(abs(d[k] / cf[k] - 1) for k in d))
    return worst


if __name__ == "__main__":
    sys.exit(main())
