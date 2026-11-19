"""The collapsed trispectrum of a helical vector loop to all orders in s/k (Sec. 7.4; App. B, "Helical lines").

Kinematics. Four inflaton legs k1..k4 with s = k1 + k2 = -(k3 + k4) soft. Side A holds the legs (12) and side B
the legs (34). Two soft vector lines join the sides: line 1 has momentum l at its A end and line 2 has w = s - l
(the Fourier momenta of the line operators at a vertex add up to the external momentum entering there). On each
side the two soft lines meet either at the quartic contact vertex of Eq. [eq:vecvertex] or at two cubic vertices
joined by a hard line. On side A the hard line runs from V1 (leg k1, soft line 1) to V2 (leg k2, soft line 2) and
its operator at V1 has momentum P = k1 - l. On side B it runs from V3 (leg k3, soft line 1) to V4 (leg k4, soft
line 2) with P = k3 + l. This is the planar attachment (13)(24); the crossed one, (14)(23), is the same function
with k3 <-> k4 (see `evaluate`).

Region. The non-analytic dependence on s comes from loop momenta l ~ s with all vertex times hard, t ~ 1/k.
Each Schwinger-Keldysh function of a soft line is then expanded on its late-time branches at both ends,

    G_ab(y; t_a, t_b)  contains  g(a, b; sig, sig') (2y)^(-1) (2 y t_a)^(1/2 + sig nu) (2 y t_b)^(1/2 + sig' nu)
                                 sum_{n, m} e_n(-a K, sig nu) e_m(-b K, sig' nu) (2 y t_a)^n (2 y t_b)^m,

with the branch series of Eq. [eq:en] (unit steps, e_1 = -K/(1 + 2 nu)) and the connection coefficients of
Eq. [eq:vecAB]. Terms with the same branch at both ends (sig = sig') are the pair pieces of Sec. 7.2. On equal
Schwinger-Keldysh branches a line with different late-time branches at its two ends is the sum of a part symmetric
in the two times, kept here with the coefficient (A_+ B_- + A_- B_+)/2, and sign(t_a - t_b) times the commutator.
The commutator is entire in the line momentum once the three polarizations (and A_0) are summed, so it drops out
of the non-analytic part. The symmetric part does not: its weight is 1 + 2 n_h, which differs between the two
transverse helicities at kappa != 0, and it multiplies the projector (delta - nn - i h eps.n)/2. See notes.

A line with exponents (sig, sig') carries (2y)^((sig + sig') nu + n + m) into the loop integral, t_A^(1/2 + sig nu + n)
into side A and t_B^(1/2 + sig' nu + m) into side B. With p1 = |k1|, p3 = |k3|, the unit-vertex loop integral is

    I = s^3 p1^-4 p3^-4 (s/p1)^(eA nu) (s/p3)^(eB nu) sum_{NA, NB} (s/p1)^NA (s/p3)^NB C_{NA NB}(c1, c3, beta),
    eA = sig_1 + sig_2,  eB = sig_1' + sig_2',

and C is a trigonometric polynomial in the dihedral angle beta, C = sum_n c_n e^(i n beta), whose coefficients are
polynomials in c1 = k1hat.shat, c3 = k3hat.shat times S1^|n| S3^|n|. Geometry: shat = z, k1hat = (S1, 0, c1),
k3hat = (S3 cos beta, S3 sin beta, c3). "Unit vertex" means: each quartic vertex is int dt t^2 e^(-k12 t), each
cubic vertex int dt e^(-k t), the loop measure is d^3l/(2 pi)^3, and all Schwinger-Keldysh branches are summed.
The external-leg factors 1/(2 k_i), the couplings and the symmetry factors are applied in parity_box_series.py.

Components. A longitudinal line has components A_0 and A_L at each end. With the neutral function G, D = t d_t - 1
and c_0 = -(y/m) tau (tau = +-i t on the two branches), and n the momentum direction of the first operator,

    (s, s): n_i n_j D_1 D_2 G/m^2,   (0, 0): c_0(1) c_0(2) G,   (s, 0): (i n_i/m) D_1 c_0(2) G,
    (0, s): -(i n_j/m) c_0(1) D_2 G,

and the vertices contract with eta^{mu nu} = diag(-1, 1, 1, 1). The time-ordered A_0 A_0 function of the hard line has
in addition the contact term of the Proca field, +(t^2/m_h^2) delta(t1 - t2) on equal branches (it first matters at
relative order (s/k)^2). These conventions are the ones checked against direct integration for the triangle.

Hard pair. Its two-time moments at line energy Y = |P| and vertex energies (1, 1 + xi) follow from those at the
doubly folded point by scaling [eq:Dfold], as a double series in u = Y^2 - 1 and xi (closed forms of
ext/helical.py: Eq. [eq:helmoment] for the ordered ones, Gamma functions for the cut ones).

Loop integral. After the azimuthal average around shat the integrand is a Laurent polynomial in y1 = |l|/s and
y2 = |s - l|/s, and each monomial is the block of Eq. [eq:C2]. The geometry is done in exact rational arithmetic
and does not depend on the masses; it is cached on disk.
"""
import os
import pickle
from fractions import Fraction
from itertools import product

import mpmath as mp

from .blocks import C2hat, e_series
from .helical import VectorLine

I = mp.mpc(0, 1)

__all__ = ["Geometry", "HardPair", "CollapsedSeries", "evaluate", "harmonics", "FAMILIES"]

# ------------------------------------------------------------------------------ exact polynomial algebra
# monomial key: exponents of (rA, rB, lx, ly, lz, y1, y2, S1, c1, S3, c3, E = e^(i beta), i)
RA, RB, LX, LY, LZ, Y1, Y2, S1, C1, S3, C3, EB, IB = range(13)
NK = 13
_NAMES = dict(ra=RA, rb=RB, lx=LX, ly=LY, lz=LZ, y1=Y1, y2=Y2, s1=S1, c1=C1, s3=S3, c3=C3, e=EB, ib=IB)
HALF = Fraction(1, 2)


def mono(c=1, **kw):
    k = [0] * NK
    for name, v in kw.items():
        k[_NAMES[name]] = v
    return {tuple(k): Fraction(c)}


def padd(*terms):
    out = {}
    for c, p in terms:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return {k: v for k, v in out.items() if v != 0}


def pmul(p, q, nmax=None):
    out = {}
    for k1, v1 in p.items():
        o1 = k1[RA] + k1[RB]
        for k2, v2 in q.items():
            if nmax is not None and o1 + k2[RA] + k2[RB] > nmax:
                continue
            k = [a + b for a, b in zip(k1, k2)]
            v = v1 * v2
            if k[IB] == 2:
                k[IB] = 0
                v = -v
            k = tuple(k)
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


ONE = mono()
ZERO = {}


def levi(i, j, k):
    return (i - j) * (j - k) * (k - i) // 2


def _dot(u, v, nmax=None):
    return padd(*[(1, pmul(a, b, nmax)) for a, b in zip(u, v)])


def _outer(u, v, nmax=None):
    return [[pmul(a, b, nmax) for b in v] for a in u]


def _eps(v):
    """[eps_jkl v_l]"""
    return [[padd(*[(levi(j, k, l), v[l]) for l in range(3) if levi(j, k, l)]) for k in range(3)] for j in range(3)]


_DELTA = [[ONE if i == j else ZERO for j in range(3)] for i in range(3)]

_ZP = {}


def _zpow(p):
    """lz^p with lz = (1 + y1^2 - y2^2)/2 (s = 1), as {(a, b): coefficient} of y1^a y2^b."""
    if p not in _ZP:
        if p == 0:
            _ZP[p] = {(0, 0): Fraction(1)}
        else:
            out = {}
            for (a, b), v in _zpow(p - 1).items():
                for (a2, b2), v2 in (((0, 0), HALF), ((2, 0), HALF), ((0, 2), -HALF)):
                    out[(a + a2, b + b2)] = out.get((a + a2, b + b2), 0) + v * v2
            _ZP[p] = out
    return _ZP[p]


def _dfact(n):
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def _binom(n, k):
    out = Fraction(1)
    for i in range(k):
        out = out * (n - i) / (i + 1)
    return out


def reduce_poly(p):
    """Azimuthal average around shat and reduction to monomials in y1, y2.

    lx = rho cos(phi), ly = rho sin(phi), rho^2 = y1^2 - lz^2, lz = (1 + y1^2 - y2^2)/2, S^2 = 1 - c^2.
    Returns {(rA, rB, y1, y2, S1, c1, S3, c3, E, i): coefficient} with S1, S3 exponents 0 or 1.
    """
    out = {}
    for key, v in p.items():
        mx, my, mz = key[LX], key[LY], key[LZ]
        if mx % 2 or my % 2:
            continue
        k = (mx + my) // 2
        f = v * Fraction(_dfact(mx - 1) * _dfact(my - 1), _dfact(mx + my))
        s1, c1, s3, c3 = key[S1], key[C1], key[S3], key[C3]
        for j in range(k + 1):                       # rho^(2k) = sum_j C(k, j) y1^(2k - 2j) (-lz^2)^j
            fj = f * _binom(k, j) * (-1) ** j
            for (a, b), zv in _zpow(2 * j + mz).items():
                for i1 in range(s1 // 2 + 1):          # S1^(2q) = (1 - c1^2)^q
                    f1 = fj * zv * _binom(s1 // 2, i1) * (-1) ** i1
                    for i3 in range(s3 // 2 + 1):
                        f3 = f1 * _binom(s3 // 2, i3) * (-1) ** i3
                        kk = (key[RA], key[RB], key[Y1] + 2 * (k - j) + a, key[Y2] + b,
                              s1 % 2, c1 + 2 * i1, s3 % 2, c3 + 2 * i3, key[EB], key[IB])
                        out[kk] = out.get(kk, 0) + f3
    return {k: v for k, v in out.items() if v != 0}


# ------------------------------------------------------------------------------ geometry
# tensor types of a side (first index at V1 or V3, second at V2 or V4) and their index types
SIDE_TYPES = {("s", "s"): ("d", "PP", "eP"), ("s", "0"): ("Ps",), ("0", "s"): ("sP",), ("0", "0"): ("1",)}
CONTACT_TYPES = {("s", "s"): ("d",), ("0", "0"): ("1",)}
# tensor types of a soft line (first index at its A end, second at its B end)
LINE_TYPES = {("s", "s"): ("E", "O", "nn"), ("s", "0"): ("nF",), ("0", "s"): ("nS",), ("0", "0"): ("1",)}
LINE_EXT = {"E": 0, "O": 0, "nn": 0, "nF": 1, "nS": 1, "1": 2}       # explicit powers of y from A_0 ends


class Geometry:
    """Contractions (side A) x (side B) x (line 1) x (line 2), times u_A^jA u_B^jB, azimuthally averaged.

    table[(tA, jA, tB, jB, t1, t2)] = {(rA, rB, y1, y2, S1, c1, S3, c3, E, i): Fraction}.
    Side tensors: 'd' delta_jk, 'PP' P_j P_k, 'eP' eps_jkl P_l, 'Ps' P_j, 'sP' P_k, '1' (scalar), with
    P_A = k1hat - rA l and P_B = k3hat + rB l (l in units of s). u_A = |P_A|^2 - 1, u_B = |P_B|^2 - 1.
    Line tensors: 'E' (delta - n n)/2, 'O' -(i/2) eps_jml n_l (so P^h = E + h O), 'nn' n_j n_m, 'nF' n_j, 'nS' n_m,
    '1', with n = l/y1 for line 1 and (s - l)/y2 for line 2.
    """

    def __init__(self, nmax, cache_dir=None, verbose=False):
        self.nmax = nmax
        self.table = None
        path = os.path.join(cache_dir, f"box_geometry_N{nmax}.pkl") if cache_dir else None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.table = pickle.load(f)
        else:
            self.table = self._build(verbose)
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(self.table, f)

    def _build(self, verbose):
        nmax = self.nmax
        L = [mono(lx=1), mono(ly=1), mono(lz=1)]
        W = [mono(-1, lx=1), mono(-1, ly=1), padd((1, ONE), (-1, mono(lz=1)))]
        N1 = [pmul(c, mono(y1=-1)) for c in L]
        N2 = [pmul(c, mono(y2=-1)) for c in W]
        K1 = [mono(s1=1), ZERO, mono(c1=1)]
        K3 = [padd((HALF, mono(s3=1, e=1)), (HALF, mono(s3=1, e=-1))),
              padd((-HALF, mono(s3=1, e=1, ib=1)), (HALF, mono(s3=1, e=-1, ib=1))),
              mono(c3=1)]
        PA = [padd((1, K1[i]), (-1, pmul(mono(ra=1), L[i]))) for i in range(3)]
        PB = [padd((1, K3[i]), (1, pmul(mono(rb=1), L[i]))) for i in range(3)]
        uA = padd((-2, pmul(mono(ra=1), _dot(L, K1))), (1, mono(ra=2, y1=2)))
        uB = padd((2, pmul(mono(rb=1), _dot(L, K3))), (1, mono(rb=2, y1=2)))

        def side(P):
            return {"d": _DELTA, "PP": _outer(P, P, nmax), "eP": _eps(P), "Ps": [[P[j]] for j in range(3)],
                    "sP": [[P[k] for k in range(3)]], "1": [[ONE]]}

        def line(N):
            nn = _outer(N, N)
            E = [[padd((HALF if i == j else 0, ONE), (-HALF, nn[i][j])) for j in range(3)] for i in range(3)]
            eps = _eps(N)
            O = [[pmul(mono(-HALF, ib=1), eps[i][j]) for j in range(3)] for i in range(3)]
            return {"E": E, "O": O, "nn": nn, "nF": [[N[j]] for j in range(3)], "nS": [[N[m] for m in range(3)]],
                    "1": [[ONE]]}

        TA, TB, T1, T2 = side(PA), side(PB), line(N1), line(N2)
        upA, upB = [ONE], [ONE]
        for _ in range(nmax):
            upA.append(pmul(upA[-1], uA, nmax))
            upB.append(pmul(upB[-1], uB, nmax))
        table = {}
        for A1, A2, B1, B2 in product("s0", repeat=4):
            ntemp = (A1 == "0") + (A2 == "0") + (B1 == "0") + (B2 == "0")
            if ntemp > nmax:
                continue
            nrem = nmax - ntemp
            for tA in SIDE_TYPES[(A1, A2)]:
                for tB in SIDE_TYPES[(B1, B2)]:
                    for t1 in LINE_TYPES[(A1, B1)]:
                        for t2 in LINE_TYPES[(A2, B2)]:
                            a, b, l1, l2 = TA[tA], TB[tB], T1[t1], T2[t2]
                            # sum_{jkmn} a[j][k] b[m][n] l1[j][m] l2[k][n] = sum_{j n} (a l2)[j][n] (l1 b)[j][n]
                            dj, dk, dm, dn = len(a), len(a[0]), len(b), len(b[0])
                            al2 = [[padd(*[(1, pmul(a[j][k], l2[k][n], nrem)) for k in range(dk)]) for n in range(dn)]
                                   for j in range(dj)]
                            l1b = [[padd(*[(1, pmul(l1[j][m], b[m][n], nrem)) for m in range(dm)]) for n in range(dn)]
                                   for j in range(dj)]
                            con = padd(*[(1, pmul(al2[j][n], l1b[j][n], nrem)) for j in range(dj) for n in range(dn)])
                            if not con:
                                continue
                            for jA in range(nrem + 1):
                                cA = pmul(con, upA[jA], nrem) if jA else con
                                if not cA:
                                    continue
                                for jB in range(nrem - jA + 1):
                                    full = pmul(cA, upB[jB], nrem) if jB else cA
                                    red = reduce_poly(full)
                                    if red:
                                        table[(tA, jA, tB, jB, t1, t2)] = red
                            if verbose:
                                print("  geometry", tA, tB, t1, t2, flush=True)
        return table


# ------------------------------------------------------------------------------ series helpers
def _binom_series(beta, jmax):
    """(1 + u)^beta = sum_j c_j u^j"""
    c = [mp.mpc(1)]
    for j in range(1, jmax + 1):
        c.append(c[-1] * (beta - j + 1) / j)
    return c


def _ser2_mul(a, b, N):
    """product of double series {(j, l): c} in (u, xi), truncated at j + l <= N"""
    out = {}
    for (j1, l1), v1 in a.items():
        for (j2, l2), v2 in b.items():
            if j1 + l1 + j2 + l2 <= N:
                k = (j1 + j2, l1 + l2)
                out[k] = out.get(k, 0) + v1 * v2
    return out


_WW = {}


def _ww_series(N):
    """w1^p1 w2^p2 as double series in (u, xi): w1 = 1 - 1/Y, w2 = 1 - (1 + xi)/Y, Y = (1 + u)^(1/2)."""
    key = (N, mp.mp.dps)
    if key not in _WW:
        isq = {(j, 0): cj for j, cj in enumerate(_binom_series(-mp.mpf(1) / 2, N))}
        w1 = {k: -v for k, v in isq.items() if k != (0, 0)}
        w2 = dict(w1)
        for (j, l), v in isq.items():
            if j + 1 <= N:
                w2[(j, 1)] = w2.get((j, 1), 0) - v
        one = {(0, 0): mp.mpc(1)}
        p1s, p2s = [one], [one]
        for _ in range(N):
            p1s.append(_ser2_mul(p1s[-1], w1, N))
            p2s.append(_ser2_mul(p2s[-1], w2, N))
        _WW[key] = {(a, b): _ser2_mul(p1s[a], p2s[b], N) for a in range(N + 1) for b in range(N + 1 - a)}
    return _WW[key]


def xi_powers(nmax, sign):
    """xi^l for xi = sqrt(1 + 2 sign r c + r^2) - 1, as [{(n, mc): Fraction}] (r^n c^mc), l = 0..nmax."""
    def mul(p, q):
        out = {}
        for (n1, m1), v1 in p.items():
            for (n2, m2), v2 in q.items():
                if n1 + n2 <= nmax:
                    out[(n1 + n2, m1 + m2)] = out.get((n1 + n2, m1 + m2), 0) + v1 * v2
        return out
    v = {(1, 1): Fraction(2 * sign), (2, 0): Fraction(1)}
    vp, xi = {(0, 0): Fraction(1)}, {}
    cj = Fraction(1)
    for j in range(1, nmax + 1):
        cj = cj * (HALF - j + 1) / j
        vp = mul(vp, v)
        for k, val in vp.items():
            xi[k] = xi.get(k, 0) + cj * val
    xl = [{(0, 0): Fraction(1)}]
    for _ in range(nmax):
        xl.append(mul(xl[-1], xi))
    return xl


# ------------------------------------------------------------------------------ the hard pair
class HardPair:
    """Two-time moments of a hard line between two cubic vertices, as double series in u = Y^2 - 1 and xi.

    base:    int dt1 dt2 e^(-t1 - t2) t1^x1 t2^x2 G^lam_{a1 a2}(t1, t2)  at Y = 1, lam = +1, -1 (transverse), 0
    moment:  the same with D^d1 D^d2 on the line functions (by parts; equal-time Wronskian term for d1 = d2 = 1)
    series2: Y^mY Q(x1, x2; Y, xi) = sum_{j l} F_{jl} u^j xi^l, with vertex energies 1 and 1 + xi and
             Q = Y^(-x1 - x2 - 3) sum_{p1 p2} w1^p1 w2^p2/(p1! p2!) Q(x1 + p1, x2 + p2; 1)   [eq:Dfold].
    """

    def __init__(self, mu_h, kappa_h):
        self.lines = {h: VectorLine(mu_h, kappa_h, h) for h in (1, -1, 0)}
        self.m2 = mp.mpf(mu_h) ** 2 + mp.mpf(1) / 4
        self._base, self._mom, self._ser = {}, {}, {}

    def base(self, lam, a1, a2, x1, x2):
        key = (lam, a1, a2, mp.nstr(x1, 20), mp.nstr(x2, 20))
        if key not in self._base:
            L = self.lines[lam]
            if a1 == a2:
                v = L._Qsym(x1, x2, conj=(a1 == -1))
            elif a1 == 1:
                v = L._P(x1) * L._P(x2, conj=True)
            else:
                v = L._P(x1, conj=True) * L._P(x2)
            self._base[key] = v
        return self._base[key]

    def moment(self, lam, a1, a2, x1, x2, d1, d2):
        key = (lam, a1, a2, mp.nstr(x1, 20), mp.nstr(x2, 20), d1, d2)
        if key not in self._mom:
            tot = 0
            for i in range(d1 + 1):
                ci = 1 if (d1 == 0 or i == 1) else -(x1 + 2)
                for j in range(d2 + 1):
                    cj = 1 if (d2 == 0 or j == 1) else -(x2 + 2)
                    tot += ci * cj * self.base(lam, a1, a2, x1 + i, x2 + j)
            if d1 and d2 and a1 == a2:
                Wr = self.lines[lam]._wronskian(conj=(a1 == -1))
                tot += Wr * mp.gamma(x1 + x2 + 3) / mp.mpf(2) ** (x1 + x2 + 3)
            self._mom[key] = tot
        return self._mom[key]

    def series2(self, lam, a1, a2, x1, x2, d1, d2, mY, N):
        key = (lam, a1, a2, mp.nstr(x1, 20), mp.nstr(x2, 20), d1, d2, mY, N)
        if key in self._ser:
            return self._ser[key]
        ww = _ww_series(N)
        tot = {}
        for p1 in range(N + 1):
            for p2 in range(N + 1 - p1):
                q = self.moment(lam, a1, a2, x1 + p1, x2 + p2, d1, d2) / (mp.factorial(p1) * mp.factorial(p2))
                for k, v in ww[(p1, p2)].items():
                    tot[k] = tot.get(k, 0) + q * v
        pre = {(j, 0): cj for j, cj in enumerate(_binom_series((mY - x1 - x2 - 3) / 2, N))}
        out = _ser2_mul(pre, tot, N)
        self._ser[key] = out
        return out

    def phi(self, t, a1, a2, x1, x2, N):
        """Coefficient series {(j, l): F} of the side tensor of type t (the eta signs of temporal contractions
        included). x1, x2: time exponents of the soft-line ends at the two vertices."""
        m2 = self.m2

        def comb(*terms):
            out = {}
            for c, s in terms:
                for k, v in s.items():
                    out[k] = out.get(k, 0) + c * v
            return out
        if t == "d":
            return comb(*[(mp.mpf(1) / 2, self.series2(lam, a1, a2, x1, x2, 0, 0, 0, N)) for lam in (1, -1)])
        if t == "PP":
            return comb(*([(-mp.mpf(1) / 2, self.series2(lam, a1, a2, x1, x2, 0, 0, -2, N)) for lam in (1, -1)]
                          + [(1 / m2, self.series2(0, a1, a2, x1, x2, 1, 1, -2, N))]))
        if t == "eP":
            return comb(*[(-I * lam / 2, self.series2(lam, a1, a2, x1, x2, 0, 0, -1, N)) for lam in (1, -1)])
        if t == "Ps":        # (s, 0): (i P_j/m) D_1 c_0(2); eta^00 = -1
            return comb((-a2 / m2, self.series2(0, a1, a2, x1, x2 + 1, 1, 0, 0, N)))
        if t == "sP":        # (0, s): -(i P_k/m) c_0(1) D_2; eta^00 = -1
            return comb((a1 / m2, self.series2(0, a1, a2, x1 + 1, x2, 0, 1, 0, N)))
        if t == "1":         # (0, 0): c_0 c_0 G plus the contact term; two temporal contractions
            out = comb((-a1 * a2 / m2, self.series2(0, a1, a2, x1 + 1, x2 + 1, 0, 0, 2, N)))
            if a1 == a2:
                for k, v in contact_series(x1, x2, N).items():
                    out[k] = out.get(k, 0) + v / m2
            return out
        raise ValueError(t)


def contact_series(x1, x2, N):
    """int dt t^2 e^(-(2 + xi) t) t^(x1 + x2) = Gamma(X) (2 + xi)^(-X), X = x1 + x2 + 3, as {(0, l): c}."""
    X = x1 + x2 + 3
    g = mp.gamma(X)
    return {(0, l): g / mp.mpf(2) ** (X + l) * cl for l, cl in enumerate(_binom_series(-X, N))}


def contact_phi(t, a1, a2, x1, x2, N):
    """Side tensor of the quartic vertex: 'd' delta_jk, '1' the temporal contraction (eta^00 = -1)."""
    if a1 != a2:
        return {}
    s = contact_series(x1, x2, N)
    return s if t == "d" else {k: -v for k, v in s.items()}


# ------------------------------------------------------------------------------ assembly
BR = ((1, 1), (1, -1), (-1, 1), (-1, -1))
# the nine families, labelled by the powers (eA, eB) of nu on the two sides, with their line assignments
# ((sig1, sig1'), (sig2, sig2'))
FAMILIES = {
    "pair": [((1, 1), (1, 1))],                                    # (s/p1)^(2 nu) (s/p3)^(2 nu)
    "singleA": [((1, 1), (1, -1)), ((1, -1), (1, 1))],              # (s/p1)^(2 nu)
    "singleB": [((1, 1), (-1, 1)), ((-1, 1), (1, 1))],              # (s/p3)^(2 nu)
    "ratio": [((1, -1), (1, -1))],                                  # (p3/p1)^(2 nu)
    "ratio_c": [((-1, 1), (-1, 1))],                                # (p1/p3)^(2 nu), conjugate to "ratio"
    "zero": [((1, 1), (-1, -1)), ((-1, -1), (1, 1)), ((1, -1), (-1, 1)), ((-1, 1), (1, -1))],
}


class CollapsedSeries:
    """Coefficients C_{NA NB}(c1, c3, beta) of the collapsed limit for one diagram class and one family.

    mu, kappa: the soft lines. mu_h, kappa_h: the hard lines (default: the same). nmax: highest NA + NB.
    """

    def __init__(self, mu, kappa, mu_h=None, kappa_h=None, nmax=0, geometry=None, cache_dir=None, eps=0):
        self.mu, self.kappa = mp.mpf(mu), mp.mpf(kappa)
        self.mu_h = self.mu if mu_h is None else mp.mpf(mu_h)
        self.kappa_h = self.kappa if kappa_h is None else mp.mpf(kappa_h)
        self.nu = I * self.mu
        self.nmax = nmax
        self.m2 = self.mu ** 2 + mp.mpf(1) / 4
        self.lines = {h: VectorLine(self.mu, self.kappa, h) for h in (1, -1, 0)}
        self.hard = HardPair(self.mu_h, self.kappa_h)
        self.geo = geometry or Geometry(nmax, cache_dir)
        self.eps = mp.mpf(eps)                         # analytic regulator y1^(2 eps) (integer-power families only)
        self._e, self._phi, self._blk = {}, {}, {}
        self._xiA = xi_powers(nmax, -1)
        self._xiB = xi_powers(nmax, +1)

    # ---- soft lines
    def g(self, h, a, b, sig, sigp):
        """Coefficient of (2y)^((sig + sig') nu) t_a^(1/2 + sig nu) t_b^(1/2 + sig' nu) in G_ab of helicity h."""
        A, B = self.lines[h].A, self.lines[h].B
        if a == b:
            w = A[sig] * B[sig] if sig == sigp else (A[1] * B[-1] + A[-1] * B[1]) / 2
            if a == -1:
                w = mp.conj(A[-sig] * B[-sig]) if sig == sigp else mp.conj(w)
            return w
        if a == 1:
            return B[sig] * mp.conj(B[-sigp])
        return mp.conj(B[-sig]) * B[sigp]

    def e(self, h, a, sig):
        key = (h, a, sig)
        if key not in self._e:
            self._e[key] = e_series(-a * self.lines[h].K, sig * self.nu, self.nmax + 1)
        return self._e[key]

    def line_table(self, sig, sigp):
        """{type: {(n, m): {(a, b): coefficient}}}: the factor of y^((sig + sig') nu + n + m + ext)
        t_A^(1/2 + sig nu + n + [A end temporal]) t_B^(1/2 + sig' nu + m + [B end temporal])."""
        nu, m2, N = self.nu, self.m2, self.nmax
        tab = {t: {} for t in LINE_EXT}
        for n in range(N + 1):
            for m in range(N + 1 - n):
                two = mp.mpf(2) ** ((sig + sigp) * nu + n + m)
                DA, DB = sig * nu + n - mp.mpf(1) / 2, sigp * nu + m - mp.mpf(1) / 2
                ent = {t: {} for t in LINE_EXT}
                for a, b in BR:
                    v = {h: self.g(h, a, b, sig, sigp) * self.e(h, a, sig)[n] * self.e(h, b, sigp)[m] for h in (1, -1, 0)}
                    ent["E"][(a, b)] = two * (v[1] + v[-1])
                    ent["O"][(a, b)] = two * (v[1] - v[-1])
                    v0 = two * v[0] / m2
                    ent["nn"][(a, b)] = v0 * DA * DB
                    ent["nF"][(a, b)] = v0 * DA * b
                    ent["nS"][(a, b)] = -v0 * a * DB
                    ent["1"][(a, b)] = -v0 * a * b
                for t in LINE_EXT:
                    tab[t][(n, m)] = ent[t]
        return tab

    # ---- sides
    def phi(self, kind, t, sf, ss, a1, a2, i1, i2, N):
        """Side series {(j, l): F} for soft-line ends t^(1/2 + sf nu + i1), t^(1/2 + ss nu + i2)."""
        key = (kind, t, sf, ss, a1, a2, i1, i2)
        got = self._phi.get(key)
        if got is None or got[0] < N:
            x1 = mp.mpf(1) / 2 + sf * self.nu + i1
            x2 = mp.mpf(1) / 2 + ss * self.nu + i2
            ser = self.hard.phi(t, a1, a2, x1, x2, N) if kind == "pair" else contact_phi(t, a1, a2, x1, x2, N)
            self._phi[key] = got = (N, ser)
        return got[1]

    def block(self, P1, P2, k1, k2):
        """int d^3l/(2 pi)^3 y1^(P1 + k1) y2^(P2 + k2), s = 1 (k1, k2 integers)."""
        key = (mp.nstr(P1, 15), mp.nstr(P2, 15), k1, k2)
        if key not in self._blk:
            self._blk[key] = C2hat(P1 + k1 + 1, P2 + k2 + 1) / (2 * mp.pi) ** 2
        return self._blk[key]

    # ---- one line assignment of one family
    def assignment(self, lines, diagram, acc=None, verbose=False):
        """Add the terms of the line assignment ((sig1, sig1'), (sig2, sig2')) to acc.

        diagram: 'PP' (box, planar attachment), 'CP' (contact on side A, pair on B), 'PC', or 'CC'.
        acc: {(NA, NB): {(S1, c1, S3, c3, n): coefficient}} with n the harmonic of e^(i n beta).
        """
        (s1, s1p), (s2, s2p) = lines
        acc = {} if acc is None else acc
        N, nu = self.nmax, self.nu
        T1, T2 = self.line_table(s1, s1p), self.line_table(s2, s2p)
        P1, P2 = (s1 + s1p) * nu + 2 * self.eps, (s2 + s2p) * nu
        kindA = "pair" if diagram[0] == "P" else "contact"
        kindB = "pair" if diagram[1] == "P" else "contact"
        typesA = SIDE_TYPES if kindA == "pair" else CONTACT_TYPES
        typesB = SIDE_TYPES if kindB == "pair" else CONTACT_TYPES
        ipow = (mp.mpc(1), I)
        for A1, A2, B1, B2 in product("s0", repeat=4):
            tA_, tB_ = (A1 == "0") + (A2 == "0"), (B1 == "0") + (B2 == "0")
            if tA_ + tB_ > N or (A1, A2) not in typesA or (B1, B2) not in typesB:
                continue
            for tA, tB, t1, t2 in product(typesA[(A1, A2)], typesB[(B1, B2)], LINE_TYPES[(A1, B1)], LINE_TYPES[(A2, B2)]):
                jmaxA = N if kindA == "pair" else 0
                jmaxB = N if kindB == "pair" else 0
                geos = {(jA, jB): self.geo.table.get((tA, jA, tB, jB, t1, t2))
                        for jA in range(jmaxA + 1) for jB in range(jmaxB + 1)}
                geos = {k: v for k, v in geos.items() if v}
                if not geos:
                    continue
                ext1, ext2 = LINE_EXT[t1], LINE_EXT[t2]
                intgeo = {}
                for n1, m1, n2, m2 in product(range(N + 1), repeat=4):
                    oa, ob = n1 + n2 + tA_, m1 + m2 + tB_
                    R = N - oa - ob
                    if R < 0:
                        continue
                    L1, L2 = T1[t1][(n1, m1)], T2[t2][(n2, m2)]
                    e1, e2 = n1 + m1 + ext1, n2 + m2 + ext2
                    iA1, iA2 = n1 + (A1 == "0"), n2 + (A2 == "0")
                    iB1, iB2 = m1 + (B1 == "0"), m2 + (B2 == "0")
                    phA = {(a1, a2): self.phi(kindA, tA, s1, s2, a1, a2, iA1, iA2, N - iA1 - iA2) for a1, a2 in BR}
                    phB = {(b1, b2): self.phi(kindB, tB, s1p, s2p, b1, b2, iB1, iB2, N - iB1 - iB2) for b1, b2 in BR}
                    keysA = sorted({k for s in phA.values() for k in s if k[0] + k[1] <= R})
                    keysB = sorted({k for s in phB.values() for k in s if k[0] + k[1] <= R})
                    for (jA, lA) in keysA:
                        # X[b1, b2] = sum_{a1 a2} L1[a1, b1] L2[a2, b2] phiA[a1, a2]
                        X = {}
                        for b1, b2 in BR:
                            X[(b1, b2)] = mp.fsum(L1[(a1, b1)] * L2[(a2, b2)] * phA[(a1, a2)].get((jA, lA), 0)
                                                  for a1, a2 in BR)
                        for (jB, lB) in keysB:
                            if jA + lA + jB + lB > R or (jA, jB) not in geos:
                                continue
                            Wt = mp.fsum(X[bb] * phB[bb].get((jB, lB), 0) for bb in BR)
                            if Wt == 0:
                                continue
                            gk = (jA, jB, e1, e2)
                            if gk not in intgeo:        # geometry integrated against the loop block
                                ig = {}
                                for (ra, rb, k1, k2, S1_, c1_, S3_, c3_, E_, ib), cf in geos[(jA, jB)].items():
                                    kk = (ra, rb, S1_, c1_, S3_, c3_, E_)
                                    val = (mp.mpf(cf.numerator) / cf.denominator) * ipow[ib] * self.block(P1, P2, e1 + k1, e2 + k2)
                                    ig[kk] = ig.get(kk, 0) + val
                                intgeo[gk] = ig
                            xa, xb = self._xiA[lA], self._xiB[lB]
                            for (ra, rb, S1_, c1_, S3_, c3_, E_), gv in intgeo[gk].items():
                                base = Wt * gv
                                for (na, mca), va in xa.items():
                                    NA = oa + ra + na
                                    if NA + ob + rb > N:
                                        continue
                                    for (nb, mcb), vb in xb.items():
                                        NB = ob + rb + nb
                                        if NA + NB > N:
                                            continue
                                        d = acc.setdefault((NA, NB), {})
                                        kk = (S1_, c1_ + mca, S3_, c3_ + mcb, E_)
                                        d[kk] = d.get(kk, 0) + base * (mp.mpf(va.numerator) / va.denominator) \
                                            * (mp.mpf(vb.numerator) / vb.denominator)
                if verbose:
                    print("   ", diagram, lines, (A1, A2, B1, B2), tA, tB, t1, t2, flush=True)
        return acc

    def family(self, name, diagram, verbose=False):
        acc = {}
        for lines in FAMILIES[name]:
            self.assignment(lines, diagram, acc, verbose)
        return acc


# ------------------------------------------------------------------------------ evaluation
def harmonics(coefs, c1, c3, rA=None, rB=None, orders=None):
    """{n: value} of sum over (NA, NB) of rA^NA rB^NB C_{NA NB} at the angles (c1, c3). With rA = rB = None the
    result is {(NA, NB): {n: value}}; orders restricts to NA + NB in the given set."""
    c1, c3 = mp.mpf(c1), mp.mpf(c3)
    S1v, S3v = mp.sqrt(1 - c1 ** 2), mp.sqrt(1 - c3 ** 2)
    per = {}
    for (NA, NB), poly in coefs.items():
        if orders is not None and NA + NB not in orders:
            continue
        h = {}
        for (a, b, c, d, n), v in poly.items():
            h[n] = h.get(n, 0) + v * S1v ** a * c1 ** b * S3v ** c * c3 ** d
        per[(NA, NB)] = h
    if rA is None:
        return per
    out = {}
    for (NA, NB), h in per.items():
        for n, v in h.items():
            out[n] = out.get(n, 0) + v * mp.mpf(rA) ** NA * mp.mpf(rB) ** NB
    return out


def evaluate(coefs, c1, c3, beta, rA, rB, orders=None):
    """sum_n c_n e^(i n beta) at the given point (the series part only, without the overall powers)."""
    return mp.fsum(v * mp.exp(I * n * beta) for n, v in harmonics(coefs, c1, c3, rA, rB, orders).items())
