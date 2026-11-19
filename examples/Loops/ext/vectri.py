"""The helical-vector triangle in the squeezed limit, order by order in r = k3/k1 (Sec. 7.3; App. D).

Model of Wang and Xianyu: three cubic vertices (lambda_2/2) a phi' eta^{mu nu} A_mu A_nu joined by gauge-boson
lines with an F Ftilde chemical potential. Vertices V1, V2 (hard legs k1, k2, |k1| = |k2| = 1) and V3 (soft
leg, |k3| = r). Line C joins V3 and V1, line B joins V3 and V2, and the hard line A joins V1 and V2.

Region. The non-analytic dependence on k3 comes from loop momenta q = r q' with the soft-vertex time
t3 = T/r. There every Schwinger-Keldysh function of a soft line is (soft end at V3) x (hard end at V1 or V2),
and the hard end is expanded on its late-time branches. The pair family keeps the branch +nu on both lines:

    hard end of C on branch a1 (V3 on branch b):
        h_C(a1, b) (2 y)^(-1/2) sum_n e_n(-a1 K_C, nu) (2 y t1)^(1/2 + nu + n),          [eq:vecAB], [eq:en]
    soft end:  (2 y)^(-1/2) Nf W_{-b K_C, nu}(2 y t3),

so the powers of r advance in unit steps (e_1 = -K/(1 + 2 nu)); at kappa = 0 the odd orders vanish. The
hard line carries P = k1 + r q', with energy Y = |P|. Its two-time moments at energy Y follow from those at
the folded point Y = 1 by scaling [eq:Dfold], and its projectors are expanded in r with it. After the
azimuthal average around k3 the soft integrand is a Laurent polynomial in y1 = |q'|, y2 = |q' - k3hat|, and
each monomial is one position-space soft moment (ext/softvertex.py).

Components. A longitudinal line has the components A_0 and A_L at each end. With the neutral function g and
D = t d_t - 1, c_0 = -(y/m) tau, the propagator between end 1 (momentum direction n) and end 2 is

    (s, s): n_i n_j D_1 D_2 G/m^2,      (0, 0): c_0(1) c_0(2) G,
    (s, 0): (i n_i/m) D_1 c_0(2) G,     (0, s): -(i n_j/m) c_0(1) D_2 G,

and tau = +-i t on the two branches. The vertex contracts with eta^{mu nu} = diag(-1, 1, 1, 1). The
time-ordered function of A_0 A_0 has in addition the contact term of the Proca field, i delta(tau - tau')/M^2
with M^2 = m^2/tau^2 (the covariant propagator differs from the ordered product of the modes by it). On the
rotated contour it is +(t^2/m^2) delta(t - t'), opposite to the heavy-mass limit -(t^2/m^2) delta(t - t') of the
spatial components. It matters only on the hard line, where it first contributes at order r^2; with it each
order of the triangle tends to twice the contracted bubble when the hard line is heavy.

Normalization. With unit couplings,

    S = (k1 k2 k3)^2 <phi phi phi>'  contains  2 Re sum_n M_n r^(2 + 2 nu + n),
    M_n(triangle) = (1/8) [r^n] sum_{a1 a2 b} sum_{components} (...),
    M_n(bubble)   = (1/16) [r^n] (...),          M_0(bubble) = M_A of Eq. [eq:vecMn],

as in the hard-line calculation of Sec. 7.3. In the normalization of Wang and Xianyu the triangle is
-m_h^2 e^(-2 pi Omega) M_n(triangle), Omega = max(kappa - mu, 0), and R_hard is its ratio to M_A.

Polynomials are dictionaries {(n, mx, my, mz, ms, mc, da, db): coefficient} for the monomial
r^n qx^mx qy^my qz^mz S^ms c^mc y1^da y2^db, with k3hat = z, k1hat = (S, 0, c), S^2 = 1 - c^2. On the
isosceles family c = -r/2. With general=True the energy of V2, |k2| = 1 + xi = sqrt(1 + 2 r c + r^2), is expanded
as well, and the coefficients M_n(c) hold at general shape (S is then symmetric under k1 <-> k2 order by order).
"""
from itertools import product

import mpmath as mp

from .blocks import e_series
from .helical import VectorLine
from .blocks import C2hat
from .softvertex import B_sigma, Grid, Leg, soft_J

I = mp.mpc(0, 1)
NV = 8                                   # number of exponents in a monomial key

__all__ = ["Triangle", "SoftTable", "reduce_poly", "pmul", "padd", "mono", "xi_powers"]


# ------------------------------------------------------------------------------ polynomial algebra
def mono(c=1, n=0, mx=0, my=0, mz=0, ms=0, mc=0, da=0, db=0):
    return {(n, mx, my, mz, ms, mc, da, db): mp.mpmathify(c)}


def padd(*terms):
    """sum of (coefficient, polynomial) pairs"""
    out = {}
    for c, p in terms:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return {k: v for k, v in out.items() if v != 0}


def pmul(p, q, nmax):
    out = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            if k1[0] + k2[0] > nmax:
                continue
            k = tuple(a + b for a, b in zip(k1, k2))
            out[k] = out.get(k, 0) + v1 * v2
    return {k: v for k, v in out.items() if v != 0}


ONE = mono()
ZERO = {}
R_ = mono(n=1)
QV = [mono(mx=1), mono(my=1), mono(mz=1)]                               # q'
N1 = [mono(mx=1, da=-1), mono(my=1, da=-1), mono(mz=1, da=-1)]            # q'/y1
N2 = [mono(-1, mx=1, db=-1), mono(-1, my=1, db=-1), padd((1, mono(db=-1)), (-1, mono(mz=1, db=-1)))]
K1 = [mono(ms=1), ZERO, mono(mc=1)]                                     # k1hat
PV = [padd((1, K1[i]), (1, mono(n=1, **{("mx", "my", "mz")[i]: 1}))) for i in range(3)]     # P = k1hat + r q'
USER = padd((2, mono(n=1, mx=1, ms=1)), (2, mono(n=1, mz=1, mc=1)), (1, mono(n=2, da=2)))    # u = Y^2 - 1


def levi(i, j, k):
    return (i - j) * (j - k) * (k - i) // 2


def proj_T(h, n, nmax):
    """Pi^h_ij(n) = eps^h_i eps^h*_j = (delta_ij - n_i n_j - i h eps_ijl n_l)/2 for a unit vector n."""
    M = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            terms = [(-mp.mpf(1) / 2, pmul(n[i], n[j], nmax))]
            if i == j:
                terms.append((mp.mpf(1) / 2, ONE))
            for l in range(3):
                e = levi(i, j, l)
                if e:
                    terms.append((-I * h * e / 2, n[l]))
            M[i][j] = padd(*terms)
    return M


def outer(u, v, nmax):
    return [[pmul(a, b, nmax) for b in v] for a in u]


_ZP = {}


def _zpow(p):
    """qz^p with qz = (1 + y1^2 - y2^2)/2, as {(da, db): coefficient}."""
    if p not in _ZP:
        if p == 0:
            _ZP[p] = {(0, 0): mp.mpf(1)}
        else:
            prev, out = _zpow(p - 1), {}
            for (a, b), v in prev.items():
                for (a2, b2), v2 in (((0, 0), mp.mpf(1) / 2), ((2, 0), mp.mpf(1) / 2), ((0, 2), -mp.mpf(1) / 2)):
                    out[(a + a2, b + b2)] = out.get((a + a2, b + b2), 0) + v * v2
            _ZP[p] = out
    return _ZP[p]


def _dfact(n):
    return mp.mpf(1) if n <= 0 else mp.fac2(n)


def reduce_poly(p):
    """Azimuthal average around k3hat and reduction to monomials in y1, y2:
    {(n, mx, my, mz, ms, mc, da, db): v} -> {(n, mc, da, db): v}.

    qx = rho cos(phi), qy = rho sin(phi), rho^2 = y1^2 - qz^2, qz = (1 + y1^2 - y2^2)/2, S^2 = 1 - c^2.
    """
    out = {}
    for (n, mx, my, mz, ms, mc, da, db), v in p.items():
        if mx % 2 or my % 2:
            continue
        if ms % 2:
            raise ValueError("odd power of S survived the azimuthal average")
        k = (mx + my) // 2
        f = v * _dfact(mx - 1) * _dfact(my - 1) / _dfact(mx + my)
        for j in range(k + 1):                      # rho^(2k) = sum_j C(k, j) y1^(2k - 2j) (-qz^2)^j
            fj = f * mp.binomial(k, j) * (-1) ** j
            for (a, b), zv in _zpow(2 * j + mz).items():
                for i in range(ms // 2 + 1):          # S^ms = (1 - c^2)^(ms/2)
                    key = (n, mc + 2 * i, da + 2 * (k - j) + a, db + b)
                    out[key] = out.get(key, 0) + fj * zv * mp.binomial(ms // 2, i) * (-1) ** i
    return {k: v for k, v in out.items() if v != 0}


# ------------------------------------------------------------------------------ series in u = Y^2 - 1
def _binom_series(beta, jmax):
    """(1 + u)^beta = sum_j c_j u^j"""
    c = [mp.mpc(1)]
    for j in range(1, jmax + 1):
        c.append(c[-1] * (beta - j + 1) / j)
    return c


def _smul(a, b, jmax):
    return [mp.fsum(a[i] * b[j - i] for i in range(j + 1)) for j in range(jmax + 1)]


def xi_powers(nmax):
    """xi^l for xi = sqrt(1 + 2 r c + r^2) - 1, as [{(n, mc): coefficient}], l = 0..nmax, truncated at r^nmax."""
    v = padd((2, mono(n=1, mc=1)), (1, mono(n=2)))
    vp, xi = ONE, {}
    for j, cj in enumerate(_binom_series(mp.mpf(1) / 2, nmax)):
        if j:
            vp = pmul(vp, v, nmax)
            xi = padd((1, xi), (cj, vp))
    xl = [ONE]
    for _ in range(nmax):
        xl.append(pmul(xl[-1], xi, nmax))
    return [{(k[0], k[5]): val for k, val in p.items()} for p in xl]


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
    """w1^p1 w2^p2 as double series, w1 = 1 - 1/Y, w2 = 1 - (1 + xi)/Y, Y = (1 + u)^(1/2)."""
    if N not in _WW:
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
        _WW[N] = {(a, b): _ser2_mul(p1s[a], p2s[b], N) for a in range(N + 1) for b in range(N + 1 - a)}
    return _WW[N]


# ------------------------------------------------------------------------------ the hard line
class HardLine:
    """Two-time moments of the hard line at energy Y and vertex energies 1, as series in u = Y^2 - 1."""

    def __init__(self, mu_h, kappa_h):
        self.lines = {h: VectorLine(mu_h, kappa_h, h) for h in (1, -1, 0)}
        self.m2 = mp.mpf(mu_h) ** 2 + mp.mpf(1) / 4
        self._base, self._ser = {}, {}

    def base(self, lam, a1, a2, x1, x2):
        """int dt1 dt2 e^(-t1-t2) t1^x1 t2^x2 G^lam_{a1 a2}(t1, t2) at Y = 1."""
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
        """The same with D_1^d1 D_2^d2 acting on the line functions (by parts; D = t d_t - 1)."""
        tot = 0
        for i in range(d1 + 1):
            ci = 1 if (d1 == 0 or i == 1) else -(x1 + 2)
            for j in range(d2 + 1):
                cj = 1 if (d2 == 0 or j == 1) else -(x2 + 2)
                tot += ci * cj * self.base(lam, a1, a2, x1 + i, x2 + j)
        if d1 and d2 and a1 == a2:
            W = self.lines[lam]._wronskian(conj=(a1 == -1))
            tot += W * mp.gamma(x1 + x2 + 3) / mp.mpf(2) ** (x1 + x2 + 3)
        return tot

    def series(self, lam, a1, a2, x1, x2, d1, d2, mY, jmax):
        """Coefficients F_j of Y^mY Q(x1, x2; Y) = sum_j F_j u^j, with
        Q(x1, x2; Y) = Y^(-x1-x2-3) sum_p (1 - 1/Y)^p/p! sum_{p1+p2=p} C(p, p1) Q(x1 + p1, x2 + p2; 1)."""
        key = (lam, a1, a2, mp.nstr(x1, 20), mp.nstr(x2, 20), d1, d2, mY, jmax)
        if key in self._ser:
            return self._ser[key]
        w = [-c for c in _binom_series(-mp.mpf(1) / 2, jmax)]
        w[0] = mp.mpc(0)                                      # w = 1 - (1 + u)^(-1/2)
        pre = _binom_series((mY - x1 - x2 - 3) / 2, jmax)
        tot = [mp.mpc(0)] * (jmax + 1)
        wp = [mp.mpc(1)] + [mp.mpc(0)] * jmax
        for p in range(jmax + 1):
            Qp = mp.fsum(mp.binomial(p, p1) * self.moment(lam, a1, a2, x1 + p1, x2 + p - p1, d1, d2)
                         for p1 in range(p + 1)) / mp.factorial(p)
            tot = [t + Qp * c for t, c in zip(tot, wp)]
            wp = _smul(wp, w, jmax)
        out = _smul(pre, tot, jmax)
        self._ser[key] = out
        return out


    def series2(self, lam, a1, a2, x1, x2, d1, d2, mY, N):
        """{(j, l): F} with Y^mY Q(x1, x2; Y, xi) = sum F u^j xi^l (j + l <= N), for the energy 1 + xi at V2:
        Q = Y^(-x1-x2-3) sum_{p1 p2} w1^p1 w2^p2/(p1! p2!) Q(x1 + p1, x2 + p2; 1)."""
        key = ("2", lam, a1, a2, mp.nstr(x1, 20), mp.nstr(x2, 20), d1, d2, mY, N)
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


# ------------------------------------------------------------------------------ soft lines
class SoftLine:
    """One soft line of type h (+1, -1 transverse; 0 longitudinal)."""

    def __init__(self, mu, kappa, h, nmax):
        self.h = h
        L = self.L = VectorLine(mu, kappa, h)
        self.nu, self.K, self.Nf = L.nu, L.K, L.Nf
        self.m2 = mp.mpf(mu) ** 2 + mp.mpf(1) / 4
        self.hard = {(1, 1): L.A[1], (-1, 1): mp.conj(L.B[-1]), (1, -1): L.B[1], (-1, -1): mp.conj(L.A[-1])}
        self.e = {a: e_series(-a * L.K, L.nu, nmax + 2) for a in (1, -1)}

    def soft_terms(self, kind, b):
        """Soft-end function as [(coefficient, first index, power of y, power of T)]:
        'W': Nf W_{-bK}(x);  'W0': W_0(x);  'D0': (x d_x - 1) W_0(x) = (x/2 - 1) W_0 - W_1,  x = 2 y T.
        The first index is a label: 'z' (0), 'o' (1), 'p' (+i kappa), 'm' (-i kappa)."""
        if kind == "W":                      # first index -b K = i b h kappa: label 'p' or 'm'
            return [(self.Nf, "p" if b * self.h > 0 else "m", 0, 0)]
        if kind == "W0":
            return [(mp.mpf(1), "z", 0, 0)]
        return [(mp.mpf(1), "z", 1, 1), (mp.mpf(-1), "z", 0, 0), (mp.mpf(-1), "o", 0, 0)]


def _soft_components(which, s3, sv, nmax):
    """Components of a soft line between V3 (index type s3) and its hard vertex (index type sv).

    Returns [(h, tensor, soft kind, D at the hard end, sign rule, dr, dy, dT, dt, 1/m^2 power)], where the
    tensor has shape (d3, dv), sign rule is a function (a, b) -> +-1 and dy counts explicit powers of y.
    """
    n = N1 if which == "C" else N2
    out = []
    if s3 == "s" and sv == "s":
        for h in (1, -1):
            out.append((h, proj_T(h, n, nmax), "W", 0, lambda a, b: 1, 0, 0, 0, 0, 0))
        out.append((0, outer(n, n, nmax), "D0", 1, lambda a, b: 1, 0, 0, 0, 0, 1))
    elif s3 == "s" and sv == "0":
        out.append((0, [[n[i]] for i in range(3)], "D0", 0, lambda a, b: a, 1, 1, 0, 1, 1))
    elif s3 == "0" and sv == "s":
        out.append((0, [[n[j] for j in range(3)]], "W0", 1, lambda a, b: -b, 0, 1, 1, 0, 1))
    else:
        out.append((0, [[ONE]], "W0", 0, lambda a, b: -a * b, 1, 2, 1, 1, 1))
    return out


def _hard_components(s1, s2, nmax):
    """Components of the hard line: [(lam, tensor (d1, d2), mY, d1, d2, sign rule (a1, a2), dt1, dt2, 1/m_h^2 power)]."""
    P = PV
    out = []
    if s1 == "s" and s2 == "s":
        delta = [[ONE if i == j else ZERO for j in range(3)] for i in range(3)]
        PP = outer(P, P, nmax)
        for lam in (1, -1):
            eps = [[padd(*[(-I * lam * levi(j, k, l) / 2, P[l]) for l in range(3) if levi(j, k, l)]) for k in range(3)]
                   for j in range(3)]
            out.append((lam, [[padd((mp.mpf(1) / 2, delta[j][k])) for k in range(3)] for j in range(3)], 0, 0, 0,
                        lambda a1, a2: 1, 0, 0, 0))
            out.append((lam, [[padd((-mp.mpf(1) / 2, PP[j][k])) for k in range(3)] for j in range(3)], -2, 0, 0,
                        lambda a1, a2: 1, 0, 0, 0))
            out.append((lam, eps, -1, 0, 0, lambda a1, a2: 1, 0, 0, 0))
        out.append((0, PP, -2, 1, 1, lambda a1, a2: 1, 0, 0, 1))
    elif s1 == "s" and s2 == "0":
        out.append((0, [[P[j]] for j in range(3)], 0, 1, 0, lambda a1, a2: a2, 0, 1, 1))
    elif s1 == "0" and s2 == "s":
        out.append((0, [[P[k] for k in range(3)]], 0, 0, 1, lambda a1, a2: -a1, 1, 0, 1))
    else:
        out.append((0, [[ONE]], 2, 0, 0, lambda a1, a2: -a1 * a2, 1, 1, 1))
        # contact term of the time-ordered A_0 A_0 function: +(t^2/m_h^2) delta(t1 - t2) on equal branches
        out.append(("c", [[ONE]], 0, 0, 0, lambda a1, a2: 1 if a1 == a2 else 0, 0, 0, 1))
    return out


# ------------------------------------------------------------------------------ soft moments
class SoftTable:
    """Soft moments  J(ia, ib, q, K1, K2) = int d^3q'/(2 pi)^3 y1^(nu - 1/2 + ia) y2^(nu - 1/2 + ib)
    int dT T^(q-1) e^(-T) W_{K1,nu}(2 y1 T) W_{K2,nu}(2 y2 T)   (ext/softvertex.py).

    For ia <= -2 or ib <= -2 a leg has a removable Gamma(0); both powers are then shifted by +-delta and
    the two values averaged (as in gauge_loop.py). The poles of single monomials cancel in the sum. With
    delta = 10^(-dps/3) the error of the average is of order delta^2 and a third of the digits are lost.
    """

    def __init__(self, nu, kappa, grid=None, delta=None, nterms=None, cache=None):
        self.nu = nu
        self.K = {"z": mp.mpf(0), "o": mp.mpf(1), "p": I * mp.mpf(kappa), "m": -I * mp.mpf(kappa)}
        self.cache = cache                        # optional path of a JSON file with computed moments
        self._new = 0
        self.grid = grid or Grid()
        self.delta = delta if delta is not None else mp.mpf(10) ** (-(mp.mp.dps // 3))
        self.nterms = nterms
        self.legs, self.vals = {}, {}

    def leg(self, ia, K, sgn):
        key = (ia, K, sgn)
        if key not in self.legs:
            self.legs[key] = Leg(self.nu - mp.mpf(1) / 2 + ia + sgn * self.delta, self.K[K], self.nu, nterms=self.nterms)
        return self.legs[key]

    def __call__(self, ia, ib, q, K1, K2):
        key = (ia, ib, q, K1, K2)
        if key not in self.vals:
            if ia <= -2 or ib <= -2:
                v = (soft_J(self.leg(ia, K1, 1), self.leg(ib, K2, 1), q, self.grid)
                     + soft_J(self.leg(ia, K1, -1), self.leg(ib, K2, -1), q, self.grid)) / 2
            else:
                v = soft_J(self.leg(ia, K1, 0), self.leg(ib, K2, 0), q, self.grid)
            self.vals[key] = v
            self._new += 1
            if self._new >= 200:                 # keep long runs restartable
                self.save()
        return self.vals[key]

    def late(self, ia, ib, q, K1, K2):
        """The same moment with each W replaced by its late-time branch B_+ (2 y T)^(1/2 + nu): the loop
        integral is then the block of Eq. [eq:C2] (dimensionally continued) and the time integral a Gamma function."""
        nu = self.nu
        return (B_sigma(self.K[K1], nu, 1) * B_sigma(self.K[K2], nu, 1) * mp.mpf(2) ** (1 + 2 * nu)
                * mp.gamma(q + 1 + 2 * nu) * C2hat(2 * nu + ia + 1, 2 * nu + ib + 1) / (2 * mp.pi) ** 2)

    # moments on disk, as strings at the working precision
    def load(self):
        import json, os
        if self.cache and os.path.exists(self.cache):
            with open(self.cache) as f:
                d = json.load(f)
            if d.get("meta") == self._meta():
                for k, (re, im) in d["vals"].items():
                    ia, ib, q, K1, K2 = k.split(",")
                    self.vals[(int(ia), int(ib), int(q), K1, K2)] = mp.mpc(re, im)
        return self

    def save(self):
        import json
        if self.cache and self._new:
            d = {"meta": self._meta(),
                 "vals": {",".join(map(str, k)): [mp.nstr(v.real, mp.mp.dps), mp.nstr(v.imag, mp.mp.dps)]
                          for k, v in self.vals.items()}}
            with open(self.cache, "w") as f:
                json.dump(d, f)
            self._new = 0

    def _meta(self):
        g = self.grid.key
        return [mp.nstr(self.nu, 25), mp.nstr(self.K["p"], 25), mp.mp.dps, [g[1], list(g[2]), g[3]],
                mp.nstr(self.delta, 5), self.nterms]


# ------------------------------------------------------------------------------ assembly
class Triangle:
    """Coefficients M_n of the pair family for the triangle and for the contracted bubble.

    mu, kappa: soft lines. mu_h, kappa_h: hard line (default: the same). nmax: highest order in r.
    """

    def __init__(self, mu, kappa, mu_h=None, kappa_h=None, nmax=0, grid=None, table=None, soft_pairs=None,
                 hard_helicities=(1, -1, 0), cache=None, general=False):
        self.general = general                        # expand the energy of V2 (general shape)
        self.mu, self.kappa = mp.mpf(mu), mp.mpf(kappa)
        self.mu_h = self.mu if mu_h is None else mp.mpf(mu_h)
        self.kappa_h = self.kappa if kappa_h is None else mp.mpf(kappa_h)
        self.nu = I * self.mu
        self.nmax = nmax
        self.soft = {h: SoftLine(self.mu, self.kappa, h, nmax) for h in (1, -1, 0)}
        self.hardline = HardLine(self.mu_h, self.kappa_h)
        self.table = table or SoftTable(self.nu, self.kappa, grid, cache=cache).load()
        self.soft_pairs = soft_pairs                  # optional restriction to some (hC, hB)
        self.hard_helicities = hard_helicities
        self._upow = None
        self._xis = None

    # ---- geometry: reduced polynomials of (contraction) x u^j
    def _upowers(self):
        if self._upow is None:
            up = [ONE]
            for _ in range(self.nmax):
                up.append(pmul(up[-1], USER, self.nmax))
            self._upow = up
        return self._upow

    @staticmethod
    def _contract(X, A, nmax):
        tot = []
        for j in range(len(X)):
            for k in range(len(X[0])):
                if X[j][k] and A[j][k]:
                    tot.append((1, pmul(X[j][k], A[j][k], nmax)))
        return padd(*tot)

    def _soft_tensor(self, tC, tB, s3, nmax):
        """X_jk = sum_i C_ij B_ik (times -1 for a temporal contraction at V3)."""
        sign = 1 if s3 == "s" else -1
        d1, d2 = len(tC[0]), len(tB[0])
        X = [[padd(*[(sign, pmul(tC[i][j], tB[i][k], nmax)) for i in range(len(tC))]) for k in range(d2)] for j in range(d1)]
        return X

    def accumulate(self, diagram="triangle"):
        """{(n, mc, ia, ib, q, K1, K2): coefficient} such that M_n(c) = sum coefficient c^mc J(ia, ib, q, K1, K2)."""
        nmax, nu = self.nmax, self.nu
        acc = {}
        up = self._upowers()
        two = mp.mpf(2)
        for s3, s1, s2 in product("s0", repeat=3):
            if diagram == "bubble" and s1 != s2:
                continue
            vsign = (-1) ** ((s1 == "0") + (s2 == "0")) if diagram == "triangle" else (-1) ** (s1 == "0")
            compC = _soft_components("C", s3, s1, nmax)
            compB = _soft_components("B", s3, s2, nmax)
            if diagram == "triangle":
                compA = [c for c in _hard_components(s1, s2, nmax)
                         if (0 if c[0] == "c" else c[0]) in self.hard_helicities]
            else:
                d = 3 if s1 == "s" else 1
                compA = [(None, [[ONE if j == k else ZERO for k in range(d)] for j in range(d)], 0, 0, 0,
                          lambda a1, a2: 1, 0, 0, 0)]
            for (hC, tC, kC, DC, sgC, drC, dyC, dTC, dtC, mC) in compC:
                for (hB, tB, kB, DB, sgB, drB, dyB, dTB, dtB, mB) in compB:
                    if self.soft_pairs is not None and (hC, hB) not in self.soft_pairs:
                        continue
                    X = self._soft_tensor(tC, tB, s3, nmax)
                    LC, LB = self.soft[hC], self.soft[hB]
                    dr0 = drC + drB
                    if dr0 > nmax:
                        continue
                    for (lam, tA, mY, dA1, dA2, sgA, dtA1, dtA2, mA) in compA:
                        N = self._contract(X, tA, nmax)
                        if not N:
                            continue
                        red = []
                        for j in range(nmax + 1):
                            red.append(reduce_poly(pmul(N, up[j], nmax)) if (diagram == "triangle" or j == 0) else {})
                        pref = vsign / (LC.m2 ** mC * LB.m2 ** mB)
                        if diagram == "triangle":
                            pref = pref / self.hardline.m2 ** mA
                        for n1 in range(nmax - dr0 + 1):
                            opC = (nu + n1 - mp.mpf(1) / 2) if DC else 1
                            for n2 in range(nmax - dr0 - n1 + 1):
                                opB = (nu + n2 - mp.mpf(1) / 2) if DB else 1
                                nb = dr0 + n1 + n2
                                x1 = mp.mpf(1) / 2 + nu + n1 + dtC + dtA1
                                x2 = mp.mpf(1) / 2 + nu + n2 + dtB + dtA2
                                p2 = two ** (2 * nu - 1 + n1 + n2)
                                for b in (1, -1):
                                    stC, stB = LC.soft_terms(kC, b), LB.soft_terms(kB, b)
                                    if self.general and diagram == "triangle":
                                        self._general_terms(acc, red, pref * opC * opB * p2, LC, LB, stC, stB, b,
                                                            (sgC, sgB, sgA), lam, (x1, x2, dA1, dA2, mY), nb,
                                                            (n1, n2, dyC, dyB, dTC, dTB))
                                        continue
                                    # sum over the hard branches of (hard ends) x (hard series)
                                    F = [mp.mpc(0)] * (nmax - nb + 1)
                                    for a1, a2 in product((1, -1), repeat=2):
                                        if diagram == "bubble" and a1 != a2:
                                            continue
                                        w = (LC.hard[(a1, b)] * LB.hard[(a2, b)] * LC.e[a1][n1] * LB.e[a2][n2]
                                             * sgC(a1, b) * sgB(a2, b) * sgA(a1, a2))
                                        if w == 0:
                                            continue
                                        if lam == "c":
                                            ser = [mp.gamma(x1 + x2 + 3) / two ** (x1 + x2 + 3)]
                                        elif diagram == "triangle":
                                            ser = self.hardline.series(lam, a1, a2, x1, x2, dA1, dA2, mY, nmax - nb)
                                        else:        # hard vertex: int dt t^2 e^(-2t) t^(x1 + x2)
                                            ser = [mp.gamma(x1 + x2 + 3) / two ** (x1 + x2 + 3)]
                                        for j in range(len(ser)):
                                            F[j] += w * ser[j]
                                    for j in range(nmax - nb + 1):
                                        if F[j] == 0 or not red[j]:
                                            continue
                                        cj = pref * opC * opB * p2 * F[j]
                                        for (nr, mc, da, db), v in red[j].items():
                                            n = nb + nr
                                            if n > nmax:
                                                continue
                                            for (c1, Kp1, y1p, T1) in stC:
                                                for (c2, Kp2, y2p, T2) in stB:
                                                    key = (n, mc, n1 + dyC + da + y1p, n2 + dyB + db + y2p,
                                                           1 + dTC + dTB + T1 + T2, Kp1, Kp2)
                                                    acc[key] = acc.get(key, 0) + cj * v * c1 * c2
        return acc

    def _general_terms(self, acc, red, pref, LC, LB, stC, stB, b, sgn, lam, hard, nb, soft):
        """The inner sum of accumulate() at general shape: the hard series is a double series in (u, xi), the
        merged vertex of the contact term has the energy 2 + xi, and S carries the factor k2 = 1 + xi."""
        nmax = self.nmax
        sgC, sgB, sgA = sgn
        x1, x2, dA1, dA2, mY = hard
        n1, n2, dyC, dyB, dTC, dTB = soft
        N = nmax - nb
        if self._xis is None:
            self._xis = xi_powers(nmax)
        F = {}
        for a1, a2 in product((1, -1), repeat=2):
            w = (LC.hard[(a1, b)] * LB.hard[(a2, b)] * LC.e[a1][n1] * LB.e[a2][n2]
                 * sgC(a1, b) * sgB(a2, b) * sgA(a1, a2))
            if w == 0:
                continue
            if lam == "c":
                X = x1 + x2 + 3
                ser = {(0, l): mp.gamma(X) / mp.mpf(2) ** (X + l) * cl for l, cl in enumerate(_binom_series(-X, N))}
            else:
                ser = self.hardline.series2(lam, a1, a2, x1, x2, dA1, dA2, mY, N)
            for k, v in ser.items():
                F[k] = F.get(k, 0) + w * v
        for (j, l), Fv in F.items():
            if Fv == 0 or not red[j]:
                continue
            cj = pref * Fv
            for lx in (l, l + 1):                       # xi^l (1 + xi)
                if lx > nmax:
                    continue
                for (nx, mcx), xv in self._xis[lx].items():
                    for (nr, mc, da, db), v in red[j].items():
                        n = nb + nr + nx
                        if n > nmax:
                            continue
                        for (c1, Kp1, y1p, T1) in stC:
                            for (c2, Kp2, y2p, T2) in stB:
                                key = (n, mc + mcx, n1 + dyC + da + y1p, n2 + dyB + db + y2p,
                                       1 + dTC + dTB + T1 + T2, Kp1, Kp2)
                                acc[key] = acc.get(key, 0) + cj * v * xv * c1 * c2

    # ---- evaluation
    def coefficients(self, diagram="triangle", isosceles=True, acc=None, late=False):
        """isosceles: [M_0, ..., M_nmax] at c = -r/2. Otherwise {(n, mc): coefficient of r^n c^mc}.
        late=True: the late-time estimate (soft ends on their leading branch; only meaningful at n = 0)."""
        acc = acc if acc is not None else self.accumulate(diagram)
        pre = mp.mpf(1) / 8 if diagram == "triangle" else mp.mpf(1) / 16
        parts = {}
        for (n, mc, ia, ib, q, K1, K2), cf in acc.items():
            if cf == 0:
                continue
            if late:
                if q == 1:
                    parts.setdefault((n, mc), []).append(cf * self.table.late(ia, ib, q, K1, K2))
                continue
            parts.setdefault((n, mc), []).append(cf * self.table(ia, ib, q, K1, K2))
        self.table.save()
        res = {k: pre * mp.fsum(v) for k, v in parts.items()}
        if not isosceles:
            return res
        out = [mp.mpc(0)] * (self.nmax + 1)
        for (n, mc), v in res.items():
            if n + mc <= self.nmax:
                out[n + mc] += v * (-mp.mpf(1) / 2) ** mc
        return out
