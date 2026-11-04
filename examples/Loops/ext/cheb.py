"""Chebyshev panels in u = ln t with a cumulative integration operator (mpmath).

Used for the nested time integrals of the hard part of the fermion triangle. On each panel the integrand
is interpolated at the n + 1 Chebyshev-Lobatto points, and the matrix Q maps values to the running integral
from the left end of the panel (spectral accuracy). Panels share their end points.
"""
import mpmath as mp

_CACHE = {}


def _cheb_matrices(n):
    """Lobatto nodes x_j = -cos(pi j/n) and the matrix Q with (Q f)_j = int_{-1}^{x_j} p_f(x) dx."""
    key = (n, mp.mp.prec)
    if key in _CACHE:
        return _CACHE[key]
    x = [-mp.cos(mp.pi * j / n) for j in range(n + 1)]
    # values -> Chebyshev coefficients: c_k = (2/n) sum'' f_j T_k(x_j), with halved end terms
    T = [[mp.cos(k * mp.acos(xj)) for xj in x] for k in range(n + 1)]   # T[k][j]
    def coeffs(f):
        c = []
        for k in range(n + 1):
            s = mp.fsum((f[j] * T[k][j] * (mp.mpf(1) / 2 if j in (0, n) else 1)) for j in range(n + 1))
            c.append(s * 2 / n * (mp.mpf(1) / 2 if k in (0, n) else 1))
        return c
    def antideriv(c):
        # int T_0 = T_1, int T_1 = T_2/4, int T_k = T_{k+1}/(2(k+1)) - T_{k-1}/(2(k-1))
        a = [mp.mpf(0)] * (n + 2)
        for k, ck in enumerate(c):
            if k == 0:
                a[1] += ck
            elif k == 1:
                a[2] += ck / 4
            else:
                a[k + 1] += ck / (2 * (k + 1))
                a[k - 1] -= ck / (2 * (k - 1))
        return a
    def evalc(a, xx):
        th = mp.acos(xx)
        return mp.fsum(ak * mp.cos(k * th) for k, ak in enumerate(a))
    Q = [[None] * (n + 1) for _ in range(n + 1)]
    for m in range(n + 1):
        e = [mp.mpf(1) if j == m else mp.mpf(0) for j in range(n + 1)]
        a = antideriv(coeffs(e))
        a0 = evalc(a, -1)
        for j in range(n + 1):
            Q[j][m] = evalc(a, x[j]) - a0
    _CACHE[key] = (x, Q)
    return x, Q


class Panels:
    """Panels covering u = ln t in segments [(u0, u1, width), ...], n + 1 Lobatto points each.

    Use narrower panels at large u, where e^(-c t) varies quickly in u.
    """

    def __init__(self, segments, n):
        self.n = n
        x, Q = _cheb_matrices(n)
        self.Q = Q
        self.u, self.hs = [], []
        for (u0, u1, width) in segments:
            u0, u1 = mp.mpf(u0), mp.mpf(u1)
            npan = int(mp.ceil((u1 - u0) / width))
            h = (u1 - u0) / npan
            for p in range(npan):
                a = u0 + p * h
                self.u.append([a + h * (1 + xj) / 2 for xj in x])
                self.hs.append(h)
        self.t = [[mp.exp(uj) for uj in row] for row in self.u]

    def nodes(self):
        for p, row in enumerate(self.t):
            for j, t in enumerate(row):
                yield p, j, t

    def cumulative(self, f):
        """F[p][j] = int_{u0}^{u_pj} f du for values f[p][j] (f already includes the dt = t du factor)."""
        F, run = [], mp.mpf(0)
        for row, h in zip(f, self.hs):
            half = h / 2
            Fr = [run + half * mp.fsum(self.Q[j][m] * row[m] for m in range(self.n + 1)) for j in range(self.n + 1)]
            F.append(Fr)
            run = Fr[-1]
        return F

    def integral(self, f):
        return mp.fsum(h / 2 * mp.fsum(self.Q[self.n][m] * row[m] for m in range(self.n + 1)) for row, h in zip(f, self.hs))
