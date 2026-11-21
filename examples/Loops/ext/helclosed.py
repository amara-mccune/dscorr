"""The soft vertex for any pair of first indices and any monomial, as one Euler integral of one Gauss function.

The soft moment of ext/softvertex.py,

    J(ia, ib, q; K1, K2) = int d^3l/(2 pi)^3 y1^(nu - 1/2 + ia) y2^(nu - 1/2 + ib)
                           int_0^oo dt t^(q-1) e^(-t) W_{K1,nu}(2 y1 t) W_{K2,nu}(2 y2 t),      k = 1,

is evaluated here for integers ia, ib >= -2 and q >= 0 and for any first indices K1, K2. This covers what
ext/softclosed.py leaves open for the helical vector: pairs with one transverse and one longitudinal line
(first indices neither equal nor opposite), and the monomials with y^-2 of the polarization sums.

Steps (closed_form_helical.tex has the derivation).
 1. Euler integral of each line in p = 1 + 2 s,
        e^(-y t) U(c, 1 - 2 nu, 2 y t) = 2^(2 nu)/Gamma(c) int_1^oo dp w(p) e^(-p y t),
        w(p) = (p - 1)^(c - 1) (p + 1)^(-2 nu - c),      c = 1/2 - nu - K.
    The time and loop integrals of the monomial y1^(ia+1) y2^(ib+1) give derivatives of one kernel,
        K_E(p1, p2) = 2 Gamma(E) [(1 + p2)^(-E) - (1 + p1)^(-E)]/(p1^2 - p2^2),     E = q - 1 - 2 nu - a - b,
    with a = ia + 1 and b = ib + 1. The derivatives are moved to the weights. An inverse power (a = -1) replaces the
    weight w by its primitive Om(p) = int_1^p w, an incomplete Beta function.
 2. The kernel is a difference of two principal values, so
        J = N0 2^(1-E) Gamma(E) int_0^1 du u^(E-1)/(2 - u) [phi2 Delta1 + phi1 Delta2],      u = 2/(1 + x),
        N0 = 2^(1 + 2 nu)/(4 pi^2 Gamma(c1) Gamma(c2)),
    where phi = d^a w/dx^a (or Om), Delta = S^(a)(x) - S^(a)(-x) (or the primitive of S(x) + S(-x)), and
        S(z) = int_1^oo dp w(p)/(p - z) = 2^(-1-2nu) B(c, 1 + 2 nu) 2F1(1, 1 + 2 nu; c + 1 + 2 nu; (1 + z)/2)
    is the Stieltjes transform of the weight, taken as a principal value on x > 1. Everything in the bracket has
    the singular points u = 0, 1, oo only. The factor 1/(2 - u) is the pole at x = 0 (cross-ratio -1 with the other
    three), and it is what keeps the integral from being a 3F2 at unit argument.
 3. With 1/(2 - u) = sum_n u^n/2^(n+1) the integral is a Kampe de Feriet series of arguments (1, 1/2): a series of
    3F2 (for y^-2: 4F3, 5F4) at unit argument whose terms fall as 2^-n.

Evaluation. The integral over u is split at u = 1/2. On (0, 1/2) every factor is expanded in u, and on (1/2, 1) in
v = 1 - u, with the connection formulas of the Gauss function. All series then converge as 2^-k and are integrated
term by term, so the cost is a few products of power series per moment. Divergent end-point powers (from the
derivatives of the weights) are continued analytically, which is the same continuation as in the definition of J.

Coincident exponents. At v = 0 each Delta has the term -pi cot(pi c) phi of its own line. The two lines combine to
-pi [cot(pi c1) + cot(pi c2)] phi1 phi2, which is integrated separately in a form that stays finite where c1 + c2 is an
integer (two enhanced lines at kappa = mu, or an enhanced and a longitudinal line at kappa = 2 mu). No limit is taken.

Working precision: the series loses about four digits at (mu, kappa) = (1, 1) and about ten at (3, 3.75), where M_A of
the bubble comes out to 8e-15 at 24 digits, to 2e-20 at 30, and to 5e-31 at 40. The loss is in the pairs of suppressed
lines, whose moments are smaller than their terms by about e^(-pi (mu + kappa)). A bubble (165 moments) takes about
40 s at 30 digits, which is the time of soft_J.
"""
import mpmath as mp

I = mp.mpc(0, 1)
HALF = mp.mpf(1) / 2

__all__ = ["HelClosed", "HelClosedTable", "soft_J_series"]


# ---------------------------------------------------------------------------------------------------------------
# generalized power series  sum_components t^rho0 sum_{k=0}^{N} a_k t^k   (several exponent classes rho0 mod 1)
# ---------------------------------------------------------------------------------------------------------------
def _cls(rho):
    r = mp.re(rho)
    f = r - mp.floor(r + mp.mpf("1e-9"))
    return (int(mp.nint(f * 10 ** 8)) % 10 ** 8, int(mp.nint(mp.im(rho) * 10 ** 8)))


class Ser:
    """dict {class: (rho0, [a_0, ..., a_N])}"""

    def __init__(self, N, comps=None):
        self.N = N
        self.c = {}
        for rho, a in (comps or []):
            self._add(mp.mpc(rho), list(a))

    def _add(self, rho, a):
        N = self.N
        a = (a + [mp.mpc(0)] * (N + 1 - len(a)))[:N + 1]
        key = _cls(rho)
        if key not in self.c:
            self.c[key] = (rho, a)
            return
        r0, b = self.c[key]
        d = int(mp.nint(mp.re(rho - r0)))
        if d >= 0:                                    # shift the new one up
            a = [mp.mpc(0)] * d + a[:N + 1 - d] if d <= N else [mp.mpc(0)] * (N + 1)
            self.c[key] = (r0, [x + y for x, y in zip(b, a)])
        else:
            b = [mp.mpc(0)] * (-d) + b[:N + 1 + d] if -d <= N else [mp.mpc(0)] * (N + 1)
            self.c[key] = (rho, [x + y for x, y in zip(a, b)])

    def __add__(self, other):
        out = Ser(self.N)
        for rho, a in list(self.c.values()) + list(other.c.values()):
            out._add(rho, list(a))
        return out

    def scale(self, s):
        out = Ser(self.N)
        out.c = {k: (rho, [s * x for x in a]) for k, (rho, a) in self.c.items()}
        return out

    def __mul__(self, other):
        N = self.N
        out = Ser(N)
        for r1, a in self.c.values():
            for r2, b in other.c.values():
                out._add(r1 + r2, [mp.fdot(a[:n + 1], b[n::-1]) for n in range(N + 1)])
        return out

    def shift(self, s):
        """multiply by t^s"""
        out = Ser(self.N)
        for rho, a in self.c.values():
            out._add(rho + s, list(a))
        return out

    def ddt(self):
        out = Ser(self.N)
        for rho, a in self.c.values():
            out._add(rho - 1, [(rho + k) * x for k, x in enumerate(a)])
        return out

    def prim(self):
        """term-by-term primitive without a constant (analytic continuation at t = 0)"""
        out = Ser(self.N)
        for rho, a in self.c.values():
            b = []
            for k, x in enumerate(a):
                e = rho + k + 1
                if abs(e) < mp.mpf(10) ** (-mp.mp.dps // 2):
                    if abs(x) > mp.mpf(10) ** (-mp.mp.dps + 6):
                        raise ValueError("helclosed: logarithm in a primitive (coincident exponents)")
                    b.append(mp.mpc(0))
                else:
                    b.append(x / e)
            out._add(rho + 1, b)
        return out

    def at(self, t):
        tot = mp.mpc(0)
        for rho, a in self.c.values():
            tot += mp.power(t, rho) * mp.polyval(a[::-1], t)
        return tot

    def tail(self, t):
        """size of the last terms relative to the largest, at t (a convergence check)"""
        big, last = mp.mpf(0), mp.mpf(0)
        for rho, a in self.c.values():
            w = abs(mp.power(t, rho))
            for k, x in enumerate(a):
                s = abs(x) * w * mp.power(t, k)
                big = max(big, s)
                if k >= self.N - 2:
                    last = max(last, s)
        return last / big if big else mp.mpf(0)


def _binom(sigma, N):
    """(1 - t)^sigma"""
    c = [mp.mpc(1)]
    for k in range(1, N + 1):
        c.append(c[-1] * (k - 1 - sigma) / k)
    return c


def _hyp1(b, c, N):
    """2F1(1, b; c; t)"""
    a = [mp.mpc(1)]
    for k in range(N):
        a.append(a[-1] * (b + k) / (c + k))
    return a


# ---------------------------------------------------------------------------------------------------------------
class _Line:
    """Local series of one line (first index K) at u = 0 (x = oo) and at v = 1 - u = 0 (x = 1)."""

    def __init__(self, nu, K, N):
        self.nu, self.K, self.N = nu, K, N
        self.c = HALF - nu - K
        self._w, self._D, self._L, self._Om = {}, {}, {}, {}
        self.tau = mp.mpf(2) ** (-1 - 2 * nu)

    # d/dx on a series in u or v
    def _ddx(self, s, side):
        N = self.N
        if side == "u":                               # d/dx = -(u^2/2) d/du
            return s.ddt().shift(2).scale(-HALF)
        d = s.ddt()                                   # d/dx = ((1 - v)^2/2) d/dv
        return (d * Ser(N, [(0, [HALF, -1, HALF])]))

    def weight(self, side, a=0):
        """d^a w/dx^a"""
        key = (side, a)
        if key not in self._w:
            nu, c, N = self.nu, self.c, self.N
            if a == 0:
                if side == "u":
                    s = Ser(N, [(1 + 2 * nu, [self.tau * x for x in _binom(c - 1, N)])])
                else:
                    s = Ser(N, [(c - 1, [self.tau * x for x in _binom(1 + 2 * nu, N)])])
            else:
                s = self._ddx(self.weight(side, a - 1), side)
            self._w[key] = s
        return self._w[key]

    def fpm(self, side):
        """(f_+, f_-) = (S(x) as a principal value, S(-x)). At v = 0 the term -pi cot(pi c) w(x) of f_+ is left out:
        with the same term of the other line it gives -pi [cot(pi c1) + cot(pi c2)] phi1 phi2, whose integral is
        finite where c1 + c2 is an integer (for example two enhanced lines at kappa = mu). HelClosed.J adds it."""
        nu, c, N, tau = self.nu, self.c, self.N, self.tau
        if side == "u":
            A = tau * mp.beta(c, 2 * nu)
            fp = Ser(N, [(1, [-A * x for x in _hyp1(1 - c - 2 * nu, 1 - 2 * nu, N)]),
                         (1 + 2 * nu, [tau * mp.pi * mp.cot(2 * mp.pi * nu) * x for x in _binom(c - 1, N)])])
            fm = Ser(N, [(1, [A * x for x in _hyp1(c, 1 - 2 * nu, N)]),
                         (1 + 2 * nu, [-tau * mp.pi / mp.sin(2 * mp.pi * nu) * x for x in _binom(-2 * nu - c, N)])])
        else:
            one_v = Ser(N, [(0, [1, -1])])
            fp = Ser(N, [(0, [tau * mp.beta(c - 1, 1 + 2 * nu) * x for x in _hyp1(1 - c - 2 * nu, 2 - c, N)])]) * one_v
            fm = Ser(N, [(0, [tau * mp.beta(c, 1 + 2 * nu) * x for x in _hyp1(c, c + 1 + 2 * nu, N)])]) * one_v
        return fp, fm

    def Delta(self, side, a=0):
        """S^(a)(x) - S^(a)(-x) = 2 x PV int dp [d^a w](p)/(p^2 - x^2)   (at v = 0: without -pi cot(pi c) d^a w)"""
        key = (side, a)
        if key not in self._D:
            fp, fm = self.fpm(side)
            for _ in range(a):
                fp, fm = self._ddx(fp, side), self._ddx(fm, side)
            self._D[key] = fp + fm.scale(-(-1) ** a)
        return self._D[key]

    def Om(self, side):
        """Om(x) = int_1^x w = 2^(-2 nu) B_v(c, 2 nu)"""
        if side not in self._Om:
            nu, c, N = self.nu, self.c, self.N
            t2 = mp.mpf(2) ** (-2 * nu)
            if side == "u":
                a, cf = [], mp.mpc(1)
                for r in range(N + 1):
                    a.append(-t2 * cf / (2 * nu + r))
                    cf = cf * (1 - c + r) / (r + 1)
                s = Ser(N, [(0, [t2 * mp.beta(c, 2 * nu)]), (2 * nu, a)])
            else:
                a, cf = [], mp.mpc(1)
                for r in range(N + 1):
                    a.append(t2 * cf / (c + r))
                    cf = cf * (1 - 2 * nu + r) / (r + 1)
                s = Ser(N, [(c, a)])
            self._Om[side] = s
        return self._Om[side]

    def Lam(self, side):
        """Lam(x) = int_1^oo dp w(p) ln|(p + x)/(p - x)|, the primitive of S(x) + S(-x) that vanishes at x = oo
        (at v = 0: without -pi cot(pi c) Om(x))"""
        if side not in self._L:
            N = self.N
            if side == "u":                           # dx = -2 du/u^2
                fp, fm = self.fpm("u")
                self._L[side] = (fp + fm).shift(-2).prim().scale(-2)
            else:                                     # dx = 2 dv/(1 - v)^2, matched to the u series at u = v = 1/2
                fp, fm = self.fpm("v")
                inv2 = Ser(N, [(0, [mp.mpf(k + 1) for k in range(N + 1)])])       # (1 - v)^-2
                P = ((fp + fm) * inv2).prim().scale(2)
                const = self.Lam("u").at(HALF) - P.at(HALF) + mp.pi * mp.cot(mp.pi * self.c) * self.Om("v").at(HALF)
                self._L[side] = P + Ser(N, [(0, [const])])
        return self._L[side]


class HelClosed:
    """Soft moments at one order nu, for any first indices and ia, ib >= -2."""

    def __init__(self, nu, N=None):
        self.nu = mp.mpc(nu)
        self.N = N or int((mp.mp.dps + 8) * mp.log(10) / mp.log(2)) + 12
        self._lines, self._j, self._wt = {}, {}, {}
        self.worst_tail = mp.mpf(0)

    def line(self, K):
        key = mp.nstr(mp.mpc(K), 25)
        if key not in self._lines:
            self._lines[key] = _Line(self.nu, mp.mpc(K), self.N)
        return self._lines[key]

    def _measure(self, E, side):
        """u^(E-1)/(2 - u) du on (0, 1/2), as a series in u or in v"""
        key = (mp.nstr(E, 25), side)
        if key not in self._wt:
            N = self.N
            if side == "u":
                s = Ser(N, [(E - 1, [mp.mpf(2) ** (-k - 1) for k in range(N + 1)])])
            else:
                s = Ser(N, [(0, _binom(E - 1, N))]) * Ser(N, [(0, [mp.mpf((-1) ** k) for k in range(N + 1)])])
            self._wt[key] = s
        return self._wt[key]

    @staticmethod
    def _power_power(pp, c1, c2):
        """-pi [cot(pi c1) + cot(pi c2)] int_0^(1/2) dv pp(v), term by term. With e the exponent after integration,
        -pi [cot(pi c1) + cot(pi c2)]/e = -pi (-1)^m sin(pi e)/(e sin(pi c1) sin(pi c2)),  m = e - c1 - c2 (an integer),
        which is finite at e = 0."""
        den = mp.sin(mp.pi * c1) * mp.sin(mp.pi * c2)
        tot = mp.mpc(0)
        for rho, a in pp.c.values():
            m0 = rho + 1 - c1 - c2
            if abs(m0 - mp.nint(mp.re(m0))) > mp.mpf(10) ** (-mp.mp.dps // 2):
                raise ValueError("helclosed: unexpected exponent in the power-power term")
            m0 = int(mp.nint(mp.re(m0)))
            t = mp.power(HALF, rho + 1)
            for k, x in enumerate(a):
                e = rho + 1 + k
                sinc = mp.pi if abs(e) < mp.mpf(10) ** (-mp.mp.dps + 2) else mp.sin(mp.pi * e) / e
                tot += x * t * (-mp.pi) * (-1) ** (m0 + k) * sinc / den
                t = t * HALF
        return tot

    def J(self, ia, ib, q, K1, K2):
        key = (ia, ib, q, mp.nstr(mp.mpc(K1), 25), mp.nstr(mp.mpc(K2), 25))
        if key in self._j:
            return self._j[key]
        a, b = ia + 1, ib + 1
        if a < -1 or b < -1 or q < 0:
            raise ValueError("helclosed: needs ia, ib >= -2 and q >= 0")
        nu = self.nu
        L1, L2 = self.line(K1), self.line(K2)
        E = q - 1 - 2 * nu - a - b
        tot = mp.mpc(0)
        for side in ("u", "v"):
            ph1 = L1.weight(side, a) if a >= 0 else L1.Om(side)
            ph2 = L2.weight(side, b) if b >= 0 else L2.Om(side)
            D1 = L1.Delta(side, a) if a >= 0 else L1.Lam(side)
            D2 = L2.Delta(side, b) if b >= 0 else L2.Lam(side)
            meas = self._measure(E, side)
            integrand = (ph2 * D1 + ph1 * D2) * meas
            self.worst_tail = max(self.worst_tail, integrand.tail(HALF))
            tot += integrand.prim().at(HALF)
            if side == "v":                               # -pi [cot(pi c1) + cot(pi c2)] int_0^(1/2) phi1 phi2
                tot += self._power_power(ph1 * ph2 * meas, L1.c, L2.c)
        N0 = mp.mpf(2) ** (1 + 2 * nu) / (4 * mp.pi ** 2 * mp.gamma(L1.c) * mp.gamma(L2.c))
        val = N0 * mp.mpf(2) ** (1 - E) * mp.gamma(E) * tot
        self._j[key] = val
        return val


def soft_J_series(ia, ib, q, nu, K1, K2):
    """One soft moment (use HelClosed for many moments at the same nu)."""
    return HelClosed(nu).J(ia, ib, q, K1, K2)


class HelClosedTable:
    """Stand-in for the soft-moment table of ext/vectri.py: every helical moment from the series above."""

    def __init__(self, nu, kappa, N=None):
        self.nu = mp.mpc(nu)
        self.K = {"z": mp.mpf(0), "o": mp.mpf(1), "p": I * mp.mpf(kappa), "m": -I * mp.mpf(kappa)}
        self.closed = HelClosed(nu, N)

    def __call__(self, ia, ib, q, K1, K2):
        return self.closed.J(ia, ib, q, self.K[K1], self.K[K2])

    def late(self, ia, ib, q, K1, K2):
        from .softvertex import B_sigma
        from .blocks import C2hat
        nu = self.nu
        return (B_sigma(self.K[K1], nu, 1) * B_sigma(self.K[K2], nu, 1) * mp.mpf(2) ** (1 + 2 * nu)
                * mp.gamma(q + 1 + 2 * nu) * C2hat(2 * nu + ia + 1, 2 * nu + ib + 1) / (2 * mp.pi) ** 2)

    def load(self):
        return self

    def save(self):
        pass
