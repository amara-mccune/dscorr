"""The soft vertex with a chemical potential in closed form (equal or opposite first indices, modulo integers).

The soft moment of ext/softvertex.py,

    J(ia, ib, q; K1, K2) = int d^3l/(2 pi)^3 y1^(nu - 1/2 + ia) y2^(nu - 1/2 + ib)
                           int_0^oo dt t^(q-1) e^(-t) W_{K1,nu}(2 y1 t) W_{K2,nu}(2 y2 t),      k = 1,

for integers ia, ib >= -1 and q >= 0, is reduced here to Gamma functions, 2F1 at argument -1, 3F2 at unit
argument, and one-fold Barnes integrals of Gamma functions, whenever K1 - K2 or K1 + K2 is an integer. Each
Barnes integral is one Meijer function G^{3,4}_{4,4} (or G^{4,3}_{4,4}) at argument -1 on the branch
log(-1) = +-i pi, the kind of function in the closed form of Song et al. (2611.23220) for the leading coefficient
of the fermion triangle. This covers every pair of a Dirac line (K = +-1/2 +- i kappa), the transverse pairs of a
helical vector (K = +- i kappa), and the scalar line (K = 0). Not covered: pairs with one transverse and one
longitudinal line, and monomials with ia or ib below -1.

Steps (closed_form_soft_vertex.tex has the derivation).
 1. Euler integrals for both lines, W_{K,nu}(x) = e^(-x/2) x^(1/2-nu) U(c, 1-2nu, x), c = 1/2 - nu - K. The time
    integral and the loop integral (in y1 +- y2) are then elementary. With z = s1/(1+s1), xi = s2/(1+s2) every
    monomial is a finite sum of
        I(a1, a2; M, q; L, n) = int_0^1 int_0^1 dz dxi z^(a1-1) xi^(a2-1) (1-z)^M (1-xi)^q (z - xi +- i0)^(-L) (1 - z xi)^(-n),
    with M = 2 nu + integer and the pole at z = xi cancelling in the sum.
 2. (1 - xi)^q is expanded, and L is lowered to one by parts in z.
 3. Equal type, a1 - a2 integer. On the conics z xi = x the integrand has two branch points only, so the fibre
    integral is an incomplete Beta function of 1 +- sqrt(x). Its Barnes form and the remaining Beta integral give
        master(A, M, n) ~ int ds Gamma(1+s) Gamma(-s)^2 Gamma(M+s) [e^(+-i pi s) -+ 1] Gamma(A + s/2)/Gamma(A + M - n + 1 + s/2).
 4. Opposite type, a1 + a2 + M integer. The xi contour is completed to (0, oo) (residues: 2F1 at -1) and the piece
    (1, oo) is mapped back by xi -> 1/xi, which gives the same family with (1-z)^M z^(-M) on the fibre and
    Gamma(At - s/2)/Gamma(At + M - n + 1 - s/2) in the master.

With s = 2 sigma the master of step 3 is, for even and odd a1 - a2,
    -+i c G^{3,4}_{4,4}(e^(+-i pi) | 1/2, 1 - M/2, (1 - M)/2, 1 - A; 0, 1/2, 1/2, 1 - A - G)   and
        c G^{3,4}_{4,4}(e^(+-i pi) | 0,   1 - M/2, (1 - M)/2, 1 - A; 0, 0,   1/2, 1 - A - G),
with G = M - n + 1 and c = 2^(M-2) Gamma(G)/(pi Gamma(M)). The residue series of these functions do not converge
at the physical parameters, so they are evaluated on the contour.

All Barnes integrals are taken on Re s = -1/2 by the trapezoid rule on one grid (exponentially convergent), with
the residues of the poles that the analytic continuation in the exponents moves across that line. Where a pole
ladder has half-integer real parts (lines with Re(1/2 - nu - K) = 1/2, as for a helical or a scalar line) the line
Re s = -1/4 is used instead. Exponents with other real parts are not supported.

Working precision: the integrands are of the size of the result, and the sums over the monomial lose few digits.
The coefficient C of the fermion triangle at (kappa, m) = (20, 2) comes out to 6e-25 at 30 digits and to 5e-15 at
20 digits (soft_closed.py), where soft_J and the closed form of Song et al. need 60 to 80 digits.
"""
import mpmath as mp

I = mp.mpc(0, 1)
HALF = mp.mpf(1) / 2

__all__ = ["BarnesGrid", "SoftClosed", "soft_J_closed"]


def _key(x):
    return mp.nstr(mp.mpc(x), 28)


def _is_int(x):
    return abs(x - mp.nint(mp.re(x))) < mp.mpf(10) ** (-mp.mp.dps // 2)


def _ff(x, k):
    """falling factorial x (x-1) ... (x-k+1)"""
    return mp.fprod(x - j for j in range(k)) if k else mp.mpf(1)


def _multi(N):
    for i1 in range(N + 1):
        for i2 in range(N + 1 - i1):
            yield i1, i2, N - i1 - i2


def _uv_poly(a, b):
    """((u+v)/2)^a ((u-v)/2)^b = sum g[i,k] u^i v^k"""
    g = {}
    for i1 in range(a + 1):
        for i2 in range(b + 1):
            c = mp.binomial(a, i1) * mp.binomial(b, i2) * (-1) ** (b - i2) / mp.mpf(2) ** (a + b)
            key = (i1 + i2, a + b - i1 - i2)
            g[key] = g.get(key, 0) + c
    return {k: v for k, v in g.items() if v != 0}


def _beta2f1(k, be, ga):
    """int_0^1 dz z^(be-1) (1-z)^(ga-1) (1+z)^(-k) = B(be, ga) 2F1(k, be; be+ga; -1), also where be + ga is a
    non-positive integer (the Beta function then vanishes and the Gauss function has a pole)."""
    c = be + ga
    if _is_int(c) and mp.re(c) < HALF:
        m = -int(mp.nint(mp.re(c)))
        return (mp.gamma(be) * mp.gamma(ga) * mp.rf(k, m + 1) * mp.rf(be, m + 1) * (-1) ** (m + 1) / mp.factorial(m + 1)
                * mp.hyp2f1(k + m + 1, be + m + 1, m + 2, -1))
    return mp.beta(be, ga) * mp.hyp2f1(k, be, c, -1)


class BarnesGrid:
    """Nodes s = sig + i t on |t| <= T with step h, and cached tables of Gamma functions on them."""

    def __init__(self, T, sig=-HALF):
        self.prec = mp.mp.prec
        self.sig = mp.mpf(sig)
        dist = min(-self.sig, 1 + self.sig)                      # distance to the poles at s = 0 and s = -1
        self.h = 2 * mp.pi * dist / ((mp.mp.dps + 8) * mp.log(10))   # trapezoid error e^(-2 pi dist/h)
        n = int(mp.ceil(T / self.h))
        self.s = [mp.mpc(self.sig, k * self.h) for k in range(-n, n + 1)]
        self.w = self.h / (2 * mp.pi)
        self._t = {}
        self.nsum = 0

    def gam(self, a, c=1, recip=False):
        """[Gamma(a + c s)] on the nodes (or its reciprocal), by recurrence from a neighbouring table if there is one"""
        a = mp.mpc(a)
        key = (c, recip, _key(a))
        t = self._t
        if key not in t:
            lo, hi = (c, recip, _key(a - 1)), (c, recip, _key(a + 1))
            if lo in t:
                t[key] = ([x / (a - 1 + c * s) for x, s in zip(t[lo], self.s)] if recip
                          else [x * (a - 1 + c * s) for x, s in zip(t[lo], self.s)])
            elif hi in t:
                t[key] = ([x * (a + c * s) for x, s in zip(t[hi], self.s)] if recip
                          else [x / (a + c * s) for x, s in zip(t[hi], self.s)])
            else:
                f = mp.rgamma if recip else mp.gamma
                t[key] = [f(a + c * s) for s in self.s]
        return t[key]

    def fixed(self, name):
        t = self._t
        if name not in t:
            if name == "lam0":                                   # Gamma(1+s) Gamma(-s)^2
                t[name] = [mp.gamma(1 + s) * mp.gamma(-s) ** 2 for s in self.s]
            elif name == "ep":
                t[name] = [mp.exp(I * mp.pi * s) for s in self.s]
            elif name == "em":
                t[name] = [mp.exp(-I * mp.pi * s) for s in self.s]
            elif name == "gneg":                                 # Gamma(-s)
                t[name] = [mp.gamma(-s) for s in self.s]
            elif name == "psin":                                 # pi/sin(pi s)
                t[name] = [mp.pi / mp.sin(mp.pi * s) for s in self.s]
        return t[name]

    def prod(self, key, build):
        if key not in self._t:
            self._t[key] = build()
        return self._t[key]

    def integrate(self, *arrays):
        """(1/2 pi i) int ds prod(arrays), with a check that the range is long enough"""
        self.nsum += 1
        if len(arrays) == 2:
            terms = [x * y for x, y in zip(*arrays)]
        else:
            terms = [mp.fprod(xs) for xs in zip(*arrays)]
        big = max(abs(x) for x in terms)
        edge = max(abs(terms[0]), abs(terms[-1]))
        if big and edge > big * mp.mpf(10) ** (-mp.mp.dps - 3):
            raise RuntimeError(f"BarnesGrid: integrand at the edge is {mp.nstr(edge / big, 3)} of its maximum; increase T")
        return self.w * mp.fsum(terms)


class SoftClosed:
    """Closed-form soft moments at one order nu (both lines) for the first indices given at each call.

    Tmax: half-length of the Barnes contour. The default covers |Im K| <= kmax.
    """

    def __init__(self, nu, kmax=0, Tmax=None):
        self.nu = mp.mpc(nu)
        if Tmax is None:
            Tmax = 2 * abs(mp.im(self.nu)) + 2 * abs(mp.mpmathify(kmax)) + 6 + (mp.mp.dps + 10) * mp.log(10) / mp.pi
        self.Tmax = Tmax
        self.grid = BarnesGrid(Tmax)                             # Re s = -1/2: pole ladders at integer real parts
        self._gridq = None                                       # Re s = -1/4: a ladder at half-integer real parts
        self._m, self._i, self._j = {}, {}, {}

    def _grid_for(self, *ladders):
        """The contour that keeps its distance from the given pole positions (their real parts modulo one)."""
        half = False
        for x in ladders:
            f = mp.re(x) - mp.floor(mp.re(x))
            if abs(f - HALF) < mp.mpf("0.05"):
                half = True
            elif min(f, 1 - f) > mp.mpf("0.05"):
                raise ValueError("softclosed: a pole ladder lies within 1/4 of both contours; the real parts of the "
                                 "exponents must be integers or half-integers")
        if not half:
            return self.grid
        if self._gridq is None:
            self._gridq = BarnesGrid(self.Tmax, sig=-mp.mpf(1) / 4)
        return self._gridq

    # ------------------------------------------------------------------ masters
    def _lamM(self, g, M, sign, pd):

        def build():
            ph = g.fixed("ep" if sign > 0 else "em")
            return [l * gm * (p - pd) for l, gm, p in zip(g.fixed("lam0"), g.gam(M), ph)]
        return g.prod(("lamM", _key(M), sign, pd), build)

    @staticmethod
    def _lam_at(s0, M, sign, pd):
        return mp.gamma(1 + s0) * mp.gamma(-s0) ** 2 * mp.gamma(M + s0) * (mp.exp(sign * I * mp.pi * s0) - pd)

    def master(self, A, M, n, dpar, sign, twisted=False):
        """Gamma(G)/(2 Gamma(M)) (1/2 pi i) int ds Gamma(1+s) Gamma(-s)^2 Gamma(M+s) [e^(sign i pi s) - (-1)^d]
        Gamma(A + e s/2)/Gamma(A + G + e s/2),  G = M - n + 1,  e = +1 (equal type) or -1 (twisted).
        Contour: Re s in (-1, 0), with the poles of Gamma(A + e s/2) on its left (e = +1) or right (e = -1)."""
        key = (twisted, _key(A), _key(M), n, dpar % 2, sign)
        if key in self._m:
            return self._m[key]
        g = self._grid_for(2 * A)
        pd = -1 if dpar % 2 else 1
        G = M - n + 1
        e = -HALF if twisted else HALF
        ratio = g.prod(("rat", e, _key(A), _key(A + G)),
                       lambda: [x * y for x, y in zip(g.gam(A, e), g.gam(A + G, e, recip=True))])
        val = g.integrate(self._lamM(g, M, sign, pd), ratio)
        m = 0
        while True:                                              # poles of Gamma(A + e s/2) on the wrong side of the line
            s0 = (2 * A + 2 * m) if twisted else (-2 * A - 2 * m)
            if (mp.re(s0) > g.sig) if twisted else (mp.re(s0) < g.sig):
                break
            val += self._lam_at(s0, M, sign, pd) * 2 * (-1) ** m / mp.factorial(m) * mp.rgamma(G - m)
            m += 1
        val *= mp.gamma(G) / (2 * mp.gamma(M))
        self._m[key] = val
        return val

    def T3(self, beta, rho, M, n):
        """int_0^1 dx x^(beta-1) (1-x)^(-n) int_x^1 dz z^rho (1-z)^M, for rho integer or rho + M integer.

        It equals B(beta+rho+1, M+1)/beta 3F2(n, beta, beta+rho+1; beta+1, beta+rho+M+2; 1). Here it is evaluated as
        elementary Beta functions where the inner integral is elementary, and otherwise from its Barnes integral."""
        key = ("T3", _key(beta), _key(rho), _key(M), n)
        if key in self._m:
            return self._m[key]
        g = self._grid_for(beta)
        Gp = M + 2 - n
        if _is_int(rho) and mp.re(rho) >= 0:                     # polynomial in z
            r = int(mp.nint(mp.re(rho)))
            val = mp.fsum(mp.binomial(r, t) * (-1) ** t / (M + 1 + t) * mp.beta(beta, M + 2 + t - n) for t in range(r + 1))
        elif _is_int(rho + M) and mp.re(rho + M) <= -2:          # polynomial in v = (1-z)/z
            r = -int(mp.nint(mp.re(rho + M)))
            val = mp.fsum(mp.binomial(r - 2, t) / (M + t + 1) * mp.beta(beta - M - t - 1, M + t + 2 - n) for t in range(r - 1))
        else:
            # [1/(Gamma(-rho) Gamma(M+2+rho))] (1/2 pi i) int ds pi/sin(pi(s-rho)) Gamma(M+1+s) Gamma(-s) B(beta+s, G')
            if _is_int(rho):
                sg = (-1) ** int(mp.nint(mp.re(rho)))
                S = g.fixed("psin")
                Sat = lambda s0: sg * mp.pi / mp.sin(mp.pi * s0)
            else:
                j = int(mp.nint(mp.re(rho + M)))
                N0 = int(mp.nint(mp.re(M - 2 * self.nu)))            # M = 2 nu + N0
                sg = (-1) ** (j + N0)
                S = g.prod("psinnu", lambda: [mp.pi / mp.sin(mp.pi * (s + 2 * self.nu)) for s in g.s])
                Sat = lambda s0: sg * mp.pi / mp.sin(mp.pi * (s0 + 2 * self.nu))
            ker = g.prod(("T3k", _key(M)), lambda: [x * y for x, y in zip(g.gam(M + 1), g.fixed("gneg"))])
            rat = g.prod(("rat1", _key(beta), _key(beta + Gp)),
                         lambda: [x * y for x, y in zip(g.gam(beta), g.gam(beta + Gp, recip=True))])
            val = sg * g.integrate(S, ker, rat)

            def rest(s0):
                return mp.gamma(M + 1 + s0) * mp.gamma(-s0) * mp.gamma(beta + s0) * mp.rgamma(beta + Gp + s0)
            k = 0
            while mp.re(rho + k) > g.sig:                        # left poles of the sine on the right of the line
                val += (-1) ** k * rest(rho + k)
                k -= 1
            k = 1
            while mp.re(rho + k) < g.sig:                        # right poles of the sine on the left of the line
                val -= (-1) ** k * rest(rho + k)
                k += 1
            m = 0
            while mp.re(-beta - m) > g.sig:                      # poles of Gamma(beta + s) on the right of the line
                s0 = -beta - m
                val += (-1) ** m / mp.factorial(m) * Sat(s0) * mp.gamma(M + 1 + s0) * mp.gamma(-s0) * mp.rgamma(Gp - m)
                m += 1
            val *= mp.gamma(Gp) * mp.rgamma(-rho) * mp.rgamma(M + 2 + rho)
        self._m[key] = val
        return val

    # ------------------------------------------------------------------ the family I(a1, a2; M, q; L, n)
    def I1base(self, a1, a2, M, n, sign):
        """int int dz dxi z^(a1-1) xi^(a2-1) (1-z)^M / [(z - xi + sign i0) (1 - z xi)^n]"""
        G = M - n + 1
        if _is_int(a1 - a2):                                     # equal type
            d = int(mp.nint(mp.re(a1 - a2)))
            A = (a1 + a2 - 1) / 2
            val = self.master(A, M, n, d, sign)
            if d % 2:
                val -= mp.beta(A, G) / M
            if d >= 1:
                for r in range(d):
                    if (d - 1 - r) % 2:
                        val += self.T3(a2 + mp.mpf(d - 2 - r) / 2, mp.mpf(r), M, n)
            elif d <= -1:
                for r in range(1, -d + 1):
                    if (d + r - 1) % 2:
                        val -= self.T3(a2 + mp.mpf(d + r - 2) / 2, mp.mpf(-r), M, n)
            return val
        kap = int(mp.nint(mp.re(a1 - a2 + M + 1)))               # twisted type: a1 - a2 + M integer
        if not _is_int(a1 - a2 + M):
            raise ValueError("first indices are neither equal nor opposite modulo integers")
        At = (a1 + a2 - M - 1) / 2
        val = self.master(At, M, n, kap - 1, sign, twisted=True)
        if (kap - 1) % 2:
            val -= mp.beta(At, G) / M
            val += self.T3(a2 + mp.mpf(kap - 2) / 2, -M - 1, M, n)
        if kap >= 2:
            for r in range(kap - 1):
                if (kap - 2 - r) % 2:
                    val += self.T3(a2 + mp.mpf(kap - 3 - r) / 2, -M + r, M, n)
        elif kap <= 0:
            for r in range(1, 2 - kap):
                if (kap - 2 + r) % 2:
                    val -= self.T3(a2 + mp.mpf(kap - 3 + r) / 2, -M - r, M, n)
        return val

    def Ired(self, a1, a2, M, L, n, sign):
        """the same with (z - xi + sign i0)^(-L): L is lowered to one by parts in z"""
        key = (_key(a1), _key(a2), _key(M), L, n, sign)
        if key not in self._i:
            if L == 1:
                v = self.I1base(a1, a2, M, n, sign)
            else:
                v = ((a1 - 1) * self.Ired(a1 - 1, a2, M, L - 1, n, sign) - M * self.Ired(a1, a2, M - 1, L - 1, n, sign)
                     + n * self.Ired(a1, a2 + 1, M, L - 1, n + 1, sign)) / (L - 1)
            self._i[key] = v
        return self._i[key]

    def Iq(self, a1, a2, M, q, L, n, sign):
        return mp.fsum(mp.binomial(q, j) * (-1) ** j * self.Ired(a1, a2 + j, M, L, n, sign) for j in range(q + 1))

    def I2refl(self, a1, a2, M, q, L, n, sign):
        """I(a1, a2; M, q; L, n) for a1 + a2 + M integer: the xi contour is completed to (0, oo), and the piece (1, oo)
        is mapped back by xi -> 1/xi."""
        tot = mp.mpc(0)
        for i1, i2, i3 in _multi(L - 1):                         # pole at xi = z
            if i2 > q:
                continue
            cf = ((-1) ** (L + i2) / (mp.factorial(i1) * mp.factorial(i2) * mp.factorial(i3))
                  * _ff(a2 - 1, i1) * _ff(q, i2) * mp.rf(n, i3))
            be, ga = a1 + a2 - 1 - i1 + i3, M + q - i2 - n - i3 + 1
            tot += cf * _beta2f1(n + i3, be, ga)
        for i1, i2, i3 in _multi(n - 1):                         # pole at xi = 1/z
            if i2 > q:
                continue
            cf = ((-1) ** (n + q + L + i3) / (mp.factorial(i1) * mp.factorial(i2) * mp.factorial(i3))
                  * _ff(a2 - 1, i1) * _ff(q, i2) * mp.rf(L, i3))
            be, ga = a1 - a2 + 1 + i1 - q + i2 + L + i3 - n, M + q - i2 - L - i3 + 1
            tot += cf * _beta2f1(L + i3, be, ga)
        tot *= -mp.pi / mp.sin(mp.pi * a2) * mp.exp(-sign * I * mp.pi * a2)
        at2 = L + n - q - a2
        tot -= (-1) ** (q + L + n) * mp.fsum(mp.binomial(q, j) * (-1) ** j * self.Ired(a1, at2 + j, M, n, L, -sign)
                                             for j in range(q + 1))
        return tot

    def I2gen(self, a1, a2, M, q, L, n, sign):
        if _is_int(a1 - a2):
            return self.Iq(a1, a2, M, q, L, n, sign)
        if _is_int(a1 + a2 + M):
            return self.I2refl(a1, a2, M, q, L, n, sign)
        raise ValueError("first indices are neither equal nor opposite modulo integers")

    # ------------------------------------------------------------------ the soft moment
    def J(self, ia, ib, q, K1, K2):
        """J(ia, ib, q; K1, K2) for K1 - K2 or K1 + K2 integer, ia, ib >= -1, and q a non-negative integer."""
        key = (ia, ib, q, _key(K1), _key(K2))
        if key in self._j:
            return self._j[key]
        a, b = ia + 1, ib + 1
        if a < 0 or b < 0 or q < 0:
            raise ValueError("closed form needs ia, ib >= -1 and q >= 0")
        nu = self.nu
        c1, c2 = HALF - nu - K1, HALF - nu - K2
        P = q + 1 - 2 * nu
        tot = mp.mpc(0)
        for (i, k), g in _uv_poly(a, b).items():
            for l in range(k + 1):
                for n in range(i + 1):
                    N = l + n + 2
                    W = ((-1) ** (l + n + 1) * mp.factorial(k) * mp.factorial(i)
                         / (mp.factorial(k - l) * mp.factorial(i - n)) / mp.rf(1 - P, N))
                    MN = 2 * nu - 1 + N
                    L, npr = l + 1, n + 1
                    I2 = self.I2gen(c1, c2, MN, q, L, npr, +1)
                    I1 = (-1) ** L * self.I2gen(c2, c1, MN, q, L, npr, -1)
                    tot += g * W * mp.mpf(2) ** (N - P) * (I1 - (-1) ** (k - l) * I2)
        val = mp.mpf(2) ** (1 - 2 * nu) * mp.gamma(P) * mp.pi / ((2 * mp.pi) ** 3 * mp.gamma(c1) * mp.gamma(c2)) * tot
        self._j[key] = val
        return val


def soft_J_closed(ia, ib, q, nu, K1, K2):
    """One soft moment (builds its own grid; use SoftClosed for many moments at the same nu)."""
    return SoftClosed(nu, kmax=max(abs(mp.im(K1)), abs(mp.im(K2)))).J(ia, ib, q, K1, K2)


def soft_J_closed_limit(ia, ib, q, nu, K1, K2, delta=None):
    """The same at a point where two pole ladders of the Barnes integrals coincide, for example K = -i kappa at
    kappa = mu, where 1/2 - nu - K is real. The first indices are moved by +-delta and +-2 delta along the line that
    keeps K1 - K2 (or K1 + K2) fixed, and the four values are combined to order delta^4."""
    if delta is None:
        delta = mp.mpf(10) ** (-(mp.mp.dps // 6))              # error of order delta^4; about dps/6 digits are lost
    same = _is_int(K1 - K2)
    vals = {}
    for j in (1, -1, 2, -2):
        e = j * delta * I
        vals[j] = soft_J_closed(ia, ib, q, nu, K1 + e, K2 + (e if same else -e))
    return (4 * (vals[1] + vals[-1]) - (vals[2] + vals[-2])) / 6


class ClosedSoftTable:
    """Closed-form stand-in for the soft-moment tables of ext/fermitri.py and ext/vectri.py.

    K: {label: first index}. Every pair of labels that is used must have K1 - K2 or K1 + K2 integer.
    """

    def __init__(self, nu, K, Tmax=None):
        self.nu, self.K = mp.mpc(nu), dict(K)
        self.closed = SoftClosed(nu, kmax=max(abs(mp.im(k)) for k in self.K.values()), Tmax=Tmax)

    def __call__(self, ia, ib, q, K1, K2):
        return self.closed.J(ia, ib, q, self.K[K1], self.K[K2])

    def load(self):
        return self

    def save(self):
        pass
