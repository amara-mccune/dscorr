"""The pair family of the scalar triangle with inflaton legs, at general shape (Sec. 5.2).

Triangle of Fig. 6: vertex v has energy k_v and power r_v, line A (order nuA) joins vertices 1 and 2, B joins
2 and 3, C joins 3 and 1, and k3 is the soft leg. For r3 = 2 the two first Barnes lemmas of Sec. 5.2 leave,
for every monomial of the loop momentum, only the first term of the Feynman-parameter form of the two-line
block (the term with j powers of the Laplacian carries 1/Gamma(r3 - 1 - j)). The soft loop momentum is then a
point lam k3 of the segment from 0 to k3, and the family of Eq. (5.4) is one parameter integral over
three-site chains, at every shape,

    psi_3  contains  pi^2 Gamma(2 + nuBC) Gamma(-1 - nuBC) k3 int_0^1 dlam [lam (1 - lam)]^(1/2)
                     int dt1 dt2 t1^(r1-1) t2^(r2-1) e^(-k1 t1 - k2 t2)
                     I_nuC(lam k3 t1) K_nuA(y t>) I_nuA(y t<) I_nuB((1 - lam) k3 t2),
    y^2 = |k1 + lam k3|^2 = (1 - lam) k1^2 + lam k2^2 - lam (1 - lam) k3^2,        nuBC = nuB + nuC.

The time integral converges for k3 < k1 + k2. In the variables

    K = (k1 + k2)/2,   R = k3/K,   delta = (k1 - k2)/k3      (k1 = K (1 + R delta/2), k2 = K (1 - R delta/2))

it expands as

    psi_3  contains  K^(1 - r1 - r2) R^(1 + nuBC) sum_N D_N(delta) R^N,

where D_N is a polynomial of degree <= N and parity (-1)^N in delta. Its coefficients are finite sums of
Gamma functions times the hard moments H_nuA of Eq. (B.4) at integer-shifted exponents, through the scaling
relation (B.6) for the line energy and the two vertex energies. All of these moments follow from the single
value H_nuA(r1 - 1 + nuC, r2 - 1 + nuB) by a first-order recurrence (see ScalarTriangle.H). With eps = R delta/2, u = y^2/K^2 - 1
= 2 (1 - 2 lam) eps + eps^2 - lam (1 - lam) R^2, and alpha = r1 - 1 + nuC + 2 i, beta = r2 - 1 + nuB + 2 j,

    D_N(delta) = pi^2 Gamma(2 + nuBC) Gamma(-1 - nuBC) 2^(-nuBC)
                 sum 4^(-i-j)/[i! j! Gamma(1 + nuC + i) Gamma(1 + nuB + j)]
                     n!/(n1! n2! n3!) 2^n1 (-1)^n3 (delta/2)^(s + n1 + 2 n2)  T_(s,n)  Lam(n1, n3),
    n = n1 + n2 + n3,      N = 2 i + 2 j + s + n1 + 2 n2 + 2 n3,
    T_(s,n)  = sum_(a+b=s) (-1)^a tau_n(alpha + a, beta + b)/(a! b!),
    tau_n(alpha, beta) = coefficient of u^n in y^(-alpha-beta-2) sum_(p0 p1) (1 - 1/y)^(p0+p1) H(alpha + p0, beta + p1)/(p0! p1!),
    Lam(n1, n3) = sum_c binom(n1, c) (-1)^c Beta(3/2 + nuC + 2 i + n3 + c, 3/2 + nuB + 2 j + n3 + n1 - c).

Exchanging the hard legs maps delta -> -delta together with (nuB, r2) <-> (nuC, r1). So D_N is even in delta
when the two soft lines and the two hard vertices are equal, and then only even N appear. On k1 = k2,
D_2N(0) is c_N of Eq. (5.6).
"""
import mpmath as mp

from .blocks import H_moment

__all__ = ["ScalarTriangle", "cN_isosceles", "in_in_weights", "correlator_family", "to_symmetric", "exchanged_angle"]

HALF = mp.mpf(1) / 2


def _binom_series(beta, N):
    c = [mp.mpc(1)]
    for j in range(1, N + 1):
        c.append(c[-1] * (beta - j + 1) / j)
    return c


def _conv(a, b, N):
    return [mp.fsum(a[i] * b[n - i] for i in range(n + 1)) for n in range(N + 1)]


class ScalarTriangle:
    """Coefficients D_N(delta) of the pair family of psi_3 at r3 = 2.

    nuA, nuB, nuC: orders of the three lines (nu = i mu on the principal series; any complex orders for which
    the moments converge). r1, r2: powers of the hard vertices (2 for inflaton legs). Nmax: highest order in R.
    recurrence: get the shifted hard moments from one seed (default) or each from Eq. (B.4).
    """

    def __init__(self, nuA, nuB, nuC, r1=2, r2=2, Nmax=4, recurrence=True):
        self.recurrence = recurrence
        self.nuA, self.nuB, self.nuC = (mp.mpmathify(x) for x in (nuA, nuB, nuC))
        self.r1, self.r2 = mp.mpmathify(r1), mp.mpmathify(r2)
        self.Nmax = Nmax
        self.nuBC = self.nuB + self.nuC
        self._H, self._g, self._c, self._tau, self._beta = {}, {}, {}, {}, {}
        self._coef = None
        self._w = None

    # ---- building blocks
    def H(self, m0, m1):
        """H_nuA(r1 - 1 + nuC + m0, r2 - 1 + nuB + m1).

        Only the seed (m0, m1) = (0, 0) is evaluated from Eq. (B.4), one 3F2 at unit argument. The shifted
        moments follow from the Bessel equation of the kernel, whose source is the unit jump,

            (2 a0 + 3) H(a0 + 1, a1) = [(a0 + 1)^2 - nu^2] H(a0, a1) + Gamma(a0 + a1 + 2)/2^(a0 + a1 + 2),

        and the same relation in a1 (the moment is symmetric). With recurrence=False every moment is evaluated
        from Eq. (B.4) instead (about a second each).
        """
        if (m0, m1) not in self._H:
            a0, a1 = self.r1 - 1 + self.nuC + m0, self.r2 - 1 + self.nuB + m1
            if (m0 == 0 and m1 == 0) or not self.recurrence:
                self._H[(m0, m1)] = H_moment(self.nuA, a0, a1)
            elif m0 > 0:
                b = a0 - 1
                self._H[(m0, m1)] = (((b + 1) ** 2 - self.nuA ** 2) * self.H(m0 - 1, m1)
                                     + mp.gamma(b + a1 + 2) / mp.mpf(2) ** (b + a1 + 2)) / (2 * b + 3)
            else:
                b = a1 - 1
                self._H[(m0, m1)] = (((b + 1) ** 2 - self.nuA ** 2) * self.H(m0, m1 - 1)
                                     + mp.gamma(a0 + b + 2) / mp.mpf(2) ** (a0 + b + 2)) / (2 * b + 3)
        return self._H[(m0, m1)]

    def prefactor(self):
        """pi^2 Gamma(2 + nuBC) Gamma(-1 - nuBC) 2^(-nuBC)"""
        return mp.pi ** 2 * mp.gamma(2 + self.nuBC) * mp.gamma(-1 - self.nuBC) * mp.mpf(2) ** (-self.nuBC)

    def g(self, m0, m1, P):
        """sum_(p0+p1=P) H(m0 + p0, m1 + p1)/(p0! p1!)"""
        if (m0, m1, P) not in self._g:
            self._g[(m0, m1, P)] = mp.fsum(self.H(m0 + p0, m1 + P - p0) / (mp.factorial(p0) * mp.factorial(P - p0))
                                           for p0 in range(P + 1))
        return self._g[(m0, m1, P)]

    def cPn(self, t):
        """c[P][n]: coefficient of u^n in (1 + u)^(-L/2) [1 - (1 + u)^(-1/2)]^P, L = r1 + r2 + nuBC + t"""
        if t not in self._c:
            N = self.Nmax
            if self._w is None:
                isq = _binom_series(-HALF, N)
                w = [mp.mpf(0)] + [-mp.re(c) for c in isq[1:]]
                wp = [[mp.mpf(1)] + [mp.mpf(0)] * N]
                for _ in range(N):
                    wp.append(_conv(wp[-1], w, N))
                self._w = wp
            pre = _binom_series(-(self.r1 + self.r2 + self.nuBC + t) / 2, N)
            self._c[t] = [_conv(pre, self._w[P], N) for P in range(N + 1)]
        return self._c[t]

    def tau(self, m0, m1, n):
        """coefficient of u^n of the hard moment at exponents shifted by (m0, m1), k1 = k2 = 1, y^2 = 1 + u"""
        if (m0, m1, n) not in self._tau:
            c = self.cPn(m0 + m1)
            self._tau[(m0, m1, n)] = mp.fsum(c[P][n] * self.g(m0, m1, P) for P in range(n + 1))
        return self._tau[(m0, m1, n)]

    def beta(self, a, b):
        """Beta(3/2 + nuC + a, 3/2 + nuB + b)"""
        if (a, b) not in self._beta:
            self._beta[(a, b)] = (mp.gamma(mp.mpf(3) / 2 + self.nuC + a) * mp.gamma(mp.mpf(3) / 2 + self.nuB + b)
                                  / mp.gamma(3 + self.nuBC + a + b))
        return self._beta[(a, b)]

    def _cij(self, i, j):
        return (mp.mpf(4) ** (-i - j) / (mp.factorial(i) * mp.factorial(j)
                                         * mp.gamma(1 + self.nuC + i) * mp.gamma(1 + self.nuB + j)))

    # ---- the series in (R, delta)
    def coefficients(self):
        """{(N, m): coefficient of R^N delta^m} (without the prefactor K^(1 - r1 - r2) R^(1 + nuBC))."""
        if self._coef is not None:
            return self._coef
        Nmax = self.Nmax
        acc = {}
        for i in range(Nmax // 2 + 1):
            for j in range(Nmax // 2 + 1 - i):
                M = Nmax - 2 * i - 2 * j
                cij = self._cij(i, j)
                for s in range(M + 1):
                    for n in range(M - s + 1):
                        T = mp.fsum((-1) ** a * self.tau(2 * i + a, 2 * j + s - a, n) / (mp.factorial(a) * mp.factorial(s - a))
                                    for a in range(s + 1))
                        for n1 in range(n + 1):
                            for n2 in range(n - n1 + 1):
                                n3 = n - n1 - n2
                                N = 2 * i + 2 * j + s + n1 + 2 * n2 + 2 * n3
                                if N > Nmax:
                                    continue
                                m = s + n1 + 2 * n2
                                lam = mp.fsum(mp.binomial(n1, c) * (-1) ** c * self.beta(2 * i + n3 + c, 2 * j + n3 + n1 - c)
                                              for c in range(n1 + 1))
                                mult = mp.factorial(n) / (mp.factorial(n1) * mp.factorial(n2) * mp.factorial(n3))
                                val = cij * mult * mp.mpf(2) ** (n1 - m) * (-1) ** n3 * T * lam
                                acc[(N, m)] = acc.get((N, m), 0) + val
        pf = self.prefactor()
        self._coef = {k: pf * v for k, v in sorted(acc.items()) if v != 0}
        return self._coef

    def isosceles(self, Nmax=None):
        """[c_0, ..., c_Nmax] of Eq. (5.6): the coefficients of (k3/k1)^(1 + nuBC + 2 N) on k1 = k2."""
        Nmax = self.Nmax // 2 if Nmax is None else Nmax
        assert 2 * Nmax <= self.Nmax
        pf = self.prefactor()
        out = []
        for N in range(Nmax + 1):
            tot = mp.mpc(0)
            for n in range(N + 1):
                for i in range(N - n + 1):
                    j = N - n - i
                    tot += self._cij(i, j) * self.beta(2 * i + n, 2 * j + n) * (-1) ** n * self.tau(2 * i, 2 * j, n)
            out.append(pf * tot)
        return out

    # ---- the same family in r = k3/k1 at fixed c = k1hat . k3hat (k2 = k1 sqrt(1 + 2 r c + r^2))
    def coefficients_angle(self):
        """{(n, mc): coefficient of r^n c^mc}, with psi_3 containing k1^(1 - r1 - r2) r^(1 + nuBC) sum C_n(c) r^n.

        Here y^2 = 1 + 2 lam r c + lam^2 r^2 and k2 = 1 + xi. The second expansion is not symmetric under the
        exchange of the hard legs, so the exchange identity tests it (see exchanged_angle).
        """
        Nmax = self.Nmax
        # (-xi)^b = (1 - sqrt(1 + v))^b = sum_p e[b][p] v^p,  v = 2 r c + r^2
        sq = _binom_series(HALF, Nmax)
        x = [mp.mpf(0)] + [-mp.re(c) for c in sq[1:]]
        e = [[mp.mpf(1)] + [mp.mpf(0)] * Nmax]
        for _ in range(Nmax):
            e.append(_conv(e[-1], x, Nmax))
        acc = {}
        for i in range(Nmax // 2 + 1):
            for j in range(Nmax // 2 + 1 - i):
                M = Nmax - 2 * i - 2 * j
                cij = self._cij(i, j)
                bj = [mp.binomial(2 * j, q) * (-1) ** q for q in range(2 * j + 1)]
                for b in range(M + 1):
                    for p in range(b, M + 1):                       # e[b][p] starts at p = b
                        if e[b][p] == 0:
                            continue
                        for n in range(M - p + 1):
                            base = cij * e[b][p] * self.tau(2 * i, 2 * j + b, n) / mp.factorial(b)
                            for p1 in range(p + 1):
                                p2 = p - p1
                                for n1 in range(n + 1):
                                    n2 = n - n1
                                    order = 2 * i + 2 * j + p1 + 2 * p2 + n1 + 2 * n2
                                    if order > Nmax:
                                        continue
                                    la = 2 * i + n1 + 2 * n2
                                    lam = mp.fsum(bj[q] * self.beta(la + q, 0) for q in range(2 * j + 1))
                                    val = (base * mp.binomial(p, p1) * mp.binomial(n, n1) * mp.mpf(2) ** (p1 + n1) * lam)
                                    key = (order, p1 + n1)
                                    acc[key] = acc.get(key, 0) + val
        pf = self.prefactor()
        return {k: pf * v for k, v in sorted(acc.items()) if v != 0}

    # ---- evaluation
    def at_delta(self, delta, coef=None):
        """[D_0(delta), ..., D_Nmax(delta)]"""
        coef = coef if coef is not None else self.coefficients()
        out = [mp.mpc(0)] * (self.Nmax + 1)
        for (n, m), v in coef.items():
            out[n] += v * mp.mpmathify(delta) ** m
        return out

    def series(self, R, delta, coef=None, Nmax=None):
        """sum_N D_N(delta) R^N (the family divided by K^(1 - r1 - r2) R^(1 + nuBC))"""
        D = self.at_delta(delta, coef)
        Nmax = self.Nmax if Nmax is None else Nmax
        return mp.fsum(D[n] * mp.mpmathify(R) ** n for n in range(Nmax + 1))

    def family(self, k1, k2, k3, coef=None, Nmax=None):
        """The pair family of psi_3 at the momenta (k1, k2, k3), from the series."""
        k1, k2, k3 = (mp.mpmathify(x) for x in (k1, k2, k3))
        K = (k1 + k2) / 2
        R, delta = k3 / K, (k1 - k2) / k3
        return K ** (1 - self.r1 - self.r2) * R ** (1 + self.nuBC) * self.series(R, delta, coef, Nmax)


# ------------------------------------------------------------------------------ changes of variables
def _pm(a, b, N):
    out = {}
    for (n1, m1), v1 in a.items():
        for (n2, m2), v2 in b.items():
            if n1 + n2 <= N:
                out[(n1 + n2, m1 + m2)] = out.get((n1 + n2, m1 + m2), 0) + v1 * v2
    return out


def _pa(*terms):
    out = {}
    for c, p in terms:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return out


def _binser(x, beta, N):
    out, xp, cj = {(0, 0): mp.mpc(1)}, {(0, 0): mp.mpc(1)}, mp.mpc(1)
    for j in range(1, N + 1):
        cj = cj * (beta - j + 1) / j
        xp = _pm(xp, x, N)
        out = _pa((1, out), (cj, xp))
    return out


def _compose(coef, cser, power_of, N):
    mcmax = max(k[1] for k in coef)
    cp = [{(0, 0): mp.mpc(1)}]
    for _ in range(mcmax):
        cp.append(_pm(cp[-1], cser, N))
    out = {}
    for n in range(N + 1):
        Cn = _pa(*[(val, cp[mc]) for (nn, mc), val in coef.items() if nn == n])
        if Cn:
            out = _pa((1, out), (1, _pm(_pm(Cn, power_of(n), N), {(n, 0): mp.mpc(1)}, N)))
    return {k: v for k, v in out.items() if v != 0}


def to_symmetric(coef, expo, N):
    """From k1^(-dim) r^e sum_n C_n(c) r^n to K^(-dim) R^e sum_N D_N(delta) R^N, with expo = dim + e.

    r = R/(1 + R delta/2), c = -(delta + R/2)/(1 + R delta/2), k1 = K (1 + R delta/2).
    """
    x = {(1, 1): mp.mpf(1) / 2}
    c = _pm({(0, 1): mp.mpf(-1), (1, 0): -mp.mpf(1) / 2}, _binser(x, -1, N), N)
    return _compose(coef, c, lambda n: _binser(x, -expo - n, N), N)


def exchanged_angle(coef, expo, N):
    """The series of the triangle with the hard legs exchanged, from that of the original one.

    If F(k1, k2, k3) = k1^(-dim) r^e sum_n C_n(c) r^n, this returns the coefficients of F(k2, k1, k3) in the same
    variables (x2 = k2/k1 = sqrt(1 + 2 r c + r^2), c2 = -(c + r)/x2). They must equal the coefficients computed
    with (nuB, r2) and (nuC, r1) exchanged. expo = dim + e.
    """
    v = {(1, 1): mp.mpf(2), (2, 0): mp.mpf(1)}
    c2 = _pm({(0, 1): mp.mpf(-1), (1, 0): mp.mpf(-1)}, _binser(v, -HALF, N), N)
    return _compose(coef, c2, lambda n: _binser(v, (-expo - n) / 2, N), N)


# ------------------------------------------------------------------------------ Eq. (5.6) and the in-in weights
def cN_isosceles(nuA, nuB, nuC, N, r1=2, r2=2):
    """c_N of Eq. (5.6): the coefficient of (k3/k1)^(1 + nuBC + 2 N) on k1 = k2, r3 = 2."""
    return ScalarTriangle(nuA, nuB, nuC, r1, r2, Nmax=2 * N).isosceles(N)[N]


def in_in_weights(mus):
    """Weights of the in-in assembly of Eq. (2.10) for three heavy lines:
    {(sA, sB, sC): W}, with the correlator's families given by -(1/8)(2 pi)^-3 sum W psi(sA nuA, sB nuB, sC nuC)."""
    import itertools
    cf = []
    for m in mus:
        s = mp.sinh(mp.pi * m)
        F = {1: mp.exp(mp.pi * m) / (2 * s), -1: -mp.exp(-mp.pi * m) / (2 * s)}
        X = {1: -1 / (2j * s), -1: 1 / (2j * s)}
        cf.append((F, X))
    W = {}
    for sg in itertools.product((1, -1), repeat=3):
        F = [cf[l][0][sg[l]] for l in range(3)]
        X = [cf[l][1][sg[l]] for l in range(3)]
        W[sg] = F[0] * F[1] * F[2] + F[0] * X[1] * X[2] + X[0] * F[1] * X[2] + X[0] * X[1] * F[2]
    return W


def correlator_family(mus, sB=1, sC=1, Nmax=4, r1=2, r2=2):
    """The pair (sB sC = +1) or difference (sB sC = -1) family of the in-in triangle at general shape.

    With S = (k1 k2 k3)^2 <pi pi pi>' (unit couplings, heavy lines mus = (muA, muB, muC)),

        S  contains  (1 - R^2 delta^2/4) 2 Re[ R^(2 + i (sB muB + sC muC)) sum_N A_N(delta) R^N ],

    where the first factor is k1 k2/K^2. Returns {(N, m): coefficient of R^N delta^m in A_N}. The assembly is
    that of Eq. (2.10): A_N = -[Z_N + conj(Y_N)]/(8 (2 pi)^3), with Z_N the sum over the sign of nuA of the
    weights times D_N(sA nuA, sB nuB, sC nuC), and Y_N the same at the opposite signs of nuB and nuC. On k1 = k2
    the leading coefficients are M_tri (pair) and D_tri (difference) of Ref. [26].
    """
    W = in_in_weights(mus)
    nu = [mp.mpc(0, m) for m in mus]
    out = {}
    for sA in (1, -1):
        Z = ScalarTriangle(sA * nu[0], sB * nu[1], sC * nu[2], r1, r2, Nmax).coefficients()
        Y = ScalarTriangle(sA * nu[0], -sB * nu[1], -sC * nu[2], r1, r2, Nmax).coefficients()
        for k, v in Z.items():
            out[k] = out.get(k, 0) + W[(sA, sB, sC)] * v
        for k, v in Y.items():
            out[k] = out.get(k, 0) + mp.conj(W[(sA, -sB, -sC)] * v)
    norm = -1 / (8 * (2 * mp.pi) ** 3)
    return {k: norm * v for k, v in sorted(out.items())}
