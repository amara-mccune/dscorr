"""The complete two-vertex loop (bubble) at every shape, as single sums over trees (Secs. 3, 6.3, 7.5).

Rotated wavefunction block of Eq. [eq:psin] with two lines of orders (n1, n2), vertex powers r1, r2 and
energies x1 > x2 >= k (the loop carries momentum k; in a bispectrum x2 = k = k3 and x1 = k12),

    P(r1, r2) = int dt1 dt2 t1^(r1-1) t2^(r2-1) e^(-x1 t1 - x2 t2) int d^3l/(2 pi) G_n1(y1) G_n2(y2).

The product identity [eq:product] turns the loop into trees of orders nu'_t = n1 + n2 + 3/2 + 2t with
weights 4 G_t, and each ordered tree is a product of two single-time moments plus an ordered remainder,

    T_nu(rho1, rho2) = Ihat_nu(rho1, x1) Khat_nu(rho2, x2)
                       + Gamma(rho2) (k/2)^(-rho2) sum_m (-1)^m (rho2+m) C_m^(rho2)(x2/k) Ihat_(rho2+m)(rho1, x1)
                                                          / (nu^2 - (rho2+m)^2),          rho_v = r_v - 3/2,

with C_m^(a) the Gegenbauer polynomials (the expansion of t^rho2 e^(-x2 t) in I_(rho2+m)(k t)). Hence

    P = sum_t 4 G_t Ihat Khat   [nonlocal and local families, A + F_0]
      + sum_m L_m S(rho2 + m)   [background, F_BG],   L_m = Gamma(rho2)(k/2)^(-rho2)(-1)^m (rho2+m) C_m Ihat_(rho2+m),

where S(z) = sum_t 4 G_t / (nu'_t^2 - z^2) is the tree sum of Eq. [eq:S0]. Both sums converge geometrically
for k <= x2 < x1, with ratio (x2 + sqrt(x2^2-k^2))/(x1 + sqrt(x1^2-k^2)); this resums the double series of
Eq. [eq:threefamilies] and holds on the bispectrum locus x2 = k, where that series is on the boundary of its
domain. S(z) is logarithmically divergent; with the constant of Eq. [eq:rung],

    P^MSbar(mu_R) = sum_t 4 G_t Ihat Khat + sum_m L_m [S_0(rho2+m) + 1/2 + (1/2) ln(mu_R/2)].

With additional factors 1/(nu'^2 - p^2) under the tree sum (the derivative form of a non-derivative vertex,
Sec. 7.5) every sum converges and no subtraction is needed.

The combination sum_eps eps1 eps2 S^eps(z) over the four sign patterns (the cut of the bubble) is closed,

    M(z) = sin(pi nu1) sin(pi nu2)/pi^2 prod_{+-+-} Gamma((3/2 + z +- nu1 +- nu2)/2) Gamma((3/2 - z +- nu1 +- nu2)/2)
           / [Gamma(3/2 + z) Gamma(3/2 - z)],

(the spectral density of the pair at imaginary argument), so the cut is a single sum of Gamma functions.
"""
import mpmath as mp
import numpy as np

PATTERNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_H = mp.mpf(3) / 2


# ----------------------------------------------------------------------------------- single-time moments
def Ihat(nu, rho, x, k):
    """int_0^oo dt t^(rho-1) e^(-x t) I_nu(k t), Eq. [eq:Ihat]."""
    a = rho + nu
    return ((k / 2) ** nu * mp.gamma(a) / (mp.gamma(1 + nu) * x ** a)
            * mp.hyp2f1(a / 2, (a + 1) / 2, 1 + nu, (k / x) ** 2))


def Khat(nu, rho, x, k):
    """int_0^oo dt t^(rho-1) e^(-x t) K_nu(k t); a ratio of Gamma functions at x = k."""
    if x == k:
        return mp.sqrt(mp.pi) * (2 * k) ** (-rho) * mp.gamma(rho + nu) * mp.gamma(rho - nu) / mp.gamma(rho + mp.mpf(1) / 2)
    return mp.pi / (2 * mp.sin(mp.pi * nu)) * (Ihat(-nu, rho, x, k) - Ihat(nu, rho, x, k))


def gegenbauer_row(a, X, M):
    """C_m^(a)(X) for m < M by the three-term recurrence (stable for X >= 1)."""
    out = [mp.mpf(1), 2 * a * X]
    for m in range(1, M - 1):
        out.append((2 * (m + a) * X * out[m] - (m + 2 * a - 1) * out[m - 1]) / (m + 1))
    return out[:M]


def ladder(rho1, rho2, k, x1, x2, M):
    """L_m for m < M: the functions that multiply the tree sums S(rho2 + m) in the background."""
    C = gegenbauer_row(rho2, x2 / k, M)
    pre = mp.gamma(rho2) * (k / 2) ** (-rho2)
    return [pre * (-1) ** m * (rho2 + m) * C[m] * Ihat(rho2 + m, rho1, x1, k) for m in range(M)]


def tree_remainder(nu, rho1, rho2, k, x1, x2, tol=None, Mmax=2000):
    """Ordered remainder of one tree, T_nu - Ihat Khat (single sum)."""
    tol = tol or mp.mpf(10) ** (-mp.mp.dps + 5)
    C0, C1 = mp.mpf(1), 2 * rho2 * x2 / k
    pre = mp.gamma(rho2) * (k / 2) ** (-rho2)
    tot, small = mp.mpc(0), 0
    for m in range(Mmax):
        if m == 0:
            C = C0
        elif m == 1:
            C = C1
        else:
            C = (2 * (m - 1 + rho2) * x2 / k * C1 - (m + 2 * rho2 - 2) * C0) / m
            C0, C1 = C1, C
        z = rho2 + m
        term = (-1) ** m * z * C * Ihat(z, rho1, x1, k) / (nu ** 2 - z ** 2)
        tot += term
        small = small + 1 if abs(term) < tol * abs(tot) else 0
        if small >= 3:
            break
    return pre * tot


def tree(nu, rho1, rho2, k, x1, x2, tol=None):
    """Ordered tree T_nu(rho1, rho2) = int t1^(rho1-1) t2^(rho2-1) e^(-x1 t1 - x2 t2) K_nu(k t>) I_nu(k t<)."""
    return Ihat(nu, rho1, x1, k) * Khat(nu, rho2, x2, k) + tree_remainder(nu, rho1, rho2, k, x1, x2, tol)


# ------------------------------------------------------------------------------------------ tree weights
def G0(n1, n2):
    """G_0 of Eq. [eq:product]."""
    nb, g = n1 + n2, mp.gamma
    return (g(_H) * g(_H + n1) * g(_H + n2) * g(_H + nb) * (_H + nb) * g(1 + nb)
            / (g(1 + n1) * g(1 + n2) * g(1 + nb) * g(3 + nb)))


def G_ratio(t, n1, n2):
    """G_(t+1)/G_t, rational in t (works on mpmath numbers and on numpy arrays)."""
    nb = n1 + n2
    h = 1.5
    return ((h + t) * (h + n1 + t) * (h + n2 + t) * (h + nb + t) * (h + nb + 2 * t + 2) * (1 + nb + 2 * t) * (2 + nb + 2 * t)
            / ((h + nb + 2 * t) * (t + 1) * (1 + n1 + t) * (1 + n2 + t) * (1 + nb + t) * (3 + nb + 2 * t) * (4 + nb + 2 * t)))


class Trees:
    """Tree weights and tree sums for two lines of orders nu1, nu2, for the four sign patterns."""

    def __init__(self, nu1, nu2, T=20000):
        self.nu1, self.nu2, self.T = mp.mpmathify(nu1), mp.mpmathify(nu2), T
        self._Gmp = {}
        self._tab = {}

    # mpmath weights (for the family sums)
    def orders(self, eps):
        return eps[0] * self.nu1, eps[1] * self.nu2

    def nup(self, eps, t):
        n1, n2 = self.orders(eps)
        return n1 + n2 + _H + 2 * t

    def G(self, eps, t):
        lst = self._Gmp.setdefault(eps, [])
        n1, n2 = self.orders(eps)
        if not lst:
            lst.append(G0(n1, n2))
        while len(lst) <= t:
            lst.append(lst[-1] * G_ratio(len(lst) - 1, n1, n2))
        return lst[t]

    # numpy tables (for the tree sums)
    def table(self, eps):
        if eps not in self._tab:
            n1, n2 = (complex(z) for z in self.orders(eps))
            t = np.arange(self.T, dtype=float)
            G = np.empty(self.T, dtype=complex)
            G[0] = complex(G0(*self.orders(eps)))
            G[1:] = G[0] * np.cumprod(G_ratio(t[:-1], n1, n2))
            self._tab[eps] = (4 * G, (n1 + n2 + 1.5 + 2 * t) ** 2)
        return self._tab[eps]

    def tsum(self, eps, z, poles=()):
        """sum_t 4 G_t / [(nu'^2 - z^2) prod_p (nu'^2 - p^2)], convergent if poles is not empty (float64)."""
        G4, n2 = self.table(eps)
        den = n2 - complex(z) ** 2
        for p in poles:
            den = den * (n2 - complex(p) ** 2)
        return complex(np.sum(G4 / den))

    def S0(self, eps, z):
        """S_0 of Eq. [eq:S0] by direct summation with Richardson extrapolation in 1/T (about 1e-11)."""
        G4, n2 = self.table(eps)
        a = G4 / (n2 - complex(z) ** 2)
        a = a - 0.5 / (np.arange(self.T) + 1.0)                 # removes the logarithm; adds gamma_E/2 below
        c = np.cumsum(a)
        Ts = [self.T // 8, self.T // 4, self.T // 2, self.T]
        v = [c[T - 1] for T in Ts]
        # tails are series in 1/T: eliminate 1/T, 1/T^2, 1/T^3
        for order in (1, 2, 3):
            v = [(2 ** order * v[i + 1] - v[i]) / (2 ** order - 1) for i in range(len(v) - 1)]
        return v[0] + 0.5 * np.euler_gamma

    # closed form of the cut
    def Mcut(self, z):
        """sum_eps eps1 eps2 sum_t 4 G_t^eps / (nu'^2 - z^2), closed form."""
        z = mp.mpmathify(z)
        num = mp.mpf(1)
        for a in (1, -1):
            for b in (1, -1):
                s = a * self.nu1 + b * self.nu2
                num *= mp.gamma((_H + z + s) / 2) * mp.gamma((_H - z + s) / 2)
        return (mp.sin(mp.pi * self.nu1) * mp.sin(mp.pi * self.nu2) / mp.pi ** 2
                * num / (mp.gamma(_H + z) * mp.gamma(_H - z)))

    def Mcut_prime(self, z):
        """dM/dz of the closed form (digamma functions)."""
        z = mp.mpmathify(z)
        d = -mp.digamma(_H + z) + mp.digamma(_H - z)
        for a in (1, -1):
            for b in (1, -1):
                s = a * self.nu1 + b * self.nu2
                d += (mp.digamma((_H + z + s) / 2) - mp.digamma((_H - z + s) / 2)) / 2
        return self.Mcut(z) * d

    def cut_sum(self, z, poles=()):
        """sum_eps eps1 eps2 tsum(eps, z, poles) by partial fractions of the closed form.

        If z^2 coincides with one of the poles (a double pole of the summand) the confluent form is used.
        """
        z = mp.mpmathify(z)
        pts = [z] + [mp.mpmathify(p) for p in poles]
        tiny = mp.mpf(10) ** (-mp.mp.dps + 10)
        dup = [j for j in range(1, len(pts)) if abs(pts[j] ** 2 - z ** 2) < tiny * abs(z) ** 2]
        if dup:
            rest = [pts[j] for j in range(1, len(pts)) if j not in dup]
            if len(dup) > 1 or len(rest) != 1:
                raise NotImplementedError("confluence of more than two points")
            A, B, b = z ** 2, rest[0] ** 2, rest[0]
            return self.Mcut_prime(z) / (2 * z * (A - B)) - (self.Mcut(z) - self.Mcut(b)) / (A - B) ** 2
        tot = mp.mpc(0)
        for i, a in enumerate(pts):
            den = mp.mpf(1)
            for j, b in enumerate(pts):
                if j != i:
                    den *= a ** 2 - b ** 2
            tot += self.Mcut(a) / den
        return tot


# ---------------------------------------------------------------------------------------- family sums
def family_sum(trees, eps, rho1, rho2, k, x1, x2, weight=None, tol=None, nmax=3000, terms=False):
    """sum_t 4 G_t w(nu'_t) Ihat_(nu'_t)(rho1, x1) Khat_(nu'_t)(rho2, x2): the families A + F_0 of one pattern."""
    tol = tol or mp.mpf(10) ** (-mp.mp.dps + 5)
    tot, small, lst = mp.mpc(0), 0, []
    big = mp.mpf(0)
    for t in range(nmax):
        nu = trees.nup(eps, t)
        term = 4 * trees.G(eps, t) * Ihat(nu, rho1, x1, k) * Khat(nu, rho2, x2, k)
        if weight is not None:
            term *= weight(nu)
        tot += term
        lst.append(term)
        big = max(big, abs(tot))
        small = small + 1 if abs(term) < tol * big else 0
        if small >= 3:
            break
    return (tot, lst) if terms else tot
