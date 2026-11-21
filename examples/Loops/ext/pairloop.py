"""Two-vertex loops of a heavy bilinear with spinning lines, at every shape (Secs. 4.1, 6.2, 6.3).

For a bilinear O that is a de Sitter scalar, the two lines enter a two-vertex loop only through the pair density
<O(x) O(x')>, a function of the invariant distance Z. For the bilinears of the Standard Model background the
density is a polynomial in the Laplacian acting on a product of two scalar-type lines,

    Ghat(Z) = w(Box) [g_n(Z)^2],     Box = (Z^2 - 1) d^2/dZ^2 + 4 Z d/dZ,     Box g_nu = (nu^2 - 9/4) g_nu,

with g_nu = -Q'_(nu-1/2) the line of Eq. [eq:gnu]. With M^2 = nu^2 - 9/4:

    F_mn F^mn, Proca line of order nu:   Ghat = 24 {(2 Z g' - M^2 g)^2 + 4 g'^2},              g = g_nu,
                                         w(P) = 4 P^2 - (16 M^2 + 24) P + 24 M^4 + 48 M^2,     n = nu;
    psibar psi, Dirac line (nu = i m):   Ghat = 4 [g+'^2 + g-'^2 - 2 Z g+' g-'],               g+- = g_(nu +- 1/2),
                                         w(P) = -2 [P - (2 nu - 1/2)^2 + 9/4],                 n = nu + 1/2;
    F_mn Ftilde^mn, Proca line:          Ghat = 96 g' (M^2 g - 2 Z g'),                         g = g_nu,
                                         w(P) = -4 (P + 2) [P - (2 nu - 1/2)^2 + 9/4] on g_nu g_(nu+1).

<F^2 F^2> = Ghat/(16 pi^4) and <psibar psi psibar psi> = -Ghat/(16 pi^4) (one real Proca field, one Dirac field).
The first follows from the field strengths of the Proca line,
    <F_mn F_r's'> = S [d_m d_r' u d_n d_s' u - (r' <-> s')] + T [d_m u d_r' u d_n d_s' u -+ 3 terms],
    S = -2 h',  T = -h'',  h = g_nu/(4 pi^2),  u = Z - 1,
the second from the trace of two Dirac propagators (the orders 1/2 -+ i mu of App. D).

Since each tree of the product identity [eq:product] is an eigenfunction of Box with the Casimir
P_t = nu'_t^2 - 9/4, the block of ext/bubble.py only acquires the weight w under its two sums,

    block = sum_t 4 G_t w(P_t) Ihat_t Khat_t  +  sum_m L_m w(zeta_m^2 - 9/4) [S_0(zeta_m) + 1/2 + (1/2) ln(mu_R/2)].

The second sum is the scalar MSbar bubble with w(Box) acting on the soft vertex. This is the reference scheme
of the scripts. Any other subtraction of the composite operator adds a polynomial of the degree of w in
zeta_m^2 under the second sum, that is, contact terms (three constants for F^2, two for psibar psi, one for
the scalar bilinear).

The direct check uses the invariant-distance form with the densities written through g and g' (not through w):
local terms are polynomials of degree <= 2 in k^2 at fixed vertex energies, so the third difference in k^2 is
scheme independent, and its invariant-distance integral converges at coincidence without subtraction.
"""
import mpmath as mp

from .bubble import Ihat, Khat, Trees, ladder

__all__ = ["w_scalar", "w_F2", "w_psi", "w_dphi2", "w_FFdual", "PairLoop", "g_and_dg", "dens_scalar", "dens_F2", "dens_psi",
           "dens_FFdual", "dens_dphi2", "kernel", "third_difference", "zside_third_difference", "S0_lattice_equal"]

_H = mp.mpf(1) / 2
_Q = mp.mpf(9) / 4


# --------------------------------------------------------------------------------------- weights
def w_scalar(nu=None):
    return [mp.mpf(1)]


def w_F2(nu):
    """Weight of F_mn F^mn for two Proca lines of order nu, as a polynomial in the Casimir P (lowest power first)."""
    M2 = mp.mpmathify(nu) ** 2 - _Q
    return [24 * M2 ** 2 + 48 * M2, -(16 * M2 + 24), mp.mpf(4)]


def w_psi(nu):
    """Weight of psibar psi for two Dirac lines (nu = i m), on the product of two scalar lines of order nu + 1/2."""
    nu = mp.mpmathify(nu)
    return [2 * ((2 * nu - _H) ** 2 - _Q), mp.mpf(-2)]


def w_dphi2(nu):
    """Weight of (d sigma)^2 = (1/2)(Box - 2 M^2) sigma^2 on the product of two scalar lines of order nu
    (so that <(d sigma)^2 (d sigma)^2> = 2 w(Box) g^2/(16 pi^4), as <sigma^2 sigma^2> = 2 g^2/(16 pi^4))."""
    M2 = mp.mpmathify(nu) ** 2 - _Q
    return [M2 ** 2, -M2, mp.mpf(1) / 4]


def w_FFdual(nu):
    """Weight of F_mn Ftilde^mn for two Proca lines of order nu, on the product g_nu g_(nu+1) of two scalar lines
    (trees nu'_t = 2 nu + 5/2 + 2 t, dimensions 4 + 2 nu + 2 t):  w(P) = -4 (P + 2) [P + 9/4 - (2 nu - 1/2)^2]."""
    nu = mp.mpmathify(nu)
    Ps = (2 * nu - _H) ** 2 - _Q
    return [8 * Ps, 4 * Ps - 8, mp.mpf(-4)]


def polyval(w, P):
    return sum(c * P ** j for j, c in enumerate(w))


# ------------------------------------------------------------------- S_0 on the half-integer lattice
def S0_lattice_equal(nu, mmax):
    """S_0(m + 1/2; nu, nu) for m <= mmax: Eqs. [eq:S0half], [eq:S0eq] at m = 0, 1, 2, then the recurrence [eq:S0rec]."""
    nu = mp.mpmathify(nu)
    om = nu ** 2 - mp.mpf(1) / 4
    psi = mp.digamma(nu + _H)
    Psi = mp.euler + psi
    cK = mp.euler / 2 + mp.log(2) / 2 - mp.mpf(1) / 4
    out = [-(psi + 1 - mp.log(2)) / 2,
           -psi / 2 - om / (4 * nu) * mp.polygamma(1, nu + _H) + mp.log(2) / 2 - mp.mpf(1) / 4,
           -psi / 2 - (om - 1) / (4 * om) + mp.log(2) / 2 - mp.mpf(1) / 4]
    for m in range(1, mmax - 1):
        z = _H + m
        w = (2 * z + 1) * (2 * z + 3)
        p1 = (z + 1.5) * (z + 2.5) * (z + 0.5) ** 2 * (z + 0.5 + 2 * nu) * (z + 0.5 - 2 * nu)
        p0 = -(z - 0.5) * (z + 0.5) * (z + 1.5) ** 2 * (z + 1.5 + 2 * nu) * (z + 1.5 - 2 * nu)
        rhs = om * (z + 1) * w * Psi - (z + 1) * w * (w - 8 * om) / 16 + cK * (p1 + p0)
        out.append((rhs - p0 * out[m]) / p1)
    return out[:mmax + 1]


# ------------------------------------------------------------------------------------- the loop
class PairLoop:
    """Two scalar-type lines of equal order n with the polynomial weight w (one sign pattern of the correlator).

    block(r1, r2, k, x1, x2) is the rotated block of ext/bubble.py with the weight, for k <= x2 < x1 and integer
    r2 >= 2, in the reference scheme of the module docstring at the scale mu_R.
    """

    def __init__(self, n, w, muR=2, mmax=300):
        self.n = mp.mpmathify(n)
        self.w = list(w)
        self.muR = mp.mpf(muR)
        self.tr = Trees(self.n, self.n)
        self.mmax = mmax
        with mp.workdps(mp.mp.dps + 15):
            self.S0 = S0_lattice_equal(self.n, mmax)
        self.cst = _H + mp.log(self.muR / 2) / 2

    def nup(self, t):
        return 2 * self.n + mp.mpf(3) / 2 + 2 * t

    def wt(self, z):
        return polyval(self.w, mp.mpmathify(z) ** 2 - _Q)

    def C(self, t):
        """Coefficient of the tree t in the pair density, Ghat = sum_t C_t g_(nu'_t)."""
        return 4 * self.tr.G((1, 1), t) * self.wt(self.nup(t))

    def families(self, r1, r2, k, x1, x2, tol=None, coef=None):
        """sum_t C_t Khat_t(r2 - 3/2, x2) Ihat_t(r1 - 3/2, x1): the nonlocal and local signals of the pattern.

        coef: {J: c_J} replaces Ihat_t(r1 - 3/2) by sum_J c_J Ihat_t(J - 3/2) (the hard vertex of Sec. 6.3)."""
        tol = tol or mp.mpf(10) ** (-mp.mp.dps + 6)
        coef = coef or {r1: 1}
        rho2 = r2 - mp.mpf(3) / 2
        tot, small, big = mp.mpc(0), 0, mp.mpf(0)
        for t in range(100000):
            nu = self.nup(t)
            term = self.C(t) * Khat(nu, rho2, x2, k) * sum(c * Ihat(nu, J - mp.mpf(3) / 2, x1, k) for J, c in coef.items())
            tot += term
            big = max(big, abs(tot))
            small = small + 1 if abs(term) < tol * big else 0
            if small >= 3:
                return tot
        raise RuntimeError("family sum did not converge")

    def background(self, r1, r2, k, x1, x2, tol=None, coef=None):
        """sum_m L_m w(zeta_m^2 - 9/4) [S_0(zeta_m) + 1/2 + (1/2) ln(mu_R/2)], zeta_m = r2 - 3/2 + m."""
        tol = tol or mp.mpf(10) ** (-mp.mp.dps + 6)
        coef = coef or {r1: 1}
        rho2 = r2 - mp.mpf(3) / 2
        off = int(r2) - 2
        for M in sorted({min(60 * 2 ** j, self.mmax - off) for j in range(12)}):
            Ls = {J: ladder(J - mp.mpf(3) / 2, rho2, k, x1, x2, M) for J in coef}
            tot, small = mp.mpc(0), 0
            for m in range(M):
                term = sum(c * Ls[J][m] for J, c in coef.items()) * (self.S0[m + off] + self.cst) * self.wt(rho2 + m)
                tot += term
                small = small + 1 if abs(term) < tol * abs(tot) else 0
                if small >= 3:
                    return tot
        raise RuntimeError("background ladder did not converge; raise mmax")

    def block(self, r1, r2, k, x1, x2, **kw):
        return self.families(r1, r2, k, x1, x2, **kw) + self.background(r1, r2, k, x1, x2, **kw)

    def contact(self, j, r1, r2, k, x1, x2, M=120):
        """Contact shape sum_m L_m zeta_m^(2 j), the local term that a change of scheme multiplies."""
        rho2 = r2 - mp.mpf(3) / 2
        L = ladder(r1 - mp.mpf(3) / 2, rho2, k, x1, x2, M)
        return mp.fsum(L[m] * (rho2 + m) ** (2 * j) for m in range(M))

    def series(self, N, r1, r2):
        """Coefficients e_0..e_N of the pair signal at k = x2 = 1: families = x1^-(r1 - 3/2 + nu'_0) sum_N e_N x1^(-2N)."""
        rho1, rho2 = r1 - mp.mpf(3) / 2, r2 - mp.mpf(3) / 2
        out = [mp.mpc(0)] * (N + 1)
        for t in range(N + 1):
            nu = self.nup(t)
            Kt = self.C(t) * Khat(nu, rho2, 1, 1)
            for p in range(N + 1 - t):
                out[t + p] += Kt * mp.mpf(2) ** (-nu - 2 * p) * mp.gamma(rho1 + nu + 2 * p) / (mp.factorial(p) * mp.gamma(1 + nu + p))
        return out


# ------------------------------------------------------------------ invariant-distance side
def g_and_dg(nu, Z):
    """g_nu(Z) = -Q'_(nu-1/2)(Z) and g'_nu(Z), from Q_lam, Q_(lam-1) and the Legendre equation."""
    lam = nu - _H
    q0, q1 = mp.legenq(lam, 0, Z, type=3), mp.legenq(lam - 1, 0, Z, type=3)
    w = Z * Z - 1
    g = -lam * (Z * q0 - q1) / w
    dg = -(lam * (lam + 1) * q0 + 2 * Z * g) / w
    return g, dg


def dens_scalar(nu, Z):
    return g_and_dg(nu, Z)[0] ** 2


def dens_F2(nu, Z):
    """24 {(2 Z g' - M^2 g)^2 + 4 g'^2}: 16 pi^4 <F^2 F^2> for one Proca field of order nu."""
    g, dg = g_and_dg(nu, Z)
    M2 = nu * nu - _Q
    return 24 * ((2 * Z * dg - M2 * g) ** 2 + 4 * dg * dg)


def dens_psi(nu, Z):
    """4 [g+'^2 + g-'^2 - 2 Z g+' g-'], g+- = g_(nu +- 1/2): -16 pi^4 <psibar psi psibar psi> for one Dirac field."""
    dp = g_and_dg(nu + _H, Z)[1]
    dm = g_and_dg(nu - _H, Z)[1]
    return 4 * (dp * dp + dm * dm - 2 * Z * dp * dm)


def dens_dphi2(nu, Z):
    """2 [(M^2 g - 3 Z g')^2 + 3 g'^2] = 2 grad_m grad_n' g grad^m grad^n' g: 16 pi^4 <(d sigma)^2 (d sigma)^2> for a real
    scalar of order nu (CWX, Eq. (4.18)); equal to 2 w_dphi2(Box) g^2."""
    g, dg = g_and_dg(nu, Z)
    return 2 * (((nu * nu - _Q) * g - 3 * Z * dg) ** 2 + 3 * dg * dg)


def dens_FFdual(nu, Z):
    """96 g' (M^2 g - 2 Z g'): 16 pi^4 <F Ftilde F Ftilde> for one Proca field of order nu."""
    g, dg = g_and_dg(nu, Z)
    return 96 * dg * ((nu * nu - _Q) * g - 2 * Z * dg)


def kernel(j1, e, k, x1, x2):
    """(-d/dx1)^j1 K(Z) at Z = 1 + e, with K of Eq. [eq:Zform] (j1 = 0, 1, 2: vertex powers r1 = 2 + j1, r2 = 2)."""
    E = x1 + x2
    D = E * E + 2 * x1 * x2 * e + k * k * e * (2 + e)
    s = mp.sqrt(D)
    N = k * s + k * k * (1 + e) + x1 * x2
    if j1 == 0:
        num = k * (2 * x1 * x2 * e + k * k * e * (2 + e)) / (s + E) + k * k * e      # N - (x1 + k)(x2 + k), no cancellation
        return mp.log1p(num / ((x1 + k) * (x2 + k))) / (2 * k)
    sx = (E + x2 * e) / s
    Nx = k * sx + x2
    if j1 == 1:
        return -(Nx / N - 1 / (x1 + k)) / (2 * k)
    if j1 == 2:
        sxx = (1 - sx * sx) / s
        return (k * sxx / N - (Nx / N) ** 2 + 1 / (x1 + k) ** 2) / (2 * k)
    raise NotImplementedError("j1 <= 2")


def third_difference(f, K0, h):
    """f(K0) - 3 f(K0 - h) + 3 f(K0 - 2h) - f(K0 - 3h) in K = k^2 (f takes k)."""
    return f(mp.sqrt(K0)) - 3 * f(mp.sqrt(K0 - h)) + 3 * f(mp.sqrt(K0 - 2 * h)) - f(mp.sqrt(K0 - 3 * h))


def zside_third_difference(dens, r1, x1, x2, K0, h, emin=mp.mpf('1e-7')):
    """2 int_1^oo dZ dens(Z) D3[(-d/dx1)^(r1-2) K(Z)]: the third difference in k^2 of the block P(r1, 2).

    The kernel difference vanishes as (Z - 1)^4, so the integrand is finite at coincidence (with logarithms). The
    difference is taken with 30 extra digits; below Z - 1 = emin the integrand is fitted to
    a ln^2 e + b ln e + c + e (a' ln^2 e + b' ln e + c') and integrated in closed form (a 1e-9 part of the result
    for F^2, with a relative error of order emin).
    """
    j1 = r1 - 2

    def F(e):
        with mp.workdps(mp.mp.dps + 30):
            dk = third_difference(lambda k: kernel(j1, e, k, x1, x2), K0, h)
        return 2 * dens(1 + e) * dk

    pts = [emin, mp.mpf('1e-5'), mp.mpf('1e-3'), mp.mpf('0.03'), mp.mpf('0.3'), 1, 3, 10, 40, 200, 2000, mp.inf]
    main = mp.quad(F, pts)
    es = [emin * f for f in (1, 2, 4, 8, 16, 32)]
    A, b = mp.matrix(6, 6), mp.matrix(6, 1)
    for i, e in enumerate(es):
        l = mp.log(e)
        for j, v in enumerate((l * l, l, 1, e * l * l, e * l, e)):
            A[i, j] = v
        b[i] = F(e)
    c = mp.lu_solve(A, b)
    l = mp.log(emin)
    near = (c[0] * emin * (l * l - 2 * l + 2) + c[1] * emin * (l - 1) + c[2] * emin
            + emin ** 2 * (c[3] * (l * l / 2 - l / 2 + mp.mpf(1) / 4) + c[4] * (l / 2 - mp.mpf(1) / 4) + c[5] / 2))
    return main + near
