"""Charged scalars with a chemical potential on the vertex (Sec. 7.5).

Model (Bodas, Broadberry, Sundrum, Xu, arXiv:2507.22978, Eq. (3.7)): after the field redefinition
chi_I -> chi_I e^(i phi/2 Lambda),

    L  >  - alpha chi1 chi2 e^(i lam t) e^(i dphi/Lambda) + h.c.,     lam = phidot_0/Lambda,

with lines that are ordinary massive scalars of mass parameters mu1, mu2 (H = 1). The bispectrum at
O(|alpha|^2) is a bubble with a quartic vertex (legs k1, k2) and a cubic vertex (leg k3). Its two vertices
carry e^(+-i lam t) = (-tau)^(-+i lam), a complex shift of the vertex powers.

In the squeezed limit the coefficient of each nonlocal family is the hard vertex (a Gamma function, with
the soft lines at their late-time ends) times the soft moment of Eq. [eq:twomoment] at the complex soft
powers r2 = i lam and 1 + i lam. We quote it in the normalization of BBSX:

    F = (5/6) <RRR>' / (P_R P_R + 2 perms) = -(10/3) phidot_0 (k1 k2 k3)^3 / (k1^3 + k2^3 + k3^3) <dphi^3>',
    F  >  f p^(-3 - i(lam - s1 mu1 - s2 mu2)) + c.c.,     p = (k1 + k2)/k3,   k1 = k2,

at |alpha|^2/Lambda^2 = 1. The signal of BBSX is the family s1 = s2 = +1, with amplitude f_oscil
(their Eq. (3.35), stationary phase plus threshold expansion of the spectral density).

Checks (Sec. 7.5):
  * late-time coefficients of the modes against mpmath's hankel1;
  * the rotated soft vertex (prefactor times K functions) against the conformal-time integral on a
    partially rotated ray, at fixed loop momentum;
  * reality: the coefficient of p^(-3 - i(lam - mu12)) is the conjugate of that of p^(-3 + i(lam - mu12));
  * comparison with BBSX at large lam and mu, near and below threshold;
  * the resonance lam = mu1 + mu2 (pole of the soft moment, residue R), and the difference line.

The complete bispectrum at every shape (families, background, three channels) is `ChargedShape`:

Usage:  python charged_scalars.py                              (squeezed limit, about 30 s)
        python charged_scalars.py --shape [--quick] [--json f]  (every shape, about 8 min; --quick 2 min)
"""
import json
import sys

import mpmath as mp

from ext.blocks import two_moment
from ext.bubble import Ihat, Khat, Trees
from ext.record import compare, report

mp.mp.dps = 50
I = mp.mpc(0, 1)


# ----------------------------------------------------------------------------------------- modes
def mode(mu, y, tau):
    """v_y(tau) = (sqrt(pi)/2) e^(-pi mu/2) (-tau)^(3/2) H1_{i mu}(-y tau) (phase dropped; it cancels)."""
    return mp.sqrt(mp.pi) / 2 * mp.exp(-mp.pi * mu / 2) * (-tau) ** 1.5 * mp.hankel1(I * mu, -y * tau)


def late_coeffs(mu):
    """v_y(tau) -> sum_s c_s (y/2)^(s i mu) (-tau)^(3/2 + s i mu) at late times."""
    pre = -I / (2 * mp.sqrt(mp.pi))
    return {+1: pre * mp.exp(mp.pi * mu / 2) * mp.gamma(-I * mu),
            -1: pre * mp.exp(-mp.pi * mu / 2) * mp.gamma(I * mu)}


def G_ext(b, k, tau):
    """Bulk-to-boundary function of the inflaton on branch b: G_+ = (1 - i k tau) e^(i k tau)/(2 k^3), G_- = conj."""
    return (1 - b * I * k * tau) * mp.exp(b * I * k * tau) / (2 * k ** 3)


class ChargedBubble:
    """Squeezed families of the charged-scalar bubble at masses mu1, mu2 and chemical potential lam."""

    def __init__(self, mu1, mu2, lam):
        self.mu1, self.mu2, self.lam = mp.mpf(mu1), mp.mpf(mu2), mp.mpf(lam)
        self.c1, self.c2 = late_coeffs(self.mu1), late_coeffs(self.mu2)

    # In the squeezed region tau1 (hard) is later than tau2 (soft). On every Schwinger-Keldysh sector the
    # line is then v(tau1) v*(tau2) if the soft vertex is on the + branch (b = +1) and v*(tau1) v(tau2) if
    # b = -1, so the hard end is v or v* at late times.
    def L(self, c, b, s):
        return c[s] if b == 1 else mp.conj(c[-s])

    @staticmethod
    def hard(z):
        """sum_a (-i a) int dtau1 (-tau1)^(z-1) G_a(k1, tau1) G_a(k2, tau1) at k1 = k2 = 1/2, times k1^3 k2^3.

        = -2 sin(pi z/2) [Gamma(z) + Gamma(z+1) + (k1 k2/k12^2) Gamma(z+2)] / 4,  k12 = 1.
        """
        return -2 * mp.sin(mp.pi * z / 2) * (mp.gamma(z) + mp.gamma(z + 1) + mp.gamma(z + 2) / 4) / 4

    def soft(self, b, s1, s2, lam):
        """S_b = int dtau2 (-tau2)^(-4 + i lam) G_b(k3) int d^3l/(2pi)^3 (y1/2)^(s1 i mu1) (y2/2)^(s2 i mu2) w_b w_b,
        at k3 = 1, with w_+ = v*, w_- = v. Rotating tau2 = i t (b = +) gives K functions and Eq. [eq:twomoment]:
            S_+ = -e^(pi lam/2) / (2 pi (2 pi)^3) [M(r2 = i lam) + M(r2 = 1 + i lam)],
        and S_-(lam, s) = conj(S_+(-lam, -s)).
        """
        if b == -1:
            return mp.conj(self.soft(1, -s1, -s2, -lam))
        n1, n2 = I * self.mu1, I * self.mu2
        M = two_moment(n1, n2, s1, s2, I * lam) + two_moment(n1, n2, s1, s2, 1 + I * lam)
        return -mp.exp(mp.pi * lam / 2) / (2 * mp.pi * (2 * mp.pi) ** 3) * M

    def family(self, sign, s1, s2):
        """Coefficient of p^(-3 + i(sign lam - s1 mu1 - s2 mu2)) in F (|alpha|^2/Lambda^2 = 1).

        B = (i |alpha|^2/Lambda^3) [I(lam) - I(-lam)]: the vertex chi1 chi2 e^(i lam t) (quartic) with
        chibar1 chibar2 e^(-i lam t) (cubic), minus the conjugate assignment. sign = +1 is I(lam).
        """
        lam = sign * self.lam
        z = -I * lam + I * (s1 * self.mu1 + s2 * self.mu2)
        tot = 0
        for b in (1, -1):
            tot += (-I * b) * self.L(self.c1, b, s1) * self.L(self.c2, b, s2) * self.soft(b, s1, s2, lam)
        IB = self.hard(z) * tot                      # times (k12/k3)^(i lam - Sigma) / (k1 k2)^3
        Bcoef = sign * I * IB                        # times |alpha|^2/Lambda^3
        return -mp.mpf(40) / 3 * self.lam * Bcoef    # F = -(40/3) phidot0 p^-3 k1^3 k2^3 B, phidot0 = lam Lambda

    def f(self):
        """BBSX's signal amplitude: coefficient of p^(-3 - i(lam - mu12))."""
        return self.family(-1, -1, -1)

    def residue(self):
        """Residue R of f at the resonance lam = mu1 + mu2, in the variable b = i(mu12 - lam): f ~ R/b."""
        mu12 = self.mu1 + self.mu2
        # f is meromorphic in lam near mu12; b f(b) at b -> 0 by a symmetric limit (Richardson in eps)
        def bf(eps):
            B = ChargedBubble(self.mu1, self.mu2, mu12 + eps)
            return I * (-eps) * B.f()
        e = mp.mpf(10) ** -12
        return (bf(e) + bf(-e)) / 2


def f_bbsx(mu1, mu2, lam):
    """BBSX Eq. (3.35): stationary phase plus threshold, |alpha|^2/Lambda^2 = 1."""
    mu1, mu2, lam = mp.mpf(mu1), mp.mpf(mu2), mp.mpf(lam)
    m = mu1 + mu2
    th = lam * mp.log(2 * lam) + m * mp.log(2 * m) - (lam + m) * mp.log(lam + m)
    return (-5 / (12 * mp.sqrt(mp.pi)) * lam ** 1.5 * m ** -0.5 / (lam ** 2 - m ** 2)
            * mp.sqrt(mu1 * mu2 / m) * (2 * m / (lam + m)) ** -1.5 * mp.exp(I * (th + 3 * mp.pi / 4)))


# ------------------------------------------------------------------------------------- checks
def check_late(mu):
    """Late-time expansion of the mode against hankel1 at y(-tau) = 1e-6 (relative error)."""
    c, y, tau = late_coeffs(mp.mpf(mu)), mp.mpf(1), -mp.mpf(10) ** -6
    approx = sum(c[s] * (y / 2) ** (s * I * mu) * (-tau) ** (1.5 + s * I * mu) for s in (1, -1))
    return abs(approx / mode(mp.mpf(mu), y, tau) - 1)


def check_rotation(mu1, mu2, lam, y1, y2, shift=mp.mpf('0.7'), theta=mp.mpf('0.4')):
    """Fixed loop momentum: the conformal-time soft vertex on the ray tau = -x e^(-i theta) (b = +) against
    the rotated form -e^(pi lam/2)/(2 pi) int dt t^(-1+i lam+shift) e^(-pi i shift/2)... with K functions.

    The vertex power is shifted by `shift` so that the integral converges absolutely at tau -> 0 (the
    identity is the same at every power). Returns the relative difference.
    """
    mu1, mu2, lam = mp.mpf(mu1), mp.mpf(mu2), mp.mpf(lam)
    r = I * lam + shift
    def f_eta(x):
        tau = -x * mp.exp(-I * theta)
        w1 = mp.conj(mode(mu1, y1, mp.conj(tau)))          # v* continued off the real axis (Schwarz reflection)
        w2 = mp.conj(mode(mu2, y2, mp.conj(tau)))
        return (-tau) ** (r - 4) * G_ext(1, 1, tau) * w1 * w2 * mp.exp(-I * theta)
    eta = mp.quad(f_eta, [0, 0.5, 2, 6, 15, 35, 80])        # integrand ~ e^(-E x sin(theta)), E > 2
    # rotated: tau = i t; v* = e^(-i pi/4) t^(3/2) K_{i mu}(y t)/sqrt(pi); (-tau)^q = t^q e^(-i pi q/2)
    g = mp.quad(lambda t: t ** (r - 1) * (1 + t) * mp.exp(-t) * mp.besselk(I * mu1, y1 * t)
                * mp.besselk(I * mu2, y2 * t), [0, 0.5, 2, 6, 15, 35, 80])
    rot = -I * mp.exp(-I * mp.pi * (r - 4) / 2) * mp.exp(-I * mp.pi / 2) / mp.pi / 2 * g
    return abs(eta / rot - 1)


# ---------------------------------------------------------------------- the complete bispectrum (--shape)
PATTERNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class ChargedShape:
    """The complete bispectrum of the charged-scalar loop at every shape, at O(|alpha|^2).

    Vertex phases. On the rotated contour the two vertices carry t^(-+i lam), i.e. r1 = j - i lam (j = 0, 1, 2
    with coefficients 1, k12, k1 k2) and r2 = j' + i lam (j' = 0, 1 with coefficients 1, k3). The sum over the
    Schwinger-Keldysh branches gives, for the sign pattern eps of the two lines (M_eps = eps1 mu1 + eps2 mu2),

        w_eps = eps1 eps2 sinh[pi (M_eps - lam)/2] sinh[pi (M_eps + lam)/2] / (sinh pi mu1 sinh pi mu2)
              = 1/2 + eps1 eps2 beta,      beta = (cosh pi mu1 cosh pi mu2 - cosh pi lam)/(2 sinh pi mu1 sinh pi mu2),

    (the sines of Eq. [eq:patterns] with their arguments shifted by the vertex phase), and

        F(k1 k2 | k3) = 5 lam / [24 pi^2 (k1^3 + k2^3 + k3^3)]  Im sum_eps w_eps P_eps(lam),

    in the normalization of BBSX at |alpha|^2/Lambda^2 = 1. The logarithmic divergence of the loop is real
    and drops out. The bispectrum is the sum over the three channels.

    Derivative form. The non-derivative vertices have R = j + j' - 3 <= 0, so the background of a single
    diagram has the poles Gamma(R + m) (late-time divergences). They cancel in the sum of diagrams by the
    shift symmetry. Tree by tree, the contiguous relation of the single-time moments,
        (nu^2 - rho^2) V_rho = -(2 rho + 1) x V_(rho+1) + (x^2 - k^2) V_(rho+2)   (+ contact terms),
    trades them for derivative vertices (those of BBSX's Eq. (4.9)) and no contact term remains:

        P_eps -> -2 i lam k3^2 sum_t 4 G_t w(nu'_t) sum_{J=2,3,4} H_J T_(nu'_t)(J - i lam - 3/2, 1/2 + i lam),
        w(nu) = 1/[(nu^2 - (3/2 + i lam)^2)(nu^2 - (3/2 - i lam)^2)],
        H_2 = 2 i lam (k1^2 + k2^2) - 2 k1.k2,  H_3 = k12 (2 i lam k1 k2 - 2 k1.k2),  H_4 = 2 k1 k2 (k1 k2 - k1.k2).

    Every sum then converges, and the result is finite and scheme independent. With ext/bubble.py it is two
    single sums: the families (one sum per pattern) and the background on the ladder zeta_m = 1/2 + i lam + m,
    whose coefficients are (1/2) sum_eps S_w^eps(zeta_m) (four convergent tree sums) plus beta times the cut,
    which is a closed form in Gamma functions.
    """

    def __init__(self, mu1, mu2, lam, dps=None, T=20000):
        self.dps = dps or 40 + int(1.5 * abs(float(lam)))
        with mp.workdps(self.dps):
            self.mu1, self.mu2, self.lam = mp.mpf(mu1), mp.mpf(mu2), mp.mpf(lam)
            mu1, mu2, lam = self.mu1, self.mu2, self.lam
            self.tr = Trees(I * mu1, I * mu2, T=T)
            self.rho0, self.rho0p = -mp.mpf(3) / 2 - I * lam, -mp.mpf(3) / 2 + I * lam
            self.rho2 = mp.mpf(1) / 2 + I * lam
            s1, s2 = mp.sinh(mp.pi * mu1), mp.sinh(mp.pi * mu2)
            self.wts = {}
            for e in PATTERNS:
                M = e[0] * mu1 + e[1] * mu2
                self.wts[e] = e[0] * e[1] * mp.sinh(mp.pi * (M - lam) / 2) * mp.sinh(mp.pi * (M + lam) / 2) / (s1 * s2)
            self.beta = (mp.cosh(mp.pi * mu1) * mp.cosh(mp.pi * mu2) - mp.cosh(mp.pi * lam)) / (2 * s1 * s2)
        self._coef = []

    def w(self, nu):
        return 1 / ((nu ** 2 - self.rho0 ** 2) * (nu ** 2 - self.rho0p ** 2))

    def coef(self, m):
        """Background coefficients on the ladder: ((1/2) sum_eps S_w^eps, beta * cut) at zeta_m."""
        while len(self._coef) <= m:
            z = self.rho2 + len(self._coef)
            poles = (self.rho0, self.rho0p)
            plus = sum(self.tr.tsum(e, z, poles) for e in PATTERNS)
            self._coef.append((mp.mpmathify(plus) / 2, self.beta * self.tr.cut_sum(z, poles)))
        return self._coef[m]

    def Hs(self, k1, k2, k3):
        lam, x = self.lam, k1 + k2
        d = x ** 2 - k3 ** 2                                  # = 2 (k1 k2 - k1.k2)
        return {2: -(2 + 4 * I * lam) * k1 * k2 + d + 2 * I * lam * x ** 2,
                3: x * (d - (2 - 2 * I * lam) * k1 * k2),
                4: d * k1 * k2}

    def pref(self, k1, k2, k3):
        return 5 * self.lam / (24 * mp.pi ** 2 * (k1 ** 3 + k2 ** 3 + k3 ** 3))

    def X(self, k1, k2, k3, tol=None, parts=False, only=None):
        """sum_eps w_eps P_eps(lam) for the channel (k1 k2 | k3), in the derivative form.

        only = eps returns the family sum of that pattern alone (with its weight), without the background.
        """
        with mp.workdps(self.dps):
            k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
            tol = tol or mp.mpf(10) ** -25
            lam, x1, k = self.lam, k1 + k2, k3
            H = self.Hs(k1, k2, k3)
            rho = {J: J - I * lam - mp.mpf(3) / 2 for J in (2, 3, 4)}
            sig, nterms = {}, 0
            for e in (PATTERNS if only is None else (only,)):
                tot, small, big = mp.mpc(0), 0, mp.mpf(0)
                for t in range(100000):
                    nu = self.tr.nup(e, t)
                    term = (4 * self.tr.G(e, t) * self.w(nu) * Khat(nu, self.rho2, k, k)
                            * sum(H[J] * Ihat(nu, rho[J], x1, k) for J in (2, 3, 4)))
                    tot += term
                    big = max(big, abs(tot))
                    small = small + 1 if abs(term) < tol * big else 0
                    if small >= 3:
                        break
                sig[e] = tot
                nterms = max(nterms, t)
            if only is not None:
                return -2 * I * lam * k3 ** 2 * self.wts[only] * sig[only]
            pre = mp.gamma(self.rho2) * (k / 2) ** (-self.rho2)
            bg_plus, bg_cut, small, C = mp.mpc(0), mp.mpc(0), 0, mp.mpf(1)
            for m in range(100000):
                if m > 0:
                    C *= (2 * self.rho2 + m - 1) / m          # Gegenbauer C_m^(rho2)(1)
                z = self.rho2 + m
                L = pre * (-1) ** m * z * C * sum(H[J] * Ihat(z, rho[J], x1, k) for J in (2, 3, 4))
                cp, cc = self.coef(m)
                bg_plus += L * cp
                bg_cut += L * cc
                small = small + 1 if abs(L) * (abs(cp) + abs(cc)) < tol * max(abs(bg_plus), abs(bg_cut)) else 0
                if small >= 3:
                    break
            fac = -2 * I * lam * k3 ** 2
            if parts:
                return {'families': {e: fac * self.wts[e] * sig[e] for e in PATTERNS},
                        'bg_ordered': fac * bg_plus, 'bg_cut': fac * bg_cut, 'terms': (nterms, m)}
            return fac * (sum(self.wts[e] * sig[e] for e in PATTERNS) + bg_plus + bg_cut)

    def channel(self, k1, k2, k3, tol=None):
        """F of the channel (k1 k2 | k3) and its pieces: {'F', 'families': {eps: F}, 'background', 'Ftilde'}.

        'Ftilde' is the complex sum-line amplitude, F > Ftilde + c.c. with Ftilde ~ f p^(-3 - i(lam - mu12)).
        """
        with mp.workdps(self.dps):
            P = self.X(k1, k2, k3, tol=tol, parts=True)
            pre = self.pref(mp.mpf(k1), mp.mpf(k2), mp.mpf(k3))
            fam = {e: pre * mp.im(v) for e, v in P['families'].items()}
            bg = pre * mp.im(P['bg_ordered'] + P['bg_cut'])
            return {'F': sum(fam.values()) + bg, 'families': fam, 'background': bg,
                    'bg_ordered': pre * mp.im(P['bg_ordered']), 'bg_cut': pre * mp.im(P['bg_cut']),
                    'Ftilde': mp.conj(pre * P['families'][(1, 1)] / (2 * I)), 'terms': P['terms']}

    def F(self, k1, k2, k3, tol=None):
        """The complete bispectrum: the three channels."""
        return (self.channel(k1, k2, k3, tol)['F'] + self.channel(k2, k3, k1, tol)['F']
                + self.channel(k3, k1, k2, tol)['F'])

    def family_leading(self, e, k1=mp.mpf(1) / 2, k2=mp.mpf(1) / 2):
        """Leading squeezed coefficient of p^(-3 + i lam - i M_eps) (closed form, from the tree t = 0), at k12 = 1.

        At k1 = k2 it equals ChargedBubble.family(+1, eps1, eps2); the signal of BBSX is its conjugate at eps = (+, +).
        """
        with mp.workdps(self.dps):
            lam = self.lam
            H = self.Hs(mp.mpf(k1), mp.mpf(k2), mp.mpf(0))
            nu = self.tr.nup(e, 0)
            hard = sum(H[J] * mp.gamma(J - I * lam - mp.mpf(3) / 2 + nu) for J in (2, 3, 4)) / (2 ** nu * mp.gamma(1 + nu))
            soft = (mp.sqrt(mp.pi) * 2 ** (-self.rho2) * mp.gamma(self.rho2 + nu) * mp.gamma(self.rho2 - nu)
                    / mp.gamma(self.rho2 + mp.mpf(1) / 2))
            Xe = -2 * I * lam * self.wts[e] * 4 * self.tr.G(e, 0) * self.w(nu) * hard * soft
            return 5 * lam / (24 * mp.pi ** 2 * (k1 ** 3 + k2 ** 3)) * Xe / (2 * I)


def rho_pair(m, nu1, nu2):
    """Spectral density of the pair with the measure d^3l/(2 pi): the loop of two cut lines is
    (t1 t2)^(-3/2) int_0^oo dm rho(m) K_{im}(k t1) K_{im}(k t2)/pi. It tends to m/2 at large m."""
    num = mp.mpf(1)
    for a in (1, -1):
        for b in (1, -1):
            for c in (1, -1):
                num *= mp.gamma((mp.mpf(3) / 2 + c * I * m + a * nu1 + b * nu2) / 2)
    return m * mp.sinh(mp.pi * m) / (2 * mp.pi ** 3) * num / (mp.gamma(1.5 + I * m) * mp.gamma(1.5 - I * m))


def spectral_X(S, k1, k2, k3, tol=mp.mpf(10) ** -22):
    """Direct evaluation of ChargedShape.X by the spectral integral over tree exchanges (independent of the
    product identity, of the tree sums, and of the closed form of the cut).

    On every Schwinger-Keldysh branch the pair propagator is int dm rho(m) G_m, so
        X = int_0^oo dm rho(m) sum_{eps=+-} [-1 + i eps cosh(pi lam)/sinh(pi m)] T^D_(eps i m),
    with T^D the tree in the derivative form. The two moments are integrated with Gauss-Legendre panels; the
    ordered remainder depends on m only through 1/(m^2 + zeta^2), so its integral is done once per ladder point.
    """
    import numpy as np
    with mp.workdps(S.dps):
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        lam, mu1, mu2 = S.lam, S.mu1, S.mu2
        nu1, nu2 = I * mu1, I * mu2
        x1, k = k1 + k2, k3
        H = S.Hs(k1, k2, k3)
        rho = {J: J - I * lam - mp.mpf(3) / 2 for J in (2, 3, 4)}
        ch = mp.cosh(mp.pi * lam)

        def f(m):
            if m == 0:
                return mp.mpc(0)
            nu = I * m
            Ip = sum(H[J] * Ihat(nu, rho[J], x1, k) for J in (2, 3, 4))
            Im = sum(H[J] * Ihat(-nu, rho[J], x1, k) for J in (2, 3, 4))
            return (rho_pair(m, nu1, nu2) * S.w(nu) * Khat(nu, S.rho2, k, k)
                    * (ch * (Im - Ip) / (I * mp.sinh(mp.pi * m)) - (Ip + Im)))

        step = min(mp.mpf(2), 2 * mp.pi / abs(mp.log(k / x1)) / 1.2)
        pts = [mp.mpf(0)]
        while pts[-1] < lam + mu1 + mu2 + 28:
            pts.append(pts[-1] + step)
        moments = mp.quad(f, pts, method='gauss-legendre')
        # ordered remainder: 2 int dm rho(m) w(i m)/(m^2 + zeta^2) on fixed panels (float64), tail from rho -> m/2
        edges = np.concatenate([np.arange(0, 60, 0.25), np.arange(60, 400, 2.0), np.geomspace(400, 2.0e5, 240)])
        xg, wg = np.polynomial.legendre.leggauss(24)
        mid, hw = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        mm = (mid[:, None] + hw[:, None] * xg[None, :]).ravel()
        ww = (hw[:, None] * wg[None, :]).ravel()
        with mp.workdps(20):
            dens = np.array([float(mp.re(rho_pair(mp.mpf(float(v)), nu1, nu2))) for v in mm])
        r0, r0p = complex(S.rho0), complex(S.rho0p)
        base = 2 * dens / ((mm ** 2 + r0 ** 2) * (mm ** 2 + r0p ** 2)) * ww
        pre = mp.gamma(S.rho2) * (k / 2) ** (-S.rho2)
        C, bg, small = mp.mpf(1), mp.mpc(0), 0
        for m in range(100000):
            if m > 0:
                C *= (2 * S.rho2 + m - 1) / m
            z = S.rho2 + m
            L = pre * (-1) ** m * z * C * sum(H[J] * Ihat(z, rho[J], x1, k) for J in (2, 3, 4))
            cm = complex(np.sum(base / (mm ** 2 + complex(z) ** 2))) + 1.0 / (4 * edges[-1] ** 4)
            bg += L * cm
            small = small + 1 if abs(L * cm) < tol * abs(bg) else 0
            if small >= 3:
                break
        return -2 * I * lam * k3 ** 2 * (moments + bg)


def f_bkg_bbsx(mu1, mu2, lam, k1, k2, k3):
    """Smooth background of the channel (k1 k2 | k3) in BBSX, their Eq. (5.21) (large lam and masses)."""
    mu1, mu2, lam = mp.mpf(mu1), mp.mpf(mu2), mp.mpf(lam)
    p, chi = (k1 + k2) / k3, (k1 - k2) / k3
    prod = mp.sqrt((lam + mu1 + mu2) * (lam + mu1 - mu2) * (lam - mu1 + mu2) * (lam - mu1 - mu2))
    L = mp.log((p + 1) / (p - 1))
    br = (p - L / 2 - (p ** 2 + 3 * chi ** 2) / 4 * (1 - p / 2 * L) - (p ** 2 - chi ** 2) / (4 * (p + 1)))
    return 5 / (18 * mp.pi) * prod / (lam * (p ** 3 + 3 * chi ** 2 * p + 4)) * br



def nonderivative_pole(S, k1, k2, k3, delta=mp.mpf(10) ** -7, Mmax=40):
    """Residue at delta = 0 of Im sum_eps w_eps P_eps for the non-derivative vertices at r_v + delta.

    It comes from Gamma(R + m + 2 delta) in the background ladder (R = j + j' - 3) and is nonzero: a single
    diagram with the non-derivative vertices has a late-time pole. By the shift symmetry it is proportional
    to k1^3 + k2^3 - k3^3 and cancels in the sum of all diagrams.
    """
    with mp.workdps(S.dps):
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        lam, x1, k = S.lam, k1 + k2, k3
        hs, ss = (1, x1, k1 * k2), (1, k3)

        def X(d):
            tot = mp.mpc(0)
            for jp in (0, 1):
                rho2 = jp + I * lam - mp.mpf(3) / 2 + d
                pre, C = mp.gamma(rho2) * (k / 2) ** (-rho2), mp.mpf(1)
                for m in range(Mmax):
                    if m > 0:
                        C *= (2 * rho2 + m - 1) / m
                    z = rho2 + m
                    coef = sum(S.wts[e] * (mp.mpmathify(S.tr.S0(e, z)) + mp.mpf(1) / 2) for e in PATTERNS)
                    L = sum(hs[j] * Ihat(z, j - I * lam - mp.mpf(3) / 2 + d, x1, k) for j in (0, 1, 2))
                    tot += ss[jp] * pre * (-1) ** m * z * C * L * coef
            return tot

        return mp.im(delta * (X(delta) - X(-delta)) / 2)


def sumline_ratio(S, k3, lead=None):
    """Full sum line over its leading squeezed term, for the bispectrum at k1 = k2 = 1/2 (complex, -> 1)."""
    with mp.workdps(S.dps):
        k3, half = mp.mpf(k3), mp.mpf(1) / 2
        lead = lead or mp.conj(S.family_leading((1, 1)))
        Ft = mp.conj(S.pref(half, half, k3) * S.X(half, half, k3, only=(1, 1)) / (2 * I))
        osc = (1 / k3) ** (-3 - I * (S.lam - S.mu1 - S.mu2))
        return Ft * (1 + 4 * k3 ** 3) / (lead * osc)


def _bisect(f, a, b, n=14):
    fa = f(a)
    for _ in range(n):
        c = (a + b) / 2
        fc = f(c)
        if (fa < 0) == (fc < 0):
            a, fa = c, fc
        else:
            b = c
    return (a + b) / 2


def main_shape(quick=False, json_path=None):
    res, data = [], {}
    half = mp.mpf(1) / 2
    print("S1. Squeezed limit of the complete bispectrum: the four families against ChargedBubble.family")
    worst = 0
    for (m1, m2, lam) in (('1', '3', '10'), ('2', '2', '20'), ('0.7', '1.1', '2')):
        S, B = ChargedShape(m1, m2, lam), ChargedBubble(m1, m2, lam)
        with mp.workdps(S.dps):
            worst = max(worst, max(abs(S.family_leading(e) / B.family(1, e[0], e[1]) - 1) for e in PATTERNS))
    print(f"   max relative difference over three parameter points: {mp.nstr(worst, 3)}")
    res.append(compare("charged_shape/squeezed families", worst))

    print("S2. Direct evaluation: sum over trees against the spectral integral over tree exchanges")
    worst = 0
    pts = [('0.7', '1.1', '2')] if quick else [('0.7', '1.1', '2'), ('1', '1', '10')]
    for (m1, m2, lam) in pts:
        S = ChargedShape(m1, m2, lam)
        shapes = [(half, half, mp.mpf(u)) for u in ('0.05', '0.2', '0.5', '0.7', '0.9')]
        if not quick:
            shapes += [(mp.mpf('0.6'), mp.mpf('0.4'), mp.mpf('0.3')), (half, mp.mpf('0.3'), half)]
        for (k1, k2, k3) in shapes:
            with mp.workdps(S.dps):
                Xt, Xs = S.X(k1, k2, k3), spectral_X(S, k1, k2, k3)
                err = abs(mp.im(Xt - Xs) / mp.im(Xt))
                worst = max(worst, err, abs(Xt - Xs) / abs(Xt))
            print(f"   (lam; mu1, mu2) = ({lam}; {m1}, {m2}), (k1, k2 | k3) = ({mp.nstr(k1, 3)}, {mp.nstr(k2, 3)} | {mp.nstr(k3, 3)}):"
                  f" F = {mp.nstr(S.pref(k1, k2, k3) * mp.im(Xt), 12)}, relative difference {mp.nstr(err, 3)}", flush=True)
    res.append(compare("charged_shape/direct check (trees vs spectral)", worst))

    print("S3. Resonance lam = mu1 + mu2 (mu = 2, k3/k12 = 0.03): poles of the sum line and of the background")
    with mp.workdps(60):
        e, u = mp.mpf(10) ** -4, mp.mpf('0.03')
        cp = ChargedShape('2', '2', 4 + e, dps=60).channel(half, half, u)
        cm = ChargedShape('2', '2', 4 - e, dps=60).channel(half, half, u)
        R = ChargedBubble('2', '2', 4).residue()
        norm = (1 + 4 * u ** 3) / u ** 3
        fam_odd = e * (cp['families'][(1, 1)] - cm['families'][(1, 1)]) / 2 * norm
        bg_odd = e * (cp['background'] - cm['background']) / 2 * norm
        print(f"   eps * odd part * p^3: sum line {mp.nstr(fam_odd, 9)}, background {mp.nstr(bg_odd, 9)}, -2 Im R = {mp.nstr(-2 * mp.im(R), 9)}")
        print(f"   F(4 + eps) = {mp.nstr(cp['F'], 9)}, F(4 - eps) = {mp.nstr(cm['F'], 9)}  (sum line alone: {mp.nstr(cp['families'][(1, 1)], 4)})")
        res.append(compare("charged_shape/resonance pole of sum line / (-2 Im R) - 1", abs(fam_odd / (-2 * mp.im(R)) - 1)))
        res.append(compare("charged_shape/resonance pole cancellation", abs(fam_odd + bg_odd) / abs(fam_odd)))
        res.append(compare("charged_shape/F on resonance mu=2 u=0.03", (cp['F'] + cm['F']) / 2))
        # the logarithm of the sum line on resonance: its even part is p^-3 (2 Re R ln p + const)
        u2 = mp.mpf('0.01')
        ev = lambda a, b, x: (a['families'][(1, 1)] + b['families'][(1, 1)]) / 2 * (1 + 4 * x ** 3) / x ** 3
        dp = ChargedShape('2', '2', 4 + e, dps=60).channel(half, half, u2)
        dm = ChargedShape('2', '2', 4 - e, dps=60).channel(half, half, u2)
        slope = (ev(dp, dm, u2) - ev(cp, cm, u)) / mp.log(u / u2)
        print(f"   coefficient of p^-3 ln p in the sum line (from k3/k12 = 0.01 and 0.03): {mp.nstr(slope, 7)}, 2 Re R = {mp.nstr(2 * mp.re(R), 7)}")
        res.append(compare("charged_shape/resonance logarithm / (2 Re R) - 1", abs(slope / (2 * mp.re(R)) - 1)))

    print("S4. Non-derivative form of one diagram: residue of the late-time pole, over (k1^3 + k2^3 - k3^3)")
    S = ChargedShape('0.7', '1.1', '2', dps=30)
    with mp.workdps(30):
        cs = []
        for (k1, k2, k3) in ((0.5, 0.5, 0.3), (0.6, 0.4, 0.3), (0.5, 0.5, 0.6)):
            cs.append(nonderivative_pole(S, k1, k2, k3) / (mp.mpf(k1) ** 3 + mp.mpf(k2) ** 3 - mp.mpf(k3) ** 3))
        print("   " + ", ".join(mp.nstr(c, 9) for c in cs))
        res.append(compare("charged_shape/late-time pole coefficient (0.7,1.1;2)", cs[0]))
        res.append(compare("charged_shape/late-time pole shape independence", max(abs(c / cs[0] - 1) for c in cs)))
        # The diagram with all three legs on one vertex (loop momentum 0, the other vertex without legs, r2 = i lam)
        # has the block Gamma(R) k_T^(-R) [S_0(i lam - 3/2) + const], R = j - 3, with the coefficients
        # (1, k_T, k1 k2 + k2 k3 + k3 k1, k1 k2 k3) of the three legs. Its residue is (sum k^3 / 6) Im sum_eps w_eps S_0^eps,
        # and it enters with the opposite sign, so it cancels the three channels, whose residues add up to cs * sum k^3.
        three = mp.im(sum(S.wts[e] * mp.mpmathify(S.tr.S0(e, S.rho0p)) for e in PATTERNS)) / 6
        print(f"   diagram with three legs on one vertex: {mp.nstr(three, 9)} sum k^3 (the three channels: {mp.nstr(cs[0], 9)} sum k^3)")
        res.append(compare("charged_shape/late-time pole cancels in the sum of diagrams", abs(three / cs[0] - 1)))

    print("S5. Full shape at the benchmarks (k1 = k2 = 1/2; F of the channel (k1 k2 | k3) and of the three channels)")
    us = ('0.02', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9')
    cases = [('4', '40'), ('2', '20'), ('1', '10'), ('4.5', '10'), ('5.1', '10')]
    if quick:
        us, cases = ('0.05', '0.3'), [('1', '10')]
    data['us'], data['benchmarks'] = [float(u) for u in us], {}
    for mu, lam in cases:
        S = ChargedShape(mu, mu, lam)
        with mp.workdps(S.dps):
            lead, fb = mp.conj(S.family_leading((1, 1))), f_bbsx(mu, mu, lam)
            print(f"   (lam, mu) = ({lam}, {mu}):  |f exact/f BBSX| = {mp.nstr(abs(lead / fb), 5)}")
            print("      k3/k12  |sum line/leading|  arg     |sum line/template|  background/BBSX(5.21)   F(k1 k2|k3)     F (three channels)")
            rows = []
            for u in us:
                k3 = mp.mpf(u)
                c = S.channel(half, half, k3)
                osc = (1 / k3) ** (-3 - I * (S.lam - S.mu1 - S.mu2))
                Rb = c['Ftilde'] * (1 + 4 * k3 ** 3) / (lead * osc)
                Tr = c['Ftilde'] / (fb * osc)
                bgb = f_bkg_bbsx(mu, mu, lam, half, half, k3) if S.lam > S.mu1 + S.mu2 else mp.nan   # (5.21) needs lam > mu12
                Ftot = c['F'] + 2 * S.channel(half, k3, half, tol=mp.mpf(10) ** -18)['F']
                rows.append(dict(u=float(u), absR=float(abs(Rb)), argR=float(mp.arg(Rb)), absT=float(abs(Tr)), argT=float(mp.arg(Tr)),
                                 F_channel=float(c['F']), F_total=float(Ftot), background=float(c['background']),
                                 background_bbsx=float(bgb), sumline=float(c['families'][(1, 1)]),
                                 template=float(2 * mp.re(fb * osc)), amp=float(2 * abs(c['Ftilde']))))
                print(f"      {u:>5s}    {mp.nstr(abs(Rb), 6):>9s}   {mp.nstr(mp.arg(Rb), 4):>8s}    {mp.nstr(abs(Tr), 5):>8s}        "
                      f"{mp.nstr(c['background'] / bgb, 6):>9s}        {mp.nstr(c['F'], 8):>14s}  {mp.nstr(Ftot, 8):>14s}", flush=True)
            data['benchmarks'][f"lam={lam},mu={mu}"] = dict(f_exact=[float(mp.re(lead)), float(mp.im(lead))],
                                                           f_bbsx=[float(mp.re(fb)), float(mp.im(fb))], rows=rows)
            key = f"({lam},{mu})"
            if not quick and (lam, mu) in (('40', '4'), ('20', '2'), ('10', '1')):
                uc = _bisect(lambda x: abs(sumline_ratio(S, x, lead) - 1) - mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.6'))
                um = _bisect(lambda x: abs(abs(sumline_ratio(S, x, lead)) - 1) - mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.6'))
                print(f"      leading squeezed term good to 10%: as a complex amplitude for k3/k12 < {mp.nstr(uc, 4)}, in modulus for k3/k12 < {mp.nstr(um, 4)}")
                data['benchmarks'][f"lam={lam},mu={mu}"].update(u10_complex=float(uc), u10_modulus=float(um))
                res.append(compare(f"charged_shape/u10 complex {key}", uc))
                res.append(compare(f"charged_shape/u10 modulus {key}", um))
                r = {x['u']: x for x in rows}
                res.append(compare(f"charged_shape/|sum line/template| {key} u=0.3", r[0.3]['absT']))
                res.append(compare(f"charged_shape/background/BBSX {key} u=0.1", r[0.1]['background'] / r[0.1]['background_bbsx']))
                res.append(compare(f"charged_shape/background/BBSX {key} u=0.5", r[0.5]['background'] / r[0.5]['background_bbsx']))
                res.append(compare(f"charged_shape/F channel {key} u=0.5", r[0.5]['F_channel']))
                res.append(compare(f"charged_shape/F total {key} u=0.5", r[0.5]['F_total']))
            if quick:
                r = {x['u']: x for x in rows}
                res.append(compare("charged_shape/F channel (10,1) u=0.3", r[0.3]['F_channel']))
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=1)
    return report(res)


def main():
    res = []
    print("1. Late-time coefficients of the modes against hankel1 (y|tau| = 1e-6)")
    err = max(check_late(mu) for mu in (1, 2.5, 4))
    print(f"   max relative error {mp.nstr(err, 3)} (leading neglected term ~ (y tau)^2 = 1e-12)")
    res.append((err < 1e-11, err))

    print("2. Rotated soft vertex against the conformal-time integral (fixed loop momentum)")
    with mp.workdps(30):
        err = max(check_rotation(mu1, mu2, lam, y1, y2) for (mu1, mu2, lam, y1, y2) in
                  [(1, 1, 3, mp.mpf('0.6'), mp.mpf('0.9')), (2, 1.5, 5, mp.mpf('1.3'), mp.mpf('0.4'))])
    print(f"   max relative difference {mp.nstr(err, 3)}")
    res.append((err < 1e-20, err))

    print("3. Reality of the bispectrum: f(lam - mu12) = conj of the coefficient of p^(-3 + i(lam - mu12))")
    err = 0
    for (m1, m2, lam) in [(1, 1, 4), (1, 3, 10), (2, 2, 9)]:
        B = ChargedBubble(m1, m2, lam)
        for (s1, s2) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            a, bb = B.family(-1, -s1, -s2), B.family(1, s1, s2)
            err = max(err, abs(a - mp.conj(bb)) / abs(a))
    print(f"   max |f - conj(f')|/|f| over four families, three parameter points: {mp.nstr(err, 3)}")
    res.append((err < mp.mpf(10) ** -40, err))

    print("4. Comparison with BBSX Eq. (3.35), mu1 = mu2 = mu")
    print("     lam    mu     |f exact|        |f BBSX|      ratio   phase offset (rad)")
    rows = [(40, 8), (50, 10), (40, 4), (20, 2), (10, 1), (10, 5.1), (10, 6)]
    vals = {}
    for lam, mu in rows:
        fe, fb = ChargedBubble(mu, mu, lam).f(), f_bbsx(mu, mu, lam)
        q = fe / fb
        vals[(lam, mu)] = (fe, fb, q)
        print(f"   {lam:5}  {mu:5}  {mp.nstr(abs(fe), 6):>12s}  {mp.nstr(abs(fb), 6):>12s}  {mp.nstr(abs(q), 4):>8s}  {mp.nstr(abs(mp.arg(q)), 3):>8s}")
    res.append(compare("charged/|ratio| (40,8) - 1", abs(abs(vals[(40, 8)][2]) - 1)))
    res.append(compare("charged/|ratio| (50,10) - 1", abs(abs(vals[(50, 10)][2]) - 1)))
    res.append(compare("charged/|ratio| (40,4)", abs(vals[(40, 4)][2])))
    res.append(compare("charged/|ratio| (20,2)", abs(vals[(20, 2)][2])))
    res.append(compare("charged/|ratio| (10,1)", abs(vals[(10, 1)][2])))
    res.append(compare("charged/phase offset (40,4)", abs(mp.arg(vals[(40, 4)][2]))))
    res.append(compare("charged/phase offset (10,1)", abs(mp.arg(vals[(10, 1)][2]))))
    res.append(compare("charged/|ratio| (10,5.1)", abs(vals[(10, 5.1)][2])))
    res.append(compare("charged/|f exact| (10,6)", abs(vals[(10, 6)][0])))
    res.append(compare("charged/|f BBSX| (10,6)", abs(vals[(10, 6)][1])))
    # suppression below threshold, ~ e^(-pi (mu12 - lam))
    s1, s2 = abs(ChargedBubble(6, 6, 10).f()), abs(ChargedBubble(6.5, 6.5, 10).f())
    rate = mp.log(s1 / s2) / (mp.pi * 1)
    print(f"   below threshold: d ln|f| / d(pi mu12) between mu12 = 12 and 13: {mp.nstr(-rate, 4)} (e^(-pi(mu12 - lam)): -1)")

    print("5. Resonance lam = mu1 + mu2: pole of the soft moment, residue R; f at lam = mu12 + 1")
    print("     mu     |R|        |f(mu12+1)|   |R|/|f|")
    rows5 = []
    for mu in (1, 2, 4, 8):
        B = ChargedBubble(mu, mu, 2 * mu)
        R = B.residue()
        fo = ChargedBubble(mu, mu, 2 * mu + 1).f()
        print(f"   {mu:4}  {mp.nstr(abs(R), 4):>9s}  {mp.nstr(abs(fo), 4):>10s}   {mp.nstr(abs(R) / abs(fo), 3)}")
        rows5.append((mu, R, fo))
    for mu, R, fo in rows5:
        res.append(compare(f"charged/|R| mu={mu}", abs(R)))
        res.append(compare(f"charged/|f| off resonance mu={mu}", abs(fo)))

    print("6. Difference line lam - |mu1 - mu2| relative to the sum line (falls as e^(-2 pi min(mu1, mu2)))")
    for (m1, m2, lam) in [(1, 3, 10), (1, 3, 40), (2, 6, 40), (4, 8, 40), (8, 12, 50)]:
        B = ChargedBubble(m1, m2, lam)
        fd = B.family(-1, 1, -1)    # coefficient of p^(-3 - i(lam - (mu2 - mu1)))
        rel = abs(fd) / abs(B.f())
        print(f"   (lam; mu1, mu2) = ({lam}; {m1}, {m2}): {mp.nstr(rel, 3):>9s}   / e^(-2 pi mu1) = {mp.nstr(rel / mp.exp(-2 * mp.pi * m1), 3)}")
        if (m1, m2, lam) == (1, 3, 10):
            res.append(compare("charged/difference line (10;1,3)", rel))
        if (m1, m2, lam) == (4, 8, 40):
            res.append(compare("charged/difference line (40;4,8)", rel))
    return report(res)


if __name__ == "__main__":
    if '--shape' in sys.argv:
        jp = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
        sys.exit(main_shape(quick='--quick' in sys.argv, json_path=jp))
    sys.exit(main())
