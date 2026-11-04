"""Charged scalars with a chemical potential on the vertex (Sec. 8.5).

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

Checks (Sec. 8.5):
  * late-time coefficients of the modes against mpmath's hankel1;
  * the rotated soft vertex (prefactor times K functions) against the conformal-time integral on a
    partially rotated ray, at fixed loop momentum;
  * reality: the coefficient of p^(-3 - i(lam - mu12)) is the conjugate of that of p^(-3 + i(lam - mu12));
  * comparison with BBSX at large lam and mu, near and below threshold;
  * the resonance lam = mu1 + mu2 (pole of the soft moment, residue R), and the difference line.

Usage:  python charged_scalars.py          (about 20 s)
"""
import sys

import mpmath as mp

from ext.blocks import two_moment
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
    sys.exit(main())
