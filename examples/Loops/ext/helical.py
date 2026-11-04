"""Massive vector with an F Ftilde chemical potential: line data for the collapsed box (Sec. 8; App. C).

Rotated contour, + branch tau = i t, momentum y = 1 unless stated. Transverse helicity h:
    u(tau) = N W_{K,nu}(2 i y tau),  N = e^(pi h kappa/2)/sqrt(2y),  K = -i h kappa,  nu = i mu,
    e*(t) := u*(i t) = (Nf/sqrt 2) W_{-K,nu}(2t)                       (decaying),
    e(t)  := u(i t)  = sum_s (A_s/sqrt 2) M_{-K, s nu}(2t)              (growing),
    e*(t) = sum_s (B_s/sqrt 2) M_{-K, s nu}(2t),
with Nf = e^(pi h kappa/2), B_s = Nf Gamma(-2 s nu)/Gamma(1/2 - s nu + K) and
A_s = Nf e^(-i pi (1/2 + s nu)) Gamma(-2 s nu)/Gamma(1/2 - s nu - K)   [Eq. (vecAB)].
Longitudinal line: the neutral mode e0 (K = 0) with e_L = (i/m)(t d_t - 1) e0, e_L* = (-i/m)(t d_t - 1) e0*,
m^2 = mu^2 + 1/4 [Eq. (vecL)].

Hard-line Schwinger-Keldysh moments at the doubly folded point (vertex energies = line energy = 1), with
the soft-line ends t^alpha (alpha = 1/2 + nu_soft) at both vertices:
    Q_ab = int dt1 dt2 (t1 t2)^alpha e^(-t1 - t2) G_ab(t1, t2),
    G_++ = e(t<) e*(t>),  G_-- = conj(e)(t<) conj(e*)(t>),  G_+- = e*(t1) conj(e*)(t2),  G_-+ = conj(e*)(t1) e*(t2),
(complex conjugation acts on the line functions only, not on the powers). Closed forms: the ordered ones
from Eq. [eq:helmoment]; the cut ones are Gamma functions; the longitudinal ones by parts, with the
equal-time Wronskian W[M_{0,mu}, W_{0,mu}] = -Gamma(1+2mu)/Gamma(1/2+mu).
"""
import mpmath as mp

from .blocks import hel_moment

I = mp.mpc(0, 1)


class VectorLine:
    def __init__(self, mu, kappa, h):
        """h = +1, -1 (transverse) or 0 (longitudinal)."""
        self.mu, self.kappa, self.h = mp.mpf(mu), mp.mpf(kappa), h
        self.nu = I * self.mu
        self.m = mp.sqrt(self.mu ** 2 + mp.mpf(1) / 4)
        hk = self.kappa * h if h != 0 else 0
        self.K = -I * hk
        self.Nf = mp.exp(mp.pi * hk / 2)
        nu, K = self.nu, self.K
        self.B = {s: self.Nf * mp.gamma(-2 * s * nu) / mp.gamma(0.5 - s * nu + K) for s in (1, -1)}
        self.A = {s: self.Nf * mp.exp(-I * mp.pi * (0.5 + s * nu)) * mp.gamma(-2 * s * nu) / mp.gamma(0.5 - s * nu - K)
                  for s in (1, -1)}

    # ------------------------------------------------------------------ soft line: late-time ends
    def g(self, a, b):
        """Coefficient of (2y)^(2 nu) (t_a t_b)^(1/2 + nu) in the two-point function of a soft line with ends on
        branches a (side 12) and b (side 34), both at late times (same-branch +nu pieces)."""
        A, B = self.A, self.B
        if a == 1 and b == 1:
            w = A[1] * B[1]
        elif a == -1 and b == -1:
            w = mp.conj(A[-1] * B[-1])
        else:
            w = B[1] * mp.conj(B[-1])
        if self.h == 0:
            w *= (self.nu - 0.5) ** 2 / self.m ** 2
        return w

    # ------------------------------------------------------------------ hard line: closed forms
    def _Qsym(self, b1, b2, conj=False):
        """int dt1 dt2 t1^b1 t2^b2 e^(-t1-t2) [e(t<) e*(t>)]  (or with conj(e), conj(e*)), K-line of this object."""
        nu, K = self.nu, self.K
        if not conj:
            return mp.fsum(self.A[s] * self.Nf * hel_moment(s * nu, -K, b1 + 0.5, b2 + 0.5)
                           * mp.gamma(1 + 2 * s * nu) / mp.gamma(0.5 + s * nu + K) for s in (1, -1))
        return mp.fsum(mp.conj(self.A[s]) * self.Nf * hel_moment(-s * nu, K, b1 + 0.5, b2 + 0.5)
                       * mp.gamma(1 - 2 * s * nu) / mp.gamma(0.5 - s * nu - K) for s in (1, -1))

    def _P(self, beta, conj=False):
        """int dt t^beta e^-t e*(t)  (or conj(e*)) = (Nf/sqrt 2) 2^(-beta-1) Gamma(beta+3/2+nu) Gamma(beta+3/2-nu)/Gamma(beta+2+-K)."""
        nu, K = self.nu, self.K
        Kp = K if not conj else -K
        return (self.Nf / mp.sqrt(2) * mp.mpf(2) ** (-beta - 1) * mp.gamma(beta + 1.5 + nu) * mp.gamma(beta + 1.5 - nu)
                / mp.gamma(beta + 2 + Kp))

    def _wronskian(self, conj=False):
        """W_t[e, e*] = e de*/dt - de/dt e* (constant); with conj(e), conj(e*) for conj=True."""
        nu = self.nu
        if not conj:
            return mp.fsum(self.A[s] * self.Nf * (-mp.gamma(1 + 2 * s * nu) / mp.gamma(0.5 + s * nu + self.K))
                           for s in (1, -1))
        return mp.fsum(mp.conj(self.A[s]) * self.Nf * (-mp.gamma(1 - 2 * s * nu) / mp.gamma(0.5 - s * nu - self.K))
                       for s in (1, -1))

    def Q(self, alpha):
        """{(a, b): Q_ab} at soft exponent alpha (closed form)."""
        al = mp.mpmathify(alpha)
        if self.h != 0:
            Qpp, Qmm = self._Qsym(al, al), self._Qsym(al, al, conj=True)
            Qpm = self._P(al) * self._P(al, conj=True)
            return {(1, 1): Qpp, (-1, -1): Qmm, (1, -1): Qpm, (-1, 1): Qpm}
        m2 = self.m ** 2
        c = {(0, 0): (al + 2) ** 2, (0, 1): -(al + 2), (1, 0): -(al + 2), (1, 1): 1}
        eq = mp.gamma(2 * al + 3) / mp.mpf(2) ** (2 * al + 3)
        out = {}
        for key, cj in ((1, False), (-1, True)):
            ordd = mp.fsum(cij * self._Qsym(al + i, al + j, conj=cj) for (i, j), cij in c.items()) / 2
            out[(key, key)] = 2 / m2 * (ordd + self._wronskian(conj=cj) * eq / 2)
        S = (al + 2) * self._P(al) - self._P(al + 1)
        out[(1, -1)] = out[(-1, 1)] = S * S / m2
        return out

    # ------------------------------------------------------------------ functions for direct quadrature
    def estar(self, t):
        return self.Nf / mp.sqrt(2) * mp.whitw(-self.K, self.nu, 2 * t)

    def e(self, t):
        return mp.fsum(self.A[s] / mp.sqrt(2) * mp.whitm(-self.K, s * self.nu, 2 * t) for s in (1, -1))

    def ebar(self, t):
        """conj(e(t)) as an analytic function of t (conjugated coefficients and indices)."""
        return mp.fsum(mp.conj(self.A[s]) / mp.sqrt(2) * mp.whitm(self.K, -s * self.nu, 2 * t) for s in (1, -1))

    def estarbar(self, t):
        return self.Nf / mp.sqrt(2) * mp.whitw(self.K, self.nu, 2 * t)
