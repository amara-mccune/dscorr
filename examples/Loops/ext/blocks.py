"""Closed-form building blocks shared by the scripts of the loops paper.

All functions take mpmath numbers (or anything mpmath accepts) and return mpmath
numbers at the current working precision. Equation numbers refer to the paper
("Massive de Sitter correlators at one loop"):

    C2, C2hat     loop block, Eq. (2.4)
    two_moment    soft moment of two scalar lines, Eq. (3.17)
    H_moment      scalar hard-line moment, Eq. (B.4)
    hel_moment    helical hard-line moment, Eq. (8.14)
    e_series      branch series of M_{K,nu}, Eq. (D.6)
    G_K           Mellin transform of W_{K,nu}, Eq. (D.8)
    laplace_W     Laplace transform of z^{p-1} W_{K,nu}(z), Eq. (9.8)

(Equation numbers of the Oct. 29 draft, loops_v3; the LaTeX labels are given in brackets elsewhere.)
"""
import mpmath as mp

__all__ = ["C2", "C2hat", "two_moment", "two_moment_late", "H_moment",
           "hel_moment", "e_series", "G_K", "laplace_W", "K_late"]


def C2(d1, d2):
    """int d^3 l  y1^(-2 d1) y2^(-2 d2) = k^(3 - 2 S) C2(d1, d2), with S = d1 + d2 [eq:C2].

    1/Gamma is used for the denominators, so the block vanishes where a line power is an even
    non-negative integer (the integral is then analytic in k and drops out of the nonlocal part).
    """
    S = d1 + d2
    return (mp.pi ** 1.5 * mp.gamma(1.5 - d1) * mp.gamma(1.5 - d2) * mp.gamma(S - 1.5)
            * mp.rgamma(d1) * mp.rgamma(d2) * mp.rgamma(3 - S))


def C2hat(p1, p2):
    """int d^3 l/(2 pi)  y1^(p1-1) y2^(p2-1) = k^(p1+p2+1) C2hat(p1, p2)."""
    return C2((1 - p1) / mp.mpf(2), (1 - p2) / mp.mpf(2)) / (2 * mp.pi)


def two_moment(nu1, nu2, s1, s2, r2, kL=1):
    """Soft moment of two scalar lines [eq:twomoment].

    M_sigma = int d^3 l (y1/2)^(s1 nu1) (y2/2)^(s2 nu2) int_0^oo dt t^(r2-1) e^(-kL t) K_nu1(y1 t) K_nu2(y2 t),
    continued in r2 from r2 > 3.
    """
    Sg = s1 * nu1 + s2 * nu2
    return (mp.pi ** 2 * mp.mpf(2) ** (2 - r2) * kL ** (3 - r2) * (kL / mp.mpf(2)) ** Sg
            * mp.gamma(1.5 + s1 * nu1) * mp.gamma(1.5 + s2 * nu2)
            * mp.gamma(r2 + Sg) * mp.gamma(r2 - 3 - Sg)
            / (mp.gamma(3 + Sg) * mp.gamma(r2 - 1)))


def K_late(nu):
    """Coefficient of the late-time branch of K_nu kept by the late-time estimate:
    K_nu(x) -> (1/2) Gamma(-nu) (x/2)^nu."""
    return mp.gamma(-nu) / 2


def two_moment_late(nu1, nu2, s1, s2, r2, kL=1):
    """The same moment with K_nu_l(y_l t) at the soft vertex replaced by its branch
    (1/2) Gamma(-s_l nu_l) (y_l t / 2)^(s_l nu_l), i.e. the late-time soft vertex.

    The loop integral is the block C2 at d_l = -s_l nu_l, the time integral a Gamma function.
    """
    a1, a2 = s1 * nu1, s2 * nu2
    pref = K_late(a1) * K_late(a2) * mp.mpf(2) ** (-2 * a1 - 2 * a2)
    return pref * C2(-a1, -a2) * kL ** (3 + 2 * a1 + 2 * a2) * mp.gamma(r2 + a1 + a2) * kL ** (-r2 - a1 - a2)


def H_moment(nu, a0, a1):
    """H_nu(a0, a1) = int dt0 dt1 t0^a0 t1^a1 e^(-t0-t1) K_nu(t>) I_nu(t<) [eq:Hmoment]."""
    L = a0 + a1 + 2
    pref = (mp.sqrt(mp.pi) * mp.gamma(L) * mp.gamma(a0 + 1 + nu) * mp.gamma(a1 + 1 + nu)
            / (mp.mpf(2) ** (L + 2 * nu) * mp.gamma(1 + nu) * mp.gamma(L + nu + 0.5)))
    return pref * mp.hyp3f2(nu + 0.5, a0 + 1 + nu, a1 + 1 + nu, L + nu + 0.5, 2 * nu + 1, 1)


def hel_moment(nu, K, a0, a1):
    """int dt0 dt1 t0^a0 t1^a1 e^(-t0-t1) K_h(1; t0, t1) for the helical kernel [eq:helmoment].

    K_h(y; t, t') = Gamma(a+)/Gamma(1+2nu) W_{K,nu}(2y t>) M_{K,nu}(2y t<) / (2y sqrt(t t')),
    a+ = 1/2 + nu - K. At K = 0 this is H_moment.
    """
    lam = a0 + a1 + 2
    ap = 0.5 + nu - K
    pref = (mp.gamma(lam) * mp.gamma(a0 + 1 + nu) * mp.gamma(a1 + 1 + nu) * mp.gamma(ap)
            / (mp.mpf(2) ** lam * mp.gamma(1 + 2 * nu) * mp.gamma(lam + ap)))
    return pref * mp.hyp3f2(ap, a0 + 1 + nu, a1 + 1 + nu, lam + ap, 1 + 2 * nu, 1)


def e_series(K, nu, N):
    """Coefficients e_0..e_N of M_{K,nu}(z) = z^(1/2+nu) sum_n e_n z^n [eq:en] (three-term recurrence)."""
    e = [mp.mpc(1)]
    if N >= 1:
        e.append(-K / (1 + 2 * nu))
    for n in range(1, N):
        e.append((-K * e[n] + e[n - 1] / 4) / ((n + 1) * (n + 1 + 2 * nu)))
    return e[:N + 1]


def G_K(K, nu, s):
    """int_0^oo z^(s-1) W_{K,nu}(z) dz [eq:GK]."""
    return (mp.gamma(0.5 + nu + s) * mp.gamma(0.5 - nu + s) / mp.gamma(1 - K + s)
            * mp.hyp2f1(0.5 + nu + s, 0.5 - nu + s, 1 - K + s, 0.5))


def laplace_W(p, K, nu, s):
    """int_0^oo dz e^(-s z) z^(p-1) W_{K,nu}(z) [eq:WLaplace], continued in s."""
    a, b, c = p + 0.5 + nu, p + 0.5 - nu, p + 1 - K
    return mp.gamma(a) * mp.gamma(b) / mp.gamma(c) * mp.hyp2f1(a, b, c, 0.5 - s)
