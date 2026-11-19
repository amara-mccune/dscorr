"""Dirac line with an axial chemical potential (Sec. 8.1, Eqs. [eq:diracmodes], [eq:diracocc]).

Helicity-h chiral amplitudes of momentum y:
    alpha_h = N c_alpha (-tau)^(-1/2) W_{K_alpha, nu}(2 i y tau),   beta_h = N c_beta (-tau)^(-1/2) W_{K_beta, nu}(2 i y tau),
    N = e^(pi h kappa/2)/sqrt(2 y),  nu = i mubar,  mubar^2 = m^2 + kappa^2,
    K_alpha = h (1/2 - i kappa),  K_beta = -h (1/2 + i kappa),
    (c_alpha, c_beta) = (1, -i m) for h = +1 and (m, i) for h = -1,
so that |alpha|^2 + |beta|^2 = 1. On the rotated contour the decaying (conjugate) solution has c_X and K_X
replaced by their conjugates at fixed nu.
"""
import mpmath as mp

from .blocks import e_series

I = mp.mpc(0, 1)


class DiracLine:
    def __init__(self, m, kappa):
        self.m, self.kappa = mp.mpf(m), mp.mpf(kappa)
        self.mubar = mp.sqrt(self.m ** 2 + self.kappa ** 2)
        self.nu = I * self.mubar

    def K(self, X, h):
        return h * (0.5 - I * self.kappa) if X == 'a' else -h * (0.5 + I * self.kappa)

    def c(self, X, h):
        if h == 1:
            return mp.mpc(1) if X == 'a' else -I * self.m
        return mp.mpc(self.m) if X == 'a' else I

    @staticmethod
    def eps(X):
        return 1 if X == 'a' else -1

    def B(self, s, K):
        """Branch weight: W_{K,nu} = B_+ M_{K,nu} + B_- M_{K,-nu}."""
        return mp.gamma(-2 * s * self.nu) / mp.gamma(0.5 - s * self.nu - K)

    def E(self, X, h, s, N, conj=False):
        """E^s_{X,h}[n] = c_X B_s(K_X) e_n(K_X, s nu), n = 0..N (conj: c_X, K_X conjugated at fixed nu)."""
        c, K = self.c(X, h), self.K(X, h)
        if conj:
            c, K = mp.conj(c), mp.conj(K)
        Bs = self.B(s, K)
        return [c * Bs * en for en in e_series(K, s * self.nu, N)]

    def occupation(self, h):
        """n_h of Eq. [eq:diracocc]."""
        mb, k = self.mubar, self.kappa
        return (mp.exp(2 * mp.pi * h * k) - mp.exp(-2 * mp.pi * mb)) / (2 * mp.sinh(2 * mp.pi * mb))

    def mode(self, X, h, y, tau):
        """alpha_h (X = 'a') or beta_h (X = 'b') at real tau < 0."""
        N = mp.exp(mp.pi * h * self.kappa / 2) / mp.sqrt(2 * y)
        return N * self.c(X, h) * (-tau) ** -0.5 * mp.whitw(self.K(X, h), self.nu, 2 * I * y * tau)
