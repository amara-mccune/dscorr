"""Dirac fermion with an axial chemical potential in the conventions of Aoki and Strumia (arXiv:2604.05548).

Weyl basis, Psi = (psi_L, psibar_R), gamma^mu = [[0, sigma^mu], [sigmabar^mu, 0]], gamma^5 = diag(-1, +1).
With mu_V = 0 and kappa = mu_A (H = 1), lam = sqrt(m^2 + kappa^2), nu = i lam, the spinors of momentum y and
helicity h are (their Eqs. (26)-(27))

    u_{+} = (-2 y eta)^(-1/2) e^(+pi kappa/2) ( m W_{-1/2 - i kappa}(2 i y eta) chi_+ ;  i W_{+1/2 - i kappa}(2 i y eta) chi_+ ),
    u_{-} = (-2 y eta)^(-1/2) e^(-pi kappa/2) ( i W_{+1/2 + i kappa}(2 i y eta) chi_- ;  m W_{-1/2 + i kappa}(2 i y eta) chi_- ),
    v_{+} = (-2 y eta)^(-1/2) e^(+pi kappa/2) ( i W_{+1/2 + i kappa}(-2 i y eta) chi_+ ; m W_{-1/2 + i kappa}(-2 i y eta) chi_+ ),
    v_{-} = (-2 y eta)^(-1/2) e^(-pi kappa/2) ( m W_{-1/2 - i kappa}(-2 i y eta) chi_- ; i W_{+1/2 - i kappa}(-2 i y eta) chi_- ),

with W_K = W_{K, nu}. Each spinor is stored as two components X = 'L', 'R' given by (c_X, K_X), times the
argument sign s_W = +1 (u) or -1 (v). The vertex of delta tau is Gamma^mu = gamma^mu (mu_V + mu_A gamma^5)
= [[0, mu_R sigma^mu], [mu_L sigmabar^mu, 0]], with mu_R = kappa and mu_L = -kappa.

Functions of eta are evaluated on rotated rays eta = -t e^(-+ i theta), by analytic continuation from eta < 0.
"Conjugate" components are the continuation of the complex conjugate from real eta.
"""
import mpmath as mp

I = mp.mpc(0, 1)


class ASFermion:
    def __init__(self, m, kappa):
        self.m, self.kappa = mp.mpf(m), mp.mpf(kappa)
        self.lam = mp.sqrt(self.m ** 2 + self.kappa ** 2)
        self.nu = I * self.lam
        self.muR, self.muL = self.kappa, -self.kappa

    def comps(self, kind, h):
        """{'L': (c, K), 'R': (c, K)} for kind 'u' or 'v' and helicity h."""
        m, k = self.m, self.kappa
        if kind == 'u':
            if h == 1:
                return {'L': (mp.mpc(m), -0.5 - I * k), 'R': (I, 0.5 - I * k)}
            return {'L': (I, 0.5 + I * k), 'R': (mp.mpc(m), -0.5 + I * k)}
        if h == 1:
            return {'L': (I, 0.5 + I * k), 'R': (mp.mpc(m), -0.5 + I * k)}
        return {'L': (mp.mpc(m), -0.5 - I * k), 'R': (I, 0.5 - I * k)}

    @staticmethod
    def sW(kind):
        return 1 if kind == 'u' else -1

    def comp(self, kind, h, X, eta, y=1, conj=False, cache=None):
        """Component X of the spinor (or of its conjugate) at complex eta (on a rotated ray).

        cache: optional dict for the Whittaker values at this (eta, y) (u, v and their conjugates share them).
        """
        c, K = self.comps(kind, h)[X]
        s = self.sW(kind)
        if conj:
            c, K, s = mp.conj(c), mp.conj(K), -s
        if cache is None:
            W = mp.whitw(K, self.nu, s * 2 * I * y * eta)
        else:
            key = (mp.nstr(K, 15), s)
            if key not in cache:
                cache[key] = mp.whitw(K, self.nu, s * 2 * I * y * eta)
            W = cache[key]
        return (-2 * y * eta) ** -0.5 * mp.exp(mp.pi * h * self.kappa / 2) * c * W

    def B(self, sg, K):
        return mp.gamma(-2 * sg * self.nu) / mp.gamma(0.5 - sg * self.nu - K)

    def late(self, kind, h, X, sg, conj=False):
        """Late-time coefficient: comp -> late * (2 y)^(sg nu) (-eta)^(sg nu) on branch sg.

        W_K(z) = B_+(K) M_{K,nu}(z) + B_-(K) M_{K,-nu}(z), M_{K,s nu}(z) = z^(1/2 + s nu)(1 + ...), and
        s_W 2 i y eta = (-2 y eta) e^(i theta0), theta0 = -s_W pi/2.
        """
        c, K = self.comps(kind, h)[X]
        s = self.sW(kind)
        if conj:
            c, K, s = mp.conj(c), mp.conj(K), -s
        th0 = -s * mp.pi / 2
        return mp.exp(mp.pi * h * self.kappa / 2) * c * self.B(sg, K) * mp.exp(I * th0 * (0.5 + sg * self.nu))

    def column(self, kind, h, eta, y=1, cache=None):
        """Spinor as a block column (L; R)."""
        return [self.comp(kind, h, 'L', eta, y, cache=cache), self.comp(kind, h, 'R', eta, y, cache=cache)]

    def row(self, kind, h, eta, y=1, cache=None):
        """Dirac conjugate psibar = psi^dagger gamma^0 as a block row (conj R, conj L)."""
        return [self.comp(kind, h, 'R', eta, y, conj=True, cache=cache),
                self.comp(kind, h, 'L', eta, y, conj=True, cache=cache)]

    def late_column(self, kind, h, sg):
        return [self.late(kind, h, 'L', sg), self.late(kind, h, 'R', sg)]

    def late_row(self, kind, h, sg):
        return [self.late(kind, h, 'R', sg, conj=True), self.late(kind, h, 'L', sg, conj=True)]

    def vertex(self, d, a, eta, s):
        """Gamma^mu F_mu(k, eta) on branch a, for k = d z-hat (|k| = 1), at sigma_z = s, as a 2x2 block matrix.

        F_0 = eta e^(i a eta)/2, F_i = -i k_i G_a, G_a = (1 - i a eta) e^(i a eta)/2.
        sigma^mu F_mu = F_0 - i d s G,  sigmabar^mu F_mu = F_0 + i d s G.
        """
        e = mp.exp(I * a * eta)
        F0 = eta * e / 2
        G = (1 - I * a * eta) * e / 2
        return [[0, self.muR * (F0 - I * d * s * G)], [self.muL * (F0 + I * d * s * G), 0]]
