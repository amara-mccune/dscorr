"""The soft vertex in position space (Sec. 9.2 and App. D; Eqs. [eq:WLaplace], [eq:Jpos], [eq:vecJpos]).

We need loop integrals of the form

    J = int d^3l/(2 pi)^3  y1^A y2^B  int_0^oo dt t^(q-1) e^(-x t) W_{K1,nu}(2 y1 t) W_{K2,nu}(2 y2 t),

with y1 = |l|, y2 = |k - l| and k = x = 1 (the soft leg ends on the vertex). A and B are complex (they carry
the late-time powers that the hard vertices leave on the soft lines), and the integral is defined by
continuation in q, as Eq. [eq:twomoment].

Method. By the convolution theorem the loop integral is the Fourier transform of a product of the two
lines' Fourier transforms. Each of those is a Laplace transform of z^(p-1) W_{K,nu}(z) at imaginary argument,
a Gauss function [eq:WLaplace]. With x = w t the vertex time integral is a Gamma function, and

    J = 2^(-A-B-4) Gamma(Q) / (2 pi^3)  int_0^oo dw  Lam_A(w) Lam_B(w) ang_Q(w),      Q = q - A - B - 3,
    Lam_A(w) = [L(A+2, -i w/2) - L(A+2, i w/2)] / (2i),     L(p, s) = int dz e^(-s z) z^(p-1) W_{K,nu}(z),
    ang_Q(w) = [(1 + i w)^(1-Q) - (1 - i w)^(1-Q)] / (i w (1 - Q))     (the angular integral).

Gamma(Q) continues the ultraviolet region of the loop. The radial integrand falls only as w^(-q-2) times
w^(+-2 nu), with coefficients of size e^(pi |nu|), so the tail matters. Here it is done exactly: for w > 1
both Lam and ang have convergent expansions in powers of 1/w. For Lam this is the small-z series of W
(Watson's lemma, convergent since the branches of W grow only as e^(z/2)):

    L(p, s) = sum_sigma B_sigma(K) sum_n e_n(K, sigma nu) Gamma(p + 1/2 + sigma nu + n) s^(-(p + 1/2 + sigma nu + n)),
    B_sigma(K) = Gamma(-2 sigma nu)/Gamma(1/2 - sigma nu - K),     |s| > 1/2.

So the integral is split at w0 > 1: [0, w0] by tanh-sinh quadrature, [w0, oo) term by term in closed form.
At large chemical potential the integrand reaches 1e11 to 1e27 and cancels to O(1-100), so everything is
done in mpmath at the caller's working precision (30 to 60 digits; see the fermion-triangle script).
"""
import mpmath as mp

from .blocks import e_series, laplace_W

__all__ = ["Grid", "Leg", "soft_J", "B_sigma"]


def B_sigma(K, nu, s):
    """W_{K,nu} = B_+ M_{K,nu} + B_- M_{K,-nu}:  B_s = Gamma(-2 s nu)/Gamma(1/2 - s nu - K)."""
    return mp.gamma(-2 * s * nu) / mp.gamma(0.5 - s * nu - K)


class Grid:
    """Gauss-Legendre nodes and weights (exact, at the working precision) on panels of [0, w0]."""

    def __init__(self, w0=6, breaks=(0.5, 1, 2, 3, 4.5), npts=24):
        self.w0 = mp.mpmathify(w0)
        edges = [mp.mpf(0)] + [mp.mpmathify(b) for b in breaks] + [self.w0]
        x, wt = _gauss_legendre(npts)
        self.nodes, self.weights = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            h = (b - a) / 2
            self.nodes += [a + h * (1 + xi) for xi in x]
            self.weights += [h * wi for wi in wt]
        self.key = (mp.mp.prec, str(self.w0), tuple(str(b) for b in breaks), npts)


def _gauss_legendre(n):
    """Nodes and weights on [-1, 1] by Newton iteration on P_n (mpmath precision)."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mp.mpf(2) ** (-mp.mp.prec + 4):
                break
        p0, p1 = mp.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


class Leg:
    """Fourier-space leg Lam_A(w) of one soft line, y^A W_{K,nu}(2 y t), with its large-w series.

    Values on a Grid are cached, so the same leg can enter many monomials at the cost of one table.
    """

    def __init__(self, A, K, nu, nterms=None):
        self.A, self.K, self.nu = mp.mpmathify(A), mp.mpmathify(K), mp.mpmathify(nu)
        self.p = self.A + 2
        # the coefficients e_n of the branch series grow like (|K|/n)^n before they fall, so the 1/w series
        # at w0 = 6 needs more terms at large first index (160 are needed at |K| = 20)
        self.nterms = nterms or int(60 + 7 * max(abs(self.K), abs(self.nu)))
        self._tail = None
        self._vals = {}

    def __call__(self, w):
        s = mp.mpc(0, 1) * w / 2
        return (laplace_W(self.p, self.K, self.nu, -s) - laplace_W(self.p, self.K, self.nu, s)) / mp.mpc(0, 2)

    def values(self, grid):
        if grid.key not in self._vals:
            self._vals[grid.key] = [self(w) for w in grid.nodes]
        return self._vals[grid.key]

    def tail(self):
        """{sigma: (a0, [c_n])} with Lam_A(w) = sum_sigma w^(-a0) sum_n c_n w^(-n) for w > 1."""
        if self._tail is None:
            out = {}
            for s in (1, -1):
                e = e_series(self.K, s * self.nu, self.nterms)
                a0 = self.p + 0.5 + s * self.nu
                Bs = B_sigma(self.K, self.nu, s)
                out[s] = (a0, [Bs * e[n] * mp.gamma(a0 + n) * mp.sin(mp.pi * (a0 + n) / 2) * mp.mpf(2) ** (a0 + n)
                               for n in range(self.nterms + 1)])
            self._tail = out
        return self._tail


def _ang(Q, w):
    I = mp.mpc(0, 1)
    return ((1 + I * w) ** (1 - Q) - (1 - I * w) ** (1 - Q)) / (I * w * (1 - Q))


def _ang_tail(Q, nterms):
    """ang_Q(w) = w^(-Q) sum_j c_j w^(-j) for w > 1."""
    return [2 * mp.binomial(1 - Q, j) * mp.sin(mp.pi * (1 - Q - j) / 2) / (1 - Q) for j in range(nterms + 1)]


def _conv(a, b, N):
    return [mp.fsum(a[i] * b[n - i] for i in range(n + 1)) for n in range(N + 1)]


def soft_J(legA, legB, q, grid=None, nterms=None, parts=False):
    """J of the module docstring for two legs (Leg objects) and vertex power q, at k = x = 1.

    grid: a Grid for the near region [0, w0] (default Grid()). nterms: number of tail terms.
    With parts=True also returns (near, tail) before the common prefactor.
    """
    grid = grid or Grid()
    q = mp.mpmathify(q)
    Q = q - legA.A - legB.A - 3
    N = nterms or min(legA.nterms, legB.nterms)
    va, vb = legA.values(grid), legB.values(grid)
    near = mp.fsum(wt * a * b * _ang(Q, w) for w, wt, a, b in zip(grid.nodes, grid.weights, va, vb))
    ta, tb, tq = legA.tail(), legB.tail(), _ang_tail(Q, N)
    far, big, last = 0, mp.mpf(0), mp.mpf(0)
    for s1 in (1, -1):
        a1, c1 = ta[s1]
        for s2 in (1, -1):
            a2, c2 = tb[s2]
            c = _conv(_conv(c1, c2, N), tq, N)
            for n in range(N + 1):
                beta = a1 + a2 + Q + n
                term = c[n] * mp.power(grid.w0, 1 - beta) / (beta - 1)
                far += term
                big = max(big, abs(term))
                if n >= N - 4:
                    last = max(last, abs(term))
    if last > big * mp.mpf(10) ** (-mp.mp.dps + 8):
        import warnings
        warnings.warn(f"soft_J: 1/w tail series not converged (last terms {mp.nstr(last / big, 3)} of the largest); "
                      f"increase Leg(nterms=...) or w0")
    pref = mp.mpf(2) ** (-legA.A - legB.A - 4) * mp.gamma(Q) / (2 * mp.pi ** 3)
    if parts:
        return pref * (near + far), (near, far)
    return pref * (near + far)
