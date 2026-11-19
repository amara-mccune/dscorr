"""Mode functions of the Proca field with an F Ftilde chemical potential on the
rotated contour (independent implementation, agent ninepairs).

Conventions (derived in README / report):
  transverse helicity h: u_h(tau) = N W_{K,nu}(2 i y tau), N = e^{pi h kappa/2}/sqrt(2y),
  K = -i h kappa, nu = i mu, m^2 = mu^2 + 1/4.  h kappa > 0 is enhanced.
  On the + branch tau = i t:   u*(it) = N W_{-K,nu}(z),  z = 2 y t  (decaying)
                               u(it)  = N W_{K,nu}(z e^{-i pi})     (growing)
  Longitudinal (A_i = qhat_i A_L):  v_0 = -(q/m) tau g,  v_L = (i/m)(tau d_tau - 1) g,
  g = u at kappa = 0.

Whittaker functions are evaluated with python-flint (arb) at 100 bits and
tabulated as e^{x/2} W(x) on Chebyshev panels in ln x for x in [XLO, XHI];
below XLO the two-branch connection formula is used, above XHI the function
is set to zero (|W| < e^{-XHI/2} x^{|K|}).
"""
from __future__ import annotations
import math, cmath, os, hashlib
import numpy as np
from flint import acb, ctx

ctx.prec = 110
XLO = 1e-14
XHI = 150.0
WIDTH = 0.25
DEG = 16

def _a(z):
    return acb(complex(z).real, complex(z).imag)

def whitW(k, m, x):
    """W_{k,m}(x), x>0, via U: W = e^{-x/2} x^{m+1/2} U(1/2+m-k, 1+2m, x)."""
    X = acb(float(x)); K = _a(k); M = _a(m)
    v = (-X / 2).exp() * X ** (M + acb(0.5)) * X.hypgeom_u(acb(0.5) + M - K, 1 + 2 * M)
    return complex(float(v.real.mid()), float(v.imag.mid()))

def whitM(k, m, x):
    """M_{k,m}(x) = e^{-x/2} x^{m+1/2} 1F1(1/2+m-k; 1+2m; x), x>0."""
    X = acb(float(x)); K = _a(k); M = _a(m)
    v = (-X / 2).exp() * X ** (M + acb(0.5)) * X.hypgeom_1f1(acb(0.5) + M - K, 1 + 2 * M)
    return complex(float(v.real.mid()), float(v.imag.mid()))

def gamma(z):
    v = _a(z).gamma()
    return complex(float(v.real.mid()), float(v.imag.mid()))

def conn(k, nu):
    """Connection coefficients W_{k,nu} = C+ M_{k,nu} + C- M_{k,-nu}."""
    return gamma(-2 * nu) / gamma(0.5 - nu - k), gamma(2 * nu) / gamma(0.5 + nu - k)


def _cheb(deg):
    j = np.arange(deg + 1)
    s = -np.cos(np.pi * j / deg)
    w = ((-1.0) ** j) * np.where((j == 0) | (j == deg), 0.5, 1.0)
    return s, w


class Table:
    """f(x) = e^{x/2} F(x) tabulated on panels in ln x; F given by a callable
    returning complex values at a float x; below XLO: small(x) callable (numpy)."""

    def __init__(self, func, small, key=None, cache_dir=None):
        self.L0 = math.log(XLO)
        self.npan = int(math.ceil((math.log(XHI) - self.L0) / WIDTH))
        self.s, self.bw = _cheb(DEG)
        self.lnxc = self.L0 + WIDTH * (np.arange(self.npan) + 0.5)
        X = np.exp(self.lnxc[:, None] + 0.5 * WIDTH * self.s[None, :])
        self.small = small
        path = None
        if cache_dir and key:
            h = hashlib.sha1(repr(key).encode()).hexdigest()[:20]
            path = os.path.join(cache_dir, f"tab_{h}.npy")
        if path and os.path.exists(path):
            self.V = np.load(path)
        else:
            V = np.empty(X.shape, complex)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x = X[i, j]
                    V[i, j] = math.exp(x / 2) * func(x)
            self.V = V
            if path:
                np.save(path, V)

    def __call__(self, x):
        """F(x) (not scaled) for an array x."""
        x = np.asarray(x, float)
        out = np.zeros(x.shape, complex)
        lo = x < XLO
        hi = x > XHI
        mid = ~(lo | hi)
        if np.any(lo):
            out[lo] = self.small(x[lo])
        if np.any(mid):
            xm = x[mid]
            lx = np.log(xm)
            p = np.clip(np.floor((lx - self.L0) / WIDTH).astype(np.int64), 0, self.npan - 1)
            loc = (lx - self.lnxc[p]) / (0.5 * WIDTH)
            d = loc[:, None] - self.s[None, :]
            ex = d == 0.0
            d = np.where(ex, 1.0, d)
            c = self.bw[None, :] / d
            v = (c * self.V[p]).sum(1) / c.sum(1)
            if np.any(ex.any(1)):
                r = np.nonzero(ex.any(1))[0]
                v[r] = self.V[p[r], np.argmax(ex[r], 1)]
            out[mid] = v * np.exp(-xm / 2)
        return out


class Line:
    """Soft-end tables and late-time coefficients for one line type at (mu, kappa).
    kind: 'T' with helicity h = +-1, or 'L' (longitudinal; h = 0 functions)."""

    def __init__(self, mu, kappa, kind, h=0, cache_dir=None):
        self.mu, self.kappa, self.kind, self.h = float(mu), float(kappa), kind, int(h)
        nu = 1j * self.mu
        self.nu = nu
        self.m = math.sqrt(self.mu ** 2 + 0.25)
        hk = self.h * self.kappa if kind == 'T' else 0.0
        self.hk = hk
        K = -1j * hk                    # first index of u
        self.K = K
        # u*(it) = Nfac/sqrt(2y) * W_{-K,nu}(z)
        self.Nfac = math.exp(math.pi * hk / 2)
        Cp, Cm = conn(-K, nu)           # W_{-K,nu} = Cp M_{-K,nu} + Cm M_{-K,-nu}
        self.Cp, self.Cm = Cp, Cm
        # late-time coefficients (without 1/sqrt(2y)) of z^{1/2+nu} [branch +] and z^{1/2-nu}
        Bp, Bm = self.Nfac * Cp, self.Nfac * Cm
        Ap = self.Nfac * cmath.exp(-1j * math.pi * (0.5 + nu)) * gamma(-2 * nu) / gamma(0.5 - nu - K)
        Am = self.Nfac * cmath.exp(-1j * math.pi * (0.5 - nu)) * gamma(2 * nu) / gamma(0.5 + nu - K)
        self.A = {+1: Ap, -1: Am}
        self.B = {+1: Bp, -1: Bm}
        key = (round(self.mu, 12), round(-K.imag, 12))
        kw = complex(-K); m_ = nu
        cp, cm = Cp, Cm
        def small_W(x, cp=cp, cm=cm, kw=kw, nu=nu):
            e1p = -kw / (1 + 2 * nu); e1m = -kw / (1 - 2 * nu)
            return cp * x ** (0.5 + nu) * (1 + e1p * x) + cm * x ** (0.5 - nu) * (1 + e1m * x)
        self.W = Table(lambda x: whitW(kw, m_, x), small_W, key=("W",) + key, cache_dir=cache_dir)
        if kind == 'L':
            # (x d_x - 1) W_{0,nu}(x) = (x/2 - 1) W_{0,nu} - W_{1,nu}   (DLMF 13.15.23 with k=0)
            def small_D(x, cp=cp, cm=cm, nu=nu):
                return cp * (nu - 0.5) * x ** (0.5 + nu) + cm * (-nu - 0.5) * x ** (0.5 - nu)
            self.D = Table(lambda x: (x / 2 - 1) * whitW(0, m_, x) - whitW(1, m_, x), small_D,
                           key=("D",) + key, cache_dir=cache_dir)

    # ----- hard-end coefficient of (2 y t1)^{1/2+nu}, without the 1/sqrt(2y)
    def hard(self, a, b):
        """a, b = +-1 (branch of vertex 1 and vertex 2)."""
        if a == 1 and b == 1:
            c = self.A[1]                    # e(it1)
            f = 1j / self.m
        elif a == 1 and b == -1:
            c = self.B[1]                    # e*(it1)
            f = -1j / self.m
        elif a == -1 and b == 1:
            c = np.conj(self.B[-1])          # e(-it1) = conj(e*(it1))
            f = 1j / self.m
        else:
            c = np.conj(self.A[-1])          # e*(-it1) = conj(e(it1))
            f = -1j / self.m
        if self.kind == 'T':
            return c
        return f * (self.nu - 0.5) * c       # A_L end, leading rung

    # ----- soft-end functions for b=+ (e*(it2)), without 1/sqrt(2y); b=- is conj.
    def soft(self, comp, x):
        """comp in {'T','L','0'}; x = 2 y tau."""
        if comp == 'T':
            return self.Nfac * self.W(x)
        if comp == 'L':
            return (-1j / self.m) * self.D(x)
        if comp == '0':
            return (-0.5j / self.m) * x * self.W(x)
        raise ValueError(comp)

    def soft_late(self, comp):
        """Coefficient of x^{1/2+nu} in soft(comp, x) (late-time +nu branch) for b=+."""
        Bp = self.Nfac * self.Cp
        if comp == 'T':
            return Bp
        if comp == 'L':
            return (-1j / self.m) * (self.nu - 0.5) * self.Cp
        if comp == '0':
            return 0.0
