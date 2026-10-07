"""Residue families of the one-loop bubble.

All family coefficients are computed in the unit frame x_A = 2, x_B = 2, k = 2 (r = 1); the
r-dependence is restored per family.  mpmath at `dps` digits for the line functions and the
B-functions; the two-fold integrals by discrete convolution (float64 direct convolution of the
normalised lines when npconv is on); the J and M families of D_IK and XX on a float64 path.
"""
from __future__ import annotations

import math
import os
import time

import mpmath as mp
import numpy as np
from scipy.special import loggamma as lg64

ELL = None          # (4 pi)^(-3/2), set at the working precision


# ============================================================================ problem frame


class Problem:
    """Frame of the bubble: k1 = k2 = 1 on vertex A, k3 = 2r on vertex B, c_s = 1."""

    def __init__(self, name="fig3c"):
        if name == "fig3c":
            mu = "1.472"
        elif name.startswith("mu") and name[2:].isdigit():
            mu = str(int(name[2:]) / 100.0)
        else:
            raise ValueError(f"unknown problem {name!r}")
        self.name = name
        self.mu_str = mu
        self.cs = 1
        self.pair = 2

    def setup(self):
        self.mu = mp.mpf(self.mu_str)
        self.nu1 = mp.mpc(0, self.mu)
        self.nu2 = mp.mpc(0, self.mu)
        self.muR = mp.mpf(self.mu_str)
        self.xA = mp.mpf(2)
        self.xB = mp.mpf(2)
        self.k = mp.mpf(2)


# ============================================================================ Gamma products


def _is_nonpos_int(z):
    z = mp.mpmathify(z)
    if isinstance(z, mp.mpc):
        if z.imag != 0:
            return False
        z = z.real
    return z <= 0 and z == mp.floor(z)


def gprod(num, den, extra=0):
    """exp(sum loggamma(num) - sum loggamma(den) + extra); exact zero at a reciprocal pole."""
    for d in den:
        if _is_nonpos_int(d):
            return mp.mpc(0)
    for n in num:
        if _is_nonpos_int(n):
            raise ValueError("gamma pole in numerator")
    s = extra
    for n in num:
        s += mp.loggamma(n)
    for d in den:
        s -= mp.loggamma(d)
    return mp.exp(s)


def gprod64(num, den, extra=0.0):
    """float64 version on arrays; non-finite results replaced by zero."""
    s = extra
    for n in num:
        s = s + lg64(n)
    for d in den:
        s = s - lg64(d)
    with np.errstate(all="ignore"):
        v = np.exp(s)
    v = np.where(np.isfinite(v), v, 0.0)
    return v


# ============================================================================ grids and integrals


class Line:
    """Vertical line Re b = c sampled at b_j = c + i j h, j = -n..n, n = ceil(T/h)."""

    def __init__(self, c, h, T):
        self.c = mp.mpf(c)
        self.h = mp.mpf(h)
        self.n = int(mp.ceil(mp.mpf(T) / self.h))
        self.pts = [mp.mpc(self.c, j * self.h) for j in range(-self.n, self.n + 1)]

    def sum_points(self, other):
        """B_k = c1 + c2 + i k h, k = -(n1+n2)..(n1+n2) (same h)."""
        n = self.n + other.n
        return [mp.mpc(self.c + other.c, k * self.h) for k in range(-n, n + 1)]


def onefold(vals, h):
    return h / (2 * mp.pi) * mp.fsum(vals)


def convolve(f1, f2, h, npconv=True):
    """F_k = h/(2pi) sum_j f1[j] f2[k-j] (list of mp numbers)."""
    if npconv:
        s1 = max(abs(v) for v in f1)
        s2 = max(abs(v) for v in f2)
        if s1 == 0 or s2 == 0:
            return [mp.mpc(0)] * (len(f1) + len(f2) - 1)
        a1 = np.array([complex(v / s1) for v in f1])
        a2 = np.array([complex(v / s2) for v in f2])
        F = np.convolve(a1, a2)
        sc = s1 * s2 * h / (2 * mp.pi)
        return [sc * mp.mpc(z.real, z.imag) for z in F]
    n1, n2 = len(f1), len(f2)
    out = []
    for k in range(n1 + n2 - 1):
        out.append(h / (2 * mp.pi) * mp.fsum(f1[j] * f2[k - j] for j in range(max(0, k - n2 + 1), min(n1, k + 1))))
    return out


def twofold(F, g, h):
    return h / (2 * mp.pi) * mp.fsum(gk * Fk for gk, Fk in zip(g, F))


# ============================================================================ line and B functions


class Families:
    """All families of the three sections for one problem and one parameter set."""

    def __init__(self, prob, dps=20, Nmax=24, Mmax=24, h="0.05", hU="0.025", T=12, npconv=True,
                 cb=None, cb0=None, cb2=None):
        global ELL
        self.prob = prob
        self.dps = dps
        mp.mp.dps = dps
        prob.setup()
        ELL = (4 * mp.pi) ** mp.mpf(-1.5)
        self.ell = ELL
        self.Nmax, self.Mmax, self.T = Nmax, Mmax, T
        self.h, self.hU = mp.mpf(h), mp.mpf(hU)
        self.npconv = npconv
        env = os.environ
        self.cb = mp.mpf(env.get("LB_CB", cb or "0.625"))      # D_II two-folds
        self.cb0 = mp.mpf(env.get("LB_CB0", cb0 or "0.75"))    # one-folds
        self.cb2 = mp.mpf(env.get("LB_CB2", cb2 or "-0.28"))   # KK line of D_IK
        self.cx = mp.mpf("0.05")                               # XX lines
        self.hw = float(env.get("LB_HW", "0.03"))
        self.tw = float(env.get("LB_TW", "6"))
        self.hwx = float(env.get("LB_HWX", "0.025"))
        self.twx = float(env.get("LB_TWX", "3"))
        self._cache = {}

    # -- elementary functions --------------------------------------------------------
    def fKI(self, n, nu, b):
        return gprod([b, mp.mpf(1) / 2 + b, 1 + nu + n - b], [1 + nu + n, b - n])

    def fKK(self, sig, m, nu, b):
        return gprod([-sig * nu - m, 1 - b + sig * nu / 2 + m + nu / 2, 1 - b + sig * nu / 2 + m - nu / 2,
                      mp.mpf(1) / 2 + b], [1 - b])

    def L(self, b):
        return gprod([b, mp.mpf(1) / 2 + b], [])

    def LX(self, b):
        return gprod([mp.mpf(1) / 2 + b], [1 - b])

    def vA(self, Sig):
        p = self.prob
        return gprod([4 + Sig], [], extra=-(4 + Sig) * mp.log(p.xA))

    def g(self, Sig, B):
        p = self.prob
        return gprod([2 * B - 2 - Sig, mp.mpf(1) / 2 - B], [1 + B],
                     extra=(2 + Sig - 2 * B) * mp.log(p.xB) + (2 * B - 1) * mp.log(p.k)
                     + (4 - 2 * B) * mp.log(2))

    def G0(self, m, B):
        p = self.prob
        return gprod([2 + 2 * B + m, mp.mpf(1) / 2 - B], [1 + B],
                     extra=(-2 - 2 * B - m) * mp.log(p.xA) + m * mp.log(p.xB)
                     + (2 * B - 1) * mp.log(p.k) + (4 - 2 * B) * mp.log(2))

    # -- cached tables ---------------------------------------------------------------------
    def line(self, c, h):
        key = ("line", str(c), str(h))
        if key not in self._cache:
            self._cache[key] = Line(c, h, self.T)
        return self._cache[key]

    def tab(self, key, fn, pts):
        if key not in self._cache:
            self._cache[key] = [fn(b) for b in pts]
        return self._cache[key]

    def tabKI(self, n, nu, c, h):
        ln = self.line(c, h)
        return self.tab(("KI", n, str(nu), str(c), str(h)), lambda b: self.fKI(n, nu, b), ln.pts)

    def tabKK(self, s, m, nu, c, h):
        ln = self.line(c, h)
        return self.tab(("KK", s, m, str(nu), str(c), str(h)), lambda b: self.fKK(s, m, nu, b), ln.pts)

    # -- r-dependence -----------------------------------------------------------------------
    @staticmethod
    def scale_power(val, Sig, rs):
        return [val * mp.mpf(r) ** (1 + Sig) for r in rs]

    def array_mode(self, P, Bs, mprime, rs):
        """sum_k P_k r^(2 B_k - 1 + m') for every ratio."""
        out = []
        for r in rs:
            lr = mp.log(mp.mpf(r))
            out.append(mp.fsum(Pk * mp.exp((2 * Bk - 1 + mprime) * lr) for Pk, Bk in zip(P, Bs)))
        return out

    # ===================================================================== section D_II
    def towers(self, rs):
        """T_N, R0T_N for N = 0..Nmax: dict key -> (Sigma, values per ratio)."""
        p = self.prob
        h = self.h
        ln = self.line(self.cb, h)
        Bs = ln.sum_points(ln)
        l1 = self.line(self.cb0, h)
        out = {}
        for N in range(self.Nmax + 1):
            Sig = p.nu1 + p.nu2 + 2 * N
            gt = [self.g(Sig, B) for B in Bs]
            acc = mp.mpc(0)
            acc0 = mp.mpc(0)
            for n1 in range(N + 1):
                n2 = N - n1
                f1 = self.tabKI(n1, p.nu1, self.cb, h)
                f2 = self.tabKI(n2, p.nu2, self.cb, h)
                F = convolve(f1, f2, h, self.npconv)
                fac = 1 / (mp.factorial(n1) * mp.factorial(n2))
                acc += fac * twofold(F, gt, h)
                vals = [self.fKI(n1, p.nu1, b) * self.fKI(n2, p.nu2, mp.mpf(1) / 2 - b) for b in l1.pts]
                acc0 += fac * onefold(vals, h)
            sgn = (-1) ** N
            T = self.ell / 4 * self.vA(Sig) * sgn * acc
            R0 = (self.ell / 4 * self.vA(Sig) * gprod([-1 - Sig], []) * p.xB ** (1 + Sig) * 8
                  / mp.gamma(mp.mpf(3) / 2) * sgn * acc0)
            out[f"T{N}"] = (Sig, self.scale_power(T, Sig, rs), T)
            out[f"R0T{N}"] = (Sig, self.scale_power(R0, Sig, rs), R0)
        return out

    def _G0tab(self, m, Bs):
        key = ("G0", m, len(Bs), str(Bs[0]))
        if key not in self._cache:
            self._cache[key] = [self.G0(m, B) for B in Bs]
        return self._cache[key]

    def U_term(self, m, n2, rs):
        """U_{m', n2} for all ratios (array mode)."""
        p = self.prob
        h = self.hU
        ln = self.line(self.cb, h)
        Bs = ln.sum_points(ln)
        Lt = self.tab(("L", str(self.cb), str(h)), self.L, ln.pts)
        d = mp.mpf(m) / 2 - n2 + (p.nu1 - p.nu2) / 2
        fKI = self.tabKI(n2, p.nu2, self.cb, h)
        fU = [fk * gprod([b + d], [1 + p.nu1 - b - d]) for fk, b in zip(fKI, ln.pts)]
        G0 = self._G0tab(m, Bs)
        GU = []
        for B, g0 in zip(Bs, G0):
            s = B - 1 + mp.mpf(m) / 2 - (p.nu1 + p.nu2) / 2 - n2
            GU.append(gprod([-s], [1 + p.nu1 + s]) * g0)
        F = convolve(Lt, fU, h, self.npconv)
        pref = self.ell / 4 / 2 * (-1) ** (m + n2) / (mp.factorial(m) * mp.factorial(n2))
        P = [pref * h / (2 * mp.pi) * gk * Fk for gk, Fk in zip(GU, F)]
        return self.array_mode(P, Bs, m, rs)

    def V_term(self, m, j, rs):
        p = self.prob
        h = self.hU
        dp = mp.mpf(m) / 2 + (p.nu1 - p.nu2) / 2 + j
        if _is_nonpos_int(-dp):
            return [mp.mpc(0)] * len(rs)
        ln = self.line(self.cb, h)
        Bs = ln.sum_points(ln)
        f1 = [fk * mp.gamma(1 + p.nu1 + j) for fk in self.tabKI(j, p.nu1, self.cb, h)]
        f2 = [gprod([b, mp.mpf(1) / 2 + b, -b - dp], [1 + p.nu2 + b + dp]) for b in ln.pts]
        G0 = self._G0tab(m, Bs)
        F = convolve(f1, f2, h, self.npconv)
        pref = (self.ell / 4 / 2 * (-1) ** (m + j) / (mp.factorial(m) * mp.factorial(j))
                * gprod([1 + p.nu2 + dp], [1 + p.nu1 + j, -dp]))
        P = [pref * h / (2 * mp.pi) * gk * Fk for gk, Fk in zip(G0, F)]
        return self.array_mode(P, Bs, m, rs)

    # -- rung families ------------------------------------------------------
    def _P(self, m, eps):
        p = self.prob
        return (self.ell / 4 / 2 * mp.exp(2 * mp.pi * 1j * eps) * (4 * mp.pi * p.muR ** 2) ** eps
                * mp.mpf(2) ** (3 - 2 * eps) / mp.gamma(mp.mpf(3) / 2 - eps) * (-1) ** m
                / mp.factorial(m) * mp.gamma(3 - 2 * eps + m) * p.xA ** (-3 + 2 * eps - m) * p.xB ** m)

    @staticmethod
    def neville0(xs, ys):
        """Polynomial extrapolation to x = 0 (Neville)."""
        P = list(ys)
        n = len(xs)
        for k in range(1, n):
            for i in range(n - k):
                P[i] = ((0 - xs[i + k]) * P[i] + (xs[i] - 0) * P[i + 1]) / (xs[i] - xs[i + k])
        return P[0]

    def rung(self, m, kind="U", delta="1.0e-8", N0=24, K=8):
        """(finite part, pole coefficient) of R0U_{m'} or R0V_{m'} in the unit frame."""
        p = self.prob
        nu1, nu2 = p.nu1, p.nu2
        sig = (-1 + m - nu1 - nu2) / 2
        z = 1 + nu1 + sig
        ln = self.line(self.cb0, self.hU)
        h = self.hU
        half = mp.mpf(1) / 2
        delta = mp.mpf(delta)

        def Gcal(b1, eps):
            b2 = half + eps - b1
            base = gprod([b1, half - eps + b1, b2, 1 - b1], [])
            if kind == "U":
                return base * mp.sin(mp.pi * b2) / (mp.pi * mp.sin(mp.pi * (z - b1)))
            B = half + eps
            B0 = B - 1 - nu1 - sig
            c = sig + 1 + nu1 - b1
            return -base * mp.sin(mp.pi * b1) * mp.sin(mp.pi * B0) / (mp.pi * mp.sin(mp.pi * c))

        def Phi(eps):
            vals = [Gcal(b1, eps) for b1 in ln.pts]
            I = onefold(vals, h)
            if kind == "U":
                return self._P(m, eps) * mp.sin(mp.pi * z) * I
            return self._P(m, eps) * I

        def Q(n, b1):
            b2 = half - b1
            if kind == "U":
                return gprod([n - sig, n + 1 + nu2 - b2, n + 1 - b2],
                             [n + 1, n + b1 - nu1 - sig, n + b1 - sig, n + 1 + nu2, z - n],
                             extra=mp.log(mp.pi)) * (-1) ** n
            B = half
            B0 = B - 1 - nu1 - sig
            c = sig + 1 + nu1 - b1
            A0 = 2 + nu1 + nu2 + sig - B
            return gprod([1 + nu1 + n - b1, 1 - b1 + n, A0 + n, 1 - B0 + n],
                         [1 + n, 1 + nu1 + n, 1 + c + n, 1 + nu2 + c + n])

        sub = mp.sin(mp.pi * z) if kind == "U" else mp.mpf(1)
        remv = []
        for b1 in ln.pts:
            s = Q(0, b1)
            seq = []
            for j in range(1, N0 + K + 1):
                s += Q(j, b1) - sub / j
                if j >= N0:
                    seq.append(s)
            xs = [mp.mpf(1) / n for n in range(N0, N0 + K + 1)]
            ext = self.neville0(xs, seq)
            remv.append(Gcal(b1, 0) * ext)
        if kind == "U":
            Rem = self._P(m, 0) * onefold(remv, h)
        else:
            Rem = self._P(m, 0) * onefold(remv, h)
        P0 = Phi(0)
        dP = (Phi(delta) - Phi(-delta)) / (2 * delta)
        eg = mp.euler
        fin = dP / 2 + P0 * (eg + (eg - mp.log(4 * mp.pi)) / 2) + Rem
        return fin, P0 / 2

    def R00(self, r):
        p = self.prob
        kT = 2 + 2 * mp.mpf(r)
        eg = mp.euler
        f1 = (-2 * mp.log(2) + mp.log(4 * mp.pi * p.muR ** 2) + mp.digamma(mp.mpf(3) / 2)
              - 2 * mp.digamma(3) + 2 * mp.log(kT) - 2 * mp.digamma(mp.mpf(1) / 2 + p.nu1)
              - 2 * mp.digamma(mp.mpf(1) / 2))
        return kT ** -3 / (4 * mp.pi ** 2) * (f1 - 2 * eg + 2 * mp.pi * 1j - (-eg + mp.log(4 * mp.pi)))

    # ===================================================================== section D_IK
    def TIK(self, n1, sig, m, rs):
        p = self.prob
        h = self.hU
        l1, l2 = self.line(self.cb, h), self.line(self.cb2, h)
        Bs = l1.sum_points(l2)
        Sig = p.nu1 + 2 * n1 + sig * p.nu2 + 2 * m
        f1 = self.tabKI(n1, p.nu1, self.cb, h)
        f2 = self.tabKK(sig, m, p.nu2, self.cb2, h)
        gt = [self.g(Sig, B) for B in Bs]
        val = (self.ell / 8 * self.vA(Sig) * (-1) ** (n1 + m) / (mp.factorial(n1) * mp.factorial(m))
               * twofold(convolve(f1, f2, h, self.npconv), gt, h))
        return Sig, self.scale_power(val, Sig, rs), val

    # -- float64 path for J and M ------------------------------------------------------------
    def _np_lines(self, c, h):
        n = int(math.ceil(self.T / h))
        return c + 1j * h * np.arange(-n, n + 1)

    def _G0_64(self, m, B):
        p = self.prob
        xA, xB, k = float(p.xA), float(p.xB), float(p.k)
        return gprod64([2 + 2 * B + m, 0.5 - B], [1 + B],
                       extra=(-2 - 2 * B - m) * math.log(xA) + m * math.log(xB)
                       + (2 * B - 1) * math.log(k) + (4 - 2 * B) * math.log(2.0))

    def _finish64(self, P, Bs, m, rs):
        Pm = [mp.mpc(z.real, z.imag) for z in P]
        Bm = [mp.mpc(z.real, z.imag) for z in Bs]
        return self.array_mode(Pm, Bm, m, rs)

    def c_IK(self, m):
        return 0.4675 if m == 0 else (0.65 if m % 2 else 0.33)

    def c_XX(self, m):
        return 0.2 if m % 2 else 0.45

    def JIK(self, m, rs):
        p = self.prob
        nu1, nu2 = complex(p.nu1), complex(p.nu2)
        h = float(self.hU)
        b1 = self._np_lines(float(self.cb), h)
        b2 = self._np_lines(float(self.cb2), h)
        Bs = (float(self.cb) + float(self.cb2)) + 1j * h * np.arange(-(len(b1) - 1), len(b1))
        Lt = gprod64([b1, 0.5 + b1], [])
        nw = int(round(self.tw / self.hw))
        ws = self.c_IK(m) + 1j * self.hw * np.arange(-nw, nw + 1)
        G0 = self._G0_64(m, Bs)
        acc = np.zeros(Bs.size, dtype=complex)
        for w in ws:
            a = m / 2 + nu1 / 2 + w
            d = 1 - m / 2 + nu1 / 2 - w
            l2 = gprod64([w + nu2 / 2, w - nu2 / 2, 1 - b2 - w + nu2 / 2, 1 - b2 - w - nu2 / 2, 0.5 + b2,
                          b2 + a], [1 - b2, d - b2])
            s = Bs - 1 + m / 2 - nu1 / 2 + w
            G = gprod64([-s], [1 + nu1 + s]) * G0
            F = np.convolve(Lt, l2) * (h / (2 * np.pi))
            acc += G * F * (h / (2 * np.pi))
        acc *= self.hw / (2 * np.pi)
        pref = float(self.ell) / 8 / 2 * (-1) ** m / math.factorial(m)
        return self._finish64(acc * pref, Bs, m, rs)

    def MIK_included(self, n1, m):
        return 0.655 + n1 - m / 2 < self.c_IK(m)

    def MIK(self, n1, m, rs):
        p = self.prob
        nu1, nu2 = complex(p.nu1), complex(p.nu2)
        h = float(self.hU)
        b1 = self._np_lines(float(self.cb), h)
        b2 = self._np_lines(float(self.cb2), h)
        Bs = (float(self.cb) + float(self.cb2)) + 1j * h * np.arange(-(len(b1) - 1), len(b1))
        e = -nu1 / 2 - n1 + m / 2
        fki = gprod64([b1, 0.5 + b1, 1 + nu1 + n1 - b1], [1 + nu1 + n1, b1 - n1])
        l1 = fki * gprod64([b1 + e + nu2 / 2, b1 + e - nu2 / 2], [])
        LX = gprod64([0.5 + b2], [1 - b2])
        w = 1 - Bs + nu1 / 2 + n1 - m / 2
        G = gprod64([w + nu2 / 2, w - nu2 / 2], []) * self._G0_64(m, Bs)
        F = np.convolve(l1, LX) * (h / (2 * np.pi))
        P = G * F * (h / (2 * np.pi))
        pref = (float(self.ell) / 8 * (-1) ** n1 / math.factorial(n1) / 2 * (-1) ** m
                / math.factorial(m))
        return self._finish64(P * pref, Bs, m, rs)

    # ===================================================================== section XX
    def TXX(self, s1, m1, s2, m2, rs):
        p = self.prob
        h = self.hU
        ln = self.line(self.cx, h)
        Bs = ln.sum_points(ln)
        Sig = s1 * p.nu1 + s2 * p.nu2 + 2 * (m1 + m2)
        key = ("gXX", str(Sig))
        if key not in self._cache:
            self._cache[key] = [self.g(Sig, B) for B in Bs]
        gt = self._cache[key]
        f1 = self.tabKK(s1, m1, p.nu1, self.cx, h)
        f2 = self.tabKK(s2, m2, p.nu2, self.cx, h)
        val = (self.ell / 16 * self.vA(Sig) * (-1) ** (m1 + m2) / (mp.factorial(m1) * mp.factorial(m2))
               * twofold(convolve(f1, f2, h, self.npconv), gt, h))
        return Sig, self.scale_power(val, Sig, rs), val

    def JXX(self, m, rs):
        p = self.prob
        nu1, nu2 = complex(p.nu1), complex(p.nu2)
        h = float(self.hU)
        b = self._np_lines(float(self.cx), h)
        Bs = 2 * float(self.cx) + 1j * h * np.arange(-(len(b) - 1), len(b))
        LX = gprod64([0.5 + b], [1 - b])
        nw = int(round(self.twx / self.hwx))
        ws = self.c_XX(m) + 1j * self.hwx * np.arange(-nw, nw + 1)
        G0 = self._G0_64(m, Bs)
        acc = np.zeros(Bs.size, dtype=complex)
        for w2 in ws:
            a = w2 + m / 2
            l2 = gprod64([w2 + nu2 / 2, w2 - nu2 / 2, 1 - b - w2 + nu2 / 2, 1 - b - w2 - nu2 / 2, 0.5 + b,
                          b + a + nu1 / 2, b + a - nu1 / 2], [1 - b])
            w1 = 1 - Bs - w2 - m / 2
            G = gprod64([w1 + nu1 / 2, w1 - nu1 / 2], []) * G0
            F = np.convolve(LX, l2) * (h / (2 * np.pi))
            acc += G * F * (h / (2 * np.pi))
        acc *= self.hwx / (2 * np.pi)
        pref = float(self.ell) / 16 / 2 * (-1) ** m / math.factorial(m)
        return self._finish64(acc * pref, Bs, m, rs)

    def MXX_included(self, m1, m):
        return 0.9 + m1 - m / 2 < self.c_XX(m)

    def MXX(self, s1, m1, m, rs):
        p = self.prob
        nu1, nu2 = complex(p.nu1), complex(p.nu2)
        h = float(self.hU)
        b = self._np_lines(float(self.cx), h)
        Bs = 2 * float(self.cx) + 1j * h * np.arange(-(len(b) - 1), len(b))
        w1 = -s1 * nu1 / 2 - m1
        e = w1 + m / 2
        fkk = gprod64([-s1 * nu1 - m1 + 0 * b, 1 - b + s1 * nu1 / 2 + m1 + nu1 / 2,
                       1 - b + s1 * nu1 / 2 + m1 - nu1 / 2, 0.5 + b], [1 - b])
        l1 = fkk * gprod64([b + e + nu2 / 2, b + e - nu2 / 2], [])
        LX = gprod64([0.5 + b], [1 - b])
        w2 = 1 - Bs - w1 - m / 2
        G = gprod64([w2 + nu2 / 2, w2 - nu2 / 2], []) * self._G0_64(m, Bs)
        F = np.convolve(l1, LX) * (h / (2 * np.pi))
        P = G * F * (h / (2 * np.pi))
        pref = float(self.ell) / 16 * (-1) ** m1 / math.factorial(m1) / 2 * (-1) ** m / math.factorial(m)
        return self._finish64(P * pref, Bs, m, rs)


# ============================================================================ n-sums and tail rule

TAIL_TABLE = {"U": {0: (5.3, 1.6e-09), 2: (6.2, 1.1e-09), 10: (7.2, 5.1e-07), 20: (8.8, 1e-07)},
              "V": {1: (6.3, 1.2e-08)}}


def tail_params(fam, m):
    tab = TAIL_TABLE[fam]
    ms = sorted(tab)
    if m <= ms[0]:
        return tab[ms[0]]
    if m >= ms[-1]:
        if fam == "U":
            p20, e20 = tab[20]
            p10 = tab[10][0]
            return min(12.0, p20 + (m - 20) * (p20 - p10) / 10.0), e20
        return tab[ms[-1]]
    for a, b in zip(ms[:-1], ms[1:]):
        if a <= m <= b:
            pa, ea = tab[a]
            pb, eb = tab[b]
            t = (m - a) / (b - a)
            return pa + t * (pb - pa), math.exp(math.log(ea) + t * (math.log(eb) - math.log(ea)))


def n_terms(fam, m, rmax, tol=1e-11, tailscale=1.3):
    """Number of terms of the n-sum of family U or V at m'."""
    if fam == "U" and m % 2 == 1:
        return 16
    if fam == "V" and m % 2 == 0:
        return 0
    p, e32 = tail_params(fam, m)
    e = e32 * (rmax / 0.52) ** (m + 3)
    Nfit = 32 * (e / tol) ** (1.0 / p) if e > tol else 8
    Nrule = math.ceil(max(Nfit, m + 35))
    return math.ceil(tailscale * Nrule)


def plan(rmax, mpmax=80, tailscale=1.3):
    tasks = [("T", None, 0)]
    for m in range(mpmax + 1):
        tasks.append(("U", m, n_terms("U", m, rmax, tailscale=tailscale)))
        if m % 2 == 1:
            tasks.append(("V", m, n_terms("V", m, rmax, tailscale=tailscale)))
    return tasks


def hurwitz_tail(terms, partials, K=3):
    """Value and error of an n-sum with the Hurwitz-zeta tail (one ratio)."""
    N = len(terms)
    last = terms[-1]
    S = partials[-1]
    if abs(last) < mp.mpf("1e-30") * max(abs(S), mp.mpf("1e-300")):
        return S, abs(last)
    ns = list(range(N - K, N))
    A = mp.matrix(K, K)
    rhs = mp.matrix(K, 1)
    for i, n in enumerate(ns):
        for j in range(K):
            A[i, j] = mp.mpf(n) ** (-(3 + j))
        rhs[i] = terms[n]
    c = mp.lu_solve(A, rhs)
    tail = mp.fsum(c[j] * mp.zeta(3 + j, N) for j in range(K))
    val = S + tail
    xs = [mp.mpf(1) / (n + 1) for n in range(N - K - 1, N)]
    ext = Families.neville0(xs, partials[N - K - 1:N])
    return val, abs(val - ext)


def geometric_sum(terms, partials):
    t1, t2 = terms[-1], terms[-2]
    q = abs(t1 / t2) if t2 != 0 else mp.mpf(1)
    err = abs(t1) * q / (1 - q) if q < mp.mpf("0.9") else 10 * abs(t1)
    return partials[-1], err


def nsum_family(fam_obj, fam, m, rs, N, Ktail=3, termdump=None):
    """U_{m'} or V_{m'} with N terms; returns (values per ratio, errors per ratio)."""
    nr = len(rs)
    terms = [[] for _ in range(nr)]
    parts = [[] for _ in range(nr)]
    acc = [mp.mpc(0)] * nr
    for n in range(N):
        t = fam_obj.U_term(m, n, rs) if fam == "U" else fam_obj.V_term(m, n, rs)
        for i in range(nr):
            if not mp.isfinite(t[i]):
                raise FloatingPointError(f"non-finite term {fam}{m} n={n}")
            terms[i].append(t[i])
            acc[i] += t[i]
            parts[i].append(acc[i])
    if termdump is not None:
        termdump.append({"fam": fam, "mp": m, "N": N, "terms": [[mp.nstr(z, 25) for z in tt] for tt in terms]})
    vals, errs = [], []
    for i in range(nr):
        if fam == "U" and m % 2 == 1:
            v, e = geometric_sum(terms[i], parts[i])
        else:
            v, e = hurwitz_tail(terms[i], parts[i], Ktail)
        vals.append(v)
        errs.append(e)
    return vals, errs


# ============================================================================ assembly


def grading(key):
    """Grading of a residue key; None for E and P keys."""
    import re
    if key.startswith(("E", "P")):
        return None
    m = re.fullmatch(r"(R0)?T(\d+)", key)
    if m:
        return 2 * int(m.group(2))
    m = re.fullmatch(r"T(\d+)_[+-](\d+)", key)
    if m:
        return 2 * (int(m.group(1)) + int(m.group(2)))
    m = re.fullmatch(r"T[+-](\d+)_[+-](\d+)", key)
    if m:
        return 2 * (int(m.group(1)) + int(m.group(2)))
    m = re.fullmatch(r"(U|V|R0U|R0V|J)(\d+)", key)
    if m:
        return int(m.group(2))
    m = re.fullmatch(r"M[+-]?\d+_(\d+)", key)
    if m:
        return int(m.group(1))
    return None


def assemble(cache, fam_obj, rs, gmax=None):
    """House-scheme assembly.  cache: {'DII': {key: (Sig, vals)}, ...}."""
    p = fam_obj.prob
    lam = -1j / mp.pi * mp.exp(-mp.pi * p.mu)
    out = []
    for i, r in enumerate(rs):
        sec = {}
        pole = mp.mpc(0)
        err = mp.mpf(0)
        for s in ("DII", "DIK", "XX"):
            tot = mp.mpc(0)
            for key, (Sig, vals) in cache[s].items():
                if key.startswith("P"):
                    pole += vals[i]
                    continue
                if key.startswith("E"):
                    err += abs(vals[i])
                    continue
                if key.startswith("M"):
                    mm = key.split("_")[-1]
                    pre = key[1:].split("_")[0]
                    m = int(mm)
                    if s == "DIK":
                        if not fam_obj.MIK_included(int(pre), m):
                            continue
                    else:
                        if not fam_obj.MXX_included(int(pre[1:]), m):
                            continue
                if gmax is not None:
                    gr = grading(key)
                    if gr is not None and gr > gmax:
                        continue
                tot += vals[i]
            sec[s] = tot
        rr = mp.mpf(r)
        sec["DII"] += fam_obj.R00(rr)
        Bc = sec["DII"] + 2 * lam * sec["DIK"] + (lam * lam + 1 / mp.pi ** 2) * sec["XX"]
        cA_cB = rr ** 2 / 2
        Sh = 2 * (2 * rr / p.cs ** 3 * 2 * mp.re(cA_cB * p.pair * Bc))
        kT = 2 + 2 * rr
        RS = rr ** 3 / (8 * mp.pi ** 2 * (1 + rr) ** 3)
        Gp = 8 * mp.pi ** 2 * kT ** 3 * (pole + kT ** -3 / (4 * mp.pi ** 2)) - 1
        out.append({"r": r, "S_house": Sh, "R_S": RS, "DII": sec["DII"], "DIK": sec["DIK"],
                    "XX": sec["XX"], "G_pole_rel": Gp, "err_richardson": 8 * rr ** 3 * err})
    return out





def scheme_chain(S_house, r):
    r = mp.mpf(r)
    eg = mp.euler
    RS = r ** 3 / (8 * mp.pi ** 2 * (1 + r) ** 3)
    kT = 2 * (1 + r)
    S5 = S_house - RS * ((eg - mp.log(4 * mp.pi)) + 4 * (mp.log(kT) - mp.mpf(3) / 2 + eg))
    c0 = (3 * eg - mp.log(mp.pi) - 3) / 2
    Ss = S5 + 2 * RS * (c0 + mp.log(1 + r))
    return S5, Ss, RS


def signal_coeff(cache, fam_obj, rs):
    p = fam_obj.prob
    lam = -1j / mp.pi * mp.exp(-mp.pi * p.mu)
    w = {"DII": 1, "DIK": 2 * lam, "XX": lam * lam + 1 / mp.pi ** 2}
    target = p.nu1 + p.nu2
    r0 = mp.mpf(rs[0])
    C = mp.mpc(0)
    for s in ("DII", "DIK", "XX"):
        for key, (Sig, vals) in cache[s].items():
            if Sig is None or not key.startswith(("T", "R0T")):
                continue
            if abs(Sig - target) < mp.mpf("1e-20"):
                C += w[s] * vals[0] / r0 ** (1 + Sig)
            elif abs(Sig + target) < mp.mpf("1e-20") and key.startswith("T"):
                C += mp.conj(w[s] * vals[0] / r0 ** (1 + Sig))
    csig = 2 * 2 / p.cs ** 3 * 2 * mp.mpf(1) / 2 * p.pair * C
    return csig
