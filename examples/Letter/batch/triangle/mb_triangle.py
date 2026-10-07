"""Mellin-Barnes route of the one-loop triangle.

float64/complex128 throughout; Gamma products are exponentials of signed sums of log-Gamma.
Term A on the production lattice (routeA_point), the loop integral W_3, the closed Barnes-
residue form of the interchange term R and of D_0 (class TriPole), and the fixed-loop-momentum
representation (psi3_mb) used as a check.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import loggamma

EULER = 0.5772156649015329


def arange_sym(T, h):
    """Floating-point range -T, -T+h, ... with stop value T + h/2."""
    return np.arange(-T, T + h / 2.0, h)


def lg(z):
    return loggamma(z)


def line_log(s, u, nu):
    """log f_l(s; u) = log[Gamma(-s) Gamma(s-u) / (Gamma(nu+s+1) Gamma(nu+u-s+1))]."""
    return lg(-s) + lg(s - u) - lg(nu + s + 1.0) - lg(nu + u - s + 1.0)


def vertex_row(nu1, nu2, u, x, d, rv=1.0):
    """Gamma(rv + Sigma) x^-(rv + Sigma), Sigma = nu1 + nu2 + 2u + 2 i d (arrays broadcast)."""
    z = rv + nu1 + nu2 + 2.0 * u + 2j * d
    return np.exp(lg(z) - z * math.log(x))


def outer_weight(delta):
    """w_l = pi / sin(pi b_l) 2^(2 delta_l), b_l = 1 - delta_l."""
    return np.pi / np.sin(np.pi * (1.0 - delta)) * np.exp(2.0 * delta * math.log(2.0))


# ============================================================================ trace T(u)


class Trace:
    """T[a, b, c] = (h_s/2pi)^3 tr(M1^(a) M2^(b) M0^(c)) on a uniform s-lattice."""

    def __init__(self, mus, x, sig_s, taus, us, rv=1.0):
        self.mus = mus
        self.nus = [1j * m for m in mus]
        self.x = x
        self.taus = taus
        self.hs = taus[1] - taus[0]
        s = sig_s + 1j * taus
        n = len(taus)
        dd = (np.arange(-(n - 1), n) * self.hs)                       # tau_j - tau_i
        nuA, nuB, nuC = self.nus
        self.us = us                                                   # (3, Nv)
        # line factors (Nv, Ns)
        self.fA = np.exp(line_log(s[None, :], us[0][:, None], nuA))
        self.fB = np.exp(line_log(s[None, :], us[1][:, None], nuB))
        self.fC = np.exp(line_log(s[None, :], us[2][:, None], nuC))
        # vertex rows on the difference lattice (Nv, 2Ns-1)
        self.g1 = vertex_row(nuA, nuB, us[0][:, None], x[1], dd[None, :], rv)
        self.g2 = vertex_row(nuB, nuC, us[1][:, None], x[2], dd[None, :], rv)
        self.g0 = vertex_row(nuA, nuC, us[2][:, None], x[0], dd[None, :], rv)
        idx = np.arange(n)
        self.toep = (idx[None, :] - idx[:, None]) + (n - 1)            # index of tau_j - tau_i
        self.n = n

    def M(self, f, g, a):
        return f[a][:, None] * g[a][self.toep]

    def compute(self, chunk=4):
        nv = self.fA.shape[0]
        T = np.empty((nv, nv, nv), dtype=complex)
        M0T = np.stack([self.M(self.fC, self.g0, c).T for c in range(nv)])   # (Nv, n, n)
        M0T = M0T.reshape(nv, -1).T                                         # (n*n, Nv)
        M2 = [self.M(self.fB, self.g2, b) for b in range(nv)]
        for a0 in range(0, nv, chunk):
            al = range(a0, min(nv, a0 + chunk))
            P = np.stack([self.M(self.fA, self.g1, a) @ M2[b] for a in al for b in range(nv)])
            T[a0:a0 + len(al)] = (P.reshape(len(al) * nv, -1) @ M0T).reshape(len(al), nv, nv)
        return T * (self.hs / (2 * np.pi)) ** 3


# ============================================================================ W_3


class W3Lattice:
    """W_3(delta_A, delta_B, delta_C) for delta_l = rd - i v_j on a uniform v grid."""

    def __init__(self, k, vgrid, rd=1.0 / 3.0, hst=0.035, tmax=14.0, h1=0.03, t1max=14.0,
                 sig=-1.0 / 9.0, sig1=-1.0 / 6.0, with_c1=True):
        k1, k2, k3 = k
        self.z = k1 ** 2 / k3 ** 2
        self.w = k2 ** 2 / k3 ** 2
        self.k3sq = k3 ** 2
        self.v = vgrid
        nv = len(vgrid)
        self.nv = nv
        hv = vgrid[1] - vgrid[0]
        self.d1 = rd - 1j * vgrid                                  # delta on single lattice
        jj = np.arange(2 * nv - 1)
        self.dp = 2 * rd - 1j * (2 * vgrid[0] + jj * hv)          # pair sums
        jt = np.arange(3 * nv - 2)
        self.dt = 3 * rd - 1j * (3 * vgrid[0] + jt * hv)          # triple sums
        self.with_c1 = with_c1
        # double integral C on Re s = Re t = sig
        ts = arange_sym(tmax, hst)
        wt = np.full(ts.size, hst)
        wt[0] = wt[-1] = hst / 2.0
        self.s = sig + 1j * ts
        self.ws = wt
        lam = self.dt - 1.5                                        # lambda on triple lattice
        self.lam = lam
        # factors: F1[p, s] = Gamma(-s) z^s Gamma(3/2 - p - s)  (p = delta_A + delta_C pair)
        s = self.s
        base = lg(-s)
        self.F1 = np.exp(base[None, :] + s[None, :] * math.log(self.z)
                         + lg(1.5 - self.dp[:, None] - s[None, :])) * wt[None, :]
        self.F2 = np.exp(base[None, :] + s[None, :] * math.log(self.w)
                         + lg(1.5 - self.dp[:, None] - s[None, :])) * wt[None, :]
        self.nst = s.size
        L = 1
        while L < 2 * self.nst:
            L *= 2
        self.L = L
        self.FF1 = np.fft.fft(self.F1, L, axis=1)
        self.FF2 = np.fft.fft(self.F2, L, axis=1)
        self.sig2 = 2 * sig + 1j * (2 * ts[0] + hst * np.arange(2 * self.nst - 1))   # s + t
        # C_1 single integral
        t1 = arange_sym(t1max, h1)
        w1 = np.full(t1.size, h1)
        w1[0] = w1[-1] = h1 / 2.0
        self.s1 = sig1 + 1j * t1
        self.w1 = w1

    def _tables(self):
        if hasattr(self, "_T3"):
            return
        sig2 = self.sig2
        self._T3 = np.exp(lg(self.lam[:, None] + sig2[None, :]))           # Gamma(lambda + s + t)
        self._TA = np.exp(lg(self.d1[:, None] + sig2[None, :]))            # Gamma(delta_A + s + t)
        s = self.s1
        lzw = math.log(self.z / self.w)
        self._A0 = np.exp(lg(-s) + s * lzw) * self.w1 / (2 * np.pi)
        self._G3 = np.exp(lg(self.lam[:, None] + s[None, :]))               # Gamma(lambda + s)
        self._Gp = np.exp(lg(1.5 - self.dp[:, None] - s[None, :]))          # Gamma(delta_B - lambda - s)
        self._Gc = np.exp(lg(self.d1[:, None] + s[None, :]))                # Gamma(delta_C + s)

    def C_for_a(self, a):
        """W_3[a, :, :] (Nv x Nv)."""
        self._tables()
        nv = self.nv
        dA = self.d1[a]
        b = np.arange(nv)
        # pair index a+c for F1 (delta_B - lambda = 3/2 - (dA + dC)), a+b for F2
        X = self.FF1[a + b][None, :, :] * self.FF2[a + b][:, None, :]   # [b, c, L]
        conv = np.fft.ifft(X, axis=2)[:, :, :2 * self.nst - 1]          # sum_{s+t} F1(s) F2(t)
        tri = a + b[:, None] + b[None, :]                                 # triple index [b, c]
        C = np.einsum("bcs,bcs->bc", conv, self._T3[tri] * self._TA[a][None, None, :])
        C = C / (2 * np.pi) ** 2
        dB = self.d1[:, None]
        dC = self.d1[None, :]
        S = dA + dB + dC
        lam = S - 1.5
        dAl = 1.5 - dB - dC
        dCl = 1.5 - dA - dB
        S1 = np.exp(lg(lam) - lam * math.log(self.z) + lg(dAl) + lg(dB) + lg(dCl))
        if self.with_c1:
            # sum_s A0(s) Gamma(lambda+s) Gamma(delta_B-lambda-s) Gamma(delta_C+s)
            M = self._G3[tri] * (self._Gp[a + b] * self._Gc[b])[None, :, :]    # [b, c, s]
            C1 = M @ self._A0
            C1 = np.exp(lg(dAl) - lam * math.log(self.w)) * C1
        else:
            C1 = 0.0
        pre = np.exp(1.5 * math.log(np.pi) + (1.5 - S) * math.log(self.k3sq)
                     - lg(3.0 - S) - lg(dA) - lg(dB) - lg(dC))
        return pre * (C + C1 + S1)


# ============================================================================ term A


def routeA_point(mus, k, hs=0.075, taumax=40.0, Vhalf=6.0, hv=0.075,
                 w3set=(0.035, 14.0, 0.03, 14.0), chunk=4, lambdas=(10, 20, 40, 80, 160),
                 verbose=False):
    """Term A of the separated loop triangle and the large-l moments J0, J2."""
    t0 = time.time()
    rho = -1.0 / 3.0
    sig_s = -1.0 / 6.0
    taus = arange_sym(taumax, hs)
    v = arange_sym(Vhalf, hv)
    us = np.array([rho + 1j * (-m + v) for m in mus])
    delta = -rho - 1j * v                                    # common to the three lines
    tr = Trace(mus, k, sig_s, taus, us)
    T = tr.compute(chunk=chunk)
    w = outer_weight(delta)
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    pref = 0.125 * (hv / (2 * np.pi)) ** 3
    w3 = W3Lattice(k, v, rd=-rho, hst=w3set[0], tmax=w3set[1], h1=w3set[2], t1max=w3set[3])
    A = 0j
    nv = len(v)
    for a in range(nv):
        A += np.sum(W[a] * T[a] * w3.C_for_a(a))
    A *= pref
    k1, k2, k3 = k
    S = delta[:, None, None] + delta[None, :, None] + delta[None, None, :]
    dA = delta[:, None, None]
    dB = delta[None, :, None]
    P2 = (dA * (2 * dA - 1) * k1 ** 2 / 3.0 + dB * (2 * dB - 1) * k3 ** 2 / 3.0
          + 4.0 / 3.0 * dA * dB * (k1 ** 2 + k3 ** 2 - k2 ** 2) / 2.0)
    rec = {"A": [A.real, A.imag], "k": list(k), "mus": list(mus), "Ns": int(len(taus)),
           "Nv": int(nv), "hs": hs, "hv": hv}
    for L in lambdas:
        J0 = pref * np.sum(W * T * 4 * np.pi * L ** (3 - 2 * S) / (3 - 2 * S))
        J2 = pref * np.sum(W * T * 4 * np.pi * P2 * L ** (1 - 2 * S) / (1 - 2 * S))
        rec[f"J0_{L:g}"] = [J0.real, J0.imag]
        rec[f"J2_{L:g}"] = [J2.real, J2.imag]
    rec["wall_s"] = time.time() - t0
    return rec


def psi3_mb(mus, x, y, hs=0.3, hv=0.3, taumax=40.0, Vhalf=6.0, chunk=4):
    """Fixed-loop-momentum psi_3 by the six-fold Mellin-Barnes representation (Letter Eq. (8) at fixed loop momentum)."""
    rho = -1.0 / 3.0
    taus = arange_sym(taumax, hs)
    v = arange_sym(Vhalf, hv)
    us = np.array([rho + 1j * (-m + v) for m in mus])
    delta = -rho - 1j * v
    T = Trace(mus, x, -1.0 / 6.0, taus, us).compute(chunk=chunk)
    wl = [np.pi / np.sin(np.pi * (1 - delta)) * np.exp(-2 * delta * math.log(yl / 2.0)) for yl in y]
    W = wl[0][:, None, None] * wl[1][None, :, None] * wl[2][None, None, :]
    return complex(0.125 * (hv / (2 * np.pi)) ** 3 * np.sum(W * T))


# ============================================================================ TriPole: D0 and R


class TriPole:
    """Closed Barnes-residue form of D_0 and of the interchange term R."""

    def __init__(self, mus, k, sigma=-0.166667, cperp=0.5, theta=0.25, taumax=40.0, hs=0.15,
                 Y=6.0, hy=0.15):
        self.mus = tuple(float(m) for m in mus)
        self.nus = [1j * m for m in self.mus]
        self.x = tuple(float(v) for v in k)
        self.sigma, self.cperp, self.theta = sigma, cperp, theta
        self.taumax, self.hs, self.Y, self.hy = taumax, hs, Y, hy
        self.tau = arange_sym(taumax, hs)
        self.y = arange_sym(Y, hy)
        self.ny = len(self.y)

    # -- outer variables -------------------------------------------------------
    def _deltas(self, S):
        dB = self.cperp - 1j * self.y
        dC = self.cperp - 1j * self.y
        m = np.arange(2 * self.ny - 1)
        dA = S - 2 * self.cperp + 1j * (2 * self.y[0] + m * self.hy)
        return dA, dB, dC

    def _u(self, S):
        dA, dB, dC = self._deltas(S)
        nuA, nuB, nuC = self.nus
        return -nuA - dA, -nuB - dB, -nuC - dC, dA, dB, dC

    def G_surface(self, S):
        nuA, nuB, nuC = self.nus
        uA, uB, uC, dA, dB, dC = self._u(S)
        wA, wB, wC = outer_weight(dA), outer_weight(dB), outer_weight(dC)
        sA = self.sigma + 1j * self.tau
        alpha = -0.5 * (1 + nuA + nuB + 2 * uA)                  # on pair lattice m = b+c
        beta = -0.5 * (1 + nuB + nuC + 2 * uB)                   # on b
        fA = np.exp(line_log(sA[None, :], uA[:, None], nuA))     # (Nm, Ntau)
        tot = 0j
        ny = self.ny
        for b in range(ny):
            m = b + np.arange(ny)                                 # c = 0..ny-1
            sB = sA[None, :] + alpha[m][:, None]
            fB = np.exp(line_log(sB, uB[b], nuB))
            sC = sB + beta[b]
            fC = np.exp(line_log(sC, uC[:, None], nuC))
            val = np.sum(fA[m] * fB * fC, axis=1)
            tot += np.sum(wA[m] * wB[b] * wC * val)
        return tot / 32.0 * (self.hy / (2 * np.pi)) ** 2 * (self.hs / (2 * np.pi))

    def D0(self):
        return 0.5 * self.G_surface(1.5)

    def _dG(self, eta):
        return (self.G_surface(1.5 + eta) - self.G_surface(1.5 - eta)) / (2 * eta)

    def Gprime(self, h=0.02):
        """Two-level Richardson derivative (the class method used by r_worker)."""
        d1, d2 = self._dG(h), self._dG(h / 2)
        return (4 * d2 - d1) / 3, abs(d2 - d1)

    def Gprime3(self, h=0.02):
        """Three-level Richardson derivative (the production driver)."""
        d1, d2, d3 = self._dG(h), self._dG(h / 2), self._dG(h / 4)
        r1 = (4 * d2 - d1) / 3
        r2 = (4 * d3 - d2) / 3
        return (16 * r2 - r1) / 15, abs(r2 - r1)

    def _crosscorr(self, U, V):
        """C(d) = sum_tau U(tau) V(tau + d), d = -(N-1)..N-1 (rows of U, V)."""
        n = U.shape[-1]
        L = 1
        while L < 2 * n:
            L *= 2
        FU = np.fft.fft(U[..., ::-1], L, axis=-1)
        FV = np.fft.fft(V, L, axis=-1)
        c = np.fft.ifft(FU * FV, axis=-1)[..., :2 * n - 1]
        # index j corresponds to d = j - (n - 1)
        return c

    def piece_res1(self, S=1.5):
        """L0: V1 residue taken, V2 contour pushed to Re V2 = -theta."""
        nuA, nuB, nuC = self.nus
        uA, uB, uC, dA, dB, dC = self._u(S)
        wA, wB, wC = outer_weight(dA), outer_weight(dB), outer_weight(dC)
        alpha = -0.5 * (1 + nuA + nuB + 2 * uA)
        beta = -0.5 * (1 + nuB + nuC + 2 * uB)
        sA = self.sigma + 1j * self.tau
        sC = self.sigma - self.theta / 2 + 1j * self.tau
        n = len(self.tau)
        d = (np.arange(2 * n - 1) - (n - 1)) * self.hs
        x0, x2 = self.x[0], self.x[2]
        fA = np.exp(line_log(sA[None, :], uA[:, None], nuA))
        fC = np.exp(line_log(sC[None, :], uC[:, None], nuC))
        tot = 0j
        ny = self.ny
        for b in range(ny):
            m = b + np.arange(ny)
            U = fA[m] * np.exp(line_log(sA[None, :] + alpha[m][:, None], uB[b], nuB))
            C = self._crosscorr(U, fC)                               # (c, 2n-1)
            V2 = -self.theta + 2j * d[None, :] - 2 * (alpha[m] + beta[b])[:, None]
            wgt = np.exp(lg(V2) + lg(-V2) + V2 * math.log(x0 / x2))
            tot += np.sum(wA[m] * wB[b] * wC * np.sum(C * wgt, axis=1))
        return 0.5 * (self.hs / (2 * np.pi)) ** 2 * tot

    def piece_res2(self, S=1.5):
        """0L: V2 residue taken, V1 contour pushed to Re V1 = -theta."""
        nuA, nuB, nuC = self.nus
        uA, uB, uC, dA, dB, dC = self._u(S)
        wA, wB, wC = outer_weight(dA), outer_weight(dB), outer_weight(dC)
        alpha = -0.5 * (1 + nuA + nuB + 2 * uA)
        beta = -0.5 * (1 + nuB + nuC + 2 * uB)
        sA = self.sigma + 1j * self.tau
        sB = self.sigma - self.theta / 2 + 1j * self.tau
        n = len(self.tau)
        d = (np.arange(2 * n - 1) - (n - 1)) * self.hs
        x0, x1 = self.x[0], self.x[1]
        fA = np.exp(line_log(sA[None, :], uA[:, None], nuA))
        tot = 0j
        ny = self.ny
        for b in range(ny):
            m = b + np.arange(ny)
            fB = np.exp(line_log(sB, uB[b], nuB))
            V = fB[None, :] * np.exp(line_log(sB[None, :] + beta[b], uC[:, None], nuC))  # (c, n)
            U = fA[m]
            C = self._crosscorr(U, V)
            V1 = -self.theta + 2j * d[None, :] - 2 * alpha[m][:, None]
            wgt = np.exp(lg(V1) + lg(-V1) + V1 * math.log(x0 / x1))
            tot += np.sum(wA[m] * wB[b] * wC * np.sum(C * wgt, axis=1))
        return 0.5 * (self.hs / (2 * np.pi)) ** 2 * tot

    def piece_LL(self, S=1.5):
        """LL: trace with the shifted s contours Re s = sigma, sigma - theta/2, sigma - theta."""
        nuA, nuB, nuC = self.nus
        uA, uB, uC, dA, dB, dC = self._u(S)
        wA, wB, wC = outer_weight(dA), outer_weight(dB), outer_weight(dC)
        n = len(self.tau)
        tau = self.tau
        sA = self.sigma + 1j * tau
        sB = self.sigma - self.theta / 2 + 1j * tau
        sC = self.sigma - self.theta + 1j * tau
        dd = (np.arange(-(n - 1), n)) * self.hs
        idx = np.arange(n)
        toep = (idx[None, :] - idx[:, None]) + (n - 1)
        x0, x1, x2 = self.x
        ny = self.ny
        # M0^(c)[k, i] = f_C(sC_k) Gamma(V0) x0^-V0, V0 = 1+nuA+nuC+2uC+2(sA_i - sC_k)
        M0 = []
        for c in range(ny):
            fC = np.exp(line_log(sC, uC[c], nuC))
            z = 1 + nuA + nuC + 2 * uC[c] + 2 * (self.theta) + 2j * dd       # sA - sC = theta + i d
            g0 = np.exp(lg(z) - z * math.log(x0))
            M0.append(fC[:, None] * g0[toep])
        M0 = np.stack(M0)                                                     # (c, n, n)
        Acc = np.zeros((2 * ny - 1, n, n), dtype=complex)
        for b in range(ny):
            fB = np.exp(line_log(sB, uB[b], nuB))
            z = 1 + nuB + nuC + 2 * uB[b] + 2 * (-self.theta / 2) + 2j * dd   # sC - sB
            g2 = np.exp(lg(z) - z * math.log(x2))
            M2 = fB[:, None] * g2[toep]
            P = np.matmul(M2[None, :, :], M0)                                 # (c, n, n)
            for c in range(ny):
                Acc[b + c] += wB[b] * wC[c] * P[c]
        tot = 0j
        for m in range(2 * ny - 1):
            fA = np.exp(line_log(sA, uA[m], nuA))
            z = 1 + nuA + nuB + 2 * uA[m] + 2 * (-self.theta / 2) + 2j * dd   # sB - sA
            g1 = np.exp(lg(z) - z * math.log(x1))
            M1 = fA[:, None] * g1[toep]
            tot += wA[m] * np.sum(M1 * Acc[m].T)                              # tr(M1 Acc)
        return (self.hs / (2 * np.pi)) ** 3 * tot

    def R(self, verbose=False, gp_h=0.01, levels=2):
        """R = 2 pi PF g with the pieces; returns a record dictionary."""
        t0 = time.time()
        D0 = self.D0()
        if levels == 3:
            Gp, Gperr = self.Gprime3(gp_h)
        else:
            Gp, Gperr = self.Gprime(gp_h)
        pref = 0.125 * (self.hy / (2 * np.pi)) ** 2
        LL = pref * self.piece_LL(1.5)
        L0 = pref * self.piece_res1(1.5)
        OL = pref * self.piece_res2(1.5)
        greg = LL + L0 + OL
        PF = greg - 2 * D0 * (EULER + math.log(self.x[0])) - 0.5 * Gp
        R = 2 * np.pi * PF
        c = lambda z: [complex(z).real, complex(z).imag]
        return {"hs": self.hs, "hy": self.hy, "Y": self.Y, "taumax": self.taumax,
                "mus": list(self.mus), "k": list(self.x), "D0": c(D0), "Gprime": c(Gp),
                "Gprime_err": float(Gperr), "g_reg": c(greg), "LL": c(LL), "L0": c(L0),
                "OL": c(OL), "R": c(R), "wall": time.time() - t0}
