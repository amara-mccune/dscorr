"""Arbitrary-precision series route to M2, M3.

Working precision 150 digits; no multi-precision Bessel calls: two power series per
hard line, Gauss-Legendre panels in ln t.  No code is shared with the float64 route.
"""
from __future__ import annotations

import mpmath as mp

WORK_DPS = 150
N_IC = 312          # coefficients of I_n(u) = u^n sum c_j u^(2j)
N_A = 621           # coefficients of e^-u I_n(u) = u^n sum a_m u^m
L_MAX = "3.92"
L_MIN_DOUBLE = -42
L_MIN_TRIPLE = -56
PANEL = "0.7"
GL_N = 26


class SeriesLine:
    """Hard line of mass mu at unit momentum from power series."""

    def __init__(self, mu):
        mu = mp.mpf(mu)
        self.mu = mu
        nu = mp.mpc(0, mu)
        self.nu = nu
        self.coef = {}
        for s in (1, -1):
            n = s * nu
            c = [mp.mpf(2) ** (-n - 2 * j) / (mp.factorial(j) * mp.gamma(n + j + 1))
                 for j in range(N_IC)]
            a = []
            for m in range(N_A):
                acc = 0
                for j in range(m // 2 + 1):
                    acc += c[j] * (-1) ** (m - 2 * j) / mp.factorial(m - 2 * j)
                a.append(acc)
            self.coef[s] = (n, c, a)
        self.epm = mp.exp(-mp.pi * mu)
        self.sh = mp.sinh(mp.pi * mu)
        self.phi = mp.pi * mp.sqrt(mp.pi / 2) / mp.cosh(mp.pi * mu)

    def _IP(self, s, t):
        n, c, a = self.coef[s]
        eps = mp.eps
        t2 = t * t
        tot = 0
        p = mp.mpf(1)
        for j in range(N_IC):
            term = c[j] * p
            tot += term
            if j > 4 and abs(term) < eps * abs(tot) / 100:
                break
            p *= t2
        tn = t ** n
        I = tn * tot
        tot = 0
        big = 0
        p = mp.mpf(1)
        for m in range(N_A):
            term = a[m] * p / (n + m + mp.mpf(1) / 2)
            tot += term
            at = abs(term)
            if at > big:
                big = at
            if m > 8 and at < eps * (abs(tot) + big) / 100:
                break
            p *= t
        P = tn * mp.sqrt(t) * tot
        return I, P

    def at(self, t):
        """(K, FF, g) at time t."""
        Ip, Pp = self._IP(1, t)
        Im, Pm = self._IP(-1, t)
        d = 2j * self.sh
        K = mp.pi * (Im - Ip) / d
        CK = mp.pi * (Pm - Pp) / d
        c0 = 1j / mp.pi * self.epm
        J = Ip - c0 * K
        CJ = Pp - c0 * CK
        BK = self.phi - CK
        FF = K * CJ + J * BK
        g = K * Pp + Ip * BK
        return K, FF, g


def gauss_legendre_newton(N):
    """N-point Gauss-Legendre rule on [-1, 1] at the working precision (4.8 step 3)."""
    eps = mp.eps
    xs, ws = [], []
    for i in range(1, N + 1):
        x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (N + mp.mpf(1) / 2))
        for _ in range(60):
            p0, p1 = mp.mpf(1), x
            for j in range(2, N + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = N * (x * p1 - p0) / (x * x - 1)
            step = p1 / dp
            x = x - step
            if abs(step) < 10 * eps:
                break
        p0, p1 = mp.mpf(1), x
        for j in range(2, N + 1):
            p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
        dp = N * (x * p1 - p0) / (x * x - 1)
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def outer_nodes(Lmin):
    Lmax = mp.mpf(L_MAX)
    Lmin = mp.mpf(Lmin)
    Np = int(mp.ceil((Lmax - Lmin) / mp.mpf(PANEL)))
    width = (Lmax - Lmin) / Np
    x, w = gauss_legendre_newton(GL_N)
    nodes = []
    for p in range(Np):
        c = Lmin + (p + mp.mpf(1) / 2) * width
        for xj, wj in zip(x, w):
            nodes.append((c + width / 2 * xj, width / 2 * wj))
    return nodes


def _lam(mu):
    d = 2 * mp.sinh(mp.pi * mu)
    return (mp.exp(mp.pi * mu) - 1j) / d, -(mp.exp(-mp.pi * mu) + 1j) / d


def _phi(mu):
    return mp.pi * mp.sqrt(mp.pi / 2) / mp.cosh(mp.pi * mu)


def H_moment(a, mu1):
    return (mp.sqrt(mp.pi) * mp.gamma(a + 1j * mu1) * mp.gamma(a - 1j * mu1)
            / (mp.mpf(2) ** a * mp.gamma(a + mp.mpf(1) / 2)))


def T_KK(p, muA, muB):
    z = mp.mpf(3) / 2 + p
    prod = 1
    for s in (1, -1):
        for sp in (1, -1):
            prod *= mp.gamma((z + s * 1j * muA + sp * 1j * muB) / 2)
    return mp.mpf(2) ** (z - 3) / mp.gamma(z) * prod


def M2_mp(mu1, mu2):
    """Engine dictionary at the working precision; masses as decimal strings or mp numbers."""
    with mp.workdps(WORK_DPS):
        mu1 = mp.mpf(mu1)
        mu2 = mp.mpf(mu2)
        line = SeriesLine(mu1)
        ap = 2 + 1j * mu2
        am = 2 - 1j * mu2
        HFp = HFm = psp = psm = Hn = 0
        for L, w in outer_nodes(L_MIN_DOUBLE):
            t = mp.exp(L)
            K, FF, g = line.at(t)
            et = w * mp.exp(-t)
            ep = et * mp.exp(ap * L)
            em = et * mp.exp(am * L)
            HFp += ep * FF
            HFm += em * FF
            psp += ep * g
            psm += em * g
            Hn += ep * K
        H = H_moment(ap, mu1)
        ph1, ph2 = _phi(mu1), _phi(mu2)
        lp, lm = _lam(mu2)
        csch = 1 / mp.sinh(mp.pi * mu2)
        pref = ph2 * mp.mpf(2) ** (-1j * mu2) / (8 * mp.gamma(1 + 1j * mu2))
        A = pref * (lp * HFp + lm * mp.conj(HFm) - ph1 / mp.pi * (1 - 1j * csch) * H)
        A_alt = pref * (lp * psp + lm * mp.conj(psm) - ph1 * H / mp.pi
                        * ((1 - 1j * csch) + 1j * mp.exp(-mp.pi * mu1)
                           * mp.cosh(mp.pi * mu2) / mp.sinh(mp.pi * mu2)))
        return {"A": A, "A_alt": A_alt, "HFp": HFp, "HFm": HFm, "psip": psp,
                "psim": mp.conj(psm), "errH": abs(Hn / H - 1), "errAB": abs(A / A_alt - 1)}


def M3_mp(muA, muB, muC):
    with mp.workdps(WORK_DPS):
        muA, muB, muC = mp.mpf(muA), mp.mpf(muB), mp.mpf(muC)
        la = SeriesLine(muA)
        lb = la if muB == muA else SeriesLine(muB)
        acc = {k: 0 for k in ("TFFp", "TFFm", "Up", "Um", "Vp", "Vm", "TKKnum")}
        pp = mp.mpf(3) / 2 + 1j * muC
        pm = mp.mpf(3) / 2 - 1j * muC
        for L, w in outer_nodes(L_MIN_TRIPLE):
            t = mp.exp(L)
            KA, FA, _ = la.at(t)
            KB, FB, _ = (KA, FA, None) if lb is la else lb.at(t)
            wp = w * mp.exp(pp * L)
            wm = w * mp.exp(pm * L)
            acc["TFFp"] += wp * FA * FB
            acc["TFFm"] += wm * FA * FB
            acc["Up"] += wp * KA * FB
            acc["Um"] += wm * KA * FB
            acc["Vp"] += wp * FA * KB
            acc["Vm"] += wm * FA * KB
            acc["TKKnum"] += wp * KA * KB
        TKK = T_KK(1j * muC, muA, muB)
        phA, phB, phC = _phi(muA), _phi(muB), _phi(muC)
        Tp = acc["TFFp"] - phA / mp.pi * acc["Up"] - phB / mp.pi * acc["Vp"]
        Tm = mp.conj(acc["TFFm"] - phA / mp.pi * acc["Um"] - phB / mp.pi * acc["Vm"])
        lp, lm = _lam(muC)
        csch = 1 / mp.sinh(mp.pi * muC)
        pref = -phC * mp.mpf(2) ** (-1j * muC) / (8 * mp.gamma(1 + 1j * muC))
        A = pref * (lp * Tp + lm * Tm + phA * phB / mp.pi ** 2 * (1 - 1j * csch) * TKK)
        d = {"A": A, "errT": abs(acc["TKKnum"] / TKK - 1)}
        for k in ("TFFp", "TFFm", "Up", "Um", "Vp", "Vm", "TKKnum"):
            d[k] = acc[k]
        return d
