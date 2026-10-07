"""Direct Schwinger-Keldysh time integration.

Double and triple exchange: the sector-summed leg functions L = -F + (C/pi) K of every
line are built at the nodes of an outer Gauss-Legendre grid in the cubic-vertex time,
entirely in multi-precision arithmetic (mpmath Bessel functions).  Single exchange:
mode functions (Hankel functions at 25 digits) on a Gauss-Legendre grid in ln t, with
double-precision quadrature of the four sector propagators.
"""
from __future__ import annotations

import math
import multiprocessing as _mpc
import os

import mpmath as mp
import numpy as np

# -----------------------------------------------------------------------------
# multi-precision Gauss-Legendre rule (4.1)
# -----------------------------------------------------------------------------

_GL_CACHE = {}


def _legendre_pair(n, x):
    """(P_n(x), P_{n-1}(x)) by the three-term recurrence."""
    p0, p1 = mp.mpf(1), x
    if n == 0:
        return p0, mp.mpf(0)
    for j in range(2, n + 1):
        p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
    return p1, p0


def gauss_legendre_mp(n):
    """n-point Gauss-Legendre rule on [-1, 1] at the current working precision."""
    key = (n, mp.mp.dps)
    r = _GL_CACHE.get(key)
    if r is not None:
        return r
    seeds, _ = np.polynomial.legendre.leggauss(n)
    pairs = []
    for s in seeds:
        x = mp.mpf(float(s))
        for _ in range(6):
            pn, pm = _legendre_pair(n, x)
            dp = n * (pm - x * pn) / (1 - x * x)
            x = x - pn / dp
        pn, pm = _legendre_pair(n, x)
        dp = n * (pm - x * pn) / (1 - x * x)
        w = 2 / ((1 - x * x) * dp * dp)
        pairs.append((x, w))
    pairs.sort(key=lambda p: p[0])
    r = ([p[0] for p in pairs], [p[1] for p in pairs])
    _GL_CACHE[key] = r
    return r


# -----------------------------------------------------------------------------
# double and triple exchange (4.2 - 4.5, 4.8)
# -----------------------------------------------------------------------------

TMIN = "1e-12"
SEG_NODES = 12
TAIL_NODES = 24
SERIES_ORDER = 48


class DirectSettings:
    """Quadrature constants of the double/triple direct route (defaults: package functions)."""

    def __init__(self, mpanels=22, nn=24, window=50, tail_width="0.75", tail_extent=15,
                 soft_below="0.1", min_soft_panels=None):
        self.mpanels = int(mpanels)
        self.nn = int(nn)
        self.window = window
        self.tail_width = tail_width
        self.tail_extent = tail_extent
        self.soft_below = soft_below
        self.min_soft_panels = min_soft_panels


def _mp(x):
    return x if isinstance(x, (mp.mpf, mp.mpc)) else mp.mpf(x if isinstance(x, str) else repr(x))


def outer_grid(q, st):
    """Outer nodes and weights in the cubic-vertex time (linear-in-t GL on log panels)."""
    tmin = mp.mpf(TMIN)
    tmax = _mp(st.window) / (q[0] + q[1] + q[2])
    M = st.mpanels
    x, w = gauss_legendre_mp(st.nn)
    la, lb = mp.log(tmin), mp.log(tmax)
    edges = [mp.exp(la + (lb - la) * i / M) for i in range(M + 1)]
    nodes, wts = [], []
    for i in range(M):
        c = (edges[i] + edges[i + 1]) / 2
        h = (edges[i + 1] - edges[i]) / 2
        for xj, wj in zip(x, w):
            nodes.append(c + h * xj)
            wts.append(h * wj)
    order = sorted(range(len(nodes)), key=lambda i: nodes[i])
    return [nodes[i] for i in order], [wts[i] for i in order], tmax


def _tail_panels(y, tmax, st):
    if y < _mp(st.soft_below):
        delta = _mp(st.tail_width) / y
        P = int(mp.ceil((_mp(st.tail_extent) / y - tmax) / delta)) + 2
        if st.min_soft_panels is not None:
            P = max(P, st.min_soft_panels)
    else:
        delta = 8 / y
        P = 7
    return delta, P


class LineTables:
    """Bessel tables and cumulative leg integrals of one line (4.3, 4.4)."""

    def __init__(self, y, mu, nodes, tmax, st):
        self.y, self.mu = y, mu
        nu = mp.mpc(0, mu)
        self.nu = nu
        N = len(nodes)
        xs, ws = gauss_legendre_mp(SEG_NODES)
        xt, wt = gauss_legendre_mp(TAIL_NODES)
        delta, P = _tail_panels(y, tmax, st)
        self.tail = (delta, P)

        def IK(u):
            z = y * u
            return mp.besseli(nu, z), mp.besselk(nu, z)

        def wy(u):
            return mp.exp(-y * u) / mp.sqrt(u)

        self.I, self.K = [], []
        for t in nodes:
            i_, k_ = IK(t)
            self.I.append(i_)
            self.K.append(k_)
        # segment sums (A upward, D downward)
        segI, segK = [], []
        for i in range(N - 1):
            a, b = nodes[i], nodes[i + 1]
            c = (a + b) / 2
            h = (b - a) / 2
            sI = 0
            sK = 0
            for xj, wj in zip(xs, ws):
                u = c + h * xj
                i_, k_ = IK(u)
                ww = h * wj * wy(u)
                sI += ww * i_
                sK += ww * k_
            segI.append(sI)
            segK.append(sK)
        # tail beyond the last outer node
        tN = nodes[-1]
        DN = 0
        for j in range(P):
            a = tN + j * delta
            b = a + delta
            c = (a + b) / 2
            h = (b - a) / 2
            for xj, wj in zip(xt, wt):
                u = c + h * xj
                _, k_ = IK(u)
                DN += h * wj * wy(u) * k_
        A0 = self._head(nodes[0], nu)
        Am0 = self._head(nodes[0], -nu)
        A = [A0]
        for i in range(N - 1):
            A.append(A[-1] + segI[i])
        D = [None] * N
        D[N - 1] = DN
        for i in range(N - 2, -1, -1):
            D[i] = D[i + 1] + segK[i]
        self.A, self.D = A, D
        self.C = D[0] + mp.pi / (2 * mp.sin(mp.pi * nu)) * (Am0 - A0)
        self.npoints = N + (N - 1) * SEG_NODES + P * TAIL_NODES

    def _head(self, t0, nu):
        y = self.y
        s = 0
        for m in range(0, SERIES_ORDER):
            if 4 * m + 1 > SERIES_ORDER:
                break
            cm = (y / 2) ** (nu + 2 * m) / (mp.factorial(m) * mp.gamma(nu + m + 1))
            j = 0
            while 4 * m + 2 * j + 1 <= SERIES_ORDER:
                a = nu + 2 * m + j + mp.mpf(1) / 2
                s += cm * (-y) ** j / mp.factorial(j) * t0 ** a / a
                j += 1
        return s

    def legs(self):
        """(F, L) at the outer nodes."""
        c0 = 1j / mp.pi * mp.exp(-mp.pi * self.mu)
        C = self.C
        Fs, Ls = [], []
        for I, K, A, D in zip(self.I, self.K, self.A, self.D):
            F = K * (A - c0 * (C - D)) + (I - c0 * K) * D
            Fs.append(F)
            Ls.append(-F + C / mp.pi * K)
        return Fs, Ls


def _lines(specs, nodes, tmax, st):
    cache = {}
    out = []
    for y, mu in specs:
        key = (y, mu)
        if key not in cache:
            cache[key] = LineTables(y, mu, nodes, tmax, st)
        out.append(cache[key])
    return out


def double_direct_mp(q, mus, st):
    """(S, F_unit, psiF_time) in multi-precision; q, mus are mp numbers."""
    nodes, wts, tmax = outer_grid(q, st)
    l1, l2 = _lines([(q[1], mus[0]), (q[2], mus[1])], nodes, tmax, st)
    F1, L1 = l1.legs()
    F2, L2 = l2.legs()
    Sig = 0
    psi = 0
    for i, t in enumerate(nodes):
        m = wts[i] * t * mp.exp(-q[0] * t)
        Sig += m * L1[i] * L2[i]
        psi += m * F1[i] * F2[i]
    F_unit = 2 * mp.re(Sig)
    S = q[0] * q[1] * q[2] * F_unit / 8
    return S, F_unit, psi


def triple_direct_mp(q, mus, st):
    nodes, wts, tmax = outer_grid(q, st)
    la, lb, lc = _lines([(q[0], mus[0]), (q[1], mus[1]), (q[2], mus[2])], nodes, tmax, st)
    FA, LA = la.legs()
    FB, LB = lb.legs()
    FC, LC = lc.legs()
    Sig = 0
    psi = 0
    for i, t in enumerate(nodes):
        m = wts[i] * mp.sqrt(t)
        Sig += m * LA[i] * LB[i] * LC[i]
        psi += m * FA[i] * FB[i] * FC[i]
    F_unit = 2 * mp.re(Sig)
    S = q[0] * q[1] * q[2] * F_unit / 8
    return S, F_unit, psi


# -----------------------------------------------------------------------------
# single exchange (4.7)
# -----------------------------------------------------------------------------

U_MIN = -75.0
HANKEL_CUT = 45.0
CLIP = 700.0
MODE_DPS = 25
CHUNK = 64


def cumulative_matrix(n):
    """S_jm = int_{-1}^{x_j} l_m(x) dx for the Lagrange basis on the n GL nodes."""
    x, _ = np.polynomial.legendre.leggauss(n)
    V = np.polynomial.legendre.legvander(x, n - 1)          # V_jm = P_m(x_j)
    P = np.polynomial.legendre.legvander(x, n)              # up to P_n
    Q = np.empty((n, n))
    Q[:, 0] = x + 1.0
    for m in range(1, n):
        Q[:, m] = (P[:, m + 1] - P[:, m - 1]) / (2 * m + 1)
    return Q @ np.linalg.inv(V)


def single_grid(y, h, n):
    umax = math.log(HANKEL_CUT / y)
    P = int(math.ceil((umax - U_MIN) / h))
    x, w = np.polynomial.legendre.leggauss(n)
    j = np.arange(P)[:, None]
    u = U_MIN + j * h + 0.5 * h * (x[None, :] + 1.0)
    t = np.exp(u)
    W = 0.5 * h * w[None, :] * t
    return u, t, W, w, P


def mode_functions(args):
    """Worker: u(it) and u*(it) for a chunk of times at 25 digits."""
    ts, mu, y = args
    out = []
    with mp.workdps(MODE_DPS):
        mu_ = mp.mpf(mu)
        y_ = mp.mpf(y)
        N = mp.sqrt(mp.pi) / 2 * mp.exp(-mp.pi * mu_ / 2) * mp.expjpi(mp.mpf(1) / 4)
        Nb = mp.conj(N)
        nu = mp.mpc(0, mu_)
        for t in ts:
            tt = mp.mpf(t)
            pw = mp.mpc(0, -tt) ** mp.mpf(1.5)
            z = mp.mpc(0, -y_ * tt)
            u = N * pw * mp.hankel1(nu, z)
            us = Nb * pw * mp.hankel2(-nu, z)
            out.append((complex(u), complex(us)))
    return out


def _cum_tail(f, u, h, Smat, w):
    """Cumulative and tail integrals of f (panel x node arrays) on the ln t grid."""
    g = f * np.exp(u)
    p = 0.5 * h * (g @ w)                     # full-panel integrals
    within = 0.5 * h * (g @ Smat.T)           # (h/2) sum_m' S_mm' g_jm'
    before = np.concatenate(([0.0], np.cumsum(p)[:-1]))
    after = np.concatenate((np.cumsum(p[::-1])[::-1][1:], [0.0]))
    cum = before[:, None] + within
    tail = after[:, None] + p[:, None] - within
    return cum, tail


def single_channel_direct(q, leg, mu, h, n, pool):
    y = q[leg - 1]
    x1 = q[0] + q[1] + q[2] - y
    x2 = y
    u, t, W, w, P = single_grid(y, h, n)
    tf = t.ravel().tolist()
    chunks = [(tf[i:i + CHUNK], mu, y) for i in range(0, len(tf), CHUNK)]
    res = pool.map(mode_functions, chunks)
    flat = [v for c in res for v in c]
    up = np.array([v[0] for v in flat]).reshape(t.shape)      # u(it)
    usp = np.array([v[1] for v in flat]).reshape(t.shape)     # u*(it)
    um = np.conj(usp)                                         # u(-it)
    usm = np.conj(up)                                         # u*(-it)
    eA = np.exp(-np.clip(x1 * t, 0.0, CLIP))
    eB = t ** -2 * np.exp(-np.clip(x2 * t, 0.0, CLIP))
    Smat = cumulative_matrix(n)
    c1, t1 = _cum_tail(eA * up, u, h, Smat, w)
    c2, t2 = _cum_tail(eA * usp, u, h, Smat, w)
    Vpp = usp * c1 + up * t2
    c3, t3 = _cum_tail(eA * usm, u, h, Smat, w)
    c4, t4 = _cum_tail(eA * um, u, h, Smat, w)
    Vmm = um * c3 + usm * t4
    Vpm = um * np.sum(W * eA * usp)
    Vmp = usp * np.sum(W * eA * um)
    f = 0j
    for V in (Vpp, Vpm, Vmp, Vmm):
        f += np.sum(W * eB * V)
    return {"exchanged_leg": leg, "y": y, "x_hard": x1, "F_sectors": complex(f),
            "S": q[0] * q[1] * q[2] / 4.0 * f.real}


def default_pool_size():
    return max(1, min(4, os.cpu_count() or 1))


def make_pool(n):
    ctx = _mpc.get_context("fork") if "fork" in _mpc.get_all_start_methods() else _mpc
    return ctx.Pool(n)


# -----------------------------------------------------------------------------
# refined reference: function ``direct_refined`` of examples/Letter/tree_validation.py
# -----------------------------------------------------------------------------

SETTING_A = dict(mpanels=36, tmax_fac=70, tail_width="0.25", tail_extent=22, soft_below="0.1",
                 nn=24)
SETTING_B = dict(mpanels=28, tmax_fac=60, tail_width="0.35", tail_extent=18, soft_below="1.5",
                 nn=24)


def direct_refined(diagram, masses, k, mpanels, tmax_fac, tail_width, tail_extent, soft_below,
                   nn=24, digits=25, return_mp=False):
    """Refined direct reference; momenta used as given (no validation, no normalisation).

    Returns the double-precision S (or the multi-precision S with ``return_mp=True``).
    """
    with mp.workdps(int(digits)):
        q = tuple(_mp(x) for x in k)
        mus = tuple(_mp(m) for m in masses)
        st = DirectSettings(mpanels=mpanels, nn=nn, window=tmax_fac, tail_width=tail_width,
                            tail_extent=tail_extent, soft_below=soft_below, min_soft_panels=7)
        if diagram == "double":
            S = double_direct_mp(q, mus, st)[0]
        elif diagram == "triple":
            S = triple_direct_mp(q, mus, st)[0]
        else:
            raise ValueError("diagram must be 'double' or 'triple'")
        return +S if return_mp else float(S)
