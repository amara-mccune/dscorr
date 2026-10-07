"""Brute-force route of the one-loop triangle.

Self-contained (numpy, scipy only): scaled Bessel functions of imaginary order and their
table, the geometric Gauss-Legendre time grid, the weighted cumulative operator with exact
polynomial-times-exponential moments, the fixed-loop-momentum triangle psi_3 (sum over the six
time orderings), the angular rule in energy variables, the radial ladder, the far patch and the
regulated constant B_reg.  No Mellin-Barnes machinery.

Conventions: vertices 0, 1, 2 with energies x = (k1, k2, k3); lines A = (0, 1), B = (1, 2),
C = (2, 0) with masses mus = (muA, muB, muC); y_C = |l| = r, y_A = |l + q_A| (|q_A| = k1),
y_B = |l + q_B| (|q_B| = k3); bare d^3 l.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import loggamma

LINES = ((0, 1), (1, 2), (2, 0))          # A, B, C
LADDER = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)

# ============================================================================ Bessel functions


class BesselImag:
    """Scaled K^ = e^x K_{i mu}(x) (real) and I^ = e^-x I_{i mu}(x) (complex), direct."""

    X_SWITCH = 30.0

    def __init__(self, mu):
        self.mu = float(mu)
        x, w = np.polynomial.legendre.leggauss(14)
        width = 0.35
        npan = int(math.ceil(50.0 / width - 1e-12))
        u, wu = [], []
        for p in range(npan):
            a = p * width
            u.append(a + 0.5 * width * (x + 1.0))
            wu.append(0.5 * width * w)
        self._u = np.concatenate(u)
        self._wc = np.concatenate(wu) * np.cos(self.mu * self._u)
        self._chm1 = np.cosh(self._u) - 1.0
        m = np.arange(71)
        self._lc = -loggamma(m + 1.0) - loggamma(m + 1.0 + 1j * self.mu)   # 1/(m! Gamma(m+1+i mu))
        a = [1.0]
        for k in range(1, 23):
            a.append(a[-1] * (-4.0 * self.mu ** 2 - (2 * k - 1) ** 2) / (8.0 * k))
        self._a = np.array(a)

    def K(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty(x.shape)
        lo = x <= self.X_SWITCH
        if lo.any():
            xl = x[lo]
            out[lo] = np.exp(-np.outer(xl, self._chm1)) @ self._wc
        hi = ~lo
        if hi.any():
            xh = x[hi]
            k = np.arange(23)
            out[hi] = np.sqrt(np.pi / (2.0 * xh)) * (xh[:, None] ** (-k[None, :]) @ self._a)
        return out

    def I(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty(x.shape, dtype=complex)
        lo = x <= self.X_SWITCH
        if lo.any():
            xl = x[lo]
            lh = np.log(xl / 2.0)
            m = np.arange(71)
            terms = np.exp(2.0 * np.outer(lh, m) + self._lc[None, :])
            out[lo] = np.exp(-xl + 1j * self.mu * lh) * terms.sum(axis=1)
        hi = ~lo
        if hi.any():
            xh = x[hi]
            k = np.arange(23)
            out[hi] = ((-xh)[:, None] ** (-k[None, :]) @ self._a) / np.sqrt(2.0 * np.pi * xh)
        return out


class TabBessel:
    """K^ and I^ tabulated on a uniform ln x grid (1e-17 .. 1e7), four-point Lagrange."""

    XMIN = 1e-17
    XMAX = 1e7

    def __init__(self, mu, step=None):
        self.mu = float(mu)
        h0 = 0.004 / max(1.0, self.mu) if step is None else step
        lo, hi = math.log(self.XMIN), math.log(self.XMAX)
        n = int(math.ceil((hi - lo) / h0)) + 1          # intervals
        self.d = (hi - lo) / n
        self.L0 = lo - self.d                               # guard node below
        self.N = n + 3                                      # nodes incl. two guards
        L = self.L0 + self.d * np.arange(self.N)
        direct = BesselImag(mu)
        x = np.exp(L)
        self.tK = direct.K(x)
        self.tI = direct.I(x)
        self.xlow = self.XMIN * math.exp(self.d)

    def _w(self, x):
        x = np.asarray(x, dtype=float)
        assert np.all(x < self.XMAX * (1 + 1e-12)), "TabBessel range exceeded"
        x = np.maximum(x, self.xlow)
        p = (np.log(x) - self.L0) / self.d
        i = np.floor(p).astype(int)
        i = np.clip(i, 1, self.N - 3)
        f = p - i
        w = np.stack([-f * (f - 1) * (f - 2) / 6.0, (f + 1) * (f - 1) * (f - 2) / 2.0,
                      -(f + 1) * f * (f - 2) / 2.0, (f + 1) * f * (f - 1) / 6.0], axis=-1)
        idx = i[..., None] + np.arange(-1, 3)
        return idx, w

    def K(self, x):
        idx, w = self._w(x)
        return np.sum(self.tK[idx] * w, axis=-1)

    def I(self, x):
        idx, w = self._w(x)
        return np.sum(self.tI[idx] * w, axis=-1)


# ============================================================================ time grid


class TimeGrid:
    """Geometric panels on [tmin, tmax] with ngl Gauss-Legendre nodes each."""

    def __init__(self, tmin=1e-12, tmax=60.0, npan=156, ngl=12):
        self.tmin, self.tmax, self.npan, self.ngl = float(tmin), float(tmax), int(npan), int(ngl)
        self.ratio = (self.tmax / self.tmin) ** (1.0 / self.npan)
        self.edges = self.tmin * self.ratio ** np.arange(self.npan + 1)
        self.edges[-1] = self.tmax
        x, w = np.polynomial.legendre.leggauss(self.ngl)
        self.xg, self.wg = x, w
        a, b = self.edges[:-1], self.edges[1:]
        self.half = 0.5 * (b - a)
        self.t = (0.5 * (a + b))[:, None] + self.half[:, None] * x[None, :]   # (npan, ngl)
        self.w = self.half[:, None] * w[None, :]
        # monomial-moment machinery
        V = np.vander(x, self.ngl, increasing=True)          # V[j, q] = x_j^q
        self.Vinv = np.linalg.inv(V)                          # c_q = sum_j Vinv[q, j] g_j
        self.targets = np.concatenate([x, [1.0]])             # nodes and right edge
        self._series = _series_coeffs(self.targets, self.ngl, 30)

    @property
    def nodes(self):
        return self.t.size


def _series_coeffs(targets, nq, order):
    """C[t, q, p] = int_{-1}^{a_t} x^q (a_t - x)^p dx / p!  (exact, by binomial expansion)."""
    nt = len(targets)
    C = np.zeros((nt, nq, order + 1))
    for it, a in enumerate(targets):
        for q in range(nq):
            for p in range(order + 1):
                # int_{-1}^{a} x^q (a-x)^p dx; substitute x = a - s: int_0^{a+1} (a-s)^q s^p ds
                s = 0.0
                for j in range(q + 1):
                    c = math.comb(q, j) * a ** (q - j) * (-1) ** j
                    s += c * (a + 1.0) ** (p + j + 1) / (p + j + 1)
                C[it, q, p] = s / math.factorial(p)
    return C


def _moments(zeta, tg):
    """m_q(zeta, a) for all targets a: array (len(zeta), ntargets, nq)."""
    zeta = np.asarray(zeta, dtype=float)
    nq = tg.ngl
    out = np.empty(zeta.shape + (len(tg.targets), nq))
    small = zeta <= 4.0
    if small.any():
        z = zeta[small]
        pw = (-z)[:, None] ** np.arange(tg._series.shape[2])[None, :]     # (n, P)
        out[small] = np.einsum("tqp,np->ntq", tg._series, pw)
    big = ~small
    if big.any():
        z = zeta[big][:, None]
        a = tg.targets[None, :]
        E = np.exp(-z * (a + 1.0))
        m = (1.0 - E) / z
        res = [m]
        for q in range(1, nq):
            m = (a ** q - (-1.0) ** q * E) / z - (q / z) * m
            res.append(m)
        out[big] = np.stack(res, axis=-1)
    return out


def cum_weights(kappa, tg):
    """Per-panel weights W[p] (n, ntargets, ngl) of the weighted cumulative operator."""
    out = []
    for p in range(tg.npan):
        h = tg.half[p]
        M = _moments(kappa * h, tg)                    # (n, nt, nq)
        out.append(h * (M @ tg.Vinv))                  # (n, nt, ngl)
    return out


def weighted_cumulative(g, kappa, tg, W=None):
    """J(t) = int_{tmin}^t g(s) exp(-kappa (t - s)) ds at all nodes.

    g: (n, npan, ngl) complex; kappa: (n,).  Returns (n, npan, ngl).  W: precomputed
    cum_weights(kappa, tg) (optional).
    """
    n = g.shape[0]
    J = np.empty_like(g)
    edge = np.zeros(n, dtype=g.dtype)
    if W is None:
        W = cum_weights(kappa, tg)
    for p in range(tg.npan):
        h = tg.half[p]
        loc = np.einsum("ntj,nj->nt", W[p], g[:, p, :])  # (n, nt)
        left = tg.edges[p]
        dec = np.exp(-kappa[:, None] * (tg.t[p][None, :] - left))
        J[:, p, :] = dec * edge[:, None] + loc[:, :-1]
        edge = np.exp(-kappa * 2.0 * h) * edge + loc[:, -1]
    return J


# ============================================================================ fixed-l triangle

ORDERINGS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


class Triangle:
    """Fixed-loop-momentum triangle psi_3 on a time grid, batched over (yA, yB, yC)."""

    def __init__(self, mus, tg=None, tab=True, rv=1.0, kernels=None):
        self.mus = tuple(float(m) for m in mus)
        self.tg = TimeGrid() if tg is None else tg
        cache = {}
        self.bes = []
        for m in self.mus:
            if m not in cache:
                cache[m] = TabBessel(m) if tab else BesselImag(m)
            self.bes.append(cache[m])
        self.rv = float(rv)

    def _line_factors(self, y, t, l):
        """(K^, I^) of line l at y t; arrays (n, npan, ngl)."""
        b = self.bes[l]
        z = y[:, None, None] * t[None, :, :]
        return b.K(z), b.I(z)

    def psi3(self, x, yA, yB, yC, orderings=False, block=320):
        """psi_3 for arrays of line momenta (complex array); x = (x0, x1, x2)."""
        yA, yB, yC = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (yA, yB, yC))
        n = yA.size
        out = np.zeros(n, dtype=complex)
        per = np.zeros((6, n), dtype=complex) if orderings else None
        for i0 in range(0, n, block):
            sl = slice(i0, i0 + block)
            r = self._psi3_block(x, yA[sl], yB[sl], yC[sl])
            out[sl] = r.sum(axis=0)
            if orderings:
                per[:, sl] = r
        return (out, per) if orderings else out

    def _psi3_block(self, x, yA, yB, yC):
        tg = self.tg
        t = tg.t
        ys = (yA, yB, yC)
        KI = [self._line_factors(ys[l], t, l) for l in range(3)]
        ex = [np.exp(-float(x[v]) * t)[None] for v in range(3)]
        pw = (t ** (self.rv - 1.0))[None] if self.rv != 1.0 else 1.0
        kv = {}
        for v in range(3):
            kv[v] = sum(ys[l] for l, (a, b) in enumerate(LINES) if v in (a, b))
        Wc = {v: cum_weights(kv[v], tg) for v in range(3)}
        res = []
        for order in ORDERINGS:
            rank = {v: i for i, v in enumerate(order)}
            g = []
            kap = []
            for v in order:
                f = ex[v] * pw
                k = 0.0
                for l, (a, b) in enumerate(LINES):
                    if v not in (a, b):
                        continue
                    other = b if v == a else a
                    K, I = KI[l]
                    f = f * (I if rank[other] > rank[v] else K)
                    k = k + ys[l]
                g.append(f)
                kap.append(k)
            J1 = weighted_cumulative(g[0], kap[0], tg, Wc[order[0]])
            J2 = weighted_cumulative(g[1] * J1, kap[2], tg, Wc[order[2]])
            val = np.einsum("npg,pg->n", g[2] * J2, tg.w)
            res.append(val)
        return np.array(res)


# ============================================================================ angular rule


def u_edges(umin, umax, ustar, dstar):
    span = umax - umin
    e = [umin, umax]
    if umin < span / 8.0:
        top = umin + span / 8.0
        u = max(6.0 * umin, umin + 1e-7 * span)
        while u < top:
            e.append(u)
            u *= 6.0
        lo = top
    else:
        lo = umin
    return e, lo


def angular_nodes(r, k, orders):
    """Nodes (yA, yB) and weights of int dOmega at radius r."""
    k1, k2, k3 = k
    nuni, ngl, nB = orders
    umin, umax = (r - k1) ** 2, (r + k1) ** 2
    span = umax - umin
    ca = (k1 ** 2 + k3 ** 2 - k2 ** 2) / (2.0 * k1 * k3)
    sa = math.sqrt(max(0.0, 1.0 - ca * ca))
    edges = [umin, umax]
    if umin < span / 8.0:
        top = umin + span / 8.0
        u = max(6.0 * umin, umin + 1e-7 * span)
        while u < top:
            edges.append(u)
            u *= 6.0
        start = top
        edges.append(top)
    else:
        start = umin
    edges.extend(start + (umax - start) * np.arange(1, nuni) / nuni)
    ustar = r * r + k1 * k1 - 2.0 * r * k1 * ca
    dstar = max(1e-8 * span, 2.0 * r * k1 * max(sa, 1e-3) * abs(r - k3) / math.sqrt(r * k3))
    cand = [ustar]
    d = dstar
    while d < span / 6.0:
        cand += [ustar - d, ustar + d]
        d *= 6.0
    for c in cand:
        if umin + 1e-9 * span < c < umax - 1e-9 * span:
            edges.append(c)
    edges = np.unique(np.array(edges))
    x, w = np.polynomial.legendre.leggauss(ngl)
    a, b = edges[:-1], edges[1:]
    uA = (0.5 * (a + b))[:, None] + (0.5 * (b - a))[:, None] * x[None, :]
    wA = (0.5 * (b - a))[:, None] * w[None, :]
    uA, wA = uA.ravel(), wA.ravel()
    cth = (uA - r * r - k1 * k1) / (2.0 * r * k1)
    sth = np.sqrt(np.maximum(0.0, 1.0 - cth * cth))
    psi = np.pi * (np.arange(nB) + 0.5) / nB
    uB = (r * r + k3 * k3 + 2.0 * r * k3 * cth[:, None] * ca
          + 2.0 * r * k3 * sth[:, None] * sa * np.cos(psi)[None, :])
    uB = np.maximum(uB, 1e-28)
    yA = np.repeat(np.sqrt(uA), nB)
    yB = np.sqrt(uB).ravel()
    w = np.repeat(wA, nB) * (np.pi / nB) / (r * k1)
    return yA, yB, w


# ============================================================================ radial ladder

SETTINGS = {
    "P1": dict(ang_lo=(6, 8, 24), ang_hi=(9, 10, 40), ang_far=(4, 6, 16), nr=8, nr_mid=10,
               kink_w=0.004, kink_ratio=3.5, kink_zone=0.18, Lam1=2560, geo_lo=8,
               main_ratio=1.3, far_ratio=1.45, nfit=4),
    "P2": dict(ang_lo=(8, 9, 32), ang_hi=(11, 11, 52), ang_far=(5, 7, 20), nr=10, nr_mid=12,
               kink_w=0.002, kink_ratio=3, kink_zone=0.24, Lam1=2560, geo_lo=10,
               main_ratio=1.22, far_ratio=1.35, nfit=5),
}


def _grow(a, b, ratio):
    e = [a]
    while e[-1] * ratio < b:
        e.append(e[-1] * ratio)
    return e


def radial_edges(k, st):
    k1, k2, k3 = k
    e = list(1e-4 * (0.25 / 1e-4) ** (np.arange(st["geo_lo"] + 1) / st["geo_lo"]))
    e += _grow(0.25, 8.0, st["main_ratio"])
    e += _grow(8.0, 160.0, st["main_ratio"] ** 2)
    e += _grow(160.0, float(st["Lam1"]), st["far_ratio"]) + [float(st["Lam1"])]
    e += [float(L) for L in LADDER if L <= st["Lam1"]]
    for kk in (k1, k3):
        s = max(kk, 0.3)
        d = st["kink_w"] * s
        e.append(kk)
        while d < st["kink_zone"] * s:
            e += [kk - d, kk + d]
            d *= st["kink_ratio"]
    e = np.array(sorted(set(float(v) for v in e if 1e-4 <= v <= st["Lam1"])))
    return e


def radial_nodes(k, st):
    """Radial nodes, weights and angular zone per node."""
    k1, k2, k3 = k
    edges = radial_edges(k, st)
    rs, ws, zones = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        n = st["nr_mid"] if 0.25 <= a < 8.0 else st["nr"]
        x, w = np.polynomial.legendre.leggauss(n)
        mid = 0.5 * (a + b)
        if any(abs(mid - K) < st["kink_zone"] * max(K, 0.3) for K in (k1, k3)):
            z = "hi"
        elif a >= 160.0:
            z = "far"
        else:
            z = "lo"
        rs.append(mid + 0.5 * (b - a) * x)
        ws.append(0.5 * (b - a) * w)
        zones += [z] * n
    return np.concatenate(rs), np.concatenate(ws), zones, edges


def f_of_r(tri, k, r, orders):
    """f(r) = r^2 int dOmega psi_3 (complex)."""
    yA, yB, w = angular_nodes(r, k, orders)
    yC = np.full(yA.shape, float(r))
    psi = tri.psi3(k, yA, yB, yC)
    return r * r * np.sum(w * psi), yA.size


# ============================================================================ regulated constant


def B_reg_single(r, f, B_L, Lam1, rfit, D0):
    """Regulated brute-force value B_reg(Lambda_1, r_fit) from stored far nodes (r, f) and B(Lambda_1)."""
    r = np.asarray(r)
    f = np.asarray(f)
    sel = r >= rfit
    rr = r[sel]
    y = f[sel] - 4.0 * np.pi * D0 / rr
    X = np.column_stack([rr ** -2, rr ** -2 * np.log(rr), rr ** -3]).astype(complex)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, b = c
    L = float(Lam1)
    return (B_L - 4.0 * np.pi * D0 * math.log(L) + a0 / L + a1 * (math.log(L) + 1.0) / L
            + b / (2.0 * L * L))


def B_reg(record, D0):
    """Mean over Lambda_1 in {150000, 67000, 30000} x r_fit in {3000, 8000, 20000}; spread."""
    r = np.asarray(record["r_far"])
    f = np.asarray(record["f_far_re"]) + 1j * np.asarray(record["f_far_im"])
    vals = []
    for L in (150000, 67000, 30000):
        BL = complex(*record[f"B_{L}"])
        for rf in (3000, 8000, 20000):
            vals.append(B_reg_single(r, f, BL, L, rf, D0))
    vals = np.array(vals)
    m = vals.mean()
    return m, float(np.max(np.abs(vals - m)))
