"""Full time-domain bispectrum of the helical-vector bubble with all nine
helicity pairs and the four Schwinger-Keldysh sectors (agent ninepairs).

The loop momentum is cut at P = y1 + y2 < Pc with Pc fixed (independent of
r = k3/k1, below the hard vertex energy x1 = 2).  The cut changes S(r) only by
terms analytic in r (the soft-region integrand at P >> r is analytic in r),
so the coefficients of r^{2+2i mu+n} are those of the uncut diagram.  With
Pc < x1 the factorized form of the time-ordered sector is integrable over the
whole quadrant, and
   F_{++} = F_fact + R,   R = int_{t2<t1} [prod e*(it1) prod e(it2) - prod e(it1) prod e*(it2)],
so the ordered remainder R of the draft (N147) is obtained separately.
S = (k1 k2 k3/16) int d^3l/(2pi)^3 sum Pol * 2 Re[F_fact + F_{+-} + R].
"""
from __future__ import annotations
import math, cmath, os, sys, json, time, hashlib
import numpy as np
from flint import acb, ctx
from modes import whitW, gamma, conn, Line, XLO, XHI, _cheb
from softmoment import loop_points, CACHE, gl01

ctx.prec = 110
WID = 0.25
DEG = 16


def _c(v):
    return complex(float(v.real.mid()), float(v.imag.mid()))


def M_and_zdM(k, nu, x):
    """(M_{k,nu}(x), x d/dx M_{k,nu}(x)) via 1F1 (arb)."""
    X = acb(float(x)); K = acb(complex(k).real, complex(k).imag); N = acb(complex(nu).real, complex(nu).imag)
    a = acb(0.5) + N - K; b = 1 + 2 * N
    pre = (-X / 2).exp() * X ** (N + acb(0.5))
    F = X.hypgeom_1f1(a, b); F1 = X.hypgeom_1f1(a + 1, b + 1)
    M = pre * F
    zdM = (acb(0.5) + N - X / 2) * M + pre * X * (a / b) * F1
    return _c(M), _c(zdM)


class STable:
    """Scaled function s(x) on Chebyshev panels in ln x, x in [XLO, XHI];
    small(x), big(x) callables for the two ends (all scaled)."""

    def __init__(self, func, small, big, key, ):
        self.L0 = math.log(XLO)
        self.npan = int(math.ceil((math.log(XHI) - self.L0) / WID))
        self.s, self.bw = _cheb(DEG)
        self.lnxc = self.L0 + WID * (np.arange(self.npan) + 0.5)
        X = np.exp(self.lnxc[:, None] + 0.5 * WID * self.s[None, :])
        h = hashlib.sha1(repr(key).encode()).hexdigest()[:20]
        path = os.path.join(CACHE, f"stab_{h}.npy")
        if os.path.exists(path):
            self.V = np.load(path)
        else:
            self.V = np.array([[func(x) for x in row] for row in X])
            np.save(path, self.V)
        self.small, self.big = small, big

    def __call__(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape, complex)
        lo = x < XLO; hi = x > XHI; mid = ~(lo | hi)
        if lo.any():
            out[lo] = self.small(x[lo])
        if hi.any():
            out[hi] = self.big(x[hi])
        if mid.any():
            xm = x[mid]; lx = np.log(xm)
            p = np.clip(np.floor((lx - self.L0) / WID).astype(np.int64), 0, self.npan - 1)
            loc = (lx - self.lnxc[p]) / (0.5 * WID)
            d = loc[:, None] - self.s[None, :]
            ex = d == 0.0
            d = np.where(ex, 1.0, d)
            c = self.bw[None, :] / d
            v = (c * self.V[p]).sum(1) / c.sum(1)
            if ex.any():
                r = np.nonzero(ex.any(1))[0]
                v[r] = self.V[p[r], np.argmax(ex[r], 1)]
            out[mid] = v
        return out


def asym_coefs(a, b, n=40):
    c = [1.0 + 0j]
    for j in range(1, n):
        c.append(c[-1] * (a + j - 1) * (b + j - 1) / j)
    return np.array(c)


class FullLine:
    """Scaled end functions of one line type:
       dec(comp, z) = e^{+z/2} sqrt(2y) e*_comp(it)   (b=+ soft-type end, decaying)
       grow(comp, z) = e^{-z/2} sqrt(2y) e_comp(it)   (growing end)
    comp in {'T'} for transverse lines, {'L','0'} for the longitudinal line."""

    def __init__(self, mu, kappa, kind, h=0):
        L = Line(mu, kappa, kind, h, CACHE)
        self.L = L
        nu = L.nu; m = L.m; K = L.K; Nf = L.Nfac
        self.kind = kind
        key = (round(mu, 12), round(-K.imag, 12))
        Ap, Am = L.A[1], L.A[-1]
        Cp, Cm = L.Cp, L.Cm
        kw = -K
        # ---- decaying W_{-K,nu}: scaled e^{x/2} W
        cdec = asym_coefs(0.5 + nu - kw, 0.5 - nu - kw)       # W_{kw,nu}(x) ~ e^{-x/2} x^{kw} sum c_n (-x)^{-n}
        def big_dec(x, c=cdec, kw=kw):
            s = sum(c[n] * (-1.0 / x) ** n for n in range(len(c)))
            return x ** kw * s
        def small_dec(x):
            return np.exp(x / 2) * (Cp * x ** (0.5 + nu) * (1 - kw / (1 + 2 * nu) * x)
                                    + Cm * x ** (0.5 - nu) * (1 - kw / (1 - 2 * nu) * x))
        self.Wd = STable(lambda x: math.exp(x / 2) * whitW(kw, nu, x), small_dec, big_dec, ("Wd",) + key)
        # ---- growing: G(x) = A+ M_{-K,nu}(x) + A- M_{-K,-nu}(x) = Nf W_{K,nu}(x e^{-i pi}); scaled e^{-x/2} G
        cgr = asym_coefs(0.5 + nu - K, 0.5 - nu - K)
        ph = cmath.exp(-1j * math.pi * K) * Nf
        def big_gr(x, c=cgr, K=K, ph=ph):
            s = sum(c[n] * (1.0 / x) ** n for n in range(len(c)))
            return ph * x ** K * s
        def small_gr(x):
            e1p = -kw / (1 + 2 * nu); e1m = -kw / (1 - 2 * nu)
            return np.exp(-x / 2) * (Ap * x ** (0.5 + nu) * (1 + e1p * x) + Am * x ** (0.5 - nu) * (1 + e1m * x))
        def f_gr(x):
            Mp, _ = M_and_zdM(kw, nu, x); Mm, _ = M_and_zdM(kw, -nu, x)
            return math.exp(-x / 2) * (Ap * Mp + Am * Mm)
        self.Gg = STable(f_gr, small_gr, big_gr, ("Gg",) + key)
        if kind == "L":
            # (x d - 1) W_{0,nu}  decaying, scaled
            def f_dd(x):
                return math.exp(x / 2) * ((x / 2 - 1) * whitW(0, nu, x) - whitW(1, nu, x))
            def big_dd(x, c=cdec):
                s = sum(c[n] * (x / 2 - n - 1) * (-1.0 / x) ** n for n in range(len(c)))
                return s
            def small_dd(x):
                return np.exp(x / 2) * (Cp * (nu - 0.5) * x ** (0.5 + nu) + Cm * (-nu - 0.5) * x ** (0.5 - nu))
            self.Dd = STable(f_dd, small_dd, big_dd, ("Dd",) + key)
            def f_dg(x):
                Mp, dMp = M_and_zdM(0, nu, x); Mm, dMm = M_and_zdM(0, -nu, x)
                return math.exp(-x / 2) * (Ap * (dMp - Mp) + Am * (dMm - Mm))
            def big_dg(x, c=cgr):
                s = sum(c[n] * (x / 2 - n - 1) * (1.0 / x) ** n for n in range(len(c)))
                return Nf * s
            def small_dg(x):
                return np.exp(-x / 2) * (Ap * (nu - 0.5) * x ** (0.5 + nu) + Am * (-nu - 0.5) * x ** (0.5 - nu))
            self.Dg = STable(f_dg, small_dg, big_dg, ("Dg",) + key)
        self.m = m

    def dec(self, comp, z):
        if comp == "T":
            return self.L.Nfac * self.Wd(z)
        if comp == "L":
            return (-1j / self.m) * self.Dd(z)
        if comp == "0":
            return (-0.5j / self.m) * z * self.Wd(z)

    def grow(self, comp, z):
        if comp == "T":
            return self.Gg(z)
        if comp == "L":
            return (1j / self.m) * self.Dg(z)
        if comp == "0":
            return (-0.5j / self.m) * z * self.Gg(z)


# ------------------------------------------------------------- time grid
def cheb_cumulative(n):
    """Chebyshev-Lobatto nodes s (ascending) on [-1,1], Clenshaw-Curtis weights,
    and the cumulative integration matrix Q: (Q f)_k = int_{-1}^{s_k} f."""
    j = np.arange(n + 1)
    s = -np.cos(np.pi * j / n)
    # Vandermonde in Chebyshev basis
    T = np.cos(np.outer(np.arccos(s), np.arange(n + 1)))
    Tinv = np.linalg.inv(T)
    # integrals of T_k from -1 to s
    I = np.zeros((n + 1, n + 1))
    th = np.arccos(s)
    for k in range(n + 1):
        if k == 0:
            I[:, k] = s + 1
        elif k == 1:
            I[:, k] = (s ** 2 - 1) / 2
        else:
            Tk1 = np.cos((k + 1) * th); Tkm = np.cos((k - 1) * th)
            I[:, k] = Tk1 / (2 * (k + 1)) - Tkm / (2 * (k - 1)) - ((-1) ** (k + 1) / (2 * (k + 1)) - (-1) ** (k - 1) / (2 * (k - 1)))
    Q = I @ Tinv
    w = Q[-1]
    return s, w, Q


NC = 12
S_N, W_N, Q_N = cheb_cumulative(NC)


def tgrid(tlo, thi, width=0.3):
    L0, L1 = math.log(tlo), math.log(thi)
    npan = max(1, int(math.ceil((L1 - L0) / width)))
    edges = np.linspace(L0, L1, npan + 1)
    h = (edges[1] - edges[0]) / 2
    u = (edges[:-1, None] + h * (S_N[None, :] + 1))           # npan x (NC+1)
    return np.exp(u), h, npan


class FullBubble:
    def __init__(self, mu, kappa, Pc=0.5, x1=2.0, loopset=None, twidth=0.3):
        self.mu, self.kappa, self.Pc, self.x1 = mu, kappa, Pc, x1
        self.lines = {"+": FullLine(mu, kappa, "T", 1), "-": FullLine(mu, kappa, "T", -1),
                      "L": FullLine(mu, kappa, "L", 0)}
        self.twidth = twidth
        self.tlo_fac = 1e-7
        self.loopset = loopset or dict(rho_min=1e-8, rho_ratio=2.0, nrho=10, nth=12, P_ratio=1.5, nP=10,
                                       nq=16, Pmax=None, ntail=1, nfit=1)
        # channels: pair -> list of (i1, i2, j1, j2, polycoefs in c)
        ch = {}
        for a in "+-L":
            for b in "+-L":
                if a in "+-" and b in "+-":
                    s = (1 if a == "+" else -1) * (1 if b == "+" else -1)
                    ch[a + b] = [("T", "T", "T", "T", (0.25, -0.5 * s, 0.25))]
                elif a == "L" and b == "L":
                    ch[a + b] = [("L", "L", "L", "L", (0, 0, 1.0)), ("L", "L", "0", "0", (0, -1.0, 0)),
                                 ("0", "0", "L", "L", (0, -1.0, 0)), ("0", "0", "0", "0", (1.0, 0, 0))]
                else:
                    ca = "T" if a != "L" else "L"; cb = "T" if b != "L" else "L"
                    ch[a + b] = [(ca, cb, ca, cb, (0.5, 0, -0.5))]
        self.ch = ch

    def point_values(self, y1, y2, x2):
        """For arrays y1, y2 (loop energies) at soft vertex energy x2, return per pair
        dict of (Ffact + F+-, R) summed over channels with Pol, complex (2Re taken later)."""
        x1 = self.x1
        n = y1.size
        Y = y1 + y2
        # common time grid for all points of the batch, in units of 1/(x1)... per point scale
        tlo = 1e-7 / (x1 + Y)
        thi_hard = 45.0 / (x1 - Y)              # hard vertex with growing ends (Y < x1)
        thi_soft = 50.0 / (x2 + Y)
        out = {p: [np.zeros(n, complex), np.zeros(n, complex)] for p in self.ch}
        for i in range(n):
            res = self._one(y1[i], y2[i], x2, tlo[i], thi_hard[i], thi_soft[i])
            for p in self.ch:
                out[p][0][i] = res[p][0]; out[p][1][i] = res[p][1]
        return out

    def _one(self, y1, y2, x2, tlo, thh, ths):
        x1 = self.x1
        Y = y1 + y2
        thi = max(thh, ths)
        T, h, npan = tgrid(tlo, thi, self.twidth)
        t = T.ravel()
        wt = (h * np.tile(W_N, npan)) * t                    # d t = t du
        z1 = 2 * y1 * t; z2 = 2 * y2 * t
        c = (x2 ** 2 - y1 ** 2 - y2 ** 2) / (2 * y1 * y2)
        cache = {}
        def fn(line, comp, kind, which):
            key = (line, comp, kind, which)
            if key not in cache:
                L = self.lines[line]
                z = z1 if which == 1 else z2
                cache[key] = (L.dec(comp, z) if kind == "d" else L.grow(comp, z))
            return cache[key]
        e_h_g = t ** 2 * np.exp(-(x1 - Y) * t)       # hard vertex, growing ends
        e_h_d = t ** 2 * np.exp(-(x1 + Y) * t)       # hard vertex, decaying ends
        e_s_g = np.where(t <= thh, np.exp(-(x2 - Y) * np.minimum(t, thh)), 0.0)   # only inside R (t2 < t1 <~ thh)
        e_h_g = np.where(t <= thh, e_h_g, 0.0)
        e_s_d = np.exp(-(x2 + Y) * t)
        # mask the hard-growing integrand beyond its range (avoid overflow is not an issue: Y < x1)
        res = {}
        norm = 1.0 / (4 * y1 * y2)
        for p, chans in self.ch.items():
            Ff = 0j; RR = 0j
            for (i1, i2, j1, j2, pc) in chans:
                pol = pc[0] + pc[1] * c + pc[2] * c * c
                if pol == 0:
                    continue
                hg = e_h_g * fn(p[0], i1, "g", 1) * fn(p[1], i2, "g", 2)
                hd = e_h_d * fn(p[0], i1, "d", 1) * fn(p[1], i2, "d", 2)
                sd = e_s_d * fn(p[0], j1, "d", 1) * fn(p[1], j2, "d", 2)
                sg = e_s_g * fn(p[0], j1, "g", 1) * fn(p[1], j2, "g", 2)
                Hg = np.dot(wt, hg); Hd = np.dot(wt, hd); Sd = np.dot(wt, sd)
                fact = Hg * Sd + Hd * np.conj(Sd)
                # R = int_{t2<t1} [hd(t1) sg(t2) - hg(t1) sd(t2)]
                A = self._cum(sg * t, h, npan)          # int_0^{t1} sg dt2 at the nodes
                Bc = self._cum(sd * t, h, npan)
                Rv = np.dot(wt, hd * A - hg * Bc)
                Ff += pol * fact; RR += pol * Rv
            res[p] = (Ff * norm, RR * norm)
        return res

    @staticmethod
    def _cum(f, h, npan):
        F = f.reshape(npan, NC + 1)
        loc = (F @ Q_N.T) * h                         # within-panel cumulative
        tot = loc[:, -1]
        off = np.concatenate([[0.0], np.cumsum(tot)[:-1]])
        return (loc + off[:, None]).ravel()

    def groups(self, r):
        """Loop points (k = r) in groups sharing a time grid: list of (y1, y2, w)."""
        st = self.loopset
        pre = 1.0 / (4 * math.pi ** 2)
        G = []
        edges = [1.0]
        while edges[-1] > st["rho_min"]:
            edges.append(edges[-1] / st["rho_ratio"])
        edges = edges[::-1]
        tr, wr = gl01(st["nrho"]); tt, wt = gl01(st["nth"])
        segs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            lr = math.log(hi / lo); rho = lo * np.exp(lr * tr)
            for ri, wri in zip(rho, lr * wr * rho):
                segs.append((ri, wri, tt, wt))
        for lo, hi in ((1.0, 1.5), (1.5, 2.0)):
            for ri, wri in zip(lo + (hi - lo) * tr, (hi - lo) * wr):
                t0, t1 = 1 - 1 / ri, 1 / ri
                segs.append((ri, wri, t0 + (t1 - t0) * tt, (t1 - t0) * wt))
        for ri, wri, th, wth in segs:
            a = ri * th; b = ri * (1 - th)
            for sgn in (1, -1):
                P = 1 + a; q = sgn * (1 - b)
                G.append(((P + q) / 2 * r, (P - q) / 2 * r, pre * (P * P - q * q) / 8 * wri * ri * wth * r ** 3))
        xq, wq = np.polynomial.legendre.leggauss(st["nq"])
        Pmax = self.Pc / r
        pe = [2.0]
        while pe[-1] * st["P_ratio"] < Pmax:
            pe.append(pe[-1] * st["P_ratio"])
        pe.append(Pmax)
        tp, wp = gl01(st["nP"])
        for lo, hi in zip(pe[:-1], pe[1:]):
            lr = math.log(hi / lo)
            for P, wP in zip(lo * np.exp(lr * tp), lr * wp * lo * np.exp(lr * tp)):
                G.append(((P + xq) / 2 * r, (P - xq) / 2 * r, pre * (P * P - xq ** 2) / 8 * wP * wq * r ** 3))
        return G

    def group_values(self, y1, y2, x2):
        """Vectorized over a group: per pair (F, R) arrays (before Pol-sum over channels done inside)."""
        x1 = self.x1
        Y = y1 + y2
        Ymax, Ymin = Y.max(), Y.min()
        tlo = self.tlo_fac / (x1 + Ymax)
        thh = 45.0 / (x1 - Ymax)
        ths = 50.0 / (x2 + Ymin)
        T, h, npan = tgrid(tlo, max(thh, ths), self.twidth)
        t = T.ravel()[None, :]
        wt = ((h * np.tile(W_N, npan)) * T.ravel())[None, :]
        y1c = y1[:, None]; y2c = y2[:, None]; Yc = Y[:, None]
        z1 = 2 * y1c * t; z2 = 2 * y2c * t
        c = (x2 ** 2 - y1 ** 2 - y2 ** 2) / (2 * y1 * y2)
        cache = {}
        def fn(line, comp, kind, which):
            key = (line, comp, kind, which)
            if key not in cache:
                L = self.lines[line]
                z = z1 if which == 1 else z2
                cache[key] = (L.dec(comp, z) if kind == "d" else L.grow(comp, z))
            return cache[key]
        inh = t <= thh
        e_h_g = np.where(inh, t ** 2 * np.exp(-(x1 - Yc) * t), 0.0)
        e_h_d = t ** 2 * np.exp(-(x1 + Yc) * t)
        e_s_g = np.where(inh, np.exp(-(x2 - Yc) * np.minimum(t, thh)), 0.0)
        e_s_d = np.exp(-(x2 + Yc) * t)
        norm = 1.0 / (4 * y1 * y2)
        res = {}
        for p, chans in self.ch.items():
            Ff = 0j; RR = 0j
            for (i1, i2, j1, j2, pc) in chans:
                pol = pc[0] + pc[1] * c + pc[2] * c * c
                hg = e_h_g * fn(p[0], i1, "g", 1) * fn(p[1], i2, "g", 2)
                hd = e_h_d * fn(p[0], i1, "d", 1) * fn(p[1], i2, "d", 2)
                sd = e_s_d * fn(p[0], j1, "d", 1) * fn(p[1], j2, "d", 2)
                sg = e_s_g * fn(p[0], j1, "g", 1) * fn(p[1], j2, "g", 2)
                Hg = (wt * hg).sum(1); Hd = (wt * hd).sum(1); Sd = (wt * sd).sum(1)
                fact = Hg * Sd + Hd * np.conj(Sd)
                A = self._cum2(sg * t, h, npan); Bc = self._cum2(sd * t, h, npan)
                Rv = (wt * (hd * A - hg * Bc)).sum(1)
                Ff = Ff + pol * fact; RR = RR + pol * Rv
            res[p] = (Ff * norm, RR * norm)
        return res

    @staticmethod
    def _cum2(f, h, npan):
        n = f.shape[0]
        F = f.reshape(n, npan, NC + 1)
        loc = (F @ Q_N.T) * h
        tot = loc[:, :, -1]
        off = np.concatenate([np.zeros((n, 1)), np.cumsum(tot, 1)[:, :-1]], 1)
        return (loc + off[:, :, None]).reshape(n, -1)

    def S2(self, r):
        t0 = time.process_time()
        acc = {p: [0.0, 0.0] for p in self.ch}
        npts = 0
        for (y1, y2, w) in self.groups(r):
            v = self.group_values(np.atleast_1d(y1), np.atleast_1d(y2), r)
            npts += np.size(y1)
            for p, (F, R) in v.items():
                acc[p][0] += (r / 16) * 2 * np.sum(w * F).real
                acc[p][1] += (r / 16) * 2 * np.sum(w * R).real
        self.cpu = time.process_time() - t0
        self.npts = npts
        return {p: tuple(v) for p, v in acc.items()}

    def S(self, r):
        """S(r) per pair: (S_fact, S_R) with k1 = k2 = 1, k3 = r (x2 = r)."""
        t0 = time.process_time()
        st = dict(self.loopset); st["Pmax"] = self.Pc / r
        y1, y2, w, _ = loop_points(st)
        y1 = y1 * r; y2 = y2 * r; w = w * r ** 3
        vals = self.point_values(y1, y2, r)
        out = {}
        for p, (F, R) in vals.items():
            out[p] = ((r / 16) * 2 * np.sum(w * F).real, (r / 16) * 2 * np.sum(w * R).real)
        self.cpu = time.process_time() - t0
        self.npts = y1.size
        return out
