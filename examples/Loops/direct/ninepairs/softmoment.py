"""Leading squeezed coefficient M_A of the helical-vector bubble, all nine
helicity pairs, by direct time-domain integration (agent ninepairs).

Method (independent of the draft's Mellin-Barnes formulas): in the region
t1 ~ 1/k1, t2 ~ 1/k3, l ~ k3 that produces (k3/k1)^{2+2i mu}, the hard vertex
is later than the soft one, so every Schwinger-Keldysh line function is a
product of a hard end and a soft end.  The hard ends are replaced by their
late-time +nu branch, the hard time integral is Gamma(4+2nu)/x1^{4+2nu}, and
the soft vertex (time tau, loop momentum l, k3 = 1) is integrated
numerically.  The loop integral diverges at large l as P^{2nu} (P = y1+y2);
it is continued analytically by fitting the large-P expansion
P^{2nu} sum_j a_j P^{-j} and integrating the fit term by term.

  S = (k1 k2 k3)^2 <...>' > 2 Re[M_A r^{2+2i mu}],  r = k3/k1, k1=k2,
  M_A = 2^{-8-2nu} Gamma(4+2nu) sum_{pairs,ch} sum_b int d^3l/(2pi)^3 Pol_ch(c) H_b T_b.
"""
from __future__ import annotations
import math, cmath, time, json, sys, os
import numpy as np
from modes import Line, gamma

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)


def gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


SET = {
    "A": dict(rho_min=1e-10, rho_ratio=2.0, nrho=10, nth=14, P_ratio=1.5, nP=10, nq=20,
              Pmax=1024.0, ntail=8, nfit=6, du=1.0, nu_=12, ulo=-21.0, uhi=4.1),
    "C": dict(rho_min=1e-8, rho_ratio=2.5, nrho=8, nth=10, P_ratio=1.8, nP=8, nq=14,
              Pmax=512.0, ntail=7, nfit=5, du=1.0, nu_=10, ulo=-18.0, uhi=4.1),
    "B": dict(rho_min=1e-11, rho_ratio=1.6, nrho=12, nth=18, P_ratio=1.3, nP=12, nq=26,
              Pmax=2048.0, ntail=9, nfit=7, du=0.7, nu_=12, ulo=-23.0, uhi=4.3),
}


def loop_points(st):
    """Points (y1, y2) with weights for int d^3l/(2pi)^3 at k=1 (without the tail).
    Returns y1, y2, w, and the tail sampling (y1t, y2t, qnode index, P)."""
    pre = 1.0 / (4 * math.pi ** 2)
    Y1, Y2, W = [], [], []
    # corner regions (P in [1,2]); rho = a + b, theta = a/rho, a = P-1, b = 1 -+ q
    edges = [1.0]
    while edges[-1] > st["rho_min"]:
        edges.append(edges[-1] / st["rho_ratio"])
    edges = edges[::-1]
    tr, wr = gl01(st["nrho"])
    tt, wt = gl01(st["nth"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        lr = math.log(hi / lo)
        rho = lo * np.exp(lr * tr); wrho = lr * wr * rho
        for ri, wri in zip(rho, wrho):
            th = tt; wth = wt
            a = ri * th; b = ri * (1 - th)
            for sgn in (1, -1):
                P = 1 + a; q = sgn * (1 - b)
                y1 = (P + q) / 2; y2 = (P - q) / 2
                Y1.append(y1); Y2.append(y2)
                W.append(pre * (P * P - q * q) / 8 * wri * ri * wth)
    # rho in [1, 2]: theta in [1 - 1/rho, 1/rho]
    for lo, hi in ((1.0, 1.5), (1.5, 2.0)):
        rho = lo + (hi - lo) * tr; wrho = (hi - lo) * wr
        for ri, wri in zip(rho, wrho):
            t0, t1 = 1 - 1 / ri, 1 / ri
            th = t0 + (t1 - t0) * tt; wth = (t1 - t0) * wt
            a = ri * th; b = ri * (1 - th)
            for sgn in (1, -1):
                P = 1 + a; q = sgn * (1 - b)
                Y1.append((P + q) / 2); Y2.append((P - q) / 2)
                W.append(pre * (P * P - q * q) / 8 * wri * ri * wth)
    # region P in [2, Pmax]
    xq, wq = np.polynomial.legendre.leggauss(st["nq"])
    pe = [2.0]
    while pe[-1] * st["P_ratio"] < st["Pmax"]:
        pe.append(pe[-1] * st["P_ratio"])
    pe.append(st["Pmax"])
    tp, wp = gl01(st["nP"])
    for lo, hi in zip(pe[:-1], pe[1:]):
        lr = math.log(hi / lo)
        P = lo * np.exp(lr * tp); wP = lr * wp * P
        PP, QQ = np.meshgrid(P, xq, indexing="ij")
        WW = np.outer(wP, wq)
        Y1.append(((PP + QQ) / 2).ravel()); Y2.append(((PP - QQ) / 2).ravel())
        W.append((pre * (PP ** 2 - QQ ** 2) / 8 * WW).ravel())
    y1 = np.concatenate([np.atleast_1d(v) for v in Y1])
    y2 = np.concatenate([np.atleast_1d(v) for v in Y2])
    w = np.concatenate([np.atleast_1d(v) for v in W])
    # tail samples
    Pt = st["Pmax"] * 2.0 ** np.arange(st["ntail"])
    PP, QQ = np.meshgrid(Pt, xq, indexing="ij")
    tail = dict(y1=((PP + QQ) / 2).ravel(), y2=((PP - QQ) / 2).ravel(), P=PP.ravel(), q=QQ.ravel(),
                Pt=Pt, xq=xq, wq=wq, meas=(pre * (PP ** 2 - QQ ** 2) / 8).ravel())
    return y1, y2, w, tail


# ------------------------------------------------------------------ channels
# each ordered pair: (line1, line2, [ (comp1, comp2, polyfun) ])  -- vertex-1 comps are T or L
def pol_TT(h1, h2):
    return lambda c: 0.25 * (1 - h1 * h2 * c) ** 2

POL_TL = lambda c: 0.5 * (1 - c * c)


class Bubble:
    def __init__(self, mu, kappa, setting="A"):
        self.mu, self.kappa = float(mu), float(kappa)
        self.nu = 1j * self.mu
        self.st = SET[setting] if isinstance(setting, str) else setting
        self.lines = {"+": Line(mu, kappa, "T", 1, CACHE), "-": Line(mu, kappa, "T", -1, CACHE),
                      "L": Line(mu, kappa, "L", 0, CACHE)}
        self.pairs = {}
        for a in "+-L":
            for b in "+-L":
                if a in "+-" and b in "+-":
                    ch = [("T", "T", pol_TT(1 if a == "+" else -1, 1 if b == "+" else -1))]
                elif a == "L" and b == "L":
                    ch = [("L", "L", lambda c: c * c), ("0", "0", lambda c: -c)]
                else:
                    ch = [("T" if a != "L" else "L", "T" if b != "L" else "L", POL_TL)]
                self.pairs[a + b] = ch
        # hard coefficients H_b = sum_a h1(a,b) h2(a,b)   (times y-dependence added later)
        self.Hb = {}
        for p in self.pairs:
            L1, L2 = self.lines[p[0]], self.lines[p[1]]
            self.Hb[p] = {b: sum(L1.hard(a, b) * L2.hard(a, b) for a in (1, -1)) for b in (1, -1)}

    def _soft(self, line, comp, x):
        return self.lines[line].soft(comp, x)

    def integrand(self, y1, y2, chunk=1500):
        """Per ordered pair, sum over channels and b of Pol * H_b * T_b at (y1, y2) (k=1),
        without the measure.  Returns dict pair -> complex array."""
        st = self.st
        nu = self.nu
        u0 = np.arange(st["ulo"], st["uhi"] - 1e-9, st["du"])
        tg, wg = gl01(st["nu_"])
        uu = (u0[:, None] + st["du"] * tg[None, :]).ravel()
        wu = np.tile(wg * st["du"], u0.size)
        out = {p: np.zeros(y1.size, complex) for p in self.pairs}
        for i0 in range(0, y1.size, chunk):
            sl = slice(i0, min(y1.size, i0 + chunk))
            a1, a2 = y1[sl], y2[sl]
            tpk = 1.0 / (1.0 + a1 + a2)
            T = tpk[:, None] * np.exp(uu)[None, :]
            WT = T * wu[None, :] * np.exp(-T)
            x1 = 2 * a1[:, None] * T; x2 = 2 * a2[:, None] * T
            cache = {}
            def S(line, comp, which):
                key = (line, comp, which)
                if key not in cache:
                    cache[key] = self._soft(line, comp, x1 if which == 1 else x2)
                return cache[key]
            c = (1 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
            # hard y-dependence: (2y1)^{1/2+nu}(2y2)^{1/2+nu}/sqrt(2y1 2y2) = (2y1)^nu (2y2)^nu
            hy = (2 * a1) ** nu * (2 * a2) ** nu
            norm_soft = 1.0 / np.sqrt(4 * a1 * a2)
            for p, chans in self.pairs.items():
                acc = np.zeros(a1.size, complex)
                for (c1, c2, pol) in chans:
                    Tp = (WT * S(p[0], c1, 1) * S(p[1], c2, 2)).sum(1) * norm_soft
                    acc += pol(c) * (self.Hb[p][1] * Tp + self.Hb[p][-1] * np.conj(Tp))
                out[p][sl] = acc * hy
        return out

    def run(self, verbose=False):
        t0 = time.process_time()
        st = self.st
        y1, y2, w, tail = loop_points(st)
        f = self.integrand(y1, y2)
        body = {p: np.sum(w * v) for p, v in f.items()}
        ft = self.integrand(tail["y1"], tail["y2"])
        nu = self.nu
        Pt = tail["Pt"]; nq = tail["xq"].size; nt = Pt.size
        res = {}
        info = {}
        pre = 2.0 ** (-8 - 2 * nu) * gamma(4 + 2 * nu)
        for p in self.pairs:
            g = (tail["meas"] * ft[p]).reshape(nt, nq)
            tl = 0.0; tl2 = 0.0; resid = 0.0
            for j in range(nq):
                for nf, store in ((st["nfit"], 1), (st["nfit"] - 1, 2)):
                    cols = np.stack([Pt ** (2 * nu - i) for i in range(nf)], 1)
                    sc = np.abs(cols).max(0)
                    cf, *_ = np.linalg.lstsq(cols / sc, g[:, j], rcond=None)
                    cf = cf / sc
                    T = sum(cf[i] * (-(Pt[0] ** (1 + 2 * nu - i)) / (1 + 2 * nu - i)) for i in range(nf))
                    if store == 1:
                        tl += tail["wq"][j] * T
                        resid = max(resid, np.max(np.abs(cols @ cf - g[:, j])) / np.max(np.abs(g[:, j])))
                    else:
                        tl2 += tail["wq"][j] * T
            tot = body[p] + tl
            res[p] = pre * tot
            info[p] = dict(tail_diff=abs(pre * (tl - tl2)), fit_resid=resid, body=pre * body[p], tail=pre * tl)
        self.cpu = time.process_time() - t0
        self.npts = y1.size
        return res, info


def ordered_sum(res):
    return sum(res.values())


# ------------------------------------------------------- late-time rung N = 0
def Iloop(a, b):
    """int d^3l/(2pi)^3 y1^a y2^b at k=1 (dimensional continuation)."""
    A, B = -a / 2, -b / 2
    return (gamma(1.5 - A) * gamma(1.5 - B) * gamma(A + B - 1.5)
            / ((4 * math.pi) ** 1.5 * gamma(A) * gamma(B) * gamma(3 - A - B)))


def rung0(mu, kappa):
    """M_{A,N=0} per ordered pair (soft ends at late time, +nu branch)."""
    nu = 1j * mu
    lines = {"+": Line(mu, kappa, "T", 1, CACHE), "-": Line(mu, kappa, "T", -1, CACHE),
             "L": Line(mu, kappa, "L", 0, CACHE)}
    def soft_coef(line, comp, b):
        Lx = lines[line]
        if comp == "T":
            cp, cm = Lx.Nfac * Lx.Cp, Lx.Nfac * Lx.Cm
        elif comp == "L":
            cp, cm = (-1j / Lx.m) * (nu - 0.5) * Lx.Cp, (-1j / Lx.m) * (-nu - 0.5) * Lx.Cm
        else:
            return 0.0
        return cp if b == 1 else np.conj(cm)
    # polynomial weights as monomials in c; c = (1 - y1^2 - y2^2)/(2 y1 y2)
    # c^0 -> (0,0); c^1 -> 1/2 [y1^-1 y2^-1 - y1 y2^-1 - y1^-1 y2]; c^2 = square
    def cpow(n):
        mono = {(0, 0): 1.0}
        base = {(-1, -1): 0.5, (1, -1): -0.5, (-1, 1): -0.5}
        for _ in range(n):
            new = {}
            for (d1, d2), v in mono.items():
                for (e1, e2), u in base.items():
                    new[(d1 + e1, d2 + e2)] = new.get((d1 + e1, d2 + e2), 0) + v * u
            mono = new
        return mono
    def loopint(coefs):  # coefs: list over c^n
        tot = 0
        for n, cn in enumerate(coefs):
            if cn == 0:
                continue
            for (d1, d2), v in cpow(n).items():
                tot += cn * v * Iloop(2 * nu + d1, 2 * nu + d2) * 2 ** (4 * nu)
        return tot
    pre = 2.0 ** (-8 - 2 * nu) * gamma(4 + 2 * nu) * gamma(2 + 2 * nu)
    out = {}
    for a in "+-L":
        for b in "+-L":
            if a in "+-" and b in "+-":
                s = (1 if a == "+" else -1) * (1 if b == "+" else -1)
                chans = [("T", "T", [0.25, -0.5 * s, 0.25])]
            elif a == "L" and b == "L":
                chans = [("L", "L", [0, 0, 1.0]), ("0", "0", [0, -1.0, 0])]
            else:
                chans = [("T" if a != "L" else "L", "T" if b != "L" else "L", [0.5, 0, -0.5])]
            L1, L2 = lines[a], lines[b]
            tot = 0
            for (c1, c2, coefs) in chans:
                li = loopint(coefs)
                for bb in (1, -1):
                    H = sum(L1.hard(aa, bb) * L2.hard(aa, bb) for aa in (1, -1))
                    tot += H * soft_coef(a, c1, bb) * soft_coef(b, c2, bb) * li
            out[a + b] = pre * tot
    return out


if __name__ == "__main__":
    mu, kappa = float(sys.argv[1]), float(sys.argv[2])
    setting = sys.argv[3] if len(sys.argv) > 3 else "A"
    B = Bubble(mu, kappa, setting)
    res, info = B.run()
    r0 = rung0(mu, kappa)
    MA = ordered_sum(res); M0 = ordered_sum(r0)
    rec = dict(mu=mu, kappa=kappa, setting=setting, cpu=B.cpu, npts=int(B.npts),
               MA=[MA.real, MA.imag], M0=[M0.real, M0.imag],
               pairs={p: [v.real, v.imag] for p, v in res.items()},
               rung0={p: [v.real, v.imag] for p, v in r0.items()},
               tail_diff={p: info[p]["tail_diff"] for p in res},
               fit_resid={p: info[p]["fit_resid"] for p in res})
    print(json.dumps(rec), flush=True)
