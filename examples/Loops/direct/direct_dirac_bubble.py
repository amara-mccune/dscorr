"""Direct time-domain evaluation of the Dirac-pair bubble with an axial chemical potential, as a check of the
squeezed series of examples/Loops/dirac_pair.py (Secs. 8.2 and 8.3).

Nothing is expanded. The bubble joins a quartic vertex at x1 = k12 = 1 (time weight t^(r1 - 2), r1 = 4) to a cubic
vertex at x2 = k3 = r (time weight t^(r2 - 2), r2 = 2), the vertices of Han, Lei, Qin and Yin (2611.20919). Its two
lines are the exact Schwinger-Keldysh functions of the scalar bilinear, in the conventions of ext/dirac.py:

    <O(tau) O(tau')> = int d^3l/(2 pi)^3 sum_{h1 h2} (1 - h1 h2 c)/2  conj(A(tau)) A(tau'),
    A(tau) = [abar_1 abar_2 - bbar_1 bbar_2](tau),

with abar, bbar the conjugate modes of helicities h1, h2 and momenta y1 = |l|, y2 = |k3 - l|. All four branch
assignments of the two vertices are kept, with both time orderings of the time-ordered ones. On the fully rotated
contour the in-in sum is  -2 I  (for r1 - r2 = 2 and r1 + r2 = 6; all couplings and external normalizations set to
one), with

    I(r) = Re int d^3l/(2 pi)^3 sum_{h1 h2} (1 - h1 h2 c)/2 e^(pi (h1 + h2) kappa)/(4 y1 y2) [ T + D ],
    D = P_1 conj(P_2),                P_v = int dt t^(r_v - 2) e^(-x_v t) Wbar(t),
    T = int_{t1 < t2} t1^(r1 - 2) t2^(r2 - 2) e^(-x1 t1 - x2 t2) G(t1) Wbar(t2) + (1 <-> 2),
    Wbar(t) = sum_{X = a, b} eps_X cbar_X^(h1) cbar_X^(h2) W_{Kbar_X^(h1), nu}(2 y1 t) W_{Kbar_X^(h2), nu}(2 y2 t),
    G(t)    = sum_X eps_X c_X^(h1) c_X^(h2) W_{K_X^(h1), nu}(2 y1 t e^(-i pi)) W_{K_X^(h2), nu}(2 y2 t e^(-i pi)).

D is the doubly cut term of dirac_pair.py and T the time-ordered one. The loop momentum is cut at y1 + y2 < Pc (in
units of k12), which changes I only by terms analytic in r. The series predicts (dirac_bubble_fit.py)

    I(r)  contains  Re sum_n Mcal_n r^(1 + 2 nu + n),
    Mcal_n = [1 - (-1)^n e^(-2 pi mubar)] Gamma(4 + 2 nu + n) S(n; +) + [1 - (-1)^n e^(2 pi mubar)] conj[Gamma(4 - 2 nu + n) S(n; -)],

with S(n; s) the soft moments of dirac_pair.py on the two pair branches.

Mode functions: e^(z/2) W_{K,nu}(z) and e^(-z/2) W_{K,nu}(z e^(-i pi)) on Chebyshev panels in ln z (class Modes of
direct_fermi2.py, python-flint). Time grid and cumulative integration as in direct_fermi2.py.

Usage: direct_dirac_bubble.py m kappa out.jsonl r1 r2 ...
       [env: PC (1.5), P0 (0.5), GRID (std | fine), LOOPJSON, ONQ (12), ONP (6), ORATIO (1.4), TWIDTH (0.3), NCHEB (12),
        TLO (1e-6), THI (50), NINEPAIRS]
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import direct_fermi2 as F2  # noqa: E402  (mode tables, time grid; reads NCHEB, GRID, LOOPJSON, TLO)
from direct_fermi2 import Modes, tgrid, loop_points, gl01  # noqa: E402

LOOP = dict(F2.LOOP)
TLO = F2.TLO
THI = float(os.environ.get("THI", 50.0))
HEL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def comps(m, kappa, h):
    """{X: (c_X, K_X)} of ext/dirac.py for helicity h."""
    Ka, Kb = h * (0.5 - 1j * kappa), -h * (0.5 + 1j * kappa)
    ca, cb = (1.0 + 0j, -1j * m) if h == 1 else (m + 0j, 1j)
    return {"a": (ca, Ka), "b": (cb, Kb)}


class Bubble:
    def __init__(self, m, kappa, Pc=1.5, p0=0.5, r1=4, r2=2, twidth=0.3, chunk=200, oratio=1.4, onP=6, onq=12):
        self.m, self.kappa = float(m), float(kappa)
        self.M = Modes(m, kappa)
        self.Pc, self.p0, self.r1, self.r2 = Pc, p0, r1, r2
        self.twidth, self.chunk, self.oratio, self.onP, self.onq = twidth, chunk, oratio, onP, onq

    def loop(self, r):
        """(y1, y2, weight) of the loop grid in units of k3, for int d^3l/(2 pi)^3 with y1 + y2 < Pc/r."""
        st = dict(LOOP)
        p0 = min(self.p0, self.Pc)
        st["Pmax"] = p0 / r
        y1, y2, w, _ = loop_points(st)
        Y1, Y2, W = [y1], [y2], [w]
        if self.Pc > p0 * (1 + 1e-12):          # outer shells, with edges fixed in units of k12
            npan = max(1, math.ceil(math.log(self.Pc / p0) / math.log(self.oratio) - 1e-9))
            edges = p0 * (self.Pc / p0) ** (np.arange(npan + 1) / npan)
            tp, wp = gl01(self.onP)
            xq, wq = np.polynomial.legendre.leggauss(self.onq)
            pre = 1.0 / (4 * math.pi ** 2)
            for lo, hi in zip(edges[:-1], edges[1:]):
                lr = math.log(hi / lo)
                P = lo / r * np.exp(lr * tp)
                wP = lr * wp * P
                PP, QQ = np.meshgrid(P, xq, indexing="ij")
                Y1.append(((PP + QQ) / 2).ravel())
                Y2.append(((PP - QQ) / 2).ravel())
                W.append((pre * (PP ** 2 - QQ ** 2) / 8 * np.outer(wP, wq)).ravel())
        return np.concatenate(Y1), np.concatenate(Y2), np.concatenate(W)

    def I(self, r, parts=False):
        """I(r) at k12 = 1 and k3 = r. With parts=True, also the time-ordered and the doubly cut pieces."""
        M, m, ka = self.M, self.m, self.kappa
        NC, W_N, Q_N = F2.NC, F2.W_N, F2.Q_N
        y1, y2, w = self.loop(r)
        tg, hstep, npan = tgrid(TLO, THI / r, self.twidth)
        tt = tg.ravel()

        def cum(Fn):
            Fp = Fn.reshape(Fn.shape[:-1] + (npan, NC + 1))
            loc = (Fp @ Q_N.T) * hstep
            off = np.cumsum(loc[..., -1], axis=-1)
            off = np.concatenate([np.zeros_like(off[..., :1]), off[..., :-1]], -1)
            return (loc + off[..., None]).reshape(Fn.shape)

        def integ(Fn):
            return (Fn.reshape(Fn.shape[:-1] + (npan, NC + 1)) * W_N).sum((-1, -2)) * hstep

        cp = {h: comps(m, ka, h) for h in (1, -1)}
        Ks = sorted({K for h in cp for (_, K) in cp[h].values()}, key=lambda z: (z.real, z.imag))
        tot_T = tot_D = 0.0
        for i0 in range(0, y1.size, self.chunk):
            a1, a2, ww = y1[i0:i0 + self.chunk], y2[i0:i0 + self.chunk], w[i0:i0 + self.chunk]
            E1, E2 = r * a1, r * a2                              # line energies in units of k12
            P = (E1 + E2)[:, None]
            z = {1: 2 * E1[:, None] * tt[None, :], 2: 2 * E2[:, None] * tt[None, :]}
            dec, grow = {}, {}
            for line in (1, 2):
                zf = z[line].ravel()
                for K in Ks:
                    dec[(line, K)] = M.dec(K, zf).reshape(z[line].shape)            # e^(z/2) W_{K,nu}(z)
                    grow[(line, K)] = M.grow(K, 1, zf).reshape(z[line].shape)       # e^(-z/2) W_{K,nu}(z e^(-i pi))
            lt = np.log(tt)[None, :]
            # vertex weights with the Jacobian of ln t; d: both lines decaying, g: both growing
            wd = {1: np.exp(np.clip((self.r1 - 1) * lt - (1.0 + P) * tt, -700.0, 600.0)),
                  2: np.exp(np.clip((self.r2 - 1) * lt - (r + P) * tt, -700.0, 600.0))}
            wg = {1: np.exp(np.clip((self.r1 - 1) * lt - (1.0 - P) * tt, -700.0, 600.0)),
                  2: np.exp(np.clip((self.r2 - 1) * lt - (r - P) * tt, -700.0, 600.0))}
            c = (1 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
            accT = np.zeros(a1.size, complex)
            accD = np.zeros(a1.size, complex)
            for h1, h2 in HEL:
                Wb, G = 0, 0
                for X, eps in (("a", 1), ("b", -1)):
                    (c1, K1), (c2, K2) = cp[h1][X], cp[h2][X]
                    Wb = Wb + eps * np.conj(c1 * c2) * dec[(1, K1.conjugate())] * dec[(2, K2.conjugate())]
                    G = G + eps * c1 * c2 * grow[(1, K1)] * grow[(2, K2)]
                d1, d2 = wd[1] * Wb, wd[2] * Wb
                g1, g2 = wg[1] * G, wg[2] * G
                wh = (1 - h1 * h2 * c) / 2 * math.exp(math.pi * (h1 + h2) * ka) / (4 * E1 * E2)
                accD += wh * integ(d1) * np.conj(integ(d2))
                accT += wh * (integ(d2 * cum(g1)) + integ(d1 * cum(g2)))
            tot_T += np.sum(ww * accT)
            tot_D += np.sum(ww * accD)
        # measure r^3 from l = r q
        T, D = r ** 3 * tot_T, r ** 3 * tot_D
        if parts:
            return (T + D).real, T, D, int(y1.size), int(tt.size)
        return (T + D).real


if __name__ == "__main__":
    m, ka, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    B = Bubble(m, ka, Pc=float(os.environ.get("PC", 1.5)), p0=float(os.environ.get("P0", 0.5)),
               twidth=float(os.environ.get("TWIDTH", 0.3)), oratio=float(os.environ.get("ORATIO", 1.4)),
               onP=int(os.environ.get("ONP", 6)), onq=int(os.environ.get("ONQ", 12)))
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["r"] for l in open(out)}
    for s in sys.argv[4:]:
        r = float(s)
        if r in done:
            continue
        t0 = time.process_time()
        val, T, D, npts, nt = B.I(r, parts=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(m=m, kappa=ka, r=r, I=val, T=[T.real, T.imag], D=[D.real, D.imag], npts=npts, nt=nt, Pc=B.Pc, P0=B.p0,
                                     r1=B.r1, r2=B.r2, onq=B.onq, onP=B.onP, oratio=B.oratio,
                                     grid=os.environ.get("GRID", "std") + ("+" + os.environ["LOOPJSON"] if os.environ.get("LOOPJSON") else ""),
                                     twidth=B.twidth, ncheb=F2.NC, tlo=TLO, thi=THI, cpu=round(time.process_time() - t0, 1))) + "\n")
