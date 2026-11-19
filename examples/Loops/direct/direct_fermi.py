"""Direct time-domain evaluation of the fermion triangle with an axial chemical potential at fixed shape,
as a check of the series of examples/Loops/ext/fermitri.py.

Nothing is expanded. One orientation of the loop,

    I(r, c) = sum_{a1 a2 a3} a1 a2 a3 int d^3q/(2pi)^3 int dt1 dt2 dt3
              Tr[Gamma.F(k1) D_{a1a2}(q + k1) Gamma.F(k2) D_{a2a3}(q - k3) Gamma.F(k3) D_{a3a1}(q)],

is evaluated on the fully rotated contour with the exact Schwinger-Keldysh functions of the three lines
(conventions of ext/fermion.py), all time orderings, the exact energies and helicity projectors, and
k1 = 1, |k3| = r, c = k1hat . k3hat. The sum over branches is 2 Re of the terms with a3 = +. The loop momentum
is cut at |q| + |q - k3| < Pc, which changes I only by terms analytic in r. The series predicts

    2 I(r, c)  contains  sum_n C_n(c) r^(2 nu + n) + c.c.

Mode functions: e^(z/2) W_{K,nu}(z) and e^(-z/2) W_{K,nu}(z e^(-i pi)) on Chebyshev panels in ln z (python-flint),
K = +-1/2 +- i kappa. Time grid and cumulative integration as in direct_triangle.py.

Usage: direct_fermi.py m kappa c out.jsonl r1 r2 ...     (c = 'iso' for k1 = k2)   [env: NPHI, PC, STRUCT, NINEPAIRS]
"""
import cmath
import json
import math
import os
import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.environ.get("NINEPAIRS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ninepairs")))
from full import STable, tgrid, W_N, NC, Q_N  # noqa: E402
from modes import whitW, whitM, gamma  # noqa: E402
from softmoment import loop_points  # noqa: E402

LOOP = dict(rho_min=1e-7, rho_ratio=2.5, nrho=8, nth=8, P_ratio=1.8, nP=8, nq=10, Pmax=None, ntail=1, nfit=1)
if os.environ.get("GRID", "std") == "std":
    LOOP.update(nrho=6, nth=6, nP=6, nq=8, rho_ratio=3.0)
TLO = float(os.environ.get("TLO", 1e-6))
if os.environ.get("QUICK"):
    LOOP.update(nrho=3, nth=3, nP=3, nq=4, rho_ratio=30.0, P_ratio=4.0)
SIG = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
PIECES = {(1, -1): [(1j, "v", "all")], (-1, 1): [(-1j, "u", "all")],
          (1, 1): [(-1j, "u", "12"), (1j, "v", "21")], (-1, -1): [(1j, "v", "12"), (-1j, "u", "21")]}
LINES = {"A": (1, 2), "B": (2, 3), "C": (3, 1)}          # (vertex of Psi, vertex of Psibar)
IN = {1: "C", 2: "A", 3: "B"}                             # line whose Psibar sits at the vertex
OUT = {1: "A", 2: "B", 3: "C"}                            # line whose Psi sits at the vertex


def asym(a, b, n=40):
    c = [1.0 + 0j]
    for j in range(1, n):
        c.append(c[-1] * (a + j - 1) * (b + j - 1) / j)
    return c


class Modes:
    def __init__(self, m, kappa):
        self.m, self.kappa = float(m), float(kappa)
        self.lam = math.sqrt(self.m ** 2 + self.kappa ** 2)
        nu = self.nu = 1j * self.lam
        self.Wd, self.Gp = {}, {}
        for sr, si in product((1, -1), repeat=2):
            K = 0.5 * sr + 1j * si * self.kappa
            B = {s: gamma(-2 * s * nu) / gamma(0.5 - s * nu - K) for s in (1, -1)}
            ph = {s: cmath.exp(-1j * math.pi * (0.5 + s * nu)) for s in (1, -1)}
            cd = asym(0.5 + nu - K, 0.5 - nu - K)

            def small_d(x, B=B, K=K):
                return sum(B[s] * x ** (0.5 + s * nu) * (1 - K / (1 + 2 * s * nu) * x) for s in (1, -1))

            def big_d(x, cd=cd, K=K):
                return x ** K * sum(cn * (-1.0 / x) ** n for n, cn in enumerate(cd))

            def small_g(x, B=B, K=K, ph=ph):
                return sum(B[s] * ph[s] * x ** (0.5 + s * nu) * (1 + K / (1 + 2 * s * nu) * x) for s in (1, -1))

            def big_g(x, cd=cd, K=K):
                return cmath.exp(-1j * math.pi * K) * x ** K * sum(cn * (1.0 / x) ** n for n, cn in enumerate(cd))

            def f_d(x, K=K):
                return math.exp(x / 2) * whitW(K, nu, x)

            def f_g(x, K=K, B=B, ph=ph):
                return math.exp(-x / 2) * sum(B[s] * ph[s] * whitM(-K, s * nu, x) for s in (1, -1))

            key = (round(self.m, 12), round(self.kappa, 12), sr, si)
            self.Wd[(sr, si)] = STable(f_d, small_d, big_d, ("fWd",) + key)
            self.Gp[(sr, si)] = STable(f_g, small_g, big_g, ("fGp",) + key)

    @staticmethod
    def lab(K):
        return (1 if K.real > 0 else -1, 1 if K.imag > 0 else -1)

    def dec(self, K, z):
        """e^(z/2) W_{K,nu}(z)"""
        return self.Wd[self.lab(K)](z)

    def grow(self, K, a, z):
        """e^(-z/2) W_{K,nu}(z e^(-i pi a))"""
        if a == 1:
            return self.Gp[self.lab(K)](z)
        return np.conj(self.Gp[self.lab(K.conjugate())](z))

    # spinor components (c, K, s): s = sign of the argument of W
    def comps(self, kind, h):
        m, k = self.m + 0j, self.kappa
        first = {"L": (m, -0.5 - 1j * k), "R": (1j, 0.5 - 1j * k)}
        second = {"L": (1j, 0.5 + 1j * k), "R": (m, -0.5 + 1j * k)}
        return first if (kind == "u") == (h == 1) else second

    def col(self, kind, h, j):
        c, K = self.comps(kind, h)["L" if j == 0 else "R"]
        return c, K, (1 if kind == "u" else -1)

    def row(self, kind, h, i):
        c, K = self.comps(kind, h)["R" if i == 0 else "L"]
        return c.conjugate(), K.conjugate(), (-1 if kind == "u" else 1)


class Direct:
    def __init__(self, m, kappa, Pc=0.5, nphi=4, twidth=0.3, chunk=60, struct=None):
        self.M = Modes(m, kappa)
        self.kappa = float(kappa)
        self.Pc, self.nphi, self.twidth, self.chunk = Pc, nphi, twidth, chunk
        self.struct = struct                      # optional restriction to some (S1, S2, S3)

    def vertex_monomials(self, a, i, k):
        """block [i][1-i] of Gamma.F at energy k on branch a: [(coefficient, power of t, structure)]"""
        mu = self.kappa if i == 0 else -self.kappa
        sg = -1j if i == 0 else 1j
        return [(mu * 1j * a / (2 * k), 1, "1"), (mu * sg / (2 * k ** 3), 0, "k"), (mu * sg / (2 * k ** 2), 1, "k")]

    def I(self, r, c):
        M = self.M
        st = dict(LOOP)
        st["Pmax"] = self.Pc / r
        y1, y2, w, _ = loop_points(st)
        tg, hstep, npan = tgrid(TLO, 50.0 / r, self.twidth)
        tt = tg.ravel()
        k1 = np.array([math.sqrt(max(1 - c * c, 0.0)), 0.0, c])
        k2 = -(k1 + r * np.array([0.0, 0.0, 1.0]))
        kv = {1: k1, 2: k2, 3: np.array([0.0, 0.0, r])}
        kn = {v: float(np.linalg.norm(kv[v])) for v in kv}
        phis = math.pi * (np.arange(self.nphi) + 0.5) / self.nphi
        hels = (1, -1)

        def cum(F):
            Fp = F.reshape(F.shape[:-1] + (npan, NC + 1))
            loc = (Fp @ Q_N.T) * hstep
            off = np.cumsum(loc[..., -1], axis=-1)
            off = np.concatenate([np.zeros_like(off[..., :1]), off[..., :-1]], -1)
            return (loc + off[..., None]).reshape(F.shape)

        def integ(F):
            return (F.reshape(F.shape[:-1] + (npan, NC + 1)) * W_N).sum((-1, -2)) * hstep

        def sdot(vec):
            return sum(vec[..., i, None, None] * SIG[i + 1] for i in range(3))

        total = 0.0
        for i0 in range(0, y1.size, self.chunk):
            a1_, a2_, ww = y1[i0:i0 + self.chunk], y2[i0:i0 + self.chunk], w[i0:i0 + self.chunk]
            n = a1_.size
            qz = (1 + a1_ ** 2 - a2_ ** 2) / 2
            rho = np.sqrt(np.maximum(a1_ ** 2 - qz ** 2, 0.0))
            q = np.stack([rho[:, None] * np.cos(phis)[None, :], rho[:, None] * np.sin(phis)[None, :],
                          np.broadcast_to(qz[:, None], (n, self.nphi))], -1)                # units of k3
            nC = q / a1_[:, None, None]
            nB = (q - np.array([0.0, 0.0, 1.0])) / a2_[:, None, None]
            P = k1 + r * q
            Y = np.linalg.norm(P, axis=-1)
            nA = P / Y[..., None]
            en = {"A": Y[..., None], "B": (r * a2_)[:, None, None], "C": (r * a1_)[:, None, None]}
            dirs = {"A": nA, "B": nB, "C": nC}
            proj = {L: {h: 0.5 * (SIG[0] + h * sdot(dirs[L])) for h in hels} for L in "ABC"}
            Smat = {v: {"1": np.broadcast_to(SIG[0], (1, 1, 2, 2)), "k": sdot(kv[v])[None, None]} for v in (1, 2, 3)}
            structs = self.struct or list(product("1k", repeat=3))
            Wsp = {}
            for (s1, s2, s3) in structs:
                for hA, hB, hC in product(hels, repeat=3):
                    Mx = Smat[1][s1] @ proj["A"][hA] @ Smat[2][s2] @ proj["B"][hB] @ Smat[3][s3] @ proj["C"][hC]
                    # the imaginary part of the trace is odd in q_y (triple products) and drops out of the
                    # azimuthal integral, which is done over half of the circle
                    Wsp[(s1, s2, s3, hA, hB, hC)] = np.trace(Mx, axis1=-2, axis2=-1).real
            zz = {L: 2 * en[L] * tt for L in "ABC"}
            ftab, vcache = {}, {}

            def fn(L, K, mode, a):
                key = (L, M.lab(K), mode, a)
                if key not in ftab:
                    z = zz[L]
                    ftab[key] = (M.dec(K, z.ravel()) if mode == "d" else M.grow(K, a, z.ravel())).reshape(z.shape) / np.sqrt(z)
                return ftab[key]

            def vf(v, a, kin, kout, hin, hout, S):
                """vertex function: sum over chirality and monomials of structure S"""
                key = (v, a, kin, kout, hin, hout, S)
                if key in vcache:
                    return vcache[key]
                Lin, Lout = IN[v], OUT[v]
                tot, E = 0, None
                for i in (0, 1):
                    cr, Kr, sr = M.row(kin, hin, i)
                    cc, Kc, sc = M.col(kout, hout, 1 - i)
                    mr, mc = ("d" if sr == -a else "g"), ("d" if sc == -a else "g")
                    if E is None:
                        E = -kn[v] + (en[Lin] if mr == "g" else -en[Lin]) + (en[Lout] if mc == "g" else -en[Lout])
                    poly = 0
                    for cf, p, Sm in self.vertex_monomials(a, i, kn[v]):
                        if Sm == S:
                            poly = poly + cf * tt ** p
                    tot = tot + cr * cc * poly * fn(Lin, Kr, mr, a) * fn(Lout, Kc, mc, a)
                out = tot * np.exp(np.clip(E * tt, -700.0, 600.0)) * tt * math.exp(math.pi * (hin + hout) * self.kappa / 2)
                vcache[key] = out          # the factor tt is the Jacobian of ln t
                return out

            def line_piece(L, a, rank):
                f, s = LINES[L]
                pcs = PIECES[(a[f], a[s])]
                if len(pcs) == 1:
                    return pcs[0][0], pcs[0][1]
                reg = "12" if rank[f] < rank[s] else "21"
                for fac, kind, region in pcs:
                    if region == reg:
                        return fac, kind

            def hkey(v, hh):                      # helicities of (in line, out line) at vertex v
                return hh[IN[v]], hh[OUT[v]]

            def ordering(a, order):
                """integral over t_order[0] < t_order[1] < t_order[2] (vertices on other branches unordered)"""
                rank = {v: i for i, v in enumerate(order)}
                fac, kind = 1, {}
                for L in "ABC":
                    f_, k_ = line_piece(L, a, rank)
                    fac *= f_
                    kind[L] = k_
                va, vb, vc = order
                res = 0
                inner, mids = {}, {}
                for hA, hB, hC in product(hels, repeat=3):
                    hh = {"A": hA, "B": hB, "C": hC}
                    for (s1, s2, s3) in structs:
                        S = {1: s1, 2: s2, 3: s3}
                        ka = (hkey(va, hh), S[va])
                        if ka not in inner:
                            inner[ka] = cum(vf(va, a[va], kind[IN[va]], kind[OUT[va]], *hkey(va, hh), S[va]))
                        kb = (hA, hB, hC, S[va], S[vb])
                        if kb not in mids:
                            mids[kb] = cum(vf(vb, a[vb], kind[IN[vb]], kind[OUT[vb]], *hkey(vb, hh), S[vb]) * inner[ka])
                        last = integ(vf(vc, a[vc], kind[IN[vc]], kind[OUT[vc]], *hkey(vc, hh), S[vc]) * mids[kb])
                        res = res + Wsp[(s1, s2, s3, hA, hB, hC)] * last
                return fac * res

            def pair_ordered(a, single, pair):
                """one vertex on its own branch (factorized), the other two ordered"""
                res = 0
                for order2 in (pair, pair[::-1]):
                    rank = {single: 0, order2[0]: 1, order2[1]: 2}        # rank of `single` is irrelevant
                    fac, kind = 1, {}
                    for L in "ABC":
                        f_, k_ = line_piece(L, a, rank)
                        fac *= f_
                        kind[L] = k_
                    va, vb = order2
                    inner, sing = {}, {}
                    for hA, hB, hC in product(hels, repeat=3):
                        hh = {"A": hA, "B": hB, "C": hC}
                        for (s1, s2, s3) in structs:
                            S = {1: s1, 2: s2, 3: s3}
                            ka = (hkey(va, hh), S[va])
                            if ka not in inner:
                                inner[ka] = cum(vf(va, a[va], kind[IN[va]], kind[OUT[va]], *hkey(va, hh), S[va]))
                            ks = (hkey(single, hh), S[single])
                            if ks not in sing:
                                sing[ks] = integ(vf(single, a[single], kind[IN[single]], kind[OUT[single]], *hkey(single, hh), S[single]))
                            last = integ(vf(vb, a[vb], kind[IN[vb]], kind[OUT[vb]], *hkey(vb, hh), S[vb]) * inner[ka])
                            res = res + fac * Wsp[(s1, s2, s3, hA, hB, hC)] * last * sing[ks]
                return res

            acc = np.zeros((n, self.nphi), complex)
            a = {1: 1, 2: 1, 3: 1}
            for order in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)):
                acc += ordering(a, order)
            vcache.clear()
            acc += -1 * pair_ordered({1: 1, 2: -1, 3: 1}, 2, (3, 1))
            vcache.clear()
            acc += -1 * pair_ordered({1: -1, 2: 1, 3: 1}, 1, (2, 3))
            vcache.clear()
            acc += pair_ordered({1: -1, 2: -1, 3: 1}, 3, (1, 2))
            total += np.sum(ww * acc.mean(1)).real
        # a3 = + terms and their conjugates; measure r^3 from q = r q'
        return 2 * r ** 3 * total, int(y1.size)


if __name__ == "__main__":
    m, ka, cs, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
    struct = None
    if os.environ.get("STRUCT"):
        struct = [tuple(s) for s in os.environ["STRUCT"].split(",")]
    D = Direct(m, ka, Pc=float(os.environ.get("PC", 0.5)), nphi=int(os.environ.get("NPHI", 4)), struct=struct)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["r"] for l in open(out)}
    for s in sys.argv[5:]:
        r = float(s)
        if r in done:
            continue
        c = -r / 2 if cs == "iso" else float(cs)
        t0 = time.process_time()
        val, npts = D.I(r, c)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(m=m, kappa=ka, r=r, c=c, I=val, npts=npts, Pc=D.Pc, nphi=D.nphi, grid=os.environ.get("GRID", "std"), tlo=TLO,
                                     struct=os.environ.get("STRUCT", "all"), cpu=round(time.process_time() - t0, 1))) + "\n")
