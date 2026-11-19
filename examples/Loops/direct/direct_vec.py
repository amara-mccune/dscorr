"""Direct time-domain evaluation of the gauge-boson triangle at fixed shape, as a check of the series of
examples/Loops/ext/vectri.py at general shape (Triangle(general=True)).

Nothing is expanded. The three lines are the exact Schwinger-Keldysh functions (mode functions and time grid of
the direct bubble code, numcheck/ninepairs/full.py), the hard line has its exact energy Y = |k1 + q| and its
exact projectors, and the three vertex times are integrated over all orderings. Kinematics: k1 = 1, |k3| = r,
c = k1hat . k3hat, |k2| = (1 + 2 r c + r^2)^(1/2); c = 'iso' is the family k1 = k2 (c = -r/2). The loop momentum
is cut at |q| + |q - k3| < Pc, which changes S only by terms analytic in r. Then

    S(r, c) = (r |k2|/8) I,   I = 2 Re sum_{a1 a2} int d^3q/(2 pi)^3 int dt1 dt2 dt3 e^(-t1 - |k2| t2 - r t3) (b = + at V3),

with the components of ext/vectri.py (longitudinal lines with D = t d_t - 1 and c_0 = -(y/m) tau, the vertex
metric, and the contact term +(t^2/m_h^2) delta(t1 - t2) of the hard line's A_0 A_0 function). The series
predicts

    S(r, c)  contains  2 Re sum_n M_n(c) r^(2 + 2 i mu + n).

At c = +-1 the integrand does not depend on the azimuth of q around k3 (NPHI=1 is exact). Otherwise the azimuth
is integrated over half of the circle with the real part of the polarization contraction (its imaginary part
is odd under the reflection q_y -> -q_y), and NPHI=6 is used.

Usage: direct_vec.py mu kappa c out.jsonl r1 r2 ...     (c = 'iso' or a number)
Environment: NPHI (4), PC (0.5), SUBSET (all | T: transverse components only), CONTACT (1 | 0: without the
contact term), GRID (fine | std), QUICK (coarse loop grid, for timing), TLO (1e-7), PART (i/n: only every n-th
loop point, starting at the i-th, so that a value can be computed in n pieces and summed), NUMCHECK (path of
numcheck/ninepairs from loops_numcheck_code.zip; default ./numcheck/ninepairs next to this file).
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("NUMCHECK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "numcheck", "ninepairs")))
from full import FullLine, tgrid, W_N, NC, Q_N  # noqa: E402
from softmoment import loop_points  # noqa: E402

EPS = np.zeros((3, 3, 3))
for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[i, j, k] = 1
    EPS[j, i, k] = -1
LOOP = dict(rho_min=1e-7, rho_ratio=2.5, nrho=8, nth=8, P_ratio=1.8, nP=8, nq=10, Pmax=None, ntail=1, nfit=1)
if os.environ.get("GRID", "fine") == "std":
    LOOP.update(nrho=6, nth=6, nP=6, nq=8, rho_ratio=3.0)
if os.environ.get("QUICK"):
    LOOP.update(nrho=3, nth=3, nP=3, nq=4, rho_ratio=30.0, P_ratio=4.0)
TLO = float(os.environ.get("TLO", 1e-7))


def projT(h, n):
    return 0.5 * (np.eye(3) - n[..., :, None] * n[..., None, :] - 1j * h * np.einsum("ijl,...l->...ij", EPS, n))


class Direct:
    def __init__(self, mu, kappa, Pc=0.5, nphi=4, subset="all", twidth=0.3, chunk=100, contact=True, part=None):
        self.mu, self.kappa, self.Pc, self.nphi, self.subset, self.twidth, self.chunk = mu, kappa, Pc, nphi, subset, twidth, chunk
        self.contact = contact
        self.part = part                          # (i, n): loop points i, i + n, i + 2n, ...
        self.lines = {1: FullLine(mu, kappa, "T", 1), -1: FullLine(mu, kappa, "T", -1), 0: FullLine(mu, kappa, "L", 0)}
        self.m2 = mu * mu + 0.25

    # ---- scaled end functions of one line on the time grid: {(lam, op, kind): array}, kind 'g' (growing) or 'd'
    def tables(self, z):
        out = {}
        for lam, L in self.lines.items():
            if self.subset == "T" and lam == 0:
                continue
            nf = L.L.Nfac if lam != 0 else 1.0
            out[(lam, 1, "d")] = nf * L.Wd(z)
            out[(lam, 1, "g")] = L.Gg(z)
            if lam == 0:
                out[(lam, "D", "d")] = L.Dd(z)
                out[(lam, "D", "g")] = L.Dg(z)
        return out

    def S(self, r, c=None):
        """S at k1 = 1, k3 = r and c = k1hat . k3hat (default: c = -r/2, the family k1 = k2)."""
        st = dict(LOOP)
        st["Pmax"] = self.Pc / r
        y1, y2, w, _ = loop_points(st)
        if self.part:
            y1, y2, w = y1[self.part[0]::self.part[1]], y2[self.part[0]::self.part[1]], w[self.part[0]::self.part[1]]
        tg, h, npan = tgrid(TLO, 50.0 / r, self.twidth)
        tt = tg.ravel()
        nt = tt.size
        if c is None:
            c = -r / 2
        k1 = np.array([math.sqrt(max(1 - c * c, 0.0)), 0.0, c])
        x2 = math.sqrt(1 + 2 * r * c + r * r)                                                    # |k2|
        phis = math.pi * (np.arange(self.nphi) + 0.5) / self.nphi
        m2, mh2 = self.m2, self.m2

        def cum(F):
            Fp = F.reshape(F.shape[:-1] + (npan, NC + 1))
            loc = (Fp @ Q_N.T) * h
            off = np.cumsum(loc[..., -1], axis=-1)
            off = np.concatenate([np.zeros_like(off[..., :1]), off[..., :-1]], -1)
            return (loc + off[..., None]).reshape(F.shape)

        def integ(F):
            return (F.reshape(F.shape[:-1] + (npan, NC + 1)) * W_N).sum((-1, -2)) * h

        total = 0.0
        for i0 in range(0, y1.size, self.chunk):
            a1, a2, ww = y1[i0:i0 + self.chunk], y2[i0:i0 + self.chunk], w[i0:i0 + self.chunk]
            n = a1.size
            qz = (1 + a1 ** 2 - a2 ** 2) / 2
            rho = np.sqrt(np.maximum(a1 ** 2 - qz ** 2, 0.0))
            q = np.stack([rho[:, None] * np.cos(phis)[None, :], rho[:, None] * np.sin(phis)[None, :],
                          np.broadcast_to(qz[:, None], (n, self.nphi))], -1)                     # (n, nphi, 3), units of k3
            n1 = q / a1[:, None, None]
            n2 = (np.array([0.0, 0.0, 1.0]) - q) / a2[:, None, None]
            P = k1 + r * q
            Y = np.linalg.norm(P, axis=-1)
            Ph = P / Y[..., None]
            eC, eB, eA = (r * a1)[:, None, None], (r * a2)[:, None, None], Y[..., None]         # line energies
            F = {"C": {k: v[:, None, :] for k, v in self.tables(2 * r * a1[:, None] * tt[None, :]).items()},
                 "B": {k: v[:, None, :] for k, v in self.tables(2 * r * a2[:, None] * tt[None, :]).items()},
                 "A": self.tables(2 * Y[..., None] * tt)}
            en = {"C": eC, "B": eB, "A": eA}
            xv = {1: 1.0, 2: x2, 3: r}
            at = {1: ("C", "A"), 2: ("A", "B"), 3: ("C", "B")}                                   # lines at each vertex
            ends = {"C": (3, 1), "B": (3, 2), "A": (1, 2)}
            ecache = {}

            def expw(v, kinds):
                """t e^(E t) at vertex v, E = -x_v + sum over its lines of +-(line energy) for a growing/decaying end."""
                key = (v, kinds)
                if key not in ecache:
                    E = -xv[v] + sum((1 if k == "g" else -1) * en[l] for l, k in zip(at[v], kinds))
                    ecache[key] = np.exp(np.clip(E * tt, -700.0, 600.0)) * tt
                return ecache[key]

            def vert(v, kinds, conj, fl, pw):
                """integrand at vertex v: fl = {line: (lam, op)} for its two lines."""
                out = expw(v, kinds)
                if pw:
                    out = out * tt ** pw
                for l, k in zip(at[v], kinds):
                    lam, op = fl[l]
                    f = F[l][(lam, op, k)]
                    out = out * (np.conj(f) if conj else f)
                return out

            def ordered2(u, v, line, conj, fu, fv, pu, pv, ku_other, kv_other):
                """two vertices joined by one ordered line; the other line at each vertex has a fixed kind."""
                iu = at[u].index(line)
                iv = at[v].index(line)
                tot = 0
                for first, second, i1, i2, f1, f2, p1, p2, ko1, ko2 in ((u, v, iu, iv, fu, fv, pu, pv, ku_other, kv_other),
                                                                         (v, u, iv, iu, fv, fu, pv, pu, kv_other, ku_other)):
                    k1_ = ["", ""]
                    k1_[i1] = "g"
                    k1_[1 - i1] = ko1
                    k2_ = ["", ""]
                    k2_[i2] = "d"
                    k2_[1 - i2] = ko2
                    inner = cum(vert(first, tuple(k1_), conj[first], f1, p1))
                    tot = tot + integ(vert(second, tuple(k2_), conj[second], f2, p2) * inner)
                return tot

            def time_part(lamC, opC, lamB, opB, lamA, opA, pw):
                """{(a1, a2): int dt1 dt2 dt3 (...)} with b = +. opX = (op at first end, op at second end) of `ends`."""
                fl = {1: {"C": (lamC, opC[1]), "A": (lamA, opA[0])}, 2: {"A": (lamA, opA[1]), "B": (lamB, opB[1])},
                      3: {"C": (lamC, opC[0]), "B": (lamB, opB[0])}}
                res = {}
                # (+,+): all six orderings
                tot = 0
                for order in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)):
                    rank = {v: i for i, v in enumerate(order)}
                    fs = []
                    for v in order:
                        kinds = tuple("g" if rank[v] < rank[[e for e in ends[l] if e != v][0]] else "d" for l in at[v])
                        fs.append(vert(v, kinds, False, fl[v], pw[v]))
                    tot = tot + integ(fs[2] * cum(fs[1] * cum(fs[0])))
                res[(1, 1)] = tot
                # (+,-): V2 on the - branch; A and B are cut, C is ordered between V3 and V1
                i2 = integ(vert(2, ("d", "d"), True, fl[2], pw[2]))
                res[(1, -1)] = i2 * ordered2(3, 1, "C", {3: False, 1: False}, fl[3], fl[1], pw[3], pw[1], "d", "d")
                # (-,+): V1 on the - branch; A and C are cut, B is ordered between V3 and V2
                i1 = integ(vert(1, ("d", "d"), True, fl[1], pw[1]))
                res[(-1, 1)] = i1 * ordered2(3, 2, "B", {3: False, 2: False}, fl[3], fl[2], pw[3], pw[2], "d", "d")
                # (-,-): V3 alone on +; A is ordered (conjugated) between V1 and V2
                i3 = integ(vert(3, ("d", "d"), False, fl[3], pw[3]))
                res[(-1, -1)] = i3 * ordered2(1, 2, "A", {1: True, 2: True}, fl[1], fl[2], pw[1], pw[2], "d", "d")
                return res

            def contact_part(lamC, opC, lamB, opB, pw):
                """hard A_0 A_0 contact term: V1 and V2 at one time t with weight e^(-(1 + |k2|) t) t^2; {(1,1), (-1,-1)}."""
                def merged(kC, kB, conj):
                    E = -(1.0 + x2) + (1 if kC == "g" else -1) * eC + (1 if kB == "g" else -1) * eB
                    fC, fB = F["C"][(lamC, opC[1], kC)], F["B"][(lamB, opB[1], kB)]
                    if conj:
                        fC, fB = np.conj(fC), np.conj(fB)
                    return np.exp(np.clip(E * tt, -700.0, 600.0)) * tt * tt ** (2 + pw[1] + pw[2]) * fC * fB
                f3 = {"C": (lamC, opC[0]), "B": (lamB, opB[0])}
                pp = integ(vert(3, ("d", "d"), False, f3, pw[3]) * cum(merged("g", "g", False))) \
                    + integ(merged("d", "d", False) * cum(vert(3, ("g", "g"), False, f3, pw[3])))
                mm = integ(vert(3, ("d", "d"), False, f3, pw[3])) * integ(merged("d", "d", True))
                return {(1, 1): pp, (-1, -1): mm}

            one = np.ones((n, self.nphi, 1, 1))

            def soft_comps(nv, e, s3, sv):
                yy = e[..., 0]
                if s3 == "s" and sv == "s":
                    out = [(h_, projT(h_, nv), (1, 1), (0, 0), lambda a: 1, 1.0) for h_ in (1, -1)]
                    if self.subset != "T":
                        out.append((0, nv[..., :, None] * nv[..., None, :], ("D", "D"), (0, 0), lambda a: 1, 1 / m2))
                    return out
                if self.subset == "T":
                    return []
                if s3 == "s":
                    return [(0, nv[..., :, None], ("D", 1), (0, 1), lambda a: a, yy / m2)]
                if sv == "s":
                    return [(0, nv[..., None, :], (1, "D"), (1, 0), lambda a: -1, yy / m2)]
                return [(0, one, (1, 1), (1, 1), lambda a: -a, yy ** 2 / m2)]

            def hard_comps(s1, s2):
                if s1 == "s" and s2 == "s":
                    out = [(l_, projT(l_, Ph), (1, 1), (0, 0), lambda a, b: 1, 1.0) for l_ in (1, -1)]
                    if self.subset != "T":
                        out.append((0, Ph[..., :, None] * Ph[..., None, :], ("D", "D"), (0, 0), lambda a, b: 1, 1 / mh2))
                    return out
                if self.subset == "T":
                    return []
                if s1 == "s":
                    return [(0, Ph[..., :, None], ("D", 1), (0, 1), lambda a, b: b, Y / mh2)]
                if s2 == "s":
                    return [(0, Ph[..., None, :], (1, "D"), (1, 0), lambda a, b: -a, Y / mh2)]
                out = [(0, one, (1, 1), (1, 1), lambda a, b: -a * b, Y ** 2 / mh2)]
                if self.contact:
                    out.append(("c", one, None, None, None, 1 / mh2))
                return out

            acc = np.zeros((n, self.nphi), complex)
            norm = 1.0 / (8 * eC[..., 0] * eB[..., 0] * Y)
            normc = 1.0 / (4 * eC[..., 0] * eB[..., 0])
            for s3 in "s0":
                for s1 in "s0":
                    for s2 in "s0":
                        vs = (-1) ** ((s3 == "0") + (s1 == "0") + (s2 == "0"))
                        for (lC, tC, oC, pC, sgC, cC) in soft_comps(n1, eC, s3, s1):
                            for (lB, tB, oB, pB, sgB, cB) in soft_comps(n2, eB, s3, s2):
                                X = np.einsum("...ij,...ik->...jk", tC, tB)
                                for (lA, tA, oA, pA, sgA, cA) in hard_comps(s1, s2):
                                    Wt = vs * np.real(np.einsum("...jk,...jk->...", X, tA)) * cC * cB * cA
                                    if lA == "c":
                                        tp = contact_part(lC, oC, lB, oB, {1: pC[1], 2: pB[1], 3: pC[0] + pB[0]})
                                        for (b1, b2), val in tp.items():
                                            acc += Wt * normc * sgC(b1) * sgB(b2) * val
                                        continue
                                    pw = {1: pC[1] + pA[0], 2: pB[1] + pA[1], 3: pC[0] + pB[0]}
                                    tp = time_part(lC, oC, lB, oB, lA, oA, pw)
                                    for (b1, b2), val in tp.items():
                                        acc += Wt * norm * sgC(b1) * sgB(b2) * sgA(b1, b2) * val
            total += np.sum(ww * acc.mean(1)).real
        # S = (k1 k2 k3/8) I, I = 2 Re (...), measure r^3 from q = r q'
        return (r * x2 / 8) * 2 * r ** 3 * total, int(y1.size)


if __name__ == "__main__":
    mu, ka, cs, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
    D = Direct(mu, ka, Pc=float(os.environ.get("PC", 0.5)), nphi=int(os.environ.get("NPHI", 4)), subset=os.environ.get("SUBSET", "all"),
               contact=os.environ.get("CONTACT", "1") != "0",
               part=tuple(int(x) for x in os.environ["PART"].split("/")) if os.environ.get("PART") else None)
    part = os.environ.get("PART")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["r"] for l in open(out) if json.loads(l).get("part") == part}
    for s in sys.argv[5:]:
        r = float(s)
        if r in done:
            continue
        c = -r / 2 if cs == "iso" else float(cs)
        t0 = time.process_time()
        val, npts = D.S(r, c)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(mu=mu, kappa=ka, r=r, c=c, shape=cs, S=val, npts=npts, Pc=D.Pc, nphi=D.nphi, subset=D.subset,
                                     contact=D.contact, grid=os.environ.get("GRID", "fine"), quick=bool(os.environ.get("QUICK")), tlo=TLO, part=part,
                                     cpu=round(time.process_time() - t0, 1))) + "\n")
