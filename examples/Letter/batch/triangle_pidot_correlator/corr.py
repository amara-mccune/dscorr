"""One-loop triangle correlator with pi-dot legs by brute force.

S = k1 k2 k3 / 8 int d^3l/(2pi)^3 [-2 Re U - 2 Re(V2 + V0 + V1)], the four structures being
triple time integrals (r_v = 2) with Feynman (F) or cut (X) kernels on the lines.  Reuses the
weighted cumulative operator and time-grid machinery of ../triangle (must stay beside it).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "triangle"))
from bf_triangle import BesselImag, LINES, ORDERINGS, TimeGrid, cum_weights, weighted_cumulative  # noqa: E402

STRUCTS = {"U": "FFF", "V2": "FXX", "V0": "XFX", "V1": "XXF"}
NAMES = ("U", "V2", "V0", "V1")


class SplineBessel:
    """K^ and I^ (x/2)^(-i mu) by cubic splines in ln x on [1e-13, 30], 420 points per decade."""

    XLO, XHI = 1e-13, 30.0

    def __init__(self, mu):
        self.mu = float(mu)
        self.direct = BesselImag(mu)
        n = int(round(420 * math.log10(self.XHI / self.XLO))) + 1
        x = np.geomspace(self.XLO, self.XHI, n)
        L = np.log(x)
        K = self.direct.K(x)
        I = self.direct.I(x) * np.exp(-1j * self.mu * np.log(x / 2.0))
        self.sK = CubicSpline(L, K)
        self.sIr = CubicSpline(L, I.real)
        self.sIi = CubicSpline(L, I.imag)

    def K(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        hi = x > self.XHI
        lo = ~hi
        xl = np.maximum(x[lo], self.XLO)
        out[lo] = self.sK(np.log(xl))
        if hi.any():
            out[hi] = self.direct.K(x[hi])
        return out

    def I(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape, dtype=complex)
        hi = x > self.XHI
        lo = ~hi
        xl = np.maximum(x[lo], self.XLO)
        L = np.log(xl)
        out[lo] = (self.sIr(L) + 1j * self.sIi(L)) * np.exp(1j * self.mu * (L - math.log(2.0)))
        if hi.any():
            out[hi] = self.direct.I(x[hi])
        return out


def time_grid(k, tpan_base, tngl, tmin=1e-9):
    tmax = 50.0 / min(k)
    npan = int(math.ceil(math.log(tmax / tmin) * tpan_base / math.log(6e10)))
    return TimeGrid(tmin, tmax, npan, tngl)


class Structs:
    """The four fixed-loop-momentum structures U, V2, V0, V1 (r_v = 2), batched."""

    def __init__(self, mu, tg, rv=2.0):
        self.mu = float(mu)
        self.bes = SplineBessel(mu)
        self.tg = tg
        self.rv = rv
        self.c = math.exp(-math.pi * self.mu)

    def __call__(self, x, yA, yB, yC, block=320, names=NAMES):
        yA, yB, yC = (np.atleast_1d(np.asarray(v, float)) for v in (yA, yB, yC))
        out = {n: np.zeros(yA.size, dtype=complex) for n in names}
        for i0 in range(0, yA.size, block):
            sl = slice(i0, i0 + block)
            r = self._block(x, yA[sl], yB[sl], yC[sl], names)
            for n in names:
                out[n][sl] = r[n]
        return out

    def _block(self, x, yA, yB, yC, names):
        tg = self.tg
        t = tg.t
        ys = (yA, yB, yC)
        fac = []
        for l in range(3):
            z = ys[l][:, None, None] * t[None]
            K = self.bes.K(z)
            I = self.bes.I(z)
            e2 = np.exp(-2.0 * z)
            fac.append({"K": K, "I": I, "F": I - (1j / math.pi) * self.c * e2 * K,
                        "X": (1.0 / math.pi) * e2 * K})
        ex = [(t ** (self.rv - 1.0) * np.exp(-float(x[v]) * t))[None] for v in range(3)]
        kv = {v: sum(ys[l] for l, (a, b) in enumerate(LINES) if v in (a, b)) for v in range(3)}
        Wc = {v: cum_weights(kv[v], tg) for v in range(3)}
        res = {}
        for name in names:
            kern = STRUCTS[name]
            tot = np.zeros(yA.size, dtype=complex)
            for order in ORDERINGS:
                rank = {v: i for i, v in enumerate(order)}
                g, kap = [], []
                for v in order:
                    f = ex[v]
                    kk = 0.0
                    for l, (a, b) in enumerate(LINES):
                        if v not in (a, b):
                            continue
                        other = b if v == a else a
                        f = f * (fac[l][kern[l]] if rank[other] > rank[v] else fac[l]["K"])
                        kk = kk + ys[l]
                    g.append(f)
                    kap.append(kk)
                J1 = weighted_cumulative(g[0], kap[0], tg, Wc[order[0]])
                J2 = weighted_cumulative(g[1] * J1, kap[2], tg, Wc[order[2]])
                tot += np.einsum("npg,pg->n", g[2] * J2, tg.w)
            res[name] = tot
        return res


# ============================================================================ loop grids

LEVELS = {
    "Q0": dict(lmin=1e-6, LMAX=1024.0, ppo_r=2.5, nr=5, nct=(5, 5), nph=6, tpan_base=80, tngl=8),
    "Q1": dict(lmin=1e-6, LMAX=1024.0, ppo_r=3.5, nr=6, nct=(8, 7), nph=10, tpan_base=80, tngl=8),
    "Q2": dict(lmin=3e-7, LMAX=2048.0, ppo_r=4.5, nr=7, nct=(11, 10), nph=14, tpan_base=110, tngl=10),
    "Q3": dict(lmin=1e-7, LMAX=4096.0, ppo_r=5.5, nr=8, nct=(14, 12), nph=18, tpan_base=110, tngl=10),
}
G2 = dict(lmin=1e-5, ppo_r=3.0, nr=5, J=8, ct_edges=(0, 0.03, 0.15, 0.5, 1.0),
          nct_pan=(6, 5, 5, 5), nph=5, nph_base=10, r_coarse=12.0, LMAX=1024.0, nct_out=8,
          nph_out=12, tpan_base=80, tngl=8)
CT_EDGES = (0.0, 0.3, 1.0)
NPOU = 4


def kvectors(k):
    k1, k2, k3 = k
    c = (k3 ** 2 - k1 ** 2 - k2 ** 2) / (2 * k1 * k2)
    c = min(1.0, max(-1.0, c))
    K1 = np.array([k1, 0.0, 0.0])
    K2 = k2 * np.array([c, math.sqrt(1 - c * c), 0.0])
    K3 = -(K1 + K2)
    return K1, K2, K3


def _merge(edges, rel):
    e = sorted(edges)
    out = [e[0]]
    for v in e[1:]:
        if v - out[-1] > rel * max(abs(v), 1e-300):
            out.append(v)
    return out


def _gl_panels(edges, n):
    x, w = np.polynomial.legendre.leggauss(n)
    e = np.asarray(edges, float)
    a, b = e[:-1], e[1:]
    nodes = (0.5 * (a + b))[:, None] + (0.5 * (b - a))[:, None] * x[None, :]
    wts = (0.5 * (b - a))[:, None] * w[None, :]
    return nodes, wts


def pou_nodes(k, lv):
    """Nodes of the partition-of-unity grid: list per centre of (l vectors, weights, panel ids)."""
    K1, K2, K3 = kvectors(k)
    cents = {"C": np.zeros(3), "A": -K1, "B": K3}
    names = ("C", "A", "B")
    out = []
    redges = {}
    for s in names:
        ps = cents[s]
        others = [cents[o] for o in names if o != s]
        dists = [float(np.linalg.norm(o - ps)) for o in others]
        nlog = int(lv["ppo_r"] * math.log10(lv["LMAX"] / lv["lmin"]))
        e = [lv["lmin"] * 10 ** (j / lv["ppo_r"]) for j in range(nlog + 1)]
        e[-1] = lv["LMAX"]                                       # last edge set to LMAX
        for d in dists:
            for f in (0.5, 0.75, 1.0, 1.5, 2.0):
                if lv["lmin"] < f * d < lv["LMAX"]:
                    e.append(f * d)
        e = _merge(e, 1e-6)
        redges[s] = e
        rn, rw = _gl_panels(e, lv["nr"])                       # (npan, nr)
        cts, cws = [], []
        for (a, b), n in zip(zip(CT_EDGES[:-1], CT_EDGES[1:]), lv["nct"]):
            x, w = _gl_panels([a, b], n)
            cts.append(x.ravel())
            cws.append(w.ravel())
        ct = np.concatenate(cts)
        cw = 2.0 * np.concatenate(cws)                          # mirror half l_z -> -l_z
        az = [0.0, 2 * math.pi]
        for o in others:
            dv = o - ps
            phi = math.atan2(dv[1], dv[0]) % (2 * math.pi)
            az += [phi, (phi + math.pi) % (2 * math.pi)]
        az = _merge(az, 1e-6)
        if az[-1] < 2 * math.pi - 1e-12:
            az.append(2 * math.pi)
        while len(az) - 1 < 4:
            gaps = np.diff(az)
            i = int(np.argmax(gaps))
            az.insert(i + 1, 0.5 * (az[i] + az[i + 1]))
        ph, pw = _gl_panels(az, lv["nph"])
        ph, pw = ph.ravel(), pw.ravel()
        st = np.sqrt(1 - ct ** 2)
        dirs = np.stack([st[:, None] * np.cos(ph)[None, :], st[:, None] * np.sin(ph)[None, :],
                         np.repeat(ct[:, None], ph.size, axis=1)], axis=-1).reshape(-1, 3)
        dw = (cw[:, None] * pw[None, :]).ravel()
        npan = rn.shape[0]
        r = rn.ravel()
        wr = (rw * rn ** 2).ravel()
        L = ps[None, None, :] + r[:, None, None] * dirs[None, :, :]       # (nrad, ndir, 3)
        L = L.reshape(-1, 3)
        W = (wr[:, None] * dw[None, :]).ravel()
        panel = np.repeat(np.repeat(np.arange(npan), lv["nr"]), dirs.shape[0])
        # partition of unity
        ds = {o: np.linalg.norm(L - cents[o][None, :], axis=1) for o in names}
        den = np.ones(L.shape[0])
        for o in names:
            if o != s:
                den = den + (ds[s] / ds[o]) ** NPOU
        W = W / den
        out.append((s, L, W, panel))
    return out, redges


def line_energies(L, k):
    K1, K2, K3 = kvectors(k)
    yC = np.linalg.norm(L, axis=1)
    yA = np.linalg.norm(L + K1[None, :], axis=1)
    yB = np.linalg.norm(L - K3[None, :], axis=1)
    return yA, yB, yC


def _eval(structs, k, L, block=320):
    yA, yB, yC = line_energies(L, k)
    return structs(k, yA, yB, yC, block=block)


def run_point_pou(mus, k, grid, verbose=False):
    """Raw panel sums of the four structures on the partition-of-unity grid (one process)."""
    t0 = time.time()
    lv = LEVELS[grid]
    mu = float(mus[0])
    tg = time_grid(k, lv["tpan_base"], lv["tngl"])
    st = Structs(mu, tg)
    cents, redges = pou_nodes(k, lv)
    # all nodes sorted by yA + yB + yC (fixed order of summation)
    allL = np.concatenate([c[1] for c in cents])
    allW = np.concatenate([c[2] for c in cents])
    cid = np.concatenate([np.full(len(c[2]), i) for i, c in enumerate(cents)])
    pid = np.concatenate([c[3] for c in cents])
    yA, yB, yC = line_energies(allL, k)
    order = np.argsort(yA + yB + yC, kind="stable")
    vals = st(k, yA[order], yB[order], yC[order])
    panels = {}
    for i, (s, L, W, panel) in enumerate(cents):
        npan = len(redges[s]) - 1
        re = np.zeros((npan, 4))
        im = np.zeros((npan, 4))
        sel = cid[order] == i
        pp = pid[order][sel]
        ww = allW[order][sel]
        for j, n in enumerate(NAMES):
            v = ww * vals[n][sel]
            re[:, j] = np.bincount(pp, weights=v.real, minlength=npan)
            im[:, j] = np.bincount(pp, weights=v.imag, minlength=npan)
        panels[s] = {"re": re.tolist(), "im": im.tolist()}
    return {"mus": list(mus), "k": list(k), "grid": grid,
            "gspec": [tg.tmin, tg.tmax, tg.npan, tg.ngl], "n_nodes": int(allW.size),
            "wall_s": time.time() - t0, "names": list(NAMES), "centres": ["C", "A", "B"],
            "r_edges": {s: list(map(float, redges[s])) for s in redges}, "panels": panels}


def norm_factor(k):
    return k[0] * k[1] * k[2] / (8.0 * (2 * math.pi) ** 3)


def assemble_pou(raw):
    """S, S_cut, tail_rel, tail_power, struct from a raw file."""
    k = raw["k"]
    names = raw["names"]
    Icut = 0.0
    struct = np.zeros(4)
    for s in raw["centres"]:
        P = np.array(raw["panels"][s]["re"])
        struct += P.sum(axis=0)
        Icut += float(np.sum(-2 * P[:, 0] - 2 * (P[:, 1] + P[:, 2] + P[:, 3])))
    # tail
    kmax = max(k)
    far = [np.array(raw["r_edges"][s]) for s in raw["centres"]]
    tail = 0.0
    p = None
    ok = all(abs(f[-1] - far[0][-1]) <= 1e-6 * far[0][-1] for f in far)
    shells = {}
    if ok:
        for s, e in zip(raw["centres"], far):
            P = np.array(raw["panels"][s]["re"])
            b = -2 * P[:, 0] - 2 * (P[:, 1] + P[:, 2] + P[:, 3])
            for j in range(len(e) - 1):
                if e[j] >= 16 * kmax:
                    key = round(e[j], 9)
                    shells.setdefault(key, [e[j], e[j + 1], 0.0])
                    shells[key][2] += b[j]
    LMAX = far[0][-1]
    if ok and len(shells) >= 3:
        sh = sorted(shells.values())
        r = np.array([math.sqrt(a * b) for a, b, _ in sh])
        y = np.array([v / math.log(b / a) for a, b, v in sh])
        sgn = np.sign(y[-1])
        keep = np.sign(y) == sgn
        A = np.column_stack([np.ones(keep.sum()), -np.log(r[keep])])
        (a, p), *_ = np.linalg.lstsq(A, np.log(np.abs(y[keep])), rcond=None)
        tail = sgn * math.exp(a) * LMAX ** (-p) / p if p > 0 else float("nan")
    N = norm_factor(k)
    Scut = N * Icut
    S = N * (Icut + tail)
    return {"S": S, "S_cut": Scut, "tail_rel": (S - Scut) / S if S else float("nan"),
            "tail_power": p, "struct": dict(zip(names, struct.tolist())),
            "n_nodes": raw["n_nodes"], "gspec": raw["gspec"]}


# ============================================================================ level G2


def g2_nodes(k, lv=G2):
    """Single clustered spherical grid about l = 0."""
    K1, K2, K3 = kvectors(k)
    k1, k2, k3 = k
    nlog = int(lv["ppo_r"] * math.log10(lv["LMAX"] / lv["lmin"]))
    e = [lv["lmin"] * 10 ** (j / lv["ppo_r"]) for j in range(nlog + 1)]
    e[-1] = lv["LMAX"]
    for c in (k1, k3):
        for j in range(lv["J"] + 1):
            e += [c - 0.5 * c * 2.0 ** -j, c + 0.5 * c * 2.0 ** -j]
    e += [k1, k2, k3, k1 + k2, k1 + k3, k2 + k3]
    e = [v for v in e if lv["lmin"] <= v <= lv["LMAX"]]
    e = _merge(e, 1e-7)
    rn, rw = _gl_panels(e, lv["nr"])
    phi3 = math.atan2(K3[1], K3[0]) % (2 * math.pi)
    dph = 2 * math.pi / lv["nph_base"]
    az = list(dph * np.arange(lv["nph_base"] + 1))
    for c in (math.pi, phi3):
        az.append(c)
        for j in range(lv["J"] + 1):
            az += [c - dph * 2.0 ** -j, c + dph * 2.0 ** -j]
    az = _merge([a for a in az if 0 <= a <= 2 * math.pi], 1e-7)
    Ls, Ws, pid = [], [], []
    for p in range(rn.shape[0]):
        inner = e[p] < lv["r_coarse"]
        if inner:
            ct_e, ct_n, azp, nph = lv["ct_edges"], lv["nct_pan"], az, lv["nph"]
        else:
            ct_e, ct_n, azp, nph = (0.0, 0.3, 1.0), (lv["nct_out"], lv["nct_out"]), \
                list(np.linspace(0, 2 * math.pi, 5)), lv["nph_out"]
        cts, cws = [], []
        for (a, b), n in zip(zip(ct_e[:-1], ct_e[1:]), ct_n):
            x, w = _gl_panels([a, b], n)
            cts.append(x.ravel())
            cws.append(w.ravel())
        ct = np.concatenate(cts)
        cw = 2.0 * np.concatenate(cws)
        ph, pw = _gl_panels(azp, nph)
        ph, pw = ph.ravel(), pw.ravel()
        st = np.sqrt(1 - ct ** 2)
        dirs = np.stack([st[:, None] * np.cos(ph)[None, :], st[:, None] * np.sin(ph)[None, :],
                         np.repeat(ct[:, None], ph.size, axis=1)], axis=-1).reshape(-1, 3)
        dw = (cw[:, None] * pw[None, :]).ravel()
        L = (rn[p][:, None, None] * dirs[None, :, :]).reshape(-1, 3)
        W = ((rw[p] * rn[p] ** 2)[:, None] * dw[None, :]).ravel()
        Ls.append(L)
        Ws.append(W)
        pid.append(np.full(W.size, p))
    return np.concatenate(Ls), np.concatenate(Ws), np.concatenate(pid), e


def run_point(mus, k, grid="G2"):
    t0 = time.time()
    lv = G2
    tg = time_grid(k, lv["tpan_base"], lv["tngl"])
    st = Structs(float(mus[0]), tg)
    L, W, pid, e = g2_nodes(k, lv)
    yA, yB, yC = line_energies(L, k)
    order = np.argsort(yC, kind="stable")
    vals = st(k, yA[order], yB[order], yC[order])
    npan = len(e) - 1
    re = np.zeros((npan, 4))
    im = np.zeros((npan, 4))
    for j, n in enumerate(NAMES):
        v = W[order] * vals[n]
        re[:, j] = np.bincount(pid[order], weights=v.real, minlength=npan)
        im[:, j] = np.bincount(pid[order], weights=v.imag, minlength=npan)
    return {"mus": list(mus), "k": list(k), "grid": grid, "gspec": [tg.tmin, tg.tmax, tg.npan, tg.ngl],
            "n_nodes": int(W.size), "wall_s": time.time() - t0, "names": list(NAMES),
            "r_edges": list(map(float, e)), "panel_re": re.tolist(), "panel_im": im.tolist()}


def assemble(raw):
    """Assembly of a G2 raw file: the same as assemble_pou with a single set of panels."""
    r = dict(raw)
    r["centres"] = ["G"]
    r["r_edges"] = {"G": raw["r_edges"]}
    r["panels"] = {"G": {"re": raw["panel_re"], "im": raw["panel_im"]}}
    return assemble_pou(r)
