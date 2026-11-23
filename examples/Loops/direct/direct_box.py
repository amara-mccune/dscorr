"""Direct time-domain evaluation of the box of the helical-vector loop (the diagram with a hard pair on both sides
of the collapsed trispectrum, Sec. 8.4 of the paper and ext/boxseries.py), at fixed external momenta.

Nothing is expanded. All four lines are the exact Schwinger-Keldysh functions (mode tables of ninepairs/full.py, as in
direct_vec.py), both hard lines have their exact energies and projectors and their A_0 A_0 contact terms, the four
vertex times are integrated over all orderings, and all sixteen branch assignments are summed.

Kinematics (those of ext/boxseries.py, units k1 = 1): s = k1 + k2 = -(k3 + k4) along z, |s| = r,
k1hat = (S1, 0, c1), k3hat = (S3 cos beta, S3 sin beta, c3). Soft line 1 joins V1 (leg k1) to V3 (leg k3) and its
operator at V1 has momentum l. Soft line 2 joins V2 to V4 with w = s - l at V2. The hard line A runs from V1 to V2
with P = k1 - l at V1, and A' from V3 to V4 with P' = k3 + l at V3 (planar attachment). The crossed box is the planar
one with k3 <-> k4, which at equal external momenta is beta -> beta + pi.

    I_box(beta) = int_{|l| + |s - l| < Pc} d^3l/(2 pi)^3 sum_{branches} int dt1 dt2 dt3 dt4 e^(-sum_v k_v t_v)
                  (product of the four line functions, contracted with eta at each vertex),

the loop integral with unit vertices in the normalization of parity_box_series.py (class PP). The cut changes it only
by terms analytic in the vector s. I_box is complex: its real part is even in beta and its imaginary part is odd (the
parity-odd trispectrum), and the harmonics I_n of I_box = sum_n I_n e^(i n beta) are real.

Method. A line with first end e1 and second end e2 (operator momentum n at e1) has three modes a = +, -, L, with
polarization vectors eps^h(n) at e1 and their conjugates at e2 (eps^h eps^h* = (1 - n n - i h eps.n)/2), and n for L.
The mode L also has a temporal component, -+ b (y/m) t g at the first/second end on branch b. At each vertex the two
line ends are contracted with eta, which gives a 3 x 3 matrix in the mode labels at each time. A fixed time ordering of
the vertices is a nested integral of products of these matrices, done with cumulative Chebyshev quadrature on panels
in ln t. The loop azimuth and beta are on the same uniform grid of M points, so that the two sides are tabulated M
times each and combined M^2 times. The numba kernel `box_kernel` does the 4-cycle. `cycle_sum` is a slow reference
for any cycle of vertices (all permutations, all branches); it serves as a test of the kernel, evaluates the pieces
with a contact term (cycles of three and two vertices), and gives the contact-pair diagram for a test against
direct_vec.py.

Large loop momenta. Above |l| + |s - l| = 2.5 k the late panels have a fixed width in t (`tgrid_uv`): a panel in ln t
across which e^(E t) changes by many orders of magnitude spoils the nested integrals. Near the ring |l| = |s - l| = k
the hard lines have zero momentum (at the azimuth of k1 and opposite to that of k3), the integrand is bounded but not
smooth, and the uniform azimuth grid converges slowly (`ring_grid` grades the other two coordinates).

Needs numpy, numba, and python-flint (for the mode tables of ninepairs/).

Usage: direct_box.py mu kappa out.jsonl r1 r2 ...
Environment: PC (0.5), M (8), GRID (std | fine | coarse, or a JSON dict of the loop grid), PART (i/n), TW (late panel width, 0.45), TLO (1e-5),
NUMCHECK (path of ninepairs; default ./ninepairs next to this file), CONTACT (1 | 0), PMIN (only the shell
PMIN < |l| + |s - l| < PC), RING (with PMIN: panels graded towards the ring where the hard lines have zero momentum).
"""
import itertools
import json
import math
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("NUMCHECK") or os.path.join(HERE, "ninepairs"))
from full import FullLine, cheb_cumulative  # noqa: E402
from softmoment import loop_points  # noqa: E402

NC = 12
S_N, W_N, Q_N = cheb_cumulative(NC)
KINDS = (("g", "g"), ("g", "d"), ("d", "g"), ("d", "d"))
KK = {k: i for i, k in enumerate(KINDS)}
GRIDS = {"std": dict(rho_min=1e-4, rho_ratio=3.0, nrho=6, nth=6, P_ratio=1.8, nP=6, nq=8),
         "fine": dict(rho_min=1e-5, rho_ratio=2.5, nrho=8, nth=8, P_ratio=1.6, nP=8, nq=10),
         "coarse": dict(rho_min=1e-3, rho_ratio=4.0, nrho=4, nth=4, P_ratio=2.5, nP=4, nq=6)}


# ------------------------------------------------------------------------------------------------ grids
def tgrid2(tlo=1e-5, tsplit=5e-3, thi=40.0, w_early=1.5, w_late=0.45):
    """Chebyshev panels in u = ln t, wide below tsplit (powers of t) and narrow above. Returns t (N), the half
    widths h (npan) and the weights wint of int du. The Jacobian t of dt = t du is part of the vertex tensors."""
    e1 = np.linspace(math.log(tlo), math.log(tsplit), max(1, int(math.ceil(math.log(tsplit / tlo) / w_early))) + 1)
    e2 = np.linspace(math.log(tsplit), math.log(thi), max(1, int(math.ceil(math.log(thi / tsplit) / w_late))) + 1)
    edges = np.concatenate([e1, e2[1:]])
    h = np.diff(edges) / 2
    t = np.exp(edges[:-1, None] + h[:, None] * (S_N[None, :] + 1)).ravel()
    wint = (h[:, None] * W_N[None, :]).ravel()
    return t, h, wint


def tgrid_uv(emax, thi, tlo=1e-5, tsplit=5e-3, w_early=1.5, w_late=0.45, dlin=6.0):
    """Time grid for large line energies: panels of width w_late in ln t up to the time where their width in t reaches
    dlin/emax, and of that width in t from there to thi. A panel across which e^(emax t) changes by many orders of
    magnitude spoils the nested integrals, so the grid of `tgrid2` only serves loop momenta of order k."""
    t1 = dlin / (emax * (math.exp(w_late) - 1))
    if t1 >= thi:
        return tgrid2(tlo, tsplit, thi, w_early, w_late)
    tsplit = min(tsplit, t1 / 4)
    e1 = np.linspace(math.log(tlo), math.log(tsplit), max(1, int(math.ceil(math.log(tsplit / tlo) / w_early))) + 1)
    e2 = np.linspace(math.log(tsplit), math.log(t1), max(1, int(math.ceil(math.log(t1 / tsplit) / w_late))) + 1)
    e3 = np.log(np.linspace(t1, thi, max(1, int(math.ceil((thi - t1) * emax / dlin))) + 1))
    edges = np.concatenate([e1, e2[1:], e3[1:]])
    h = np.diff(edges) / 2
    t = np.exp(edges[:-1, None] + h[:, None] * (S_N[None, :] + 1)).ravel()
    return t, h, (h[:, None] * W_N[None, :]).ravel()


def cum(F, h):
    """int_0^t F on the panels (F holds the integrand times the Jacobian t), along the last axis"""
    Fp = F.reshape(F.shape[:-1] + (h.size, NC + 1))
    loc = (Fp @ Q_N.T) * h[:, None]
    off = np.cumsum(loc[..., -1], axis=-1)
    off = np.concatenate([np.zeros_like(off[..., :1]), off[..., :-1]], -1)
    return (loc + off[..., None]).reshape(F.shape)


def loop_grid(r, Pc, st, Pmin=None):
    """Points (y1, y2) = (|l|, |s - l|)/s and weights of int d^3l/(2 pi)^3 (in units of s^3, azimuth averaged) for
    y1 + y2 < Pc/r. The outer panels have edges fixed in units of k, so that their quadrature error is analytic in r.
    With Pmin (in units of k, at least 2 r) only the shell Pmin < |l| + |s - l| < Pc is returned."""
    if Pmin is None:
        base = dict(st, Pmax=2.0, ntail=1, nfit=1)
        y1, y2, w, _ = loop_points(base)
        keep = w != 0
        y1, y2, w = [y1[keep]], [y2[keep]], [w[keep]]
    else:
        y1, y2, w = [], [], []
    Pmax = Pc / r
    if Pmax > 2.0:
        edges = [Pc]
        lowest = 2.0 * r * 1.25 if Pmin is None else Pmin * 1.25
        while edges[-1] / st["P_ratio"] > lowest:
            edges.append(edges[-1] / st["P_ratio"])
        edges = [2.0 if Pmin is None else Pmin / r] + [e / r for e in edges[::-1]]
        xq, wq = np.polynomial.legendre.leggauss(st["nq"])
        tp, wp = np.polynomial.legendre.leggauss(st["nP"])
        tp, wp = 0.5 * (tp + 1), 0.5 * wp
        pre = 1.0 / (4 * math.pi ** 2)
        for lo, hi in zip(edges[:-1], edges[1:]):
            lr = math.log(hi / lo)
            P = lo * np.exp(lr * tp)
            PP, QQ = np.meshgrid(P, xq, indexing="ij")
            y1.append(((PP + QQ) / 2).ravel())
            y2.append(((PP - QQ) / 2).ravel())
            w.append((pre * (PP ** 2 - QQ ** 2) / 8 * np.outer(lr * wp * P, wq)).ravel())
    return np.concatenate(y1), np.concatenate(y2), np.concatenate(w)


def ring_grid(r, Plo, Phi, nP=5, nq=5, Pring=2.0):
    """Shell Plo < |l| + |s - l| < Phi (units of k) with panels graded towards the ring |l| + |s - l| = Pring,
    |l| = |s - l|, on which the two hard lines of the box have zero momentum at equal external momenta (at the azimuth
    of k1 and opposite to that of k3). There the integrand is bounded but not smooth."""
    offs = (-0.9, -0.6, -0.4, -0.2, -0.08, 0.0, 0.08, 0.2, 0.4, 0.6)
    edges = [Plo] + [Pring + o for o in offs if Plo + 0.04 < Pring + o < Phi - 0.04] + [Phi]
    qe = (-1.0, -0.35, -0.1, 0.0, 0.1, 0.35, 1.0)
    x, wx = np.polynomial.legendre.leggauss(nP)
    xq, wq = np.polynomial.legendre.leggauss(nq)
    pre = 1.0 / (4 * math.pi ** 2)
    y1, y2, w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        P = (0.5 * (lo + hi) + 0.5 * (hi - lo) * x) / r
        wP = 0.5 * (hi - lo) * wx / r
        for qa, qb in zip(qe[:-1], qe[1:]):
            q = 0.5 * (qa + qb) + 0.5 * (qb - qa) * xq
            PP, QQ = np.meshgrid(P, q, indexing="ij")
            y1.append(((PP + QQ) / 2).ravel())
            y2.append(((PP - QQ) / 2).ravel())
            w.append((pre * (PP ** 2 - QQ ** 2) / 8 * np.outer(wP, 0.5 * (qb - qa) * wq)).ravel())
    return np.concatenate(y1), np.concatenate(y2), np.concatenate(w)


# ------------------------------------------------------------------------------------------------ lines
class Modes:
    """End functions of the three modes of a line of mass mu and chemical potential kappa (tables of full.py)."""

    def __init__(self, mu, kappa):
        self.mu, self.kappa = mu, kappa
        self.m = math.sqrt(mu * mu + 0.25)
        self.lines = {1: FullLine(mu, kappa, "T", 1), -1: FullLine(mu, kappa, "T", -1), 0: FullLine(mu, kappa, "L", 0)}

    def ends(self, y, t):
        """{kind: (fS[3, ...], fT0[...])}: the spatial-component functions of the modes (+, -, L) and the temporal
        function (y/m) t g of the mode L without its sign, at line energy y (array broadcast against t)."""
        z = 2 * y * t
        Lp, Lm, L0 = self.lines[1], self.lines[-1], self.lines[0]
        out = {}
        for k in "gd":
            if k == "d":
                fs = [Lp.L.Nfac * Lp.Wd(z), Lm.L.Nfac * Lm.Wd(z), L0.Dd(z) / self.m]
                f0 = L0.Wd(z)
            else:
                fs = [Lp.Gg(z), Lm.Gg(z), L0.Dg(z) / self.m]
                f0 = L0.Gg(z)
            out[k] = (np.stack(fs), (y / self.m) * t * f0)
        return out


def pol_vectors(n):
    """U[..., a, 3] at the first end of a line with operator momentum direction n[..., 3]: (e1 + i h e2)/sqrt2 for
    a = +, -, and n for a = L, with (e1, e2, n) the right-handed spherical basis (covariant under rotations about z)."""
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    st = np.sqrt(nx * nx + ny * ny)
    st = np.where(st > 0, st, 1.0)
    cph, sph = np.where(st > 0, nx / st, 1.0), np.where(st > 0, ny / st, 0.0)
    e1 = np.stack([nz * cph, nz * sph, -np.sqrt(nx * nx + ny * ny)], -1)
    e2 = np.stack([-sph, cph, np.zeros_like(nx)], -1)
    U = np.stack([(e1 + 1j * e2) / math.sqrt(2), (e1 - 1j * e2) / math.sqrt(2), n.astype(complex)], -2)
    return U


def vertex_tensor(t, x, pw, yp, endp, Up, firstp, yn, endn, Un, firstn, norm=1.0):
    """T[br, kk, a, a', ..., N] of one vertex: energy x, extra power pw of the time, 'prev' line (energy yp, end
    functions endp from Modes.ends, polarization vectors Up[..., a, 3] of its first end, firstp = whether this vertex
    is its first end) and 'next' line likewise. br = 0, 1 for the + and - branch, kk indexes KINDS (prev, next)."""
    gp = Up if firstp else np.conj(Up)
    gn = Un if firstn else np.conj(Un)
    g = np.einsum("...ai,...bi->ab...", gp, gn)[..., None]           # (3, 3, ..., 1)
    sig = (-1.0 if firstp else 1.0) * (-1.0 if firstn else 1.0)
    shape = np.broadcast(g[0, 0], yp * t, yn * t).shape
    out = np.empty((2, 4, 3, 3) + shape, complex)
    for (kp, kn), kk in KK.items():
        E = -x + (1 if kp == "g" else -1) * yp + (1 if kn == "g" else -1) * yn
        wexp = norm * t ** (1 + pw) * np.exp(np.clip(E * t, -700.0, 600.0))
        fsp, ftp = endp[kp]
        fsn, ftn = endn[kn]
        B = wexp * fsp[:, None] * fsn[None, :]                       # (3, 3, ..., N)
        TT = sig * wexp * ftp * ftn
        Mp = g * B
        Mm = g * np.conj(B)
        Mp[2, 2] -= TT
        Mm[2, 2] -= np.conj(TT)
        out[0, kk], out[1, kk] = Mp, Mm
    return out


def contact_tensor(t, x, yp, endp, firstp, yn, endn, firstn, mh2):
    """Merged vertex of a hard pair whose A_0 A_0 line is replaced by its contact term +(t^2/m_h^2) delta(t - t'):
    T[br, kk, 1, 1, ..., N], only the temporal components of the mode L of the two soft lines."""
    sig = (-1.0 if firstp else 1.0) * (-1.0 if firstn else 1.0)
    shape = np.broadcast(yp * t, yn * t).shape
    out = np.empty((2, 4, 1, 1) + shape, complex)
    for (kp, kn), kk in KK.items():
        E = -x + (1 if kp == "g" else -1) * yp + (1 if kn == "g" else -1) * yn
        val = sig * t ** 3 * np.exp(np.clip(E * t, -700.0, 600.0)) * endp[kp][1] * endn[kn][1] / mh2
        out[0, kk, 0, 0], out[1, kk, 0, 0] = val, np.conj(val)
    return out


# ------------------------------------------------------------------------------------------------ reference
def cycle_sum(T, h, wint, only=None):
    """Sum over the branch assignments with the first vertex on +, and over all time orderings, of the cyclic
    contraction of the vertex tensors T[p][br, kk, a_prev, a_next, ..., N] (line p joins vertices p and p + 1).
    Slow reference implementation; returns an array over the batch dimensions."""
    n = len(T)
    let = "abcdefg"[:n]
    total = 0
    for bits in itertools.product((0, 1), repeat=n - 1):
        br = (0,) + bits
        if only is not None and br != tuple(only):
            continue
        for perm in itertools.permutations(range(n)):
            rank = {v: i for i, v in enumerate(perm)}

            def kind(p, q):
                if br[p] != br[q]:
                    return "d"
                return "g" if rank[p] < rank[q] else "d"
            R, sub = None, ""
            for i, p in enumerate(perm):
                kk = KK[(kind(p, (p - 1) % n), kind(p, (p + 1) % n))]
                Mv = T[p][br[p], kk]
                subm = let[(p - 1) % n] + let[p]
                if R is None:
                    R, sub = Mv, subm
                else:
                    new = "".join(sorted(set(sub) ^ set(subm))) if n > 2 or i < n - 1 else ""
                    if n == 2:
                        new = ""
                    R = np.einsum(f"{sub}...,{subm}...->{new}...", R, Mv)
                    sub = new
                R = cum(R, h) if i < n - 1 else (R * wint).sum(-1)
            total = total + R
    return total


# ------------------------------------------------------------------------------------------------ kernel
@njit(cache=True, fastmath=True)
def _mm(A, B, out):
    for a in range(3):
        for c in range(3):
            for t in range(A.shape[2]):
                out[a, c, t] = A[a, 0, t] * B[0, c, t] + A[a, 1, t] * B[1, c, t] + A[a, 2, t] * B[2, c, t]


@njit(cache=True, fastmath=True)
def _cum(X, out, Q, h):
    nn = Q.shape[0]
    for a in range(3):
        for c in range(3):
            off = 0j
            for p in range(h.size):
                b0 = p * nn
                for k in range(nn):
                    s = 0j
                    for m in range(nn):
                        s += Q[k, m] * X[a, c, b0 + m]
                    out[a, c, b0 + k] = off + h[p] * s
                off = out[a, c, b0 + nn - 1]


@njit(cache=True, fastmath=True)
def _trint(A, B, w):
    s = 0j
    for a in range(3):
        for b in range(3):
            for t in range(w.size):
                s += w[t] * A[a, b, t] * B[b, a, t]
    return s


@njit(cache=True, fastmath=True)
def _intmm(A, B, w, out):
    for a in range(3):
        for c in range(3):
            s = 0j
            for t in range(w.size):
                s += w[t] * (A[a, 0, t] * B[0, c, t] + A[a, 1, t] * B[1, c, t] + A[a, 2, t] * B[2, c, t])
            out[a, c] += s


@njit(cache=True, fastmath=True)
def _tr2(X, Y):
    s = 0j
    for a in range(3):
        for b in range(3):
            s += X[a, b] * Y[b, a]
    return s


@njit(cache=True, fastmath=True)
def box_kernel(T, Q, h, w):
    """T[p, jj, br, kk, a, a', t] for the cycle positions p = 0..3 = (V1, V2, V4, V3), with the azimuth index jj of
    the side (side 12 for p = 0, 1 and side 34 for p = 2, 3). Returns Z[j, j'], the sum over the eight branch
    assignments with V1 on + and over all orderings of the four times, and Zs[j, j', 8], the same by sector: all on +;
    one vertex on the other branch (four); the two sides on opposite branches; V1, V3 against V2, V4; alternating."""
    M, N = T.shape[1], T.shape[6]
    GG, GD, DG, DD = 0, 1, 2, 3
    C = np.zeros((4, M, 2, 3, 3, 3, N), np.complex128)          # cumulative integrals of the kinds gg, gd, dg
    Idd = np.zeros((4, M, 2, 3, 3), np.complex128)
    for p in range(4):
        for jj in range(M):
            for br in range(2):
                for kk in range(3):
                    _cum(T[p, jj, br, kk], C[p, jj, br, kk], Q, h)
                for a in range(3):
                    for c in range(3):
                        s = 0j
                        for t in range(N):
                            s += w[t] * T[p, jj, br, DD, a, c, t]
                        Idd[p, jj, br, a, c] = s
    tmp = np.zeros((3, 3, N), np.complex128)
    tmp2 = np.zeros((3, 3, N), np.complex128)
    R2 = np.zeros((3, 3, N), np.complex128)
    R3 = np.zeros((3, 3, N), np.complex128)
    R3o = np.zeros((3, 3, N), np.complex128)
    R3b = np.zeros((3, 3, N), np.complex128)
    # single-side pieces: R2cw for p = 0, 2 and R2ccw for p = 1, 3 on the + branch; the split sector
    S2cw = np.zeros((2, M, 3, 3, N), np.complex128)
    S2ccw = np.zeros((2, M, 3, 3, N), np.complex128)
    X12 = np.zeros((M, 3, 3), np.complex128)
    X43 = np.zeros((M, 3, 3), np.complex128)
    for jj in range(M):
        for i in range(2):
            p = 2 * i
            _mm(C[p, jj, 0, GG], T[p + 1, jj, 0, DG], tmp)
            _cum(tmp, S2cw[i, jj], Q, h)
            p = 2 * i + 1
            _mm(T[p - 1, jj, 0, GD], C[p, jj, 0, GG], tmp)
            _cum(tmp, S2ccw[i, jj], Q, h)
        _intmm(C[0, jj, 0, DG], T[1, jj, 0, DD], w, X12[jj])
        _intmm(T[0, jj, 0, DD], C[1, jj, 0, GD], w, X12[jj])
        _intmm(C[2, jj, 1, DG], T[3, jj, 1, DD], w, X43[jj])
        _intmm(T[2, jj, 1, DD], C[3, jj, 1, GD], w, X43[jj])
    Z = np.zeros((M, M), np.complex128)
    Zs = np.zeros((M, M, 8), np.complex128)
    W = np.zeros((3, 3), np.complex128)
    Xa = np.zeros((3, 3), np.complex128)
    Xb = np.zeros((3, 3), np.complex128)
    jx = np.zeros(4, np.int64)
    for j in range(M):
        for jp in range(M):
            jx[0], jx[1], jx[2], jx[3] = j, j, jp, jp
            res = 0j
            # ---- (+ + + +)
            for p in range(4):
                p1, p2, p3 = (p + 1) % 4, (p + 2) % 4, (p + 3) % 4
                if p == 0 or p == 2:
                    R2[:] = S2cw[p // 2, jx[p]]
                else:
                    _mm(C[p, jx[p], 0, GG], T[p1, jx[p1], 0, DG], tmp)
                    _cum(tmp, R2, Q, h)
                _mm(R2, T[p2, jx[p2], 0, DG], tmp)
                _cum(tmp, R3, Q, h)
                res += _trint(R3, T[p3, jx[p3], 0, DD], w)
                _mm(T[p3, jx[p3], 0, GD], R2, tmp)
                _cum(tmp, R3o, Q, h)
                if p == 1 or p == 3:
                    R2[:] = S2ccw[p // 2, jx[p]]
                else:
                    _mm(T[p3, jx[p3], 0, GD], C[p, jx[p], 0, GG], tmp)
                    _cum(tmp, R2, Q, h)
                _mm(T[p2, jx[p2], 0, GD], R2, tmp)
                _cum(tmp, R3, Q, h)
                res += _trint(T[p1, jx[p1], 0, DD], R3, w)
                _mm(R2, T[p1, jx[p1], 0, DG], tmp)
                _cum(tmp, R3b, Q, h)
                for a in range(3):
                    for c in range(3):
                        for t in range(N):
                            R3o[a, c, t] += R3b[a, c, t]
                res += _trint(R3o, T[p2, jx[p2], 0, DD], w)
            for p in range(2):
                p1, p2, p3 = p + 1, p + 2, (p + 3) % 4
                _mm(C[p, jx[p], 0, GG], T[p1, jx[p1], 0, DD], tmp)
                _mm(tmp, C[p2, jx[p2], 0, GG], tmp2)
                _cum(tmp2, R3, Q, h)
                res += _trint(R3, T[p3, jx[p3], 0, DD], w)
                _mm(C[p2, jx[p2], 0, GG], T[p3, jx[p3], 0, DD], tmp)
                _mm(tmp, C[p, jx[p], 0, GG], tmp2)
                _cum(tmp2, R3, Q, h)
                res += _trint(R3, T[p1, jx[p1], 0, DD], w)
            Zs[j, jp, 0] = res
            # ---- one vertex on the other branch
            for o in range(4):
                o1, o2, o3 = (o + 1) % 4, (o + 2) % 4, (o + 3) % 4
                bo = 0 if o == 0 else 1
                bp = 1 - bo
                W[:] = 0
                _mm(C[o1, jx[o1], bp, DG], T[o2, jx[o2], bp, DG], tmp)
                _cum(tmp, R3, Q, h)
                _intmm(R3, T[o3, jx[o3], bp, DD], w, W)
                _mm(T[o2, jx[o2], bp, GD], C[o3, jx[o3], bp, GD], tmp)
                _cum(tmp, R3, Q, h)
                _intmm(T[o1, jx[o1], bp, DD], R3, w, W)
                _mm(T[o1, jx[o1], bp, DD], C[o2, jx[o2], bp, GG], tmp)
                _cum(tmp, R3, Q, h)
                _intmm(R3, T[o3, jx[o3], bp, DD], w, W)
                _mm(C[o2, jx[o2], bp, GG], T[o3, jx[o3], bp, DD], tmp)
                _cum(tmp, R3, Q, h)
                _intmm(T[o1, jx[o1], bp, DD], R3, w, W)
                _mm(C[o1, jx[o1], bp, DG], T[o2, jx[o2], bp, DD], tmp)
                _intmm(tmp, C[o3, jx[o3], bp, GD], w, W)
                res += _tr2(Idd[o, jx[o], bo], W)
                Zs[j, jp, 1 + o] = _tr2(Idd[o, jx[o], bo], W)
            # ---- two and two: the two sides on opposite branches
            res += _tr2(X12[j], X43[jp])
            Zs[j, jp, 5] = _tr2(X12[j], X43[jp])
            # ---- V1, V3 on + and V2, V4 on -: lines A and A' cut, line 1 ordered, line 2 anti-ordered
            Xa[:] = 0
            Xb[:] = 0
            _intmm(C[3, jp, 0, DG], T[0, j, 0, DD], w, Xa)
            _intmm(T[3, jp, 0, DD], C[0, j, 0, GD], w, Xa)
            _intmm(C[1, j, 1, DG], T[2, jp, 1, DD], w, Xb)
            _intmm(T[1, j, 1, DD], C[2, jp, 1, GD], w, Xb)
            res += _tr2(Xa, Xb)
            Zs[j, jp, 6] = _tr2(Xa, Xb)
            # ---- alternating around the cycle: all four lines cut
            s = 0j
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            s += Idd[0, j, 0, a, b] * Idd[1, j, 1, b, c] * Idd[2, jp, 0, c, d] * Idd[3, jp, 1, d, a]
            res += s
            Zs[j, jp, 7] = s
            Z[j, jp] = res
    return Z, Zs


# ------------------------------------------------------------------------------------------------ the box
class Box:
    def __init__(self, mu, kappa, Pc=0.5, M=8, grid="std", contact=True, tlo=1e-5, w_late=0.45, part=None, Pmin=None):
        self.mu, self.kappa, self.Pc, self.M, self.contact, self.part, self.Pmin = mu, kappa, Pc, M, contact, part, Pmin
        self.modes = Modes(mu, kappa)
        self.mh2 = mu * mu + 0.25
        self.st = GRIDS[grid] if isinstance(grid, str) else grid
        self.tlo, self.w_late, self.dlin, self.puv = tlo, w_late, 6.0, 2.5
        self.ring = False
        self.set_tgrid(40.0)
        self.phis = 2 * math.pi * np.arange(M) / M

    def set_tgrid(self, thi, emax=None):
        """Time grid up to thi. The default (40 in units of 1/k, panels in ln t) serves loop momenta below about 2 k.
        With emax (the largest line energies) the late panels have a fixed width in t instead (`tgrid_uv`)."""
        if getattr(self, "tkey", None) != (thi, emax):
            self.tkey = (thi, emax)
            if emax is None:
                self.t, self.h, self.wint = tgrid2(tlo=self.tlo, thi=thi, w_late=self.w_late)
            else:
                self.t, self.h, self.wint = tgrid_uv(emax, thi, tlo=self.tlo, w_late=self.w_late, dlin=self.dlin)

    def soft(self, r, y1, y2):
        """The two soft lines at one loop point: loop momentum l[M, 3] on the azimuth grid, polarization vectors and
        end functions of line 1 (momentum l at its first end) and line 2 (momentum s - l)."""
        t, M = self.t, self.M
        qz = (1 + y1 * y1 - y2 * y2) / 2
        rho = math.sqrt(max(y1 * y1 - qz * qz, 0.0))
        l = r * np.stack([rho * np.cos(self.phis), rho * np.sin(self.phis), np.full(M, qz)], -1)      # (M, 3)
        svec = np.array([0.0, 0.0, r])
        e1, e2 = self.modes.ends(r * y1, t), self.modes.ends(r * y2, t)        # (3, N), (N,)
        return dict(l=l, svec=svec, y1=r * y1, y2=r * y2, U1=pol_vectors(l / (r * y1)), U2=pol_vectors((svec - l) / (r * y2)),
                    e1={k: (v[0][:, None, :], v[1][None, :]) for k, v in e1.items()},      # -> (3, 1, N), (1, N)
                    e2={k: (v[0][:, None, :], v[1][None, :]) for k, v in e2.items()})

    def side(self, S, c, p, which):
        """Vertex tensors of one side for one loop point (S from `soft`), T[2 (vertex), M, br, kk, a, a', N], and its
        contact tensor. which = 12: vertices (V1, V2), hard momentum k1 - l; which = 34: (V4, V3) in cycle order, hard
        momentum k3 + l, in the frame where k3 has azimuth zero (the loop azimuth is then phi - beta)."""
        t = self.t
        l, svec, y1, y2, U1, U2, e1b, e2b = (S[k] for k in ("l", "svec", "y1", "y2", "U1", "U2", "e1", "e2"))
        kv = p * np.array([math.sqrt(max(1 - c * c, 0.0)), 0.0, c])
        if which == 12:
            P = kv - l
            xa, xb = p, float(np.linalg.norm(svec - kv))                      # |k1|, |k2|
        else:
            P = kv + l
            xa, xb = p, float(np.linalg.norm(-svec - kv))                     # |k3|, |k4|
        Y = np.linalg.norm(P, axis=-1)                                         # (M,)
        UA = pol_vectors(P / Y[:, None])
        ya = Y[:, None]
        eA = self.modes.ends(ya, t)                                            # (3, M, N), (M, N)
        if which == 12:
            # V1: prev = line 1 (first end), next = A (first end); V2: prev = A (second end), next = line 2 (first end)
            Ta = vertex_tensor(t, xa, 0, y1, e1b, U1, True, ya, eA, UA, True, norm=1 / (2 * ya))
            Tb = vertex_tensor(t, xb, 0, ya, eA, UA, False, y2, e2b, U2, True)
            Tc = contact_tensor(t, xa + xb, y1, e1b, True, y2, e2b, True, self.mh2)
            out = np.stack([Ta, Tb])                                           # (2, br, kk, a, a', M, N)
        else:
            # V4: prev = line 2 (second end), next = A' (second end); V3: prev = A' (first end), next = line 1 (second end)
            Tb = vertex_tensor(t, xb, 0, y2, e2b, U2, False, ya, eA, UA, False)
            Ta = vertex_tensor(t, xa, 0, ya, eA, UA, True, y1, e1b, U1, False, norm=1 / (2 * ya))
            Tc = contact_tensor(t, xa + xb, y2, e2b, False, y1, e1b, False, self.mh2)
            out = np.stack([Tb, Ta])
        return np.ascontiguousarray(np.moveaxis(out, 5, 1)), Tc               # (2, M, br, kk, a, a', N)

    def point(self, r, y1, y2, c1=None, c3=None, p1=1.0, p3=1.0):
        """Z[j, j'] of one loop point, and the three pieces with a contact term (arrays over j', over j, scalar)."""
        c1 = r / 2 if c1 is None else c1
        c3 = -r / 2 if c3 is None else c3
        Pph = r * (y1 + y2)
        if Pph <= self.puv:
            self.set_tgrid(40.0)
        else:           # every vertex then decays at least as e^(-(1 + elo) t), and the overall time as e^(-4 t)
            elo = max(Pph / 2 - 1.0, 0.0)
            self.set_tgrid(round(max(5.0, 40.0 / (1.0 + elo)), 3), round(Pph + 1.0, 6))
        S = self.soft(r, y1, y2)
        T12, Tc12 = self.side(S, c1, p1, 12)
        T34, Tc34 = self.side(S, c3, p3, 34)
        T = np.concatenate([T12, T34])
        Z, _ = box_kernel(T, Q_N, self.h, self.wint)
        zc = [0.0, 0.0, 0.0]
        if self.contact:
            L = slice(2, 3)
            # (contact on 12, pair on 34): cycle (Vc12, V4, V3); lines 1 and 2 carry the mode L only
            A = [Tc12, np.moveaxis(T34[0], 0, -2)[:, :, L, :], np.moveaxis(T34[1], 0, -2)[:, :, :, L]]
            zc[0] = cycle_sum(A, self.h, self.wint)
            A = [np.moveaxis(T12[0], 0, -2)[:, :, L, :], np.moveaxis(T12[1], 0, -2)[:, :, :, L], Tc34]
            zc[1] = cycle_sum(A, self.h, self.wint)
            zc[2] = complex(np.ravel(cycle_sum([Tc12, Tc34], self.h, self.wint))[0])
        return Z, zc

    def run(self, r, verbose=False):
        """Z(beta_m), m = 0..M-1, of the planar box with the cut Pc at equal external momenta, and its pieces."""
        if self.ring:
            y1, y2, w = ring_grid(r, self.Pmin, self.Pc)
        else:
            y1, y2, w = loop_grid(r, self.Pc, self.st, self.Pmin)
        if self.part:
            y1, y2, w = (v[self.part[0]::self.part[1]] for v in (y1, y2, w))
        M = self.M
        Zb, Zc = np.zeros(M, complex), np.zeros(3, complex)
        idx = np.arange(M)
        t0 = time.time()
        for i in range(y1.size):
            Z, zc = self.point(r, y1[i], y2[i])
            pref = w[i] * r ** 3 / (4 * r * r * y1[i] * y2[i])
            for m in range(M):
                Zb[m] += pref * Z[idx, (idx - m) % M].mean()
            Zc += pref * np.array([np.mean(zc[0]), np.mean(zc[1]), zc[2]])
            if verbose and i % 100 == 0:
                print(i, y1.size, round(time.time() - t0, 1), flush=True)
        return Zb, Zc, int(y1.size)


def harmonics(Zb, Zc):
    """I(beta_m) = Z(beta_m) + conj Z(-beta_m), with the contact pieces (independent of beta), and its harmonics."""
    M = Zb.size
    Zt = Zb + Zc.sum()
    I = Zt + np.conj(Zt[(-np.arange(M)) % M])
    F = np.fft.fft(I) / M                                  # F[n] = coefficient of e^(i n beta), n mod M
    return I, F


if __name__ == "__main__":
    mu, ka, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    part = os.environ.get("PART")
    grid = os.environ.get("GRID", "std")
    B = Box(mu, ka, Pc=float(os.environ.get("PC", 0.5)), M=int(os.environ.get("M", 8)), grid=json.loads(grid) if grid.startswith("{") else grid,
            contact=os.environ.get("CONTACT", "1") != "0", tlo=float(os.environ.get("TLO", 1e-5)),
            w_late=float(os.environ.get("TW", 0.45)), part=tuple(int(x) for x in part.split("/")) if part else None,
            Pmin=float(os.environ["PMIN"]) if os.environ.get("PMIN") else None)
    B.ring = bool(os.environ.get("RING"))
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["r"], json.loads(l).get("Pmin"), json.loads(l)["Pc"]) for l in open(out) if json.loads(l).get("part") == part}
    for s in sys.argv[4:]:
        r = float(s)
        if (r, B.Pmin, B.Pc) in done:
            continue
        t0 = time.process_time()
        Zb, Zc, npts = B.run(r, verbose=bool(os.environ.get("VERBOSE")))
        I, F = harmonics(Zb, Zc)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(mu=mu, kappa=ka, r=r, Pc=B.Pc, Pmin=B.Pmin, ring=B.ring, M=B.M, grid=os.environ.get("GRID", "std"), part=part, npts=npts,
                                     nt=int(B.t.size), contact=B.contact, cpu=round(time.process_time() - t0, 1),
                                     Z=[[z.real, z.imag] for z in Zb], Zc=[[z.real, z.imag] for z in Zc],
                                     I=[[z.real, z.imag] for z in I], F=[[z.real, z.imag] for z in F])) + "\n")
