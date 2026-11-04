"""The fermion triangle with an axial chemical potential: leading squeezed coefficient (Sec. 9.3).

Model of Aoki and Strumia (arXiv:2604.05548): delta tau couples to the fermion currents through
Gamma^mu = gamma^mu (mu_V + mu_A gamma^5)/taudot_0 (we set mu_V = 0, kappa = mu_A, and strip the 1/taudot_0).
The bispectrum is their Eq. (45),

    <dtau^3>' = - int deta_123 sum_{a_i} a1 a2 a3 F_mu1(k1) F_mu2(k2) F_mu3(k3) int d^3q/(2pi)^3 A^{mu1 mu2 mu3},
    A = Tr[Gamma^mu1 D_a1a2(q + k1) Gamma^mu2 D_a2a3(q + k1 + k2) Gamma^mu3 D_a3a1(q)] + (1 <-> 2),

and we write the leading squeezed term as <dtau^3>' > C k1^-6 (k3/k1)^(2 i lam), so that S = -4 pi^2 P_zeta C
(N_c = 1). Collinear kinematics, k3 parallel to k1 (z axis), as in their Eq. (49).

Method of regions (Sec. 7.1): the soft lines (momenta q and q - k3) end on the soft vertex at eta3 ~ 1/k3 and
on the hard vertices at late times. Then
  * the hard part has the hard vertices, the hard line (both helicities, all orderings) and the soft lines at
    their late-time ends on the branch (-eta)^(i lam). It is a nested double time integral for the ordered
    Schwinger-Keldysh sectors and a product of single integrals for the cut ones. We evaluate it on partially
    rotated rays eta = -t e^(-+ i theta) with Chebyshev panels in ln t (ext/cheb.py). On the fully rotated ray
    terms of size e^(pi lam) cancel, so theta is kept small at large kappa.
  * the soft vertex has the two soft lines in full. With the helicity projectors traced, it is a sum of
    monomials y1^a y2^b times two Whittaker functions, done in position space (ext/softvertex.py).
In the spin basis along z every hard factor is diagonal, so the trace over the soft helicities is
    Tr[M Pi_h1(qhat) N Pi_h2(phat)] = sum_s m_s n_s (Pi_h1)_ss (Pi_h2)_ss + (h1 h2/4) SS (m_+ n_- + m_- n_+),
with SS = qhat_perp . phat_perp. The (1 <-> 2) term is the same hard part with z -> -z, i.e. n_s -> n_-s.

Late-time soft vertex, for comparison: 'nl' keeps the branch (2 y t)^(1/2 + i lam) of both soft lines at the
soft vertex (the approximation of Aoki and Strumia), 'all0' keeps both branches of each line at leading order.

Usage:
    python fermion_triangle.py               kappa = 1.3 and 5 (m = 1), about 10 min
    python fermion_triangle.py --point 10 1  one point (kappa, m); --theta and --dps-soft override SETTINGS
    python fermion_triangle.py --all         all four points of Sec. 9.3 (about 1.5 h on one core)
The hard part is cached in cache/ (delete it to recompute).
"""
import argparse
import hashlib
import os
import pickle
import sys
import time

import mpmath as mp

from ext.blocks import C2hat
from ext.cheb import Panels
from ext.fermion import ASFermion
from ext.record import compare, load, report
from ext.softvertex import Grid, Leg, soft_J

I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# geometry polynomials in y1 = |q|, y2 = |q - k3| (k3 = 1): {(a, b): coef} for y1^a y2^b
QZ = {(1, 0): 0.5, (-1, 0): 0.5, (-1, 2): -0.5}        # qhat . z
PZ = {(2, -1): 0.5, (0, -1): -0.5, (0, 1): -0.5}       # phat . z, p = q - k3
QP = {(1, -1): 0.5, (-1, -1): -0.5, (-1, 1): 0.5}      # qhat . phat


def padd(*ps):
    out = {}
    for c, p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return {k: v for k, v in out.items() if v != 0}


def pmul(p, q):
    out = {}
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            k = (a1 + a2, b1 + b2)
            out[k] = out.get(k, 0) + c1 * c2
    return {k: v for k, v in out.items() if v != 0}


ONE = {(0, 0): 1}
SS = padd((1, QP), (-1, pmul(QZ, PZ)))                  # qhat_perp . phat_perp


def diag_poly(h1, h2, s):
    """(Pi_h1(qhat))_ss (Pi_h2(phat))_ss = (1 + h1 s qz)(1 + h2 s pz)/4."""
    return pmul(padd((0.5, ONE), (0.5 * h1 * s, QZ)), padd((0.5, ONE), (0.5 * h2 * s, PZ)))


# soft-vertex entries of sigma^mu F_mu (R) and sigmabar^mu F_mu (L) on the rotated ray, as (coef, power of t),
# with e^(-t) and the 1/(2 k3^3) included (k3 = 1): ENTRY[a3][X][s]
ENTRY = {
    1: {'R': {1: [(-I / 2, 0)], -1: [(I / 2, 0), (I, 1)]},
        'L': {1: [(I / 2, 0), (I, 1)], -1: [(-I / 2, 0)]}},
    -1: {'R': {1: [(-I / 2, 0), (-I, 1)], -1: [(I / 2, 0)]},
         'L': {1: [(I / 2, 0)], -1: [(-I / 2, 0), (-I, 1)]}},
}
KINDS = {1: ('u', 'v'), -1: ('v', 'u')}   # a3 -> (barred soft line q - k3, unbarred soft line q) at the soft vertex
HEL = (1, -1)


def mat_vec(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]


def vec_mat(v, M):
    return [v[0] * M[0][0] + v[1] * M[1][0], v[0] * M[0][1] + v[1] * M[1][1]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


# ------------------------------------------------------------------------------------------------ hard part
class HardPart:
    """n_s(a1, a2; a3, h1, h2) of the module docstring, family (-eta)^(sg nu) on both soft lines.

    Returns Ntil[(a3, h1, h2, s)] = sum_{a1 a2} a1 a2 [n_s + n_-s] (both orientations).
    """

    def __init__(self, F, theta, segments=((-40, 1.5, 0.25), (1.5, 6.0, 0.08)), n=32, sg=1):
        self.F, self.theta, self.sg = F, mp.mpf(theta), sg
        self.P = Panels(segments, n)
        self.key = (str(F.m), str(F.kappa), str(theta), tuple(segments), n, sg, mp.mp.dps)

    def _tables(self, ray):
        """Vertex integrands on the ray of branch a = ray: f_U, f_V (vertex 1) and g_U, g_V (vertex 2)."""
        F, sg, nu = self.F, self.sg, self.F.nu
        ph = mp.exp(-I * ray * self.theta)          # eta = -t ph, d eta -> ph dt
        late_r = {(a3, h1): F.late_row(KINDS[a3][1], h1, sg) for a3 in (1, -1) for h1 in HEL}
        late_c = {(a3, h2): F.late_column(KINDS[a3][0], h2, sg) for a3 in (1, -1) for h2 in HEL}
        f = {}
        for key in [(a3, h, s, w) for a3 in (1, -1) for h in HEL for s in HEL for w in ('U', 'V')]:
            f[('f',) + key] = [[None] * (self.P.n + 1) for _ in self.P.u]
            f[('g',) + key] = [[None] * (self.P.n + 1) for _ in self.P.u]
        for p, j, t in self.P.nodes():
            eta = -t * ph
            meas = ph * t * (-eta) ** (sg * nu)      # Jacobian, dt = t du, late-end power
            wc = {}
            for s in HEL:
                h = s                                 # hard helicity along +z
                U, V = F.column('u', h, eta, cache=wc), F.column('v', h, eta, cache=wc)
                Ub, Vb = F.row('u', h, eta, cache=wc), F.row('v', h, eta, cache=wc)
                G1, G2 = F.vertex(1, ray, eta, s), F.vertex(-1, ray, eta, s)
                G1U, G1V = mat_vec(G1, U), mat_vec(G1, V)
                UbG2, VbG2 = vec_mat(Ub, G2), vec_mat(Vb, G2)
                for a3 in (1, -1):
                    for hh in HEL:
                        r1, c2 = late_r[(a3, hh)], late_c[(a3, hh)]
                        f[('f', a3, hh, s, 'U')][p][j] = meas * dot(r1, G1U)
                        f[('f', a3, hh, s, 'V')][p][j] = meas * dot(r1, G1V)
                        f[('g', a3, hh, s, 'U')][p][j] = meas * dot(UbG2, c2)
                        f[('g', a3, hh, s, 'V')][p][j] = meas * dot(VbG2, c2)
        return f

    def compute(self, verbose=True):
        path = os.path.join(CACHE, "hard_" + hashlib.sha1(repr(self.key).encode()).hexdigest()[:16] + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            if d["key"] == self.key:
                return d["N"]
        t0 = time.time()
        T = {ray: self._tables(ray) for ray in (1, -1)}
        if verbose:
            print(f"   hard part: tables on both rays in {time.time() - t0:.0f} s")
        P = self.P
        cum = {ray: {k: P.cumulative(v) for k, v in T[ray].items()} for ray in (1, -1)}
        def single(ray, k):
            return cum[ray][k][-1][-1]
        def nested(ray, kf, kg, order):
            """order 'f<g': int_{t1 < t2} f(t1) g(t2); 'g<f': int_{t2 < t1}."""
            if order == 'f<g':
                F_, g = cum[ray][kf], T[ray][kg]
                return P.integral([[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(F_, g)])
            G_, f_ = cum[ray][kg], T[ray][kf]
            return P.integral([[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(G_, f_)])
        n = {}
        for a3 in (1, -1):
            for h1 in HEL:
                for h2 in HEL:
                    for s in HEL:
                        fU, fV = ('f', a3, h1, s, 'U'), ('f', a3, h1, s, 'V')
                        gU, gV = ('g', a3, h2, s, 'U'), ('g', a3, h2, s, 'V')
                        npm = I * single(1, fV) * single(-1, gV)          # D_{+-} = i v vbar
                        nmp = -I * single(-1, fU) * single(1, gU)         # D_{-+} = -i u ubar
                        npp = -I * nested(1, fU, gU, 'f<g') + I * nested(1, fV, gV, 'g<f')
                        nmm = I * nested(-1, fV, gV, 'f<g') - I * nested(-1, fU, gU, 'g<f')
                        n[(a3, h1, h2, s)] = {(1, -1): npm, (-1, 1): nmp, (1, 1): npp, (-1, -1): nmm}
        N = {}
        for (a3, h1, h2, s), d in n.items():
            tot = 0
            for (a1, a2), v in d.items():
                tot += a1 * a2 * (v + n[(a3, h1, h2, -s)][(a1, a2)])
            N[(a3, h1, h2, s)] = tot
        if verbose:
            print(f"   hard part done in {time.time() - t0:.0f} s")
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"key": self.key, "N": N, "n": n}, fh)
        return N


# ------------------------------------------------------------------------------------------------ soft part
class SoftPart:
    """m_s[P](a3, h1, h2): the soft vertex with the polynomial weight P, exact or late-time."""

    def __init__(self, F, sg=1, grid=None):
        self.F, self.sg = F, sg
        self.grid = grid or Grid()
        self._legs, self._J = {}, {}

    def leg(self, A, K):
        key = (mp.nstr(A, 30), mp.nstr(K, 30))
        if key not in self._legs:
            self._legs[key] = Leg(A, K, self.F.nu)
        return self._legs[key]

    def J(self, a, b, Kk, Kr, p, mode):
        """2^(2 sg nu) int d^3q/(2pi)^3 y1^(a + sg nu - 1/2) y2^(b + sg nu - 1/2) int dt t^(p-1) e^-t W_Kk(2 y1 t) W_Kr(2 y2 t)."""
        key = (a, b, mp.nstr(Kk, 30), mp.nstr(Kr, 30), p, mode)
        if key in self._J:
            return self._J[key]
        F, sg, nu = self.F, self.sg, self.F.nu
        A, B = a + sg * nu - 0.5, b + sg * nu - 0.5
        if mode == 'exact':
            val = soft_J(self.leg(A, Kk), self.leg(B, Kr), p, self.grid)
        else:
            br = (sg,) if mode == 'nl' else (1, -1)
            val = 0
            for b1 in br:
                for b2 in br:
                    pre = F.B(b1, Kk) * F.B(b2, Kr) * mp.mpf(2) ** (1 + (b1 + b2) * nu)
                    e1, e2 = A + 0.5 + b1 * nu, B + 0.5 + b2 * nu
                    Sm = -(e1 + e2) / 2 - 1.5             # Gamma(S - 3/2) of the block
                    if abs(Sm - mp.nint(Sm.real)) < 1e-20 and mp.nint(Sm.real) <= 0:
                        continue      # integer line powers (b1 = b2 = -sg): UV-divergent or scaleless; dropped
                    loop = C2hat(e1 + 1, e2 + 1) / (2 * mp.pi) ** 2
                    val += pre * loop * mp.gamma(p + 1 + (b1 + b2) * nu)
        val *= mp.mpf(2) ** (2 * sg * nu)
        self._J[key] = val
        return val

    def m(self, a3, h1, h2, s, poly, mode='exact', hel_soft=None):
        F = self.F
        if hel_soft is not None and (h1 != hel_soft or h2 != hel_soft):
            return 0
        kr, kk = KINDS[a3]
        cr, ck = F.comps(kr, h2), F.comps(kk, h1)
        E = mp.exp((h1 + h2) * mp.pi * F.kappa / 2)
        tot = 0
        for X, mu in (('L', F.muL), ('R', F.muR)):
            c = mp.conj(cr[X][0]) * ck[X][0]
            Kr, Kk = mp.conj(cr[X][1]), ck[X][1]
            for f, p in ENTRY[a3][X][s]:
                for (a, b), coef in poly.items():
                    tot += mu * c * f * coef * self.J(a, b, Kk, Kr, p, mode)
        return E / 2 * tot


def coefficient(N, soft, mode='exact', hel_soft=None):
    """Assemble C = + sum_a3 a3 sum_h1h2 [sum_s N_s m_s[diag] + (h1 h2/4)(N_- m_+[SS] + N_+ m_-[SS])].

    Overall sign. Each vertex is i a int L_int with L_int = + d_mu(dtau) Psibar Gamma^mu Psi (the sign fixed by
    mu_P = f taudot_0 in their Eq. (43)); the fermion loop gives -Tr[Gamma S Gamma S Gamma S] with
    S = <T Psi Psibar> = i D. Hence <dtau^3>' = (i)^3 a1 a2 a3 (-1) (i)^3 Tr[...] = + a1 a2 a3 Tr[...], the
    opposite of the leading minus printed in Eq. (45) of Aoki-Strumia. With this sign the values of the
    paper (computed Oct. 28) are reproduced; the moduli do not depend on it.
    """
    C = 0
    for a3 in (1, -1):
        for h1 in HEL:
            for h2 in HEL:
                tr = 0
                for s in HEL:
                    tr += N[(a3, h1, h2, s)] * soft.m(a3, h1, h2, s, diag_poly(h1, h2, s), mode, hel_soft)
                tr += h1 * h2 / mp.mpf(4) * (N[(a3, h1, h2, -1)] * soft.m(a3, h1, h2, 1, SS, mode, hel_soft)
                                            + N[(a3, h1, h2, 1)] * soft.m(a3, h1, h2, -1, SS, mode, hel_soft))
                C += a3 * tr
    return C


def run_point(kappa, m, theta, dps_hard, dps_soft, segments, n=32, grid_pts=24, verbose=True):
    """C for one (kappa, m): hard part at dps_hard (cached), soft vertex and assembly at dps_soft."""
    with mp.workdps(dps_hard):
        F = ASFermion(m, kappa)
        N = HardPart(F, theta, segments=segments, n=n).compute(verbose)
    with mp.workdps(dps_soft):
        F = ASFermion(m, kappa)
        N = {k: mp.mpmathify(v) for k, v in N.items()}
        S = SoftPart(F, grid=Grid(npts=grid_pts))
        t0 = time.time()
        out = {mode: coefficient(N, S, mode) for mode in ('nl', 'all0')}
        out['nl h=+'] = coefficient(N, S, 'nl', hel_soft=1)
        out['exact'] = coefficient(N, S, 'exact')
        if verbose:
            print(f"   soft vertex and assembly in {time.time() - t0:.0f} s")
        return out


# production settings: (kappa, m) -> theta, dps_hard, dps_soft, panels in ln t, Lobatto points - 1
SETTINGS = {
    (1.3, 1): dict(theta=0.6, dps_hard=25, dps_soft=30, segments=((-32, 1.5, 0.4), (1.5, 5.0, 0.12)), n=28),
    (5, 1): dict(theta=0.5, dps_hard=30, dps_soft=30, segments=((-32, 1.5, 0.3), (1.5, 5.0, 0.1)), n=32),
    (10, 1): dict(theta=0.25, dps_hard=30, dps_soft=40, segments=((-32, 1.5, 0.25), (1.5, 5.0, 0.08)), n=32),
    (20, 2): dict(theta=0.15, dps_hard=40, dps_soft=60, segments=((-32, 1.5, 0.2), (1.5, 5.6, 0.06)), n=32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs=2, type=float, metavar=("KAPPA", "M"))
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--dps-soft", type=int, default=None)
    ap.add_argument("--grid-pts", type=int, default=24)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.point:
        k, m = args.point
        pts = [(int(k) if k.is_integer() else k, int(m) if m.is_integer() else m)]
    elif args.all:
        pts = list(SETTINGS)
    else:
        pts = [(1.3, 1), (5, 1)]
    res = []
    for kappa, m in pts:
        st = dict(SETTINGS.get((kappa, m), SETTINGS[(10, 1)]))
        if args.theta:
            st["theta"] = args.theta
        if args.dps_soft:
            st["dps_soft"] = args.dps_soft
        print(f"kappa = {kappa}, m = {m}: {st}")
        out = run_point(kappa, m, grid_pts=args.grid_pts, **st)
        for k, v in out.items():
            print(f"   C[{k:6s}] = {mp.nstr(v, 12)}   |S|/P_zeta = {mp.nstr(4 * mp.pi ** 2 * abs(v), 5)}")
        ex = out['exact']
        print(f"   |exact/nl| = {mp.nstr(abs(ex / out['nl']), 4)},  |all0/exact| = {mp.nstr(abs(out['all0'] / ex), 4)}")
        tag = f"kappa={kappa},m={m}"
        table = load()
        for key in ("exact", "nl", "nl h=+", "all0"):
            rk = f"fermion_triangle/C {key} {tag}"
            if rk in table:
                res.append(compare(rk, out[key], table))
        for rk, val in ((f"fermion_triangle/|S|/Pz {tag}", 4 * mp.pi ** 2 * abs(ex)),
                        (f"fermion_triangle/|exact/nl| {tag}", abs(ex / out['nl'])),
                        (f"fermion_triangle/|all0/exact| {tag}", abs(out['all0'] / ex))):
            if rk in table:
                res.append(compare(rk, val, table))
    return report(res) if res else 0


if __name__ == "__main__":
    sys.exit(main())
