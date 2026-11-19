"""The fermion triangle with an axial chemical potential, order by order in r = k3/k1 and at general shape.

Model and conventions of ext/fermion.py and fermion_triangle.py (Aoki and Strumia, arXiv:2604.05548): a Dirac
fermion of mass m with axial chemical potential kappa, vertices dtau Psibar Gamma^mu Psi with
Gamma^mu = gamma^mu gamma^5 kappa (the 1/taudot_0 stripped). One orientation of the loop is

    sum_{a1 a2 a3} a1 a2 a3 int d^3q/(2pi)^3 Tr[Gamma.F(k1) D_{a1a2}(q + k1) Gamma.F(k2) D_{a2a3}(q - k3) Gamma.F(k3) D_{a3a1}(q)],

and the other one is equal to it (charge conjugation; the axial current is even), so the correlator is twice
this. We write k1 = 1, k3 = r, c = k1hat . k3hat, |k2| = 1 + xi = sqrt(1 + 2 r c + r^2), and

    k1^6 <dtau^3>'  contains  sum_n C_n(c) r^(2 nu + n)  + c.c.,       nu = i sqrt(m^2 + kappa^2),

so that C_0(1) is the coefficient C of fermion_triangle.py (collinear, k3 parallel to k1).

Region. As for the vector triangle (ext/vectri.py), the non-analytic dependence on k3 comes from q = r q' and
t3 = T/r. There the soft lines are (soft end at V3) x (hard end at V1 or V2), independently of the branches of
the hard vertices,

    D_{a3 a1}(q)      ->  +i a3 X(3) Xbar(1),     X  = v (a3 = +), u (a3 = -),
    D_{a2 a3}(q - k3) ->  -i a3 X'(2) X'bar(3),   X' = u (a3 = +), v (a3 = -),

up to pieces that are analytic in the momentum of the line once the helicities are summed. On the fully rotated
contour eta = i a t every component of a spinor is (2 y t)^(-1/2) e^(i a pi/4) e^(pi h kappa/2) c times

    W_{K,nu}(2 y t)                                                         (decaying), or
    sum_s B_s(K) e^(-i pi a (1/2 + s nu)) M_{-K, s nu}(2 y t)               (growing),

and the phases e^(i a pi/4) cancel against the measures -i a dt. The hard ends are expanded on the branch
s = + (the pair family), M_{K,nu}(z) = z^(1/2+nu) sum_n e_n(K, nu) z^n. The hard line is kept in full. Its
moments at the folded point are Gamma functions (cut) and 4F3(-1) (ordered, the two ends have first indices
that differ by 0 or 1):

    int dt2 t2^b e^(-t2) W_{K1,nu}(2 t2) int_0^t2 dt1 t1^a e^(-t1) M_{K2,mu}(2 t1)        (mu = +-nu)
      = 2^(1/2 + mu - P) Gamma(P + 1/2 + nu) Gamma(P + 1/2 - nu) / [Gamma(P + 1 - K1) (a + 3/2 + mu)]
        x 4F3[1/2 + mu + K2, P + 1/2 + nu, P + 1/2 - nu, a + 3/2 + mu; 1 + 2 mu, P + 1 - K1, a + 5/2 + mu; -1],

with P = a + b + 5/2 + mu. At energy Y = |k1 + r q'| and with the energy 1 + xi at V2 the moments follow by
scaling, as a double series in u = Y^2 - 1 and xi. The spin traces are polynomials in the loop momentum, and
after the azimuthal average each monomial is one position-space soft moment (ext/softvertex.py).
"""
from itertools import product

import mpmath as mp

from .blocks import e_series
from .fermion import ASFermion
from .softvertex import B_sigma, Grid
from .vectri import (K1, N1, N2, ONE, PV, USER, ZERO, SoftTable, mono, padd, pmul, reduce_poly)

I = mp.mpc(0, 1)
HALF = mp.mpf(1) / 2

__all__ = ["FermiTriangle", "FermiSoftTable", "ordered_moment"]


# ------------------------------------------------------------------------------ hard-line building blocks
def _f43m1(al, be):
    """4F3(al; be; -1), by mpmath (which continues the series) with a check on failure."""
    return mp.hyper(al, be, -1)


def ordered_moment(a, b, Km, mu, Kw, nu):
    """int_0^oo dt2 t2^b e^(-t2) W_{Kw,nu}(2 t2) int_0^t2 dt1 t1^a e^(-t1) M_{Km,mu}(2 t1),  mu = +-nu."""
    P = a + b + mp.mpf(5) / 2 + mu
    pre = (mp.mpf(2) ** (HALF + mu - P) * mp.gamma(P + HALF + nu) * mp.gamma(P + HALF - nu)
           / (mp.gamma(P + 1 - Kw) * (a + mp.mpf(3) / 2 + mu)))
    return pre * _f43m1([HALF + mu + Km, P + HALF + nu, P + HALF - nu, a + mp.mpf(3) / 2 + mu],
                        [1 + 2 * mu, P + 1 - Kw, a + mp.mpf(5) / 2 + mu])


def single_moment(x, K, nu):
    """int_0^oo dt t^x e^(-t) W_{K,nu}(2t) = 2^(-x-1) Gamma(x + 3/2 + nu) Gamma(x + 3/2 - nu)/Gamma(x + 2 - K)."""
    return mp.mpf(2) ** (-x - 1) * mp.gamma(x + mp.mpf(3) / 2 + nu) * mp.gamma(x + mp.mpf(3) / 2 - nu) / mp.gamma(x + 2 - K)


# ------------------------------------------------------------------------------ truncated double series
def _ser_mul(a, b, N):
    """product of {(j, l): c} series in (u, xi), truncated at j + l <= N"""
    out = {}
    for (j1, l1), v1 in a.items():
        for (j2, l2), v2 in b.items():
            if j1 + l1 + j2 + l2 <= N:
                k = (j1 + j2, l1 + l2)
                out[k] = out.get(k, 0) + v1 * v2
    return out


def _binom(beta, N):
    c = [mp.mpc(1)]
    for j in range(1, N + 1):
        c.append(c[-1] * (beta - j + 1) / j)
    return c


# ------------------------------------------------------------------------------ Pauli algebra with polynomial entries
def _vdot(v, w, nmax):
    return padd(*[(1, pmul(v[i], w[i], nmax)) for i in range(3) if v[i] and w[i]])


def _vcross(v, w, nmax):
    out = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        out.append(padd((1, pmul(v[j], w[k], nmax)), (-1, pmul(v[k], w[j], nmax))))
    return out


def _smul(A, B, nmax):
    """(s + v.sigma)(t + w.sigma) = (s t + v.w) + (s w + t v + i v x w).sigma"""
    s, v = A
    t, w = B
    cr = _vcross(v, w, nmax)
    return (padd((1, pmul(s, t, nmax)), (1, _vdot(v, w, nmax))),
            [padd((1, pmul(s, w[i], nmax)), (1, pmul(t, v[i], nmax)), (I, cr[i])) for i in range(3)])


def _scal(p):
    return (p, [ZERO, ZERO, ZERO])


def _vec(v, c=1):
    return (ZERO, [padd((c, x)) for x in v])


# ------------------------------------------------------------------------------ soft moments
KLAB = {(1, 1): "a", (1, -1): "b", (-1, 1): "c", (-1, -1): "d"}        # (sign Re K, sign Im K) -> label


class FermiSoftTable(SoftTable):
    """Soft moments with the first indices of a Dirac line, K = +-1/2 +- i kappa (labels a, b, c, d)."""

    def __init__(self, nu, kappa, grid=None, delta=None, nterms=None, cache=None):
        super().__init__(nu, kappa, grid, delta, nterms, cache)
        k = mp.mpf(kappa)
        self.K = {lab: sr * HALF + si * I * k for (sr, si), lab in KLAB.items()}

    def _meta(self):
        g = self.grid.key
        return ["fermi", mp.nstr(self.nu, 25), mp.nstr(self.K["a"], 25), mp.mp.dps, [g[1], list(g[2]), g[3]],
                mp.nstr(self.delta, 5), self.nterms]


# ------------------------------------------------------------------------------ the triangle
PIECES = {(1, -1): [(I, "v", "all")], (-1, 1): [(-I, "u", "all")],
          (1, 1): [(-I, "u", "12"), (I, "v", "21")], (-1, -1): [(I, "v", "12"), (-I, "u", "21")]}


class FermiTriangle:
    """Coefficients C_n(c) of the pair family of one fermion species (both orientations included).

    m, kappa: mass and axial chemical potential (H = 1). nmax: highest order in r.
    """

    def __init__(self, m, kappa, nmax=0, grid=None, table=None, cache=None, orientations=2, structures=None):
        self.structures = structures              # optional restriction to some (S1, S2, S3), e.g. [('1', '1', '1')]
        self.F = ASFermion(m, kappa)
        self.m, self.kappa, self.nu = self.F.m, self.F.kappa, self.F.nu
        self.nmax = nmax
        self.orientations = orientations
        self.table = table or FermiSoftTable(self.nu, self.kappa, grid, cache=cache).load()
        self._ord, self._q0, self._ser, self._e, self._geo = {}, {}, {}, {}, None

    # ---- spinor components: (c, K, s) with s the sign of the argument of W
    def col(self, kind, h, j):
        c, K = self.F.comps(kind, h)["L" if j == 0 else "R"]
        return mp.mpc(c), mp.mpc(K), self.F.sW(kind)

    def row(self, kind, h, i):
        c, K = self.F.comps(kind, h)["R" if i == 0 else "L"]
        return mp.conj(mp.mpc(c)), mp.conj(mp.mpc(K)), -self.F.sW(kind)

    def klabel(self, K):
        return KLAB[(1 if mp.re(K) > 0 else -1, 1 if mp.im(K) > 0 else -1)]

    def eser(self, K):
        key = self.klabel(K)
        if key not in self._e:
            self._e[key] = e_series(K, self.nu, self.nmax + 1)
        return self._e[key]

    def hard_end(self, comp, a):
        """Pair-branch coefficient and series index of a soft-line end at a hard vertex on branch a:
        component -> coefficient x sum_n e_n(Ke, nu) (2 y t)^(nu + n)."""
        c, K, s = comp
        co = c * B_sigma(K, self.nu, 1)
        if s == -a:
            return co, K
        return co * mp.exp(-I * mp.pi * a * (HALF + self.nu)), -K

    # ---- hard line at the folded point
    def _ordered(self, xg, xd, Km, sg, Kw):
        key = (mp.nstr(xg, 25), mp.nstr(xd, 25), self.klabel(Km), sg, self.klabel(Kw))
        if key not in self._ord:
            self._ord[key] = ordered_moment(xg, xd, Km, sg * self.nu, Kw, self.nu)
        return self._ord[key]

    def q0(self, a1, a2, hA, j1, i2, n1, n2):
        """int dt1 dt2 t1^(nu + n1) t2^(nu + n2) e^(-t1 - t2) [D_{a1 a2}]_{j1 i2}(t1, t2) at Y = 1, with the
        factors (2t)^(-1/2) of the two components and without the helicity projector."""
        key = (a1, a2, hA, j1, i2, n1, n2)
        if key in self._q0:
            return self._q0[key]
        nu = self.nu
        x1, x2 = nu + n1 - HALF, nu + n2 - HALF
        tot = mp.mpc(0)
        for fac, kind, region in PIECES[(a1, a2)]:
            c1, Ka, s1 = self.col(kind, hA, j1)
            c2, Kb, s2 = self.row(kind, hA, i2)
            if region == "all":
                assert s1 == -a1 and s2 == -a2
                v = single_moment(x1, Ka, nu) * single_moment(x2, Kb, nu)
            elif region == "12":                         # t1 < t2: growing at V1, decaying at V2
                assert s1 == a1 and s2 == -a2
                v = mp.fsum(B_sigma(Ka, nu, sg) * mp.exp(-I * mp.pi * a1 * (HALF + sg * nu))
                            * self._ordered(x1, x2, -Ka, sg, Kb) for sg in (1, -1))
            else:                                        # t2 < t1
                assert s2 == a2 and s1 == -a1
                v = mp.fsum(B_sigma(Kb, nu, sg) * mp.exp(-I * mp.pi * a2 * (HALF + sg * nu))
                            * self._ordered(x2, x1, -Kb, sg, Ka) for sg in (1, -1))
            tot += fac * c1 * c2 * v
        tot *= mp.exp(mp.pi * hA * self.kappa) / 2
        self._q0[key] = tot
        return tot

    def hard_series(self, a1, a2, hA, j1, i2, n1, n2, mY, mX, N):
        """{(j, l): F} with  Y^mY (1 + xi)^mX Q(Y, xi) = sum F u^j xi^l  (j + l <= N), where
        Q(Y, xi) = int dt1 dt2 t1^(nu+n1) t2^(nu+n2) e^(-t1 - (1+xi) t2) [D]_{j1 i2}(Y t1, Y t2)
                 = Y^(-2 nu - n1 - n2 - 2) sum_{p1 p2} w1^p1 w2^p2/(p1! p2!) Q0(n1 + p1, n2 + p2),
        w1 = 1 - 1/Y, w2 = 1 - (1 + xi)/Y."""
        key = (a1, a2, hA, j1, i2, n1, n2, mY, mX, N)
        if key in self._ser:
            return self._ser[key]
        g = self._geom(N)
        tot = {}
        for p1 in range(N + 1):
            for p2 in range(N + 1 - p1):
                q = self.q0(a1, a2, hA, j1, i2, n1 + p1, n2 + p2) / (mp.factorial(p1) * mp.factorial(p2))
                for k, v in g["ww"][(p1, p2)].items():
                    tot[k] = tot.get(k, 0) + q * v
        pre = {(j, 0): cj for j, cj in enumerate(_binom((mY - 2 * self.nu - n1 - n2 - 2) / 2, N))}
        pre = _ser_mul(pre, g["xpow"][mX], N)
        out = _ser_mul(pre, tot, N)
        self._ser[key] = out
        return out

    def _geom(self, N):
        """Series in (u, xi) of w1^p1 w2^p2 and of (1 + xi)^mX."""
        if self._geo is not None and self._geo["N"] >= N:
            return self._geo
        isq = {(j, 0): cj for j, cj in enumerate(_binom(-HALF, N))}              # (1 + u)^(-1/2)
        w1 = {k: -v for k, v in isq.items() if k != (0, 0)}                       # 1 - (1 + u)^(-1/2)
        w2 = dict(w1)
        for (j, l), v in isq.items():                                             # - xi (1 + u)^(-1/2)
            if j + 1 <= N:
                w2[(j, 1)] = w2.get((j, 1), 0) - v
        one = {(0, 0): mp.mpc(1)}
        p1s, p2s = [one], [one]
        for _ in range(N):
            p1s.append(_ser_mul(p1s[-1], w1, N))
            p2s.append(_ser_mul(p2s[-1], w2, N))
        ww = {(a, b): _ser_mul(p1s[a], p2s[b], N) for a in range(N + 1) for b in range(N + 1 - a)}
        xpow = {mX: {(0, l): cl for l, cl in enumerate(_binom(mX, N))} for mX in (0, -1, -2, -3)}
        self._geo = {"N": N, "ww": ww, "xpow": xpow}
        return self._geo

    # ---- geometry: spin traces as reduced polynomials
    def _traces(self):
        """T[(S1, S2, S3, pC, pB, pA)][j]: azimuthally averaged tr[S1 PiA S2 PiB S3 PiC] u^j, where each projector
        Pi_h(n) = 1/2 + (h/2) sigma.n is split into its two parts (p = 0, 1; the factors h are left out, and
        the hard part p = 1 carries 1/Y). Also xi^l as {(n, mc): coefficient}."""
        nmax = self.nmax
        half = _scal(padd((HALF, ONE)))
        zhat = [ZERO, ZERO, ONE]
        k2 = [padd((-1, K1[0])), ZERO, padd((-1, K1[2]), (-1, mono(n=1)))]          # k2 = -(k1hat + r zhat)
        pB = [padd((-1, x)) for x in N2]                                             # (q' - zhat)/y2
        S1 = {"1": _scal(ONE), "k": _vec(K1)}
        S2 = {"1": _scal(ONE), "k": _vec(k2)}
        S3 = {"1": _scal(ONE), "k": _vec(zhat)}
        PA = {0: half, 1: _vec(PV, HALF)}
        PB = {0: half, 1: _vec(pB, HALF)}
        PC = {0: half, 1: _vec(N1, HALF)}
        up = [ONE]
        for _ in range(nmax):
            up.append(pmul(up[-1], USER, nmax))
        T = {}
        for s1, s2, s3, c, b, a in product("1k", "1k", "1k", (0, 1), (0, 1), (0, 1)):
            M = _smul(_smul(_smul(_smul(_smul(S1[s1], PA[a], nmax), S2[s2], nmax), PB[b], nmax), S3[s3], nmax), PC[c], nmax)
            tr = padd((2, M[0]))
            T[(s1, s2, s3, c, b, a)] = [reduce_poly(pmul(tr, up[j], nmax)) if tr else {} for j in range(nmax + 1)]
        # xi = sqrt(1 + 2 r c + r^2) - 1
        v = padd((2, mono(n=1, mc=1)), (1, mono(n=2)))
        vp, xi = ONE, {}
        for j, cj in enumerate(_binom(HALF, nmax)):
            if j:
                vp = pmul(vp, v, nmax)
                xi = padd((1, xi), (cj, vp))
        xl = [ONE]
        for _ in range(nmax):
            xl.append(pmul(xl[-1], xi, nmax))
        xis = [{(k[0], k[5]): val for k, val in p.items()} for p in xl]
        return T, xis

    def vertex_terms(self, a, i, which):
        """Monomials of the block [i][1-i] of Gamma.F at a vertex on branch a, e^(-k t) stripped:
        [(coefficient, power of t, spin structure, power of the vertex energy)].
        Upper block (i = 0): mu_R (F0 - i G sigma.k); lower: mu_L (F0 + i G sigma.k);
        F0 = i a t/(2 k), G = (1 + k t)/(2 k^3). For V3 the common 1/r^2 is left out and sigma.k is sigma_z."""
        mu = self.F.muR if i == 0 else self.F.muL
        sg = -I if i == 0 else I
        if which == 3:
            return [(mu * I * a / 2, 1, "1", 0), (mu * sg / 2, 0, "k", 0), (mu * sg / 2, 1, "k", 0)]
        return [(mu * I * a / 2, 1, "1", -1), (mu * sg / 2, 0, "k", -3), (mu * sg / 2, 1, "k", -2)]

    def accumulate(self, verbose=False):
        """{(n, mc, ia, ib, q, K1, K2): coefficient} with C_n(c) = sum coefficient c^mc J(ia, ib, q, K1, K2)."""
        nmax, nu, ka = self.nmax, self.nu, self.kappa
        T, xis = self._traces()
        acc = {}
        two = mp.mpf(2)
        for a3 in (1, -1):
            kC, kB = ("v", "u") if a3 == 1 else ("u", "v")
            for h1, h2 in product((1, -1), repeat=2):
                hel = mp.exp(mp.pi * (h1 + h2) * ka)
                # ---- hard weights W[(i3-independent)]: sum over a1, a2, i1, i2, hA and the monomials of V1, V2
                Wt = {}
                for i1, i2 in product((0, 1), repeat=2):
                    rowC = self.row(kC, h1, i1)                       # hard end of line C at V1
                    colB = self.col(kB, h2, 1 - i2)                   # hard end of line B at V2
                    for a1, a2 in product((1, -1), repeat=2):
                        cC, KeC = self.hard_end(rowC, a1)
                        cB, KeB = self.hard_end(colB, a2)
                        eC, eB = self.eser(KeC), self.eser(KeB)
                        for (f1, p1, s1, _), (f2, p2, s2, mX) in product(self.vertex_terms(a1, i1, 1),
                                                                        self.vertex_terms(a2, i2, 2)):
                            base = a1 * a2 * cC * cB * f1 * f2
                            for n1 in range(nmax + 1):
                                for n2 in range(nmax + 1 - n1):
                                    w0 = base * eC[n1] * eB[n2] * two ** (2 * nu + n1 + n2)
                                    N = nmax - n1 - n2
                                    for hA in (1, -1):
                                        for pA in (0, 1):
                                            ser = self.hard_series(a1, a2, hA, 1 - i1, i2, n1 + p1, n2 + p2, -pA, mX, N)
                                            d = Wt.setdefault((s1, s2, pA, n1, n2), {})
                                            wf = w0 * (hA if pA else 1)
                                            for k, v in ser.items():
                                                d[k] = d.get(k, 0) + wf * v
                # ---- soft vertex
                for i3 in (0, 1):
                    cB3, KB3, sB3 = self.row(kB, h2, i3)
                    cC3, KC3, sC3 = self.col(kC, h1, 1 - i3)
                    assert sB3 == -a3 and sC3 == -a3
                    labC, labB = self.klabel(KC3), self.klabel(KB3)
                    for f3, q, s3, _ in self.vertex_terms(a3, i3, 3):
                        soft = a3 * hel * cB3 * cC3 * f3 / 2
                        for (s1, s2, pA, n1, n2), d in Wt.items():
                            if self.structures is not None and (s1, s2, s3) not in self.structures:
                                continue
                            for pC, pB in product((0, 1), repeat=2):
                                red = T[(s1, s2, s3, pC, pB, pA)]
                                hf = (h1 if pC else 1) * (h2 if pB else 1)
                                for (j, l), w in d.items():
                                    if w == 0 or not red[j]:
                                        continue
                                    cf = soft * hf * w
                                    for (nr, mc, da, db), v in red[j].items():
                                        for (nx, mcx), xv in xis[l].items():
                                            n = n1 + n2 + nr + nx
                                            if n > nmax:
                                                continue
                                            key = (n, mc + mcx, n1 + da, n2 + db, q, labC, labB)
                                            acc[key] = acc.get(key, 0) + cf * v * xv
                if verbose:
                    print(f"   a3 = {a3:+d}, h1 = {h1:+d}, h2 = {h2:+d}: {len(acc)} soft moments so far", flush=True)
        return {k: v for k, v in acc.items() if v != 0}

    def coefficients(self, acc=None, late=False):
        """{(n, mc): coefficient of r^(2 nu + n) c^mc} (both orientations)."""
        acc = acc if acc is not None else self.accumulate()
        parts = {}
        for (n, mc, ia, ib, q, K1_, K2_), cf in acc.items():
            J = self.table.late(ia, ib, q, K1_, K2_) if late else self.table(ia, ib, q, K1_, K2_)
            parts.setdefault((n, mc), []).append(cf * J)
        self.table.save()
        return {k: self.orientations * mp.fsum(v) for k, v in parts.items()}

    # ---- evaluation on families of shapes
    def at_angle(self, coef, c):
        """[C_0(c), ..., C_nmax(c)] at fixed c = k1hat . k3hat."""
        out = [mp.mpc(0)] * (self.nmax + 1)
        for (n, mc), v in coef.items():
            out[n] += v * mp.mpf(c) ** mc
        return out

    def isosceles(self, coef):
        """Coefficients of r^(2 nu + n) on k1 = k2, where c = -r/2."""
        out = [mp.mpc(0)] * (self.nmax + 1)
        for (n, mc), v in coef.items():
            if n + mc <= self.nmax:
                out[n + mc] += v * (-HALF) ** mc
        return out
