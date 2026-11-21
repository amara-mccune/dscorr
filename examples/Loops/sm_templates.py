"""Complete templates for the Standard Model background: the W/Z loop and the top-quark loop at every shape
(Sec. 6.3), in the same form as the Higgs portal of higgs_portal.py.

Vertices (Chen, Wang, Xianyu, arXiv:1612.08122, Sec. 4, with X = (d phi)^2):
    L  >  2 f' phidot_0 dphidot O - f' (d_mu dphi)^2 O,
    O = F_mn F^mn (one real Proca field; W/Z loop),     O = m psibar psi (top quark; see the basis check below).
The bispectrum is a bubble with the hard vertex -[dG dG + k1.k2 G G] and the soft vertex of the leg k3, so
that at f' = phidot_0 = H = 1, exactly as for the Higgs portal,

    <dphi^3>' = I / (2 k1^3 k2^3 k3^3),
    I = -(k3^2 / 8 pi^2) N_O sum_patterns [ c4 P(4, 2) + c3 P(3, 2) + c2 P(2, 2) ],
    c4 = k1^2 k2^2 - k1.k2 k1 k2,   c3 = -k1.k2 k12,   c2 = -k1.k2,

with P the block of ext/pairloop.py at x1 = k12, x2 = k = k3. The pair densities are (ext/pairloop.py)

    <F^2 F^2>                   =  Ghat_F/(16 pi^4),    Ghat_F   = w_F(Box) g_nu^2,          N_O = 2,
    <psibar psi psibar psi>     = -Ghat_psi/(16 pi^4),  Ghat_psi = w_psi(Box) g_(nu+1/2)^2,  N_O = -2 m^2,

(for the Higgs portal <O O> = g^2/(2 16 pi^4), N_O = 1; kind 'dh' is O = (d h)^2/2 with the weight (P - 2 M^2)^2/4). The two patterns are nu -> +-nu; for heavy lines
they are complex conjugates. In the normalization of CWX's Eq. (4.31), -(1/4) f_A F^2, I is 1/16 of the above.

Everything is closed. The families are one sum over trees with the weights w(P_t), and the background is one
sum over the half-integer ladder with the weights w(zeta_m^2 - 9/4) and the digamma coefficients S_0(m + 1/2)
of the scalar bubble at the orders nu (W/Z) and nu + 1/2 (top quark). The background is quoted in the reference
scheme of ext/pairloop.py (the scalar MSbar bubble with w(Box) on the soft vertex) at mu_R = 2. Another scheme
adds a polynomial in zeta_m^2 of degree two (W/Z) or one (top), that is, three or two contact shapes.

The script
  1. checks the pair densities pointwise against w(Box) on the scalar product;
  2. checks the single sums against the invariant-distance form of Sec. 4.1 at five shapes (third difference in
     k^2, which is scheme independent and needs no subtraction);
  3. W/Z: reproduces the nine-pair soft moments of gauge_loop.py (orders 0 and 1 of the hard vertex) from the
     closed tree coefficients, and compares the late-time coefficient with C_A of CWX, Eq. (4.35) (not
     reproduced; the factor is printed), with C_DH of their Eq. (4.21) for |D_mu H|^2 as a control (reproduced);
  4. top quark: basis check (kinetic operator against m psibar psi), the series of dirac_pair.py --orders,
     and the spectral density of Aoki, Qin, Yamaguchi, and Zhu (arXiv:2605.28054);
  5. gives the pair signal against its leading squeezed term, and the complete I, at every shape.

Usage:  python sm_templates.py [--quick] [--json out.json]      (about 15 min; --quick 2 min)
"""
import json
import sys

import mpmath as mp

from ext.bubble import G0, G_ratio, Khat
from ext.pairloop import (PairLoop, dens_dphi2, dens_F2, dens_psi, dens_scalar, g_and_dg, polyval, third_difference, w_dphi2,
                          w_F2, w_psi, w_scalar, zside_third_difference)
from ext.record import compare, load, report
from universal import U

mp.mp.dps = 30
I = mp.mpc(0, 1)
_H = mp.mpf(1) / 2
_Q = mp.mpf(9) / 4


def coeffs(k1, k2, k3):
    """Hard vertex -[dG dG + k1.k2 G G] as {J: c_J} on the powers r1 = J."""
    d = (k3 ** 2 - k1 ** 2 - k2 ** 2) / 2
    return {4: k1 ** 2 * k2 ** 2 - d * k1 * k2, 3: -d * (k1 + k2), 2: -d}


def shape(R, delta=0, k12=1):
    """(k1, k2, k3) at R = 2 k3/k12 and delta = (k1 - k2)/k3, Eq. [eq:shapevars]."""
    k12, R, delta = mp.mpf(k12), mp.mpf(R), mp.mpf(delta)
    k3 = R * k12 / 2
    return (k12 + delta * k3) / 2, (k12 - delta * k3) / 2, k3


class SMLoop:
    """One loop of the Standard Model background: patterns (+, -) of a PairLoop with the hard vertex of Sec. 6.3."""

    def __init__(self, kind, mu, muR=2, mmax=300):
        self.kind, self.mu = kind, mp.mpmathify(mu)
        self.heavy = mp.im(self.mu) == 0 and kind != 'light'
        nu = I * self.mu
        self.nu = nu
        if kind == 'higgs':
            mk = lambda n: PairLoop(n, w_scalar(), muR, mmax)
            self.norm = mp.mpf(1)
        elif kind == 'dh':                                   # O = (d h)^2/2, the operator |D_mu H|^2 of CWX, Sec. 4.2
            mk = lambda n: PairLoop(n, w_dphi2(n), muR, mmax)
            self.norm = mp.mpf(1)
        elif kind == 'wz':
            mk = lambda n: PairLoop(n, w_F2(n), muR, mmax)
            self.norm = mp.mpf(2)
        elif kind == 'top':
            mk = lambda n: PairLoop(n + _H, w_psi(n), muR, mmax)
            self.norm = -2 * self.mu ** 2
        else:
            raise ValueError(kind)
        self.pat = {1: mk(nu), -1: mk(-nu)}
        self.Delta0 = {e: self.pat[e].nup(0) + mp.mpf(3) / 2 for e in (1, -1)}

    def families(self, eps, k1, k2, k3):
        return self.pat[eps].families(None, 2, k3, k1 + k2, k3, coef=coeffs(k1, k2, k3))

    def background(self, eps, k1, k2, k3):
        return self.pat[eps].background(None, 2, k3, k1 + k2, k3, coef=coeffs(k1, k2, k3))

    def calI(self, k1, k2, k3, parts=False):
        """I of the module docstring (reference scheme at mu_R): <dphi^3>' = I/(2 k1^3 k2^3 k3^3)."""
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        pre = -k3 ** 2 / (8 * mp.pi ** 2) * self.norm
        fam = {e: pre * self.families(e, k1, k2, k3) for e in (1, -1)}
        bg = pre * sum(self.background(e, k1, k2, k3) for e in (1, -1))
        tot = sum(fam.values()) + bg
        if parts:
            return {'I': mp.re(tot), 'families': fam, 'background': mp.re(bg), 'imag': mp.im(tot)}
        return mp.re(tot)

    def leading(self, eps, k1, k2, k3):
        """Leading squeezed term of the pattern eps: tree t = 0, hard vertex at k3 -> 0 on k1 = k2 = k12/2."""
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        L, x1 = self.pat[eps], k1 + k2
        c = coeffs(x1 / 2, x1 / 2, 0 * k3)
        nu = L.nup(0)
        hard = sum(cJ * (k3 / 2) ** nu * mp.gamma(J - mp.mpf(3) / 2 + nu) / (mp.gamma(1 + nu) * x1 ** (J - mp.mpf(3) / 2 + nu))
                   for J, cJ in c.items())
        return -k3 ** 2 / (8 * mp.pi ** 2) * self.norm * L.C(0) * Khat(nu, _H, k3, k3) * hard

    def ratio(self, R, delta=0, eps=1):
        """Pair signal of the pattern eps over its leading squeezed term."""
        k1, k2, k3 = shape(R, delta)
        pre = -k3 ** 2 / (8 * mp.pi ** 2) * self.norm
        return pre * self.families(eps, k1, k2, k3) / self.leading(eps, k1, k2, k3)

    def ten_percent(self, crit='complex', delta=0, eps=1):
        """Largest R for which the leading term reproduces the pair signal to 10% (complex amplitude or modulus)."""
        if crit == 'complex':
            f = lambda R: abs(self.ratio(R, delta, eps) - 1) - mp.mpf('0.1')
        else:
            f = lambda R: abs(abs(self.ratio(R, delta, eps)) - 1) - mp.mpf('0.1')
        lo, hi = mp.mpf('0.02'), mp.mpf('0.04')
        while f(hi) < 0:
            lo, hi = hi, hi * mp.mpf('1.25')
            if hi > mp.mpf('1.9'):
                return None
        for _ in range(34):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2


# ----------------------------------------------------------------------------------- checks
def box(f, Z):
    return (Z * Z - 1) * mp.diff(f, Z, 2) + 4 * Z * mp.diff(f, Z)


def check_densities():
    """Pointwise: the densities written through g and g' against w(Box) on the scalar product."""
    worst = {'F2': 0, 'psi': 0, 'dh': 0}
    for nu in (I * mp.mpf(1), I * mp.mpf(2), mp.mpf('0.3')):
        for Z in (mp.mpf('1.3'), mp.mpf(3), mp.mpf(12)):
            g2 = lambda z: dens_scalar(nu, z)
            b1 = lambda z: box(g2, z)
            w = w_F2(nu)
            worst['F2'] = max(worst['F2'], abs((w[0] * g2(Z) + w[1] * b1(Z) + w[2] * box(b1, Z)) / dens_F2(nu, Z) - 1))
            gp = lambda z: dens_scalar(nu + _H, z)
            w = w_psi(nu)
            worst['psi'] = max(worst['psi'], abs((w[0] * gp(Z) + w[1] * box(gp, Z)) / dens_psi(nu, Z) - 1))
            w = w_dphi2(nu)
            worst['dh'] = max(worst['dh'], abs(2 * (w[0] * g2(Z) + w[1] * b1(Z) + w[2] * box(b1, Z)) / dens_dphi2(nu, Z) - 1))
    return worst


def two_routes(kind, mu, us, mmax=800, r1s=(2, 3, 4)):
    """Third difference in k^2 of the block P(r1, 2) at x2 = 1, x1 = 1/u, k^2 = 1, 0.85, 0.70, 0.55: single sums
    against the invariant-distance integral of the density written through g and g'."""
    nu = I * mp.mpf(mu)
    if kind == 'wz':
        L, dens = PairLoop(nu, w_F2(nu), mmax=mmax), (lambda Z: dens_F2(nu, Z))
    elif kind == 'top':
        L, dens = PairLoop(nu + _H, w_psi(nu), mmax=mmax), (lambda Z: dens_psi(nu, Z))
    else:
        L, dens = PairLoop(nu, w_scalar(), mmax=mmax), (lambda Z: dens_scalar(nu, Z))
    K0, h, x2 = mp.mpf(1), mp.mpf('0.15'), mp.mpf(1)
    worst = 0
    for u in us:
        x1 = 1 / mp.mpf(u)
        for r1 in r1s:
            a = third_difference(lambda k: L.block(r1, 2, k, x1, x2), K0, h)
            b = zside_third_difference(dens, r1, x1, x2, K0, h)
            worst = max(worst, abs(a / b - 1))
        print(f"   {kind} mu = {mu}, k/x1 = {u}: max over r1 so far {mp.nstr(worst, 3)}", flush=True)
    return worst


def wz_soft_moment(mu, order, dq):
    """Closed form of the nine-pair soft moments of gauge_loop.py (GaugeLoop.order(order, dq)), r2 = 2 + dq.

    Order 0: -Gamma(1+nu)^2 2^(2nu)/(16 pi^3) C_0 2^(-nu'_0) Khat_0(r2 - 3/2)/Gamma(1 + nu'_0); order 1 adds the
    second term of Ihat_0 and the tree t = 1, with the sign of tau^2 = -t^2."""
    nu = I * mp.mpf(mu)
    w = w_F2(nu)
    n0, n1 = 2 * nu + mp.mpf(3) / 2, 2 * nu + mp.mpf(7) / 2
    g0 = G0(nu, nu)
    C0, C1 = 4 * g0 * polyval(w, n0 ** 2 - _Q), 4 * g0 * G_ratio(0, nu, nu) * polyval(w, n1 ** 2 - _Q)
    rho = _H + dq
    pre = -mp.gamma(1 + nu) ** 2 * 2 ** (2 * nu) / (16 * mp.pi ** 3)
    if order == 0:
        return pre * C0 * 2 ** (-n0) * Khat(n0, rho, 1, 1) / mp.gamma(1 + n0)
    return -pre * (C0 * Khat(n0, rho, 1, 1) / mp.gamma(2 + n0) + C1 * Khat(n1, rho, 1, 1) / mp.gamma(1 + n1)) * 2 ** (-n0 - 2)


def wz_E1_over_L1(mu):
    """|E_1/L_1| of gauge_loop.py --orders from the closed tree coefficients."""
    nu = I * mp.mpf(mu)
    w = w_F2(nu)
    n0, n1 = 2 * nu + mp.mpf(3) / 2, 2 * nu + mp.mpf(7) / 2
    c0 = polyval(w, n0 ** 2 - _Q) / mp.gamma(2 + n0)
    c1 = G_ratio(0, nu, nu) * polyval(w, n1 ** 2 - _Q) / mp.gamma(1 + n1)
    e0, e1 = c0 * Khat(n0, _H, 1, 1), c1 * Khat(n1, _H, 1, 1)
    return abs((e0 + e1) / (e0 / U(3 + 2 * nu, 2) + e1 / U(5 + 2 * nu, 2)))


def C_A(mu1):
    """Late-time coefficient of CWX for the gauge-boson loop, their Eq. (4.35)."""
    m = mp.mpmathify(mu1)
    return (-(2 - m) * (3 - 2 * m) * (23 - 6 * m) ** 2 * mp.gamma(m) ** 2 * mp.gamma(-4 + 4 * m) * mp.gamma(mp.mpf(5) / 2 - m) ** 2
            * mp.gamma(2 - 2 * m) ** 2 * mp.sin(mp.pi * m) ** 3 * mp.cos(mp.pi * m))


def C_DH(mu):
    """Late-time coefficient of CWX for |D_mu H|^2, their Eq. (4.21)."""
    m = mp.mpmathify(mu)
    return ((2 - m) * (3 - 2 * m) ** 3 * mp.cos(mp.pi * m) * mp.sin(mp.pi * m) ** 3 * mp.gamma(-4 + 4 * m) * mp.gamma(m) ** 2
            * mp.gamma(mp.mpf(5) / 2 - m) ** 2 * mp.gamma(2 - 2 * m) ** 2)


def cwx_factor(mu1):
    """Our late-time W/Z coefficient over CWX's Eq. (4.34): (8/9)(1/2 - mu_1)^2 (1/4 - mu_1^2)^2/(23 - 6 mu_1)^2."""
    m = mp.mpmathify(mu1)
    return mp.mpf(8) / 9 * (_H - m) ** 2 * (mp.mpf(1) / 4 - m ** 2) ** 2 / (23 - 6 * m) ** 2


def top_series(m, N, r2=2):
    """S(2N + 1) of dirac_pair.py at kappa = 0 in closed form:
    S(2N+1) = [16 pi^2 cosh^2(pi m)]^-1 sum_{t<=N} C_t Khat_t(r2 - 3/2) 2^(-nu'_0 - 2N)/[(N-t)! Gamma(1 + nu'_0 + N + t)]."""
    nu = I * mp.mpf(m)
    L = PairLoop(nu + _H, w_psi(nu), mmax=4)
    n0 = L.nup(0)
    tot = mp.fsum(L.C(t) * Khat(L.nup(t), r2 - mp.mpf(3) / 2, 1, 1) / (mp.factorial(N - t) * mp.gamma(1 + n0 + N + t)) for t in range(N + 1))
    return tot * mp.mpf(2) ** (-n0 - 2 * N) / (16 * mp.pi ** 2 * mp.cosh(mp.pi * mp.mpf(m)) ** 2)


def basis_check(m=mp.mpf('0.7')):
    """f(X) psibar i gamma.D psi against m f(X) psibar psi, in position space (rotated, psi = t^(-3/2) Psi).

    Returns (a) the residual of (gamma.d - m/t) S = 0 for the propagator S = (gamma.d + m/t)[P_+ G_+ + P_- G_-],
    which makes the pair density of the kinetic operator m^2 times that of psibar psi at separated points;
    (b) tr[gamma^a S(x, x') S(x', x)] relative to tr[S S], the correlator of the vector current with psibar psi;
    (c) tr[S S] (t t')^3 16 pi^4 against the density of ext/pairloop.py.
    """
    s0, s1, s2, s3 = mp.eye(2), mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, -I], [I, 0]]), mp.matrix([[1, 0], [0, -1]])

    def kron(a, b):
        out = mp.matrix(4, 4)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        out[2 * i + k, 2 * j + l] = a[i, j] * b[k, l]
        return out

    gam = [kron(s3, s0), kron(s1, s1), kron(s1, s2), kron(s1, s3)]
    Pp, Pm = (mp.eye(4) + gam[0]) / 2, (mp.eye(4) - gam[0]) / 2

    def X(x, xp):
        t, tp = x[0], xp[0]
        Z = (t * t + tp * tp + sum((x[i] - xp[i]) ** 2 for i in (1, 2, 3))) / (2 * t * tp)
        return -(Pp * g_and_dg(m + _H, Z)[0] + Pm * g_and_dg(m - _H, Z)[0]) / (t * tp)

    def dmat(F, x, a):
        def comp(i, j):
            def f(s):
                y = list(x)
                y[a] = s
                return F(y)[i, j]
            return mp.diff(f, x[a])
        return mp.matrix([[comp(i, j) for j in range(4)] for i in range(4)])

    def S(x, xp):
        tot = (m / x[0]) * X(x, xp)
        for a in range(4):
            tot += gam[a] * dmat(lambda y: X(y, xp), x, a)
        return tot

    x, xp = [mp.mpf('0.9'), mp.mpf('0.3'), mp.mpf('-0.2'), mp.mpf('0.5')], [mp.mpf('0.4'), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    with mp.workdps(40):
        Sx = S(x, xp)
        DS = -(m / x[0]) * Sx
        for a in range(4):
            DS += gam[a] * dmat(lambda y: S(y, xp), x, a)
        size = max(abs(v) for v in Sx)
        res_eom = max(abs(v) for v in DS) / size * x[0]
        Sb = S(xp, x)
        tr = lambda A: sum(A[i, i] for i in range(4))
        trSS = tr(Sx * Sb)
        res_cur = max(abs(tr(gam[a] * Sx * Sb)) for a in range(4)) / abs(trSS)
        Z = (x[0] ** 2 + xp[0] ** 2 + sum((x[i] - xp[i]) ** 2 for i in (1, 2, 3))) / (2 * x[0] * xp[0])
        res_dens = abs(trSS * (x[0] * xp[0]) ** 3 / dens_psi(m, Z) - 1)
    return res_eom, res_cur, res_dens


# ------------------------------------------------------------------------------------- main
def main(quick=False, json_path=None):
    res, data = [], {}
    table = load()
    num = lambda v: mp.mpc(mp.mpf(v[0]), mp.mpf(v[1]))

    print("1. Pair densities: written through g, g' against w(Box) on the scalar product (three orders, three Z)")
    w = check_densities()
    print(f"   F^2: {mp.nstr(w['F2'], 3)};  psibar psi: {mp.nstr(w['psi'], 3)};  (d h)^2: {mp.nstr(w['dh'], 3)}")
    res += [compare("sm/density F2 vs w(Box)", w['F2'], table, quiet=True), compare("sm/density psi vs w(Box)", w['psi'], table, quiet=True),
            compare("sm/density dh vs w(Box)", w['dh'], table, quiet=True)]

    print("2. Direct evaluation: single sums against the invariant-distance form (third difference in k^2)")
    us = ('0.2',) if quick else ('0.05', '0.2', '0.5', '0.7', '0.9')
    for kind, mus in (('wz', ('1',) if quick else ('1', '2', '4')), ('top', ('1',) if quick else ('1', '2'))):
        for mu in mus:
            worst = two_routes(kind, mu, us)
            res.append(compare(f"sm/two routes {kind} mu={mu}", worst, table, quiet=True))

    print("3. W/Z loop: nine-pair soft moments of gauge_loop.py from the closed tree coefficients")
    with mp.workdps(45):
        for mu in ('1', '2'):
            worst = 0
            for order, dqs in ((0, (0, 1, 2)), (1, (0, 1, 2, 3))):
                for dq in dqs:
                    ref = num(table[f"gauge_ref/order {order} mu={mu} r2={2 + dq}"]['value'])
                    worst = max(worst, abs(wz_soft_moment(mu, order, dq) / ref - 1))
            print(f"   mu = {mu}: orders 0 (r2 = 2, 3, 4) and 1 (r2 = 2..5), max |closed/gauge_loop - 1| = {mp.nstr(worst, 3)}")
            res.append(compare(f"sm/wz nine-pair moments mu={mu}", worst, table, quiet=True))
            res.append(compare(f"gauge_orders/|E_1/L_1| mu={mu}", wz_E1_over_L1(mu), table))
    worst = 0
    for mu in (_H, mp.mpf(1), mp.mpf(2), mp.mpf(4)):
        A = SMLoop('wz', mu, mmax=4)
        kS, kL = mp.mpf(1), mp.mpf(10) ** -6
        for eps in (1, -1):
            nu = eps * A.nu
            ours = A.leading(eps, kS, kS, kL) / 16 / U(3 + 2 * nu, 2) / (2 * kS ** 6 * kL ** 3)
            cwx = 27 / (mp.pi ** 4 * (mp.mpf(1) / 4 - nu ** 2) ** 2 * (2 * kS) ** 6) * C_A(-nu) * (kL / (2 * kS)) ** (2 * nu)
            worst = max(worst, abs(ours / cwx / cwx_factor(-nu) - 1))
    wdh = 0
    for nu_ in (I * _H, I * 1, I * 2, mp.mpf('1.4')):
        A = SMLoop('dh', -I * nu_, mmax=4)
        kS, kL = mp.mpf(1), mp.mpf(10) ** -6
        for eps in (1, -1):
            nu = eps * A.nu
            ours = A.leading(eps, kS, kS, kL) / U(3 + 2 * nu, 2) / (2 * kS ** 6 * kL ** 3)
            cwx = -1 / (32 * mp.pi ** 4 * kS ** 6) * C_DH(-nu) * (kL / (2 * kS)) ** (2 * nu)
            wdh = max(wdh, abs(ours / cwx - 1))
    print(f"   control, |D_mu H|^2: late-time leading term against C_DH of CWX, Eq. (4.21): max |ours/CWX - 1| = {mp.nstr(wdh, 3)}")
    res.append(compare("sm/dh late-time term vs C_DH", wdh, table, quiet=True))
    print(f"   late-time leading term over CWX's Eq. (4.34): equal to (8/9)(1/2 - mu_1)^2 (1/4 - mu_1^2)^2/(23 - 6 mu_1)^2 to {mp.nstr(worst, 3)}")
    print("   (CWX's C_A is not reproduced; modulus of this factor at mu = 1/2, 1, 2, 4: "
          + ", ".join(mp.nstr(abs(cwx_factor(I * m)), 4) for m in (_H, 1, 2, 4)) + ")")
    res.append(compare("sm/wz late-time term vs CWX times factor", worst, table, quiet=True))
    for m_ in ('1', '2', '4'):
        res.append(compare(f"sm/wz |ours/CWX| late-time mu={m_}", abs(cwx_factor(I * mp.mpf(m_))), table))

    print("4. Top quark")
    e, c, d = basis_check()
    print(f"   basis check: (gamma.d - m/t) S = 0 to {mp.nstr(e, 3)};  tr[gamma^a S S]/tr[S S] = {mp.nstr(c, 3)};  tr[S S] against the density: {mp.nstr(d, 3)}")
    res += [compare("sm/top equation of motion of S", e, table, quiet=True), compare("sm/top vector current", c, table, quiet=True),
            compare("sm/top trace vs density", d, table, quiet=True)]
    for m in ('1', '2'):
        worst = 0
        for N in range(4):
            ref = num(table[f"dirac_ref/S({2 * N + 1}) kappa=0 m={m}"]['value'])
            worst = max(worst, abs(top_series(m, N) / ref - 1))
        for r2 in (3, 4):
            for N in (0, 1):
                ref = num(table[f"dirac_ref/S({2 * N + 1}) kappa=0 m={m} r2={r2}"]['value'])
                worst = max(worst, abs(top_series(m, N, r2) / ref - 1))
        print(f"   m = {m}: S(1), S(3), S(5), S(7) of dirac_pair.py (r2 = 2; S(1), S(3) also at r2 = 3, 4): max |closed/dirac_pair - 1| = {mp.nstr(worst, 3)}")
        res.append(compare(f"sm/top series m={m}", worst, table, quiet=True))
        res.append(compare(f"dirac_orders/|S(1)| kappa=0 m={m}", abs(top_series(m, 0)), table))
        res.append(compare(f"dirac_orders/|S(3)/S(1)| kappa=0 m={m}", abs(top_series(m, 1) / top_series(m, 0)), table))
    # Aoki et al., Eqs. (3.15), (3.35): fermion tree coefficient over the scalar one at orders shifted by 1/2 is -4 (3 + 2k)(k + 1 + 2 nu)
    worst = 0
    for m in (1, 2):
        nu = I * mp.mpf(m)
        L = PairLoop(nu + _H, w_psi(nu), mmax=4)
        for k in range(6):
            worst = max(worst, abs(L.wt(L.nup(k)) / (-4 * (3 + 2 * k) * (k + 1 + 2 * nu)) - 1))
    print(f"   tree weights against the spectral density of Aoki et al., -4 (3 + 2k)(k + 1 + i M): {mp.nstr(worst, 3)}")
    res.append(compare("sm/top weights vs Aoki et al.", worst, table, quiet=True))

    print("5. The pair signal over its leading squeezed term, and the complete I (reference scheme, mu_R = 2)")
    cases = [('wz', '1'), ('top', '1')] if quick else [('wz', '1'), ('wz', '2'), ('wz', '4'), ('top', '1'), ('top', '2'), ('higgs', '1')]
    data['signal'] = {}
    for kind, mu in cases:
        A = SMLoop(kind, mu)
        key = f"{kind} mu={mu}"
        rs = {R: A.ratio(R) for R in ('0.3', '0.5', '1')}
        Rc, Rm = A.ten_percent(), A.ten_percent('modulus')
        print(f"   {key:11s} |signal/leading| at R = 0.3, 0.5, 1: " + ", ".join(f"{mp.nstr(abs(r), 5)} (arg {mp.nstr(mp.arg(r), 3)})" for r in rs.values())
              + f";  10%: R < {mp.nstr(Rc, 3)} (complex), {mp.nstr(Rm, 3) if Rm else '> 1.9'} (modulus)", flush=True)
        data['signal'][key] = dict(ratio={R: [float(mp.re(r)), float(mp.im(r))] for R, r in rs.items()},
                                   R10_complex=float(Rc), R10_modulus=float(Rm) if Rm else None,
                                   Delta0=[float(mp.re(A.Delta0[1])), float(mp.im(A.Delta0[1]))],
                                   absU=float(abs(U(A.Delta0[1], 2))))
        for R in rs:
            res.append(compare(f"sm/|signal/leading| {key} R={R}", abs(rs[R]), table))
        res.append(compare(f"sm/R10 complex {key}", Rc, table))
        if kind == 'top' and not quick:
            for dl in ('0.5', '1'):
                r = A.ratio('0.5', dl)
                Rd = A.ten_percent(delta=dl)
                print(f"               delta = {dl}: |signal/leading| at R = 0.5: {mp.nstr(abs(r), 5)} (arg {mp.nstr(mp.arg(r), 3)});  10%: R < {mp.nstr(Rd, 3)}")
                data['signal'][key][f"delta={dl}"] = dict(ratio_R05=[float(mp.re(r)), float(mp.im(r))], R10_complex=float(Rd))
    if not quick:
        A = SMLoop('wz', '1')
        for dl in ('0.5', '1'):
            r = A.ratio('0.5', dl)
            Rd = A.ten_percent(delta=dl)
            print(f"   wz mu=1     delta = {dl}: |signal/leading| at R = 0.5: {mp.nstr(abs(r), 5)} (arg {mp.nstr(mp.arg(r), 3)});  10%: R < {mp.nstr(Rd, 3)}")
            data['signal']['wz mu=1'][f"delta={dl}"] = dict(ratio_R05=[float(mp.re(r)), float(mp.im(r))], R10_complex=float(Rd))
    Rs = ('0.1', '1') if quick else ('0.04', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1', '1.2', '1.4', '1.6', '1.8')
    data['shape'] = {}
    for kind, mu in cases:
        A = SMLoop(kind, mu)
        key = f"{kind} mu={mu}"
        print(f"   {key}:   R     |signal/leading|   arg        I (complete)     families       background")
        rows = []
        for R in Rs:
            k1, k2, k3 = shape(R)
            P = A.calI(k1, k2, k3, parts=True)
            r = P['families'][1] / A.leading(1, k1, k2, k3)
            fam = mp.re(sum(P['families'].values()))
            assert abs(P['imag']) < mp.mpf(10) ** -18 * (abs(fam) + abs(P['background']))
            rows.append(dict(R=float(R), absR=float(abs(r)), argR=float(mp.arg(r)), I=float(P['I']), families=float(fam), background=float(P['background'])))
            print(f"              {R:>5s}   {mp.nstr(abs(r), 6):>9s}   {mp.nstr(mp.arg(r), 4):>9s}   {mp.nstr(P['I'], 8):>14s}  {mp.nstr(fam, 6):>12s}  {mp.nstr(P['background'], 6):>12s}", flush=True)
        data['shape'][key] = rows
        if '1' in Rs:
            res.append(compare(f"sm/I {key} R=1", rows[Rs.index('1')]['I'], table))
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=1)
    return report(res)


if __name__ == "__main__":
    jp = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    sys.exit(main(quick='--quick' in sys.argv, json_path=jp))
