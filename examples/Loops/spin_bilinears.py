"""The universal factor for heavy bilinears with spin (Sec. 6.2): which operators of the basis of
Craig, Kumar, and McCune (arXiv:2401.10976, Table 5) it covers.

Eq. [eq:Univ] holds for a bilinear that is a de Sitter scalar, whatever the spin of its lines. For a tensor
bilinear T_rs contracted with two gradients of the inflaton the soft vertex is 2 grad^r phi_0 grad^s dphi T_rs.
With the hard end late, T_rs is a sum over boundary operators, and each reaches the soft vertex through a
bulk-to-boundary function fixed by the de Sitter isometries (X^2 = t^2 + rho^2, v = grad ln(t/X^2), v^2 = 1):

    S   scalar operator, trace part        g_rs (t/X^2)^D
    V   scalar operator, traceless part    (v_r v_s - g_rs/4)(t/X^2)^D
    T2  spin-2 operator                    t^-2 J_ri J_sj eps^ij (t/X^2)^D,   J_ri = delta_ri - 2 X_r X_i/X^2

After the Fourier transform every soft-vertex integral is a moment of K_(D-3/2) and closes in Gamma functions.
With U = U(D, r2) of Eq. [eq:Univ] and the late-time version defined by the leading late-time term of every line
function at the soft vertex (the massless mode is kept exactly), the exact over late-time ratios are

    S :  U                                                             (Eq. [eq:Univ])
    V :  -U (D^2 - 3D + 2 r2^2 - 14 r2 + 20)/[(D + 1)(D + 4 r2 - 8)]    (= U (3 - D)/(D + 1) at r2 = 2)
    T2:   U (D - r2 - 1)/[2 (D - 2)]                                    (= U (D - 3)/[2 (D - 2)] at r2 = 2)

(for T2 the component T_tt starts two powers of t later than T_tl and is absent from the late-time version)

and the exact soft vertices relative to the scalar one, V/S = (D^2 - 3D + 2 r2^2 - 14 r2 + 20)/[4 D (D + 1)].

Operators of the basis:
    O_8,1 = F F (grad phi)^2          scalar bilinear, D = 3 + 2 nu:  U                      (sm_templates.py)
    O_8,2 = F Ftilde (grad phi)^2     pseudoscalar bilinear, D = 4 + 2 nu:  U, family by family
    O_8,4 = |D H|^2 (grad phi)^2      scalar bilinear:  U
    O_8,5 = (D_m H)^dag D_n H grad^m phi grad^n phi     T = v v (t/X^2)^D0 = V + S/4 at leading order: the late-time
            soft vertex vanishes at r2 = 2 (a total derivative, as Box dphi = 0), and the exact one is (D0 - 1)/[2 (D0 + 1)]
            times the scalar one. Spin-2 operator at dimension D0 + 2 (two more powers of k3/k1): T2.
    O_8,6 = F_mr F_n^r grad^m phi grad^n phi            T = S + (4/3) V at leading order: exact/late = U 2 D0/(D0 + 1);
            the spin-2 operator has the leading dimension D0 = 3 + 2 nu: exact/late = nu U(D0, 2)/(1 + 2 nu).
            Both are checked in momentum space with Proca lines and all polarizations (--ninepair).

The script checks
  1. the closed forms against quadrature of the K-Bessel moments (trivially) and their simplified forms;
  2. the structures S, V, T2 in momentum space with scalar lines (position-space soft vertex of ext/softvertex.py);
  3. the tensor structure of F_mr F_n^r from the field strengths of the Proca line (b/a = 4/3);
  4. F Ftilde: the pair density 96 g'(M^2 g - 2 Z g') = w(Box)[g_nu g_(nu+1)], and (with --ninepair, 5 min) the
     nine-pair soft moment with E.B at the soft vertex against U(4 + 2 nu, r2) and the closed tree coefficient;
  5. (with --ninepair, 2 min) O_8,6 with Proca lines: the soft moment of F_mr F_n^r in momentum space, all
     polarizations, isotropic part and P_2 part, exact and late-time, against the closed forms.

Usage:  python spin_bilinears.py [--ninepair]        (about 4 min; 11 min with --ninepair)
"""
import sys
from itertools import product

import mpmath as mp

from ext.bubble import G0, Khat
from ext.pairloop import dens_FFdual, g_and_dg, w_FFdual
from ext.record import compare, load, report
from ext.softvertex import Grid, Leg, soft_J
from universal import U

mp.mp.dps = 30
I = mp.mpc(0, 1)
_H = mp.mpf(1) / 2


# ---------------------------------------------------------------------------- closed forms
def kap(a, mu):
    """int_0^oo dz z^(a-1) e^-z K_mu(z)."""
    return mp.sqrt(mp.pi) * 2 ** (-a) * mp.gamma(a + mu) * mp.gamma(a - mu) / mp.gamma(a + _H)


def late(a, mu):
    """The same with K_mu -> (1/2) Gamma(-mu) (z/2)^mu."""
    return mp.gamma(-mu) / 2 * 2 ** (-mu) * mp.gamma(a + mu)


def V_S(D, r2, exact=True):
    mu, a = D - 3 * _H, r2 - 3 * _H
    return -(kap if exact else late)(a, mu)


def V_V(D, r2, exact=True):
    mu, a = D - 3 * _H, r2 - 3 * _H
    if exact:
        k = kap
        return (-(mp.mpf(3) / 4 * k(a, mu) - 2 / D * k(a + 1, mu + 1) + k(a + 2, mu + 2) / (D * (D + 1)))
                + (k(a, mu) + k(a + 1, mu)) / D - (k(a + 1, mu + 1) + k(a + 2, mu + 1)) / (D * (D + 1)))
    return -mp.mpf(3) / 4 * late(a, mu) + (late(a, mu) + late(a + 1, mu)) / D


def V_T2(D, r2, exact=True):
    """int dz z^(r2 - 1/2) e^-z {K_mu [1 + (D - 1)(1 + z)/z^2] + (1 + z) K_(mu-1)/z}, mu = D - 3/2.
    The term K_mu alone is the component T_tt, which starts two powers of t later than T_tl; the late-time
    version (the leading late-time term of every line function) does not contain it."""
    mu, a = D - 3 * _H, r2 + _H
    if exact:
        k = kap
        return k(a, mu) + (D - 1) * (k(a - 2, mu) + k(a - 1, mu)) + k(a - 1, mu - 1) + k(a, mu - 1)
    l1 = lambda aa: mp.gamma(1 - mu) / 2 * 2 ** (1 - mu) * mp.gamma(aa + mu - 1)
    return (D - 1) * (late(a - 2, mu) + late(a - 1, mu)) + l1(a - 1) + l1(a)


def U_V(D, r2=2):
    return -U(D, r2) * (D ** 2 - 3 * D + 2 * r2 ** 2 - 14 * r2 + 20) / ((D + 1) * (D + 4 * r2 - 8))


def U_T2(D, r2=2):
    return U(D, r2) * (D - r2 - 1) / (2 * (D - 2))


def U_O86(D, r2=2):
    """Exact over late-time soft vertex of O_8,6 at leading order: T = S + (4/3) V."""
    vex = (D ** 2 - 3 * D + 2 * r2 ** 2 - 14 * r2 + 20) / (4 * D * (D + 1))
    vl = -(D + 4 * r2 - 8) / (4 * D)
    return U(D, r2) * (1 + mp.mpf(4) / 3 * vex) / (1 + mp.mpf(4) / 3 * vl)


# ------------------------------------------------------- momentum space, scalar lines (O_8,5)
class TensorVertex:
    """Soft moment of t^(r2-3) [d_t dphi T_tt + d_l dphi T_tl], T = d sigma d sigma, with two scalar lines of order nu
    whose hard ends are (y1 y2)^nu times a polynomial (k = 1).  s = sqrt(pi/2) y^(-1/2) t W_(0,nu)(2 y t),
    d_t s = sqrt(pi/2) y^(-1/2) [c W_(0,nu) - y t W_(0,nu+1)], c = 3/2 + nu."""

    def __init__(self, nu):
        self.nu, self.grid, self.legs = mp.mpmathify(nu), Grid(), {}

    def leg(self, A, dn):
        key = (mp.nstr(A, 20), dn)
        if key not in self.legs:
            self.legs[key] = Leg(A, 0, self.nu + dn)
        return self.legs[key]

    def J(self, a1, d1, a2, d2, q):
        return soft_J(self.leg(self.nu - _H + a1, d1), self.leg(self.nu - _H + a2, d2), q, self.grid)

    def moment(self, r2, poly=None):
        c, dq = 3 * _H + self.nu, r2 - 2
        tot = 0
        for (p1, p2), pc in (poly or {(0, 0): 1}).items():
            J = lambda a1, d1, a2, d2, q: self.J(a1 + p1, d1, a2 + p2, d2, q)
            T1 = -(c * c * J(0, 0, 0, 0, dq + 1) - c * (J(1, 1, 0, 0, dq + 2) + J(0, 0, 1, 1, dq + 2)) + J(1, 1, 1, 1, dq + 3))
            T2 = 0
            for sw in (False, True):
                for (b1, b2), cf in (((0, 0), 1), ((0, 2), 1), ((2, 0), -1)):
                    if sw:
                        b1, b2 = b2, b1
                    for qq in (dq + 1, dq + 2):
                        T2 += cf * (c * J(b1, 0, b2, 0, qq) - (J(b1, 0, b2 + 1, 1, qq + 1) if sw else J(b1 + 1, 1, b2, 0, qq + 1)))
            tot += pc * (T1 + T2 / 4)
        return mp.pi / 2 * tot


# traceless part of y1_i y2_j along k: [3 (y1.k)(y2.k) - y1.y2]/2 with y1 + y2 = -k, k = 1
SPIN2 = {(0, 0): mp.mpf(1) / 8, (2, 0): mp.mpf(1) / 4, (0, 2): mp.mpf(1) / 4, (4, 0): -mp.mpf(3) / 8, (0, 4): -mp.mpf(3) / 8, (2, 2): mp.mpf(3) / 4}


# -------------------------------------------------------------- structure of F_mr F_n^r
def structure_FF(nu, t=mp.mpf('0.7'), rho=(mp.mpf('0.3'), mp.mpf('-0.2'), mp.mpf('0.5')), t1=mp.mpf('1e-12')):
    """b/a in g^ab sum_i <F_ra F_ti><F_sb F_ti> = [a g_rs + b (v_r v_s - g_rs/4)](t/X^2)^(2 DA) for t1 -> 0, and the residual."""
    with mp.workdps(40):
        DA = nu + mp.mpf(3) / 2
        r2 = sum(v * v for v in rho)
        Z = (t * t + t1 * t1 + r2) / (2 * t * t1)
        S, T = 2 * DA * Z ** (-DA - 1), -DA * (DA + 1) * Z ** (-DA - 2)          # -2h', -h'' for h = Z^-DA
        ap = [1 / t1 - Z / t] + [v / (t * t1) for v in rho]
        a0 = [1 / t - Z / t1] + [-v / (t * t1) for v in rho]
        M = mp.matrix(4, 4)
        M[0, 0] = -1 / t ** 2 - 1 / t1 ** 2 + Z / (t * t1)
        for i in range(3):
            M[0, i + 1], M[i + 1, 0], M[i + 1, i + 1] = rho[i] / (t * t * t1), -rho[i] / (t * t1 * t1), -1 / (t * t1)
        FF = lambda r, s, m, n: (S * (M[r, m] * M[s, n] - M[r, n] * M[s, m])
                                 + T * (ap[r] * a0[m] * M[s, n] - ap[s] * a0[m] * M[r, n] - ap[r] * a0[n] * M[s, m] + ap[s] * a0[n] * M[r, m]))
        B = mp.matrix(4, 4)
        for r in range(4):
            for s in range(4):
                B[r, s] = sum(FF(r, a, 0, i) * FF(s, a, 0, i) for a in range(4) for i in (1, 2, 3)) * t * t
        X2 = t * t + r2
        v = [1 / t - 2 * t / X2] + [-2 * w / X2 for w in rho]
        b = B[0, 1] / (v[0] * v[1])
        a = (B[0, 0] - b * (v[0] ** 2 - 1 / (4 * t * t))) * t * t
        d = lambda r, s: (1 if r == s else 0) / (t * t)
        res = max(abs(B[r, s] - (a * d(r, s) + b * (v[r] * v[s] - d(r, s) / 4))) for r in range(4) for s in range(4)) / abs(B[0, 0])
        return b / a, res


# ------------------------------------------------- momentum space, Proca lines (O_8,6)
def _add(*ps):
    out = {}
    for p in ps:
        for key, c in p.items():
            out[key] = out.get(key, 0) + c
    return out


def _mul(*ps):
    out = {(0, 0): mp.mpf(1)}
    for p in ps:
        if not isinstance(p, dict):
            p = {(0, 0): p}
        new = {}
        for (a, b), c in out.items():
            for (a2, b2), c2 in p.items():
                new[(a + a2, b + b2)] = new.get((a + a2, b + b2), 0) + c * c2
        out = new
    return out


class ProcaTensorVertex:
    """Soft moment of t^(r2-3) [d_t dphi T_tt + d_l dphi T_tl], T_rs = F_ra F_s^a, with two Proca lines of order nu.
    The hard ends are electric fields E_i, E_j on the late-time branch (k = (0, 0, 1), y1 + y2 = -k), and the hard
    indices are contracted with delta_ij ('iso') or with (3 k_i k_j - delta_ij)/2 ('spin2').

    Line functions with a = t^(1/2) K_nu(y t) = sqrt(pi/2) y^(-1/2) W_(0,nu)(2 y t), common factors dropped:
        <E_i E_m> = PiT_im d_t a + (1/2 - nu) yh_i yh_m a/t,   <E_i A_m> = PiT_im a,   F_lm = i (y_l A_m - y_m A_l),
        d_t a -> y W_0 - W_1/t.
    The longitudinal polarization enters through the second term of <E_i E_m>. Scalars are reduced to Laurent
    polynomials in s1 = y1^2, s2 = y2^2 with y1.y2 = (1 - s1 - s2)/2."""
    FUNC = {'A': [(1, 0, 0, 0)], 'At': [(1, 0, 0, -1)], 'dA': [(1, 1, 0, 0), (-1, 0, 1, -1)]}     # (coef, y power, K, t power)

    def __init__(self, nu):
        self.nu, self.grid, self.legs, self.cache = mp.mpmathify(nu), Grid(), {}, {}
        h = _H
        one, i1, i2 = {(0, 0): mp.mpf(1)}, {(-1, 0): mp.mpf(1)}, {(0, -1): mp.mpf(1)}
        c = {(0, 0): h, (1, 0): -h, (0, 1): -h}                       # y1.y2
        k1 = {(0, 0): -h, (1, 0): -h, (0, 1): h}                      # y1.k
        k2 = {(0, 0): -h, (1, 0): h, (0, 1): -h}                      # y2.k
        E = {'dA': (mp.mpf(1), mp.mpf(-1)), 'At': (mp.mpf(0), h - self.nu)}          # E = alpha delta + beta yh yh
        sw = lambda p: {(b, a): v for (a, b), v in p.items()}
        self.terms = []                                              # (function 1, function 2, power of t, trace, k.M.k)
        for f1, (a1, b1) in E.items():
            for f2, (a2, b2) in E.items():                           # -t T_tt = -2 t^3 E1.E2
                tr = _add(_mul(3 * a1 * a2 + a1 * b2 + b1 * a2, one), _mul(b1 * b2, c, c, i1, i2))
                kk = _add(_mul(a1 * a2, one), _mul(a1 * b2, k2, k2, i2), _mul(b1 * a2, k1, k1, i1), _mul(b1 * b2, c, k1, k2, i1, i2))
                self.terms.append((f1, f2, 2, _mul(-2, tr), _mul(-2, kk)))
            # i k_l (1 + t) t^2 E1_m F2_lm:  M = -(y2.k) E1 PiT2 + (E1 y2)(PiT2 k)^T
            tr = _add(_mul(-1, k2, _add(_mul(2 * a1 + b1, one), _mul(-b1, c, c, i1, i2))), _mul(b1, c, i1, _add(k1, _mul(-1, c, k2, i2))))
            kk = _add(_mul(-1, k2, _add(_mul(a1, one), _mul(-a1, k2, k2, i2), _mul(b1, k1, k1, i1), _mul(-b1, c, k1, k2, i1, i2))),
                      _mul(_add(_mul(a1, k2), _mul(b1, c, k1, i1)), _add(one, _mul(-1, k2, k2, i2))))
            for P in (1, 2):
                self.terms.append((f1, 'A', P, tr, kk))
                self.terms.append(('A', f1, P, sw(tr), sw(kk)))

    def leg(self, A, K):
        key = (mp.nstr(A, 22), K)
        if key not in self.legs:
            self.legs[key] = Leg(A, K, self.nu)
        return self.legs[key]

    def moment(self, which, r2, exact=True, delta=0):
        nu, dq, tot = self.nu, r2 - 2, 0
        Bp = lambda K: mp.gamma(-2 * nu) / mp.gamma(_H - nu - K)
        base = nu + delta - _H
        for f1, f2, P, tr, kk in self.terms:
            mono = tr if which == 'iso' else _add(_mul(mp.mpf(3) / 2, kk), _mul(-_H, tr))
            for (a, b), cf in mono.items():
                if cf == 0:
                    continue
                for c1, ys1, K1, ts1 in self.FUNC[f1]:
                    for c2, ys2, K2, ts2 in self.FUNC[f2]:
                        e1, e2, Pt = 2 * a + ys1, 2 * b + ys2, P + dq + ts1 + ts2
                        if exact:
                            key = (e1, K1, e2, K2, Pt, mp.nstr(delta, 5))
                            if key not in self.cache:
                                self.cache[key] = soft_J(self.leg(base + e1, K1), self.leg(base + e2, K2), Pt + 1, self.grid)
                            tot += cf * c1 * c2 * self.cache[key]
                        elif not ((f1 == 'dA' and K1 == 0) or (f2 == 'dA' and K2 == 0)):      # W_K -> B_+(K) (2 y t)^(1/2 + nu)
                            from ext.blocks import C2hat
                            p1, p2 = base + e1 + _H + nu, base + e2 + _H + nu
                            tot += cf * c1 * c2 * Bp(K1) * Bp(K2) * mp.mpf(2) ** (1 + 2 * nu) * C2hat(p1 + 1, p2 + 1) / (2 * mp.pi) ** 2 * mp.gamma(Pt + 2 + 2 * nu)
        return tot

    def sym(self, which, r2, exact=True, d=mp.mpf(10) ** -10):
        return (self.moment(which, r2, exact, d) + self.moment(which, r2, exact, -d)) / 2


def proca_tensor(mu, r2s=(2, 3, 4)):
    """O_8,6 with Proca lines: (max |exact/late/closed ratio - 1| isotropic, the same for the P_2 part,
    spread of exact/closed form over r2 for the two parts)."""
    nu = I * mp.mpf(mu)
    O, D0 = ProcaTensorVertex(nu), 3 + 2 * nu
    f = {'iso': lambda r2: V_S(D0, r2) + mp.mpf(4) / 3 * V_V(D0, r2), 'spin2': lambda r2: V_T2(D0, r2)}
    R = {'iso': U_O86, 'spin2': U_T2}
    wr, wc = {}, {}
    for which in ('iso', 'spin2'):
        ex = {r2: O.sym(which, r2) for r2 in r2s}
        la = {r2: O.sym(which, r2, False) for r2 in r2s}
        wr[which] = max(abs(ex[r2] / la[r2] / R[which](D0, r2) - 1) for r2 in r2s)
        rat = [ex[r2] / f[which](r2) for r2 in r2s]
        wc[which] = max(abs(x / rat[0] - 1) for x in rat)
    return wr['iso'], wr['spin2'], wc['iso'], wc['spin2']


# ------------------------------------------------------------------------- F Ftilde
def C_dual(nu, t):
    """Tree coefficient of the F Ftilde pair density on the trees nu'_t = 2 nu + 5/2 + 2t."""
    from ext.bubble import Trees
    nup = 2 * nu + mp.mpf(5) / 2 + 2 * t
    w = w_FFdual(nu)
    P = nup ** 2 - mp.mpf(9) / 4
    return 4 * Trees(nu, nu + 1).G((1, 1), t) * (w[0] + w[1] * P + w[2] * P * P)


def ninepair_dual(mu, r2s=(2, 3)):
    """Nine-pair soft moment of E.B (gauge_loop.py machinery): (max |exact/late/U - 1|, max |exact/closed - 1|)."""
    import gauge_loop as gl
    from ext.blocks import C2hat
    with mp.workdps(45):
        G = gl.GaugeLoop(mu)
        nu, m = G.nu, G.m
        wT, wL = I * (0.5 + nu), m

        def soft(l1, l2):
            if l1 and l2:
                return [(-2 * I * l2, gl.DW, gl.W0, (1, 1, 0)), (-2 * I * l1, gl.W0, gl.DW, (1, 1, 0))]
            if l1:
                return [(l1 * m, gl.W0, gl.W0, (1, 0, -1))]
            if l2:
                return [(l2 * m, gl.W0, gl.W0, (0, 1, -1))]
            return []

        def hard(l1, l2):
            out = []
            if l2:
                out.append(((0, 1), (wT if l1 else wL) * I * l2))
            if l1:
                out.append(((1, 0), I * l1 * (wT if l2 else wL)))
            return out

        def J(l1, l2, mode, delta, dq):
            tot = 0
            eE = lambda l: -2 * I * (0.5 + nu) if l else 2 * m
            for (a, b), oc in gl.overlap(l1, l2).items():
                for (ha, hb), hc in hard(l1, l2):
                    b1, b2 = nu + delta - 0.5 + a + ha, nu + delta - 0.5 + b + hb
                    if mode == 'exact':
                        for coef, f1, f2, (dy1, dy2, dt) in soft(l1, l2):
                            for c1, s1, K1, t1 in f1:
                                for c2, s2, K2, t2 in f2:
                                    tot += oc * hc * coef * c1 * c2 * soft_J(G.leg(b1 + dy1 + s1, K1), G.leg(b2 + dy2 + s2, K2), 3 + dq + dt + t1 + t2, G.grid)
                    else:
                        terms = ([(eE(l1) * l2, (0.5 + nu, 1.5 + nu))] if l2 else []) + ([(l1 * eE(l2), (1.5 + nu, 0.5 + nu))] if l1 else [])
                        for cf, (e1, e2) in terms:
                            tot += oc * hc * cf * G.Bp ** 2 * mp.mpf(2) ** (2 * nu) * C2hat(b1 + e1 + 1, b2 + e2 + 1) / (2 * mp.pi) ** 2 * mp.gamma(3 + dq + 2 * nu)
            return tot

        d = mp.mpf(10) ** -12
        tot = lambda mode, dq: mp.fsum((J(l1, l2, mode, d, dq) + J(l1, l2, mode, -d, dq)) / 2 for l1, l2 in product(gl.POLS, gl.POLS))
        n0 = 2 * nu + mp.mpf(5) / 2
        wU = wC = 0
        for r2 in r2s:
            ex, la = tot('exact', r2 - 2), tot('late', r2 - 2)
            wU = max(wU, abs(ex / la / U(4 + 2 * nu, r2) - 1))
            closed = mp.gamma(1 + nu) ** 2 * 2 ** (2 * nu) / (16 * mp.pi ** 3) * C_dual(nu, 0) * 2 ** (-n0) * Khat(n0, r2 - mp.mpf(3) / 2, 1, 1) / mp.gamma(1 + n0)
            wC = max(wC, abs(abs(ex / closed) - 1))
        return wU, wC


# ------------------------------------------------------------------------------------ main
def main(ninepair=False):
    res, table = [], load()
    print("1. Closed forms of the soft vertex for the structures S, V, T2 (simplified forms against the K-Bessel moments)")
    worst = 0
    for D in (3 + 2 * I, 5 + 4 * I, mp.mpf('2.4')):
        for r2 in (2, 3, mp.mpf('3.5')):
            worst = max(worst, abs(V_S(D, r2) / V_S(D, r2, False) / U(D, r2) - 1), abs(V_V(D, r2) / V_V(D, r2, False) / U_V(D, r2) - 1),
                        abs(V_T2(D, r2) / V_T2(D, r2, False) / U_T2(D, r2) - 1),
                        abs(V_V(D, r2) / V_S(D, r2) / ((D ** 2 - 3 * D + 2 * r2 ** 2 - 14 * r2 + 20) / (4 * D * (D + 1))) - 1))
    print(f"   max deviation {mp.nstr(worst, 3)}")
    res.append(compare("spin/closed forms", worst, table, quiet=True))
    print("   |U|, |U_V|, |U_T2| at r2 = 2:   " + ";  ".join(
        f"D = 3 + 2i mu, mu = {m}: {mp.nstr(abs(U(3 + 2 * I * m, 2)), 4)}, {mp.nstr(abs(U_V(3 + 2 * I * m, 2)), 4)}, {mp.nstr(abs(U_T2(3 + 2 * I * m, 2)), 4)}" for m in (1, 2, 3)))
    for m in (1, 2, 3):
        D0 = 3 + 2 * I * m
        res.append(compare(f"spin/|U_T2| D=3+2i mu, mu={m}", abs(U_T2(D0, 2)), table))
        res.append(compare(f"spin/|U_O86| mu={m}", abs(U_O86(D0, 2)), table))
        late85 = abs(V_V(D0, 2, False) + V_S(D0, 2, False) / 4) / abs(V_S(D0, 2, False))
        res.append(compare(f"spin/O85 late-time soft vertex mu={m}", late85, table, quiet=True))
    print("   O_8,5 at leading order: late-time soft vertex over the scalar one at r2 = 2: "
          + mp.nstr(abs(V_V(3 + 2 * I, 2, False) + V_S(3 + 2 * I, 2, False) / 4) / abs(V_S(3 + 2 * I, 2, False)), 2)
          + ";  exact over the scalar one: (D0 - 1)/[2 (D0 + 1)]")
    print("   O_8,6 at leading order: |exact/late| = |U 2 D0/(D0 + 1)| = " + ", ".join(mp.nstr(abs(U_O86(3 + 2 * I * m, 2)), 4) for m in (1, 2, 3))
          + " at mu = 1, 2, 3;  spin-2 part: |nu U/(1 + 2 nu)| = " + ", ".join(mp.nstr(abs(U_T2(3 + 2 * I * m, 2)), 4) for m in (1, 2, 3)))

    print("2. Momentum space, scalar lines: soft moments of the tensor vertex against the closed forms")
    for mu in (1, 2):
        nu = I * mu
        D0 = 3 + 2 * nu
        O = TensorVertex(nu)
        f1 = lambda r2: V_V(D0, r2) + V_S(D0, r2) / 4
        f2 = lambda r2: V_T2(D0 + 2, r2)
        lead = {r2: O.moment(r2) for r2 in (2, 3, 4)}
        rat = [lead[r2] / f1(r2) for r2 in (2, 3, 4)]
        dev = max(abs(x / rat[0] - 1) for x in rat)
        print(f"   mu = {mu}: leading moment over (V + S/4) at r2 = 2, 3, 4 constant to {mp.nstr(dev, 3)};  |moment at r2 = 2|/|scalar-vertex moment| nonzero: "
              f"{mp.nstr(abs(f1(2) / V_S(D0, 2)), 6)} = |(D0 - 1)/(2 (D0 + 1))| = {mp.nstr(abs((D0 - 1) / (2 * (D0 + 1))), 6)}", flush=True)
        res.append(compare(f"spin/O85 leading moment mu={mu}", dev, table, quiet=True))
        Bm = {r2: O.moment(r2, SPIN2) for r2 in (2, 3, 4, 5)}
        A = mp.matrix([[f1(2), f2(2)], [f1(3), f2(3)]])
        a, b = mp.lu_solve(A, mp.matrix([Bm[2], Bm[3]]))
        dev = max(abs((a * f1(r2) + b * f2(r2)) / Bm[r2] - 1) for r2 in (4, 5))
        ctrl = max(abs(Bm[r2] / Bm[2] / (f1(r2) / f1(2)) - 1) for r2 in (3, 4, 5))
        print(f"           spin-2 structure: descendant and T2 fitted at r2 = 2, 3 predict r2 = 4, 5 to {mp.nstr(dev, 3)} (descendant alone misses by {mp.nstr(ctrl, 2)})", flush=True)
        res.append(compare(f"spin/T2 structure mu={mu}", dev, table, quiet=True))

    print("3. Tensor structure of F_mr F_n^r with the hard end late")
    worst, wres = 0, 0
    for nu in (mp.mpf('0.3'), I * 1, I * 2):
        r, rs = structure_FF(nu)
        worst, wres = max(worst, abs(r - mp.mpf(4) / 3)), max(wres, rs)
    print(f"   b/a = 4/3 to {mp.nstr(worst, 3)} (two-structure form holds to {mp.nstr(wres, 3)})")
    res.append(compare("spin/FF structure b/a = 4/3", max(worst, wres), table, quiet=True))

    print("4. F Ftilde: pair density against w(Box)[g_nu g_(nu+1)]")
    box = lambda f, Z: (Z * Z - 1) * mp.diff(f, Z, 2) + 4 * Z * mp.diff(f, Z)
    worst = 0
    for nu in (I * mp.mpf(1), mp.mpf('0.3')):
        for Z in (mp.mpf('1.4'), mp.mpf(5)):
            f = lambda z: g_and_dg(nu, z)[0] * g_and_dg(nu + 1, z)[0]
            b1 = lambda z: box(f, z)
            w = w_FFdual(nu)
            worst = max(worst, abs((w[0] * f(Z) + w[1] * b1(Z) + w[2] * box(b1, Z)) / dens_FFdual(nu, Z) - 1))
    print(f"   max deviation {mp.nstr(worst, 3)};  leading dimension 4 + 2 nu, |U(4 + 2 nu, 2)| = "
          + ", ".join(mp.nstr(abs(U(4 + 2 * I * m, 2)), 4) for m in (1, 2, 3)) + " at mu = 1, 2, 3")
    res.append(compare("spin/density FFdual vs w(Box)", worst, table, quiet=True))
    if ninepair:
        wU, wC = ninepair_dual('1')
        print(f"   nine pairs, mu = 1, r2 = 2, 3: |exact/late/U(4 + 2 nu, r2) - 1| = {mp.nstr(wU, 3)};  | |exact|/|closed tree coefficient| - 1| = {mp.nstr(wC, 3)}")
        res.append(compare("spin/FFdual nine pairs vs U", wU, table, quiet=True))
        res.append(compare("spin/FFdual nine pairs vs closed form", wC, table, quiet=True))
        print("5. O_8,6 with Proca lines, all polarizations: soft moment of F_mr F_n^r in momentum space, r2 = 2, 3, 4")
        for mu in ('1', '2'):
            a, b, c, d = proca_tensor(mu)
            print(f"   mu = {mu}: isotropic part, |exact/late/[U (1 + 4 v/3)/(1 + 4 v_late/3)] - 1| = {mp.nstr(a, 3)}, exact over (V_S + 4 V_V/3) constant to {mp.nstr(c, 3)};"
                  f"  P_2 part, |exact/late/U_T2 - 1| = {mp.nstr(b, 3)}, exact over V_T2 constant to {mp.nstr(d, 3)}", flush=True)
            res.append(compare(f"spin/O86 Proca lines mu={mu}", max(a, b, c, d), table, quiet=True))
    return report(res)


if __name__ == "__main__":
    sys.exit(main(ninepair='--ninepair' in sys.argv))
