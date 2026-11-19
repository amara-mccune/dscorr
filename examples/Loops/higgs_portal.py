"""The Higgs-portal loop f_H(X) H^dagger H at every shape (Sec. 6.3), and the universal factor order by order.

Vertices (Chen, Wang, Xianyu, arXiv:1612.08122, Eqs. (4.2)-(4.4), with X = (d phi)^2):
    L  >  2 f_H' phidot_0 dphidot O - f_H' (d_mu dphi)^2 O,      O = H^dagger H,   <O O> = G_h^2 / 2.
The bispectrum is a bubble with the hard vertex -[dG dG + k1.k2 G G] (legs k1, k2) and the soft vertex of
the leg k3. On the rotated contour the soft vertex has r2 = 2 and the hard vertex r1 = 4 (time derivatives)
and r1 = 2, 3, 4 (gradients), so that, at f_H' = phidot_0 = H = 1,

    <dphi^3>' = I / (2 k1^3 k2^3 k3^3),
    I = -(k3^2 / 8 pi^2) sum_{eps=+-} [ c4 P_eps(4, 2) + c3 P_eps(3, 2) + c2 P_eps(2, 2) ],
    c4 = k1^2 k2^2 - k1.k2 k1 k2,   c3 = -k1.k2 k12,   c2 = -k1.k2,

where P_eps is the block of ext/bubble.py at equal orders (eps nu, eps nu), x1 = k12, x2 = k = k3 (at equal
masses the mixed patterns have zero weight). The loop is logarithmically divergent; we quote MSbar at mu_R.

Everything is closed. The families are one sum over trees of Gamma functions times a 2F1 in (k3/k12)^2. The
background sits on the ladder zeta_m = m + 1/2, where the functions L_m are elementary (I of half-integer
order, C_m^(1/2)(1) = 1) and the tree sums S_0(m + 1/2) are digamma functions [Eqs. (4.7)-(4.10)].

The script
  1. checks the block against the invariant-distance form of Sec. 4.1 (the direct evaluation) at five shapes;
  2. checks that the late-time version of the leading term is C_H of CWX [their Eq. (4.15)];
  3. tests the universal factor of Sec. 6.2 order by order in k3/k12;
  4. gives the full shape against the late-time template.

Usage:  python higgs_portal.py [--quick] [--json out.json]        (about 5 min; --quick 30 s)
"""
import json
import sys

import mpmath as mp

from ext.bubble import Ihat, Khat, Trees
from ext.record import compare, report
from universal import U

mp.mp.dps = 30
I = mp.mpc(0, 1)
_HALF = mp.mpf(1) / 2


def S0_lattice_equal(nu, mmax):
    """S_0(m + 1/2; nu, nu) for m <= mmax: Eqs. (4.10), (4.9) at m = 0, 1, 2, then the recurrence (4.8)."""
    nu = mp.mpmathify(nu)
    om = nu ** 2 - mp.mpf(1) / 4
    psi = mp.digamma(nu + _HALF)
    Psi = mp.euler + psi
    cK = mp.euler / 2 + mp.log(2) / 2 - mp.mpf(1) / 4
    out = [-(psi + 1 - mp.log(2)) / 2,
           -psi / 2 - om / (4 * nu) * mp.polygamma(1, nu + _HALF) + mp.log(2) / 2 - mp.mpf(1) / 4,
           -psi / 2 - (om - 1) / (4 * om) + mp.log(2) / 2 - mp.mpf(1) / 4]
    for m in range(1, mmax - 1):
        z = _HALF + m
        w = (2 * z + 1) * (2 * z + 3)
        p1 = (z + 1.5) * (z + 2.5) * (z + 0.5) ** 2 * (z + 0.5 + 2 * nu) * (z + 0.5 - 2 * nu)
        p0 = -(z - 0.5) * (z + 0.5) * (z + 1.5) ** 2 * (z + 1.5 + 2 * nu) * (z + 1.5 - 2 * nu)
        rhs = om * (z + 1) * w * Psi - (z + 1) * w * (w - 8 * om) / 16 + cK * (p1 + p0)
        out.append((rhs - p0 * out[m]) / p1)
    return out[:mmax + 1]


def C_H(mu):
    """Late-time coefficient of CWX, their Eq. (4.15)."""
    mu = mp.mpmathify(mu)
    return ((2 - mu) * (3 - 2 * mu) * mp.cos(mp.pi * mu) * mp.sin(mp.pi * mu) ** 3 * mp.gamma(-4 + 4 * mu)
            * mp.gamma(mu) ** 2 * mp.gamma(mp.mpf(3) / 2 - mu) ** 2 * mp.gamma(2 - 2 * mu) ** 2)


class HiggsPortal:
    """Bubble of two lines of equal order nu (nu = i mu for a heavy Higgs, real for a light one)."""

    def __init__(self, nu, muR=2, mmax=400):
        self.nu = mp.mpmathify(nu)
        self.heavy = mp.re(self.nu) == 0
        self.muR = mp.mpf(muR)
        self.tr = Trees(self.nu, self.nu)
        with mp.workdps(mp.mp.dps + 15):
            self.S0 = {(1, 1): S0_lattice_equal(self.nu, mmax), (-1, -1): S0_lattice_equal(-self.nu, mmax)}

    @staticmethod
    def coeffs(k1, k2, k3, part='all'):
        d = (k3 ** 2 - k1 ** 2 - k2 ** 2) / 2                      # k1.k2
        if part == 'time':
            return {4: k1 ** 2 * k2 ** 2}
        if part == 'gradient':
            return {4: -d * k1 * k2, 3: -d * (k1 + k2), 2: -d}
        return {4: k1 ** 2 * k2 ** 2 - d * k1 * k2, 3: -d * (k1 + k2), 2: -d}

    def nup(self, eps, t):
        return 2 * eps * self.nu + mp.mpf(3) / 2 + 2 * t

    def families(self, eps, k1, k2, k3, part='all', tol=None, terms=False):
        """sum_t 4 G_t Khat_t(1/2, k3) sum_J c_J Ihat_t(J - 3/2, k12): the nonlocal and local signal of one pattern."""
        tol = tol or mp.mpf(10) ** (-mp.mp.dps + 6)
        c, x1, k = self.coeffs(k1, k2, k3, part), k1 + k2, k3
        e = (eps, eps)
        tot, small, big, lst = mp.mpc(0), 0, mp.mpf(0), []
        for t in range(100000):
            nu = self.nup(eps, t)
            term = 4 * self.tr.G(e, t) * Khat(nu, _HALF, k, k) * sum(cJ * Ihat(nu, J - mp.mpf(3) / 2, x1, k) for J, cJ in c.items())
            tot += term
            lst.append(term)
            big = max(big, abs(tot))
            small = small + 1 if abs(term) < tol * big else 0
            if small >= 3:
                break
        return (tot, lst) if terms else tot

    def background(self, eps, k1, k2, k3, part='all', tol=None):
        """sum_m L_m [S_0(m + 1/2) + 1/2 + (1/2) ln(mu_R/2)], with L_m elementary."""
        tol = tol or mp.mpf(10) ** (-mp.mp.dps + 6)
        c, x1, k = self.coeffs(k1, k2, k3, part), k1 + k2, k3
        pre = mp.sqrt(mp.pi) * (2 / k) ** _HALF
        cst = _HALF + mp.log(self.muR / 2) / 2
        S0 = self.S0[(eps, eps)]
        tot, small = mp.mpc(0), 0
        for m in range(len(S0)):
            z = _HALF + m
            L = pre * (-1) ** m * z * sum(cJ * Ihat(z, J - mp.mpf(3) / 2, x1, k) for J, cJ in c.items())
            term = L * (S0[m] + cst)
            tot += term
            small = small + 1 if abs(term) < tol * abs(tot) else 0
            if small >= 3:
                return tot
        raise RuntimeError("background ladder did not converge; raise mmax")

    def calI(self, k1, k2, k3, part='all', parts=False):
        """The integral I of CWX's Eq. (4.4) in MSbar: <dphi^3>' = I/(2 k1^3 k2^3 k3^3)."""
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        pre = -k3 ** 2 / (8 * mp.pi ** 2)
        fam = {e: pre * self.families(e, k1, k2, k3, part) for e in (1, -1)}
        bg = pre * sum(self.background(e, k1, k2, k3, part) for e in (1, -1))
        tot = sum(fam.values()) + bg
        if parts:
            return {'I': mp.re(tot), 'families': fam, 'background': mp.re(bg), 'imag': mp.im(tot)}
        return mp.re(tot)

    def leading(self, eps, k1, k2, k3):
        """Leading squeezed term of the pattern eps (tree t = 0, hard vertex at lowest order), exact soft vertex."""
        k1, k2, k3 = mp.mpf(k1), mp.mpf(k2), mp.mpf(k3)
        c = self.coeffs(k1, k2, 0 * k3)
        nu, x1 = self.nup(eps, 0), k1 + k2
        hard = sum(cJ * (k3 / 2) ** nu * mp.gamma(J - mp.mpf(3) / 2 + nu) / (mp.gamma(1 + nu) * x1 ** (J - mp.mpf(3) / 2 + nu))
                   for J, cJ in c.items())
        return -k3 ** 2 / (8 * mp.pi ** 2) * 4 * self.tr.G((eps, eps), 0) * Khat(nu, _HALF, k3, k3) * hard

    def order_table(self, eps, Nmax=6, part='all'):
        """Coefficients of (k3/k12)^(2N) of the signal of one pattern at k1 = k2, relative to u^(nu'_0) k3^(-1/2).

        Returns rows (N, U(Delta_N), E_N/L_N, E_N/L0_N) with
          E_N  the exact coefficient (trees t <= N),
          L_N  its late-time version with every tree on its leading soft term (K_t -> K_t/U(Delta_t)),
          L0_N the late-time version with the leading pair operator only (tree t = 0).
        """
        e = (eps, eps)
        k1 = k2 = _HALF
        # hard coefficients as polynomials in u^2: c_J = a_J + b_J u^2 (k1.k2 = (u^2 - 1/2)/2)
        if part == 'time':
            poly = {4: (k1 ** 2 * k2 ** 2, 0)}
        else:
            base = {4: k1 * k2, 3: mp.mpf(1), 2: mp.mpf(1)}
            poly = {J: (mp.mpf(1) / 4 * b, -b / 2) for J, b in base.items()}      # -d c = (1/4 - u^2/2) * base
            if part == 'all':
                poly[4] = (poly[4][0] + k1 ** 2 * k2 ** 2, poly[4][1])
        rows = []
        E = [mp.mpc(0)] * (Nmax + 1)
        L = [mp.mpc(0)] * (Nmax + 1)
        L0 = [mp.mpc(0)] * (Nmax + 1)
        for t in range(Nmax + 1):
            nu = self.nup(eps, t)
            Kt = 4 * self.tr.G(e, t) * mp.gamma(_HALF + nu) * mp.gamma(_HALF - nu)     # common factors dropped
            Ut = U(mp.mpf(3) / 2 + nu, 2)
            for p in range(Nmax + 1 - t):
                for J, (a, b) in poly.items():
                    rho = J - mp.mpf(3) / 2
                    h = mp.gamma(rho + nu + 2 * p) / (mp.factorial(p) * mp.gamma(1 + nu + p)) * mp.mpf(2) ** (-nu - 2 * p)
                    for shift, cf in ((0, a), (1, b)):
                        N = t + p + shift
                        if N <= Nmax and cf != 0:
                            E[N] += Kt * h * cf
                            L[N] += Kt / Ut * h * cf
                            if t == 0:
                                L0[N] += Kt / Ut * h * cf
        for N in range(Nmax + 1):
            UN = U(mp.mpf(3) + 2 * eps * self.nu + 2 * N, 2)
            rows.append((N, UN, E[N] / L[N], E[N] / L0[N]))
        return rows


# ------------------------------------------------------------------------- invariant-distance check
def zform_block(nu, r1, x1, muR=2):
    """MSbar block P(r1, 2) at x2 = k = 1 from the invariant-distance form of Sec. 4.1 (direct evaluation).

    psi = 2 FP int_1^oo dZ g_nu(Z)^2 (-d/dx1)^(r1-2) K(Z) + (1/2) Gamma(R) E^(-R) (gamma_E + 1/2 - (1/2) ln 2 + ln mu_R),
    with g_nu = -Q'_(nu-1/2), K = artanh[(Z-1)/(x1 + x2 + sqrt(D))], and FP dropping the logarithm of the cut Z - 1.
    """
    x2 = mp.mpf(1)
    j = r1 - 2

    def kern(e):
        # K = artanh(w), w = e/(y + x2 + s), s = sqrt(D); derivatives in y = x1 in closed form
        y = x1
        s_ = mp.sqrt((y + x2) ** 2 + 2 * y * x2 * e + e * (2 + e))
        den = y + x2 + s_
        w = e / den
        if j == 0:
            return mp.atanh(w)
        sy = (y + x2 + x2 * e) / s_
        wy = -e * (1 + sy) / den ** 2
        if j == 1:
            return -wy / (1 - w ** 2)
        syy = (1 - sy ** 2) / s_
        wyy = -e * syy / den ** 2 + 2 * e * (1 + sy) ** 2 / den ** 3
        return wyy / (1 - w ** 2) + 2 * w * wy ** 2 / (1 - w ** 2) ** 2

    def g(e):
        Z = 1 + e
        lam = nu - _HALF
        q = lambda l: mp.legenq(l, 0, Z, type=3)
        return -lam * (Z * q(lam) - q(lam - 1)) / (e * (2 + e))

    E = x1 + x2
    R = r1 - 1
    c = mp.gamma(R) * E ** (-R) / 4                 # g^2 kern -> c / e at the cut
    f = lambda e: 2 * g(e) ** 2 * kern(e)
    I1 = mp.quad(lambda e: f(e) - c / e, [0, mp.mpf('1e-4'), mp.mpf('1e-3'), mp.mpf('0.01'), mp.mpf('0.1'), mp.mpf('0.5'), 1])
    I2 = mp.quad(f, [1, 2, 9, 49, 299, mp.inf])
    return I1 + I2 + mp.gamma(R) * E ** (-R) / 2 * (mp.euler + _HALF - mp.log(2) / 2 + mp.log(muR))


def block_trees(H, eps, r1, x1):
    """The same block from the single sums (families + background), x2 = k = 1."""
    k = mp.mpf(1)
    e = (eps, eps)
    rho1 = r1 - mp.mpf(3) / 2
    tol = mp.mpf(10) ** (-mp.mp.dps + 6)
    fam, small, big = mp.mpc(0), 0, mp.mpf(0)
    for t in range(100000):
        nu = H.nup(eps, t)
        term = 4 * H.tr.G(e, t) * Khat(nu, _HALF, k, k) * Ihat(nu, rho1, x1, k)
        fam += term
        big = max(big, abs(fam))
        small = small + 1 if abs(term) < tol * big else 0
        if small >= 3:
            break
    pre = mp.sqrt(mp.pi) * mp.sqrt(2)
    cst = _HALF + mp.log(H.muR / 2) / 2
    bg, small = mp.mpc(0), 0
    for m, s in enumerate(H.S0[e]):
        z = _HALF + m
        term = pre * (-1) ** m * z * Ihat(z, rho1, x1, k) * (s + cst)
        bg += term
        small = small + 1 if abs(term) < tol * abs(bg) else 0
        if small >= 3:
            break
    return fam + bg


# ------------------------------------------------------------------------------------------- main
def main(quick=False, json_path=None):
    res, data = [], {}
    half = _HALF
    shapes = ('0.05', '0.2', '0.5', '0.7', '0.9')

    print("1. Direct evaluation: single sums against the invariant-distance form, P(r1, 2) at x2 = k (MSbar, mu_R = 2)")
    worst = 0
    pts = [(I * mp.mpf(1), 1)] if quick else [(I * mp.mpf(1), 1), (I * mp.mpf(1), -1), (I * mp.mpf(2), 1), (mp.mpf('0.9'), 1)]
    with mp.workdps(20):
        for nu, eps in pts:
            H = HiggsPortal(nu)
            for u in (shapes[:2] if quick else shapes):
                x1 = 1 / mp.mpf(u)
                for r1 in (2, 3, 4):
                    a = block_trees(H, eps, r1, x1)
                    b = zform_block(eps * nu, r1, x1)
                    err = abs(a - b) / abs(b)
                    worst = max(worst, err)
                print(f"   nu = {mp.nstr(eps * nu, 4)}, k3/k12 = {u}: max over r1 = 2, 3, 4 so far {mp.nstr(worst, 3)}", flush=True)
    res.append(compare("higgs/direct check (trees vs Z form)", worst))

    print("2. Late-time version of the leading term against C_H of CWX, Eq. (4.15)")
    worst = 0
    for nu in (I * half, I * 1, I * 2, I * 4, mp.mpf('1.4'), mp.mpf('1.45'), mp.mpf('0.3')):
        H = HiggsPortal(nu, mmax=4)
        kS, kL = mp.mpf(1), mp.mpf(10) ** -6
        for eps in (1, -1):
            ours = H.leading(eps, kS, kS, kL) / U(3 + 2 * eps * H.nu, 2) / (2 * kS ** 6 * kL ** 3)
            mu = -eps * H.nu
            cwx = -1 / (8 * mp.pi ** 4 * kS ** 6) * C_H(mu) * (kL / (2 * kS)) ** (-2 * mu)
            worst = max(worst, abs(ours / cwx - 1))
    print(f"   heavy mu = 1/2, 1, 2, 4 and light nu = 1.4, 1.45, 0.3, both branches: max |ours/CWX - 1| = {mp.nstr(worst, 3)}")
    res.append(compare("higgs/late-time leading term vs C_H", worst))

    print("3. The universal factor order by order (k1 = k2). E_N: exact coefficient of (k3/k12)^(Delta_0 + 2N);")
    print("   L_N: every tree on its leading soft term; L0_N: leading pair operator only. U_N = U(Delta_0 + 2N, 2).")
    data['orders'] = {}
    for label, nu in (('mu=0.5', I * half), ('mu=1', I * 1), ('mu=2', I * 2), ('mu=4', I * 4), ('nu=1.4', mp.mpf('1.4')), ('nu=1.45', mp.mpf('1.45'))):
        H = HiggsPortal(nu, mmax=4)
        eps = 1 if H.heavy else -1          # the leading family: Delta_0 = 3 + 2 i mu (heavy), 3 - 2 nu (light)
        rows = H.order_table(eps, Nmax=5)
        print(f"   {label}:   N    |U_N|        |E_N/L_N|     |E_N/L0_N|    |E_N/L_N|/|U_N|")
        data['orders'][label] = []
        for N, UN, r, r0 in rows:
            print(f"            {N}   {mp.nstr(abs(UN), 6):>10s}   {mp.nstr(abs(r), 6):>10s}   {mp.nstr(abs(r0), 6):>10s}   {mp.nstr(abs(r) / abs(UN), 5):>8s}")
            data['orders'][label].append(dict(N=N, absU=float(abs(UN)), EL=[float(mp.re(r)), float(mp.im(r))],
                                              EL0=[float(mp.re(r0)), float(mp.im(r0))]))
        err0 = abs(rows[0][2] / rows[0][1] - 1)
        res.append((err0 < mp.mpf(10) ** -20, err0))
        if label in ('mu=1', 'mu=4', 'nu=1.4'):
            res.append(compare(f"higgs/|E_1/L_1| {label}", abs(rows[1][2])))
            res.append(compare(f"higgs/|E_1/L0_1| {label}", abs(rows[1][3])))
    # tree by tree the factor is exact
    H = HiggsPortal(I * 2, mmax=4)
    worst = 0
    for t in range(6):
        nu = H.nup(1, t)
        late = mp.gamma(-nu) / 2 * mp.mpf(2) ** (-nu) * mp.gamma(half + nu)          # soft moment of the branch (k t/2)^nu
        worst = max(worst, abs(Khat(nu, half, 1, 1) / late / U(mp.mpf(3) / 2 + nu, 2) - 1))
    print(f"   tree by tree (t <= 5): max |exact/late / U(Delta_t, 2) - 1| = {mp.nstr(worst, 3)}")
    res.append((worst < mp.mpf(10) ** -20, worst))

    print("4. Full shape against the leading squeezed term and the late-time template (k1 = k2 = 1/2, MSbar mu_R = 2)")
    us = ('0.02', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9') if not quick else ('0.05', '0.3')
    data['shape'] = {}
    cases = (('mu=0.5', I * half), ('mu=1', I * 1), ('mu=2', I * 2), ('mu=4', I * 4), ('nu=1.4', mp.mpf('1.4')), ('nu=1.45', mp.mpf('1.45')))
    for label, nu in (cases[1:2] if quick else cases):
        H = HiggsPortal(nu)
        eps = 1 if H.heavy else -1
        print(f"   {label}:  k3/k12   |signal/leading|  arg     signal/late-time   I (complete)     families     background")
        rows = []
        for u in us:
            k3 = mp.mpf(u)
            P = H.calI(half, half, k3, parts=True)
            lead = H.leading(eps, half, half, k3)
            sig = P['families'][eps]
            R = sig / lead
            fam = mp.re(sum(P['families'].values()))
            rows.append(dict(u=float(u), absR=float(abs(R)), argR=float(mp.arg(R)), I=float(P['I']), families=float(fam),
                             background=float(P['background']), late_ratio=float(abs(R * U(3 + 2 * eps * H.nu, 2)))))
            print(f"            {u:>5s}    {mp.nstr(abs(R), 6):>9s}   {mp.nstr(mp.arg(R), 4):>8s}   {mp.nstr(abs(R * U(3 + 2 * eps * H.nu, 2)), 6):>9s}"
                  f"      {mp.nstr(P['I'], 8):>14s}  {mp.nstr(fam, 6):>12s}  {mp.nstr(P['background'], 6):>12s}", flush=True)
            if H.heavy:
                assert abs(P['imag']) < mp.mpf(10) ** -18 * (abs(fam) + abs(P['background']))
        data['shape'][label] = rows
        if label in ('mu=1', 'mu=4', 'nu=1.4'):
            for u_rec in ('0.2', '0.5'):
                if u_rec in us:
                    r = rows[us.index(u_rec)]
                    res.append(compare(f"higgs/|signal/leading| {label} u={u_rec}", r['absR']))
            if '0.5' in us:
                res.append(compare(f"higgs/I {label} u=0.5", rows[us.index('0.5')]['I']))
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=1)
    return report(res)


if __name__ == "__main__":
    jp = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    sys.exit(main(quick='--quick' in sys.argv, json_path=jp))
