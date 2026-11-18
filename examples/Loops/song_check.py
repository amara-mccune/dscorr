"""The fermion triangle against the closed form of Song, You, Cheng, He and Chen (arXiv:2611.23220) (Sec. 9.2).

Their model is a Weyl fermion N with Majorana mass M, coupling -(d_mu phi/Lambda) N^dag sigmabar^mu N and chemical
potential lam = phidot/Lambda, with <dphi^3>'_osc = (H^6/k1^6)[C+ rho^(2 i nu) + c.c.] and C+ = Chard Csoft from their
Eqs. (12), (14) and (17a)-(17c). The four Meijer G functions G^{3,4}_{4,4}(-1) of their Eq. (E20) are evaluated as
Mellin-Barnes integrals on Re s = -3/4 plus the residue at s = -c (their continuation rule). A Dirac fermion with an axial
chemical potential is two Majorana copies, and our couplings are kappa instead of 1/Lambda, so 2 kappa^3 Lambda^3 C+ is
compared with C of fermion_triangle.py (values_of_record.json).

Usage:  python song_check.py          kappa = 1.3, 5, 10 (m = 1) at 30 digits, a few minutes
        python song_check.py --all    also kappa = 20, m = 2 at 80 digits (50 digits leave a 2% error there)
"""
import sys

import mpmath as mp

from ext.record import compare, report, load

I = mp.mpc(0, 1)


def Gline(A, c, r, sign=1):
    """G_{-1}[-A/2, (1-A)/2, -r, 1-c; - / 0, 1, -1/2; 1-A-c] of their Eqs. (17c), (E20), with Log(-1) = sign i pi.

    The integrand is Gamma(-s)Gamma(1-s)Gamma(-1/2-s) Gamma(1+A/2+s)Gamma((1+A)/2+s)Gamma(1+r+s)Gamma(c+s)/Gamma(A+c+s) z^s.
    We integrate on Re s = -3/4 and add the residue at s = -c, the one left pole (of Gamma(c+s)) that lies to the
    right of that line for imaginary c (their continuation rule after (E20)). The integrand decays as e^{-3 pi |y|}.
    """
    lz = sign * I * mp.pi
    def f(s):
        return (mp.gamma(-s) * mp.gamma(1 - s) * mp.gamma(-0.5 - s) * mp.gamma(1 + A / 2 + s) * mp.gamma((1 + A) / 2 + s)
                * mp.gamma(1 + r + s) * mp.gamma(c + s) / mp.gamma(A + c + s) * mp.exp(s * lz))
    sig = mp.mpf(-3) / 4
    ims = sorted(set([0, mp.im(-c), -mp.im(A) / 2, mp.im(A) / 2, -mp.im(A + c)]))
    pts = [-mp.inf] + [mp.mpf(x) for x in ims] + [mp.inf]
    line = mp.quad(lambda y: f(sig + I * y), pts) / (2 * mp.pi)
    s0 = -c
    res = (mp.gamma(c) * mp.gamma(1 + c) * mp.gamma(c - 0.5) * mp.gamma(1 + A / 2 - c) * mp.gamma((1 + A) / 2 - c)
           * mp.gamma(1 + r - c) / mp.gamma(A) * mp.exp(s0 * lz))
    assert mp.re(s0) > sig
    return line + res


def Hfun(A, c, sign=1):
    return sum((A * (1 - A) / 4) ** (1 - r) * Gline(A, c, r, sign) for r in (0, 1))


def Cplus(M, lam, Lam=1):
    M, lam = mp.mpf(M), mp.mpf(lam)
    nu = mp.sqrt(M ** 2 + lam ** 2)
    a, b = -I * (nu - lam), -I * (nu + lam)
    A = 2 + 2 * I * nu
    om = mp.sinh(2 * mp.pi * lam) * mp.sinh(mp.pi * lam) / mp.sinh(mp.pi * nu)
    B = mp.beta(a, b)
    Bc = mp.conj(B)
    Chard = -(I * mp.pi * nu * (A - 1) * mp.tanh(mp.pi * nu)) / (8 * Lam ** 2 * (A - 1 - M ** 2) * abs(B) ** 2) * (
        A ** 2 * (A - 1) * mp.sinh(mp.pi * lam) / ((A - 1 - M ** 2) * mp.sinh(mp.pi * nu)) - lam * M ** 2 / nu * Bc)

    def T(c):
        return mp.sin(mp.pi * c) * ((om - mp.cos(mp.pi * c)) * Hfun(A, c)
                                    + (om + mp.cos(mp.pi * c)) * mp.conj(Hfun(mp.conj(A), mp.conj(c))))
    Ta, Tb = T(a), T(b)
    Csoft = -(mp.mpf(2) ** (A - 4) * Bc) / (mp.pi * Lam * nu * A * (A - 1) * (A + 2) * mp.sinh(2 * mp.pi * nu)) * (
        (A - 2) * (4 + A - A ** 2) - 16 * A * lam ** 2 + I * 8 * (Ta + Tb) / (mp.pi ** 3 * (A + 1)))
    return Chard, Csoft, Chard * Csoft, dict(Ta=Ta, Tb=Tb)



def main(full=False):
    table = load()
    pts = [(1.3, 1, 30), (5, 1, 30), (10, 1, 30)] + ([(20, 2, 80)] if full else [])
    res = []
    for ka, m, dps in pts:
        with mp.workdps(dps):
            Cp = Cplus(m, ka)[2]
            val = 2 * Cp * mp.mpf(ka) ** 3
            key = f"fermion_triangle/C exact kappa={ka:g},m={m:g}"
            print(f"   kappa = {ka:g}, m = {m:g}:  2 kappa^3 Lambda^3 C+ = {mp.nstr(val, 12)}")
            res.append(compare(key, val, table))
    return report(res)


if __name__ == "__main__":
    sys.exit(main(full="--all" in sys.argv))
