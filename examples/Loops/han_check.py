"""The fermion bubble with a chemical potential against Han, Lei, Qin and Yin (arXiv:2611.20919) (Sec. 9.2).

They compute the bubble of the scalar bilinear (psi psi + h.c.) for a Weyl fermion with a Majorana mass M and a
chemical potential lam, and in their Sec. 5.3 its squeezed bispectrum for the vertices with q1 = -1 and q2 = -3
(our r1 = 4 and r2 = 2). This script evaluates their three-point coefficient, Eqs. (3.15)-(3.17) and
(5.39)-(5.43), in their conventions, with one change: the soft-vertex integral (after their rotation
eta2 = -i x, Eq. (3.23)) is computed in position space with ext/softvertex.py. In their notation, with
nu = sqrt(M^2 + lam^2) and k_s = k_34 = 1,

    T^{s1 s2}_{rho, lam} = 2 e^{i pi (5 + q1 + 2 i rho nu)/2} Gamma(5 + q1 + 2 i rho nu) e^{-i pi (q2 + 4)/2}
                           N*_{s1} N*_{s2} J(rho; K*_{s1}, K*_{s2}),
    J = int d^3p/(2 pi)^3 F_{s1 s2} (2 p1)^{i rho nu - 1/2} (2 p2)^{i rho nu - 1/2}
        int_0^oo dx x^{q2 + 3} e^{-x} W_{K1, i nu}(2 p1 x) W_{K2, i nu}(2 p2 x),

where v_s = N_s (-2 k tau)^{-1/2} W_{K_s, i nu}(2 i k tau) are the mode functions of their Eqs. (2.16)-(2.19). The
amplitude 2 |T~_+ + T~_-^*| and the phase arg(T~_+ + T~_-^*) are compared with their Fig. 8, digitized from
the vector graphics of the arXiv PDF (han_fig8.json; reading accuracy about 1e-3 in log10).

The script also checks that the pair coefficient S(0) of dirac_pair.py is their single-sector coefficient:
with lam = kappa, the Dirac chiral amplitudes are (alpha_h, beta_h) = phi_h (v_h, u_h) with phi_+ = -i,
phi_- = 1, and

    S(0, s) = -i e^{s pi mubar} [X_{rho = -s}]^*,
    X_rho = sum_{s1 s2} C^rho_{s1 s2} sum_{f = u, v} eps_f N*_{f s1} N*_{f s2} J(rho; K*_{f s1}, K*_{f s2}),

with eps_u = +1, eps_v = -1, so that their Eq. (5.41) reads
T~_rho = [1 + e^{-i pi (4 + q1 + 2 i rho nu)}] 2 e^{i pi (5 + q1 + 2 i rho nu)/2} Gamma(5 + q1 + 2 i rho nu)
         e^{-i pi (q2 + 4)/2} X_rho.

Usage:  python han_check.py          (four points of Fig. 8 and the identity at m = kappa = 1, about 5 min)
        python han_check.py --all    (all 21 digitized points, about 1.5 h; lam >= 15 at 50-60 digits)
"""
import json
import os
import sys

import mpmath as mp

from ext.softvertex import Grid, Leg, soft_J

I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))


def monomials(s1, s2):
    """F_{s1 s2} = (1 - s1 s2 c)/2 with c = (1 - y1^2 - y2^2)/(2 y1 y2) at k = 1, as (coef, d1, d2)."""
    ss = s1 * s2
    return [(mp.mpf(1) / 2, 0, 0), (-mp.mpf(ss) / 4, -1, -1), (mp.mpf(ss) / 4, 1, -1), (mp.mpf(ss) / 4, -1, 1)]


class HanBubble:
    """Three-point coefficient of Han et al. in their conventions, with our position-space soft vertex."""

    def __init__(self, M, lam, q1=-1, q2=-3, grid=None):
        self.M, self.lam = mp.mpf(M), mp.mpf(lam)
        self.nt = mp.sqrt(self.M ** 2 + self.lam ** 2)
        self.nu = I * self.nt
        self.q1, self.q2 = q1, q2
        self.grid = grid or Grid()
        self._legs = {}

    def modes(self, lam):
        """{(f, s): (N, K)} of their Eqs. (2.16)-(2.19) at chemical potential lam."""
        M, e = self.M, mp.exp(mp.pi * lam / 2)
        return {('u', 1): (M * e, -0.5 - I * lam), ('u', -1): (I / e, 0.5 + I * lam),
                ('v', 1): (I * e, 0.5 - I * lam), ('v', -1): (M / e, -0.5 + I * lam)}

    def C(self, f, s, rho):
        """Late-time coefficient C_f^(rho) of their Eq. (2.26)."""
        N, K = self.modes(self.lam)[(f, s)]
        mu = I * rho * self.nt
        return N * mp.gamma(-2 * mu) / mp.gamma(0.5 - mu - K) * mp.exp(-I * mp.pi / 4 + rho * mp.pi * self.nt / 2)

    def Cpair(self, s1, s2, rho):
        return self.C('u', s1, rho) * self.C('u', s2, rho) - self.C('v', s1, rho) * self.C('v', s2, rho)

    def leg(self, A, K):
        key = (mp.nstr(A, 25), mp.nstr(K, 25))
        if key not in self._legs:
            self._legs[key] = Leg(A, K, self.nu)
        return self._legs[key]

    def J(self, rho, K1, K2, s1, s2):
        a = I * rho * self.nt - 0.5
        tot = mp.fsum(c * soft_J(self.leg(a + d1, K1), self.leg(a + d2, K2), self.q2 + 4, self.grid)
                      for c, d1, d2 in monomials(s1, s2))
        return mp.mpf(2) ** (2 * a) * tot

    def X(self, rho):
        md = self.modes(self.lam)
        tot = 0
        for s1 in (1, -1):
            for s2 in (1, -1):
                inner = 0
                for f, eps in (('u', 1), ('v', -1)):
                    (N1, K1), (N2, K2) = md[(f, s1)], md[(f, s2)]
                    inner += eps * mp.conj(N1 * N2) * self.J(rho, mp.conj(K1), mp.conj(K2), s1, s2)
                tot += self.Cpair(s1, s2, rho) * inner
        return tot

    def Ttilde(self, rho):
        q1, q2, nt = self.q1, self.q2, self.nt
        hard = 2 * mp.exp(I * mp.pi * (5 + q1 + 2 * I * rho * nt) / 2) * mp.gamma(5 + q1 + 2 * I * rho * nt)
        return (1 + mp.exp(-I * mp.pi * (4 + q1 + 2 * I * rho * nt))) * hard * mp.exp(-I * mp.pi * (q2 + 4) / 2) * self.X(rho)

    def amplitude_phase(self):
        z = self.Ttilde(1) + mp.conj(self.Ttilde(-1))
        return 2 * abs(z), mp.arg(z)


# the 21 points compared in the paper; QUICK_POINTS include the kink of the M = 3 curve at lam = 3 -> 4
ALL_POINTS = [(1, 1), (1, 5), (1, 7), (1, 10), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (3, 16),
              (5, 1), (5, 5), (5, 6), (5, 15), (5, 16), (5, 17), (7, 4), (7, 8), (7, 20)]
QUICK_POINTS = [(1, 5), (3, 4), (5, 1), (7, 4)]


def dps_for(lam):
    return 30 if lam <= 7 else 40 if lam <= 12 else 50 if lam < 18 else 60


def main(full=False):
    fig = json.load(open(os.path.join(HERE, "han_fig8.json")))
    res = []
    print("1. Fig. 8 of arXiv:2611.20919 (q1 = -1, q2 = -3), with the soft vertex in position space")
    left = {float(k): v for k, v in fig["left"].items()}
    pts = ALL_POINTS if full else QUICK_POINTS
    for M, lam in pts:
        M, lam = float(M), float(lam)
        y_fig = fig["right"][f"{M:g}"][f"{lam:g}"]
        with mp.workdps(dps_for(lam)):
            amp, ph = HanBubble(M, lam).amplitude_phase()
            d = float(mp.log10(amp)) - y_fig
            okr = abs(d) < 2e-3
            line = f"   M = {M:g}, lam = {lam:>4g}:  log10 amplitude {float(mp.log10(amp)):8.4f}  (Fig. 8: {y_fig:8.4f})"
            res.append((okr, abs(d)))
            if M == 1 and lam in left:
                A_fig, ph_fig = left[lam]
                dA, dph = float(amp) / A_fig - 1, float(ph) - ph_fig
                line += f";  amplitude {float(amp):.2f} ({A_fig:.2f}), phase {float(ph):.4f} ({ph_fig:.4f})"
                res += [(abs(dA) < 2e-3, abs(dA)), (abs(dph) < 2e-3, abs(dph))]
            print(("  [ok ] " if okr else "  [FAIL] ") + line, flush=True)

    print("2. S(0) of dirac_pair.py as their single-sector coefficient, m = kappa = 1")
    from dirac_pair import DiracPair
    with mp.workdps(30):
        P, H = DiracPair(1, 1), HanBubble(1, 1)
        for s in (1, -1):
            S = P.S(0, s=s)
            pred = -I * mp.exp(s * mp.pi * P.D.mubar) * mp.conj(H.X(-s))
            err = abs(S / pred - 1)
            print(f"   s = {s:+d}:  S(0) = {mp.nstr(S, 10)},  -i e^(s pi mubar) X*_(-s) = {mp.nstr(pred, 10)},  rel {mp.nstr(err, 2)}")
            res.append((err < 1e-20, err))
    bad = sum(1 for ok, _ in res if not ok)
    print(f"  {len(res) - bad}/{len(res)} comparisons passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(full="--all" in sys.argv))
