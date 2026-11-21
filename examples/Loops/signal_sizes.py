"""How large are the loop signals? f_NL with the exact squeezed coefficients (Secs. 6.3, 7.2, 7.3, 7.5, 8.3).

One convention for every model (Hubble units, P_zeta = H^4/(4 pi^2 phidot_0^2), zeta = -H dphi/phidot_0):

    S(k1, k2, k3) = (k1 k2 k3)^2 <zeta zeta zeta>' / [(2 pi)^4 P_zeta^2],
    S  contains  A r^(2 + i omega) + c.c.,   r = k3/k1,   f_NL^osc = |A|.

This is the f_NL^osc of Aoki and Strumia (2604.05548, Eqs. (12), (15)), of Chen, Wang and Xianyu (1805.02656,
below Eq. (78)), and of Sec. 8.3 of the paper. The other template papers differ by fixed factors:

    Wang and Xianyu (2004.02887, Eqs. (45)-(48))    coefficient of (2 k3/k1)^(2 + 2 i nu):   f_WX   = |A| / 4
    Song et al. (2611.23220, Eqs. (18)-(19)),
    You et al. (2605.21419, Eqs. (4.2)-(4.3))       (18/5) f_NL (2 pi^2 P)^2 [2 cos(...)] r^2:  f_CC   = (10/9) |A|
    Bodas et al. (2507.22978, Eq. (2.6))            F = (10/3) S k1 k2 k3/(k1^3 + k2^3 + k3^3):
                                                    |F_osc| at p = (k1 + k2)/k3 = 2 is (5/3) |A|
    Chen, Wang and Xianyu (1612.08122, Eq. (4.38))  coefficient of (k3/2k1)^(2 - 2 mu):     |A| = |coef| / 4 (heavy)

A search that writes B = (18/5) f_NL (2 pi^2 P)^2 S_T/(k1 k2 k3)^2 with a template S_T -> 2 cos(omega ln r + phi) r^2
in the squeezed limit measures f_NL = (10/9) |A|.

For each model the script first reproduces a signal size from the template paper's own formulas and then puts
the exact coefficient of the paper in its place:

    A.  Aoki-Strumia fermion triangle: the two f_NL values of Sec. 8.3 from C of Table 4.
    B1. Standard Model background (Chen, Wang, Xianyu): Higgs portal, W/Z loop, top quark.
    B2. Helical vectors (Wang, Xianyu): contracted bubble and triangle with the hard line.
    B3. Charged scalars (Bodas, Broadberry, Sundrum, Xu).
    B4. Scalar-bilinear fermion bubble (vertices of Han, Lei, Qin, Yin; operator O_n5 of 1805.02656).
    B5. Fermion triangle of the right-handed neutrino (1805.02656; You et al.; Song et al.).

Inputs that take minutes each are read from signal_sizes_record.json ("inputs"): M_A and R_hard of
helical_triangle.py (rerun with --recompute-helical), C of fermion_triangle.py (values_of_record.json), and the
amplitude of Fig. 8 of Han et al. (han_fig8.json, which han_check.py reproduces). Everything else is closed form.
The numbers of the table are compared with signal_sizes_record.json ("values"); the exit status is nonzero if
one is not reproduced.

The top-quark row needs the late-time expansion of <psibar psi psibar psi> in de Sitter space, which is derived
here from the exact Dirac propagator (TopQuark). Its pair branch has no term (tau tau'/X^2)^(2 + 2 i m) [the power
of CWX's Eq. (4.28)] and none at 3 + 2 i m; it starts at 4 + 2 i m, as stated in Sec. 6.3.

Usage:  python signal_sizes.py                              (a few seconds)
        python signal_sizes.py --md table.md --tex table.tex --json table.json
        python signal_sizes.py --recompute-helical          (reruns helical_triangle.leading, 2 to 10 min per point)
"""
import argparse
import json
import os
import sys

import mpmath as mp

from ext.record import compare, report

mp.mp.dps = 30
I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
PZETA = mp.mpf("2.1e-9")


def load_record():
    with open(os.path.join(HERE, "signal_sizes_record.json")) as f:
        return json.load(f)


def phidot0(P=PZETA):
    """phidot_0/H^2 = 1/(2 pi sqrt(P_zeta)), about 3474 (59^2)."""
    return 1 / (2 * mp.pi * mp.sqrt(P))


def U(Delta, r2=2):
    """Universal factor, Eq. (6.2)."""
    Delta, r2 = mp.mpmathify(Delta), mp.mpmathify(r2)
    return mp.sqrt(mp.pi) * mp.mpf(2) ** (1 + Delta - r2) * mp.gamma(r2 - Delta) / (mp.gamma(r2 - 1) * mp.gamma(1.5 - Delta))


# ===================================================================================== A. Aoki-Strumia
def as_fnl(C, Nc=3, P=PZETA):
    """|f_NL^osc| = |S| = N_c 4 pi^2 P_zeta |C| (Sec. 8.2: S = -4 pi^2 P_zeta C per color)."""
    return Nc * 4 * mp.pi ** 2 * P * abs(C)


def as_eq60(kappa, m, Nc=3, P=PZETA):
    """Asymptotic estimate of Aoki and Strumia, their Eq. (60)."""
    kappa, m = mp.mpf(kappa), mp.mpf(m)
    return Nc * m ** 4 * P * mp.pi ** 2.5 * kappa ** 1.5 / (64 * mp.sqrt(2)) * mp.exp(-2 * mp.pi * m ** 2 / kappa)


def fermion_C(kappa, m, kind="exact"):
    """C of Table 4 (kind 'exact') or its late-time version ('nl'), from values_of_record.json."""
    with open(os.path.join(HERE, "values_of_record.json")) as f:
        tab = json.load(f)
    v = tab[f"fermion_triangle/C {kind} kappa={kappa:g},m={m:g}"]["value"]
    return mp.mpc(v[0], v[1])


# ===================================================================================== B1. Standard Model background
def t_cwx(mu):
    """C_H(mu)/[Gamma(mu) Gamma(3/2 - mu)]^2 of CWX's Eq. (4.15). It maps the late-time two-point function
    <O O> = N (tau tau'/X^2)^Delta, Delta = 3 - 2 mu, to S = 16 pi (f' phidot_0)^2 N t (k3/2k1)^(Delta - 1) (H = 1)."""
    mu = mp.mpmathify(mu)
    return (2 - mu) * (3 - 2 * mu) * mp.cos(mp.pi * mu) * mp.sin(mp.pi * mu) ** 3 * mp.gamma(-4 + 4 * mu) * mp.gamma(2 - 2 * mu) ** 2


def C_H(mu):
    """CWX Eq. (4.15)."""
    mu = mp.mpmathify(mu)
    return t_cwx(mu) * mp.gamma(mu) ** 2 * mp.gamma(mp.mpf(3) / 2 - mu) ** 2


def C_A(mu):
    """CWX Eq. (4.35)."""
    mu = mp.mpmathify(mu)
    return (-(2 - mu) * (3 - 2 * mu) * (23 - 6 * mu) ** 2 * mp.gamma(mu) ** 2 * mp.gamma(-4 + 4 * mu)
            * mp.gamma(mp.mpf(5) / 2 - mu) ** 2 * mp.gamma(2 - 2 * mu) ** 2 * mp.sin(mp.pi * mu) ** 3 * mp.cos(mp.pi * mu))


def C_Psi(m):
    """CWX Eq. (4.30)."""
    m = mp.mpf(m)
    return (m ** 4 * (-24 + 7j) * (1 + I * m) * (3 + 2 * I * m) * mp.cosh(mp.pi * m) ** 3 * mp.sinh(mp.pi * m)
            * mp.gamma(0.5 - I * m) ** 2 * mp.gamma(-2 - 4 * I * m) * mp.gamma(I * m) ** 2 * mp.gamma(2 * I * m) ** 2)


def higgs(order, g=1):
    """Higgs portal f_H(X) H^dag H, CWX Eq. (4.38): S = g^2/(2 pi^4) [C_H(mu_h) (k3/2k1)^(2 - 2 mu_h) + (mu_h -> -mu_h)],
    g = f_H' phidot_0/(1 + f_DH). order = i mu (heavy) or nu (light). Returns (template, exact, |U|) for |A|."""
    order = mp.mpmathify(order)
    if mp.re(order) == 0:                       # heavy: the two terms are conjugates
        tmpl = g ** 2 * abs(C_H(order)) / (8 * mp.pi ** 4)
        u = abs(U(3 + 2 * order))
    else:                                       # light: leading family 3 - 2 nu, a power law
        tmpl = g ** 2 * abs(C_H(order)) * mp.mpf(2) ** (2 * order - 2) / (2 * mp.pi ** 4)
        u = abs(U(3 - 2 * order))
    return tmpl, u * tmpl, u


def gauge(mu, gA=1):
    """W/Z loop f_A(X) F^2/4, CWX Eq. (4.41), one real vector of mass m_A^2 = mu^2 + 1/4 (heavy):
    S = gA^2 27/(16 pi^4 m_A^4) [C_A (k3/2k1)^(2 - 2 mu_1) + c.c.], gA = f_A' H^4 phidot_0/(1 + f_A)."""
    mu = mp.mpf(mu)
    mA2 = mu ** 2 + mp.mpf(1) / 4
    tmpl = gA ** 2 * 27 * abs(C_A(I * mu)) / (64 * mp.pi ** 4 * mA2 ** 2)
    u = abs(U(3 + 2 * I * mu))
    return tmpl, u * tmpl, u


class TopQuark:
    """Loop of f_Psi(X) Psibar i Dslash Psi (CWX Sec. 4.3), O = m Psibar Psi on shell, g = f_Psi' H^2 phidot_0/(1 + f_Psi).

    Dirac propagator in de Sitter space (H = 1, conformal time, P_s = (1 + s gamma^0)/2):
        S(x, x') = (a a')^(-1/2) [ i gamma^mu d_mu + a (m + i gamma^0) ] [ G_+(Z) P_+ + G_-(Z) P_- ],
        G_s = Gamma(a_s) Gamma(b_s)/(16 pi^2) 2F1(a_s, b_s; 2; (1 + Z)/2),   (a_s, b_s) = (1 - i s m, 2 + i s m),
    the form of CWX's Eqs. (4.24)-(4.25) with the spinor derivative written out [it gives m + i gamma^0, not m + 2].
        <psibar psi(x) psibar psi(x')> = -tr[S(x, x') S(x', x)]
                                       = -tau tau' [2 A_+ A'_+ + 2 A_- A'_- - 4 G_+' G_-' X^2/(tau tau')^2],
        A_s = i s G_s' dZ/dtau - (m + i s) G_s/tau,   A'_s: tau <-> tau'.
    Checks: flat-space limit 1/(pi^4 sigma^6); de Sitter invariance; no terms w^(2 + 2im), w^(3 + 2im) on the pair
    branch (w = tau tau'/X^2). N4 is the coefficient of w^(4 + 2 i m).
    """

    def __init__(self, m):
        self.m = mp.mpf(m)

    @staticmethod
    def _G(a, b, xi, branch):
        if branch:              # the term (-xi)^(-a) of the connection formula at xi -> -oo
            c = mp.gamma(a) * mp.gamma(b - a) / mp.gamma(2 - a) / (16 * mp.pi ** 2)
            f = lambda x: c * (-x) ** (-a) * mp.hyp2f1(a, a - 1, a - b + 1, 1 / x)
        else:
            f = lambda x: mp.gamma(a) * mp.gamma(b) / (16 * mp.pi ** 2) * mp.hyp2f1(a, b, 2, x)
        return f(xi), mp.diff(f, xi) / 2

    def two_point(self, tau, taup, X, pair=True):
        """<psibar psi psibar psi>: the full function, or its pair branch (both lines carry +i m)."""
        m = self.m
        Z = (tau ** 2 + taup ** 2 - X ** 2) / (2 * tau * taup)
        xi = (1 + Z) / 2
        G, dG = {}, {}
        for s in (1, -1):
            a, b = 1 - I * s * m, 2 + I * s * m
            if pair and s == 1:
                a, b = b, a                     # exponent 2 + i m of G_+; G_- keeps its exponent 1 + i m
            G[s], dG[s] = self._G(a, b, xi, pair)
        dZ, dZp = 1 / taup - Z / tau, 1 / tau - Z / taup
        A = {s: I * s * dG[s] * dZ - (m + I * s) * G[s] / tau for s in (1, -1)}
        Ap = {s: I * s * dG[s] * dZp - (m + I * s) * G[s] / taup for s in (1, -1)}
        return -tau * taup * (2 * A[1] * Ap[1] + 2 * A[-1] * Ap[-1] - 4 * dG[1] * dG[-1] * X ** 2 / (tau * taup) ** 2)

    def N4(self, w=mp.mpf(10) ** -12):
        with mp.workdps(max(mp.mp.dps, 40)):    # the terms at 2 + 2im and 3 + 2im cancel to the relative size w^2
            D = 4 + 2 * I * self.m
            v = [self.two_point(-mp.sqrt(x), -mp.sqrt(x), mp.mpf(1)) / x ** D for x in (w, w / 2)]
            return 2 * v[1] - v[0]              # the corrections are of order w

    def lower_powers(self, w=mp.mpf(10) ** -12):
        """|pair branch| / w^(3 + 2 i m): of order w if the terms at 2 + 2im and 3 + 2im are absent."""
        with mp.workdps(max(mp.mp.dps, 40)):
            return abs(self.two_point(-mp.sqrt(w), -mp.sqrt(w), mp.mpf(1)) / w ** (3 + 2 * I * self.m))

    def template(self, g=1):
        """|coefficient| of r^(1 + 2 i m) in CWX's Eq. (4.40). The exact signal has no such term."""
        return g ** 2 * self.m ** 2 * abs(C_Psi(self.m)) / (2 * mp.pi ** 4)

    def first_term(self, g=1):
        """|coefficient| of r^(3 + 2 i m): (late-time soft vertex, exact, |U(4 + 2 i m, 2)|)."""
        D = 4 + 2 * I * self.m
        late = g ** 2 * self.m ** 2 * 16 * mp.pi * abs(self.N4() * t_cwx((3 - D) / 2)) / 8
        u = abs(U(D))
        return late, u * late, u

    def cwx_map_check(self):
        """The map t_cwx applied to CWX's own two-point function, their Eq. (4.28), over the coefficient of their
        Eq. (4.40). The map gives the coefficient of (k3/2k1)^(1 + 2 i m) and reproduces their Eq. (4.29) with k3/2k1
        in place of k3/k1; converting (4.29) to S gives -4 times the coefficient printed in (4.40). So the ratio is -4:
        the Gamma functions and the factor (-24 + 7i) agree, and their two equations differ by a sign and a power of 2."""
        m = self.m
        N = m ** 4 * (7 + 24j) * mp.gamma(I * m) ** 2 * mp.gamma(0.5 - I * m) ** 2 / (32 * mp.pi ** 5)
        return 16 * mp.pi * N * t_cwx(mp.mpf(1) / 2 - I * m) / (m ** 2 * C_Psi(m) / (2 * mp.pi ** 4))


# ===================================================================================== B2. helical vectors
def wx_C(mut, nut):
    """Wang and Xianyu, Eq. (42) (their mu~ is our kappa, their nu~ our mu)."""
    mut, nut = mp.mpf(mut), mp.mpf(nut)
    num = mp.exp(2 * mp.pi * mut - 4 * mp.pi * nut) * (4 * nut ** 4 - 8j * nut ** 3 - 6 * nut ** 2 + 2j * nut + mp.mpf(3) / 8) * mp.gamma(-2j * nut) ** 2
    den = (I * nut) ** 2 * mp.gamma(5 + 4j * nut) * mp.gamma(0.5 + I * mut - I * nut) ** 2 * mp.gamma(0.5 - I * mut - I * nut) ** 2
    return num / den


def wx_bracket(kappa, mu):
    """e^(2 pi nu) Gamma(4 + 2 i nu) Gamma(2 + 2 i nu) C of their Eqs. (45), (46)."""
    mu = mp.mpf(mu)
    return mp.exp(2 * mp.pi * mu) * mp.gamma(4 + 2j * mu) * mp.gamma(2 + 2j * mu) * wx_C(kappa, mu)


def wx_M(kappa, mu):
    """Their closed form in the normalization of Eq. (7.6): S_(b) = -32 pi^2 P m^4 e^(2 pi Omega)/u^3 2 Re[M r^(2+2i mu)],
    from lambda_1 lambda_2 = 8 rho_3 rho_4, rho_3 = m^2/(u phidot_0), rho_4 = m^2 e^(2 pi Omega)/(u phidot_0)^2."""
    return wx_bracket(kappa, mu) * mp.mpf(2) ** (2 + 2j * mp.mpf(mu)) / 1024


def wx_fnl(kappa, mu, diagram="b", u=1, P=PZETA, asymptotic=False):
    """f_NL^(osc) of Wang and Xianyu in THEIR convention (coefficient of (2 k3/k1)^(2 + 2 i nu)):
    Eqs. (45), (46) at finite mass, or Eqs. (47), (48) (asymptotic=True, what their Figs. 1 and 2 show)."""
    kappa, mu = mp.mpf(kappa), mp.mpf(mu)
    m = mp.sqrt(mu ** 2 + mp.mpf(1) / 4)
    Om = max(kappa - mu, 0)
    if asymptotic:
        g = kappa - mu
        if diagram == "b":
            return mp.pi ** 1.5 / (128 * mp.sqrt(2)) * P * m ** 5.5 * mp.exp(2 * mp.pi * (g + 2 * Om)) / u ** 3
        return mp.pi ** 1.5 / (256 * mp.sqrt(2)) * P * m ** 5.5 * mp.exp(2 * mp.pi * (g + Om)) / u ** 3
    pref = mp.pi ** 2 / 32 * mp.exp(2 * mp.pi * Om) if diagram == "b" else mp.pi ** 2 / 64
    return pref * P * m ** 4 / u ** 3 * abs(wx_bracket(kappa, mu))


def wx_estimate(kappa, mu, diagram, u=1, P=PZETA):
    """The parametric estimates of Wang and Xianyu, their Eqs. (25), (27), (29), for the diagrams (a), (b), (c) of their
    Fig. 4. Diagram (a) has three axion vertices; they give no explicit result for it, and neither does the paper."""
    kappa, mu = mp.mpf(kappa), mp.mpf(mu)
    m = mp.sqrt(mu ** 2 + mp.mpf(1) / 4)
    g = kappa - mu
    Om = max(g, 0)
    if diagram == "a":
        return P / 4 * (kappa / m) ** 3 * m * mp.exp(2 * mp.pi * (g + 2 * Om))
    if diagram == "b":
        return P / 4 * m ** 4 * mp.exp(2 * mp.pi * (g + 2 * Om)) / u ** 3
    return P / 16 * m ** 4 * mp.exp(2 * mp.pi * (g + Om)) / u ** 3


def helical_inputs(rec=None):
    rec = rec or load_record()
    out = {}
    for key, v in rec["inputs"]["helical"].items():
        mu, ka = key.split(",")
        out[(mu, ka)] = dict(M_A=mp.mpc(*map(mp.mpf, v["M_A"])), M0_iso=mp.mpc(*map(mp.mpf, v["M0_isosceles"])),
                             R_hard=mp.mpc(*map(mp.mpf, v["R_hard"])), m2=mp.mpf(v["m2_hard"]))
    return out


def helical(mu, kappa, inp, u=1, P=PZETA):
    """Exact amplitudes |A| (coefficient of r^(2 + 2 i mu), k1 = k2) of the model of Wang and Xianyu at u = 1:
    their closed forms for diagrams (b) and (c), the contracted bubble with the exact M_A, the triangle with the hard
    line [diagram (b)], and the sum of (b) and (c). The quartic vertex of (c) is 1/(2 e^(2 pi Omega)) of the contracted
    hard line [their Eq. (37)]."""
    d = inp[(mu, kappa)]
    mu_, ka_ = mp.mpf(mu), mp.mpf(kappa)
    Om = max(ka_ - mu_, 0)
    m4 = d["m2"] ** 2
    norm = 32 * mp.pi ** 2 * P * m4 / u ** 3
    R_iso = -d["m2"] * mp.exp(-2 * mp.pi * Om) * d["M0_iso"] / d["M_A"]       # triangle over WX-normalized bubble, k1 = k2
    eb = mp.exp(2 * mp.pi * Om)
    return dict(tmpl_b=4 * wx_fnl(kappa, mu, "b", u, P), tmpl_c=4 * wx_fnl(kappa, mu, "c", u, P),
                asym_b=4 * wx_fnl(kappa, mu, "b", u, P, True), asym_c=4 * wx_fnl(kappa, mu, "c", u, P, True),
                bubble_b=norm * eb * abs(d["M_A"]), exact_b=norm * eb * abs(R_iso * d["M_A"]),
                exact_c=norm * abs(d["M_A"]) / 2, exact_bc=norm * abs(d["M_A"] * (eb * R_iso + mp.mpf(1) / 2)),
                MA_over_WX=abs(d["M_A"] / wx_M(kappa, mu)), R_hard=abs(d["R_hard"]), R_iso=abs(R_iso), m=mp.sqrt(d["m2"]))


def wx_gamma_max(u=1, P=PZETA, feq=10):
    """Largest kappa - mu allowed by their Eq. (56): (1/4) P u^-3 e^(6 pi gamma) < f_NL^eq."""
    return mp.log(4 * feq * u ** 3 / P) / (6 * mp.pi)


# ===================================================================================== B3. charged scalars
def bbsx(lam, mu, P=PZETA, Nchi=1):
    """Bodas et al.: F contains f p^(-3 - i(lam - 2 mu)) + c.c. at |alpha|^2/Lambda^2 = 1; S = (3/5) F/r on k1 = k2.
    Largest coupling from the parametric-resonance bound, their Eq. (5.48): alpha_PR^2 = prod(lam +- mu +- mu)/(2 pi lam),
    with Lambda = phidot_0/lam. Returns |F_osc|(p = 2) and |A| for their Eq. (3.35) and for the exact coefficient."""
    import charged_scalars as cs
    with mp.workdps(50):
        lam_, mu_ = mp.mpf(lam), mp.mpf(mu)
        aPR2 = (lam_ + 2 * mu_) * lam_ ** 2 * (lam_ - 2 * mu_) / (2 * mp.pi * lam_)
        coup = Nchi * aPR2 * lam_ ** 2 / phidot0(P) ** 2
        f_t, f_e = abs(cs.f_bbsx(mu_, mu_, lam_)), abs(cs.ChargedBubble(mu_, mu_, lam_).f())
    return dict(coupling=coup, F_tmpl=f_t / 8 * coup, F_exact=f_e / 8 * coup, A_tmpl=mp.mpf(3) / 40 * f_t * coup,
                A_exact=mp.mpf(3) / 40 * f_e * coup, ratio=f_e / f_t)


# ===================================================================================== B4. scalar-bilinear fermion bubble
def han_amplitude(M, lam):
    """T_amp,3pt of Han et al., Fig. 8 (q1 = -1, q2 = -3), digitized in han_fig8.json and reproduced by han_check.py."""
    with open(os.path.join(HERE, "han_fig8.json")) as f:
        fig = json.load(f)
    return mp.mpf(10) ** mp.mpf(fig["right"][f"{M:g}"][f"{lam:g}"])


def fermion_bubble(M, lam, P=PZETA):
    """|A| of the bubble of a Weyl fermion with Majorana mass M and chemical potential lam, vertices of Han et al.'s
    Eq. (2.46), from their Eq. (2.48): <dphi^3>'_osc = -H^7 T_amp cos(2 nu ln r_1 + phi)/(4 Lambda_1 Lambda_2^3 k1 k2 k12^4),
    so |A| = T_amp/(256 pi sqrt(P) Lambda_1 Lambda_2^3). Couplings of the operator (d_mu phi)^2 N N/Lambda^3 of
    1805.02656 [their O_n5, unit coefficient]: 1/Lambda_1 = 2 phidot_0/Lambda^3, 1/Lambda_2^3 = 1/Lambda^3, with
    Lambda = phidot_0/lam. Then |A| = (pi^4/4) P^2 lam^6 T_amp (time-derivative vertices only; the gradient term of
    the covariant vertex multiplies it by |1 + (3 + i nu)/(1 + i nu)|, Sec. 8.3)."""
    T = han_amplitude(M, lam)
    nu = mp.sqrt(mp.mpf(M) ** 2 + mp.mpf(lam) ** 2)
    A = mp.pi ** 4 / 4 * P ** 2 * mp.mpf(lam) ** 6 * T
    return dict(T_amp=T, A=A, gradient=abs(1 + (3 + I * nu) / (1 + I * nu)))


# ===================================================================================== B5. right-handed neutrino triangle
def song_eq20(M, lam, P=PZETA):
    """Song et al., Eq. (20): the complete amplitude f_NL^CC at large lam."""
    M, lam = mp.mpf(M), mp.mpf(lam)
    return 5 * mp.pi / 18 * P * M ** 2 * lam ** 3 * mp.exp(-11 * mp.pi * M ** 2 / (4 * lam))


def song_eq21(M, lam, P=PZETA):
    """Song et al., Eq. (21): the nonlocal late-time amplitude that they attribute to You et al."""
    M, lam = mp.mpf(M), mp.mpf(lam)
    return 5 * mp.pi ** 2.5 / (288 * mp.sqrt(2)) * P * M ** 4 * lam ** 1.5 * mp.exp(-9 * mp.pi * M ** 2 / (4 * lam))


def you_eq46(M, lam, P=PZETA):
    """You et al. (v2), Eq. (4.6). It is lam^2/4 times Song et al.'s Eq. (21) at large lam."""
    M, lam = mp.mpf(M), mp.mpf(lam)
    nu = mp.sqrt(M ** 2 + lam ** 2)
    return 5 * mp.pi ** 2.5 / (2 ** 7 * 9 * mp.sqrt(2)) * P * M ** 4 * lam ** 3.5 * mp.exp(-9 * mp.pi / 2 * (nu - lam))


def cwx_eq79(m, lam, P=PZETA):
    """Chen, Wang and Xianyu (1805.02656), Eq. (79): the first estimate of the neutrino triangle."""
    m, lam = mp.mpf(m), mp.mpf(lam)
    return 3 * mp.pi ** 2 / 2 * P * lam ** 5 * m ** 3 * mp.exp(-5 * mp.pi * m ** 2 / (4 * lam))


def majorana_fcc(C_dirac, P=PZETA):
    """f_NL^CC of Song et al. from C of Table 4: one Majorana fermion is half of a Dirac fermion, and
    f_CC = (5/(9 pi sqrt(P))) |C_+|/Lambda^3 with C = 2 kappa^3 Lambda^3 C_+, so f_CC = (10/9) 2 pi^2 P |C|."""
    return mp.mpf(10) / 9 * 2 * mp.pi ** 2 * P * abs(C_dirac)


# ===================================================================================== the table
def fmt(x, n=3):
    return mp.nstr(x, n, min_fixed=-3, max_fixed=4)


def song_cplus_fcc(M, lam, rec, P=PZETA):
    """f_CC = (5/(9 pi sqrt(P))) |C_+| with the closed form of Song et al. evaluated by song_check.Cplus (record, Lambda = 1)."""
    for e in rec["inputs"]["song_cplus"]:
        if abs(float(e["M"]) - float(M)) < 1e-9 and abs(float(e["lam"]) - float(lam)) < 1e-3:
            return 5 / (9 * mp.pi * mp.sqrt(P)) * mp.mpf(e["abs_Cplus"]) * (mp.mpf(e["lam"]) / phidot0(P)) ** 3
    raise KeyError((M, lam))


def build(rec=None, verbose=True):
    """Compute every row. Returns (rows, results): rows of the table, results of the comparisons with the record.
    Each row gives |A| ('template', 'exact') and the factor 'own' that converts |A| to the template paper's convention."""
    rec = rec or load_record()
    vals = rec["values"]
    res, rows = [], []
    say = print if verbose else (lambda *a, **k: None)

    def chk(key, value):
        res.append(compare(key, value, vals, quiet=not verbose))

    def row(**kw):
        kw.setdefault("own", 1)
        kw.setdefault("own_label", "f_NL^osc = |A|")
        if "ratio" not in kw:
            none = kw.get("template") is None or kw.get("exact") is None or kw.get("no_ratio")
            kw["ratio"] = None if none else kw["exact"] / kw["template"]
        rows.append(kw)
        return kw

    Phi = phidot0()
    say(f"P_zeta = {mp.nstr(PZETA, 3)},  phidot_0/H^2 = {mp.nstr(Phi, 6)} = ({mp.nstr(mp.sqrt(Phi), 4)})^2\n")

    # ------------------------------------------------------------------ A
    say("A. Aoki-Strumia fermion triangle: |f_NL^osc| = N_c 4 pi^2 P_zeta |C|, N_c = 3 (Sec. 8.3)")
    say("     kappa    m     their Eq. (60)   late-time soft vertex   exact      exact/Eq. (60)")
    for ka, m in ((10, 1), (20, 2), (5, 1), (1.3, 1)):
        C, Cl = fermion_C(ka, m), fermion_C(ka, m, "nl")
        f, fl, f60 = as_fnl(C), as_fnl(Cl), as_eq60(ka, m)
        chk(f"A/f_NL exact kappa={ka:g},m={m:g}", f)
        chk(f"A/f_NL late kappa={ka:g},m={m:g}", fl)
        chk(f"A/f_NL Eq.(60) kappa={ka:g},m={m:g}", f60)
        say(f"     {ka:>5g}  {m:>3g}     {fmt(f60):>10s}       {fmt(fl):>10s}        {fmt(f):>10s}     {fmt(f / f60)}")
        row(model="Fermion triangle, minimal scenario (Aoki-Strumia)", point=f"kappa = {ka:g}, m = {m:g} H, N_c = 3",
            coupling="1/f = kappa H/taudot_0 (tied to P_zeta)", constraint="kappa < 60 (unitarity, their Sec. 3.3)",
            template=f60, template_src="their Eq. (60) (reproduced from the formula)", late=fl, exact=f,
            exact_src="C of Table 4 (taken from the paper's code)", own_label="f_NL^osc of Aoki-Strumia")
    say(f"   with P_zeta = 2e-9 (their value): {fmt(as_fnl(fermion_C(10, 1), P=mp.mpf('2e-9')))} and "
        f"{fmt(as_fnl(fermion_C(20, 2), P=mp.mpf('2e-9')))}\n")

    # ------------------------------------------------------------------ B1
    say("B1. Standard Model background (Chen, Wang, Xianyu 1612.08122), |A| at unit coupling g = 1")
    for mu in ("0.5", "1", "2", "4"):
        t, e, u = higgs(I * mp.mpf(mu))
        chk(f"B1/Higgs template mu={mu}", t)
        chk(f"B1/Higgs |U| mu={mu}", u)
        row(model="Higgs portal (CWX)", point=f"mu_H = {mu} (M_H^2 = {mp.nstr(mp.mpf(mu) ** 2 + 2.25, 3)} H^2)",
            coupling="g_H = f_H' phidot_0 = 1 (f_H = X/Lambda^2 at Lambda^2 = phidot_0)",
            constraint="Lambda^2 > phidot_0; M_H must then be tuned against f_H(X_0) = g_H phidot_0",
            template=t, template_src="their Eq. (4.38) with (4.15) (reproduced)", exact=e,
            exact_src="Eq. (6.3), U(3 + 2 i mu, 2)", own=4, own_label="|coefficient of (k3/2k1)^(2 - 2 mu_h)|, scales as g_H^2")
    for nu in ("1.4", "1.45"):
        t, e, u = higgs(mp.mpf(nu))
        chk(f"B1/Higgs template nu={nu}", t)
        chk(f"B1/Higgs U nu={nu}", u)
        row(model="Higgs portal (CWX), light Higgs", point=f"nu_H = {nu} (M_H = {mp.nstr(mp.sqrt(2.25 - mp.mpf(nu) ** 2), 2)} H)",
            coupling="g_H = 1", constraint="Lambda^2 > phidot_0; a power law r^(2 - 2 nu), no oscillation",
            template=t, template_src="their Eq. (4.38) (reproduced)", exact=e, exact_src="Eq. (6.3), U(3 - 2 nu, 2)",
            own=mp.mpf(2) ** (2 - 2 * mp.mpf(nu)), own_label="coefficient of (k3/2k1)^(2 - 2 nu), scales as g_H^2")
    for r in rows[-6:]:
        say(f"   {r['point']:<34s} template {fmt(r['template']):>9s}   exact {fmt(r['exact']):>9s}   ratio {fmt(r['ratio'], 4)}")
    cH = abs(C_H(I * mp.mpf("0.1"))) * mp.cosh(mp.pi / 5)
    chk("B1/|C_H| cosh(2 pi mu) mu=0.1", cH)
    say(f"   |C_H| cosh(2 pi mu) = {fmt(cH)} at mu = 0.1 and {fmt(abs(C_H(I)) * mp.cosh(2 * mp.pi))} at mu = 1 (their Eq. (4.42): about 1.5)")
    t, e, u = higgs(I, g=(1 + mp.mpf(9) / 4) / Phi)
    chk("B1/Higgs exact mu=1 untuned", e)
    say(f"   without tuning, g_H = M_H^2/phidot_0: |A| = {fmt(t)} (template), {fmt(e)} (exact) at mu = 1")
    for mu in ("1", "2"):
        t, e, u = gauge(mu)
        chk(f"B1/gauge template mu={mu}", t)
        r = row(model="W/Z loop (CWX), one real vector", point=f"mu = {mu} (m_A^2 = {mp.nstr(mp.mpf(mu) ** 2 + 0.25, 3)} H^2)",
                coupling="g_A = f_A' H^4 phidot_0 = 1", constraint="f_A = X/Lambda^4 with Lambda^4 > phidot_0^2 has g_A < H^2/phidot_0 = 2.9e-4",
                template=t, template_src="their Eq. (4.41) with (4.35) (reproduced)", exact=e,
                exact_src="Eq. (6.2) at r_2 = 2 (nine helicity pairs)", own=4, own_label="|coefficient of (k3/2k1)^(2 - 2 mu_1)|, scales as g_A^2")
        say(f"   W/Z, {r['point']:<28s} template {fmt(t):>9s}   exact {fmt(e):>9s}   ratio {fmt(r['ratio'], 4)}")
    cA = abs(C_A(I * mp.mpf("0.1"))) * mp.cosh(mp.pi / 5)
    chk("B1/|C_A| cosh(2 pi mu) mu=0.1", cA)
    say(f"   |C_A| cosh(2 pi mu) = {fmt(cA)} at mu = 0.1 and {fmt(abs(C_A(I)) * mp.cosh(2 * mp.pi))} at mu = 1 (their Eq. (4.45): about 1800);"
        f" 27 x 1800/(16 pi^4) = {fmt(27 * 1800 / (16 * mp.pi ** 4))} (their Eq. (4.46): 30)")
    for m in (1, 2):
        Tq = TopQuark(m)
        late, ex, u = Tq.first_term()
        chk(f"B1/top N4 m={m}", Tq.N4())
        chk(f"B1/top lower powers m={m}", Tq.lower_powers())
        chk(f"B1/top template m={m}", Tq.template())
        chk(f"B1/top first term exact m={m}", ex)
        chk(f"B1/top CWX map m={m}", abs(Tq.cwx_map_check() + 4))
        row(model="Top-quark loop (CWX)", point=f"m = {m} H", coupling="g_Psi = f_Psi' H^2 phidot_0 = 1",
            constraint="as for W/Z: g_Psi < 2.9e-4 for f_Psi = X/Lambda^4", template=Tq.template(),
            template_src="their Eq. (4.40) with (4.30) (reproduced): the coefficient of r^(1 + 2 i m), a power that the exact signal does not contain",
            late=late, exact=ex, no_ratio=True,
            exact_src=f"recomputed here: the first term is r^(3 + 2 i m); late-time coefficient {fmt(late)} times |U(4 + 2 i m, 2)| = {fmt(u)}",
            own_label="coefficients of different powers of r, scale as g_Psi^2")
        say(f"   top quark, m = {m}: their coefficient of r^(1 + 2im) {fmt(Tq.template())};  first term of the exact signal, r^(3 + 2im): "
            f"{fmt(late)} (late-time) x {fmt(u)} = {fmt(ex)};  N4 = {mp.nstr(Tq.N4(), 8)}")
    Tq = TopQuark(1)
    X = mp.mpf("1e-4")
    with mp.workdps(40):
        flat = Tq.two_point(mp.mpf(-1), mp.mpf(-1), X, pair=False) * mp.pi ** 4 * X ** 6
    chk("B1/top flat-space limit", abs(flat - 1))
    say("")

    # ------------------------------------------------------------------ B2
    say("B2. Helical vectors (Wang, Xianyu 2004.02887), u = 1, in THEIR convention f_WX = |A|/4")
    inp = helical_inputs(rec)
    say("    mu  kappa   (b): Eq.(45)  [Eq.(47)]   bubble, exact M_A   triangle (exact)   exact/(45)   M_A/M_WX   |R_hard|   (c): Eq.(46)   (c) exact")
    gmax = wx_gamma_max()
    for (mu, ka) in sorted(inp, key=lambda k: (float(k[0]), float(k[1]))):
        h = helical(mu, ka, inp)
        say(f"   {mu:>3s} {ka:>6s}   {fmt(h['tmpl_b'] / 4):>10s}  [{fmt(h['asym_b'] / 4):>9s}]   {fmt(h['bubble_b'] / 4):>12s}      {fmt(h['exact_b'] / 4):>12s}"
            f"      {fmt(h['exact_b'] / h['tmpl_b']):>8s}   {fmt(h['MA_over_WX']):>8s}   {fmt(h['R_hard']):>8s}   {fmt(h['tmpl_c'] / 4):>10s}   {fmt(h['exact_c'] / 4):>10s}")
        chk(f"B2/their (b) mu={mu} kappa={ka}", h["tmpl_b"])
        chk(f"B2/exact (b) mu={mu} kappa={ka}", h["exact_b"])
        chk(f"B2/exact (b)+(c) mu={mu} kappa={ka}", h["exact_bc"])
        g = mp.mpf(ka) - mp.mpf(mu)
        row(model="Helical-vector triangle (Wang-Xianyu), diagram (b)", point=f"mu = {mu}, kappa = {ka} (m_A = {fmt(h['m'])} H, u = 1)",
            coupling="rho_3 = m_A^2/(u phidot_0), fixed by the model",
            constraint=f"kappa - mu < {fmt(gmax)} (their Eq. (56)); kappa < 59 (Lambda_F^2 > phidot_0)" + ("; VIOLATED" if g > gmax else ""),
            template=h["tmpl_b"], template_src="their Eq. (45) (reproduced; Eq. (47) is its large-mass form)", bubble=h["bubble_b"],
            exact=h["exact_b"], exact_src="M_A (Sec. 7.2) times the hard line (Sec. 7.3) on k1 = k2; recomputed with helical_triangle.py",
            own=mp.mpf(1) / 4, own_label="f_NL^(osc) of Wang-Xianyu",
            extra=dict(asym_b=h["asym_b"], tmpl_c=h["tmpl_c"], exact_c=h["exact_c"], exact_bc=h["exact_bc"], MA_over_WX=h["MA_over_WX"], R_hard=h["R_hard"]))
    nu20 = mp.sqrt(400 - mp.mpf(1) / 4)
    chk("B2/WX Eq.(45) over Eq.(47) m=20", wx_fnl(nu20, nu20, "b") / wx_fnl(nu20, nu20, "b", asymptotic=True))
    r45 = wx_fnl(40.5, 40, "b") / wx_fnl(40.5, 40, "b", asymptotic=True)
    chk("B2/WX Eq.(45) over Eq.(47) mu=40 kappa=40.5, over (1 + e^(-pi))/2", r45 / ((1 + mp.exp(-mp.pi)) / 2))
    say(f"   their Eq. (47) is the large-mass form of Eq. (45) at kappa = mu only: for kappa != mu, (45)/(47) tends to (1 + e^(-2 pi |kappa - mu|))/2"
        f" [{fmt(r45)} at (40, 40.5), against {fmt((1 + mp.exp(-mp.pi)) / 2)}]")
    chk("B2/WX map against Eq.(7.13) mu=3", abs(abs(wx_M(3, 3)) / mp.mpf(rec["inputs"]["wx_map_check"]["M_late_pp_times_R_WX mu=kappa=3"]) - 1))
    say(f"   their Eq. (56) allows kappa - mu < {fmt(gmax)} at u = 1")
    say("   their parametric estimates, Eqs. (25), (27), (29), for the diagrams (a), (b), (c); (a) has no explicit result, there or here")
    for mu, ka in (("3", "4"), ("5", "5.75")):
        ea, eb, ec = (wx_estimate(ka, mu, d) for d in "abc")
        say(f"     mu = {mu}, kappa = {ka}:  (a) {fmt(ea)},  (b) {fmt(eb)} [Eq. (45): {fmt(wx_fnl(ka, mu, 'b'))}],  (c) {fmt(ec)} [Eq. (46): {fmt(wx_fnl(ka, mu, 'c'))}]")
        chk(f"B2/their estimate (a), Eq.(25), mu={mu} kappa={ka}", ea)
        chk(f"B2/their estimate (b), Eq.(27), mu={mu} kappa={ka}", eb)
        chk(f"B2/their estimate (c), Eq.(29), mu={mu} kappa={ka}", ec)
        row(model="Helical-vector triangle with three axion vertices (Wang-Xianyu), diagram (a)", point=f"mu = {mu}, kappa = {ka}",
            coupling="phi F Ftilde/(4 Lambda_F) at the three vertices, Lambda_F = phidot_0/kappa", constraint="as for diagram (b)",
            template=4 * ea, template_src="their Eq. (25), an order-of-magnitude estimate (reproduced)", exact=None,
            exact_src="not computed in the paper", own=mp.mpf(1) / 4, own_label="f_NL^(osc) of Wang-Xianyu", ratio=None,
            extra=dict(est_b=4 * eb, est_c=4 * ec))
    say("")

    # ------------------------------------------------------------------ B3
    say("B3. Charged scalars (Bodas et al. 2507.22978), coupling at the parametric-resonance bound, N_chi = 1")
    for lam, mu in ((40, 4), (30, 4), (50, 10), (40, 15)):
        b = bbsx(lam, mu)
        say(f"   lam = {lam}, mu = {mu}: |alpha|^2/Lambda^2 < {fmt(b['coupling'])};  |F_osc|(p = 2) = {fmt(b['F_tmpl'])} (Eq. (3.35)), {fmt(b['F_exact'])} (exact);"
            f"  |A| = {fmt(b['A_tmpl'])}, {fmt(b['A_exact'])};  ratio {fmt(b['ratio'], 4)}")
        chk(f"B3/F_osc template lam={lam} mu={mu}", b["F_tmpl"])
        chk(f"B3/ratio lam={lam} mu={mu}", b["ratio"])
        row(model="Charged scalars (Bodas et al.)", point=f"lambda = {lam}, mu_1 = mu_2 = {mu}, N_chi = 1",
            coupling=f"|alpha|^2/Lambda^2 = {fmt(b['coupling'])} (largest allowed)",
            constraint="parametric resonance, their Eq. (5.48); lambda < 59 (their Eq. (6.3))",
            template=b["A_tmpl"], template_src="their Eq. (3.35) (reproduced; it gives the green curve of their Fig. 20)", exact=b["A_exact"],
            exact_src="Eq. (7.19), charged_scalars.py", own=mp.mpf(5) / 3, own_label="|F_osc| at p = 2")
    b36 = bbsx(30, 4, P=1 / (2 * mp.pi * 3600) ** 2)
    chk("B3/F_osc template lam=30 mu=4 phidot0=3600", b36["F_tmpl"])
    say(f"   with phidot_0 = (60 H)^2 as in their text: |F_osc|(p = 2) = {fmt(b36['F_tmpl'])} at (30, 4); their Fig. 20 shows about 3e-3\n")

    # ------------------------------------------------------------------ B4
    say("B4. Scalar-bilinear fermion bubble: Han et al.'s Eq. (2.48) with the operator (d phi)^2 N N/Lambda^3 of 1805.02656")
    for M, lam in ((1, 10), (1, 20), (3, 20), (5, 20)):
        fb = fermion_bubble(M, lam)
        tri = song_cplus_fcc(M, lam, rec) * mp.mpf(9) / 10
        say(f"   M = {M}, lam = {lam}: T_amp = {fmt(fb['T_amp'])},  |A| = {fmt(fb['A'])} (x {fmt(fb['gradient'])} with the gradient term);"
            f"  triangle of the same fermion (closed form of Song et al.): {fmt(tri)}")
        chk(f"B4/bubble M={M} lam={lam}", fb["A"])
        chk(f"B4/triangle M={M} lam={lam}", tri)
        row(model="Fermion bubble, scalar bilinear (vertices of Han et al.)", point=f"M = {M} H, lambda = {lam} (Lambda = {fmt(Phi / lam)} H)",
            coupling="1/Lambda_1 = 2 phidot_0/Lambda^3, 1/Lambda_2^3 = 1/Lambda^3 (O_n5 of 1805.02656, unit coefficient)",
            constraint="lambda < 59 (Lambda^2 > phidot_0)", template=None,
            template_src="none: neither paper quotes an f_NL for this diagram", exact=fb["A"],
            exact_src="Han et al., Eq. (2.48) and Fig. 8 (taken from their paper; our soft vertex reproduces the amplitude of Fig. 8)",
            own_label="f_NL^osc = |A|, one Weyl fermion, time-derivative vertices", extra=dict(triangle_same_point=tri, gradient_factor=fb["gradient"]))
    say("")

    # ------------------------------------------------------------------ B5
    say("B5. Right-handed neutrino triangle (one Majorana fermion, coupling 1/Lambda, lam = phidot_0/Lambda), f_CC = (10/9) |A|")
    for ka, m in ((10, 1), (20, 2)):
        f = majorana_fcc(fermion_C(ka, m))
        chk(f"B5/f_CC from C kappa={ka},m={m}", f)
        chk(f"B5/Song closed form over C kappa={ka},m={m}", song_cplus_fcc(m, ka, rec) / f)
        chk(f"B5/Song Eq.(20) over C kappa={ka},m={m}", song_eq20(m, ka) / f)
        say(f"   lam = {ka}, M = {m}: f_CC = (10/9) 2 pi^2 P |C| = {fmt(f)};  closed form of Song et al.: {fmt(song_cplus_fcc(m, ka, rec))};  their Eq. (20): {fmt(song_eq20(m, ka))}")
    for e in rec["inputs"]["song_table_I"]:
        LamH, M = e["Lambda_over_H"], e["M"]
        lam = Phi / LamH
        f20, f21 = song_eq20(M, lam), song_eq21(M, lam)
        say(f"   Lambda = {LamH} H (lam = {fmt(lam, 4)}), M = {M}: Table I {e['f_CC']:.3g} (complete), {e['f_NLoc']:.3g} (late-time);"
            f"  Eq. (20): {fmt(f20)}, Eq. (21): {fmt(f21)};  You et al. Eq. (4.6): {fmt(you_eq46(M, lam))}")
        chk(f"B5/Song Eq.(20) Lambda={LamH} M={M}", f20)
        if M == 4:      # the large-lambda forms reproduce Table I at M = 4 to 1%
            res.append((abs(f20 / e["f_CC"] - 1) < 0.01, abs(f20 / e["f_CC"] - 1)))
            res.append((abs(f21 / e["f_NLoc"] - 1) < 0.02, abs(f21 / e["f_NLoc"] - 1)))
        row(model="Right-handed neutrino triangle (You et al.; Song et al.)", point=f"Lambda = {LamH} H (lambda = {fmt(lam, 4)}), M_R = {M} H",
            coupling="1/Lambda", constraint="Lambda > 59 H (phidot_0^2/Lambda^4 < 1, as in 1805.02656); Song et al. ask for phidot_0^2/(16 pi^2 Lambda^4) < 0.1, Lambda > 30 H"
            + ("; BELOW the first bound" if LamH < 59 else ""),
            template=mp.mpf(e["f_NLoc"]) * 9 / 10, template_src="Song et al., Table I, late-time column (taken from the paper)",
            exact=mp.mpf(e["f_CC"]) * 9 / 10, exact_src="Song et al., Table I (taken from the paper; their closed form equals ours, Sec. 8.2)",
            own=mp.mpf(10) / 9, own_label="f_NL^CC of Song et al.", extra=dict(you_eq46=you_eq46(M, lam) * 9 / 10))
    t1 = rec["inputs"]["song_table_I_check"]            # their inputs: Lambda/H = 60 with lam = 57.81 (their Eq. (H6))
    P_song = 1 / (2 * mp.pi * 60 * mp.mpf(t1["lam"])) ** 2
    f_t1 = 5 / (9 * mp.pi * mp.sqrt(P_song)) * mp.mpf(t1["abs_Cplus"]) / 60 ** 3
    chk("B5/Song Table I Lambda=60 M=4 from the closed form", f_t1)
    res.append((abs(f_t1 / mp.mpf("5.20e-4") - 1) < 5e-3, abs(f_t1 / mp.mpf("5.20e-4") - 1)))
    say(f"   Table I, Lambda = 60 H, M = 4, from their closed form at lam = {t1['lam']} ({t1['dps']} digits): H^3 |C_+| = "
        f"{fmt(mp.mpf(t1['abs_Cplus']) / 60 ** 3)} (Table I: 1.35e-7), f_CC = {fmt(f_t1, 4)} (Table I: 5.20e-4)")
    lam35 = Phi / 35
    chk("B5/You Eq.(4.6) Lambda=35 M=5", you_eq46(5, lam35))
    chk("B5/Song Eq.(20) Lambda=35 M=5", song_eq20(5, lam35))
    chk("B5/You Eq.(4.6) over Song Eq.(21) times 4/lam^2", you_eq46(5, lam35) / song_eq21(5, lam35) * 4 / lam35 ** 2)
    say(f"   Lambda = 35 H, M = 5: You et al. Eq. (4.6) gives {fmt(you_eq46(5, lam35))} (their Fig. 5: O(0.1)); Song Eq. (20): {fmt(song_eq20(5, lam35))};"
        f" Song Eq. (21): {fmt(song_eq21(5, lam35))}")
    row(model="Right-handed neutrino triangle (You et al., v2)", point=f"Lambda = 35 H (lambda = {fmt(lam35, 4)}), M_R = 5 H", coupling="1/Lambda",
        constraint="their bound Lambda > 34 H (footnote 5); BELOW Lambda > 59 H", template=you_eq46(5, lam35) * 9 / 10,
        template_src="You et al. (v2), Eq. (4.6) (reproduced from the formula; their Figs. 5, 6 show 0.1)", exact=song_eq20(5, lam35) * 9 / 10,
        exact_src="Song et al., Eq. (20) (formula; it matches their Table I to 1% at M = 4)", own=mp.mpf(10) / 9, own_label="f_NL^CC")
    lam3 = mp.sqrt(Phi) / 3
    say(f"   Lambda = 3 phidot_0^(1/2) = {fmt(3 * mp.sqrt(Phi), 4)} H (lam = {fmt(lam3, 6)}): Chen-Wang-Xianyu, Eq. (79), against the closed form of Song et al.")
    for m in (1, 2, 3, 5):
        t, e = cwx_eq79(m, lam3), song_cplus_fcc(m, lam3, rec) * mp.mpf(9) / 10
        chk(f"B5/CWX Eq.(79) Lambda=3 sqrt(phidot0) m={m}", t)
        chk(f"B5/exact Lambda=3 sqrt(phidot0) m={m}", e)
        say(f"      m = {m}: Eq. (79) {fmt(t)};  exact {fmt(e)};  ratio {fmt(e / t)}")
        row(model="Neutrino triangle, first estimate (Chen-Wang-Xianyu 2018)", point=f"Lambda = 3 phidot_0^(1/2) (lambda = {fmt(lam3, 4)}), m = {m} H",
            coupling="1/Lambda", constraint="Lambda > phidot_0^(1/2) (their Sec. 4)", template=t,
            template_src="their Eq. (79) (reproduced from the formula; their Fig. 2)", exact=e,
            exact_src="closed form of Song et al. (recomputed with song_check.py; equals ours, Sec. 8.2)", own_label="f_NL(clock) = |A|")
    return rows, res


def write_markdown(rows, path):
    with open(path, "w") as f:
        f.write("| Model | Parameters | Couplings | Constraint | Template: \\|A\\| | in its own convention | Source | Exact: \\|A\\| | in the same convention | Source | Exact/template |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            t = "none" if r["template"] is None else fmt(r["template"])
            to = "" if r["template"] is None else f"{fmt(r['template'] * r['own'])} ({r['own_label']})"
            ratio = "" if r["ratio"] is None else fmt(r["ratio"])
            e = "none" if r["exact"] is None else fmt(r["exact"])
            eo = "" if r["exact"] is None else fmt(r["exact"] * r["own"])
            f.write(f"| {r['model']} | {r['point']} | {r['coupling']} | {r['constraint']} | {t} | {to} | {r['template_src']} | {e} | "
                    f"{eo} | {r['exact_src']} | {ratio} |\n")


def write_tex(rows, path):
    def num(x):
        if x is None:
            return "---"
        t = mp.nstr(x, 2, min_fixed=-2, max_fixed=3)
        if "e" in t:
            a, b = t.split("e")
            return rf"${a}\times10^{{{int(b)}}}$"
        return f"${t}$"
    esc = lambda t: t.replace("_", r"\_").replace("^", r"\^{}")
    with open(path, "w") as f:
        f.write("% f_NL^osc = |A| of the loop signals (S contains A r^(2 + i omega) + c.c.): template and exact. Written by signal_sizes.py.\n")
        f.write("\\begin{tabular}{@{}llccc@{}}\n\\toprule\nModel & Parameters & Template & Exact & Ratio\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{esc(r['model'])} & {esc(r['point'])} & {num(r['template'])} & {num(r['exact'])} & {num(r['ratio'])}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_json(rows, path):
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (mp.mpf, mp.mpc)):
            return float(v) if mp.im(v) == 0 else [float(mp.re(v)), float(mp.im(v))]
        return v
    with open(path, "w") as f:
        json.dump(dict(P_zeta=float(PZETA), convention="|A|: S contains A r^(2 + i omega) + c.c., r = k3/k1; 'own' converts |A| to the template paper's convention",
                       rows=[conv(r) for r in rows]), f, indent=1)


def recompute_helical(rec):
    """Rerun helical_triangle.leading at the points of the record and compare M_A and R_hard."""
    import helical_triangle as ht
    res = []
    for key, v in rec["inputs"]["helical"].items():
        mu, ka = key.split(",")
        with mp.workdps(int(v.get("dps", 30))):
            L = ht.leading(mu, ka)
            err = abs(L["M_A"] / mp.mpc(*map(mp.mpf, v["M_A"])) - 1)
            errR = abs(L["R_hard"] / mp.mpc(*map(mp.mpf, v["R_hard"])) - 1)
        print(f"   mu = {mu}, kappa = {ka}: M_A agrees to {mp.nstr(err, 2)}, R_hard to {mp.nstr(errR, 2)}")
        res += [(err < 1e-10, err), (errR < 1e-10, errR)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="write the table as Markdown")
    ap.add_argument("--tex", help="write a short LaTeX version of the table")
    ap.add_argument("--json", help="write the rows as JSON")
    ap.add_argument("--recompute-helical", action="store_true", help="rerun helical_triangle.leading at every point of the record")
    args = ap.parse_args()
    rec = load_record()
    rows, res = build(rec)
    if args.recompute_helical:
        print("\nHelical inputs recomputed with helical_triangle.py")
        res += recompute_helical(rec)
    if args.md:
        write_markdown(rows, args.md)
    if args.json:
        write_json(rows, args.json)
    if args.tex:
        write_tex(rows, args.tex)
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
