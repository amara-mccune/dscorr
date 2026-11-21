"""The fermion triangle as a contact term plus a remainder, and the closed forms behind the total-energy phase.

Model and conventions of ext/fermitri.py (Dirac fermion, axial chemical potential kappa, vertices
kappa d_mu(dtau) Psibar gamma^mu gamma^5 Psi with the 1/taudot_0 stripped).

The identity. Rotate the fermion by the fluctuation on the two hard legs only, Psi -> exp(-i alpha gamma^5) Psi with
alpha = -+ kappa dtau_h. The kinetic term then cancels the axial vertices of the hard legs, the soft leg keeps its
axial vertex, and the mass term becomes

    m Psibar exp(-2 i alpha gamma^5) Psi = m Psibar Psi  +- 2 i kappa m dtau_h Psibar gamma^5 Psi  - 2 kappa^2 m dtau_h^2 Psibar Psi + ...

At third order in dtau the triangle of three axial vertices is therefore, exactly,

    triangle = B' + T3',
    B'   a two-vertex loop with the quartic vertex on the hard legs (the contact term, of order m^2),
    T3'  a triangle with two pseudoscalar mass vertices on the hard legs (of order m^4).

After the rescaling of the spinors by a^(3/2) a mass vertex carries a(eta) = i a/t on the rotated contour
eta = i a t, times the external modes G(k, t) = (1 + k t) e^(-k t)/(2 k^3):

    quartic vertex   t^-1 (1 + k1 t)(1 + k2 t) e^(-(k1 + k2) t)/(4 k1^3 k2^3),  no spin structure,
    cubic vertex     kappa m a gamma^5 [t^-1 k^-3 + k^-2] e^(-k t),  gamma^5 = -1, +1 on the chiral blocks (0,0), (1,1).

The hard part of B' is exact and elementary. With the hard ends of the soft lines on their pair branch,
(2 y t)^(nu + n), the vertex integral is Gamma(2 nu + n + p + 1) k12^-(2 nu + n + p + 1) for the three powers
p = -1, 0, 1 with weights 1, k12, k1 k2. In the variables R = 2 k3/k12 and delta = (k1 - k2)/k3 of
ext/shapevars.py, and in the normalization of fermion_triangle_shape.py,

    D_N(B') = -i m kappa^2 2^(-2 nu - N) Sc(N) [Gamma(2 nu + N) + Gamma(1 + 2 nu + N) + w Gamma(2 + 2 nu + N)]/(4 w)^3,
    w = (1 - R^2 delta^2/4)/4      (w = 1/4 on k1 = k2),

where Sc(N) collects the soft moments and spin traces of fermitri.py without the hard line. B' does not depend
on the angle between k1 and k3.

Closed forms (kappa >> m; eps = i (mubar - kappa), nu = i mubar). When the first index of the lines is 1/2 + nu the
loop of a two-vertex diagram with vertex powers r1 (hard) and r2 (soft) is

    int_1^oo dxi int_-1^1 deta (xi^2 - 1) (1 + xi)^-(r2 - 2 nu) (1 + R xi/2)^-(r1 + 2 nu),

xi = (y1 + y2)/k3, and its part that is analytic at R = 0 (the pair signal) is `pair_closed`. The functions below
are these forms and their first corrections. See fermion_phase.py for the comparison with the exact series.
"""
from itertools import product

import mpmath as mp

from .fermitri import FermiTriangle

I = mp.mpc(0, 1)

__all__ = ["Contact", "contact_coefficients", "contact_symmetric", "pair_closed", "bubble_closed", "eps_factor",
           "contact_leading", "contact_first_order", "segment_average", "tree_saddle_phase", "log_series",
           "strip_total_energy"]


# ------------------------------------------------------------------------------ exact: contact term and remainder
class Contact(FermiTriangle):
    """FermiTriangle with the two diagrams of the identity triangle = B' + T3'."""

    delta_t = mp.mpf(10) ** -8        # shift of the power t^-1 of the mass vertices of T3' (see t3_accumulate)

    # ---- B'
    def contact_accumulate(self, keep=None):
        """{(n, mc, ia, ib, q, K1, K2): coefficient} with Sc(n) = sum coefficient J(ia, ib, q, K1, K2).

        keep(a3, ah, h1, h2): optional filter on the branch of the soft vertex, the branch of the quartic vertex,
        and the helicities of the two soft lines."""
        nmax, nu, ka = self.nmax, self.nu, self.kappa
        T, _ = self._traces()
        acc = {}
        two = mp.mpf(2)
        for a3 in (1, -1):
            kC, kB = ("v", "u") if a3 == 1 else ("u", "v")
            for h1, h2 in product((1, -1), repeat=2):
                hel = mp.exp(mp.pi * (h1 + h2) * ka)
                Wt = {}
                for i in (0, 1):                              # Psibar Psi: chiral block (i, i), unit spin matrix
                    rowC = self.row(kC, h1, i)                # hard end of line C
                    colB = self.col(kB, h2, i)                # hard end of line B
                    for ah in (1, -1):
                        if keep is not None and not keep(a3, ah, h1, h2):
                            continue
                        cC, KeC = self.hard_end(rowC, ah)
                        cB, KeB = self.hard_end(colB, ah)
                        eC, eB = self.eser(KeC), self.eser(KeB)
                        for n1 in range(nmax + 1):
                            for n2 in range(nmax + 1 - n1):
                                Wt[(n1, n2)] = Wt.get((n1, n2), 0) + cC * cB * eC[n1] * eB[n2] * two ** (2 * nu + n1 + n2)
                for i3 in (0, 1):
                    cB3, KB3, _ = self.row(kB, h2, i3)
                    cC3, KC3, _ = self.col(kC, h1, 1 - i3)
                    labC, labB = self.klabel(KC3), self.klabel(KB3)
                    for f3, q, s3, _ in self.vertex_terms(a3, i3, 3):
                        soft = a3 * hel * cB3 * cC3 * f3 / 2
                        for (n1, n2), w in Wt.items():
                            for pC, pB in product((0, 1), repeat=2):
                                red = T[("1", "1", s3, pC, pB, 0)][0]       # (1/2) tr[Pi_B S3 Pi_C]
                                hf = (h1 if pC else 1) * (h2 if pB else 1)
                                for (nr, mc, da, db), v in red.items():
                                    n = n1 + n2 + nr
                                    if n > nmax:
                                        continue
                                    key = (n, mc, n1 + da, n2 + db, q, labC, labB)
                                    acc[key] = acc.get(key, 0) + 2 * soft * hf * w * v
        return {k: v for k, v in acc.items() if v != 0}

    def Sc(self, acc=None):
        """[Sc(0), ..., Sc(nmax)]."""
        acc = acc if acc is not None else self.contact_accumulate()
        out = [mp.mpc(0)] * (self.nmax + 1)
        for (n, mc, ia, ib, q, K1_, K2_), cf in acc.items():
            if mc:
                raise ValueError("the contact term cannot depend on the angle between k1 and k3")
            out[n] += cf * self.table(ia, ib, q, K1_, K2_)
        self.table.save()
        return out

    # ---- T3'
    def mass_terms(self, a, i, delta=0):
        """[(coefficient, power of t, power of the vertex energy)] of the cubic mass vertex on block (i, i)."""
        cf = self.kappa * self.m * a * (-1 if i == 0 else 1)
        return [(cf, -1 + delta, -3), (cf, 0, -2)]

    def t3_accumulate(self, delta=0):
        """As FermiTriangle.accumulate, with the cubic mass vertices at V1 and V2.

        A single assignment of branches has a pole when both mass vertices go to late times (the ordered hard
        moment contains Gamma(n1 + n2 + p1 + p2 + 2)). It cancels in the sum over the branches of the hard
        vertices. `delta` shifts the power t^-1, and `t3_isosceles` averages over +-delta_t."""
        nmax, nu, ka = self.nmax, self.nu, self.kappa
        T, xis = self._traces()
        acc = {}
        two = mp.mpf(2)
        for a3 in (1, -1):
            kC, kB = ("v", "u") if a3 == 1 else ("u", "v")
            for h1, h2 in product((1, -1), repeat=2):
                hel = mp.exp(mp.pi * (h1 + h2) * ka)
                Wt = {}
                for i1, i2 in product((0, 1), repeat=2):
                    rowC = self.row(kC, h1, i1)                       # block (i1, i1) at V1
                    colB = self.col(kB, h2, i2)                       # block (i2, i2) at V2
                    for a1, a2 in product((1, -1), repeat=2):
                        cC, KeC = self.hard_end(rowC, a1)
                        cB, KeB = self.hard_end(colB, a2)
                        eC, eB = self.eser(KeC), self.eser(KeB)
                        for (f1, p1, _), (f2, p2, mX) in product(self.mass_terms(a1, i1, delta), self.mass_terms(a2, i2, delta)):
                            base = a1 * a2 * cC * cB * f1 * f2
                            for n1 in range(nmax + 1):
                                for n2 in range(nmax + 1 - n1):
                                    w0 = base * eC[n1] * eB[n2] * two ** (2 * nu + n1 + n2)
                                    N = nmax - n1 - n2
                                    for hA in (1, -1):
                                        for pA in (0, 1):
                                            ser = self.hard_series(a1, a2, hA, i1, i2, n1 + p1, n2 + p2, -pA, mX, N)
                                            d = Wt.setdefault((pA, n1, n2), {})
                                            wf = w0 * (hA if pA else 1)
                                            for k, v in ser.items():
                                                d[k] = d.get(k, 0) + wf * v
                for i3 in (0, 1):
                    cB3, KB3, _ = self.row(kB, h2, i3)
                    cC3, KC3, _ = self.col(kC, h1, 1 - i3)
                    labC, labB = self.klabel(KC3), self.klabel(KB3)
                    for f3, q, s3, _ in self.vertex_terms(a3, i3, 3):
                        soft = a3 * hel * cB3 * cC3 * f3 / 2
                        for (pA, n1, n2), d in Wt.items():
                            for pC, pB in product((0, 1), repeat=2):
                                red = T[("1", "1", s3, pC, pB, pA)]
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
        return {k: v for k, v in acc.items() if v != 0}

    def t3_isosceles(self):
        """Coefficients of T3' on k1 = k2 (both orientations), and the relative difference of the two regulated values."""
        vals = []
        for dl in (self.delta_t, -self.delta_t):
            self._q0, self._ser, self._ord = {}, {}, {}
            vals.append(self.isosceles(self.coefficients(self.t3_accumulate(dl))))
        self._q0, self._ser, self._ord = {}, {}, {}
        mean = [(x + y) / 2 for x, y in zip(*vals)]
        spread = max(abs(x - y) / abs(x + y) * 2 for x, y in zip(*vals))
        return mean, spread


def contact_coefficients(Sc, nu, m, kappa, x=0):
    """D_N(B') for N = 0..len(Sc)-1 in the normalization of fermion_triangle_shape.py, at x = R^2 delta^2/4
    (x = 0 is k1 = k2). For x != 0 the coefficients depend on R through x, so this is not a power series in R."""
    w = (1 - mp.mpf(x)) / 4
    return [-I * m * kappa ** 2 * mp.mpf(2) ** (-2 * nu - N) * Sc[N] / (4 * w) ** 3
            * (mp.gamma(2 * nu + N) + mp.gamma(1 + 2 * nu + N) + w * mp.gamma(2 + 2 * nu + N)) for N in range(len(Sc))]


def contact_symmetric(Sc, nu, m, kappa):
    """{(N, md): coefficient of R^N delta^md} of B', as ext/shapevars.symmetric gives it for the triangle
    (orders N <= len(Sc) - 1). Only even powers of delta occur, through x = R^2 delta^2/4."""
    nmax = len(Sc) - 1
    D = {}
    for N in range(nmax + 1):
        pre = -I * m * kappa ** 2 * mp.mpf(2) ** (-2 * nu - N) * Sc[N]
        g01, g2 = mp.gamma(2 * nu + N) + mp.gamma(1 + 2 * nu + N), mp.gamma(2 + 2 * nu + N) / 4
        j = 0
        while N + 2 * j <= nmax:                 # (1 - x)^-3 and (1 - x)^-2
            D[(N + 2 * j, 2 * j)] = D.get((N + 2 * j, 2 * j), 0) + pre * mp.mpf(4) ** -j * (g01 * mp.binomial(j + 2, 2) + g2 * (j + 1))
            j += 1
    return D


# ------------------------------------------------------------------------------ closed forms
def pair_closed(R, nu, r1, r2):
    """Pair signal of the model loop over its leading term, sum_n [M(n)/M(0)] (R/2)^n at first index 1/2 + nu:
    (1 + R/2)^-(r1 + 2 nu) 2F1(r1 + 2 nu, 2; 4 - r2 + 2 nu; 2R/(2 + R))."""
    R = mp.mpmathify(R)
    return (1 + R / 2) ** (-(r1 + 2 * nu)) * mp.hyp2f1(r1 + 2 * nu, 2, 4 - r2 + 2 * nu, 2 * R / (2 + R))


def bubble_closed(R, nu):
    """pair_closed at (r1, r2) = (4, 2), the Dirac bubble of dirac_pair.py, in elementary form:
    (1 + R/2)^(-2 nu) (1 - R^2/4)^-2 [1 + 4 rho/c + 6 rho^2/(c (c + 1))],  rho = 2R/(2 - R), c = 2 + 2 nu."""
    R = mp.mpmathify(R)
    rho, c = 2 * R / (2 - R), 2 + 2 * nu
    return (1 + R / 2) ** (-2 * nu) * (1 - R * R / 4) ** -2 * (1 + 4 * rho / c + 6 * rho ** 2 / (c * (c + 1)))


def eps_factor(R, nu, kappa, eta2=None):
    """First order in eps = nu - i kappa (leading in 1/nu): (1 + R/2)^(2 eps) <(1 - R^2 eta^2/4)^(-eps)>.
    eta2 = None averages uniformly over -1 <= eta <= 1 (the bubble); otherwise <eta^2> is given and only the
    order R^2 of the average is kept."""
    R = mp.mpmathify(R)
    eps = nu - I * kappa
    if eta2 is None:
        avg = 1 + eps * mp.nsum(lambda k: (R / 2) ** (2 * k) / (k * (2 * k + 1)), [1, mp.inf])
    else:
        avg = 1 + eps * R * R * eta2 / 4
    return (1 + R / 2) ** (2 * eps) * avg


def contact_leading(R, delta, kappa):
    """Leading form of B'(R, delta)/B'(0) for kappa >> m, without R^(2 nu):
    (1 + R/2)^(-2 i kappa) (1 - R/2)^-2 (1 - R^2 delta^2/4)^-2."""
    R = mp.mpmathify(R)
    return (1 + R / 2) ** (-2 * I * kappa) * (1 - R / 2) ** -2 * (1 - (R * delta / 2) ** 2) ** -2


def contact_first_order(R, nu, kappa, eta2=mp.mpf(1) / 3):
    """contact_leading on k1 = k2 with its corrections of order 1/kappa:
    exp[-R^2/(nu (2 - R)) + eps R^2 <eta^2>/4] (1 + R/2)^(-2 i kappa) (1 - R/2)^-2."""
    R = mp.mpmathify(R)
    eps = nu - I * kappa
    return contact_leading(R, 0, kappa) * mp.exp(-R * R / (nu * (2 - R)) + eps * R * R * eta2 / 4)


def segment_average(q, kappa):
    """<(y1 + y2)/k3> over the soft moment of two lines with first index 1/2 - i kappa and vertex power q
    (ext/softvertex.soft_J with A = B = nu - 1/2), for kappa >> m: the weight is (2 xi^2 - 2/3)(1 + xi)^-(q + 1 - 2 i kappa)
    on xi >= 1, continued in q. Equals 1 + i/kappa + O(kappa^-2)."""
    g = q + 1 - 2 * I * mp.mpf(kappa)
    Ia = lambda a: mp.mpf(2) ** a / (g - a - 1)
    return (2 * Ia(3) - 6 * Ia(2) + mp.mpf(16) / 3 * Ia(1) - mp.mpf(4) / 3 * Ia(0)) / (2 * Ia(2) - 4 * Ia(1) + mp.mpf(4) / 3 * Ia(0))


def tree_saddle_phase(X, mu, kappa):
    """Phase of the time integral of a tree-level exchange at its saddle point, for a line with the dispersion
    omega^2 = s^2 + 2 s kappa/tau + mu^2/tau^2 and a vertex of energy k12 = X s: Eq. (3.49) of Qin, Renaux-Petel,
    Tong, Werth, and Zhu (2506.01555) at unit sound speed, where the integral is proportional to exp(-i Phi),

        Phi = mu arccosh(X/sqrt(1 - kappa^2/mu^2)) + kappa [arccoth X - arctanh(X kappa/(mu Delta))],
        Delta = sqrt(X^2 - 1 + kappa^2/mu^2).

    Their mu is the late-time frequency, mubar here. For kappa -> mu,
    Phi = kappa ln(1 + X) + (mu - kappa) ln X + const + O((mu - kappa)^2/mu): the total energy carries the
    exponent kappa and the squeezed frequency stays mu (the statement of first order in eps, for one line)."""
    X, mu, kappa = mp.mpmathify(X), mp.mpmathify(mu), mp.mpmathify(kappa)
    e2 = 1 - (kappa / mu) ** 2
    return mu * mp.acosh(X / mp.sqrt(e2)) + kappa * (mp.acoth(X) - mp.atanh(X * kappa / (mu * mp.sqrt(X * X - e2))))


# ------------------------------------------------------------------------------ series tools
def log_series(c):
    """Coefficients L_0 = 0, L_1, ... of log(sum_n c_n R^n / c_0)."""
    a = [x / c[0] for x in c]
    L = [mp.mpc(0)] * len(a)
    for n in range(1, len(a)):
        L[n] = a[n] - mp.fsum(k * L[k] * a[n - k] for k in range(1, n)) / n
    return L


def strip_total_energy(c, nu):
    """Coefficients of (1 + R/2)^(2 nu) sum_n c_n R^n / c_0."""
    N = len(c) - 1
    b = [mp.mpc(1)]
    for j in range(1, N + 1):
        b.append(b[-1] * (2 * nu - j + 1) / j / 2)
    return [mp.fsum(c[i] / c[0] * b[n - i] for i in range(n + 1)) for n in range(N + 1)]
