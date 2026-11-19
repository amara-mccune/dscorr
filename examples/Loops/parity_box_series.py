"""The collapsed trispectrum of a helical vector loop beyond leading order in s/k (Sec. 7.4; App. B).

parity_box.py gives the leading collapsed coefficient of the model of Wang and Xianyu. This script gives the
corrections in powers of the soft momentum s = |k1 + k2| over the hard momenta, for the same three classes of
diagram (ext/boxseries.py has the method and the conventions):

    CC  two quartic vertices (the bubble),
    CP  a quartic vertex on side (12) and a hard pair on side (34), and PC, its mirror image,
    PP  a hard pair on both sides (the box), planar attachment (13)(24) of the soft lines; the crossed
        attachment (14)(23) is the same function with k3 <-> k4.

For each class the unit-vertex loop integral in the collapsed limit is a sum over families,

    I = s^3 p1^-4 p3^-4 (s/p1)^(eA nu) (s/p3)^(eB nu) sum_{NA NB} (s/p1)^NA (s/p3)^NB sum_n c_n(c1, c3) e^(i n beta),

p1 = |k1|, p3 = |k3|, c1 = k1hat.shat, c3 = k3hat.shat, beta the dihedral angle, nu = i mu:

    pair     (eA, eB) = (2, 2)     both soft lines on the same late-time branch at both ends (Sec. 7.4),
    singleA  (eA, eB) = (2, 0)     one line as above, the other on opposite branches at its two ends;
    singleB  (eA, eB) = (0, 2)     these oscillate at half the frequency, in s/k12 or in s/k34 alone.

    zero     (eA, eB) = (0, 0)     no oscillation,
    ratio    (eA, eB) = (2, -2)    (k34/k12)^(2 nu), and its conjugate; no oscillation at k12 = k34.

The single-frequency and the non-oscillating families vanish at kappa = 0 (checked here to working precision, order
by order). At kappa != 0 they do not: the part of a helical line that is symmetric in its two times has the weight
1 + 2 n_h, and its helicity-odd part multiplies eps_ijk lhat_k, which is not analytic at l = 0. The families
conjugate to the oscillating ones follow by complex conjugation at fixed harmonic. The non-oscillating families
have integer powers of the loop momenta. Where s^(3 + N) multiplies a structure that is analytic in the vector s,
their blocks have a pole and the term is s^(3 + N) ln s (see `nonoscillating`).

Normalization. With the vertices of Eq. [eq:vecvertex], H = 1, and the legs' factors 1/(2 k_i),

    16 k1 k2 k3 k4 <phi^4>'  contains  (lambda_1^2/2) I_CC + lambda_1 lambda_2^2 (I_CP + I_PC)
                                       + lambda_2^4 (I_PP + I_PP[k3 <-> k4]),

from the Wick contractions (symmetry factor 1/2 for the bubble, 1 for the others), and "full" below is
(1/2) I_CC + f (I_CP + I_PC) + f^2 (I_PP + I_PP[k3 <-> k4]) with f = lambda_2^2/lambda_1. Two choices of f:

    scheme "model"  f = -2 m_h^2. The quartic vertex is the one of the model of Wang and Xianyu with two Higgs
                    lines contracted (their diagram (c)): rho_4 = rho_3^2/(2 m_A^2), by their Eqs. (32) and (37).
    scheme "wx"     f = -m_h^2 e^(-2 pi Omega), Omega = max(kappa - mu, 0), the weights of Secs. 7.3 and 7.4 of
                    v5.12 and of parity_box.py. There the quartic coupling is their Eq. (34), the hard line of the
                    triangle contracted to a point, so that adding it to the hard pair counts that line twice.

At leading order and for k1hat, k3hat perpendicular to s, "full" in the scheme "wx" is 128 times the "full" of
parity_box.py in the even harmonics. The odd harmonics of the box cancel there between the planar and the crossed
attachment (parity_box.py has the planar one twice).

Benchmark configuration: k1 = k2 = k3 = k4 = k, so that c1 = r/2, c3 = -r/2 with r = s/k, and the coefficients
are quoted as series in r. There the crossed box is the planar one with beta -> beta + pi.

Checks (all are run by the script; see CHECKS in the notes of the Nov 19 run):
    the N = 0 term against parity_box.py; kappa = 0 (no parity-odd part, no single-frequency family);
    kappa -> -kappa exchanges c_n and c_-n; exchange of the two sides; exchange k1 <-> k2 together with
    k3 <-> k4 (planar box), and of the two legs of one side (the other classes); the heavy-hard-line limit
    (a hard pair tends to -1/m_h^2 times the contact vertex); independence of the working precision.

Usage:
    python parity_box_series.py                 (mu, kappa) = (1, 1) through N = 4 with all checks
    python parity_box_series.py --all           also (1, 0.5) and (2, 2)
    python parity_box_series.py --nmax 6        higher orders (the geometry takes 2.5 min to build at N = 4 and
                                                about 30 min and 2 GB at N = 6; it is cached in cache/)
    python parity_box_series.py --nonosc        also the non-oscillating families (5 min per point at N = 4)
    python parity_box_series.py --quick         N = 2, for tests
    python parity_box_series.py --dump F.json   coefficients of the benchmark configuration, per class
    python parity_box_series.py --template F.json   all coefficients C_{NA NB}(c1, c3) e^(i n beta), general shape
"""
import argparse
import json
import os
import sys
import time

import mpmath as mp

from ext.boxseries import FAMILIES, CollapsedSeries, Geometry, harmonics
from ext.record import compare, report

I = mp.mpc(0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DIAGRAMS = ("CC", "CP", "PC", "PP")
OSC = ("pair", "singleA", "singleB")
NONOSC = ("zero", "ratio", "ratio_c")
EXPONENTS = {"pair": (2, 2), "singleA": (2, 0), "singleB": (0, 2), "zero": (0, 0), "ratio": (2, -2), "ratio_c": (-2, 2)}
BENCHMARKS = (("1", "1"), ("1", "0.5"), ("2", "2"))

_GEO = {}


def rec(res, key, value, quiet=False):
    """Compare with the value of record if there is one."""
    try:
        res.append(compare(key, value, quiet=quiet))
        if quiet and not res[-1][0]:
            print(f"   FAIL {key}: {mp.nstr(value, 8)}")
    except KeyError:
        if not quiet:
            print(f"  [new ] {key:<46s} computed {mp.nstr(value, 10):>28s}")


def geometry(nmax):
    if nmax not in _GEO:
        _GEO[nmax] = Geometry(nmax, CACHE)
    return _GEO[nmax]


def coupling_ratio(mu, kappa, mu_h=None, scheme="model"):
    """f = lambda_2^2/lambda_1 (see the module docstring)."""
    mu, kappa = mp.mpf(mu), mp.mpf(kappa)
    mu_h = mu if mu_h is None else mp.mpf(mu_h)
    m2 = mu_h ** 2 + mp.mpf(1) / 4
    if scheme == "model":
        return -2 * m2
    if scheme == "wx":
        return -m2 * mp.exp(-2 * mp.pi * max(kappa - mu, 0))
    raise ValueError(scheme)


class Collapsed:
    """All diagram classes of the oscillating families at one point (mu, kappa)."""

    def __init__(self, mu, kappa, nmax, families=("pair",), mu_h=None, kappa_h=None, diagrams=DIAGRAMS,
                 scheme="model", eps=0, verbose=False):
        self.mu, self.kappa, self.nmax = mp.mpf(mu), mp.mpf(kappa), nmax
        self.mu_h = mu_h
        self.S = CollapsedSeries(mu, kappa, mu_h, kappa_h, nmax, geometry(nmax), eps=eps)
        self.set_scheme(scheme)
        self.coefs = {}
        for fam in families:
            for d in diagrams:
                t0 = time.time()
                self.coefs[(fam, d)] = self.S.family(fam, d)
                if verbose:
                    print(f"      {fam:8s} {d}: {time.time() - t0:5.1f} s", flush=True)

    def set_scheme(self, scheme):
        self.scheme = scheme
        self.f = coupling_ratio(self.mu, self.kappa, self.mu_h, scheme)
        return self

    def weight(self, d):
        return {"CC": mp.mpf(1) / 2, "CP": self.f, "PC": self.f, "PP": self.f ** 2}[d]

    # ---- general kinematics
    def series(self, fam, d, c1, c3, rA, rB, orders=None):
        """{n: sum over orders of rA^NA rB^NB c_n} for one diagram class (planar attachment for PP)."""
        return harmonics(self.coefs[(fam, d)], c1, c3, rA, rB, orders)

    def full(self, fam, p1, c1, p3, c3, s=1, orders=None, parts=False):
        """Harmonics {n: value} of the weighted sum, with the powers of p1, p3 and s included (unit s by default):
        (1/2) I_CC + f (I_CP + I_PC) + f^2 (I_PP + I_PP[k3 <-> k4])."""
        eA, eB = EXPONENTS[fam]
        nu = I * self.mu
        p1, c1, p3, c3, s = (mp.mpf(x) for x in (p1, c1, p3, c3, s))
        out, byd = {}, {}

        def add(d, w, p3_, c3_, sign):
            rA, rB = s / p1, s / p3_
            pre = w * s ** 3 * p1 ** -4 * p3_ ** -4 * rA ** (eA * nu) * rB ** (eB * nu)
            for n, v in self.series(fam, d, c1, c3_, rA, rB, orders).items():
                val = pre * v * (sign ** n)
                out[n] = out.get(n, 0) + val
                byd.setdefault(d, {})[n] = byd.setdefault(d, {}).get(n, 0) + val
        for d in DIAGRAMS:
            if (fam, d) in self.coefs:
                add(d, self.weight(d), p3, c3, 1)
        if (fam, "PP") in self.coefs:               # crossed box: k3 <-> k4
            q = mp.sqrt(1 + 2 * (s / p3) * c3 + (s / p3) ** 2)
            add("PP", self.weight("PP"), p3 * q, -(s / p3 + c3) / q, -1)
        return (out, byd) if parts else out

    # ---- the benchmark configuration k1 = k2 = k3 = k4
    def equal_k(self, fam, d, crossed=True):
        """{n: [c^(0), ..., c^(nmax)]}: coefficients of r^N, r = s/k, at c1 = r/2, c3 = -r/2 (unit weights).
        For PP the crossed attachment is added (it multiplies the harmonic n by (-1)^n)."""
        N = self.nmax
        out = {}
        for (NA, NB), poly in self.coefs[(fam, d)].items():
            for (a, b, c, e, n), v in poly.items():
                # S1^a S3^c = (1 - r^2/4)^((a + c)/2), c1^b c3^e = (r/2)^b (-r/2)^e
                base = NA + NB + b + e
                if base > N:
                    continue
                pref = v * mp.mpf(1) / 2 ** (b + e) * (-1) ** e
                q, cj, j = mp.mpf(a + c) / 2, mp.mpf(1), 0
                while base + 2 * j <= N:
                    out.setdefault(n, [mp.mpc(0)] * (N + 1))[base + 2 * j] += pref * cj
                    cj = cj * (q - j) / (j + 1) * (-mp.mpf(1) / 4)
                    j += 1
        if d == "PP" and crossed:
            out = {n: [x * (1 + (-1) ** n) for x in v] for n, v in out.items()}
        return out

    def equal_k_full(self, fam):
        """The weighted sum of all classes in the benchmark configuration: {n: [c^(0), ..., c^(nmax)]}."""
        out = {}
        for d in DIAGRAMS:
            if (fam, d) not in self.coefs:
                continue
            for n, v in self.equal_k(fam, d).items():
                tot = out.setdefault(n, [mp.mpc(0)] * (self.nmax + 1))
                for k, x in enumerate(v):
                    tot[k] += self.weight(d) * x
        return out


def partial(cs, r, nmax):
    """sum_{N <= nmax} c^(N) r^N"""
    return mp.fsum(c * mp.mpf(r) ** k for k, c in enumerate(cs[:nmax + 1]))


def odd_fraction_r(full, r, nmax):
    """Parity-odd part over the monopole as in Sec. 7.4, max_n |c_n - c_-n|/(2 |c_0|), with the series through nmax."""
    c0 = partial(full[0], r, nmax)
    ns = sorted({abs(n) for n in full if n})
    zero = [mp.mpc(0)] * (nmax + 1)
    return max(abs(partial(full.get(n, zero), r, nmax) - partial(full.get(-n, zero), r, nmax)) for n in ns) / (2 * abs(c0))


# ------------------------------------------------------------------------------ non-oscillating families
def nonoscillating(mu, kappa, nmax, delta="1e-8", scheme="model", diagrams=DIAGRAMS):
    """The families with integer powers of the loop momenta (eA + eB = 0) in the benchmark configuration.

    Their blocks have poles where the result is s^(3 + N) times a structure that is analytic in the vector s, and
    the pole is a logarithm of s. With the analytic regulator y1^(2 eps) (eps = +-delta) a coefficient is
    R/eps + F, and the family is  s^3 k^-8 sum_N r^N sum_n e^(i n beta) [F_n^(N) + 2 R_n^(N) ln r].
    By parity under shat -> -shat, a term cos(n beta) r^N is of the power type for even N and of the logarithmic
    type for odd N, and a term sin(n beta) r^N the other way round. Only R is meaningful for the logarithmic
    type (F shifts with the scale of the logarithm, which the hard region sets), and R must vanish for the power
    type. Returns {"cos": {n: [(F, 2R), ...]}, "sin": {n: [...]}} with cos: c_n + c_-n (n >= 0, c_0 for n = 0) and
    sin: i (c_n - c_-n), so that the family is sum_n [cos(n beta) cos_n + sin(n beta) sin_n].
    """
    delta = mp.mpf(delta)
    zero = [mp.mpc(0)] * (nmax + 1)
    with mp.workdps(mp.mp.dps + 10):
        tot = {}
        for sgn in (1, -1):
            C = Collapsed(mu, kappa, nmax, NONOSC, diagrams=diagrams, scheme=scheme, eps=sgn * delta)
            acc = {}
            for fam in NONOSC:
                for n, v in C.equal_k_full(fam).items():
                    acc[n] = [x + y for x, y in zip(acc.get(n, zero), v)]
            tot[sgn] = acc
        out = {"cos": {}, "sin": {}}
        ns = sorted({abs(n) for n in tot[1]})
        for n in ns:
            def comb(sgn, kind):
                p, m = tot[sgn].get(n, zero), tot[sgn].get(-n, zero)
                if n == 0:
                    return p if kind == "cos" else zero
                return [x + y for x, y in zip(p, m)] if kind == "cos" else [I * (x - y) for x, y in zip(p, m)]
            for kind in ("cos", "sin"):
                if n == 0 and kind == "sin":
                    continue
                vp, vm = comb(1, kind), comb(-1, kind)
                out[kind][n] = [((a + b) / 2, delta * (a - b)) for a, b in zip(vp, vm)]
    return out


# ------------------------------------------------------------------------------ checks
def check_leading(C):
    """N = 0 against parity_box.py at generic angles: the planar box, and the sum with the weights of Sec. 7.3."""
    from parity_box import CollapsedBox
    B = CollapsedBox(C.mu, C.kappa)
    a1, a3 = mp.mpf("0.7"), mp.mpf("1.1")
    c1, c3 = mp.cos(a1), mp.cos(a3)
    box = B.harmonics(a1, a3)
    mine = C.series("pair", "PP", c1, c3, 1, 1, orders={0})
    w1 = max(abs(mine[n] / box[n] - 1) for n in range(-2, 3))
    full = B.harmonics(a1, a3, weights=(B.cc, B.cp))
    w = {"CC": mp.mpf(1) / 256, "CP": B.cp / 16, "PC": B.cp / 16, "PP": B.cp ** 2}
    tot = {}
    for d in DIAGRAMS:
        for n, v in C.series("pair", d, c1, c3, 1, 1, orders={0}).items():
            tot[n] = tot.get(n, 0) + w[d] * v
    w2 = max(abs(tot[n] / full[n] - 1) for n in range(-2, 3))
    return max(w1, w2)


def _scale(C, fam):
    """Largest coefficient of the family at a generic point (to set the scale of 'zero')."""
    return max(abs(v) for d in DIAGRAMS for h in harmonics(C.coefs[(fam, d)], "0.3", "-0.2").values() for v in h.values())


def check_parity(C0):
    """kappa = 0: c_n = c_-n at every order, and the single-frequency families vanish."""
    worst_odd, worst_single = mp.mpf(0), mp.mpf(0)
    sc = _scale(C0, "pair")
    for (fam, d), acc in C0.coefs.items():
        for h in harmonics(acc, "0.3", "-0.2").values():
            for n, v in h.items():
                if fam == "pair":
                    worst_odd = max(worst_odd, abs(v - h.get(-n, 0)) / sc)
                else:
                    worst_single = max(worst_single, abs(v) / sc)
    return worst_odd, worst_single


def check_reflection(C, Cm):
    """kappa -> -kappa exchanges c_n and c_-n, at every order and for every class."""
    worst = mp.mpf(0)
    for key, acc in C.coefs.items():
        sc = _scale(C, key[0])
        a, b = harmonics(acc, "0.3", "-0.2"), harmonics(Cm.coefs[key], "0.3", "-0.2")
        for k, h in a.items():
            for n, v in h.items():
                worst = max(worst, abs(v - b[k].get(-n, 0)) / sc)
    return worst


def check_sides(C):
    """Exchange of the two sides: C_{NA NB}(c1, c3) = C_{NB NA}(-c3, -c1) for CC and PP, and CP <-> PC."""
    worst = mp.mpf(0)
    c1, c3 = mp.mpf("0.3"), mp.mpf("-0.2")
    for (fam, d), acc in C.coefs.items():
        fam2 = {"pair": "pair", "singleA": "singleB", "singleB": "singleA"}[fam]
        if (fam2, d[::-1]) not in C.coefs:
            continue
        sc = _scale(C, fam)
        a, b = harmonics(acc, c1, c3), harmonics(C.coefs[(fam2, d[::-1])], -c3, -c1)
        for (NA, NB), h in a.items():
            for n, v in h.items():
                worst = max(worst, abs(v - b[(NB, NA)].get(n, 0)) / sc)
    return worst


def check_exchange(C, fam="pair", dps=50):
    """Exchange of the legs within a side, which mixes the orders of the series.

    With qA = |k2|/|k1| = sqrt(1 - 2 rA c1 + rA^2) and c2 = (rA - c1)/qA, and qB = |k4|/|k3|, c4 = -(rB + c3)/qB,
    the series part F(rA, c1; rB, c3) of each harmonic obeys
        planar box:  F = qA^(-4 - eA nu) qB^(-4 - eB nu) F(rA/qA, c2; rB/qB, c4)       (k1 <-> k2 and k3 <-> k4),
        CP:          F = qB^(-4 - eB nu) (-1)^n F(rA, c1; rB/qB, c4)                    (k3 <-> k4),
        CC, CP:      F = qA^(-4 - eA nu) (-1)^n F(rA/qA, c2; rB, c3)                    (k1 <-> k2),
    and the mirror statements for PC. Both sides are expanded in t along rA = a t, rB = b t, through t^nmax.
    Returns the largest difference over the largest coefficient.
    """
    eA, eB = EXPONENTS[fam]
    nu = I * C.mu
    c1, c3 = mp.mpf("0.3"), mp.mpf("-0.2")
    N = C.nmax
    worst = mp.mpf(0)
    with mp.workdps(dps):
        for d, swapA, swapB in (("PP", True, True), ("CP", False, True), ("CP", True, False), ("PC", True, False),
                                ("PC", False, True), ("CC", True, False), ("CC", False, True)):
            if (fam, d) not in C.coefs:
                continue
            acc = C.coefs[(fam, d)]
            ns = sorted({k[4] for poly in acc.values() for k in poly})
            for a, b in ((mp.mpf(1), mp.mpf("0.7")), (mp.mpf("0.4"), mp.mpf("1.3"))):
                lhs = {n: mp.taylor(lambda t, n=n: harmonics(acc, c1, c3, a * t, b * t).get(n, mp.mpc(0)), 0, N) for n in ns}

                def rhs(t, n):
                    rA, rB = a * t, b * t
                    qA = mp.sqrt(1 - 2 * rA * c1 + rA ** 2) if swapA else mp.mpf(1)
                    qB = mp.sqrt(1 + 2 * rB * c3 + rB ** 2) if swapB else mp.mpf(1)
                    c2 = (rA - c1) / qA if swapA else c1
                    c4 = -(rB + c3) / qB if swapB else c3
                    sgn = (-1) ** (n * (swapA + swapB))
                    return sgn * qA ** (-4 - eA * nu) * qB ** (-4 - eB * nu) * \
                        harmonics(acc, c2, c4, rA / qA, rB / qB).get(n, mp.mpc(0))
                sc = max(abs(x) for v in lhs.values() for x in v)
                for n in ns:
                    r_ = mp.taylor(lambda t: rhs(t, n), 0, N)
                    worst = max(worst, max(abs(x - y) for x, y in zip(lhs[n], r_)) / sc)
    return worst


def check_heavy(mu, kappa, nmax, mus=(20, 40), families=OSC):
    """Heavy hard line (kappa_h = 0): a hard pair tends to -1/m_h^2 times the contact vertex, with corrections of
    order mu_h^-2. Returns {mu_h: (w1, w2)}: the largest |-m_h^2 CP - CC| and |m_h^4 PP - CC| over all families,
    orders and harmonics at generic angles, in units of the largest coefficient of CC in the family."""
    out = {}
    for mh in mus:
        C = Collapsed(mu, kappa, nmax, families, mu_h=mh, kappa_h=0)
        m2 = mp.mpf(mh) ** 2 + mp.mpf(1) / 4
        w1 = w2 = mp.mpf(0)
        for fam in families:
            cc = harmonics(C.coefs[(fam, "CC")], "0.3", "-0.2")
            cp = harmonics(C.coefs[(fam, "CP")], "0.3", "-0.2")
            pp = harmonics(C.coefs[(fam, "PP")], "0.3", "-0.2")
            sc = max(abs(v) for h in cc.values() for v in h.values())
            for k in pp:
                for n in set(pp[k]) | set(cc.get(k, {})):
                    ref = cc.get(k, {}).get(n, 0)
                    w1 = max(w1, abs(-m2 * cp.get(k, {}).get(n, 0) - ref) / sc)
                    w2 = max(w2, abs(m2 ** 2 * pp[k].get(n, 0) - ref) / sc)
        out[mh] = (w1, w2)
    return out


def check_precision(mu, kappa, nmax, C, dps=35):
    """The same coefficients at a higher working precision."""
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        C2 = Collapsed(mu, kappa, nmax, families=sorted({k[0] for k in C.coefs}))
        worst = mp.mpf(0)
        for key, acc in C.coefs.items():
            sc = _scale(C, key[0])
            a, b = harmonics(acc, "0.3", "-0.2"), harmonics(C2.coefs[key], "0.3", "-0.2")
            for k, h in a.items():
                for n, v in h.items():
                    worst = max(worst, abs(v - b[k].get(n, 0)) / sc)
    finally:
        mp.mp.dps = old
    return worst


# ------------------------------------------------------------------------------ output
def _rows(tab, floor):
    return sorted(n for n in tab if max(abs(x) for x in tab[n]) > floor)


def single_full(C):
    """singleA + singleB in the benchmark configuration (p1 = p3, so both carry r^(2 i mu))."""
    a, b = C.equal_k_full("singleA"), C.equal_k_full("singleB")
    zero = [mp.mpc(0)] * (C.nmax + 1)
    return {n: [x + y for x, y in zip(a.get(n, zero), b.get(n, zero))] for n in set(a) | set(b)}


def classes(C, fam):
    """Unit-weight coefficients per class in the benchmark configuration: CC, CP + PC, PP (planar + crossed)."""
    zero = [mp.mpc(0)] * (C.nmax + 1)
    fams = ("singleA", "singleB") if fam == "single" else (fam,)
    out = {}
    for name, ds in (("CC", ("CC",)), ("CP+PC", ("CP", "PC")), ("PP", ("PP",))):
        tot = {}
        for f in fams:
            for d in ds:
                for n, v in C.equal_k(f, d).items():
                    tot[n] = [x + y for x, y in zip(tot.get(n, zero), v)]
        out[name] = tot
    return out


def print_table(C, tab, label, res, key, fam):
    floor = mp.mpf(10) ** (-mp.mp.dps + 8) * max(abs(x) for v in tab.values() for x in v)
    ns = _rows(tab, floor)
    lead = max(abs(tab[n][0]) for n in ns)
    print(f"   {label}")
    for n in ns:
        print(f"     n = {n:+d}: " + "  ".join(f"{mp.nstr(x, 5):>28s}" for x in tab[n]))
    print("     moduli over the largest leading harmonic:")
    for n in ns:
        print(f"     n = {n:+d}: " + "  ".join(f"{mp.nstr(abs(x) / lead, 4):>9s}" for x in tab[n]))
    for n in ns:
        for k in range(C.nmax + 1):
            rec(res, f"parity_box_series/{key} {fam} c_{n:+d} N={k}", tab[n][k], quiet=True)


def run_point(mu, kappa, nmax, res, families=OSC, checks=True):
    key = f"mu={mu} kappa={kappa}"
    print(f"== (mu, kappa) = ({mu}, {kappa}), through N = {nmax}")
    t0 = time.time()
    C = Collapsed(mu, kappa, nmax, families)
    print(f"   series: {time.time() - t0:.0f} s.  Benchmark configuration k1 = k2 = k3 = k4 = k; coefficients of r^N, r = s/k,")
    print(f"   of (1/2) I_CC + f (I_CP + I_PC) + f^2 (I_PP + crossed), f = -2 m_h^2 = {mp.nstr(C.f, 5)} (scheme 'model')")
    out = {"pair": C.equal_k_full("pair")}
    print_table(C, out["pair"], "pair family, s^3 k^-8 r^(4 i mu) sum_n c_n e^(i n beta):", res, key, "pair")
    if "singleA" in families:
        out["single"] = single_full(C)
        print_table(C, out["single"], "single-frequency families (singleA + singleB), s^3 k^-8 r^(2 i mu) sum_n c_n e^(i n beta):",
                    res, key, "single")
    lead = max(out["pair"], key=lambda n: abs(out["pair"][n][0]))
    rat = [abs(out["pair"][lead][k + 1] / out["pair"][lead][k]) for k in range(nmax)]
    print(f"   ratios |c^(N+1)/c^(N)| of the harmonic n = {lead:+d} of the pair family: " + ", ".join(mp.nstr(x, 3) for x in rat))
    print("   parity-odd part over the monopole of the pair family, max_n |c_n - c_-n|/(2 |c_0|):")
    print(f"       r      model: N <= 0   N <= {min(2, nmax)}   N <= {nmax}      wx: N <= 0   N <= {nmax}")
    wx = C.set_scheme("wx").equal_k_full("pair")
    C.set_scheme("model")
    out["odd_fraction"] = {}
    for r in ("0.02", "0.05", "0.1", "0.2", "0.3"):
        vm = [odd_fraction_r(out["pair"], r, k) for k in (0, min(2, nmax), nmax)]
        vw = [odd_fraction_r(wx, r, k) for k in (0, nmax)]
        print(f"     {r:>5s}        " + "   ".join(f"{mp.nstr(v, 4):>8s}" for v in vm) + "       " + "   ".join(f"{mp.nstr(v, 4):>8s}" for v in vw))
        out["odd_fraction"][f"odd fraction r={r} N<={nmax}"] = vm[-1]
        out["odd_fraction"][f"odd fraction (wx) r={r} N<={nmax}"] = vw[-1]
        rec(res, f"parity_box_series/{key} odd fraction r={r} N<={nmax}", vm[-1], quiet=True)
        rec(res, f"parity_box_series/{key} odd fraction (wx) r={r} N<={nmax}", vw[-1], quiet=True)
    out["classes"] = {fam: classes(C, fam) for fam in (("pair", "single") if "singleA" in families else ("pair",))}
    out["f"] = {"model": C.f, "wx": coupling_ratio(mu, kappa, scheme="wx")}
    if not checks:
        return C, out
    print("   checks")
    w = check_leading(C)
    print(f"     N = 0 against parity_box.py (planar box and weighted sum, generic angles): {mp.nstr(w, 3)}")
    rec(res, "parity_box_series/leading order vs parity_box", w)
    w = check_sides(C)
    print(f"     exchange of the two sides, every order and harmonic: {mp.nstr(w, 3)}")
    rec(res, "parity_box_series/exchange of sides", w)
    for fam in families:
        w = check_exchange(C, fam)
        print(f"     exchange of the legs within a side ({fam}), through t^{nmax}: {mp.nstr(w, 3)}")
        rec(res, "parity_box_series/exchange of legs", w)
    Cm = Collapsed(mu, "-" + str(kappa), nmax, families)
    w = check_reflection(C, Cm)
    print(f"     kappa -> -kappa exchanges c_n and c_-n: {mp.nstr(w, 3)}")
    rec(res, "parity_box_series/kappa reflection", w)
    C0 = Collapsed(mu, 0, nmax, families)
    w1, w2 = check_parity(C0)
    print(f"     kappa = 0: parity-odd part {mp.nstr(w1, 3)}; single-frequency families {mp.nstr(w2, 3)}")
    rec(res, "parity_box_series/kappa=0 odd part", w1)
    rec(res, "parity_box_series/kappa=0 single-frequency families", w2)
    hv = check_heavy(mu, kappa, nmax, families=families)
    for mh, (w1, w2) in hv.items():
        print(f"     heavy hard line mu_h = {mh}: |-m_h^2 CP - CC| <= {mp.nstr(w1, 3)}, |m_h^4 PP - CC| <= {mp.nstr(w2, 3)} "
              "(all families, orders and harmonics; units of the largest CC coefficient)")
    r1 = hv[20][0] / hv[40][0]
    r2 = hv[20][1] / hv[40][1]
    print(f"     ratio of the deviations at mu_h = 20 and 40: {mp.nstr(r1, 3)} (CP), {mp.nstr(r2, 3)} (PP); 4 for mu_h^-2")
    rec(res, "parity_box_series/heavy hard line CP mu_h=40", hv[40][0])
    rec(res, "parity_box_series/heavy hard line PP mu_h=40", hv[40][1])
    rec(res, "parity_box_series/heavy hard line ratio 20/40", r2)
    w = check_precision(mu, kappa, nmax, C)
    print(f"     {mp.mp.dps} against 35 digits: {mp.nstr(w, 3)}")
    rec(res, "parity_box_series/precision 25 vs 35", w)
    return C, out


def _ser(v):
    return [[mp.nstr(x.real, 20), mp.nstr(x.imag, 20)] for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nochecks", action="store_true")
    ap.add_argument("--point", nargs=2, metavar=("MU", "KAPPA"))
    ap.add_argument("--nonosc", action="store_true")
    ap.add_argument("--dump", help="write the coefficients of the benchmark configuration to this JSON file")
    ap.add_argument("--template", help="write all coefficients (general shape) to this JSON file")
    args = ap.parse_args()
    mp.mp.dps = 25
    nmax = 2 if args.quick else args.nmax
    points = [tuple(args.point)] if args.point else (BENCHMARKS if args.all else BENCHMARKS[:1])
    res, dump, tmpl = [], {}, {}
    t0 = time.time()
    geometry(nmax)
    print(f"geometry through N = {nmax}: {time.time() - t0:.0f} s")
    for i, (mu, ka) in enumerate(points):
        C, out = run_point(mu, ka, nmax, res, checks=(i == 0 and not args.nochecks))
        key = f"{mu},{ka}"
        dump[key] = {
            "f": {k: mp.nstr(v, 20) for k, v in out["f"].items()},
            "odd_fraction": {k: mp.nstr(v, 15) for k, v in out["odd_fraction"].items()},
            "full_model": {fam: {str(n): _ser(v) for n, v in out[fam].items()} for fam in ("pair", "single") if fam in out},
            "classes": {fam: {cl: {str(n): _ser(v) for n, v in d.items()} for cl, d in cls.items()}
                        for fam, cls in out["classes"].items()},
        }
        if args.nonosc:
            t0 = time.time()
            no = nonoscillating(mu, ka, nmax)
            print(f"   non-oscillating families (zero + ratio + conjugate; {time.time() - t0:.0f} s):")
            print("   s^3 k^-8 sum_N r^N [F + L ln r] for each of cos(n beta) and sin(n beta). Each order has F alone (a power")
            print("   term) or a logarithm, of which L is given (its finite part is not fixed by the soft region)")
            for kind in ("cos", "sin"):
                for n, v in no[kind].items():
                    if max(abs(F) + abs(L) for F, L in v) < 1e-15:
                        continue
                    cells = []
                    for k, (F, L) in enumerate(v):
                        log_type = (k % 2 == 1) if kind == "cos" else (k % 2 == 0)
                        val = L if log_type else F
                        val = val.real if kind == "cos" else val.imag
                        cells.append(f"{'L' if log_type else 'F'} = {mp.nstr(val, 6):>12s}")
                        rec(res, f"parity_box_series/mu={mu} kappa={ka} nonosc {kind} n={n} N={k}", val, quiet=True)
                    print(f"     {kind}({n} beta){' x i' if kind == 'sin' else '    '}: " + "   ".join(cells))
            dump[key]["nonosc_model"] = {kind: {str(n): [[mp.nstr(F, 20), mp.nstr(L, 20)] for F, L in v] for n, v in d.items()}
                                         for kind, d in no.items()}
        if args.template:
            tmpl[key] = {f"{fam} {d}": {f"{NA},{NB}": {",".join(map(str, k)): [mp.nstr(v.real, 20), mp.nstr(v.imag, 20)]
                                                       for k, v in poly.items()} for (NA, NB), poly in acc.items()}
                         for (fam, d), acc in C.coefs.items()}
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dump, f, indent=1)
    if args.template:
        meta = {"key": "S1^a c1^b S3^c c3^d e^(i n beta) -> 'a,b,c,d,n': [Re, Im]; orders 'NA,NB'; see the module docstring",
                "nmax": nmax}
        with open(args.template, "w") as f:
            json.dump({"meta": meta, "points": tmpl}, f)
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
