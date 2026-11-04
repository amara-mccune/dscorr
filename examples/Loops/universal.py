"""The universal factor between the exact and the late-time soft vertex (Sec. 7.2, Eq. [eq:Univ]).

    U(Delta, r2) = sqrt(pi) 2^(1 + Delta - r2) Gamma(r2 - Delta) / [Gamma(r2 - 1) Gamma(3/2 - Delta)]

with Delta = 3 + 2 nu the dimension of the pair (3 - 2 nu for the leading family of light lines) and
r2 the power of the soft vertex. This script checks three things:

1. U is the ratio of the exact two-line soft moment [eq:twomoment] to the late-time moment, in which
   K_nu(y t) at the soft vertex is replaced by its branch (1/2) Gamma(-nu) (y t/2)^nu.
2. The position-space form: with the hard end late, the pair density is the Euclidean AdS
   bulk-to-boundary propagator of dimension Delta, whose Fourier transform is t^(3/2) K_(Delta-3/2)(k t).
   The ratio of the soft-vertex integral of that function to the integral of its non-analytic
   leading power is evaluated here by direct quadrature (where it converges, r2 > 3) and equals U.
3. The values quoted in Sec. 7.2.

Usage:  python universal.py            (about 5 s)
"""
import sys

import mpmath as mp

from ext.blocks import two_moment, two_moment_late
from ext.record import compare, report

mp.mp.dps = 30


def U(Delta, r2):
    """Universal factor, Eq. [eq:Univ]."""
    Delta, r2 = mp.mpmathify(Delta), mp.mpmathify(r2)
    return (mp.sqrt(mp.pi) * mp.mpf(2) ** (1 + Delta - r2) * mp.gamma(r2 - Delta)
            / (mp.gamma(r2 - 1) * mp.gamma(1.5 - Delta)))


def U_heavy(mu, r2=2):
    return U(3 + 2j * mp.mpf(mu), r2)


def U_light(nu, r2=2):
    return U(3 - 2 * mp.mpf(nu), r2)


def ratio_from_moments(nu, sigma, r2):
    """Exact two-line moment over its late-time version, equal orders nu, branch sigma on both lines."""
    return two_moment(nu, nu, sigma, sigma, r2) / two_moment_late(nu, nu, sigma, sigma, r2)


def U_position(Delta, r2):
    """Direct quadrature of the position-space soft vertex (converges for r2 > 3).

    With the hard end at late times, the soft vertex at time t sees (2t)^Delta [t'/(t'^2 + rho^2)]^Delta.
    Its Fourier transform in rho is proportional to (k/2t')^a K_a(k t'), a = Delta - 3/2, and the late-time
    estimate keeps its non-analytic leading power (1/2) Gamma(-a) (k t'/2)^a, i.e. drops t'^2 against rho^2.
    With the soft-vertex weight t'^(r2-4) e^(-t') (k = 1) the ratio of the two integrals is
        2^a int dt t^(r2 - 5/2) e^(-t) K_a(t)  /  [ (1/2) Gamma(-a) Gamma(r2 - 3 + Delta) ].
    """
    a = mp.mpmathify(Delta) - 1.5
    num = mp.quad(lambda t: t ** (r2 - 2.5) * mp.exp(-t) * mp.besselk(a, t), [0, 1, 5, 20, mp.inf])
    den = mp.gamma(-a) / 2 * mp.gamma(r2 - 3 + Delta)
    return mp.mpf(2) ** a * num / den


def main():
    res = []
    print("1. U against the ratio of the exact to the late-time two-line moment")
    worst = 0
    for mu in (0.5, 1, 2, 4):
        for r2 in (2, 3.5, 4.25):
            nu = 1j * mp.mpf(mu)
            worst = max(worst, abs(ratio_from_moments(nu, +1, r2) / U(3 + 2 * nu, r2) - 1))
    for nu in (0.3, 0.6, 0.9, 1.3):
        for r2 in (2, 3.5):
            worst = max(worst, abs(ratio_from_moments(mp.mpf(nu), -1, r2) / U(3 - 2 * mp.mpf(nu), r2) - 1))
    print(f"   heavy (mu = 0.5..4) and light (nu = 0.3..1.3), r2 in {{2, 3.5, 4.25}}: max |ratio/U - 1| = {mp.nstr(worst, 3)}")
    res.append((worst < mp.mpf(10) ** -25, worst))

    print("2. U against direct quadrature of the position-space (EAdS) soft vertex, r2 = 3.5")
    worst = 0
    for Delta in (3 + 1j, 3 + 2j, 3 + 4j, 2.4, 1.8):
        worst = max(worst, abs(U_position(Delta, 3.5) / U(Delta, 3.5) - 1))
    print(f"   Delta = 3+i, 3+2i, 3+4i, 2.4, 1.8: max |quadrature/U - 1| = {mp.nstr(worst, 3)}")
    res.append((worst < 1e-12, worst))

    print("3. Values quoted in Sec. 7.2 (r2 = 2)")
    for mu in ("0.5", "1", "2", "4"):
        res.append(compare(f"universal/|U| heavy mu={mu}", abs(U_heavy(mu))))
    for nu in ("0.3", "0.6", "1.45"):
        res.append(compare(f"universal/U light nu={nu}", U_light(nu)))
    # growth as mu^(1/2) at large mu
    slope = mp.log(abs(U_heavy(64)) / abs(U_heavy(32))) / mp.log(2)
    print(f"   d ln|U| / d ln mu at mu = 32..64: {mp.nstr(slope, 4)} (Sec. 7.2: grows as mu^(1/2))")
    res.append((abs(slope - 0.5) < 0.01, slope))
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
