"""Shapes of a squeezed-limit series: exchange of the two hard legs, and variables symmetric under it.

A pair-family series of a triangle with soft leg k3 is computed as (k1 = 1, r = k3/k1, c = k1hat . k3hat)

    F(k1, k2, k3) = k1^(-dim) sum_n C_n(c) r^(2 nu + n),

with polynomials C_n(c) stored as {(n, mc): coefficient of r^n c^mc}. Here dim = 6 for <dtau^3>' of the fermion
triangle, and for the dimensionless shape function S = (k1 k2 k3)^2 <phi^3>' ~ r^(2 + 2 nu + n) one uses dim = 2
(the powers of r are then counted from r^(2 nu), i.e. the argument nu is the exponent without the 2).

Exchange. F is symmetric under k1 <-> k2. With x2 = |k2|/k1 = sqrt(1 + 2 r c + r^2) and c2 = k2hat . k3hat
= -(c + r)/x2 this reads

    sum_n C_n(c) r^n = sum_n C_n(c2) r^n x2^(-dim - 2 nu - n)

order by order in r, as polynomials in c. The identity mixes orders, so it tests the expansion of the second
hard leg and of the hard line together.

Symmetric variables. At fixed c the series in r converges only for r < 1 (branch points at |k2| = 0). With

    k1 + k2 = 2 K,   R = 2 k3/(k1 + k2),   delta = (k1 - k2)/k3,
    r = R/(1 + R delta/2),   c = -(delta + R/2)/(1 + R delta/2),

the same function is  K^(-dim) sum_N D_N(delta) R^(2 nu + N), the exchange is delta -> -delta, every D_N is an
even polynomial in delta, and R = r, delta = 0 on k1 = k2. The triangle inequalities are R <= 2, |delta| <= 1,
and R <= 1 when k3 is the shortest side.
"""
import mpmath as mp

__all__ = ["exchanged", "exchange_error", "symmetric", "at_delta", "at_angle"]


def _pm(a, b, N):
    out = {}
    for (n1, m1), v1 in a.items():
        for (n2, m2), v2 in b.items():
            if n1 + n2 <= N:
                out[(n1 + n2, m1 + m2)] = out.get((n1 + n2, m1 + m2), 0) + v1 * v2
    return out


def _pa(*terms):
    out = {}
    for c, p in terms:
        for k, v in p.items():
            out[k] = out.get(k, 0) + c * v
    return out


def _binser(x, beta, N):
    """(1 + x)^beta for a series x that starts at order r"""
    out, xp, cj = {(0, 0): mp.mpc(1)}, {(0, 0): mp.mpc(1)}, mp.mpc(1)
    for j in range(1, N + 1):
        cj = cj * (beta - j + 1) / j
        xp = _pm(xp, x, N)
        out = _pa((1, out), (cj, xp))
    return out


def _compose(coef, cser, power_of, N):
    """sum_n C_n(cser) r^n (1 + x)^(power_of(n)) for the series cser and the factor returned by power_of(n)"""
    mcmax = max(k[1] for k in coef)
    cp = [{(0, 0): mp.mpc(1)}]
    for _ in range(mcmax):
        cp.append(_pm(cp[-1], cser, N))
    out = {}
    for n in range(N + 1):
        Cn = _pa(*[(val, cp[mc]) for (nn, mc), val in coef.items() if nn == n])
        if Cn:
            out = _pa((1, out), (1, _pm(_pm(Cn, power_of(n), N), {(n, 0): mp.mpc(1)}, N)))
    return {k: v for k, v in out.items() if v != 0}


def exchanged(coef, nu, N, dim=6):
    """The right-hand side of the exchange identity, as {(n, mc): coefficient}."""
    v = {(1, 1): mp.mpf(2), (2, 0): mp.mpf(1)}                                    # x2^2 - 1
    c2 = _pm({(0, 1): mp.mpf(-1), (1, 0): mp.mpf(-1)}, _binser(v, -mp.mpf(1) / 2, N), N)
    return _compose(coef, c2, lambda n: _binser(v, (-dim - 2 * nu - n) / 2, N), N)


def exchange_error(coef, nu, N, dim=6):
    """{n: largest |C_n - exchanged C_n| over the powers of c, relative to the largest coefficient of that order}"""
    ex = exchanged(coef, nu, N, dim)
    worst = {}
    for k in set(coef) | set(ex):
        scale = max((abs(x) for (n, mc), x in coef.items() if n == k[0]), default=mp.mpf(1))
        worst[k[0]] = max(worst.get(k[0], 0), abs(coef.get(k, 0) - ex.get(k, 0)) / scale)
    return dict(sorted(worst.items()))


def symmetric(coef, nu, N, dim=6):
    """{(N, md): coefficient of R^N delta^md} of the series in the symmetric variables."""
    x = {(1, 1): mp.mpf(1) / 2}                                                   # R delta/2
    c = _pm({(0, 1): mp.mpf(-1), (1, 0): -mp.mpf(1) / 2}, _binser(x, -1, N), N)
    return _compose(coef, c, lambda n: _binser(x, -dim - 2 * nu - n, N), N)


def at_delta(D, delta, N):
    """[D_0(delta), ..., D_N(delta)]"""
    out = [mp.mpc(0)] * (N + 1)
    for (n, md), v in D.items():
        out[n] += v * mp.mpf(delta) ** md
    return out


def at_angle(coef, c, N):
    """[C_0(c), ..., C_N(c)]"""
    return at_delta(coef, c, N)
