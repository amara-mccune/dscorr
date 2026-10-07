"""The kernel identity (Letter Eq. (2)) and its Barnes building blocks, in arbitrary precision.

    K_nu(y t_>) I_nu(y t_<) = 1/2 iint ds ds'/(2 pi i)^2
        Gamma(-s) Gamma(-s') Gamma(b) Gamma(1-b) / [Gamma(nu+s+1) Gamma(nu+s'+1)]
        (y/2)^(2b-2) t^(nu+2s) t'^(nu+2s'),     b = 1 + nu + s + s'.

All functions set the mpmath working precision locally to ``dps`` digits (default 30)
and accept numbers or strings: a string containing ``j`` is a complex number, one
containing ``/`` a ratio of two reals, any other a real.
"""
from __future__ import annotations

from collections import namedtuple

import mpmath as mp

__all__ = ["kernel_identity_lhs", "kernel_identity_rhs", "kernel_identity_check",
           "IdentityCheck", "NumericVsClosedForm", "mellin_transform_lhs",
           "mellin_transform_rhs", "K_barnes", "barnes_2F1_lemma"]

IdentityCheck = namedtuple("IdentityCheck", ["lhs", "rhs", "rel"])
NumericVsClosedForm = namedtuple("NumericVsClosedForm", ["numeric", "closed_form", "rel"])


def _num(x, name, where):
    """Parse a number or string at the current working precision."""
    try:
        if isinstance(x, str):
            s = x.strip()
            if "j" in s:
                v = _parse_complex(s)
            elif "/" in s:
                a, b = s.split("/")
                v = mp.mpf(a.strip()) / mp.mpf(b.strip())
            else:
                v = mp.mpf(s)
        elif isinstance(x, complex):
            v = mp.mpc(x)
        elif isinstance(x, (mp.mpc,)):
            v = x
        else:
            v = mp.mpf(x)
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"{where}: {name} must be a number or a numeric string, got {x!r}") from None
    if not mp.isfinite(v):
        raise ValueError(f"{where}: {name} must be finite, got {x!r}")
    return v


def _parse_complex(s):
    """Parse 'a+bj', 'bj', '1.3j' at full working precision."""
    t = s.replace(" ", "").replace("(", "").replace(")", "")
    if not t.endswith("j"):
        raise ValueError(s)
    body = t[:-1]
    # find the split between real and imaginary parts (last +/- not following 'e')
    idx = -1
    for i in range(len(body) - 1, 0, -1):
        if body[i] in "+-" and body[i - 1] not in "eE":
            idx = i
            break
    if idx == -1:
        re, im = "0", body
    else:
        re, im = body[:idx], body[idx:]
    if im in ("", "+"):
        im = "1"
    elif im == "-":
        im = "-1"
    return mp.mpc(mp.mpf(re), mp.mpf(im))


def _pos(x, name, where):
    v = _num(x, name, where)
    if isinstance(v, mp.mpc) or not v > 0:
        raise ValueError(f"{where}: {name} must be finite and positive, got {x!r}")
    return v


def _re(v):
    return mp.re(v)


# -----------------------------------------------------------------------------
# kernel identity
# -----------------------------------------------------------------------------

def kernel_identity_lhs(nu, y, t, tp, *, dps=30):
    """K_nu(y t_>) I_nu(y t_<) from the library Bessel functions."""
    where = "kernel_identity_lhs"
    with mp.workdps(dps):
        nu = _num(nu, "nu", where)
        y = _pos(y, "y", where)
        t = _pos(t, "t", where)
        tp = _pos(tp, "tp", where)
        tg, tl = (t, tp) if t >= tp else (tp, t)
        return +(mp.besselk(nu, y * tg) * mp.besseli(nu, y * tl))


def kernel_identity_rhs(nu, y, t, tp, b0=0.7, *, dps=30):
    """Right-hand side of the kernel identity in its one-dimensional p-form on Re b = b0."""
    where = "kernel_identity_rhs"
    with mp.workdps(dps):
        nu = _num(nu, "nu", where)
        y = _pos(y, "y", where)
        t = _pos(t, "t", where)
        tp = _pos(tp, "tp", where)
        b0 = _num(b0, "b0", where)
        if isinstance(b0, mp.mpc) or not (mp.mpf(0.5) < b0 < 1):
            raise ValueError(f"{where}: b0 must lie in (1/2, 1), got {b0}")
        if not _re(nu) > b0 - 1:
            raise ValueError(f"{where}: Re nu must exceed b0 - 1, got nu = {nu}")
        tg, tl = (t, tp) if t >= tp else (tp, t)
        g1nu = mp.gamma(1 + nu)
        if t == tp:
            def f(p):
                b = b0 + 1j * p
                return (mp.gamma(1 - b) * mp.gamma(1 + nu - b) * mp.gamma(2 * b - 1)
                        / (mp.gamma(b) * mp.gamma(b + nu)) * (y * t / 2) ** (2 * b - 2))
            pref = 1 / (4 * mp.pi)
        else:
            x = (tl / tg) ** 2

            def f(p):
                b = b0 + 1j * p
                return (mp.gamma(1 - b) * mp.gamma(1 + nu - b) / g1nu
                        * mp.hyp2f1(1 + nu - b, 1 - b, 1 + nu, x) * (y * tg / 2) ** (2 * b - 2))
            pref = (tl / tg) ** nu / (4 * mp.pi)
        val = mp.quad(f, [-40, -10, 0, 10, 40], maxdegree=10)
        return +(pref * val)


def kernel_identity_check(nu, y, t, tp, b0=0.7, *, dps=30):
    """Both sides of the kernel identity and |lhs - rhs| / |lhs|."""
    with mp.workdps(dps):
        lhs = kernel_identity_lhs(nu, y, t, tp, dps=dps)
        rhs = kernel_identity_rhs(nu, y, t, tp, b0, dps=dps)
        rel = abs(lhs - rhs) / abs(lhs)
        return IdentityCheck(lhs, rhs, rel)


# -----------------------------------------------------------------------------
# double Mellin transform of the ordered kernel
# -----------------------------------------------------------------------------

def _strip(where, nu, z, zp):
    if not (_re(z + nu) > 0 and _re(zp + nu) > 0 and 0 < _re(z + zp) < 2):
        raise ValueError(f"{where}: (z, z') outside the strip Re(z+nu) > 0, Re(z'+nu) > 0, "
                         f"0 < Re(z+z') < 2; got z = {z}, z' = {zp}, nu = {nu}")


def mellin_transform_rhs(nu, y, z, zp, *, dps=30):
    """Closed form of M(z, z') = iint dt dt' t^(z-1) t'^(z'-1) K_nu(y t_>) I_nu(y t_<)."""
    where = "mellin_transform_rhs"
    with mp.workdps(dps):
        nu = _num(nu, "nu", where)
        y = _pos(y, "y", where)
        z = _num(z, "z", where)
        zp = _num(zp, "zp", where)
        _strip(where, nu, z, zp)
        val = (mp.gamma((z + nu) / 2) * mp.gamma((zp + nu) / 2) * mp.gamma((z + zp) / 2)
               * mp.gamma(1 - (z + zp) / 2)
               / (mp.gamma(1 + (nu - z) / 2) * mp.gamma(1 + (nu - zp) / 2))
               * (y / 2) ** (-z - zp) / 8)
        return +val


def mellin_transform_lhs(nu, y, z, zp, *, direct2d=False, dps=30):
    """M(z, z') from the term-by-term integrated series (3F2 form), or with
    ``direct2d=True`` by nested quadrature in ln t, ln t' over [-60, 8]."""
    where = "mellin_transform_lhs"
    with mp.workdps(dps):
        nu = _num(nu, "nu", where)
        y = _pos(y, "y", where)
        z = _num(z, "z", where)
        zp = _num(zp, "zp", where)
        _strip(where, nu, z, zp)
        if direct2d:
            def inner(L2):
                t2 = mp.exp(L2)

                def g(L1):
                    t1 = mp.exp(L1)
                    tg, tl = (t1, t2) if t1 >= t2 else (t2, t1)
                    return (mp.exp(z * L1) * mp.besselk(nu, y * tg) * mp.besseli(nu, y * tl))
                return mp.exp(zp * L2) * mp.quad(g, [-60, L2, 8], maxdegree=6)
            return +mp.quad(inner, [-60, 0, 8], maxdegree=6)
        h = (z + zp) / 2
        tot = 0
        for a in (z, zp):
            c = (a + nu) / 2
            try:
                F = mp.hyp3f2(h + nu, h, c, nu + 1, c + 1, 1)
            except Exception as exc:  # mpmath raises NoConvergence (ValueError subclass)
                raise ArithmeticError(f"{where}: series summation did not converge ({exc})") from None
            tot += mp.gamma(h + nu) * mp.gamma(h) / mp.gamma(nu + 1) / (a + nu) * F
        return +(mp.mpf(2) ** (z + zp - 2) * y ** (-(z + zp)) * tot)


# -----------------------------------------------------------------------------
# Barnes representations
# -----------------------------------------------------------------------------

def K_barnes(nu, z, sigp=None, *, dps=30):
    """K_nu(z) = 1/2 int ds'/(2 pi i) Gamma(-s') Gamma(-s'-nu) (z/2)^(nu+2s') on Re s' = sigp,
    integrated over |Im s'| <= 40, compared with the library K_nu."""
    where = "K_barnes"
    with mp.workdps(dps):
        nu = _num(nu, "nu", where)
        z = _pos(z, "z", where)
        bound = min(mp.mpf(0), -_re(nu))
        if sigp is None:
            sigp = bound - mp.mpf(1) / 4
        else:
            sigp = _num(sigp, "sigp", where)
            if isinstance(sigp, mp.mpc) or not sigp < bound:
                raise ValueError(f"{where}: sigp must be < min(0, -Re nu) = {bound}, got {sigp}")

        def f(tau):
            s = sigp + 1j * tau
            return mp.gamma(-s) * mp.gamma(-s - nu) * (z / 2) ** (nu + 2 * s)
        num = mp.quad(f, [-40, 0, 40], maxdegree=10) / (4 * mp.pi)
        cf = mp.besselk(nu, z)
        return NumericVsClosedForm(+num, +cf, abs(num - cf) / abs(cf))


def barnes_2F1_lemma(A, B, C, D, x, *, dps=30):
    """1/(2 pi i) int_{Re u = 0} du Gamma(A-u) Gamma(B+u) / [Gamma(C+u) Gamma(D-u)] x^u
    against x^A Gamma(A+B) / [Gamma(A+C) Gamma(D-A)] 2F1(A+B, 1+A-D; A+C; x)."""
    where = "barnes_2F1_lemma"
    with mp.workdps(dps):
        A = _num(A, "A", where)
        B = _num(B, "B", where)
        C = _num(C, "C", where)
        D = _num(D, "D", where)
        x = _num(x, "x", where)
        if isinstance(x, mp.mpc) or not (0 < x < 1):
            raise ValueError(f"{where}: x must lie in (0, 1), got {x}")
        if not (_re(A) > 0 and _re(B) > 0):
            raise ValueError(f"{where}: Re A and Re B must be > 0")
        if not _re(C + D - A - B) > 1:
            raise ValueError(f"{where}: Re(C + D - A - B) must be > 1")

        def f(tau):
            u = 1j * tau
            return (mp.gamma(A - u) * mp.gamma(B + u) / (mp.gamma(C + u) * mp.gamma(D - u))
                    * x ** u)
        num = mp.quad(f, [-3000, -100, 0, 100, 3000], maxdegree=12) / (2 * mp.pi)
        cf = (x ** A * mp.gamma(A + B) / (mp.gamma(A + C) * mp.gamma(D - A))
              * mp.hyp2f1(A + B, 1 + A - D, A + C, x))
        return NumericVsClosedForm(+num, +cf, abs(num - cf) / abs(cf))
