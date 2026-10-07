"""Input validation shared by all public routes of dscorr.

Public names: ``ValidityWarning``, ``TESTED``, ``check_mu``, ``check_momenta``,
``check_family``.  ``where`` is the name of the calling public function or
class; it is printed as the prefix ``"<where>: "`` of every message.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = ["ValidityWarning", "TESTED", "check_mu", "check_momenta", "check_family"]


class ValidityWarning(UserWarning):
    """Issued when an input lies outside the range tested by the package."""


#: Tested range of the package.
TESTED = {"mu_min": 0.3, "mu_max": 4.0, "momentum_ratio_min": 0.002}

_TRIANGLE_TOL = 1e-12
_FAMILY_KMAX = 2.0


def _prefix(where):
    return f"{where}: " if where else ""


def check_mu(*mus, where=""):
    """Check mass parameters; return them as a tuple of floats.

    Raises ``ValueError`` for non-numeric, non-finite or non-positive values;
    warns with ``ValidityWarning`` (once per offending value) outside
    ``[TESTED['mu_min'], TESTED['mu_max']]``.
    """
    p = _prefix(where)
    out = []
    for mu in mus:
        try:
            m = float(mu)
        except (TypeError, ValueError):
            raise ValueError(f"{p}mass parameter mu must be a real number, got {mu!r}") from None
        if not math.isfinite(m):
            raise ValueError(f"{p}mass parameter mu must be finite, got {m}")
        if m <= 0.0:
            raise ValueError(
                f"{p}mass parameter mu must be > 0 (principal series, m^2/H^2 = mu^2 + 9/4; "
                f"the correlators depend on mu^2 only, pass abs(mu)), got mu = {m}")
        out.append(m)
    for m in out:
        if m < TESTED["mu_min"] or m > TESTED["mu_max"]:
            warnings.warn(
                f"{p}mu = {m} lies outside the tested range [{TESTED['mu_min']}, "
                f"{TESTED['mu_max']}] (see dscorr.TESTED / README 'Tested range')",
                ValidityWarning, stacklevel=3)
    return tuple(out)


def check_momenta(k1, k2, k3, where=""):
    """Check a momentum triple; return it as a tuple of floats.

    Raises ``ValueError`` for non-numeric, non-finite or non-positive momenta
    and for a violation of the triangle inequality by more than
    ``1e-12 * max(k)``; warns with ``ValidityWarning`` if
    ``min(k)/max(k) < TESTED['momentum_ratio_min']``.
    """
    p = _prefix(where)
    raw = (k1, k2, k3)
    ks = []
    for name, k in zip(("k1", "k2", "k3"), raw):
        try:
            v = float(k)
        except (TypeError, ValueError):
            raise ValueError(f"{p}momentum {name} must be a real number, got {k!r}") from None
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(
                f"{p}momenta must be finite and > 0, got (k1, k2, k3) = ({k1}, {k2}, {k3})")
        ks.append(v)
    a, b, c = ks
    kmax = max(ks)
    tol = _TRIANGLE_TOL * kmax
    for big, s1, s2, txt in ((a, b, c, "k1 > k2 + k3"), (b, a, c, "k2 > k1 + k3"),
                             (c, a, b, "k3 > k1 + k2")):
        if big > s1 + s2 + tol:
            raise ValueError(
                f"{p}momenta violate the triangle inequality ({txt}): (k1, k2, k3) = "
                f"({a}, {b}, {c}); each momentum must not exceed the sum of the other two "
                f"(folded/degenerate triangles are allowed)")
    ratio = min(ks) / kmax
    rmin = TESTED["momentum_ratio_min"]
    if ratio < rmin:
        warnings.warn(
            f"{p}momentum ratio min(k)/max(k) = {ratio:.4e} < tested minimum {rmin:.4e}; "
            f"extend the ln t grid (L1=16 or larger) and check convergence",
            ValidityWarning, stacklevel=3)
    return (a, b, c)


def check_family(kappas, where=""):
    """Validate a squeezing family (1, 1, kappa); return a 1-D float array."""
    p = _prefix(where)
    kap = np.atleast_1d(np.asarray(kappas, dtype=float)).ravel()
    if kap.size == 0:
        return kap
    bad = ~(np.isfinite(kap) & (kap > 0))
    if bad.any():
        v = kap[bad][0]
        raise ValueError(f"{p}momenta must be finite and > 0, got kappa = {v!r}")
    over = kap > _FAMILY_KMAX + _TRIANGLE_TOL
    if over.any():
        v = kap[over][0]
        raise ValueError(
            f"{p}momenta violate the triangle inequality (k3 > k1 + k2): kappa = {v!r} "
            f"on the family (1, 1, kappa)")
    check_momenta(1.0, 1.0, float(kap.min()), where=where)
    return kap


def broadcast_momenta(k1, k2, k3):
    """Broadcast scalar or array momenta; return (shape, list of float triples).

    The shape is ``()`` when all three are scalars.
    """
    a, b, c = np.broadcast_arrays(np.asarray(k1, dtype=object), np.asarray(k2, dtype=object),
                                  np.asarray(k3, dtype=object))
    shape = a.shape
    triples = list(zip(a.ravel().tolist(), b.ravel().tolist(), c.ravel().tolist()))
    return shape, triples
