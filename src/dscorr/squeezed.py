"""Closed-form squeezed coefficients, templates and the fit (Letter Eqs. (10), (11); SM Sec. S4).

    S(1, 1, kappa) = S_an(kappa) + 2 Re[M_L kappa^(1/2 + i mu_soft)] (1 + O(kappa^2)),

with the exchanged line soft (last mass argument).  Public names: ``M1``, ``M2``, ``M3``,
``template``, ``signal``, ``extract``, ``SqueezedFit``.
"""
from __future__ import annotations

from collections import namedtuple

import mpmath as mp
import numpy as np

from ._engines import squeezed_f64 as _f
from ._engines import squeezed_mp as _m
from ._validate import check_mu

__all__ = ["M1", "M2", "M3", "template", "signal", "extract", "SqueezedFit"]

SqueezedFit = namedtuple("SqueezedFit", ["M", "max_residual", "coefficients", "S_an"])

MIN_POINTS = 8


def _dec(x):
    """Shortest decimal string of a float (multi-precision numbers pass unchanged)."""
    if isinstance(x, (mp.mpf,)):
        return x
    return repr(float(x))


def M1(mu, *, dps=None):
    """Single-exchange coefficient M_1 of Letter Eq. (10) (squeezed limit of Eq. (9))."""
    if isinstance(mu, mp.mpf):
        check_mu(float(mu), where="M1")
        m = mu
    else:
        (m,) = check_mu(mu, where="M1")
    if dps is None:
        return _f.M1(float(m))
    with mp.workdps(int(dps)):
        mu_ = m if isinstance(m, mp.mpf) else mp.mpf(_dec(m))
        nu = mp.mpc(0, mu_)
        ph = mp.pi * mp.sqrt(mp.pi / 2) / mp.cosh(mp.pi * mu_)
        c = 1 - 1j / mp.sinh(mp.pi * mu_)
        val = (-(ph / 4) * c * mp.mpf(2) ** (-nu) * mp.gamma(mp.mpf(5) / 2 + nu)
               * mp.mpf(2) ** (-mp.mpf(5) / 2 - nu) / mp.gamma(1 + nu))
        return +val


def M2(mu1, mu2, *, dps=None, parts=False):
    """Double-exchange coefficient (Supplemental Material Sec. S4); mu1 hard line, mu2 soft line."""
    m1, m2 = check_mu(mu1, mu2, where="M2")
    if dps is None:
        d = _f.M2_parts(m1, m2)
        return d if parts else complex(d["A"])
    d = _m.M2_mp(_dec(m1), _dec(m2))
    if parts:
        return d
    with mp.workdps(int(dps)):
        return +d["A"]


def M3(muA, muB, muC, *, dps=None, parts=False):
    """Triple-exchange coefficient (Letter Eq. (11)); muC is the soft line."""
    a, b, c = check_mu(muA, muB, muC, where="M3")
    if dps is None:
        d = _f.M3_parts(a, b, c)
        return d if parts else complex(d["A"])
    d = _m.M3_mp(_dec(a), _dec(b), _dec(c))
    if parts:
        return d
    with mp.workdps(int(dps)):
        return +d["A"]


def _kappa(kappa, where):
    k = np.asarray(kappa, dtype=float)
    if not np.all(np.isfinite(k) & (k > 0)):
        raise ValueError(f"{where}: kappa must be finite and > 0")
    return k


def template(kappa, M, mu_soft):
    """2 Re[M kappa^(1/2 + i mu_soft)]."""
    (mu,) = check_mu(mu_soft, where="template")
    k = _kappa(kappa, "template")
    return 2.0 * np.real(complex(M) * np.exp((0.5 + 1j * mu) * np.log(k)))


def signal(kappa, M, mu_soft):
    """2 Re[M kappa^(i mu_soft)] = template / sqrt(kappa)."""
    (mu,) = check_mu(mu_soft, where="signal")
    k = _kappa(kappa, "signal")
    return 2.0 * np.real(complex(M) * np.exp(1j * mu * np.log(k)))


def _osc_columns(k, exps):
    cols = []
    lk = np.log(k)
    for e in exps:
        z = np.exp(e * lk)
        cols.append(2.0 * z.real)
        cols.append(-2.0 * z.imag)
    return cols


def design_matrix(kappas, mu, extra_terms=True):
    """Columns of the public fit model."""
    k = np.asarray(kappas, dtype=float)
    exps = [0.5 + 1j * mu, 2.5 + 1j * mu]
    if extra_terms:
        exps += [1.0 + 2j * mu, 2.0 + 2j * mu]
    cols = [k, k ** 2, k ** 3] + _osc_columns(k, exps)
    return np.column_stack(cols)


def extract(kappas, S, mu_soft, *, extra_terms=True, wpow=-0.5):
    """Fit the squeezed coefficient with pinned exponents.

    Model: c1 k + c2 k^2 + c3 k^3 + 2Re[M k^(1/2+i mu)] + 2Re[M' k^(5/2+i mu)]
    (+ 2Re[D k^(1+2i mu)] + 2Re[D' k^(2+2i mu)] with ``extra_terms``), weighted
    least squares with weights k^wpow.  Returns ``SqueezedFit(M, max_residual,
    coefficients, S_an)``.
    """
    k = np.asarray(kappas)
    s = np.asarray(S)
    if k.ndim != 1 or s.ndim != 1 or k.shape != s.shape or k.size < MIN_POINTS:
        raise ValueError(f"extract: kappas and S must be 1-D arrays of equal length >= "
                         f"{MIN_POINTS}, got shapes {k.shape} and {s.shape}")
    k = k.astype(float)
    s = s.astype(float)
    if not (np.all(np.isfinite(k) & (k > 0)) and np.all(np.isfinite(s))):
        raise ValueError("extract: kappas must be finite and > 0 and S finite")
    (mu,) = check_mu(mu_soft, where="extract")
    X = design_matrix(k, mu, extra_terms)
    w = k ** float(wpow)
    c, *_ = np.linalg.lstsq(X * w[:, None], s * w, rcond=None)
    resid = float(np.max(np.abs(s - X @ c)))
    c1, c2, c3 = float(c[0]), float(c[1]), float(c[2])

    def S_an(kappa):
        kk = np.asarray(kappa, dtype=float)
        return c1 * kk + c2 * kk ** 2 + c3 * kk ** 3

    return SqueezedFit(complex(c[3], c[4]), resid, c, S_an)


# -----------------------------------------------------------------------------
# fit models 2 and 3 (model 3 builds the background S_an of Letter Fig. 2(b))
# -----------------------------------------------------------------------------

def _fit_pinned(kappas, S, powers, exps, wpow=None):
    """Linear least squares with columns kappa^p (p in powers) and the pairs
    2 Re kappa^e, -2 Im kappa^e (e in exps); weights kappa^wpow (None: unweighted)."""
    k = np.asarray(kappas, dtype=float)
    s = np.asarray(S, dtype=float)
    X = np.column_stack([k ** p for p in powers] + _osc_columns(k, exps))
    if wpow is None:
        c, *_ = np.linalg.lstsq(X, s, rcond=None)
    else:
        w = k ** float(wpow)
        c, *_ = np.linalg.lstsq(X * w[:, None], s * w, rcond=None)
    return c, X


def fit_single_exchange(kappas, S, mu, wpow=-0.5):
    """Model 2 (single exchange): returns (M, coefficients, max_residual)."""
    exps = [0.5 + 1j * mu, 1.5 + 1j * mu, 2.5 + 1j * mu, 1.5 + 2j * mu, 2.5 + 2j * mu]
    c, X = _fit_pinned(kappas, S, [1.0, 1.5, 2.0, 2.5, 3.0], exps, wpow)
    return complex(c[5], c[6]), c, float(np.max(np.abs(np.asarray(S) - X @ c)))


def fit_panel_c(kappas, S, mu):
    """Model 3 (Letter Fig. 2(b)): unweighted; returns (M_fit, coefficients, S_an callable)."""
    exps = [0.5 + 1j * mu, 2.5 + 1j * mu, 1.0 + 2j * mu, 2.0 + 2j * mu]
    c, _ = _fit_pinned(kappas, S, [1, 2, 3, 4, 5, 6], exps, None)
    cn = [float(x) for x in c[:6]]

    def S_an(kappa):
        kk = np.asarray(kappa, dtype=float)
        return sum(cn[n] * kk ** (n + 1) for n in range(6))

    return complex(c[6], c[7]), c, S_an
