"""Uniform grid in L = ln t, composite and cumulative Simpson rules."""
from __future__ import annotations

import numpy as np

POW_CLIP = 690.0
EXP_CLIP = 745.0


class LogGrid:
    """Uniform grid L_j = L0 + j hL, j = 0..n, with n even."""

    def __init__(self, L0, L1, hL):
        L0 = float(L0)
        L1 = float(L1)
        hL = float(hL)
        n = int(round((L1 - L0) / hL))
        if n % 2:
            n += 1
        self.L0, self.L1, self.hL, self.n = L0, L1, hL, n
        self.L = L0 + np.arange(n + 1) * hL
        self.t = np.exp(self.L)
        w = np.full(n + 1, 2.0)
        w[1::2] = 4.0
        w[0] = 1.0
        w[-1] = 1.0
        self.simpson_w = w * (hL / 3.0)

    @property
    def nodes(self):
        return self.n + 1

    def pow(self, a):
        """exp(a L) with the exponent clipped to [-690, 690] (a may be complex)."""
        a = complex(a) if np.iscomplexobj(a) else float(a)
        if isinstance(a, complex):
            re = np.clip(a.real * self.L, -POW_CLIP, POW_CLIP)
            return np.exp(re + 1j * a.imag * self.L)
        return np.exp(np.clip(a * self.L, -POW_CLIP, POW_CLIP))

    def damp(self, x):
        """exp(-x t) with x t clipped to [0, 745]."""
        return np.exp(-np.clip(float(x) * self.t, 0.0, EXP_CLIP))

    def simpson(self, f):
        """Composite Simpson integral over L of samples f (last axis)."""
        return f @ self.simpson_w


def cumulative_simpson(y, h):
    """Cumulative Simpson integral of equally spaced samples, initial value 0.

    Interval [L_j, L_{j+1}] with even j: h (5 y_j + 8 y_{j+1} - y_{j+2}) / 12;
    with odd j: h (-y_{j-1} + 8 y_j + 5 y_{j+1}) / 12.  The last interval always
    uses the second form.  (This is SciPy's ``cumulative_simpson`` for equal
    spacing.)
    """
    y = np.asarray(y)
    n = y.shape[-1] - 1
    inc = np.empty(y.shape[:-1] + (n,), dtype=np.result_type(y, float))
    # even j (j <= n-2): first form
    je = np.arange(0, n - 1, 2)
    inc[..., je] = h * (5.0 * y[..., je] + 8.0 * y[..., je + 1] - y[..., je + 2]) / 12.0
    # odd j: second form
    jo = np.arange(1, n, 2)
    inc[..., jo] = h * (-y[..., jo - 1] + 8.0 * y[..., jo] + 5.0 * y[..., jo + 1]) / 12.0
    if n % 2 == 1:
        # last interval (even index) uses the second form
        j = n - 1
        inc[..., j] = h * (-y[..., j - 1] + 8.0 * y[..., j] + 5.0 * y[..., j + 1]) / 12.0
    out = np.zeros(y.shape, dtype=inc.dtype)
    out[..., 1:] = np.cumsum(inc, axis=-1)
    return out
