"""Tree-level massive-exchange bispectra as finite Mellin-Barnes integrals
(kernel identity Eq. (2), master formula Eq. (4), cutting rule SM Sec. S3 of the Letter).

Public names: ``single_exchange``, ``double_exchange``, ``triple_exchange``,
``SingleExchange``, ``DoubleExchange``, ``TripleExchange``.

Normalisation: B = <pi pi pi>' = F / (8 k1 k2 k3), S = (k1 k2 k3)^2 B = k1 k2 k3 F / 8.
The exchanged line is always the last mass argument and ends on the mixing of k3.
"""
from __future__ import annotations

import math

import numpy as np

from ._engines.grid import LogGrid
from ._engines.lineblock import LineBlockCache
from ._engines.trees import (DoubleContour, TripleContour, double_assemble, double_cut_blocks,
                             single_channel, triple_assemble, triple_cut_blocks)
from ._validate import ValidityWarning, broadcast_momenta, check_family, check_momenta, check_mu

__all__ = ["single_exchange", "double_exchange", "triple_exchange",
           "SingleExchange", "DoubleExchange", "TripleExchange", "ValidityWarning"]

_OUTS = ("S", "F", "B", "blocks")
_DOUBLE_PERMS = ((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1), (0, 2, 1))
_SOFT_K3 = ("k3", "mixed", "C", "a")
_SOFT_K1 = ("k1", "unmixed", "B", "b")


def _check_kw(where, kw, allowed):
    bad = [k for k in kw if k not in allowed]
    if bad:
        raise TypeError(
            f"{where}: unexpected keyword argument(s) {', '.join(repr(b) for b in bad)}; "
            f"allowed numerical keywords: {', '.join(allowed)}")


def _check_out(where, out):
    if out not in _OUTS:
        raise ValueError(f"{where}: out must be one of 'S', 'F', 'B', 'blocks', got {out!r}")


def _masses(where, mus, n):
    if len(mus) == 1 and isinstance(mus[0], (tuple, list, np.ndarray)):
        mus = tuple(np.asarray(mus[0], dtype=object).ravel().tolist())
    if len(mus) != n:
        raise TypeError(f"{where}: expected {n} mass parameter(s), got {len(mus)}")
    return check_mu(*mus, where=where)


def _normalise(where, k1, k2, k3):
    k = check_momenta(k1, k2, k3, where=where)
    kmax = max(k)
    q = tuple(x / kmax for x in k)
    return k, q, kmax


def _finish(k, q, kmax, F_unit, out):
    S = q[0] * q[1] * q[2] * F_unit / 8.0
    F = F_unit / kmax ** 3
    if out == "S":
        return float(S)
    if out == "F":
        return float(F)
    return float(F / (8.0 * k[0] * k[1] * k[2]))


def _vectorise(fn, k1, k2, k3, out):
    shape, triples = broadcast_momenta(k1, k2, k3)
    if shape == ():
        return fn(*triples[0])
    vals = [fn(*t) for t in triples]
    if out == "blocks":
        return vals
    return np.array(vals, dtype=float).reshape(shape)


# =============================================================================
# single exchange
# =============================================================================

class SingleExchange:
    """Single exchange: pi-dot pi-dot sigma vertex, sigma(mu) exchanged to the mixing on k3.

    Numerical keywords: ``L0`` (-60.0), ``L1`` (10.6), ``hL`` (0.002).
    """

    _NUMKW = ("L0", "L1", "hL")
    _DEFAULTS = {"L0": -60.0, "L1": 10.6, "hL": 0.002}

    def __init__(self, *mus, _where=None, **kw):
        where = _where or "SingleExchange"
        _check_kw(where, kw, self._NUMKW)
        self._where = where
        (self.mu,) = _masses(where, mus, 1)
        self.masses = (self.mu,)
        p = dict(self._DEFAULTS)
        p.update({k: float(v) for k, v in kw.items()})
        self.settings = p
        self.grid = LogGrid(p["L0"], p["L1"], p["hL"])
        self._lines = LineBlockCache(self.grid)

    def _channel(self, q, leg):
        y = q[leg - 1]
        x = q[0] + q[1] + q[2] - y
        blk = self._lines.get(self.mu, y)
        ch = single_channel(self.grid, blk, x)
        return {"exchanged_leg": leg, "y": y, "x_hard": x, "bracket": ch["bracket"],
                "psi2F": ch["psi2F"], "PhiA": ch["PhiA"], "PhiB": ch["PhiB"],
                "S": q[0] * q[1] * q[2] * ch["bracket"] / 4.0}

    def _scalar(self, k1, k2, k3, sum_perms, out):
        k, q, kmax = _normalise(self._where, k1, k2, k3)
        legs = (1, 2, 3) if sum_perms else (3,)
        chans = [self._channel(q, leg) for leg in legs]
        tot = 0.0
        for c in chans:
            tot += c["bracket"]
        S = q[0] * q[1] * q[2] * tot / 4.0
        F_unit = 2.0 * tot
        if out == "blocks":
            return {"channels": chans, "k_input": k, "k_unit": q, "scale": kmax,
                    "S": float(S), "F": float(F_unit / kmax ** 3)}
        return _finish(k, q, kmax, F_unit, out)

    def __call__(self, k1, k2, k3, *, sum_perms=False, out="S"):
        _check_out(self._where, out)
        return _vectorise(lambda a, b, c: self._scalar(a, b, c, sum_perms, out), k1, k2, k3, out)

    def scan(self, kappas, *, sum_perms=False, out="S"):
        """S (or F) on the family (k1, k2, k3) = (1, 1, kappa)."""
        where = f"{self._where}.scan"
        kap = check_family(kappas, where=where)
        if kap.size == 0:
            return kap
        if not sum_perms:
            o = "S" if out == "S" else "F"
            return np.array([self(1.0, 1.0, float(x), out=o) for x in kap], dtype=float)
        hard = self._lines.get(self.mu, 1.0)
        res = np.empty(kap.size)
        for i, x in enumerate(kap):
            soft = self._lines.get(self.mu, float(x))
            br3 = single_channel(self.grid, soft, 2.0)["bracket"]
            br1 = single_channel(self.grid, hard, 1.0 + float(x))["bracket"]
            S = float(x) / 4.0 * (br3 + 2.0 * br1)
            res[i] = S if out == "S" else 8.0 * S / float(x)
        return res


# =============================================================================
# double exchange
# =============================================================================

class DoubleExchange:
    """Double exchange: pi-dot sigma1 sigma2 vertex with k1 direct; sigma1(mu1) to the
    mixing on k2; sigma2(mu2) exchanged to the mixing on k3.

    Numerical keywords: ``L0`` (-60.0), ``L1`` (10.6), ``hL`` (0.004), ``c`` (-0.25),
    ``T_sp`` (12.0), ``h`` (0.02), ``c_s`` (-0.125), ``T_s`` (20.0), ``h_s`` (0.02).
    """

    _NUMKW = ("L0", "L1", "hL", "c", "T_sp", "h", "c_s", "T_s", "h_s")
    _DEFAULTS = {"L0": -60.0, "L1": 10.6, "hL": 0.004, "c": -0.25, "T_sp": 12.0, "h": 0.02,
                 "c_s": -0.125, "T_s": 20.0, "h_s": 0.02}

    def __init__(self, *mus, _where=None, **kw):
        where = _where or "DoubleExchange"
        _check_kw(where, kw, self._NUMKW)
        self._where = where
        self.mu1, self.mu2 = _masses(where, mus, 2)
        self.masses = (self.mu1, self.mu2)
        p = dict(self._DEFAULTS)
        p.update({k: float(v) for k, v in kw.items()})
        self.settings = p
        self.grid = LogGrid(p["L0"], p["L1"], p["hL"])
        self._lines = LineBlockCache(self.grid)
        self._contour = DoubleContour(self.mu1, self.mu2, c=p["c"], T=p["T_sp"], h=p["h"],
                                      c_s=p["c_s"], T_s=p["T_s"], h_s=p["h_s"])

    # -- building blocks ------------------------------------------------------
    def psi_F(self, k1, k2, k3, *, method="twist"):
        """Contour value of the uncut Feynman block at the momenta as given."""
        m = "engine" if method == "engine" else "twist"
        return self._contour.psiF(float(k1), float(k2), float(k3), m)

    def _blocks_unit(self, q, uncut, method):
        b1 = self._lines.get(self.mu1, q[1])
        b2 = self._lines.get(self.mu2, q[2])
        cb = double_cut_blocks(self.grid, b1, b2, q[0])
        if uncut == "MB":
            psiF = self.psi_F(*q, method=method)
        else:
            psiF = cb["psiF_time"]
        uncut_v, cut1, cut2, F_unit = double_assemble(psiF, cb)
        d = {"psiF": psiF, "psiF_time": cb["psiF_time"], "uncut": uncut_v, "cut1": cut1,
             "cut2": cut2, "Phi_1": cb["Phi_1"], "Phi_2": cb["Phi_2"], "Psi_2": cb["Psi_2"],
             "Psi_1": cb["Psi_1"], "Xi": cb["Xi"], "F_unit": F_unit}
        return d

    def blocks(self, k1, k2, k3, *, uncut="MB", method="twist"):
        """Block dictionary at the normalised momenta (no key ``F``)."""
        k, q, kmax = _normalise(self._where, k1, k2, k3)
        d = {"k_input": k, "k_unit": q, "scale": kmax}
        d.update(self._blocks_unit(q, uncut, method))
        d["S"] = q[0] * q[1] * q[2] * d["F_unit"] / 8.0
        return d

    def _scalar(self, k1, k2, k3, sum_perms, out, uncut, method):
        k, q, kmax = _normalise(self._where, k1, k2, k3)
        if sum_perms:
            F_unit = 0.0
            for p in _DOUBLE_PERMS:
                qp = (q[p[0]], q[p[1]], q[p[2]])
                F_unit += self._blocks_unit(qp, uncut, method)["F_unit"]
            return _finish(k, q, kmax, F_unit, out)
        d = self._blocks_unit(q, uncut, method)
        if out == "blocks":
            r = {"k_input": k, "k_unit": q, "scale": kmax}
            r.update(d)
            r["S"] = q[0] * q[1] * q[2] * d["F_unit"] / 8.0
            r["F"] = d["F_unit"] / kmax ** 3
            return r
        return _finish(k, q, kmax, d["F_unit"], out)

    def __call__(self, k1, k2, k3, *, sum_perms=False, out="S", uncut="MB", method="twist"):
        _check_out(self._where, out)
        if out == "blocks" and sum_perms:
            raise ValueError(f"{self._where}: out='blocks' is not available with sum_perms=True")
        return _vectorise(lambda a, b, c: self._scalar(a, b, c, sum_perms, out, uncut, method),
                          k1, k2, k3, out)

    def scan(self, kappas, *, soft="k3", out="S"):
        """S (or F) on (1, 1, kappa) (soft='k3') or (kappa, 1, 1) (soft='k1')."""
        where = f"{self._where}.scan"
        if soft in _SOFT_K3:
            mode = "k3"
        elif soft in _SOFT_K1:
            mode = "k1"
        else:
            raise ValueError(f"{where}: soft must be 'k3' or 'k1' (legacy codes 'mixed', 'C', "
                             f"'a' / 'unmixed', 'B', 'b'), got {soft!r}")
        kap = check_family(kappas, where=where)
        if kap.size == 0:
            return kap
        g = self.grid
        res = np.empty(kap.size)
        if mode == "k3":
            psi = self._contour.scan_soft_k3(kap)
            b1 = self._lines.get(self.mu1, 1.0)
            for i, x in enumerate(kap):
                b2 = self._lines.get(self.mu2, float(x))
                cb = double_cut_blocks(g, b1, b2, 1.0)
                F = double_assemble(psi[i], cb)[-1]
                res[i] = float(x) * F / 8.0 if out == "S" else F
        else:
            psi = self._contour.scan_soft_k1(kap)
            b1 = self._lines.get(self.mu1, 1.0)
            b2 = self._lines.get(self.mu2, 1.0)
            for i, x in enumerate(kap):
                cb = double_cut_blocks(g, b1, b2, float(x))
                F = double_assemble(psi[i], cb)[-1]
                res[i] = float(x) * F / 8.0 if out == "S" else F
        return res


# =============================================================================
# triple exchange
# =============================================================================

class TripleExchange:
    """Triple exchange: sigma_A sigma_B sigma_C vertex without direct leg; sigma_A(muA) to
    the mixing on k1, sigma_B(muB) to k2, sigma_C(muC) exchanged to k3.

    Numerical keywords: ``L0`` (-60.0), ``L1`` (10.6), ``hL`` (0.004), ``c`` (-0.25),
    ``T`` (12.0), ``h`` (0.02), ``c_s`` (-0.125), ``T_s`` (20.0), ``h_s`` (0.02).
    """

    _NUMKW = ("L0", "L1", "hL", "c", "T", "h", "c_s", "T_s", "h_s")
    _DEFAULTS = {"L0": -60.0, "L1": 10.6, "hL": 0.004, "c": -0.25, "T": 12.0, "h": 0.02,
                 "c_s": -0.125, "T_s": 20.0, "h_s": 0.02}

    def __init__(self, *mus, _where=None, **kw):
        where = _where or "TripleExchange"
        _check_kw(where, kw, self._NUMKW)
        self._where = where
        self.muA, self.muB, self.muC = _masses(where, mus, 3)
        self.masses = (self.muA, self.muB, self.muC)
        p = dict(self._DEFAULTS)
        p.update({k: float(v) for k, v in kw.items()})
        self.settings = p
        self.grid = LogGrid(p["L0"], p["L1"], p["hL"])
        self._lines = LineBlockCache(self.grid)
        self._contour = TripleContour(self.muA, self.muB, self.muC, c=p["c"], T=p["T"],
                                      h=p["h"], c_s=p["c_s"], T_s=p["T_s"], h_s=p["h_s"])

    def psi_F(self, k1, k2, k3, *, method="twist"):
        """Contour value of the uncut Feynman block at the momenta as given."""
        m = "engine" if method == "engine" else "twist"
        return self._contour.psiF(float(k1), float(k2), float(k3), m)

    def _blocks_unit(self, q, uncut, method):
        bA = self._lines.get(self.muA, q[0])
        bB = self._lines.get(self.muB, q[1])
        bC = self._lines.get(self.muC, q[2])
        cb = triple_cut_blocks(self.grid, bA, bB, bC)
        psiF = self.psi_F(*q, method=method) if uncut == "MB" else cb["psiF_time"]
        uncut_v, cut1, cut2, cut3, F_unit = triple_assemble(psiF, cb)
        return {"psiF": psiF, "psiF_time": cb["psiF_time"], "uncut": uncut_v, "cut1": cut1,
                "cut2": cut2, "cut3": cut3, "Phi": list(cb["Phi"]), "Phic": cb["Phic"],
                "Psi": list(cb["Psi"]), "Xi": list(cb["Xi"]), "F_unit": F_unit}

    def blocks(self, k1, k2, k3, *, uncut="MB", method="twist"):
        """Block dictionary at the normalised momenta (no key ``F``)."""
        k, q, kmax = _normalise(self._where, k1, k2, k3)
        d = {"k_input": k, "k_unit": q, "scale": kmax}
        d.update(self._blocks_unit(q, uncut, method))
        d["S"] = q[0] * q[1] * q[2] * d["F_unit"] / 8.0
        return d

    def _scalar(self, k1, k2, k3, out, uncut, method):
        k, q, kmax = _normalise(self._where, k1, k2, k3)
        d = self._blocks_unit(q, uncut, method)
        if out == "blocks":
            r = {"k_input": k, "k_unit": q, "scale": kmax}
            r.update(d)
            r["S"] = q[0] * q[1] * q[2] * d["F_unit"] / 8.0
            r["F"] = d["F_unit"] / kmax ** 3
            return r
        return _finish(k, q, kmax, d["F_unit"], out)

    def __call__(self, k1, k2, k3, *, out="S", uncut="MB", method="twist"):
        _check_out(self._where, out)
        return _vectorise(lambda a, b, c: self._scalar(a, b, c, out, uncut, method),
                          k1, k2, k3, out)

    def scan(self, kappas, *, out="S"):
        """S (or F) on the family (1, 1, kappa)."""
        where = f"{self._where}.scan"
        kap = check_family(kappas, where=where)
        if kap.size == 0:
            return kap
        psi = self._contour.scan(kap)
        bA = self._lines.get(self.muA, 1.0)
        bB = self._lines.get(self.muB, 1.0)
        res = np.empty(kap.size)
        for i, x in enumerate(kap):
            bC = self._lines.get(self.muC, float(x))
            cb = triple_cut_blocks(self.grid, bA, bB, bC)
            F = triple_assemble(psi[i], cb)[-1]
            res[i] = float(x) * F / 8.0 if out == "S" else F
        return res


# =============================================================================
# function interface
# =============================================================================

_ROUTINES = {}
_ROUTINE_MAX = 16


def _routine(cls, where, mus, kw):
    _check_kw(where, kw, cls._NUMKW)
    fm = check_mu(*mus, where=where)
    key = (cls.__name__, fm, tuple(sorted((k, float(v)) for k, v in kw.items())))
    r = _ROUTINES.get(key)
    if r is None:
        if len(_ROUTINES) > _ROUTINE_MAX:
            _ROUTINES.clear()
        import warnings
        with warnings.catch_warnings():
            # masses were checked (and warned about) above
            warnings.simplefilter("ignore", ValidityWarning)
            r = cls(*fm, _where=where, **kw)
        _ROUTINES[key] = r
    return r


def single_exchange(k1, k2, k3, mu, *, sum_perms=False, out="S", **kw):
    """Single-exchange bispectrum; ``sum_perms=True`` sums the three channels (template S_1)."""
    r = _routine(SingleExchange, "single_exchange", (mu,), kw)
    return r(k1, k2, k3, sum_perms=sum_perms, out=out)


def double_exchange(k1, k2, k3, mu1, mu2, *, sum_perms=False, out="S", uncut="MB", **kw):
    """Double-exchange bispectrum; ``sum_perms=True`` sums the six leg assignments."""
    r = _routine(DoubleExchange, "double_exchange", (mu1, mu2), kw)
    return r(k1, k2, k3, sum_perms=sum_perms, out=out, uncut=uncut)


def triple_exchange(k1, k2, k3, muA, muB, muC, *, out="S", uncut="MB", **kw):
    """Triple-exchange bispectrum (one assignment of the species to the legs)."""
    r = _routine(TripleExchange, "triple_exchange", (muA, muB, muC), kw)
    return r(k1, k2, k3, out=out, uncut=uncut)
