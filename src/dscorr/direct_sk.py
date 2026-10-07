"""Direct Schwinger-Keldysh time integration of the tree diagrams (Letter SM Sec. S5).

The independent check of the Mellin-Barnes route (``dscorr.tree``): no special-function
or contour code is shared.  Scalars only; seconds to minutes per value.

Public names: ``single_exchange_direct``, ``double_exchange_direct``,
``triple_exchange_direct``.

Note: the double and triple functions
leave the mpmath working precision at ``dps`` digits after the call.
"""
from __future__ import annotations

import mpmath as mp

from ._engines import direct as _d
from ._validate import check_momenta, check_mu

__all__ = ["single_exchange_direct", "double_exchange_direct", "triple_exchange_direct"]


def _normalise(where, k1, k2, k3):
    k = check_momenta(k1, k2, k3, where=where)
    kmax = max(k)
    q = tuple(x / kmax for x in k)
    return k, q, kmax


def _mp_dec(x):
    """Multi-precision number from the shortest decimal string of a double (or a string)."""
    if isinstance(x, str):
        return mp.mpf(x)
    return mp.mpf(repr(float(x)))


def _raw(k1, k2, k3):
    ks = (k1, k2, k3)
    q = tuple(float(k) for k in ks)
    qmp = tuple(_mp_dec(k) for k in ks)
    return q, qmp


def _output(where, S, F_unit, q, kmax, out, full, extra):
    Sf = float(S)
    Fu = float(F_unit)
    F = Fu / kmax ** 3
    B = F / (8.0 * q[0] * q[1] * q[2] * kmax ** 3)
    if full:
        d = {"S": Sf, "F": F, "B": B}
        d.update(extra)
        return d
    if out == "S":
        return Sf
    if out == "F":
        return F
    if out == "B":
        return B
    raise ValueError(f"{where}: out must be 'S', 'F' or 'B' (use full=True for all values)")


def double_exchange_direct(k1, k2, k3, mu1, mu2, *, out="S", full=False, mpanels=22, nn=24,
                           dps=25, raw=False):
    """Double exchange by direct time integration (8-sector sum factorised per line)."""
    where = "double_exchange_direct"
    if not raw:
        check_momenta(k1, k2, k3, where=where)
    m1, m2 = check_mu(mu1, mu2, where=where)
    if raw:
        q, qmp = _raw(k1, k2, k3)
        kmax = 1.0
    else:
        _, q, kmax = _normalise(where, k1, k2, k3)
    mp.mp.dps = int(dps)                      # global side effect (documented)
    if not raw:
        qmp = tuple(_mp_dec(x) for x in q)
    mus = (_mp_dec(m1), _mp_dec(m2))
    st = _d.DirectSettings(mpanels=mpanels, nn=nn)
    S, F_unit, psi = _d.double_direct_mp(qmp, mus, st)
    extra = {"psiF_time": complex(psi), "k_unit": tuple(q), "scale": kmax,
             "mp_values": {"S": S, "F_unit": F_unit, "psiF": psi}}
    return _output(where, S, F_unit, q, kmax, out, full, extra)


def triple_exchange_direct(k1, k2, k3, muA, muB, muC, *, out="S", full=False, mpanels=22,
                           nn=24, dps=25, raw=False, engine_family=False):
    """Triple exchange by direct time integration (16-sector sum)."""
    where = "triple_exchange_direct"
    ms = check_mu(muA, muB, muC, where=where)
    if raw:
        q, qmp = _raw(k1, k2, k3)
        kmax = 1.0
    else:
        _, q, kmax = _normalise(where, k1, k2, k3)
    if engine_family and not (abs(q[0] - 1.0) <= 1e-15 and abs(q[1] - 1.0) <= 1e-15):
        raise ValueError(f"{where}: engine_family=True needs (k1,k2,k3) = (1,1,kappa)")
    mp.mp.dps = int(dps)
    if not raw:
        qmp = tuple(_mp_dec(x) for x in q)
    if engine_family:
        # the archived isosceles-family integrator: the same algorithm on (1, 1, kappa)
        qmp = (mp.mpf(1), mp.mpf(1), qmp[2])
    mus = tuple(_mp_dec(m) for m in ms)
    st = _d.DirectSettings(mpanels=mpanels, nn=nn)
    S, F_unit, psi = _d.triple_direct_mp(qmp, mus, st)
    extra = {"psiF_time": complex(psi), "k_unit": tuple(q), "scale": kmax,
             "mp_values": {"S": S, "F_unit": F_unit, "psiF": psi}}
    return _output(where, S, F_unit, q, kmax, out, full, extra)


def single_exchange_direct(k1, k2, k3, mu, *, sum_perms=False, out="S", full=False, h=0.125,
                           n=20, nproc=None, raw=False):
    """Single exchange by direct time integration with Hankel mode functions (4-sector sum)."""
    where = "single_exchange_direct"
    (m,) = check_mu(mu, where=where)
    if nproc is None:
        npool = _d.default_pool_size()
    else:
        npool = int(nproc)
        if npool < 1:
            raise ValueError(f"{where}: nproc must be >= 1, got {nproc}")
    pool = _d.make_pool(npool)
    try:
        if raw:
            q = tuple(float(k) for k in (k1, k2, k3))
            kmax = 1.0
        else:
            _, q, kmax = _normalise(where, k1, k2, k3)
        legs = (1, 2, 3) if sum_perms else (3,)
        chans = [_d.single_channel_direct(q, leg, m, float(h), int(n), pool) for leg in legs]
    finally:
        pool.terminate()
        pool.join()
    f = 0j
    for c in chans:
        f += c["F_sectors"]
    pq = q[0] * q[1] * q[2]
    S = pq / 4.0 * f.real
    F_unit = 8.0 * S / pq
    extra = {"k_unit": tuple(q), "scale": kmax, "imag_leak": abs(f.imag), "channels": chans}
    return _output(where, S, F_unit, q, kmax, out, full, extra)
