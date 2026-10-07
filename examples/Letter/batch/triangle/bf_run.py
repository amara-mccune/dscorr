"""Brute-force runs of the triangle: main radial ladder, far patch, P1/P2 records.
Used by b_worker.py, far_patch.py and by the
pi-dot pipeline (run_bf.py)."""
from __future__ import annotations

import math
import multiprocessing as mpc
import time

import numpy as np

from bf_triangle import LADDER, SETTINGS, TimeGrid, Triangle, f_of_r, radial_nodes

_W = {}


def _init(mus, tgrid, rv, k, st):
    _W["tri"] = Triangle(mus, TimeGrid(**tgrid), rv=rv)
    _W["k"] = k
    _W["st"] = st


def _node(args):
    i, r, zone = args
    st = _W["st"]
    orders = {"lo": st["ang_lo"], "hi": st["ang_hi"], "far": st["ang_far"]}[zone]
    f, n = f_of_r(_W["tri"], _W["k"], r, orders)
    return i, f, n


def default_tgrid(k, tgrid=None):
    g = {"tmin": 1e-12, "tmax": 60.0 / min(1.0, min(k)), "npan": 156, "ngl": 12}
    if tgrid:
        g.update(tgrid)
    return g


def resolve_setting(setting):
    if setting is None:
        setting = "P1"
    if isinstance(setting, str):
        return dict(SETTINGS[setting])
    return dict(setting)


def run_main(task, rv=1.0, store_from=300.0):
    """Main run up to Lambda_1 (round-robin over nproc processes, or serial)."""
    t0 = time.time()
    mus = tuple(task["mus"])
    k = tuple(float(v) for v in task["k"])
    st = resolve_setting(task.get("setting"))
    tg = default_tgrid(k, task.get("tgrid"))
    rs, ws, zones, edges = radial_nodes(k, st)
    jobs = [(i, float(r), z) for i, (r, z) in enumerate(zip(rs, zones))]
    nproc = task.get("nproc")
    f = np.zeros(len(rs), dtype=complex)
    nev = 0
    if nproc:
        ctx = mpc.get_context("fork")
        with ctx.Pool(int(nproc), initializer=_init, initargs=(mus, tg, rv, k, st)) as pool:
            rr = [j for q in range(int(nproc)) for j in jobs[q::int(nproc)]]   # round-robin
            for i, fi, n in pool.imap_unordered(_node, rr):
                f[i] = fi
                nev += n
    else:
        _init(mus, tg, rv, k, st)
        for job in jobs:
            i, fi, n = _node(job)
            f[i] = fi
            nev += n
    rec = {"mus": list(mus), "k": list(k), "setting": st, "n_radial": int(len(rs)),
           "n_evals": int(nev)}
    for L in LADDER:
        if L <= st["Lam1"]:
            B = complex(np.sum(ws[rs <= L] * f[rs <= L]))
            rec[f"B_{L}"] = [B.real, B.imag]
    sel = rs >= store_from
    rec["r_far"] = rs[sel].tolist()
    rec["w_far"] = ws[sel].tolist()
    rec["f_far_re"] = f[sel].real.tolist()
    rec["f_far_im"] = f[sel].imag.tolist()
    rec["tmin"], rec["tmax"], rec["npan"], rec["ngl"] = tg["tmin"], tg["tmax"], tg["npan"], tg["ngl"]
    rec["wall_s"] = time.time() - t0
    rec["task"] = task
    return rec


FAR_EDGES = (2560, 3620, 5120, 7240, 10240, 14480, 20480, 30000, 45000, 67000, 100000, 150000)
FAR_B = (5120, 10240, 20480, 30000, 67000, 150000)


def far_patch(rec, nproc=None):
    """Far patch of the radial ladder (in place); no-op if B_150000 is present."""
    if "B_150000" in rec:
        return rec
    k = tuple(rec["k"])
    mus = tuple(rec["mus"])
    tri = Triangle(mus, TimeGrid(1e-12, 60.0, 156))
    orders = (4, 6, 16)
    r = np.array(rec["r_far"])
    f = np.array(rec["f_far_re"]) + 1j * np.array(rec["f_far_im"])
    idx = [i for i in range(len(r)) if 300.0 <= r[i] <= 2560.0][::3]
    omega = 2.0 * float(np.mean(mus))
    errs = []
    for i in idx:
        fc, _ = f_of_r(tri, k, float(r[i]), orders)
        errs.append(fc - f[i])
        f[i] = fc
    errs = np.array(errs)
    ph = np.exp(-1j * omega * np.log(r[idx]))
    c = complex(np.mean(errs * ph))
    scatter = float(np.max(np.abs(errs * ph - c))) if len(errs) else 0.0
    dB = c * 2560.0 * np.exp(1j * omega * math.log(2560.0)) / (1.0 + 1j * omega)
    B2560 = complex(*rec["B_2560"]) + dB
    x, w = np.polynomial.legendre.leggauss(8)
    rn, wn = [], []
    for a, b in zip(FAR_EDGES[:-1], FAR_EDGES[1:]):
        rn.append(0.5 * (a + b) + 0.5 * (b - a) * x)
        wn.append(0.5 * (b - a) * w)
    rn, wn = np.concatenate(rn), np.concatenate(wn)
    fn = np.array([f_of_r(tri, k, float(rr), orders)[0] for rr in rn])
    for L in FAR_B:
        B = B2560 + np.sum(wn[rn <= L] * fn[rn <= L])
        rec[f"B_{L}"] = [B.real, B.imag]
    rec["tmin_err"] = {"c": [c.real, c.imag], "omega": omega, "scatter": scatter,
                       "dB_2560": [dB.real, dB.imag], "n_nodes": len(idx)}
    rec["B_2560_corr"] = [B2560.real, B2560.imag]
    r_all = np.concatenate([r, rn])
    f_all = np.concatenate([f, fn])
    rec["r_far"] = r_all.tolist()
    rec["f_far_re"] = f_all.real.tolist()
    rec["f_far_im"] = f_all.imag.tolist()
    rec["far_patched"] = True
    return rec


def tail_pidot(rec, Lcut=2560.0, rfit=300.0):
    """Three-term tail of the pi-dot brute force."""
    r = np.array(rec["r_far"])
    f = np.array(rec["f_far_re"]) + 1j * np.array(rec["f_far_im"])
    sel = (r >= rfit) & (r <= Lcut)
    rr, ff = r[sel], f[sel]
    X = np.column_stack([rr ** -3, rr ** -4, rr ** -5]).astype(complex)
    c, *_ = np.linalg.lstsq(X, ff, rcond=None)
    fit = X @ c
    resid = float(np.max(np.abs(fit - ff) / np.abs(ff)))
    tail = c[0] / (2 * Lcut ** 2) + c[1] / (3 * Lcut ** 3) + c[2] / (4 * Lcut ** 4)
    slope = float(np.polyfit(np.log(rr), np.log(np.abs(ff)), 1)[0])
    Binf = complex(*rec["B_2560"]) + tail
    rec["B_inf"] = [Binf.real, Binf.imag]
    rec["tail_info"] = {"c3": [c[0].real, c[0].imag], "fit_rel_resid": resid,
                        "tail": [tail.real, tail.imag], "exponent_free": slope}
    return rec
