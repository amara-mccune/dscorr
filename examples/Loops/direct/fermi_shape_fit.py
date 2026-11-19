"""Fit of the direct integration of the fermion triangle against the series in the symmetric variables.

Each direct value I(r, c) (k1 = 1, one orientation) contains  Re[ K^(-6) sum_N D_N(delta) R^(2 nu + N) ]  with
K = (1 + k2)/2, R = r/K, delta = (1 - k2)/r, plus a background  sum_n (a_n + b_n ln r) r^n,  n >= -2.

For every data set the script reports, for 12, 14 and 16 background parameters,
  (1) the largest residual after the full series is subtracted (no free parameter), relative to the envelope of the series,
  (2) a free complex factor on the whole series,
  (3) a free complex factor on the corrections N >= 1 only,
  (4) the residual with the leading term only (the power of the test).
The background and the free factors are fitted by linear least squares with the weight r^2 (as in fermi_direct_fit.py).

Usage: fermi_shape_fit.py series.json data1.jsonl [data2.jsonl ...] [--rmax X] [--rmin X] [--nmax N] [--json out.json]
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ext import shapevars as SV  # noqa: E402


def load_series(serfile, nmax=None):
    mp.mp.dps = 30
    d = json.load(open(serfile))
    coef = {tuple(map(int, k.split(","))): mp.mpc(mp.mpf(v[0]), mp.mpf(v[1])) for k, v in d["coef"].items()}
    N = max(k[0] for k in coef)
    nu = mp.mpc(mp.mpf(d["nu"][0]), mp.mpf(d["nu"][1]))
    D = SV.symmetric(coef, nu, N, dim=6)
    if nmax is not None:
        N = min(N, nmax)
    return dict(coef=coef, nu=nu, lam=float(mp.im(nu)), D=D, N=N)


def shape(r, c):
    k2 = np.sqrt(1 + 2 * r * c + r * r)
    K = (1 + k2) / 2
    return K, r / K, (1 - k2) / r


def terms(S, r, c):
    """complex T_N(r) = K^(-6) D_N(delta) R^(2 nu + N), N = 0..Nmax, for arrays r, c"""
    K, R, dl = shape(r, c)
    out = np.zeros((S["N"] + 1, r.size), complex)
    for i in range(r.size):
        DN = SV.at_delta(S["D"], float(dl[i]), S["N"])
        for n in range(S["N"] + 1):
            out[n, i] = complex(DN[n]) * R[i] ** n
        out[:, i] *= K[i] ** (-6) * np.exp(2j * S["lam"] * np.log(R[i]))
    return out


def terms_C(S, r, c):
    """the same in the fixed-angle form sum_n C_n(c) r^(2 nu + n) (converges for r < 1 only)"""
    out = np.zeros((S["N"] + 1, r.size), complex)
    for (n, mc), v in S["coef"].items():
        if n <= S["N"]:
            out[n] += complex(v) * c ** mc * r ** n
    return out * np.exp(2j * S["lam"] * np.log(r))[None, :]


def lsq(r, y, extra, nbg, pmin=-2):
    """weighted least squares: background columns r^n, r^n ln r (n = pmin .. pmin + nbg) plus the columns in extra"""
    L = np.log(r)
    cols = [r ** n for n in range(pmin, pmin + nbg + 1)] + [r ** n * L for n in range(pmin, pmin + nbg + 1)] + list(extra)
    A = np.stack(cols, 1) * (r ** (-pmin))[:, None]
    scl = np.abs(A).max(0)
    cf, *_ = np.linalg.lstsq(A / scl, y * r ** (-pmin), rcond=None)
    fit = ((A / scl) @ cf) * r ** pmin
    res = y - fit
    cov = None
    if extra:
        dof = max(r.size - A.shape[1], 1)
        s2 = np.sum((res * r ** (-pmin)) ** 2) / dof
        cov = s2 * np.linalg.pinv((A / scl).T @ (A / scl))[-len(extra):, -len(extra):] / np.outer(scl[-len(extra):], scl[-len(extra):])
    return res, (cf / scl)[-len(extra):] if extra else None, cov


def analyse(S, r, c, I, nbg):
    T = terms(S, r, c)
    full, lead, corr = T.sum(0), T[0], T[1:].sum(0)
    env = np.abs(full)
    out = dict(nbg=nbg, npar=2 * nbg + 2, npts=int(r.size))
    res, _, _ = lsq(r, I, [], nbg)
    out["res_none"] = float(np.max(np.abs(res) / env))
    res, _, _ = lsq(r, I - lead.real, [], nbg)
    out["res_lead"] = float(np.max(np.abs(res) / env))
    out["res_lead_abs"] = np.abs(res).tolist()
    res, _, _ = lsq(r, I - full.real, [], nbg)
    out["res_full"] = float(np.max(np.abs(res) / env))
    out["res_full_r"] = (np.abs(res) / env).tolist()
    out["res_by_order"] = []
    for n in range(S["N"] + 1):
        rr, _, _ = lsq(r, I - T[:n + 1].sum(0).real, [], nbg)
        out["res_by_order"].append(float(np.max(np.abs(rr) / env)))
    res, cf, cov = lsq(r, I, [full.real, -full.imag], nbg)
    out["s_all"] = [float(cf[0]), float(cf[1])]
    out["s_all_dev"] = float(abs(complex(cf[0], cf[1]) - 1))
    out["s_all_err"] = float(np.sqrt(np.trace(cov)))
    res, cf, cov = lsq(r, I - lead.real, [corr.real, -corr.imag], nbg)
    out["s_corr"] = [float(cf[0]), float(cf[1])]
    out["s_corr_dev"] = float(abs(complex(cf[0], cf[1]) - 1))
    out["s_corr_err"] = float(np.sqrt(np.trace(cov)))
    out["corr_over_lead_max"] = float(np.max(np.abs(corr) / np.abs(lead)))
    # one free complex factor on a single order (all the others fixed)
    out["s_single"] = {}
    for n in range(min(S["N"], 4) + 1):
        res, cf, cov = lsq(r, I - (full - T[n]).real, [T[n].real, -T[n].imag], nbg)
        out["s_single"][n] = [float(abs(complex(cf[0], cf[1]) - 1)), float(np.sqrt(np.trace(cov)))]
    # one free complex factor on all orders N >= M (lower orders fixed)
    out["s_ge"] = {}
    for M in (2, 3, 4):
        if S["N"] < M:
            continue
        hi = T[M:].sum(0)
        res, cf, cov = lsq(r, I - T[:M].sum(0).real, [hi.real, -hi.imag], nbg)
        out["s_ge"][M] = [float(abs(complex(cf[0], cf[1]) - 1)), float(np.sqrt(np.trace(cov))), float(np.max(np.abs(hi) / np.abs(lead)))]
    # free complex factors on the orders 1..M together (orders above M and the leading term fixed)
    out["s_orders"] = {}
    for M in (1, 2, 3):
        if S["N"] < M or r.size < 2 * nbg + 2 + 2 * M + 2:
            continue
        ext = []
        for n in range(1, M + 1):
            ext += [T[n].real, -T[n].imag]
        res, cf, cov = lsq(r, I - lead.real - T[M + 1:].sum(0).real, ext, nbg)
        out["s_orders"][M] = [float(abs(complex(cf[2 * i], cf[2 * i + 1]) - 1)) for i in range(M)]
    out["power"] = out["res_lead"] / out["res_full"]
    return out


def load_data(files, rmin=0.0, rmax=1e9):
    rows = [json.loads(l) for f in files for l in open(f)]
    rows = list({x["r"]: x for x in rows if rmin <= x["r"] <= rmax}.values())      # one row per r
    rows.sort(key=lambda x: x["r"])
    return np.array([x["r"] for x in rows]), np.array([x["c"] for x in rows]), np.array([x["I"] for x in rows])


def report(S, files, rmin=0.0, rmax=1e9, nbgs=(5, 6, 7), quiet=False):
    r, c, I = load_data(files, rmin, rmax)
    K, R, dl = shape(r, c)
    out = dict(files=[os.path.basename(f) for f in files], npts=int(r.size), r=[float(r[0]), float(r[-1])], R=[float(R.min()), float(R.max())],
               delta=[float(dl.min()), float(dl.max())], N=S["N"], fits={})
    for nbg in nbgs:
        if r.size < 2 * nbg + 2 + 3:
            continue
        a = analyse(S, r, c, I, nbg)
        out["fits"][nbg] = a
        if not quiet:
            print(f"  {a['npar']:2d} bkg par: residual/envelope  none {a['res_none']:.2e}  lead {a['res_lead']:.2e}  full {a['res_full']:.2e}  (power {a['power']:.0f})"
                  f" | s_all - 1 = {a['s_all'][0] - 1:+.2e} {a['s_all'][1]:+.2e}i (|.| {a['s_all_dev']:.1e}, formal {a['s_all_err']:.1e})"
                  f" | s_corr - 1 = {a['s_corr'][0] - 1:+.2e} {a['s_corr'][1]:+.2e}i (|.| {a['s_corr_dev']:.1e}, formal {a['s_corr_err']:.1e})")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = {}
    for key in ("--rmax", "--rmin", "--nmax", "--json"):
        if key in args:
            i = args.index(key)
            opt[key] = args[i + 1]
            del args[i:i + 2]
    S = load_series(args[0], int(opt["--nmax"]) if "--nmax" in opt else None)
    files = args[1:]
    r, c, I = load_data(files, float(opt.get("--rmin", 0)), float(opt.get("--rmax", 1e9)))
    K, R, dl = shape(r, c)
    T = terms(S, r, c)
    print(f"{len(r)} points, r in [{r[0]:.4g}, {r[-1]:.4g}], R up to {R.max():.4g}, delta in [{dl.min():.3f}, {dl.max():.3f}], series through N = {S['N']}")
    print(f"  largest |corrections/leading| = {np.max(np.abs(T[1:].sum(0)) / np.abs(T[0])):.3f}")
    out = report(S, files, float(opt.get("--rmin", 0)), float(opt.get("--rmax", 1e9)))
    for nbg, a in out["fits"].items():
        print(f"  {a['npar']} bkg par: residual/envelope by order kept:", " ".join(f"{x:.1e}" for x in a["res_by_order"]))
    for nbg, a in out["fits"].items():
        print(f"  {a['npar']} bkg par: one free coefficient at a time, |D_N(fit)/D_N - 1|: " + ", ".join(f"N={n}: {v[0]:.1e}" for n, v in a["s_single"].items()))
        print(f"  {a['npar']} bkg par: one free factor on all orders N >= M, |s - 1| (formal error; largest size of these orders over the leading term): "
              + "; ".join(f"M={M}: {v[0]:.1e} ({v[1]:.0e}; {v[2]:.3f})" for M, v in a["s_ge"].items()))
        print(f"  {a['npar']} bkg par: |s_N - 1| with free factors on orders 1..M: " + "; ".join(f"M={M}: " + ", ".join(f"{x:.1e}" for x in v) for M, v in a["s_orders"].items()))
    if "--json" in opt:
        json.dump(out, open(opt["--json"], "w"), indent=1)
