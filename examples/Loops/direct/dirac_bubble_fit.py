"""Fit of the direct integration of the Dirac-pair bubble (direct_dirac_bubble.py) against its squeezed series.

Each direct value I(r) (k12 = 1, k3 = r, vertices r1 = 4 and r2 = 2) contains the pair signal

    Re sum_n Mcal_n r^(1 + 2 nu + n),
    Mcal_n = [1 - (-1)^n e^(-2 pi mubar)] M_D(n) + [1 - (-1)^n e^(2 pi mubar)] conj[M_D^-(n)],
    M_D(n) = Gamma(4 + 2 nu + n) S(n; +),     M_D^-(n) = Gamma(4 - 2 nu + n) S(n; -),

plus a background  sum_j (a_j + b_j ln r) r^j,  j >= 0. Here S(n; +-) are the soft moments of dirac_pair.py on the two
pair branches (dirac_bubble_series.py), and the brackets are the sum over the four Schwinger-Keldysh sectors: the hard
vertex is later than the soft one, so the time-ordered sector has the Wightman function of the cut one, and its hard
time integral differs by the phase -e^(-i pi (4 + 2 nu + n)). The doubly cut term alone (D) and the time-ordered term
alone (T) contain

    Re D:  sum_n [M_D(n) + conj M_D^-(n)] r^(1 + 2 nu + n),
    Re T:  -(-1)^n sum_n [e^(-2 pi mubar) M_D(n) + e^(2 pi mubar) conj M_D^-(n)] r^(1 + 2 nu + n).

The option --model plus fits instead the series of the + branch alone with the full leading coefficient,
Mcal_0 M_D(n)/M_D(0), which is the series quoted in Sec. 8.3 of v5.19.

For every data set the script reports, for 12, 14 and 16 background parameters,
  (1) the largest residual after the full series is subtracted (no free parameter), relative to the envelope of the series,
  (2) the same with the leading term only, and the ratio of the two (the power of the test),
  (3) a free complex factor on the whole series and on the corrections n >= 1 only,
  (4) one free complex factor on a single order, all the others fixed.
The background and the free factors are fitted by linear least squares.

The option --wpow p weights the points by r^p (default 0; -1 gives every e-fold of r the same weight in the signal).

Usage: dirac_bubble_fit.py series.json data1.jsonl [data2.jsonl ...] [--rmax X] [--rmin X] [--nmax N] [--part I|T|D]
                           [--model full|plus] [--wpow p] [--json out.json]
       dirac_bubble_fit.py --check        (the stored runs of data_dirac/ against ../dirac_bubble_direct_record.json)
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_dirac")
RECORD = os.path.join(HERE, "..", "dirac_bubble_direct_record.json")
# stored runs (Nov 20, 2026) and the fit of record for each: (series, run, smallest r, weight power)
RUNS = {"(1,1)": ("series_1_1.json", "m1_k1_pc1.5.jsonl", 0.0, -1.0),
        "(2,2)": ("series_2_2.json", "m2_k2_pc1.5.jsonl", 0.01, 0.0),
        "(1,0.5)": ("series_1_0.5.json", "m1_k0.5_pc1.5.jsonl", 1e-5, -1.0)}


def load_series(serfile, nmax=None, part="I", model="full"):
    mp.mp.dps = 30
    d = json.load(open(serfile))
    mub = mp.mpf(d["mubar"])
    nu = mp.mpc(0, mub)
    Sp = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["S_plus"]]
    Sm = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["S_minus"]]
    N = min(len(Sp), len(Sm)) - 1
    if nmax is not None:
        N = min(N, nmax)
    r1 = d.get("r1", 4)
    E = mp.exp(2 * mp.pi * mub)
    Mp = [mp.gamma(r1 + 2 * nu + n) * Sp[n] for n in range(N + 1)]
    Mm = [mp.conj(mp.gamma(r1 - 2 * nu + n) * Sm[n]) for n in range(N + 1)]
    full = [(1 - (-1) ** n / E) * Mp[n] + (1 - (-1) ** n * E) * Mm[n] for n in range(N + 1)]
    if part == "D":
        M = [Mp[n] + Mm[n] for n in range(N + 1)]
    elif part == "T":
        M = [-(-1) ** n * (Mp[n] / E + E * Mm[n]) for n in range(N + 1)]
    else:
        M = full
    if model == "plus":                                   # the series of Sec. 8.3: + branch of the doubly cut term
        M = [M[0] * Mp[n] / Mp[0] for n in range(N + 1)]
    return dict(M=[complex(x) for x in M], Mplus=[complex(x) for x in Mp], Mminus=[complex(x) for x in Mm],
                full=[complex(x) for x in full], lam=float(mub), N=N, m=d["m"], kappa=d["kappa"])


def terms(S, r):
    """complex T_n(r) = M_n r^(1 + 2 nu + n), n = 0..N"""
    ph = r * np.exp(2j * S["lam"] * np.log(r))
    return np.array([S["M"][n] * r ** n * ph for n in range(S["N"] + 1)])


def lsq(r, y, extra, nbg, pmin=0, wpow=0.0):
    """least squares: background columns r^n and r^n ln r (n = pmin .. pmin + nbg) plus the columns in extra; weight r^wpow"""
    L = np.log(r)
    cols = [r ** n for n in range(pmin, pmin + nbg + 1)] + [r ** n * L for n in range(pmin, pmin + nbg + 1)] + list(extra)
    wt = r ** wpow
    A = np.stack(cols, 1) * wt[:, None]
    scl = np.abs(A).max(0)
    cf, *_ = np.linalg.lstsq(A / scl, y * wt, rcond=None)
    res = y - ((A / scl) @ cf) / wt
    cov = None
    if extra:
        dof = max(r.size - A.shape[1], 1)
        s2 = np.sum((res * wt) ** 2) / dof
        ne = len(extra)
        cov = s2 * np.linalg.pinv((A / scl).T @ (A / scl))[-ne:, -ne:] / np.outer(scl[-ne:], scl[-ne:])
    return res, (cf / scl)[-len(extra):] if extra else None, cov


def cfac(cf):
    return complex(cf[0], cf[1])


def analyse(S, r, I, nbg, pmin=0, wpow=0.0):
    T = terms(S, r)
    full, lead, corr = T.sum(0), T[0], T[1:].sum(0)
    env = np.abs(full)
    kw = dict(pmin=pmin, wpow=wpow)
    out = dict(nbg=nbg, npar=2 * nbg + 2, npts=int(r.size))
    res, _, _ = lsq(r, I, [], nbg, **kw)
    out["res_none"] = float(np.max(np.abs(res) / env))
    res, _, _ = lsq(r, I - lead.real, [], nbg, **kw)
    out["res_lead"] = float(np.max(np.abs(res) / env))
    res, _, _ = lsq(r, I - full.real, [], nbg, **kw)
    out["res_full"] = float(np.max(np.abs(res) / env))
    out["res_full_r"] = (np.abs(res) / env).tolist()
    out["res_full_abs"] = float(np.max(np.abs(res)))
    out["power"] = out["res_lead"] / out["res_full"]
    out["res_by_order"] = []
    for n in range(S["N"] + 1):
        rr, _, _ = lsq(r, I - T[:n + 1].sum(0).real, [], nbg, **kw)
        out["res_by_order"].append(float(np.max(np.abs(rr) / env)))
    res, cf, cov = lsq(r, I, [full.real, -full.imag], nbg, **kw)
    out["s_all"] = [float(cf[0]), float(cf[1])]
    out["s_all_dev"] = float(abs(cfac(cf) - 1))
    out["s_all_err"] = float(np.sqrt(np.trace(cov)))
    res, cf, cov = lsq(r, I - lead.real, [corr.real, -corr.imag], nbg, **kw)
    out["s_corr"] = [float(cf[0]), float(cf[1])]
    out["s_corr_dev"] = float(abs(cfac(cf) - 1))
    out["s_corr_err"] = float(np.sqrt(np.trace(cov)))
    out["corr_over_lead_max"] = float(np.max(np.abs(corr) / np.abs(lead)))
    out["s_single"] = {}                       # one free complex factor on a single order (all the others fixed)
    for n in range(min(S["N"], 4) + 1):
        res, cf, cov = lsq(r, I - (full - T[n]).real, [T[n].real, -T[n].imag], nbg, **kw)
        out["s_single"][n] = [float(abs(cfac(cf) - 1)), float(np.sqrt(np.trace(cov))), float(cf[0]), float(cf[1])]
    out["s_ge"] = {}                           # one free complex factor on all orders n >= M (lower orders fixed)
    for M in (2, 3, 4):
        if S["N"] < M:
            continue
        hi = T[M:].sum(0)
        res, cf, cov = lsq(r, I - T[:M].sum(0).real, [hi.real, -hi.imag], nbg, **kw)
        out["s_ge"][M] = [float(abs(cfac(cf) - 1)), float(np.sqrt(np.trace(cov))), float(np.max(np.abs(hi) / np.abs(lead)))]
    return out


def load_data(files, rmin=0.0, rmax=1e9, part="I"):
    rows = [json.loads(l) for f in files for l in open(f)]
    rows = list({x["r"]: x for x in rows if rmin <= x["r"] <= rmax}.values())      # one row per r
    rows.sort(key=lambda x: x["r"])
    val = {"I": lambda x: x["I"], "T": lambda x: x["T"][0], "D": lambda x: x["D"][0]}[part]
    return np.array([x["r"] for x in rows]), np.array([val(x) for x in rows])


def report(S, files, rmin=0.0, rmax=1e9, nbgs=(5, 6, 7), part="I", quiet=False, pmin=0, wpow=0.0):
    r, I = load_data(files, rmin, rmax, part)
    T = terms(S, r)
    out = dict(files=list(files), part=part, npts=int(r.size), r=[float(r[0]), float(r[-1])], N=S["N"],
               corr_over_lead_max=float(np.max(np.abs(T[1:].sum(0)) / np.abs(T[0]))),
               modulus_ratio_max=float(np.max(np.abs(T.sum(0)) / np.abs(T[0]))), fits={})
    for nbg in nbgs:
        if r.size < 2 * nbg + 2 + 3:
            continue
        a = analyse(S, r, I, nbg, pmin, wpow)
        out["fits"][nbg] = a
        if not quiet:
            print(f"  {a['npar']:2d} bkg par: residual/envelope  none {a['res_none']:.2e}  lead {a['res_lead']:.2e}  full {a['res_full']:.2e}  (power {a['power']:.0f})"
                  f" | s_all - 1 = {a['s_all'][0] - 1:+.2e} {a['s_all'][1]:+.2e}i (|.| {a['s_all_dev']:.1e})"
                  f" | s_corr - 1 = {a['s_corr'][0] - 1:+.2e} {a['s_corr'][1]:+.2e}i (|.| {a['s_corr_dev']:.1e}, formal {a['s_corr_err']:.1e})")
    return out


def cut_difference(run_a, run_b, degree=12):
    """largest residual of a polynomial fit (no logarithms) to the difference of two runs with different loop cuts"""
    a = {json.loads(l)["r"]: json.loads(l)["I"] for l in open(run_a)}
    b = {json.loads(l)["r"]: json.loads(l)["I"] for l in open(run_b)}
    r = np.array(sorted(set(a) & set(b)))
    d = np.array([a[x] - b[x] for x in r])
    A = np.stack([r ** n for n in range(degree + 1)], 1)
    scl = np.abs(A).max(0)
    cf, *_ = np.linalg.lstsq(A / scl, d, rcond=None)
    return float(np.max(np.abs(d - (A / scl) @ cf))), int(r.size)


def record_values():
    """{key: computed value} for every entry of dirac_bubble_direct_record.json"""
    out = {}
    for pt, (ser, run, rmin, wpow) in RUNS.items():
        f = [os.path.join(DATA, run)]
        S = load_series(os.path.join(DATA, ser))
        for n in (1, 2, 3):
            out[f"dirac_bubble_direct/Mcal_{n}/Mcal_0 {pt}"] = S["M"][n] / S["M"][0]
        out[f"dirac_bubble_direct/|(- branch)/(+ branch)| n=0 {pt}"] = abs(S["full"][0] - (1 - np.exp(-2 * np.pi * S["lam"])) * S["Mplus"][0]) / abs((1 - np.exp(-2 * np.pi * S["lam"])) * S["Mplus"][0])
        rep = report(S, f, rmin=rmin, quiet=True, wpow=wpow)
        for nbg, a in rep["fits"].items():
            k = f"dirac_bubble_direct/{pt} {a['npar']} par: "
            out[k + "residual, full series over leading term only"] = a["res_full"] / a["res_lead"]
            out[k + "|s_corr - 1|"] = a["s_corr_dev"]
            out[k + "|Mcal_1(fit)/Mcal_1 - 1|"] = a["s_single"][1][0]
            out[k + "|Mcal_2(fit)/Mcal_2 - 1|"] = a["s_single"][2][0]
        P = load_series(os.path.join(DATA, ser), model="plus")
        rep = report(P, f, rmin=rmin, quiet=True, wpow=wpow, nbgs=(7,))
        out[f"dirac_bubble_direct/{pt} 16 par, + branch alone: |s_corr - 1|"] = rep["fits"][7]["s_corr_dev"]
        for part, name in (("D", "doubly cut term"), ("T", "time-ordered term")):
            Sp = load_series(os.path.join(DATA, ser), part=part)
            rep = report(Sp, f, rmin=rmin, quiet=True, wpow=wpow, nbgs=(7,), part=part)
            out[f"dirac_bubble_direct/{pt} 16 par, {name}: |s_corr - 1|"] = rep["fits"][7]["s_corr_dev"]
    res, npts = cut_difference(os.path.join(DATA, "m1_k1_pc1.5.jsonl"), os.path.join(DATA, "m1_k1_pc1.0.jsonl"))
    out["dirac_bubble_direct/(1,1) cuts 1.5 and 1.0: residual of a polynomial fit to the difference"] = res
    return out


def check():
    sys.path.insert(0, os.path.join(HERE, ".."))
    from ext.record import compare, report as rep
    table = json.load(open(RECORD))
    vals = record_values()
    res = []
    for key in table:
        v = vals[key]
        res.append(compare(key, mp.mpc(v) if isinstance(v, complex) else mp.mpf(float(v)), table))
    return rep(res)


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    args = sys.argv[1:]
    opt = {}
    for key in ("--rmax", "--rmin", "--nmax", "--json", "--part", "--model", "--pmin", "--wpow"):
        if key in args:
            i = args.index(key)
            opt[key] = args[i + 1]
            del args[i:i + 2]
    part, model = opt.get("--part", "I"), opt.get("--model", "full")
    S = load_series(args[0], int(opt["--nmax"]) if "--nmax" in opt else None, part, model)
    files = args[1:]
    rmin, rmax = float(opt.get("--rmin", 0)), float(opt.get("--rmax", 1e9))
    r, I = load_data(files, rmin, rmax, part)
    print(f"{len(r)} points, r in [{r[0]:.4g}, {r[-1]:.4g}], (m, kappa) = ({S['m']}, {S['kappa']}), series through n = {S['N']}, part {part}, model {model}")
    out = report(S, files, rmin, rmax, part=part, pmin=int(opt.get("--pmin", 0)), wpow=float(opt.get("--wpow", 0.0)))
    print(f"  largest |corrections/leading| = {out['corr_over_lead_max']:.3f}, largest |series/leading| = {out['modulus_ratio_max']:.3f}")
    for nbg, a in out["fits"].items():
        print(f"  {a['npar']} bkg par: residual/envelope by order kept:", " ".join(f"{x:.1e}" for x in a["res_by_order"]))
    for nbg, a in out["fits"].items():
        print(f"  {a['npar']} bkg par: one free coefficient at a time, |M_n(fit)/M_n - 1|: " + ", ".join(f"n={n}: {v[0]:.1e}" for n, v in a["s_single"].items()))
        print(f"  {a['npar']} bkg par: one free factor on all orders n >= M, |s - 1| (formal error; largest size of these orders over the leading term): "
              + "; ".join(f"M={M}: {v[0]:.1e} ({v[1]:.0e}; {v[2]:.3f})" for M, v in a["s_ge"].items()))
    if "--json" in opt:
        json.dump(out, open(opt["--json"], "w"), indent=1)
