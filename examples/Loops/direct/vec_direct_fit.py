"""Series of the helical triangle at general shape against direct values S(r, c) of direct_vec.py.

The series predicts  S(r, c)  contains  2 Re sum_n M_n(c) r^(2 + 2 i mu + n), with M_n(c) = sum_m C[n, m] c^m from
Triangle(general=True) (on the family k1 = k2, c = -r/2 and the coefficients are those of the isosceles series).
With --sym the series is taken in the symmetric variables instead,

    S  contains  2 Re sum_N D_N(delta) R^(2 + 2 i mu + N),    R = 2 k3/(k1 + k2),  delta = (k1 - k2)/k3,

evaluated at every point. At fixed c the series in r has growing coefficients and converges only for r < 1, so
--sym is the form to use at moderate r (the two truncations differ by higher orders).

The cut on the loop momentum and the rest of the diagram give a background of powers r^n and r^n ln r, which is
fitted by linear least squares, every point weighted by 1/r^2 so that all count alike against the leading
oscillation 2 |M_0| r^2. Printed:

  1. the largest residual, in units of 2 |M_0| r^2, after subtracting the series through order n;
  2. for the first four orders, the factor a_n = M_n(fit)/M_n(series) with that coefficient left free (two more
     linear parameters), and the same from a minimax fit (as tri_direct_fit.py);
  3. one common free factor on all corrections n >= 1.

The spread of these numbers between backgrounds of 12, 14, and 16 parameters (--nbg=5, 6, 7) measures how well a
coefficient is determined. A fit needs clearly more points than parameters.

Usage: vec_direct_fit.py series.json data.jsonl [more.jsonl ...] [--sym] [--nbg=5] [--pmin=0] [--rmin=..] [--rmax=..]
                         [--nuse=..] [--weight=2] [--minus=other.jsonl] [--norm=full_series.json] [--json=out.json]
                         [--label=..]
series.json: output of helical_triangle.py --general --json (keys mu, coef, isosceles, sym).
--minus subtracts a second run point by point (for example a run with CONTACT=0, to isolate the contact term, with
the contact part of the series as series.json and the full series as --norm for the leading amplitude)."""
import json
import sys

import numpy as np
from scipy.optimize import minimize

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict((a.lstrip("-").split("=", 1) + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
nbg, pmin = int(opt.get("nbg", 5)), int(opt.get("pmin", 0))
rmin, rmax, wpow = float(opt.get("rmin", 0)), float(opt.get("rmax", 1e9)), float(opt.get("weight", 2))
sym = "sym" in opt
cz = lambda p: complex(float(p[0]), float(p[1]))


def load(files):
    rows = [json.loads(l) for f in files for l in open(f)]
    return sorted((x for x in rows if rmin <= x["r"] <= rmax), key=lambda x: x["r"])


rows = load(args[1:])
r = np.array([x["r"] for x in rows])
S = np.array([x["S"] for x in rows])
if "minus" in opt:
    other = {x["r"]: x["S"] for x in load([opt["minus"]])}
    S = S - np.array([other[x] for x in r])
L = np.log(r)
shape = rows[0].get("shape", "iso")
cs = np.array([x.get("c", -x["r"] / 2) for x in rows])
x2 = np.sqrt(1 + 2 * r * cs + r * r)                                           # |k2| at k1 = 1
Rv, dl = 2 * r / (1 + x2), (1 - x2) / r                                        # R and delta at every point


def terms(d):
    """[Z_0, Z_1, ...]: the terms of the series at every point (complex; S contains 2 Re sum Z_n), and reference
    coefficients [M_0, M_1, ...] (at the first point)."""
    mu_ = float(d["mu"])
    if sym:
        Dc = {tuple(map(int, k.split(","))): cz(v) for k, v in d["sym"].items()}
        co = [sum(v * dl ** m for (n, m), v in Dc.items() if n == nn) + 0 * r for nn in range(max(k[0] for k in Dc) + 1)]
        return [c_ * Rv ** (2 + n) * np.exp(2j * mu_ * np.log(Rv)) for n, c_ in enumerate(co)], [complex(c_[0]) for c_ in co]
    if shape == "iso":
        M_ = [cz(p) for p in d["isosceles"]]
    else:
        C = {tuple(map(int, k.split(","))): cz(v) for k, v in d["coef"].items()}
        M_ = [sum(v * float(shape) ** m for (n, m), v in C.items() if n == nn) for nn in range(max(k[0] for k in C) + 1)]
    return [m * r ** (2 + n) * np.exp(2j * mu_ * L) for n, m in enumerate(M_)], M_


d = json.load(open(args[0]))
mu = float(d["mu"])
Z, M = terms(d)
if "nuse" in opt:
    Z, M = Z[:int(opt["nuse"]) + 1], M[:int(opt["nuse"]) + 1]
nmax = len(M) - 1
M0 = terms(json.load(open(opt["norm"])))[1][0] if "norm" in opt else M[0]       # leading coefficient for the normalization
lead = 2 * abs(M0) * r ** 2
w = r ** (-wpow)
BG = np.stack([r ** n for n in range(pmin, pmin + nbg + 1)] + [r ** n * L for n in range(pmin, pmin + nbg + 1)], 1)
orders = [n for n in range(nmax + 1) if abs(M[n]) > 1e-12 * max(abs(m) for m in M)]
name = "D" if sym else "M"


def series(n_hi, scale=None):
    return sum(2 * (Z[n] * (scale[n] if scale else 1)).real for n in range(n_hi + 1)) + 0 * r


def lsq(R, extra=None):
    """weighted least squares of R on the background (and extra columns); returns coefficients, residual"""
    A = BG if extra is None else np.concatenate([BG, extra], 1)
    Aw, Rw = A * w[:, None], R * w
    sc = np.abs(Aw).max(0)
    cf, *_ = np.linalg.lstsq(Aw / sc, Rw, rcond=None)
    return cf / sc, R - A @ (cf / sc)


def resid(R):
    return float(np.abs(lsq(R)[1] / lead).max())


def free_linear(ns):
    """one complex factor a on the orders ns, as two linear parameters: 2 Re[a sum Z_n]"""
    z = sum(Z[n] for n in ns)
    rest = S - sum(2 * Z[n].real for n in range(nmax + 1) if n not in ns)
    cf, res = lsq(rest, np.stack([2 * z.real, -2 * z.imag], 1))
    return complex(cf[-2], cf[-1]), float(np.abs(res / lead).max())


def free_minimax(n):
    def f(x):
        sc = [1] * (nmax + 1)
        sc[n] = 1 + x[0] + 1j * x[1]
        return resid(S - series(nmax, sc))
    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-20))
                for x0 in ([0, 0], [0.3, 0.3], [-0.5, 0.2])), key=lambda z: z.fun)
    return complex(1 + best.x[0], best.x[1]), float(best.fun)


out = dict(label=opt.get("label", ""), files=args[1:], minus=opt.get("minus"), form="R, delta" if sym else "r at fixed c", shape=shape,
           c=rows[0].get("c"), mu=mu, kappa=rows[0]["kappa"], Pc=rows[0]["Pc"], nphi=rows[0]["nphi"], npoints=len(r),
           rmin=float(r[0]), rmax=float(r[-1]), Rmax=float(Rv[-1]), nbg=nbg, pmin=pmin, nparam=2 * nbg + 2, nmax=nmax,
           ratios=[abs(z / M0) for z in M])
print(f"{opt.get('label', '')}  [series in {'R, delta' if sym else 'r'}]  shape {shape}, mu = {mu}, Pc = {rows[0]['Pc']}, NPHI = {rows[0]['nphi']}: "
      f"{len(r)} points, r in [{r[0]:.4g}, {r[-1]:.4g}]; background r^n, r^n ln r, n = {pmin}..{pmin + nbg} ({2 * nbg + 2} parameters)")
print(f"  |{name}_n/{name}_0| = " + ", ".join(f"{abs(z / M0):.3f}" for z in M))
corr = np.abs(series(nmax) - (2 * Z[0].real if orders[0] == 0 else 0)) / lead
print(f"  {'corrections (n >= 1)' if orders[0] == 0 else 'this part of the series'} over the leading amplitude at the largest r: {corr[-1]:.3g}")
out["correction_at_rmax"] = float(corr[-1])
res = [resid(S)] + [resid(S - series(n)) for n in range(nmax + 1)]
out["residual"] = res
print("  residual / leading amplitude:  none " + f"{res[0]:.1e}" + "".join(f",  n<={n} {v:.1e}" for n, v in enumerate(res[1:]) if n >= orders[0]))
out["free"] = {}
for n in orders[:4]:
    a, rs = free_linear([n])
    am, rm = free_minimax(n)
    out["free"][str(n)] = dict(a=[a.real, a.imag], dev=abs(a - 1), residual=rs, minimax_dev=abs(am - 1), minimax_residual=rm)
    print(f"  {name}_{n} free: a - 1 = {a.real - 1:+.2e} {a.imag:+.2e}i  (|a - 1| = {abs(a - 1):.1e}, residual {rs:.1e});"
          f"  minimax |a - 1| = {abs(am - 1):.1e}")
group = [n for n in orders if n >= 1]
if group:
    a, rs = free_linear(group)
    out["free_corrections"] = dict(orders=group, a=[a.real, a.imag], dev=abs(a - 1), residual=rs)
    print(f"  orders {group[0]}..{group[-1]} with one free factor: a - 1 = {a.real - 1:+.2e} {a.imag:+.2e}i  (|a - 1| = {abs(a - 1):.1e})")
if "json" in opt:
    json.dump(out, open(opt["json"], "w"), indent=1)
