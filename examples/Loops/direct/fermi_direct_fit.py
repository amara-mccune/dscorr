"""Fermion-triangle series against direct time-domain values I(r) at fixed angle c: fit an analytic background and
see what the series leaves. The series predicts  I  contains  Re sum_n C_n(c) r^(2 nu + n)  (C_n: both orientations).
Usage: series.json data.jsonl [--nbg=5] [--rmax=..] [--nuse=..] [--pmin=-2]"""
import json, sys, numpy as np
from scipy.optimize import minimize
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(a.lstrip("-").split("=") for a in sys.argv[1:] if a.startswith("--"))
nbg = int(opt.get("nbg", 5)); rmax = float(opt.get("rmax", 1)); rmin = float(opt.get("rmin", 0)); pmin = int(opt.get("pmin", -2))
d = json.load(open(args[0]))
co = {tuple(map(int, k.split(","))): complex(float(v[0]), float(v[1])) for k, v in d["coef"].items()}
lam = float(d["nu"][1]); nmax = max(k[0] for k in co)
if "nuse" in opt: nmax = int(opt["nuse"])
rows = [json.loads(l) for f in args[1:] for l in open(f)]
rows = [x for x in rows if rmin <= x["r"] <= rmax]
r = np.array([x["r"] for x in rows]); I = np.array([x["I"] for x in rows]); c = rows[0]["c"]
o = np.argsort(r); r, I = r[o], I[o]; L = np.log(r)
C = [sum(v * c ** mc for (n, mc), v in co.items() if n == nn) for nn in range(nmax + 1)]
def series(n_hi, scale=None):
    out = np.zeros_like(r)
    for n in range(n_hi + 1):
        out += ((C[n] * (scale[n] if scale else 1)) * r ** n * np.exp(2j * lam * L)).real
    return out
def resid(R):
    cols = [r ** n for n in range(pmin, pmin + nbg + 1)] + [r ** n * L for n in range(pmin, pmin + nbg + 1)]
    A = np.stack(cols, 1); w = r ** (-pmin)            # weight so that all points count alike
    A, R = A * w[:, None], R * w
    sc = np.abs(A).max(0)
    cf, *_ = np.linalg.lstsq(A / sc, R, rcond=None)
    return np.abs((R - (A / sc) @ cf) / (w * abs(C[0]))).max()
print(f"{len(r)} points, r in [{r[0]:.4g}, {r[-1]:.4g}], c = {c}; background r^n, r^n ln r, n = {pmin}..{pmin + nbg} ({2 * nbg + 2} parameters)")
print(f"  |C_n/C_0| = {[round(abs(z / C[0]), 3) for z in C]}")
print("  residual / |C_0| after subtracting the series through order n:")
print("   none: %.2e" % resid(I), "  " + "  ".join(f"n<={n}: {resid(I - series(n)):.2e}" for n in range(nmax + 1)))
for n in range(min(3, nmax + 1)):
    def f(x):
        sc = [1] * (nmax + 1); sc[n] = 1 + x[0] + 1j * x[1]
        return resid(I - series(nmax, sc))
    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-20)) for x0 in ([0, 0], [0.3, 0.3], [-0.5, 0.2])), key=lambda z: z.fun)
    print(f"   best-fit C_{n}/C_{n}(series) - 1 = {best.x[0]:+.2e} {best.x[1]:+.2e}i   (residual {best.fun:.2e})")
