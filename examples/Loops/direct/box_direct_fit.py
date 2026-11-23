"""Fit of the direct values of the planar box (direct_box.py) against the collapsed series of parity_box_series.py.

At equal external momenta k = 1 the planar box with the loop cut Pc is I(r, beta) = sum_n I_n(r) e^(i n beta) with real
harmonics I_n. A_n = I_n + I_-n multiplies cos(n beta) (A_0 = I_0) and D_n = I_n - I_-n multiplies i sin(n beta), the
parity-odd part. For X = A_n or D_n the series predicts

    X(r) = background + c { 2 Re[r^(3 + 4 i mu) (a_0 + lam_p sum_{N>=1} a_N r^N)] + lam_s 2 Re[r^(3 + 2 i mu) sum_N b_N r^N]
                            + lam_0 sum_N r^(3 + N) [F_N or L_N ln r] }

with c = lam_p = lam_s = lam_0 = 1, where a, b, F, L are the sums (for A_n) or differences (for D_n) of the coefficients
of the harmonics n and -n of class PP (planar attachment, unit weight; pp_coefs.py). Of the non-oscillating terms,
F_N is predicted where r^(3 + N) is not analytic in the vector s (N even for A_n, odd for D_n), and L_N otherwise.
Backgrounds: 'par' = powers of r of the parity of the harmonic (even for A_n, odd for D_n: the cut, the hard region,
and the quadrature of both are analytic in the vector s), in which all three kinds of term are predictions;
'poly' = powers and logarithms of both parities, which absorb the non-oscillating terms.

The stored runs of Nov 22, 2026 (data_box/, (mu, kappa) = (1, 1), 32 values of r from 0.01 to 0.35, loop cuts 0.8 and
1.2, azimuth grid M = 10) are the default. `summary` collects the numbers of record (../box_direct_record.json).

Usage: box_direct_fit.py [DATA.jsonl COEFS.json] [--nonosc COEFS_N4.json] [--rmax 0.25] [--nmax N] [--json OUT]
       box_direct_fit.py --summary          (the numbers of record, from the stored runs)
"""
import argparse
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_box")
MU = 1.0


def load_rows(path):
    """rows of direct_box.py, with the parts of one (r, Pc) summed; returns {r: I_n dict} for the M of the file"""
    acc = {}
    for line in open(path):
        d = json.loads(line)
        key = (d["r"], d["Pc"], d["M"])
        Z = np.array([complex(*z) for z in d["Z"]])
        Zc = np.array([complex(*z) for z in d["Zc"]])
        n = 1 if not d.get("part") else int(d["part"].split("/")[1])
        e = acc.setdefault(key, dict(Z=0, Zc=0, parts=set(), n=n, npts=0, cpu=0.0))
        tag = d.get("part") or "all"
        if tag in e["parts"]:
            continue
        e["parts"].add(tag)
        e["Z"], e["Zc"], e["npts"], e["cpu"] = e["Z"] + Z, e["Zc"] + Zc, e["npts"] + d["npts"], e["cpu"] + d["cpu"]
    out = {}
    for (r, Pc, M), e in sorted(acc.items()):
        if len(e["parts"]) != e["n"]:
            continue
        Zt = e["Z"] + e["Zc"].sum()
        I = Zt + np.conj(Zt[(-np.arange(M)) % M])
        F = (np.fft.fft(I) / M).real
        out[r] = dict(I=I, F={n: F[n % M] for n in range(-(M // 2) + 1, M // 2 + 1)}, Zc=e["Zc"], Pc=Pc, M=M, npts=e["npts"], cpu=e["cpu"])
    return out


def harmonic(rows, n, kind):
    r = np.array(sorted(rows))
    if n == 0:
        return r, np.array([rows[x]["F"][0] for x in r])
    s = 1 if kind == "A" else -1
    return r, np.array([rows[x]["F"][n] + s * rows[x]["F"][-n] for x in r])


def coefs(cf, n, kind, nmax=None):
    """series coefficients of A_n or D_n: pair a_N, single b_N (complex), non-oscillating F_N, L_N (real)"""
    s = 1 if kind == "A" else -1

    def comb(tab, cast):
        p = [cast(x) for x in tab.get(str(n), [])]
        m = [cast(x) for x in tab.get(str(-n), [])]
        L = max(len(p), len(m))
        p, m = p + [0] * (L - len(p)), m + [0] * (L - len(m))
        if n == 0:
            return p
        return [x + s * y for x, y in zip(p, m)]
    c = lambda x: complex(float(x[0]), float(x[1]))          # noqa: E731
    f = lambda x: float(x[0])                                 # noqa: E731
    out = dict(a=comb(cf["pair"], c), b=comb(cf["single"], c), F=comb(cf.get("nonosc_F", {}), f), L=comb(cf.get("nonosc_L", {}), f))
    if nmax is not None:
        out = {k: v[:nmax + 1] for k, v in out.items()}
    return out


def osc(r, c, w, n0=0):
    amp = sum(c[N] * r ** N for N in range(n0, len(c))) if len(c) > n0 else 0 * r
    return 2 * (r ** 3 * np.exp(1j * w * MU * np.log(r)) * amp).real


def nonosc(C, r, kind, n):
    """predicted non-oscillating terms: F_N where r^(3 + N) has the wrong parity for the background, L_N ln r otherwise"""
    out = 0 * r
    for N in range(len(C["F"])):
        power_type = (N % 2 == 0) if kind == "A" else (N % 2 == 1)
        out = out + (C["F"][N] * r ** (3 + N) if power_type else C["L"][N] * r ** (3 + N) * np.log(r))
    return out


def background(r, kind, mode, nbg):
    if mode == "par":
        p0 = 0 if kind == "A" else 1
        return [r ** (p0 + 2 * j) for j in range(nbg)]
    cols = []
    j = 0
    while len(cols) < nbg:
        cols.append(r ** j)
        if j >= 3:
            cols.append(r ** j * np.log(r))
        j += 1
    return cols[:nbg]


def solve(y, cols, wt=None):
    A = np.stack(cols, 1)
    wt = np.ones_like(y) if wt is None else wt
    sc = np.abs(A * wt[:, None]).max(0)
    x, *_ = np.linalg.lstsq(A * wt[:, None] / sc, y * wt, rcond=None)
    return x / sc, y - (A / sc) @ x


def fit(r, y, C, kind, n, mode, nbg, free=("c", "p", "s", "0"), wpow=3.0, drop=()):
    """Least squares with weights r^-wpow (wpow = 3 weights every point by the size of the signal). Returns the fitted
    factors and the largest residual over the envelope 2 |a_0| r^3."""
    lead, corr, sing, no = osc(r, C["a"][:1], 4), osc(r, C["a"], 4, 1), osc(r, C["b"], 2), nonosc(C, r, kind, n)
    env = 2 * r ** 3 * max(abs(C["a"][0]), 1e-300)
    B = background(r, kind, mode, nbg)
    wt = r ** -wpow
    if mode == "poly":
        free = tuple(f for f in free if f != "0")
        no = 0 * r
    cols, names, fixed = [], [], no * 0
    groups = {"c": lead, "p": corr, "s": sing, "0": no}
    for k in drop:
        groups[k] = 0 * r
    # the overall constant c multiplies everything that is not given its own free factor
    base = lead + sum(groups[k] for k in ("p", "s", "0") if k not in free)
    if "c" in free:
        cols.append(base)
        names.append("c")
    else:
        fixed = fixed + base
    for k in ("p", "s", "0"):
        if k in free:
            cols.append(groups[k])
            names.append(k)
    if len(B) + len(cols) >= r.size:
        return None
    x, res = solve(y - fixed, B + cols, wt)
    out = dict(kind=kind, n=n, mode=mode, nbg=nbg, npts=int(r.size), residual=float(np.max(np.abs(res) / env)),
               rms=float(np.sqrt(np.mean((res / env) ** 2))))
    cval = x[len(B) + names.index("c")] if "c" in names else 1.0
    out["c"] = float(cval)
    for k, lab in (("p", "lam_p"), ("s", "lam_s"), ("0", "lam_0")):
        if k in names:
            out[lab] = float(x[len(B) + names.index(k)] / cval)
    return out


def fit_signed(rows, cf, n, nbg, rmax, rmin=0.0, nmax=None, free=("c", "p", "s", "0"), wpow=3.0):
    """The harmonic I_n itself (n != 0), I_n = (A_|n| + sign(n) D_|n|)/2: its background has both parities
    (nbg even and nbg odd powers), and all three kinds of term are predictions."""
    r, A = harmonic(rows, abs(n), "A")
    _, D = harmonic(rows, abs(n), "D")
    sg = 1 if n > 0 else -1
    m = (r <= rmax) & (r >= rmin)
    r, y = r[m], ((A + sg * D) / 2)[m]
    CA, CD = coefs(cf, abs(n), "A", nmax), coefs(cf, abs(n), "D", nmax)
    C = {k: [(a + sg * d) / 2 for a, d in zip(CA[k], CD[k])] for k in ("a", "b")}
    lead, corr, sing = osc(r, C["a"][:1], 4), osc(r, C["a"], 4, 1), osc(r, C["b"], 2)
    no = (nonosc(CA, r, "A", abs(n)) + sg * nonosc(CD, r, "D", abs(n))) / 2
    env = 2 * r ** 3 * abs(C["a"][0])
    B = [r ** j for j in range(2 * nbg)]
    groups = {"p": corr, "s": sing, "0": no}
    base = lead + sum(groups[k] for k in groups if k not in free)
    cols, names = [base], ["c"]
    for k in ("p", "s", "0"):
        if k in free:
            cols.append(groups[k])
            names.append(k)
    if len(B) + len(cols) >= r.size:
        return None
    x, res = solve(y, B + cols, r ** -wpow)
    cval = x[len(B)]
    out = dict(kind="I", n=n, mode="par", nbg=2 * nbg, npts=int(r.size), rmax=rmax, c=float(cval), residual=float(np.max(np.abs(res) / env)))
    for k, lab in (("p", "lam_p"), ("s", "lam_s"), ("0", "lam_0")):
        if k in names:
            out[lab] = float(x[len(B) + names.index(k)] / cval)
    return out


def leading_absent(rows, cf, n, kind, nbg, rmax, nmax=None, ref=None):
    """For a harmonic that the series does not have at leading order: fit, on top of the predicted terms, free
    amplitudes of r^(3 + 4 i mu), r^(3 + 2 i mu) and of the non-oscillating r^3 term (r^3 for A_n, r^3 ln r for D_n), and
    return their moduli over `ref` (the leading pair coefficient of a reference harmonic)."""
    r, y = harmonic(rows, n, kind)
    m = r <= rmax
    r, y = r[m], y[m]
    C = coefs(cf, n, kind, nmax)
    pred = osc(r, C["a"], 4) + osc(r, C["b"], 2) + nonosc(C, r, kind, n)
    ph4, ph2 = r ** 3 * np.exp(4j * MU * np.log(r)), r ** 3 * np.exp(2j * MU * np.log(r))
    extra = [2 * ph4.real, -2 * ph4.imag, 2 * ph2.real, -2 * ph2.imag, r ** 3 if kind == "A" else r ** 3 * np.log(r)]
    B = background(r, kind, "par", nbg)
    x, res = solve(y - pred, B + extra, r ** -3.0)
    e = x[len(B):]
    return dict(kind=kind, n=n, nbg=nbg, rmax=rmax, pair=float(abs(complex(e[0], e[1])) / ref), single=float(abs(complex(e[2], e[3])) / ref),
                nonosc=float(abs(e[4]) / ref), residual=float(np.max(np.abs(res) / (2 * r ** 3 * ref))))


def merge(cf, cf_nonosc):
    """pair and single-frequency coefficients of one file with the non-oscillating ones of another (lower order)"""
    out = dict(cf)
    for k in ("nonosc_F", "nonosc_L"):
        if k not in out and cf_nonosc is not None:
            out[k] = cf_nonosc[k]
    return out


def cut_difference(rows_a, rows_b, kind, n, terms=6):
    """The difference of two runs with different loop cuts for A_n or D_n, fitted with powers of r of the parity of the
    harmonic: (largest residual, largest difference, number of points, leading coefficient)."""
    rs = np.array(sorted(set(rows_a) & set(rows_b)))

    def h(rows):
        if n == 0:
            return np.array([rows[r]["F"][0] for r in rs])
        s = 1 if kind == "A" else -1
        return np.array([rows[r]["F"][n] + s * rows[r]["F"][-n] for r in rs])
    d = h(rows_b) - h(rows_a)
    p0 = 0 if kind == "A" else 1
    x, res = solve(d, [rs ** (p0 + 2 * j) for j in range(terms)])
    return float(np.max(np.abs(res))), float(np.max(np.abs(d))), int(rs.size), float(x[0])


def stored():
    cf = merge(json.load(open(os.path.join(DATA, "box_coefs_1_1_N6.json"))), json.load(open(os.path.join(DATA, "box_coefs_1_1_N4.json"))))
    return load_rows(os.path.join(DATA, "box_1_1_pc0.8.jsonl")), load_rows(os.path.join(DATA, "box_1_1_pc1.2.jsonl")), cf


def summary(rmaxs=(0.2, 0.25, 0.3), nbgs=(6, 7, 8, 9)):
    """Numbers of record from the stored runs: for the cut 0.8, the largest deviations from one of the fitted factors
    over the ranges and background orders, per harmonic; the test that |n| = 3, 4 are absent at leading order; and the
    difference of the two cuts."""
    a, b, cf = stored()
    out = {}
    for kind, n in (("A", 0), ("A", 2), ("D", 2), ("A", 1), ("D", 1)):
        r, y = harmonic(a, n, kind)
        C = coefs(cf, n, kind)
        dev = dict(c=0.0, lam_p=0.0, lam_s=0.0, lam_0=0.0, residual=0.0)
        without = []
        for rmax in rmaxs:
            m = r <= rmax
            for nbg in nbgs:
                e = fit(r[m], y[m], C, kind, n, "par", nbg)
                for k in ("c", "lam_p", "lam_s", "lam_0"):
                    dev[k] = max(dev[k], abs(e[k] - 1))
                dev["residual"] = max(dev["residual"], e["residual"])
                without.append(fit(r[m], y[m], C, kind, n, "par", nbg, free=("c",), drop=("p",))["residual"])
        dev["residual_without_corrections_min"] = min(without)
        out[f"{kind}_{n}"] = dev
    for n in (-2, 2):
        dev = dict(c=0.0, lam_p=0.0, lam_s=0.0, lam_0=0.0)
        for rmax in rmaxs:
            for nbg in (5, 6, 7):
                e = fit_signed(a, cf, n, nbg, rmax)
                for k in dev:
                    dev[k] = max(dev[k], abs(e[k] - 1))
        out[f"I_{n}"] = dev
    ref = abs(coefs(cf, 2, "D")["a"][0]) / 2
    out["leading_absent"] = {f"{kind}_{n}": max(max(leading_absent(a, cf, n, kind, 7, rmax, None, ref)[k] for k in ("pair", "single", "nonosc"))
                                               for rmax in (0.2, 0.25)) for kind, n in (("A", 3), ("D", 3), ("A", 4), ("D", 4))}
    out["cut_difference"] = {}
    for kind, n in (("A", 0), ("A", 2), ("D", 2), ("A", 1), ("D", 1)):
        res, size, npts, lead = cut_difference(a, b, kind, n)
        out["cut_difference"][f"{kind}_{n}"] = dict(residual=res, size=size, points=npts, leading_coefficient=lead)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.path.join(DATA, "box_1_1_pc0.8.jsonl"))
    ap.add_argument("coefs", nargs="?", default=os.path.join(DATA, "box_coefs_1_1_N6.json"))
    ap.add_argument("--nonosc", default=os.path.join(DATA, "box_coefs_1_1_N4.json"),
                    help="coefficient file with the non-oscillating terms, if `coefs` has none")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rmax", type=float, nargs="*", default=[0.2, 0.25, 0.3])
    ap.add_argument("--rmin", type=float, default=0.0)
    ap.add_argument("--nmax", type=int, default=None)
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.summary:
        print(json.dumps(summary(), indent=1))
        return
    rows = load_rows(args.data)
    cf = merge(json.load(open(args.coefs)), json.load(open(args.nonosc)) if args.nonosc else None)
    res = []
    r0 = next(iter(rows.values()))
    print(f"{len(rows)} values of r from {min(rows)} to {max(rows)}, cut {r0['Pc']}, M = {r0['M']}, pair and single-frequency series "
          f"through N = {args.nmax if args.nmax is not None else cf['nmax']}, non-oscillating through N = {len(cf['nonosc_F']['0']) - 1}")
    for kind, n in (("A", 0), ("A", 2), ("D", 2), ("A", 1), ("D", 1), ("A", 3), ("D", 3)):
        r, y = harmonic(rows, n, kind)
        C = coefs(cf, n, kind, args.nmax)
        if abs(C["a"][0]) == 0:
            continue
        for rmax in args.rmax:
            m = (r <= rmax) & (r >= args.rmin)
            print(f"--- {kind}_{n}, r <= {rmax} ({int(m.sum())} points): c, lam_p, lam_s, lam_0 (largest residual) | only c and lam_p free | "
                  f"nothing free | without the orders N >= 1")
            for mode, nbgs in (("par", (6, 7, 8, 9)), ("poly", (10, 12))):
                for nbg in nbgs:
                    e = fit(r[m], y[m], C, kind, n, mode, nbg)
                    if e is None:
                        continue
                    e1 = fit(r[m], y[m], C, kind, n, mode, nbg, free=("c", "p"))
                    e0 = fit(r[m], y[m], C, kind, n, mode, nbg, free=())
                    ex = fit(r[m], y[m], C, kind, n, mode, nbg, free=("c",), drop=("p",))
                    e.update(rmax=rmax, lam_p_alone=e1["lam_p"], c_alone=e1["c"], residual_p_alone=e1["residual"],
                             residual_no_free=e0["residual"], residual_without_corrections=ex["residual"])
                    res.append(e)
                    extra = "".join(f", {e[k]:.4f}" for k in ("lam_s", "lam_0") if k in e)
                    print(f"  {mode:4s} {nbg:2d}: {e['c']:.5f}, {e['lam_p']:.5f}{extra} ({e['residual']:.0e}) | "
                          f"{e1['c']:.5f}, {e1['lam_p']:.5f} ({e1['residual']:.0e}) | {e0['residual']:.0e} | {ex['residual']:.0e}")
    for n in (-2, 2):
        for rmax in args.rmax:
            for nbg in (5, 6, 7):
                e = fit_signed(rows, cf, n, nbg, rmax, args.rmin, args.nmax)
                e1 = fit_signed(rows, cf, n, nbg, rmax, args.rmin, args.nmax, free=("c", "p"))
                if e is None:
                    continue
                e.update(lam_p_alone=e1["lam_p"], c_alone=e1["c"])
                res.append(e)
                print(f"--- I_{n}, r <= {rmax}, {2 * nbg} background terms: c = {e['c']:.5f}, lam_p = {e['lam_p']:.5f}, lam_s = {e['lam_s']:.4f}, "
                      f"lam_0 = {e['lam_0']:.4f} ({e['residual']:.0e}) | only c and lam_p free: {e1['c']:.5f}, {e1['lam_p']:.5f}")
    ref = abs(coefs(cf, 2, "D", args.nmax)["a"][0]) / 2
    for kind, n in (("A", 3), ("D", 3), ("A", 4), ("D", 4)):
        for rmax in args.rmax[:2]:
            e = leading_absent(rows, cf, n, kind, 7, rmax, args.nmax, ref)
            res.append(dict(e, test="leading_absent"))
            print(f"--- {kind}_{n}, r <= {rmax}: free leading terms over |c_-2|: pair {e['pair']:.1e}, single {e['single']:.1e}, "
                  f"non-oscillating {e['nonosc']:.1e} (residual {e['residual']:.0e})")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
