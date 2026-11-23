"""First template of the parity-odd part of the box of the helical-vector loop at equal external momenta.

The parity-odd part of the box does not depend on the loop cut once the cut is far above the momenta where the hard
lines go soft (it has no ultraviolet divergence, and the terms with a contact line have no parity-odd part), so it
can be assembled from the regions of loop momentum computed by direct_box.py:

    inner    |l| + |s - l| < P_in             (the collapsed-region runs, loop grid graded towards l = 0 and l = s)
    ring     P_in < |l| + |s - l| < 2.6       (panels graded towards the ring where the hard lines have zero momentum)
    shells   2.6 < |l| + |s - l| < 25         (time grid for large line energies)

For each r = s/k the script sums the harmonics D_n = I_n - I_-n of the planar box, I = sum_n I_n e^(i n beta), over the
regions, and writes them with the series of the collapsed region and a fitting form to a JSON file.

Usage: box_template.py OUT.json [COEFS_N6.json COEFS_N4.json RUNS.jsonl ...]     (default: the stored runs of data_box/)
"""
import json
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_box")
MU = 1.0
NH = 4                      # harmonics kept


def load(paths):
    """{(r, Pmin, Pc, ring, M): dict(F = {n: I_n}, I = I(beta_m), npts, cpu)} with the parts of a run summed"""
    acc = {}
    for path in paths:
        for line in open(path):
            d = json.loads(line)
            key = (d["r"], d.get("Pmin") or 0.0, d["Pc"], bool(d.get("ring")), d["M"])
            tag = d.get("part") or "all"
            e = acc.setdefault(key, dict(Z=0, Zc=0, parts=set(), n=1 if tag == "all" else int(tag.split("/")[1]), npts=0, cpu=0.0))
            if tag in e["parts"]:
                continue
            e["parts"].add(tag)
            e["Z"] = e["Z"] + np.array([complex(*z) for z in d["Z"]])
            e["Zc"] = e["Zc"] + np.array([complex(*z) for z in d["Zc"]])
            e["npts"] += d["npts"]
            e["cpu"] += d["cpu"]
    out = {}
    for key, e in acc.items():
        if len(e["parts"]) != e["n"]:
            print("incomplete run, skipped:", key, sorted(e["parts"]))
            continue
        M = key[4]
        Zt = e["Z"] + e["Zc"].sum()
        I = Zt + np.conj(Zt[(-np.arange(M)) % M])
        F = (np.fft.fft(I) / M).real
        out[key] = dict(F={n: F[n % M] for n in range(-NH, NH + 1)}, I=I, npts=e["npts"], cpu=e["cpu"])
    return out


def assemble(runs, r, top=25.0, mmax=20):
    """Tile 0 < P < top with the runs at this r, taking the largest M <= mmax where a region was run twice. Returns the
    list of regions used, or None if they leave a gap."""
    mine = [k for k in runs if k[0] == r and k[4] <= mmax]
    used, edge = [], 0.0
    while edge < top - 1e-9:
        cands = [k for k in mine if abs(k[1] - edge) < 1e-9]
        if not cands:
            return None
        # prefer the run that reaches furthest only if it is the inner run; otherwise the largest M
        k = max(cands, key=lambda k: (k[4], k[2]) if edge > 0 else (k[2] <= 1.25, k[4], -k[2]))
        used.append(k)
        edge = k[2]
    return used


def series_D(cf6, cf4, n, r, nmax=6):
    """the non-analytic part of D_n in the collapsed region: pair and single-frequency terms through nmax, and the
    non-oscillating terms through N = 4 (F_N for odd N, L_N ln r for even N)"""
    def c(tab, k):
        p = [complex(float(a), float(b)) for a, b in tab.get(str(k), [])]
        return p + [0] * (nmax + 1 - len(p))
    a = [x - y for x, y in zip(c(cf6["pair"], n), c(cf6["pair"], -n))][:nmax + 1]
    b = [x - y for x, y in zip(c(cf6["single"], n), c(cf6["single"], -n))][:nmax + 1]
    F = [x.real - y.real for x, y in zip(c(cf4["nonosc_F"], n), c(cf4["nonosc_F"], -n))][:5]
    L = [x.real - y.real for x, y in zip(c(cf4["nonosc_L"], n), c(cf4["nonosc_L"], -n))][:5]
    pair = 2 * (r ** 3 * np.exp(4j * MU * np.log(r)) * sum(a[N] * r ** N for N in range(len(a)))).real
    single = 2 * (r ** 3 * np.exp(2j * MU * np.log(r)) * sum(b[N] * r ** N for N in range(len(b)))).real
    no = sum((F[N] * r ** (3 + N) if N % 2 == 1 else L[N] * r ** (3 + N) * np.log(r)) for N in range(len(F)))
    return dict(pair=pair, single=single, nonosc=no, total=pair + single + no, a=a, b=b, F=F, L=L)


def main(argv):
    out = argv[0]
    paths = argv[1:] or [os.path.join(DATA, f) for f in ("box_coefs_1_1_N6.json", "box_coefs_1_1_N4.json", "box_1_1_pc0.8.jsonl", "box_1_1_regions.jsonl")]
    cf6, cf4 = json.load(open(paths[0])), json.load(open(paths[1]))
    runs = load(paths[2:])
    rs = sorted({k[0] for k in runs})
    table = []
    for r in rs:
        used = assemble(runs, r)
        if used is None:
            continue
        D = {n: sum(runs[k]["F"][n] - runs[k]["F"][-n] for k in used) for n in range(1, NH + 1)}
        parts = {f"{k[1]:g}-{k[2]:g}{' ring' if k[3] else ''} (M={k[4]})": {n: runs[k]["F"][n] - runs[k]["F"][-n] for n in range(1, NH + 1)} for k in used}
        # the same with the finest azimuth grid available in each region (M = 40 near the ring at r = 0.3)
        fine = assemble(runs, r, mmax=40)
        Dfine = {n: sum(runs[k]["F"][n] - runs[k]["F"][-n] for k in fine) for n in D} if any(k[4] == 40 for k in fine) else None
        # M = 40 minus M = 20 on the panels next to the ring, 1.8 < P < 2.2, where both were run
        k40, k20 = (r, 1.8, 2.2, True, 40), (r, 1.8, 2.2, True, 20)
        corr = {n: (runs[k40]["F"][n] - runs[k40]["F"][-n]) - (runs[k20]["F"][n] - runs[k20]["F"][-n]) for n in D} if k40 in runs and k20 in runs else None
        # values in beta at the ten angles common to all regions
        Iodd = sum(runs[k]["I"][::runs[k]["I"].size // 10].imag for k in used)
        table.append(dict(r=r, D_M20=D, D_M40=Dfine, M40_minus_M20_central=corr, regions=parts, Im_I_beta=[float(x) for x in Iodd],
                          cpu=sum(runs[k]["cpu"] for k in used), npts=sum(runs[k]["npts"] for k in used)))
    r = np.array([e["r"] for e in table])
    # ---- correction for the azimuth grid. The change from M = 20 to 40 on the central panels is measured at three
    # values of r and interpolated as r (a + b r^2). At r = 0.3, where every region near the ring was run at M = 40, the
    # whole change is `ratio` times that of the central panels. D = D_M20 + ratio * central(r), with one third of the
    # correction as the uncertainty from the grid that remains.
    cr = np.array([e["r"] for e in table if e["M40_minus_M20_central"]])
    e3 = [e for e in table if e["D_M40"]][0]
    azimuth = {}
    for n in range(1, NH + 1):
        cv = np.array([e["M40_minus_M20_central"][n] for e in table if e["M40_minus_M20_central"]])
        A = np.stack([cr, cr ** 3], 1)
        ab, *_ = np.linalg.lstsq(A / cr[:, None], cv / cr, rcond=None)
        ratio = (e3["D_M40"][n] - e3["D_M20"][n]) / e3["M40_minus_M20_central"][n]
        azimuth[n] = dict(central_fit_a_b=[float(v) for v in ab], ratio_all_to_central=float(ratio), central_measured={float(a): float(b) for a, b in zip(cr, cv)},
                          fit_residual=float(np.max(np.abs(A @ ab - cv) / np.abs(cv))))
        for e in table:
            c = ratio * (ab[0] * e["r"] + ab[1] * e["r"] ** 3)
            e.setdefault("D", {})[n] = e["D_M20"][n] + c
            e.setdefault("D_uncertainty", {})[n] = abs(c) / 3
    res = dict(azimuth_correction=azimuth, conventions=dict(
        quantity="D_n(r) = I_n - I_-n of the planar box at equal external momenta k = 1, I_box = sum_n I_n e^(i n beta), unit vertices, "
                 "loop measure d^3l/(2 pi)^3, in units of k^-5. The parity-odd part of the planar box is i sum_n D_n sin(n beta), and that of "
                 "the planar plus the crossed box is 2 i [D_2 sin(2 beta) + D_4 sin(4 beta)].",
        trispectrum="16 k^4 <dphi^4>' contains lambda_2^4 (I_box + I_box^crossed) (Eq. (triweights) of Sec. 8.4); only this term is parity-odd",
        point="(mu, kappa) = (1, 1)", beta="dihedral angle between the planes (s, k1) and (s, k3), s along z, k1 in the xz plane",
        accuracy="D_M20: azimuth grid M = 20 between P = 0.8 and 5 (M = 10 elsewhere). D_M40 (r = 0.3 only): M = 40 between 0.8 and 3.6. "
                 "D = D_M20 plus the modeled change from M = 20 to 40 (azimuth_correction), D_uncertainty = one third of that change. "
                 "The uniform azimuth grid converges slowly near the ring |l| = |s - l| = k, where the hard lines have zero momentum. "
                 "D_2 is good to about 3 percent and D_4 to tens of percent."), points=table)
    # ---- analytic part: D_n minus the series, where the series converges (r <= 0.35)
    fits = {}
    for n in (1, 2, 3, 4):
        Dn = np.array([e["D"][n] for e in table])
        S = series_D(cf6, cf4, n, r)
        m = r <= 0.36
        rest = Dn - S["total"]
        cols = [r[m] ** (1 + 2 * j) for j in range(min(3, int(m.sum()) - 1))]
        A = np.stack(cols, 1)
        x, *_ = np.linalg.lstsq(A / r[m][:, None], rest[m] / r[m], rcond=None)
        resid = rest[m] - A @ x
        fits[n] = dict(analytic_coefficients_r1_r3_r5=[float(v) for v in x], points=int(m.sum()),
                       largest_residual_over_D=float(np.max(np.abs(resid) / np.abs(Dn[m]))),
                       series=dict(pair=[[v.real, v.imag] for v in S["a"]], single=[[v.real, v.imag] for v in S["b"]], F=S["F"], L=S["L"]),
                       D=[float(v) for v in Dn], series_total=[float(v) if rr <= 0.36 else None for v, rr in zip(S["total"], r)], pair_envelope=[float(2 * abs(S["a"][0]) * v ** 3) for v in r])
        print(f"D_{n}: " + "  ".join(f"r={a:.3g}: {b:+.4e}" + (f" (series {c:+.2e})" if a <= 0.36 else "") for a, b, c in zip(r, Dn, S["total"])))
        print(f"      analytic part b1 r + b3 r^3 + b5 r^5 on r <= 0.36 ({int(m.sum())} points): {x}, residual/D <= {fits[n]['largest_residual_over_D']:.1e}")
    res["collapsed_region"] = fits
    # ---- fitting form over the whole range: D_n = r (p0 + p1 r^2 + p2 r^4 + p3 r^6) + r^3 2 Re[(u0 + u1 r) r^(4 i mu)]
    form = {}
    for n in (2, 4):
        Dn = np.array([e["D"][n] for e in table])
        ph = r ** 3 * np.exp(4j * MU * np.log(r))
        for npoly, nosc in ((3, 0), (4, 0), (3, 1)):
            cols = [r ** (1 + 2 * j) for j in range(npoly)]
            for j in range(nosc):
                cols += [2 * (ph * r ** j).real, -2 * (ph * r ** j).imag]
            if len(cols) >= r.size:
                continue
            A = np.stack(cols, 1)
            x, *_ = np.linalg.lstsq(A / r[:, None], Dn / r, rcond=None)
            resid = Dn - A @ x
            form[f"D_{n} poly{npoly} osc{nosc}"] = dict(coefficients=[float(v) for v in x], largest_residual_over_D=float(np.max(np.abs(resid / Dn))),
                                                        form="sum_j p_j r^(1 + 2 j) + r^3 2 Re[sum_j (u_j + i v_j) r^j r^(4 i mu)], coefficients p then (u, v) pairs")
            print(f"D_{n} fitting form with {npoly} odd powers and {nosc} oscillating terms: residual/D <= {form[f'D_{n} poly{npoly} osc{nosc}']['largest_residual_over_D']:.1e}")
    res["fitting_forms"] = form
    # ---- size: <zeta^4>'/(4 P(k)^2 P(s)) = 2 pi^2 Delta_zeta^2 (m_A^4/u^4) r^6 A, A multiplying lambda_1^2 s^3 k^-8 (Eq. (tauloop));
    # the parity-odd part of A is i [2 f^2 D_2/r^3 sin(2 beta) + 2 f^2 D_4/r^3 sin(4 beta)], f = lambda_2^2/lambda_1 = -2 m_A^2
    m2 = MU ** 2 + 0.25
    pre = 2 * np.pi ** 2 * 2.1e-9 * m2 ** 2
    f2 = (2 * m2) ** 2
    Y221 = 0.106482064565690
    b1 = fits[2]["analytic_coefficients_r1_r3_r5"][0]
    res["size"] = dict(prefactor_2pi2_Delta2_mA4=pre, f_squared=f2,
                       sin2beta_coefficient_per_u4=[float(pre * 2 * f2 * e["D"][2] * e["r"] ** 3) for e in table],
                       sin2beta_coefficient_per_u4_series_only=[float(pre * 2 * f2 * v * rr ** 3) if v is not None else None for v, rr in zip(fits[2]["series_total"], r)],
                       analytic_term_sin2beta_per_u4_r4=float(pre * 2 * f2 * b1),
                       analytic_term_tau221_equivalent_per_u4_r4=float(pre * 4 * f2 * b1 / Y221),
                       note="sin2beta_coefficient: the coefficient of i sin(2 beta) in <zeta^4>'/(4 P(k)^2 P(s)) times u^4, at each r of `points`. "
                            "tau221_equivalent assumes the angular structure i shat.(k1hat x k3hat)(k1hat.k3hat), which equal external momenta "
                            "do not test; Y_221 = -0.106482 i X (k1hat.k3hat) as in runs/normalization.py of Nov 19.")
    print("size:", {k: v for k, v in res["size"].items() if k != "note"})
    if out != "-":
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
