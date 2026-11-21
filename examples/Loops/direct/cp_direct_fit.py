"""Fit of the direct contact-pair values against the collapsed series of parity_box_series.py (Sec. 7.4).

The contact-pair diagram of the collapsed trispectrum (one quartic vertex and one hard pair) is direct_vec.py with
X3=2 PW3=2 and c = 'iso': equal external momenta k = 1, soft momentum r = s/k. Each stored row holds
I = S/(r/8), the loop integral with unit vertices, cut at y1 + y2 < Pc. The series predicts

    I(r) = background + c { 2 Re[r^(3 + 4 i mu) (a_0 + lam_p sum_{N>=1} a_N r^N)] + lam_s 2 Re[r^(3 + 2 i mu) sum_N b_N r^N]
                            + lam_0 sum_N r^(3 + N) [F_N (N even) or L_N ln r (N odd)] }

with c = lam_p = lam_s = lam_0 = 1. The coefficients are those of class CP at unit weight (dump of the classes, CP + PC
halved, and parity_box_series.nonoscillating(diagrams=("CP",)) over the class weight).
Backgrounds: 'even' = even powers of r (the cut and the hard region are analytic in the vector s), in which all three
kinds of term are predictions; 'poly' = powers r^j (j = 0, 2, 3, ...) with r^j ln r for even j >= 4, which absorbs
the non-oscillating terms.

Usage: python cp_direct_fit.py [DATA.jsonl COEFS.json]      (default: data_cp/cp_1_1_pc0.5.jsonl, data_cp/cp_coefs_1_1_N4.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_cp")
MU = 1.0


def load(data=None, coefs=None):
    rows = sorted([json.loads(l) for l in open(data or os.path.join(DATA, "cp_1_1_pc0.5.jsonl"))], key=lambda d: d["r"])
    cf = json.load(open(coefs or os.path.join(DATA, "cp_coefs_1_1_N4.json")))
    return dict(r=np.array([d["r"] for d in rows]), I=np.array([d.get("I", d["S"] / (d["r"] / 8)) for d in rows]),
                a=[complex(*x) for x in cf["pair"]], b=[complex(*x) for x in cf["single"]], F=cf["F"], L=cf["L"])


def osc(r, c, w, n0=0):
    amp = sum(c[N] * r ** N for N in range(n0, len(c))) if len(c) > n0 else 0 * r
    return 2 * (r ** 3 * np.exp(1j * w * MU * np.log(r)) * amp).real


def nonosc(D, r):
    return sum((D["F"][k] * r ** (3 + k) if k % 2 == 0 else D["L"][k] * r ** (3 + k) * np.log(r)) for k in range(len(D["F"])))


def background(r, kind, nbg):
    if kind == "even":
        return [r ** (2 * j) for j in range(nbg // 2 + 1)]
    cols = []
    for j in range(nbg + 1):
        if j == 1:
            continue
        cols.append(r ** j)
        if j >= 4 and j % 2 == 0:
            cols.append(r ** j * np.log(r))
    return cols


def solve(y, cols):
    A = np.stack(cols, 1)
    sc = np.abs(A).max(0)
    x, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
    return x / sc, y - (A / sc) @ x


def fits(D, rmax, rmin=0.0):
    """One dict per background: c, lam_p, lam_s, lam_0 (even background only), and residuals over the envelope 2 |a_0| r^3."""
    m = (D["r"] >= rmin) & (D["r"] <= rmax)
    r, I = D["r"][m], D["I"][m]
    lead, corr, sing, no = osc(r, D["a"][:1], 4), osc(r, D["a"], 4, 1), osc(r, D["b"], 2), nonosc(D, r)
    env = 2 * r ** 3 * abs(D["a"][0])
    out = []
    for kind, nbgs in (("even", (10, 12)), ("poly", (6, 7, 8))):
        for nbg in nbgs:
            B = background(r, kind, nbg)
            if len(B) + 4 > r.size:
                continue
            e = dict(kind=kind, terms=len(B), npts=int(r.size))
            if kind == "even":
                x, res = solve(I, B + [lead, corr, sing, no])
                e.update(c=x[-4], lam_p=x[-3] / x[-4], lam_s=x[-2] / x[-4], lam_0=x[-1] / x[-4])
                x1, _ = solve(I, B + [lead + corr + sing + no])
                e["c_one_constant"] = x1[-1]
            else:
                x, res = solve(I, B + [lead + sing, corr])
                e.update(c=x[-2], lam_p=x[-1] / x[-2])
                _, res0 = solve(I, B + [lead + sing])
                e["residual_without_corrections"] = float(np.max(np.abs(res0) / env))
            e["residual"] = float(np.max(np.abs(res) / env))
            out.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in e.items()})
    return out


def cut_difference(run_a, run_b, terms=6):
    """largest residual of an even polynomial fitted to the difference of two runs with different loop cuts"""
    A, B = load(run_a), load(run_b)
    a = dict(zip(A["r"], A["I"]))
    r = np.array([x for x in B["r"] if x in a])
    d = np.array([dict(zip(B["r"], B["I"]))[x] - a[x] for x in r])
    _, res = solve(d, [r ** (2 * j) for j in range(terms)])
    return float(np.max(np.abs(res))), float(np.max(np.abs(d))), int(r.size)


if __name__ == "__main__":
    D = load(*sys.argv[1:3])
    for rmax in (0.1, 0.2, 0.25, 0.36):
        print(f"--- r <= {rmax}")
        for e in fits(D, rmax):
            tail = (f"lam_s = {e['lam_s']:.5f}, lam_0 = {e['lam_0']:.4f}" if e["kind"] == "even"
                    else f"residual without the pair corrections {e['residual_without_corrections']:.1e}")
            print(f"  {e['kind']}, {e['terms']:2d} terms, {e['npts']} points: c = {e['c']:.6f}, lam_p = {e['lam_p']:.5f}, {tail} (residual {e['residual']:.1e})")
    if len(sys.argv) < 3:
        res, size, n = cut_difference(None, os.path.join(DATA, "cp_1_1_pc0.8.jsonl"))
        print(f"--- cuts 0.5 and 0.8: {n} points, difference up to {size:.2e}, even polynomial of 6 terms to {res:.1e}")
