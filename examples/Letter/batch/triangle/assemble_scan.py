"""Assembly of the triangle scan.

assemble_scan.py [--mb DIR ...] [--b DIR ...] [--out FILE] [--record FILE]

MB = A + R_extrap + dT with R(h) = R0 + C exp(-w/h), w = 0.7587 + 2.521 i, from R at
h_s = 0.05 and 0.04, and dT = 2.9 D1(k1) (D1 interpolated linearly in k1, constant outside
[0.3, 2.0]).  B_reg from the P1 brute-force record (mean of nine window choices) with D0 from
data/D0_all.json.  Writes the JSON table and a CSV with the same stem.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from bf_triangle import B_reg

W_EXP = 0.7587 + 2.521j
TAIL_FACTOR = 2.9
D1_ANCHORS = ((0.3, -0.0001314079008136737 + 5.7750320303640024e-08j),
              (1.0, -0.00013691562234585675 + 6.065910684860398e-08j),
              (2.0, -0.00014301632728552782 + 6.453290923857935e-08j))
ERR_EXTRAP = 1.5e-6
ERR_D1 = 5e-6
ERR_TAILPOW = 0.045
ERR_B_ADD = 2e-6
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = ("MB = A + R_extrap + dT; R(h) = R0 + C exp(-w/h), w = 0.7587 + 2.521i from h_s = 0.05, "
         "0.04; dT = 2.9 D1(k1); Breg = mean over Lambda_1 x r_fit of the smooth-tail fit")


def cplx(v):
    return complex(v[0], v[1])


def R_extrap(R005, R004):
    C = (R005 - R004) / (np.exp(-W_EXP / 0.05) - np.exp(-W_EXP / 0.04))
    return R004 - C * np.exp(-W_EXP / 0.04)


def D1(k1):
    ks = [a for a, _ in D1_ANCHORS]
    vs = [v for _, v in D1_ANCHORS]
    if k1 <= ks[0]:
        return vs[0]
    if k1 >= ks[-1]:
        return vs[-1]
    for (a, va), (b, vb) in zip(D1_ANCHORS[:-1], D1_ANCHORS[1:]):
        if a <= k1 <= b:
            return va + (vb - va) * (k1 - a) / (b - a)


def dT(k1):
    return TAIL_FACTOR * D1(k1)


def err_MB_partial(dTv):
    return ERR_EXTRAP + ERR_D1 + ERR_TAILPOW * abs(dTv)


def classify(tag):
    if tag.startswith("k"):
        fam = "momentum"
    elif tag.startswith("Keq"):
        fam = "control_eq"
    elif "_mu" in tag:
        fam = "mass"
    else:
        fam = "control"
    mt = tag.split("_")[-1]
    if mt.startswith("mu"):
        mt = "equal"
    return fam, mt


def load_dirs(dirs):
    recs = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                recs[fn] = json.load(open(os.path.join(d, fn)))     # later overrides earlier
    return recs


def assemble(mb_dirs, b_dirs, record=None, d0_file=None):
    d0tab = json.load(open(d0_file or os.path.join(HERE, "data", "D0_all.json")))
    mb = load_dirs(mb_dirs)
    bf = load_dirs(b_dirs)
    prior = {}
    if record and os.path.exists(record):
        prior = {p["tag"]: p for p in json.load(open(record))["points"]}
    tags = sorted({fn[:-len("_A.json")] for fn in mb if fn.endswith("_A.json") and "A" in mb[fn]})
    points = []
    for tag in tags:
        A = mb.get(f"{tag}_A.json")
        R5 = mb.get(f"{tag}_R0.05.json")
        R4 = mb.get(f"{tag}_R0.04.json")
        if A is None or R5 is None or R4 is None:
            print(f"missing MB records for {tag}")
            continue
        mus, k = A["mus"], A["k"]
        Ae, r5, r4 = cplx(A["A"]), cplx(R5["R"]), cplx(R4["R"])
        Rx = R_extrap(r5, r4)
        dTv = dT(k[0])
        MB = Ae + Rx + dTv
        D0 = cplx(d0tab[str(tuple(float(m) for m in mus))]["D0"])
        fam, mt = classify(tag)
        p = {"tag": tag, "mus": mus, "k": k, "shape_x": k[0] / (k[1] + k[2]),
             "mu_eff": float(np.mean(mus)), "family": fam, "mass_tag": mt,
             "A": [Ae.real, Ae.imag], "R_005": [r5.real, r5.imag], "R_004": [r4.real, r4.imag],
             "R_extrap": [Rx.real, Rx.imag], "dT": [dTv.real, dTv.imag], "MB": [MB.real, MB.imag],
             "D0": [D0.real, D0.imag], "err_MB_partial": err_MB_partial(dTv),
             "D0_MB_vs_direct": abs(cplx(R4["D0"]) / D0 - 1)}
        if tag in prior and "err_MB_abs" in prior[tag]:
            p["err_MB_abs"] = prior[tag]["err_MB_abs"]
        b1 = bf.get(f"{tag}_P1.json")
        if b1 is not None and "B_150000" in b1:
            Br, spread = B_reg(b1, D0)
            errB = spread + ERR_B_ADD
            b2 = bf.get(f"{tag}_P2.json")
            if b2 is not None and "B_150000" in b2:
                Br2, _ = B_reg(b2, D0)
                p["P1P2_diff"] = abs(Br - Br2)
                errB = max(errB, abs(Br - Br2))
            p.update({"Breg": [Br.real, Br.imag], "B_windows_spread": spread, "err_B_abs": errB,
                      "rel_dev": abs(MB - Br) / abs(Br)})
            if "tmin_err" in b1:
                p["tmin_dB"] = b1["tmin_err"]["dB_2560"]
            if "err_MB_abs" in p:
                p["limiting"] = "MB" if p["err_MB_abs"] > errB else "B"
        points.append(p)
    return {"model": MODEL, "mb_dirs": list(mb_dirs), "b_dirs": list(b_dirs), "points": points}


def write_csv(table, fn):
    cols = ["tag", "k1", "k2", "k3", "mu1", "mu2", "mu3", "shape_x", "MB_re", "MB_im", "Breg_re",
            "Breg_im", "rel_dev", "err_MB_abs", "err_B_abs", "limiting"]
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in table["points"]:
            Br = p.get("Breg", [None, None])
            w.writerow([p["tag"], *p["k"], *p["mus"], p["shape_x"], p["MB"][0], p["MB"][1], Br[0],
                        Br[1], p.get("rel_dev"), p.get("err_MB_abs"), p.get("err_B_abs"),
                        p.get("limiting")])


def summary(table):
    rd = [p["rel_dev"] for p in table["points"] if "rel_dev" in p]
    if not rd:
        return {}
    return {"n": len(rd), "median_rel_dev": float(np.median(rd)),
            "p90_rel_dev": float(np.percentile(rd, 90)), "max_rel_dev": float(np.max(rd))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", nargs="*", default=["records/mb", "outputs/mb_local"])
    ap.add_argument("--b", nargs="*", default=["records/bruteforce", "outputs/b_local"])
    ap.add_argument("--out", default="outputs/triangle_scan.json")
    ap.add_argument("--record", default="outputs/prior/triangle_scan_data.json")
    a = ap.parse_args()
    t = assemble(a.mb, a.b, a.record)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(t, open(a.out, "w"), indent=1)
    write_csv(t, os.path.splitext(a.out)[0] + ".csv")
    s = summary(t)
    if s:
        print(f"median {s['median_rel_dev']:.2e}, 90th percentile {s['p90_rel_dev']:.2e}, "
              f"worst {s['max_rel_dev']:.2e} ({s['n']} configurations)")


if __name__ == "__main__":
    main()
