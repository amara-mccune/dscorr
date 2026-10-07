"""assemble_pidot.py --mb DIR --bf DIR --out FILE [--points FILE] .

Extrapolates every MB record (window fit, step 9), reads the brute-force records, and writes
one JSON file with meta, points, summary, MB_detail and BF_detail."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mb_pidot import extrapolate_record  # noqa: E402

QUOTED = -0.6
SECOND = -0.55


def c2(z):
    return [complex(z).real, complex(z).imag]


def assemble(mb_dir, bf_dir, points_file=None):
    here = os.path.dirname(os.path.abspath(__file__))
    pts = json.load(open(points_file or os.path.join(here, "points_pidot.json")))
    MBd, BFd = {}, {}
    order = {}
    for fn in sorted(os.listdir(mb_dir)) if os.path.isdir(mb_dir) else []:
        if not fn.endswith(".json") or "_rho" not in fn:
            continue
        rec = json.load(open(os.path.join(mb_dir, fn)))
        ex = extrapolate_record(rec)
        tag = rec["tag"]
        key = f"{rec['rho']:g}"
        MBd.setdefault(tag, {})[key] = {
            "S_inf": c2(ex["S_inf"]), "extrap_err": ex["extrap_err"], "N_last": ex["N_last"],
            "S_last": c2(ex["S_last"]), "tail_last": ex["tail_last"], "fit_resid": ex["fit_resid"],
            "p": ex["p"], "settings": {k: rec[k] for k in ("rho", "sig", "sig_u", "hst", "hs", "taumax",
                                                            "Vhalf", "hv", "Dband", "Ns", "Nv")},
            "w3_pole_margin": rec["w3_pole_margin"], "band_edge_ratio": rec["band_edge_ratio"],
            "V_ladder": rec.get("V_ladder"), "S_2hv": rec.get("S_2hv")}
        order.setdefault(tag, []).append(key)
    for fn in sorted(os.listdir(bf_dir)) if os.path.isdir(bf_dir) else []:
        if fn.startswith("bf_") and fn.endswith(".json"):
            rec = json.load(open(os.path.join(bf_dir, fn)))
            BFd[fn[3:-5]] = {k: rec[k] for k in ("B_inf", "tail_info", "B_2560", "n_radial", "n_evals",
                                                 "setting", "tmin", "tmax") if k in rec}
    points = []
    for p in pts:
        tag = p["tag"]
        row = {"tag": tag, "mus": p["mus"], "k": p["k"]}
        m = MBd.get(tag, {})
        used = f"{QUOTED:g}" if f"{QUOTED:g}" in m else (f"{SECOND:g}" if f"{SECOND:g}" in m else None)
        if used:
            row["MB"] = m[used]["S_inf"]
            row["MB_placement_used"] = float(used)
            row["err_MB_abs"] = m[used]["extrap_err"]
        if len(order.get(tag, [])) >= 2:
            a, b = (complex(*m[order[tag][0]]["S_inf"]), complex(*m[order[tag][1]]["S_inf"]))
            row["MB_placement_diff"] = abs(a - b) / abs(a)
        if tag in BFd:
            row["B"] = BFd[tag]["B_inf"]
            if "MB" in row:
                row["rel_diff"] = abs(complex(*row["MB"]) - complex(*row["B"])) / abs(complex(*row["B"]))
        points.append(row)
    rd = [r["rel_diff"] for r in points if "rel_diff" in r]
    summ = {"n_points": len(points), "n_both_routes": len(rd)}
    if rd:
        summ.update({"median_rel_diff": float(np.median(rd)), "max_rel_diff": float(np.max(rd))})
    meta = {"MB_side": {"lattice": "h_s = h_v = 0.1, tau_max = 100, V = 4, band |dtau| <= 14",
                        "window_tail": "fit S(N) = S_inf + N^-p (c0 + c1/N + c2/N^2), p = 5 + 6 rho, "
                                       "20 <= N <= 86, last 10 windows"},
            "BF_side": {"layout": "P1", "tail": "c3 r^-3 + c4 r^-4 + c5 r^-5 on r >= 300"}}
    return {"meta": meta, "points": points, "summary": summ, "MB_detail": MBd, "BF_detail": BFd}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True)
    ap.add_argument("--bf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default=None)
    a = ap.parse_args()
    json.dump(assemble(a.mb, a.bf, a.points), open(a.out, "w"), indent=1)
