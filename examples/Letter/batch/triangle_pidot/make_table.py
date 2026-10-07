"""make_table.py <assembled.json> --json <table.json> --csv <table.csv>.

Forms the table from the assembled file without numerical work."""
import argparse
import csv
import json
import math

import numpy as np


def c(z):
    return complex(z[0], z[1])


def make(asm):
    MBd, BFd = asm["MB_detail"], asm["BF_detail"]
    emethod = None
    diffs = [abs(c(BFd[t]["B_inf"]) - c(BFd[t + "_P2"]["B_inf"])) for t in BFd
             if t + "_P2" in BFd and not t.endswith("_P2")]
    if diffs:
        emethod = max(diffs)
    rows = []
    for p in asm["points"]:
        tag = p["tag"]
        k = p["k"]
        fam = "momentum" if (k[1], k[2]) == (1.0, 1.1) else "control"
        r = {"tag": tag, "family": fam, "mass_tag": tag.split("_")[-1],
             "plotted_as": "dashed line+square" if fam == "momentum" else "not drawn",
             "mus": p["mus"], "k": k, "shape_x": k[0] / (k[1] + k[2])}
        m = MBd.get(tag, {})
        for name, key in (("MB", "-0.6"), ("MBalt", "-0.55")):
            if key in m:
                d = m[key]
                rS = -3.0 * d["settings"]["rho"]
                r[name] = d["S_inf"]
                r[name + "_placement"] = float(key)
                r["err_" + name + "_abs"] = d["extrap_err"]
                r[name + "_alias_rel"] = math.exp(-2 * math.pi * (rS - 1.5) / d["settings"]["hv"])
                r[name + "_settings"] = dict(d["settings"], N_last=d["N_last"], p=d["p"])
        if "MB" in r:
            r["absMB"] = abs(c(r["MB"]))
        if tag in BFd:
            b = BFd[tag]
            r["B"] = b["B_inf"]
            r["absB"] = abs(c(b["B_inf"]))
            tail = c(b["tail_info"]["tail"])
            r["err_B_tail_abs"] = abs(tail)
            r["B_tail"] = b["tail_info"]["tail"]
            r["B_tail_exponent"] = b["tail_info"]["exponent_free"]
            if emethod is not None:
                r["err_B_abs"] = emethod + abs(tail)
            r["B_settings"] = {k2: b.get(k2) for k2 in ("setting", "tmin", "tmax", "n_radial", "n_evals")}
            if "MB" in r:
                r["rel_dev"] = abs(c(r["MB"]) - c(r["B"])) / abs(c(r["B"]))
        if "MB" in r and "MBalt" in r:
            r["placement_rel_diff"] = abs(c(r["MBalt"]) - c(r["MB"])) / abs(c(r["MB"]))
        rows.append(r)
    rd = [r["rel_dev"] for r in rows if "rel_dev" in r]
    summary = {"n_points": len(rows)}
    if rd:
        i = int(np.argmax(rd))
        summary.update({"median_rel_dev": float(np.median(rd)), "max_rel_dev": float(np.max(rd)),
                        "max_rel_dev_tag": [r["tag"] for r in rows if "rel_dev" in r][i]})
    pr = [r["placement_rel_diff"] for r in rows if "placement_rel_diff" in r]
    if pr:
        summary.update({"n_with_both_placements": len(pr), "placement_rel_diff_min": min(pr),
                        "placement_rel_diff_max": max(pr), "placement_rel_diff_mean": float(np.mean(pr))})
    meta = dict(asm["meta"])
    meta["MB_side"] = {"lattice": "per-run settings, see MB_settings",
                       "window_tail": "per-run window ladder, see MB_settings (N_last)"}
    return {"description": "Loop triangle with one pi-dot leg per vertex", "meta": meta,
            "points": rows, "summary": summary, "MB_detail": MBd, "BF_detail": BFd}


COLS = ["tag", "family", "mass_tag", "plotted_as", "k1", "k2", "k3", "mu1", "mu2", "mu3", "shape_x",
        "absMB", "absB", "MB_re", "MB_im", "B_re", "B_im", "rel_dev", "err_MB_abs", "err_B_abs",
        "err_B_tail_abs", "MB_placement", "MBalt_placement", "MBalt_re", "MBalt_im", "err_MBalt_abs",
        "placement_rel_diff", "MB_sig_u", "MB_hst", "MB_hs", "MB_hv", "MB_Vhalf", "MB_taumax",
        "MB_N_last", "MBalt_taumax", "MBalt_N_last", "B_setting", "B_tmin", "B_tmax", "B_n_radial"]


def g4(x):
    return "" if x is None else f"{x:.4g}"


def write_csv(t, fn):
    with open(fn, "w", newline="") as f:
        f.write("# loop triangle with pi-dot legs\n# columns as in the table of record\n")
        f.write(f"# summary: {json.dumps(t['summary'])}\n")
        w = csv.writer(f)
        w.writerow(COLS)
        for r in t["points"]:
            ms, ma = r.get("MB_settings", {}), r.get("MBalt_settings", {})
            bs = r.get("B_settings", {})
            MB, B, MBa = r.get("MB", [None, None]), r.get("B", [None, None]), r.get("MBalt", [None, None])
            w.writerow([r["tag"], r["family"], r["mass_tag"], r["plotted_as"], *r["k"], *r["mus"],
                        f"{r['shape_x']:.6f}", r.get("absMB", ""), r.get("absB", ""), MB[0], MB[1], B[0], B[1],
                        g4(r.get("rel_dev")), g4(r.get("err_MB_abs")), g4(r.get("err_B_abs")),
                        g4(r.get("err_B_tail_abs")), r.get("MB_placement"), r.get("MBalt_placement"),
                        MBa[0], MBa[1], g4(r.get("err_MBalt_abs")), g4(r.get("placement_rel_diff")),
                        ms.get("sig_u"), ms.get("hst"), ms.get("hs"), ms.get("hv"), ms.get("Vhalf"),
                        ms.get("taumax"), f"{ms['N_last']:.1f}" if ms else "", ma.get("taumax"),
                        f"{ma['N_last']:.1f}" if ma else "", bs.get("setting", {}).get("nr") and "P1",
                        bs.get("tmin"), g4(bs.get("tmax")) if bs.get("tmax") else "", bs.get("n_radial")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("assembled")
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    t = make(json.load(open(a.assembled)))
    json.dump(t, open(a.json, "w"), indent=1)
    write_csv(t, a.csv)
