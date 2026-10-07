"""Mellin-Barnes driver of the triangle scan: mb_point.py <tasks.json> <outdir>.

Task file {"gp_h": 0.02, "points": [{"tag", "mus", "k", "do_A", "A_hs", "hs_list", "taumax",
"Y", "hy"}, ...]}.  Writes <tag>_A.json and <tag>_R<hs>.json per point; existing files are
skipped.  Production: A on the lattice h_s = h_v = 0.075 (about 1.3 core-hours per point on one workstation core,
6-20 core-hours at 16 linear-algebra workers in the production run); R at h_s = 0.05 and 0.04
(about 1 and 2 core-hours).
"""
import json
import os
import sys

from mb_triangle import TriPole, routeA_point


def main(tasks, outdir):
    cfg = json.load(open(tasks))
    gp_h = cfg.get("gp_h", 0.02)
    os.makedirs(outdir, exist_ok=True)
    for p in cfg["points"]:
        tag = p["tag"]
        mus, k = tuple(p["mus"]), tuple(p["k"])
        if p.get("do_A", True):
            fn = os.path.join(outdir, f"{tag}_A.json")
            if not os.path.exists(fn):
                rec = routeA_point(mus, k, hs=p.get("A_hs", 0.075))
                rec["tag"] = tag
                json.dump(rec, open(fn, "w"))
        for hs in p.get("hs_list", [0.05, 0.04]):
            fn = os.path.join(outdir, f"{tag}_R{hs:g}.json")
            if os.path.exists(fn):
                continue
            tp = TriPole(mus, k, sigma=-0.125, cperp=0.5, theta=0.25, taumax=p.get("taumax", 40.0),
                         hs=hs, Y=p.get("Y", 6.0), hy=p.get("hy", 0.15))
            rec = tp.R(gp_h=gp_h, levels=3)
            rec["tag"] = tag
            json.dump(rec, open(fn, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
