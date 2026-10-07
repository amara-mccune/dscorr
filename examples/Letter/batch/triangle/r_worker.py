"""r_worker.py <task.json> <out.json>: TriPole.R for one task {mus, k, params, gp_h} with the
two-level derivative (default gp_h 0.01); produced the convergence records."""
import json
import sys

from mb_triangle import TriPole

if __name__ == "__main__":
    t = json.load(open(sys.argv[1]))
    p = t.get("params", {})
    tp = TriPole(t["mus"], t["k"], **p)
    rec = tp.R(gp_h=t.get("gp_h", 0.01), levels=2)
    rec["task"] = t
    json.dump(rec, open(sys.argv[2], "w"))
