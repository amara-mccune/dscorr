"""b_worker.py <task.json> <out.json>: brute-force main run of the triangle.

Task keys: mus, k, setting ("P1" default, "P2" or a dictionary), nproc (None = serial),
tgrid (tmin, tmax, npan, ngl).  About 5 core-hours per configuration (production)."""
import json
import sys

from bf_run import run_main

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    rec = run_main(task, rv=1.0, store_from=300.0)
    json.dump(rec, open(sys.argv[2], "w"))
