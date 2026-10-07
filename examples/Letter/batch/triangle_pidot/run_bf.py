"""run_bf.py <task.json> <out.json> [--nproc N]: brute force of the pi-dot triangle.
Reuses the brute-force modules of ../triangle (rv = 2).  About 2.1-2.9
core-hours per configuration in layout P1."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "triangle"))

from bf_run import run_main, tail_pidot  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--nproc", type=int, default=None)
    a = ap.parse_args()
    task = json.load(open(a.task))
    task.setdefault("nproc", 16)
    if a.nproc:
        task["nproc"] = a.nproc
    rec = run_main(task, rv=float(task.get("rv", 2.0)), store_from=100.0)
    rec["rv"] = float(task.get("rv", 2.0))
    rec = tail_pidot(rec)
    json.dump(rec, open(a.out, "w"))
