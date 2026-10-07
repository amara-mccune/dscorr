"""run_mb.py <tasks.json> <outdir> [--tag TAG] [--blas N].

Task file {"runs": [{tag, mus, k, rho, nblk, [rv, sig_u, hst, tmax_u, hs, hv, taumax, Vhalf]}]}.
Writes <outdir>/<tag>_rho<rho>.json; existing records are skipped.  About 5-10 minutes per entry
on one workstation core (0.3 core-hours at 16 linear-algebra cores in the production run), ~1 GB memory.
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("tasks")
ap.add_argument("outdir")
ap.add_argument("--tag", default=None)
ap.add_argument("--blas", type=int, default=None)
args = ap.parse_args()
nb = str(args.blas) if args.blas else os.environ.get("OPENBLAS_NUM_THREADS", "16")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mb_pidot import mb_loop_point  # noqa: E402

os.makedirs(args.outdir, exist_ok=True)
for e in json.load(open(args.tasks))["runs"]:
    if args.tag and e["tag"] != args.tag:
        continue
    fn = os.path.join(args.outdir, f"{e['tag']}_rho{e['rho']:g}.json")
    if os.path.exists(fn):
        continue
    kw = {k: e[k] for k in ("rv", "sig_u", "hst", "tmax_u", "hs", "hv", "taumax", "Vhalf") if k in e}
    rec = mb_loop_point(e["mus"], e["k"], rho=e["rho"], nblk=e["nblk"], **kw)
    rec["tag"] = e["tag"]
    json.dump(rec, open(fn, "w"))
