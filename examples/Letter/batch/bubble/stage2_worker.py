"""Stage-2 worker: sections D_IK and XX for one chunk of ratios.

  PYTHONPATH=. python stage2_worker.py --problem fig3c --r <chunk list ending in 0.55> --out F [--xx-only]

Parameters hard-wired: Nmax = Mmax = 24, m'max = 80, hU = 0.025, T = 12, dps 20.  --xx-only is
the second worker (XX alone, for restart recovery).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Progress, dik_section, enc, xx_section  # noqa: E402
from terms import Families, Problem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problem", required=True)
ap.add_argument("--r", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--xx-only", action="store_true")
a = ap.parse_args()
rs = [s.strip() for s in a.r.split(",")]
assert rs[-1] == "0.55", "the last ratio of a stage-2 chunk must be the sentinel 0.55"
F = Families(Problem(a.problem), dps=20, Nmax=24, Mmax=24, h="0.05", hU="0.025", T=12, npconv=True)
prog = Progress(a.out + ".progress.jsonl")
fam = {}
if not a.xx_only:
    fam["DIK"], _ = dik_section(F, rs, 80, prog)
fam["XX"], _ = xx_section(F, rs, 80, prog)
json.dump({"prob": a.problem, "rs": rs,
           "fam": {s: {k: enc(Sig, v) for k, (Sig, v) in c.items()} for s, c in fam.items()}},
          open(a.out, "w"))
