"""Merge the stage-1 pieces into the D_II cache.

  python merge_dii.py --problem fig3c --pieces pieces --out cache_fig3c_dii.json [--termdump F] [--mpmax 80]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import enc  # noqa: E402

import mpmath as mp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problem", required=True)
ap.add_argument("--pieces", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--termdump", default=None)
ap.add_argument("--mpmax", type=int, default=80)
a = ap.parse_args()
pieces = {}
for fn in sorted(os.listdir(a.pieces)):
    if fn.endswith(".json"):
        d = json.load(open(os.path.join(a.pieces, fn)))
        pieces[d["task"]] = d
need = ["T"] + [f"U:{m}" for m in range(a.mpmax + 1)] + [f"V:{m}" for m in range(1, a.mpmax + 1, 2)]
missing = [t for t in need if t not in pieces]
if missing:
    sys.exit("missing tasks: " + ", ".join(missing))
ref = pieces["T"]
for t, d in pieces.items():
    if d["rs"] != ref["rs"] or d["params"] != ref["params"] or d["prob"] != a.problem:
        sys.exit(f"task {t}: ratio list, parameters or problem differ")
ent = {}
dump = []
for t in need:
    ent.update(pieces[t]["entries"])
    if pieces[t].get("termdump"):
        dump += pieces[t]["termdump"]
zero = enc(None, [mp.mpc(0)] * len(ref["rs"]))
for m in range(0, a.mpmax + 1, 2):
    ent[f"V{m}"] = zero
    ent[f"EV{m}"] = zero
json.dump({"prob": a.problem, "rs": ref["rs"], "fam": {"DII": ent}, "params": ref["params"],
           "ntasks": len(need)}, open(a.out, "w"))
if a.termdump:
    json.dump(dump, open(a.termdump, "w"))
