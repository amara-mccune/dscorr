"""Stage-2 launcher and merger: ten round-robin chunks of the ratio list, each with the sentinel
0.55 appended if it does not own it; ten single-core workers; the merge asserts equal key sets
and Sigma per key, drops the sentinel column and writes cache_<problem>_rest.json.

  python launch_stage2.py --problem fig3c --rfile rlist_fig3.txt --dir stage2 [--merge-only]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--problem", default="fig3c")
ap.add_argument("--rfile", default=os.path.join(HERE, "rlist_fig3.txt"))
ap.add_argument("--dir", default="stage2")
ap.add_argument("--merge-only", action="store_true")
a = ap.parse_args()
rs = [s.strip() for s in open(a.rfile).read().strip().split(",")]
chunks = [rs[i::10] for i in range(10)]
for c in chunks:
    if "0.55" not in c:
        c.append("0.55")
    else:
        c.remove("0.55")
        c.append("0.55")
os.makedirs(a.dir, exist_ok=True)
outs = [os.path.join(a.dir, f"chunk{i}.json") for i in range(10)]
if not a.merge_only:
    env = dict(os.environ, PYTHONPATH=HERE, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    ps = [subprocess.Popen([sys.executable, os.path.join(HERE, "stage2_worker.py"), "--problem", a.problem,
                            "--r", ",".join(c), "--out", o], env=env)
          for c, o in zip(chunks, outs) if not os.path.exists(o)]
    for p in ps:
        assert p.wait() == 0
ds = [json.load(open(o)) for o in outs]
fam = {}
for s in ("DIK", "XX"):
    keys = set(ds[0]["fam"][s])
    for d in ds[1:]:
        assert set(d["fam"][s]) == keys, "stage-2 chunks hold different key sets"
    out = {}
    for k in keys:
        sig = ds[0]["fam"][s][k]["Sig"]
        col = {}
        for d in ds:
            e = d["fam"][s][k]
            assert e["Sig"] == sig, "Sigma differs between chunks"
            for r, v in zip(d["rs"], e["vals"]):
                col.setdefault(r, v)
        out[k] = {"Sig": sig, "vals": [col[r] for r in rs]}
    fam[s] = out
json.dump({"prob": a.problem, "rs": rs, "fam": fam}, open(f"cache_{a.problem}_rest.json", "w"))
