"""Stage-1 launcher: runs every task of the plan as a single-core process (longest first,
existing pieces skipped, failed tasks re-run once serially).  The run of record allowed 122
simultaneous processes; about 1e2 core-hours in total.

  python launch_stage1.py --problem fig3c --rfile rlist_fig3.txt --pieces pieces --procs 122
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from terms import plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problem", default="fig3c")
ap.add_argument("--rfile", default=os.path.join(HERE, "rlist_fig3.txt"))
ap.add_argument("--pieces", default="pieces")
ap.add_argument("--procs", type=int, default=122)
a = ap.parse_args()
rs = open(a.rfile).read().strip()
rmax = max(float(r) for r in rs.split(","))
os.makedirs(a.pieces, exist_ok=True)
tasks = sorted(plan(rmax), key=lambda t: -t[2])
todo = []
for fam, m, n in tasks:
    if fam == "V" and m % 2 == 0:
        continue
    name = fam if m is None else f"{fam}:{m}"
    out = os.path.join(a.pieces, name.replace(":", "_") + ".json")
    if not os.path.exists(out):
        todo.append((name, out))
env = dict(os.environ, PYTHONPATH=HERE, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
running, failed = [], []
for name, out in todo:
    while len(running) >= a.procs:
        p, nm, o = running.pop(0)
        if p.wait() != 0:
            failed.append((nm, o))
    cmd = [sys.executable, os.path.join(HERE, "letter_dii_worker.py"), "--problem", a.problem,
           "--r", rs, "--task", name, "--out", out]
    running.append((subprocess.Popen(cmd, env=env), name, out))
for p, nm, o in running:
    if p.wait() != 0:
        failed.append((nm, o))
for nm, o in failed:
    subprocess.run([sys.executable, os.path.join(HERE, "letter_dii_worker.py"), "--problem", a.problem,
                    "--r", rs, "--task", nm, "--out", o], env=env, check=True)
