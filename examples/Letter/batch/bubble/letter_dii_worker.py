"""Stage-1 worker of the bubble.

  PYTHONPATH=. python letter_dii_worker.py --problem fig3c --r <comma list> --task T|U:<m'>|V:<m'> --out FILE
  python letter_dii_worker.py --problem fig3c --r <comma list> --plan

The float64 convolution is always on.  Task T: towers T<N>, R0T<N>; task U:<m'>: U, EU, R0U, R0V,
PR0U, PR0V; task V:<m'> (odd m'): V, EV.  4 to 45 minutes per task (122 tasks for the run of record).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import dii_mprime, enc  # noqa: E402
from terms import Families, Problem, plan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--r", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dps", type=int, default=20)
    ap.add_argument("--Nmax", type=int, default=24)
    ap.add_argument("--h", default="0.05")
    ap.add_argument("--hU", default="0.025")
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--Ktail", type=int, default=3)
    ap.add_argument("--tailscale", type=float, default=1.3)
    ap.add_argument("--termdump", action="store_true")
    a = ap.parse_args(argv)
    rs = [s.strip() for s in a.r.split(",")]
    rmax = max(float(r) for r in rs)
    if a.plan:
        tot = 0
        for fam, m, n in plan(rmax, tailscale=a.tailscale):
            print(f"{fam}:{m}" if m is not None else fam, n)
            tot += n
        print("tasks", len(plan(rmax, tailscale=a.tailscale)), "terms", tot)
        return
    t0 = time.time()
    prob = Problem(a.problem)
    F = Families(prob, dps=a.dps, Nmax=a.Nmax, h=a.h, hU=a.hU, T=a.T, npconv=True)
    td = [] if a.termdump else None
    if a.task == "T":
        ent = {k: (Sig, v) for k, (Sig, v, _) in F.towers(rs).items()}
    else:
        fam, m = a.task.split(":")
        m = int(m)
        ent = dii_mprime(F, m, rs, tailrule=True, rmax=rmax, tailscale=a.tailscale, Ktail=a.Ktail,
                         termdump=td, do_U=(fam == "U"), do_V=(fam == "V"))
    out = {"prob": a.problem, "rs": rs, "task": a.task,
           "entries": {k: enc(Sig, v) for k, (Sig, v) in ent.items()},
           "termdump": td, "params": {"dps": a.dps, "Nmax": a.Nmax, "h": a.h, "hU": a.hU, "T": a.T,
                                      "Ktail": a.Ktail, "tailscale": a.tailscale, "npconv": True},
           "secs": time.time() - t0}
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
