"""Series of the Dirac-pair bubble for the direct check: S(n; s = +1) and S(n; s = -1) of dirac_pair.py.

    python dirac_bubble_series.py m kappa nmax out.json      (run from examples/Loops, or set LOOPS to that folder)

The file is rewritten after every order, so a partial run can be used.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("LOOPS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import mpmath as mp  # noqa: E402

from dirac_pair import DiracPair  # noqa: E402

mp.mp.dps = int(os.environ.get("DPS", 30))


def main(m, kappa, nmax, out):
    P = DiracPair(mp.mpf(m), mp.mpf(kappa))
    data = dict(m=m, kappa=kappa, r1=4, r2=2, dps=mp.mp.dps, nu=[0.0, float(P.D.mubar)], mubar=mp.nstr(P.D.mubar, 25), S_plus=[], S_minus=[], cpu=[])
    if os.path.exists(out):
        old = json.load(open(out))
        if old.get("m") == m and old.get("kappa") == kappa and old.get("dps") == mp.mp.dps:
            data = old
    for n in range(len(data["S_plus"]), nmax + 1):
        t0 = time.process_time()
        sp, sm = P.S(n, s=1), P.S(n, s=-1)
        data["S_plus"].append([mp.nstr(mp.re(sp), 25), mp.nstr(mp.im(sp), 25)])
        data["S_minus"].append([mp.nstr(mp.re(sm), 25), mp.nstr(mp.im(sm), 25)])
        data["cpu"].append(round(time.process_time() - t0, 1))
        with open(out, "w") as f:
            json.dump(data, f, indent=1)
        print(n, mp.nstr(sp, 12), mp.nstr(sm, 12), data["cpu"][-1], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
