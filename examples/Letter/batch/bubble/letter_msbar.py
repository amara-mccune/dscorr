"""Monolithic bubble driver: all sections for a list of ratios in one process.

  PYTHONPATH=. python letter_msbar.py --problem fig3c --r 0.1 --dps 20 --Nmax 24 --Mmax 24 \
      --mpmax 80 --h 0.05 --hU 0.025 --T 12 --tailrule --tailscale 1.3 --Ktail 3 --npconv

(parameters of record; the command-line defaults are NOT the parameters of record).
Prints one JSON line per ratio and scheme, one line with the signal coefficient, and one line
with the six largest last-ratio moduli.  About 1 to 2 core-hours per ratio.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402

from pipeline import (Progress, dii_section, dik_section, load_cache, save_cache,  # noqa: E402
                      xx_section)
from terms import Families, Problem, assemble, scheme_chain, signal_coeff  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="fig3c")
    ap.add_argument("--r", required=True)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--Nmax", type=int, default=12)
    ap.add_argument("--Mmax", type=int, default=12)
    ap.add_argument("--mpmax", type=int, default=30)
    ap.add_argument("--h", default="0.05")
    ap.add_argument("--hU", default="0.0125")
    ap.add_argument("--T", type=int, default=18)
    ap.add_argument("--N0", type=int, default=10)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--Nn", type=int, default=0)
    ap.add_argument("--tailrule", action="store_true")
    ap.add_argument("--tailscale", type=float, default=1.0)
    ap.add_argument("--Ktail", type=int, default=3)
    ap.add_argument("--npconv", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--diicache", default=None)
    a = ap.parse_args(argv)
    rs = [s.strip() for s in a.r.split(",")]
    prob = Problem(a.problem)
    F = Families(prob, dps=a.dps, Nmax=a.Nmax, Mmax=a.Mmax, h=a.h, hU=a.hU, T=a.T, npconv=a.npconv)
    cfn = f"cache_{a.tag}.json"
    prog = Progress(f"progress_{a.tag}.jsonl")
    if a.reuse and os.path.exists(cfn):
        d, fam = load_cache(cfn)
        assert d["prob"] == a.problem and d["rs"] == rs, "cache of another problem or ratio list"
        last = d.get("last_mp", {})
    else:
        rmax = max(float(r) for r in rs)
        if a.diicache:
            d, f0 = load_cache(a.diicache)
            assert d["prob"] == a.problem and d["rs"] == rs, "D_II cache of another problem or ratio list"
            dii, lm = f0["DII"], None
        else:
            dii, lm = dii_section(F, rs, a.mpmax, prog, stop=True, tailrule=a.tailrule, rmax=rmax,
                                  tailscale=a.tailscale, Ktail=a.Ktail, N0=a.N0, K=a.K, Nn=a.Nn)
        dik, lk = dik_section(F, rs, a.mpmax, prog)
        xx, lx = xx_section(F, rs, a.mpmax, prog)
        fam = {"DII": dii, "DIK": dik, "XX": xx}
        last = {"DII": lm, "DIK": lk, "XX": lx}
        save_cache(cfn, a.problem, rs, fam, {"last_mp": last})
    mp.mp.dps = 30
    rows = assemble(fam, F, rs)
    for row in rows:
        S5, Ss, RS = scheme_chain(row["S_house"], row["r"])
        for scheme, val in (("house", row["S_house"]), ("five_minimal", S5)):
            print(json.dumps({"r": row["r"], "scheme": scheme, "S": mp.nstr(val, 20),
                              "S_spec": mp.nstr(Ss, 20), "R_S": mp.nstr(RS, 20),
                              "DII": [mp.nstr(mp.re(row["DII"]), 15), mp.nstr(mp.im(row["DII"]), 15)],
                              "DIK": [mp.nstr(mp.re(row["DIK"]), 15), mp.nstr(mp.im(row["DIK"]), 15)],
                              "XX": [mp.nstr(mp.re(row["XX"]), 15), mp.nstr(mp.im(row["XX"]), 15)],
                              "G_pole_rel": mp.nstr(row["G_pole_rel"], 5),
                              "err_richardson": mp.nstr(row["err_richardson"], 5)}))
    c = signal_coeff(fam, F, rs)
    print(json.dumps({"signal_coeff": [mp.nstr(mp.re(c), 20), mp.nstr(mp.im(c), 20)],
                      "amplitude": mp.nstr(abs(c), 20), "phase": mp.nstr(mp.arg(c), 20),
                      "last_mp": last}))
    big = sorted(((abs(v[-1]), f"{s}:{k}") for s in fam for k, (_, v) in fam[s].items()
                  if not k.startswith(("E", "P"))), reverse=True)[:6]
    print(json.dumps({"largest": [[k, mp.nstr(b, 5)] for b, k in big]}))


if __name__ == "__main__":
    main()
