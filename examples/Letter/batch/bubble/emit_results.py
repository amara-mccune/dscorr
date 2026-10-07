"""Results file of the bubble.

  PYTHONPATH=. python emit_results.py --problem fig3c --cache cache_fig3c.json [--calib err_calib.json]

Writes results_<problem>.jsonl: S in the house and five-minimal schemes (25 digits) with the error
fields, and R_S, mu_slope (= 2 R_S), amplitude, phase, c_ct (= R_S(0.30)) for the house scheme.
The cache must hold the three sections DII, DIK, XX (the joining of cache_<p>_dii.json and
cache_<p>_rest.json has no fixed layout; use join_caches()).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402

from pipeline import errors, load_cache  # noqa: E402
from terms import Families, Problem, assemble, scheme_chain, signal_coeff  # noqa: E402


def join_caches(dii_fn, rest_fn, out_fn):
    """Union of the 'fam' dictionaries under common 'prob' and 'rs'."""
    a, b = json.load(open(dii_fn)), json.load(open(rest_fn))
    assert a["prob"] == b["prob"] and a["rs"] == b["rs"]
    fam = dict(a["fam"])
    fam.update(b["fam"])
    json.dump({"prob": a["prob"], "rs": a["rs"], "fam": fam}, open(out_fn, "w"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="fig3c")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", default="err_calib.json")
    a = ap.parse_args(argv)
    mp.mp.dps = 30
    d, fam = load_cache(a.cache)
    rs = d["rs"]
    F = Families(Problem(a.problem), dps=30)
    calib = None
    if os.path.exists(a.calib):
        c = json.load(open(a.calib))
        calib = {"halving": mp.mpf(str(c["halving"])), "order": mp.mpf(str(c["order"]))}
    rows = assemble(fam, F, rs)
    out = open(f"results_{a.problem}.jsonl", "w")
    for i, row in enumerate(rows):
        S5, Ss, RS = scheme_chain(row["S_house"], row["r"])
        er = errors(fam, row, i, calib)
        for scheme, val in (("house", row["S_house"]), ("five_minimal", S5)):
            out.write(json.dumps({"problem": a.problem, "quantity": "S", "r": float(row["r"]),
                                  "value": mp.nstr(val, 25), "err": mp.nstr(er["err"], 5),
                                  "scheme": scheme, "G_pole_rel": mp.nstr(row["G_pole_rel"], 5),
                                  "bound": mp.nstr(er["bound"], 5),
                                  "err_empirical": mp.nstr(er["err_empirical"], 5),
                                  "err_basis": er["err_basis"],
                                  "err_note": "bound = richardson tail + truncation proxy"}) + "\n")
        for q, v in (("R_S", RS), ("mu_slope", 2 * RS)):
            out.write(json.dumps({"problem": a.problem, "quantity": q, "r": float(row["r"]),
                                  "value": mp.nstr(v, 25), "err": "0", "scheme": "house"}) + "\n")
    c = signal_coeff(fam, F, rs)
    _, _, RS3 = scheme_chain(0, "0.30")
    for q, v in (("amplitude", abs(c)), ("phase", mp.arg(c)), ("c_ct", RS3)):
        out.write(json.dumps({"problem": a.problem, "quantity": q, "r": None, "value": mp.nstr(v, 25),
                              "err": "0", "scheme": "house"}) + "\n")


if __name__ == "__main__":
    main()
