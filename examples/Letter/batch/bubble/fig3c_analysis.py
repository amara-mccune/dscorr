"""Analysis of the bubble.

  PYTHONPATH=. python fig3c_analysis.py --problem fig3c --cache cache_fig3c.json \
      [--results results_fig3c.jsonl] [--dense data/fig3_dense_data.json]

Assembles, converts every ratio by the scheme chain, attaches the error bound, runs the truncation
scan (g_max = 0, 2, ..., 50; tol 1e-6 and 1e-8), compares with the shipped spectral and transport
values when the reference file is present, and writes fig3c_barnes_data.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402

from pipeline import load_cache, n_graded, truncation_scan  # noqa: E402
from terms import Families, Problem, assemble, scheme_chain, signal_coeff  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="fig3c")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--dense", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "data", "fig3_dense_data.json"))
    a = ap.parse_args(argv)
    mp.mp.dps = 30
    d, fam = load_cache(a.cache)
    rs = d["rs"]
    F = Families(Problem(a.problem), dps=30)
    rows = assemble(fam, F, rs)
    bounds = {}
    if a.results and os.path.exists(a.results):
        for line in open(a.results):
            e = json.loads(line)
            if e["quantity"] == "S" and e["scheme"] == "house":
                bounds[str(e["r"])] = float(e["bound"])
    ref, trans = {}, {}
    if os.path.exists(a.dense):
        pc = json.load(open(a.dense))["panel_c"]
        ref = {round(x["r"], 10): x["S_spec"] for x in pc["ours_dense"]}
        trans = {round(x["r"], 10): x["S_spec_transport"] for x in pc["transport_points"]}
    full = [row["S_house"] for row in rows]
    scan = truncation_scan(fam, F, rs, full)
    out = []
    for i, row in enumerate(rows):
        S5, Ss, RS = scheme_chain(row["S_house"], row["r"])
        r = float(row["r"])
        o = {"r": r, "S_barnes_house": float(row["S_house"]), "S_barnes_five": float(S5),
             "S_barnes_spec": float(Ss), "S_barnes_spec_str": mp.nstr(Ss, 20),
             "err_bound_house_abs": bounds.get(str(r)), "R_S": float(RS),
             "conv_grading_1e6": scan["1e-6"][i][0], "nterms_1e6": scan["1e-6"][i][1],
             "conv_grading_1e8": scan["1e-8"][i][0], "nterms_1e8": scan["1e-8"][i][1]}
        if round(r, 10) in ref:
            o["S_spectral"] = ref[round(r, 10)]
            o["rel_barnes_vs_spectral"] = float(Ss) / ref[round(r, 10)] - 1
        if round(r, 10) in trans:
            o["S_transport"] = trans[round(r, 10)]
            o["rel_barnes_vs_transport"] = float(Ss) / trans[round(r, 10)] - 1
            o["transport_point"] = True
        out.append(o)
    c = signal_coeff(fam, F, rs)
    json.dump({"problem": a.problem, "mu_R": float(F.prob.muR), "nutilde": float(F.prob.mu),
               "observable": "S(r) = -8 k1 k2 k3^4 B_XZ, k3 channel, legs (4,2)",
               "scheme_chain": "house -> five_minimal -> spectral", "route": "residue families",
               "n_residue_terms_full": n_graded(fam), "signal_amplitude": float(abs(c)),
               "signal_phase": float(mp.arg(c)), "rows": out},
              open("fig3c_barnes_data.json", "w"), indent=1)


if __name__ == "__main__":
    main()
