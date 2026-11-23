"""Coefficients of the planar box (class PP, unit weight, planar attachment only) of the collapsed series at equal
external momenta, for the direct check of the box: pair, single-frequency (singleA + singleB) and non-oscillating
families, per harmonic n. Run from examples/Loops:  python direct/box_series_coefs.py MU KAPPA NMAX OUT.json [nonosc]
(geometry and series through N = 4: 5 min, with the non-oscillating families 14 min; through N = 6: 18 min and 15 min)."""
import json
import sys
import time

import mpmath as mp

sys.path.insert(0, ".")
import parity_box_series as P  # noqa: E402

mu, ka, nmax, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
do_nonosc = len(sys.argv) > 5
mp.mp.dps = 25
t0 = time.time()
P.geometry(nmax)
print(f"geometry through N = {nmax}: {time.time() - t0:.0f} s", flush=True)
zero = [mp.mpc(0)] * (nmax + 1)


def ser(v):
    return [[mp.nstr(x.real, 20), mp.nstr(x.imag, 20)] for x in v]


t0 = time.time()
C = P.Collapsed(mu, ka, nmax, families=("pair", "singleA", "singleB"), diagrams=("PP",))
pair = C.equal_k("pair", "PP", crossed=False)
a, b = C.equal_k("singleA", "PP", crossed=False), C.equal_k("singleB", "PP", crossed=False)
single = {n: [x + y for x, y in zip(a.get(n, zero), b.get(n, zero))] for n in set(a) | set(b)}
res = dict(mu=mu, kappa=ka, nmax=nmax, note="planar box, unit weight, equal external momenta; coefficient lists over N of r^N",
           pair={str(n): ser(v) for n, v in pair.items()}, single={str(n): ser(v) for n, v in single.items()})
print(f"oscillating families: {time.time() - t0:.0f} s", flush=True)
json.dump(res, open(out, "w"), indent=1)
if do_nonosc:
    t0 = time.time()
    delta = mp.mpf("1e-8")
    with mp.workdps(mp.mp.dps + 10):
        tot = {}
        for sgn in (1, -1):
            Cn = P.Collapsed(mu, ka, nmax, P.NONOSC, diagrams=("PP",), eps=sgn * delta)
            acc = {}
            for fam in P.NONOSC:
                for n, v in Cn.equal_k(fam, "PP", crossed=False).items():
                    acc[n] = [x + y for x, y in zip(acc.get(n, zero), v)]
            tot[sgn] = acc
        F, L = {}, {}
        for n in sorted(set(tot[1]) | set(tot[-1])):
            p, m = tot[1].get(n, zero), tot[-1].get(n, zero)
            F[str(n)] = ser([(x + y) / 2 for x, y in zip(p, m)])
            L[str(n)] = ser([delta * (x - y) for x, y in zip(p, m)])
    res["nonosc_F"], res["nonosc_L"] = F, L
    res["nonosc_note"] = "sum over the non-oscillating families: s^3 k^-8 sum_N r^N sum_n e^(i n beta) [F_n + L_n ln r]"
    print(f"non-oscillating families: {time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(out, "w"), indent=1)
