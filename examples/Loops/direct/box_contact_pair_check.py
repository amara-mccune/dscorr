"""Check of the line and vertex functions of direct_box.py on the contact-pair diagram: the hard pair (V1, V2) of the
box with a quartic vertex in place of the other side, evaluated with the reference `cycle_sum` on the loop grid and
azimuth of direct_vec.py (X3=2 PW3=2 NPHI=6 GRID=std PC=0.5), against a stored value of direct_vec.py.

Usage: box_contact_pair_check.py [r S]      (default: r = 0.0991227, S = 1.0261148647086105e-06 of data_cp/cp_1_1_pc0.5.jsonl)
On Nov 22, 2026: I = 8.281573157503e-05 against S/(r/8) = 8.281573158993e-05 (ratio - 1 = -1.8e-10), 50 s."""
import sys, time, json
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import direct_box as D
from softmoment import loop_points
r, Srec = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.0991227, 1.0261148647086105e-06)
part = int(sys.argv[3]) if len(sys.argv) > 3 else 1
B = D.Box(1.0, 1.0, Pc=0.5, M=12)
B.phis = np.pi * (np.arange(12) + 0.5) / 6
y1, y2, w, _ = loop_points(dict(rho_min=1e-7, rho_ratio=3.0, nrho=6, nth=6, P_ratio=1.8, nP=6, nq=8, Pmax=0.5 / r, ntail=1, nfit=1))
t, h, wint = B.t, B.h, B.wint
L = slice(2, 3)
tot, totc = 0j, 0j
t0 = time.time()
for i in range(0, y1.size, part):
    S = B.soft(r, y1[i], y2[i])
    T12, Tc12 = B.side(S, r / 2, 1.0, 12)
    Tq = D.vertex_tensor(t, 2.0, 2, S["y2"], S["e2"], S["U2"], False, S["y1"], S["e1"], S["U1"], False)
    A = [np.moveaxis(T12[0], 0, -2), np.moveaxis(T12[1], 0, -2), Tq]
    pref = w[i] * r ** 3 / (4 * S["y1"] * S["y2"])
    tot += pref * np.mean(D.cycle_sum(A, h, wint))
    totc += pref * np.mean(D.cycle_sum([Tc12, Tq[:, :, L, L]], h, wint))
I = 2 * (tot + totc).real * part
print(f"r = {r}: I = {I:.12e} (regular {2 * tot.real * part:.6e}, contact {2 * totc.real * part:.6e}), direct_vec.py S/(r/8) = {Srec / (r / 8):.12e}, "
      f"ratio - 1 = {I / (Srec / (r / 8)) - 1:.2e}, Im Z = {(tot + totc).imag * part:.1e} (cancels in I = Z + conj Z); {time.time() - t0:.0f} s, {y1.size // part} points")
