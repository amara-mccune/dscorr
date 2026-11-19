# Direct integration of the fermion triangle at fixed shape

These scripts check the series of `fermion_triangle_shape.py` against the triangle itself. Nothing is expanded. One orientation of the loop is integrated in the time domain with the exact Schwinger-Keldysh functions of the three lines and all time orderings, on the fully rotated contour. The loop momentum is cut at |q| + |q - k3| < Pc (in units of k1), which changes the result only by terms analytic in k3. A fit then separates the pair-frequency signal from that analytic background.

| File | Content |
|---|---|
| `direct_fermi2.py` | The integration. `direct_fermi2.py m kappa c out.jsonl r1 r2 ...`, with `c = iso` for k1 = k2 and `c = d-0.5` for fixed delta = (k1 - k2)/k3 = -0.5. Settings come from the environment (list in the docstring). |
| `direct_fermi.py` | The first version (one loop grid, cut at 0.5). Kept because the earlier runs used it. `direct_fermi2.py` at `PC=0.5` returns the same numbers to all digits. |
| `fermi_shape_fit.py` | The fit in the symmetric variables R = 2 k3/(k1 + k2) and delta. It reports the residual after the full series, a free factor on the whole series, a free factor on the corrections, and the residual with the leading term only, for 12, 14, and 16 background parameters. |
| `fermi_direct_fit.py` | The earlier fit at fixed angle (series in k3/k1). |
| `data/` | The runs of Nov 19, 2026 and the series coefficients they are compared with. |

The mode tables and the loop grid come from three helper modules of the paper's numerical checks (`full.py`, `modes.py`, and `softmoment.py`, folder `ninepairs/` of `loops_numcheck_code.zip`). Put that folder next to these scripts, or set `NINEPAIRS` to its path. The scripts also need numpy and python-flint.

    cd examples/Loops
    python fermion_triangle_shape.py --point 1.3 1 --nmax 10 --save series.json
    NPHI=6 PC=1.8 python direct/direct_fermi2.py 1 1.3 iso run.jsonl 0.3 0.2 0.1 0.05 0.02 0.01
    python direct/fermi_shape_fit.py direct/data/series_1_1.3_n10.json direct/data/iso_pc1.8.jsonl

Things that matter in practice:

- **The cut and the largest r.** The background of the cut integral is singular where the cut region degenerates (r = Pc) and where the hard line goes soft on its boundary (1 + k2 = Pc). On k1 = k2 the second never happens, and Pc = 1.8 allows r up to 0.3. For collinear momenta with k2 = k1 - k3 it happens at r = 2 - Pc, so Pc = 1.0 is the best choice there. A background of 12 to 16 terms needs the nearest singularity at about four times the largest r or more.
- **Outer shells.** Between 0.5 and Pc the radial panels have edges fixed in units of k1. Their quadrature error is then analytic in r and is absorbed by the background fit (checked by raising their resolution, `OUTSCALE`).
- **Azimuth.** Six points over half of the circle away from collinear shapes (`NPHI=6`), one for collinear shapes. The imaginary part of the spin trace is dropped, because the triple products are odd under reflection through the plane of the three momenta. `FULLPHI=1` integrates the whole circle with the complex trace instead, as a test.
- **Large chemical potential.** The branch terms cancel by a factor of about 4e9 at kappa = 5. The time quadrature must then be converged to roundoff. Use `NCHEB=20` (degree of the time panels) and a finer loop grid (`GRID=fine LOOPJSON='{"rho_ratio":1.6,"nrho":8,"nth":10,"P_ratio":1.4,"nP":8,"nq":12}'`). The standard settings (degree 12) are wrong by a percent there, and the error changes each time the number of time panels changes, which looks like noise in a fit.
- **What a fit can see.** A background with 12 to 16 free terms imitates a slow oscillation well. At kappa = 1.3 (one period per 1.9 e-folds of r) the fits fix the first two corrections and the sum of the higher ones. At kappa = 5 the frequency is three times higher, and four orders are fixed one by one.
