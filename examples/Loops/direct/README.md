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

# Direct integration of the Dirac-pair bubble

`direct_dirac_bubble.py` checks the squeezed series of `dirac_pair.py` (Secs. 8.2 and 8.3) against the bubble itself, at the vertices of Han, Lei, Qin and Yin (r1 = 4 at k12, r2 = 2 at k3). It integrates both vertex times and the loop momentum with the exact Schwinger-Keldysh functions of the scalar bilinear, for all four branch assignments. The loop momentum is cut at y1 + y2 < Pc (in units of k12). The bubble depends on the shape only through r = k3/k12, and it has no azimuth, so one point takes 10 to 40 s.

| File | Content |
|---|---|
| `direct_dirac_bubble.py` | The integration. `direct_dirac_bubble.py m kappa out.jsonl r1 r2 ...`. Each row holds I(r) and, separately, the time-ordered term T and the doubly cut term D. Settings come from the environment (list in the docstring). |
| `dirac_bubble_series.py` | The soft moments S(n) of `dirac_pair.py` on both pair branches, `dirac_bubble_series.py m kappa nmax out.json`. |
| `dirac_bubble_fit.py` | The fit of an analytic background (12, 14, or 16 parameters) together with the series. `--check` reruns the fits of record on the stored runs and compares them with `../dirac_bubble_direct_record.json`. |
| `data_dirac/` | The runs of Nov 20, 2026 at (m, kappa) = (1, 1), (2, 2), and (1, 0.5), and the series they are compared with. |

    cd examples/Loops
    python direct/dirac_bubble_series.py 1 1 12 series.json
    GRID=fine LOOPJSON='{"rho_ratio":2.0,"nrho":10,"nth":10,"P_ratio":1.5,"nP":10,"nq":12}' NCHEB=16 PC=1.5 \
        python direct/direct_dirac_bubble.py 1 1 run.jsonl 0.3 0.1 0.03 0.01 0.003 0.001
    python direct/dirac_bubble_fit.py series.json run.jsonl --wpow -1
    python direct/dirac_bubble_fit.py --check

Things that matter in practice:

- **Both pair branches enter the bispectrum.** The hard vertex is later than the soft one, so the time-ordered sector has the Wightman function of the cut one, and its hard time integral differs by the phase -e^(-i pi (4 + 2 nu + n)). Summed over the four sectors, the coefficient of r^(1 + 2 nu + n) is [1 - (-1)^n e^(-2 pi mubar)] M_D(n) + [1 - (-1)^n e^(2 pi mubar)] conj M_D^-(n), where M_D^-(n) = Gamma(4 - 2 nu + n) S(n; -) belongs to the branch nu -> -nu. At n = 0 this is the combination of the two branches in Eq. (5.41) of Han et al. (`han_check.py`). The second term is 0.10, 0.85, and 6.5 times the first at (1, 1), (2, 2), and (1, 0.5), and it alternates in sign with n. The ratios M_D(n)/M_D(0) of `dirac_pair.py --orders` are therefore the series of the + branch of the doubly cut term, not of the bispectrum. The stored runs exclude them (a free factor on their corrections is off from one by 0.34 to 1.2), and they confirm the doubly cut term and the time-ordered term separately (`--part D`, `--part T`).
- **Range.** A background of 12 to 16 terms imitates a slow oscillation over one or two periods. At (1, 1) the pair signal has one period per 2.2 e-folds of r, and on 0.01 <= r <= 0.3 a free factor on the corrections is fixed only to between 5e-3 and 7e-2. The runs therefore extend to r = 1e-4 (1e-6 at (1, 0.5)), and the fits weight the points by 1/r, so that every e-fold counts equally in the signal. At (2, 2) the range 0.01 to 0.3 is enough.
- **Cut.** The background of the cut integral is singular at r = +-Pc, and that of the bubble itself at r = -1 (zero total energy). The runs use Pc = 1.5. The difference between Pc = 1.5 and 1.0 is a polynomial in r, without logarithms, to 2e-13.
- **Grids.** With the loop grid above and time panels of degree 16 the values are converged to 1e-14 at (1, 1) and to 3e-12 at (2, 2) (`data_dirac/convergence_m2_k2.jsonl`). The standard grid of `direct_fermi2.py` is good to 2e-9 only, which is 5e-7 of the signal at r = 0.1.
- **What a fit can see.** At (1, 1) the fits fix the first correction to 7e-4 or better and the second to 2.5%, and with 16 terms also the third to 1%. At (2, 2) they fix four corrections. At (1, 0.5) (one period per 2.8 e-folds) the criteria need 14 terms or more.
