# The helical-vector triangle beyond the leading squeezed term

Behind Sec. 8.3 of "Massive de Sitter correlators at one loop" (section, equation, and figure numbers are those of arXiv v1). New files only; nothing in the paper's scripts changes. `helical_triangle.py` is also the script behind two sets of numbers of the paper that had none here: the ratio of M_A to its late-time estimate at κ = μ (6.07, 16.70, 29.99; Sec. 8.2) and R_hard (Sec. 8.3 and Fig. 14).

| File | Content |
|---|---|
| `helical_triangle.py` | Leading-order checks against Secs. 8.2 and 8.3, the series of the pair family in r = k3/k on k1 = k2, the series at general shape in R and δ with the check of the symmetry k1 ↔ k2, and a template symmetric in the three legs. Compares with `helical_triangle_record.json` and exits nonzero if a value is not reproduced. |
| `ext/vectri.py` | The expansion: polynomial algebra for the polarization contractions, the hard line at energy Y from its folded moments, the energy of the second hard leg, the azimuthal reduction to monomials, and the assembly over branches and components. (The same file is in the fermion-triangle patch.) |
| `ext/shapevars.py` | Exchange identity and symmetric shape variables. (Also in the fermion-triangle patch.) |
| `helical_triangle_record.json` | Values of record (Nov 18–19, 2026). |
| `direct/direct_vec.py` | Direct time-domain integration of the triangle at fixed shape, with nothing expanded. |
| `direct/vec_direct_fit.py` | Fit of an analytic background together with the series, in r at fixed angle or in (R, δ). |
| `direct/combine_parts.py` | Sums the pieces of a value computed with `PART=i/n`. |
| `direct/data_vec/` | The direct values of Nov 19, 2026, and the series they are compared with. |
| `../../tests/test_helical_triangle.py` | Fast checks of the algebra, and slow checks of the leading order and of the exchange symmetry at μ = κ = 1. |
| `../../tests/test_helical_triangle_direct.py` | The fits of the stored direct values against the stored series (seconds). |

    cd examples/Loops
    python helical_triangle.py                          # mu = kappa = 1: leading order, the series through n = 2, general shape (about 10 min)
    python helical_triangle.py --leading                # every leading-order number of Secs. 8.2 and 8.3 (about 3 min per point, 10 points)
    python helical_triangle.py --point 2 2.5 --nmax 6 --general --json series.json
    python helical_triangle.py --all                    # benchmarks through n = 8, kappa = 0, heavy hard line (several hours)
    python helical_triangle.py --deep                   # (3, 3.75) through n = 12 and the equilateral point (about 2 hours)

Needs Python ≥ 3.10 and mpmath ≥ 1.3 (as the other scripts). Soft moments are cached in `cache/`.

## Conventions worth knowing

- **Unit steps.** S contains 2 Re Σ_n M_n r^(2 + 2iμ + n). The odd orders come from e_1 = −K/(1 + 2ν) of Eq. (C.6) and vanish at κ = 0.
- **Normalization.** `M_n` is at unit couplings (triangle: prefactor 1/8; bubble: 1/16, so that the bubble's M_0 is M_A). The triangle in the normalization of Wang and Xianyu is −m_h² e^(−2πΩ) M_n, with Ω = max(κ − μ, 0).
- **Isosceles family.** k̂1·k̂3 = −r/2, so the n = 0 coefficient is the monopole minus half the quadrupole. `leading()` gives the multipoles separately; R_hard is the monopole over M_A.
- **Contact term.** The time-ordered A_0 A_0 function of the hard line includes +(t²/m_h²) δ(t − t′) on the rotated contour (first relevant at n = 2).
- **Infrared regulator.** Monomials with y^−2 have a removable Γ(0) in one leg of `soft_J`; both powers are shifted by ±δ and averaged (`SoftTable`).
- **General shape.** `Triangle(general=True)` expands the energy of the second hard leg as well and gives M_n(c) with c = k̂1·k̂3. At fixed c the series in r converges only for k3 < k1 and its coefficients grow. In R = 2k3/(k1 + k2) and δ = (k1 − k2)/k3 the same series is 2 Re Σ_N D_N(δ) R^(2 + 2iμ + N), each D_N is an even polynomial in δ, and D_N(0) = M_N. The leading coefficient is D_0(δ) = monopole + quadrupole (3δ² − 1)/2.

## Direct integration (`direct/`)

`direct_vec.py` evaluates S(r, c) in the time domain, with the exact Schwinger–Keldysh functions of the three lines, the exact energy and projectors of the hard line, all orderings of the three vertices, the longitudinal and temporal components, and the contact term. The loop momentum is cut at |q| + |q − k3| < Pc, which changes S only by terms analytic in r. It needs numpy, python-flint, and `full.py`, `modes.py`, `softmoment.py` of `loops_numcheck_code.zip` (set `NUMCHECK` to that folder). With `c = iso` it reproduces the earlier `direct_triangle.py` of the k1 = k2 family digit for digit.

    NPHI=1 PC=0.5 python direct/direct_vec.py 1 1 -1 out.jsonl 0.003 0.01 0.03        # collinear: the azimuth is trivial
    NPHI=6 PC=0.5 PART=0/8 python direct/direct_vec.py 1 1 0.5 parts.jsonl 0.03       # one of eight pieces
    python direct/combine_parts.py out.jsonl parts.jsonl
    python direct/vec_direct_fit.py direct/data_vec/series_1_1_n6.json direct/data_vec/c-1_pc0.5.jsonl --nbg=5
    python direct/vec_direct_fit.py direct/data_vec/series_1_1_n6.json direct/data_vec/c-1_pc1.0.jsonl --sym --nbg=6

What the stored runs show (μ = κ = 1 unless stated; backgrounds of 12 and 14 parameters; `a` is the fitted factor on a coefficient of the series, which should be one):

| Run | Points | \|a − 1\| for the first three coefficients | One factor on all corrections |
|---|---|---|---|
| collinear, c = −1, Pc = 0.5, r ≤ 0.08 | 18 | 2e-6, 1e-4, 2e-3 (12) and 3e-7, 7e-5, 4e-3 (14) | 1e-4 |
| collinear, c = +1, Pc = 0.5, r ≤ 0.08 | 18 | 2e-5, 4e-4, 5e-3 and 8e-7, 3e-5, 2e-3 | 3e-4 and 4e-5 |
| c = +0.5, Pc = 0.5, NPHI = 6, r ≤ 0.08 | 18 | 2e-5, 7e-4, 1e-2 and 2e-6, 1e-4, 4e-3 | 6e-4 and 1e-4 |
| collinear, c = −1, Pc = 1.0, r ≤ 0.167 (in R, δ) | 36 | 5e-5, 1e-3, 3e-2 and 4e-6, 2e-4, 9e-3 | 1e-3 and 2e-4 |
| (μ, κ) = (2, 2.5), collinear, c = −1, Pc = 0.5 | 18 | 2e-6, 5e-5, 9e-4 and 6e-7, 2e-5, 6e-4 | 5e-5 and 2e-5 |
| (μ, κ) = (2, 2.5), c = +0.5, Pc = 0.5, NPHI = 6 | 18 | 4e-6, 8e-5, 1.5e-3 and 7e-7, 2e-5, 6e-4 | 8e-5 and 2e-5 |
| k1 = k2, Pc = 1.0, NPHI = 8, R ≤ 0.167 (series through n = 8) | 34 | 2e-5, 4e-4, 7e-3 and 4e-6, 1e-4, 3e-3 | 4e-4 and 1e-4 |

The contact term is tested by the difference of two runs (`--minus`, see `tests/test_helical_triangle_direct.py`): it reproduces the contact part of the series to 3.5% (12 parameters) and 0.5% (14). At (3, 3.75) double precision leaves a noise of 7e-5 of the leading oscillation, and only the leading coefficient (4e-4) and the first correction (0.6%) are confirmed. Three things to know when repeating the fits:

- A fit needs clearly more points than parameters, and the background must stay smooth over the fitted range. Keep r below about Pc/6.
- With a cut near the energy 1 + |k2| of the two hard vertices (Pc = 1.4 and 1.8 in `data_vec/`) the loop integrand is close to a singularity, and the default grid leaves a noise of 1e-9 to 1e-7 of the leading oscillation. Those runs are kept for the record only.
- Away from collinear shapes the azimuthal sum converges slowly in the outer part of the loop integral. The error is analytic in r and goes into the background (NPHI = 8 and 12 differ by 2e-5 to 9e-5 of the leading oscillation at Pc = 1.0, while the fit residuals are 1e-11).
