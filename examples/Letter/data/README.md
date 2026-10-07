# Data of the Letter's one-loop figures

These tables hold the values of record of the one-loop production runs behind Fig. 3, Fig. S2(c) and Fig. S3 of the Letter, and two sets of comparison values computed from other papers' formulas. They are the author's own results, computed with the batch pipelines in `../batch/`. Regenerating them takes tens to hundreds of core-hours (see `../batch/README.md`), so the plotting script `../make_figures.py` reads them from here.

All values are double precision as produced. Lines starting with `#` are comments.

| File | Figure | Content | Produced by | Cost to regenerate |
|---|---|---|---|---|
| `triangle_conformal_33config.csv` | Fig. 3, solid lines (abs MB) and circles (abs Breg) | ψ₃^{loop,reg} of Eq. (8), conformally coupled legs (r_v = 1), at 33 configurations (see note 1) | `batch/triangle/` | ~570 core-hours (MB) + ~160 (brute force) |
| `triangle_pidot_14config.csv` | Fig. 3, dashed lines (abs MB) and squares (abs B) | the finite triangle with one π̇ leg per vertex (r_v = 2), at 14 configurations (see note 2) | `batch/triangle_pidot/` | ~45 core-hours |
| `triangle_correlator_Q1.csv` | Fig. S2(c), lines | S = (k1k2k3)²⟨πππ⟩′ of the r_v = 2 triangle with equal masses μ = 0.5, 0.7 on k1 = k2 = 1 (10 values of k3 each), brute force at grid level Q1, and S/S(1,1,1) | `batch/triangle_pidot_correlator/` | ~36 core-hours at levels Q0 and Q1 |
| `triangle_correlator_ladder.csv` | (convergence) | the same at the levels Q0–Q3 and an independent grid G2, for the 8 configurations computed on more than two levels | as above | hours per point at Q2/Q3 |
| `bubble_S_of_r.csv` | Fig. S3, curve | the renormalized bubble S(r) = −8 k1 k2 k3⁴ B_XZ at μ = 1.472, k1 = k2 = 1, k3 = 2r, 50 ratios r ∈ [0.1, 0.55] (see note 3) | `batch/bubble/` | ~1e2 core-hours |
| `bubble_spectral_xianyu_zhang.csv` | Fig. S3, circles | the spectral decomposition of the same bubble (see the credit below) | separate script, not part of dscorr | — |
| `aoki_et_al_double_series.csv` | Fig. S1, diamonds | the series solution of Aoki et al. for the double-exchange bispectrum S at (1, 1, κ), 21 points (see the credit below) | separate script, not part of dscorr | — |

Notes:
1. Conformal triangle: MB = Mellin–Barnes route, Breg = brute force. The momentum family (k1, 1, 1.1) has 10 values of k1, each at the mass sets (1,1,1) and (1,1.3,0.7). The controls are (1, 1.2, 0.9) and (1,1,1), and there is an equal-mass scan at (1, 1.2, 0.9).
2. π̇ triangle: the family (k1, 1, 1.1) has 6 values of k1, and the control is (1, 1.2, 0.9). MB is quoted at contour placement Re δ = 0.6, and the columns MBalt hold a second placement. The tables also carry both routes' own error estimates.
3. Bubble: the sum of all Mellin–Barnes residue families, in the MS-bar convention of Xianyu and Zhang at μ_R = μ, with a truncation bound.

Further inputs of the triangle pipeline are in `../batch/triangle/data/`: the D₀ coefficients of Eq. (15), with errors, for the 11 mass sets.

**Credit.** `bubble_spectral_xianyu_zhang.csv` holds values of the spectral decomposition of Z.-Z. Xianyu and H. Zhang, JHEP 04 (2023) 103, arXiv:2211.03810. The author computed them for the Letter from the formulas printed in that paper, and they are credited to that paper. The script that computed them is not part of dscorr, and the values are shipped unchanged as reference data.

`aoki_et_al_double_series.csv` holds values of the series solution of S. Aoki, L. Pinol, F. Sano, M. Yamaguchi and Y. Zhu, JHEP 09 (2024) 176, arXiv:2404.09547. The author computed them for the Letter from Eqs. (4.2), (4.4)–(4.6) and (4.8) of that paper, with the single sums ordered in the soft-leg ratio 2k3/k123 and summed to N = 80 terms (omitted remainder below 1e-13 relative). They are credited to that paper. The script that computed them is not part of dscorr.

**Comparison data from other groups.** The circles of Fig. S2(a),(b) (CosmoFlow) and the circles of Fig. S2(c) (the integrator of Herderschee and Lu) are not shipped. To draw them, put the files described in the docstring of `../make_figures.py` into `external/` in this directory.

**Python access.** Use `dscorr.data.load("triangle_pidot_14config.csv")` from a source checkout. In another location, set `DSCORR_DATA` to this directory.
