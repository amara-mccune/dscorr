# dscorr

**Massive de Sitter correlators as finite Mellin–Barnes integrals.** dscorr is the code that accompanies the Letter

> A. McCune, *Massive de Sitter correlators as finite Mellin–Barnes integrals*, arXiv:2610.10921 (2026).

It computes the inflationary bispectrum ⟨πππ⟩′ mediated by heavy scalars on the principal series (m² = μ² + 9/4) at tree level:

- single, double and triple exchange at arbitrary masses and physical kinematics;
- the direct Schwinger–Keldysh time integration used as the independent check;
- the closed-form squeezed coefficients M₁, M₂, M₃ with signal templates and a fit;
- the kernel identity in arbitrary precision.

The one-loop results of the Letter (bubble, triangle, triangle correlator) come from batch pipelines that ship in `examples/Letter/batch/`, together with their values of record in `examples/Letter/data/`.

## How it works

Each massive line of a tree diagram is written with the kernel identity, Eq. (2) of the Letter:

    K_ν(y t_>) I_ν(y t_<) = ½ ∬ ds ds'/(2πi)² Γ(−s)Γ(−s')Γ(δ)Γ(1−δ)/[Γ(ν+s+1)Γ(ν+s'+1)] (y/2)^(−2δ) t^(ν+2s) t'^(ν+2s')

The vertex time integrals are then done in closed form. Every tree becomes one finite Mellin–Barnes integral, the master formula Eq. (4), evaluated on straight contours by the trapezoidal rule. The in-in correlator is assembled by the cutting rule (Supplemental Material, Sec. S3).

## Install

Requirements: Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.12, mpmath ≥ 1.3. The figures also need matplotlib ≥ 3.5, and the tests need pytest ≥ 7.

    git clone https://github.com/amara-mccune/dscorr.git
    cd dscorr
    pip install ".[figures,test]"

## Conventions

| Item | Convention |
|---|---|
| Units and couplings | Hubble rate 1, unit couplings |
| Power spectrum | P_π(k) = 1/(2k³) |
| Mass parameter | μ > 0 on every line |
| Output | `S = (k1 k2 k3)² ⟨πππ⟩′` by default (scale invariant) |
| Other outputs | `out='F'` (`S = k1 k2 k3 F/8`), `'B'` (`⟨πππ⟩′`), `'blocks'` |
| Layout | Fig. 1(a) of the Letter: the exchanged line is the **last** mass argument and ends on the mixing vertex of `k3` |

| Call | Diagram |
|---|---|
| `single_exchange(k1, k2, k3, mu)` | k1, k2 on the π̇π̇σ vertex; σ(μ) mixes into k3 |
| `double_exchange(k1, k2, k3, mu1, mu2)` | k1 on the π̇σ₁σ₂ vertex; σ₁ mixes into k2, σ₂ into k3 |
| `triple_exchange(k1, k2, k3, muA, muB, muC)` | σ_A, σ_B, σ_C mix into k1, k2, k3 |

Other channels are obtained by permuting the momentum arguments. `sum_perms=True` sums the three channels (single exchange) or the six leg assignments (double exchange). The triple exchange always returns one assignment; sum the permutations yourself.

## Quickstart

```python
import numpy as np
import dscorr

# tree level
dscorr.single_exchange(1, 1, 0.1, 1.0, sum_perms=True)       # 0.029826138118716...
dscorr.double_exchange(0.8, 1.0, 0.5, 1.0, 1.5)               # -0.0028136606935...
dscorr.triple_exchange(1, 0.8, 0.5, 1.0, 1.5, 0.7)            # -0.0192669778124...

# classes hold the per-mass-set preparation (a few seconds) and offer scans of (1, 1, kappa)
T = dscorr.TripleExchange(1.0, 1.5, 0.7)
S = T.scan(np.geomspace(0.002, 1, 100))
D = dscorr.DoubleExchange(1.0, 1.5, hL=0.002)                 # half time step (higher accuracy)

# squeezed limit: S -> S_an + 2 Re[M_L (k3/k1)^(1/2 + i mu)]   (Letter Eq. (10))
M3 = dscorr.M3(1.0, 1.5, 0.7)                                 # Letter Eq. (11), float64
M2 = dscorr.M2(1.0, 1.5, dps=30)                              # SM Sec. S4, 30 digits
M1 = dscorr.M1(1.0)
dscorr.signal(np.geomspace(1e-4, 1e-2, 5), M3, 0.7)           # 2 Re[M kappa^(i mu)]
kap = np.logspace(-4, -2, 80)
S = dscorr.TripleExchange(1.0, 1.5, 0.7, L1=16.0).scan(kap)   # warns: below the tested ratio
dscorr.extract(kap, S, 0.7).M                                 # fitted coefficient, within ~5e-10 of M3

# independent check: direct Schwinger-Keldysh time integration (mpmath, 10-60 s per value)
dscorr.double_exchange_direct(0.8, 1.0, 0.5, 1.0, 1.5)        # -0.002813660693529512
dscorr.triple_exchange_direct(1, 0.8, 0.5, 1.0, 1.5, 0.7, mpanels=28)

# kernel identity, Eq. (2), both sides in arbitrary precision
dscorr.kernel_identity_check("1.3j", 1.0, 0.7, 1.9, dps=20)
```

Inputs outside the tested range `dscorr.TESTED = {'mu_min': 0.3, 'mu_max': 4.0, 'momentum_ratio_min': 0.002}` are computed but raise `dscorr.ValidityWarning`. For min(k)/max(k) < 0.002, use a longer time grid (`L1=16`). Bad input raises `ValueError`.

**Numerical keywords** (not validated):

| Keyword | Default | Meaning |
|---|---|---|
| `L0`, `L1` | −60, 10.6 | ends of the grid in ln t |
| `hL` | 0.004; single exchange 0.002 | grid step |
| `c`, `T_sp` (double) / `T` (triple), `h` | −0.25, 12, 0.02 | contour abscissa, half-length, step |
| `c_s`, `T_s`, `h_s` | −0.125, 20, 0.02 | the same for the cut contours |

The direct route takes `mpanels` (22; use 28 for reference values), `nn` (24) and `dps` (25). It leaves the mpmath working precision at `dps` after a call.

## Accuracy and cost

**Trees.** Accuracy was measured against refined direct Schwinger–Keldysh values on 201 diagram-points (198 isosceles and scalene configurations, k3/k1 ≥ 0.002, masses up to μ = 2 on every line, plus a mass range up to 4):

| Setting | Worst | Median | Preparation per mass set | Per configuration (triple, one assignment) |
|---|---|---|---|---|
| default | 1.9e-10 | 2.8e-12 | ~6 s | 0.14–0.19 s |
| `hL=0.002` (half step) | 2.4e-11 (one point, see below) | 2.6e-13 | ~6 s | 0.28–0.41 s |
| fast (`L0=-54, L1=10.6, hL=0.005, T=8, h=0.05, T_s=8, h_s=0.025`) | 5.4e-10 | 6.5e-12 | 0.4 s | 0.10–0.13 s |

- The worst point is the triple exchange at μ = (0.5, 1, 1.5), k = (1, 1, 0.002). There S = −5.7e-6 is the sum of blocks of size ~0.8, so its floating-point floor is ~1e-11. At every other point the half-step setting is better than 2e-11.
- Times are for one core and one linear-algebra worker.
- `M2`, `M3` in float64 take ~0.2 s and agree with their 150-digit series route (`dps=N`, 10–40 s) to ~4e-12.
- Fits over 1e-4 ≤ k3/k1 ≤ 1e-2 return them to a few 1e-10.
- Direct Schwinger–Keldysh values cost 10–60 s each.

**One loop.** These are batch computations (see below):
- Conformal-leg triangle: the Mellin–Barnes and brute-force routes agree to a median 3e-6 at 33 configurations.
- π̇-leg triangle: median 5.5e-6 at 14 configurations.
- Bubble: the residue series agrees with the spectral decomposition of Xianyu–Zhang to 1e-6 for r ≤ 0.45.

## Figures of the Letter

Every figure of the Letter and of its Supplemental Material is regenerated by one script:

    cd examples/Letter
    python make_figures.py all          # writes figures/fig1.pdf ... figures/figS3.pdf

| Command | Output | Content | Source of the data |
|---|---|---|---|
| `python make_figures.py fig1` | `fig1.pdf` | diagrams: double exchange, triangle | drawing |
| `python make_figures.py fig2` | `fig2.pdf` | (a) triple exchange S(1,1,κ) for three mass triples, with direct-SK squares; (b) squeezed signal (S − S_an)(k1/k3)^(1/2) and leading term | dscorr, computed live (~15 min, mostly the 45 direct points) |
| `python make_figures.py fig3` | `fig3.pdf` | one-loop triangle along (k1, 1, 1.1): conformal legs (solid, circles) and π̇ legs (dashed, squares) | `data/` (values of record) |
| `python make_figures.py figS1` | `figS1.pdf` | double exchange S(1,1,κ) with direct-SK squares | dscorr, live (~10 min) |
| `python make_figures.py figS2` | `figS2.pdf` | (a),(b) double/triple exchange summed over the six leg assignments; (c) triangle correlator with π̇ legs, equilateral-normalized | (a),(b) dscorr, live (~1 min); (c) `data/` |
| `python make_figures.py figS3` | `figS3.pdf` | renormalized bubble: residue series (curve) and spectral decomposition (circles) | `data/` |

Options:
- `--no-direct` skips the direct Schwinger–Keldysh squares.
- `--quick` uses coarse grids, for a test in a few minutes.
- `--png` also writes PNG files.

Computed values are cached in `examples/Letter/cache/`; delete that folder to recompute.

Comparison points from other groups' codes are not part of this repository: CosmoFlow (Fig. S2(a),(b)) and the integrator of Herderschee and Lu (Fig. S2(c)). The diamonds of Fig. S1 (series solution of Aoki et al.) and the circles of Fig. S3 (spectral decomposition of Xianyu and Zhang) were computed by the author from those papers' formulas and ship in `examples/Letter/data/` with that credit. The script draws them if you place them, in the documented CSV format, in `examples/Letter/data/external/` (see the docstring of `make_figures.py`).

## One-loop batch pipelines

`examples/Letter/batch/` contains the four production pipelines behind the loop figures:

| Pipeline | Content | Cost |
|---|---|---|
| `bubble/` | bubble, Fig. S3 | ~1e2 core-hours |
| `triangle/` | conformal-leg triangle, Fig. 3 | ~570 + 160 core-hours |
| `triangle_pidot/` | π̇-leg triangle, Fig. 3 | ~45 core-hours |
| `triangle_pidot_correlator/` | triangle correlator, Fig. S2(c) | ~36 core-hours |

They are scripts, not library calls, and depend only on numpy, scipy and mpmath. Their commands are listed in [`examples/Letter/batch/README.md`](examples/Letter/batch/README.md). Their results are the tables in [`examples/Letter/data/`](examples/Letter/data/README.md), which the figure script reads. `dscorr.data.load(name)` reads the same tables from a source checkout.

## Tests

    pytest              # fast tier, about 3 minutes on one core
    pytest -m slow      # converged direct values, 30-digit closed forms, deep-squeezed fits (~20 min)

Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` for reproducible timings. The tests of the batch pipelines (`tests/test_oneloop.py`) run only from a source checkout, because they import `examples/Letter/batch/`.

## How to cite

If you use dscorr, please cite the Letter and the code:

```bibtex
@article{McCune2026dscorr,
  author        = {McCune, Amara},
  title         = {Massive de Sitter correlators as finite Mellin-Barnes integrals},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {hep-th}
}
@software{dscorr,
  author  = {McCune, Amara},
  title   = {dscorr: massive de Sitter correlators as finite Mellin-Barnes integrals},
  version = {1.0},
  year    = {2026},
  url     = {https://github.com/amara-mccune/dscorr}
}
```

See also `CITATION.cff`.

## License

MIT; see [LICENSE](LICENSE).
