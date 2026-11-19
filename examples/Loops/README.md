# Code for "Massive de Sitter correlators at one loop"

These scripts reproduce the results of the squeezed and collapsed limits of the companion paper (Sections 6.2, 6.3, 7.4, 7.5, 8.2, and 8.3):

- the universal factor between the exact and the late-time soft vertex, and the W/Z loop;
- the charged-scalar bubble with a chemical potential on the vertex;
- the parity-odd collapsed trispectrum of a helical vector loop;
- the Dirac-pair bubble with an axial chemical potential, and its comparison with Han, Lei, Qin and Yin (2611.20919);
- the fermion triangle with an axial chemical potential, and its comparison with the closed form of Song, You, Cheng, He and Chen (2611.23220);
- the fermion triangle away from the squeezed limit (all orders in k₃/k₁, at every shape), with a direct time-domain integration of the full triangle.
- the complete bispectrum of two-vertex loops at every shape: the charged-scalar bubble and the Higgs-portal loop as single sums, and the series of the W/Z, top-quark, and Dirac loops beyond their leading squeezed term.

Like `examples/Letter/batch/`, they are scripts rather than library calls, and they do not import `dscorr`. They need only Python ≥ 3.10 and mpmath ≥ 1.3 (the full-shape scripts also use numpy, for the tree sums of `ext/bubble.py`); the Letter's tests also use numpy, scipy and pytest. Every script compares its output with the numbers quoted in the paper, which are kept in [`values_of_record.json`](values_of_record.json). It prints one line per value and exits with a nonzero status if any value is not reproduced.

Run the scripts from this folder:

    cd examples/Loops
    python universal.py

## Scripts

| Script | Paper | What it computes | Time (one core) |
|---|---|---|---|
| `universal.py` | Sec. 6.2, Eq. (6.2) | U(Δ, r₂). It checks U against the ratio of the exact and late-time moments of Eq. (3.17) (to 1e-31) and against direct quadrature of the position-space (EAdS) soft vertex (to 1e-16). It also gives the values quoted for heavy and light lines. | 10 s |
| `gauge_loop.py` | Sec. 6.3 | The W/Z loop (f_A(X)F², Proca with all three polarizations). It compares the exact soft moment, with E and B kept at the soft vertex, against its late-time version. Summed over the nine helicity pairs, the ratio equals U(3+2ν, 2) (to 1e-23); the individual pairs do not. | 10 min |
| `charged_scalars.py` | Sec. 7.5 | Squeezed families of the charged-scalar bubble in the normalization of Bodas, Broadberry, Sundrum and Xu (2507.22978). The script compares them with their Eq. (3.35), near and below threshold. It also computes the resonance residue and the difference line. | 1 min |
| `charged_scalars.py --shape` | Sec. 7.5 | The complete bispectrum of the charged-scalar loop at every shape (`ChargedShape`): families, background, and the three channels, in the derivative form of the vertices. It is checked against the squeezed families (to 1e-43), against a numerical spectral integral over tree exchanges at five shapes (to 1.5e-13), and at the resonance. It tabulates the full signal against its leading term and against the template and background of BBSX. `--quick` runs one benchmark. | 8 min (`--quick`: 2 min) |
| `higgs_portal.py` | Secs. 6.2, 6.3 | The Higgs-portal loop f_H(X)H†H at every shape in MSbar, in closed form (families as one sum over trees, background on the half-integer ladder with digamma coefficients). Checks: invariant-distance form at five shapes; the late-time leading term against C_H of Chen, Wang and Xianyu (1612.08122), to 1e-29. It also tests the universal factor order by order in k₃/k₁₂. | 5 min (`--quick`: 30 s) |
| `gauge_loop.py --orders` | Sec. 6.2 | The W/Z loop beyond the leading order: the nine-pair soft moment at orders 0 and 1 of the hard vertex is a combination of the soft factors of the boundary operators of dimension 3+2ν and 5+2ν (tested through the dependence on the soft power, to 1e-23). | 25 min |
| `dirac_pair.py --orders` | Secs. 6.3, 8.2 | The series S(n) of the Dirac-pair bubble through n = 8, exact against the late-time pair term, and the top-quark loop (κ = 0): only odd orders, soft factors of dimension 4+2ν and 6+2ν, vanishing Yukawa signal. Pair signal only. | 25 min |
| `parity_box.py` | Sec. 7.4, Eqs. (7.15)–(7.16) | Hard-line moments in closed form, checked against quadrature to 1e-20. Box harmonics c_n in the dihedral angle. Chirality ratios \|c₋₂/c₊₂\|. Parity-odd fraction of the full collapsed trispectrum. | 5 min |
| `dirac_pair.py` | Sec. 8.2 | The squeezed coefficient M_D(n) of the Dirac-pair bubble, with the soft moment in position space. The script compares it with the late-time pair term and with the all-branch late-time estimate. `--quick` runs only m = κ = 1. | 4 min |
| `han_check.py` | Sec. 8.2 | The squeezed bispectrum of the fermion bubble of Han, Lei, Qin and Yin (2611.20919), in their conventions, with our position-space soft vertex. It compares the amplitude and phase with their Fig. 8 (digitized in [`han_fig8.json`](han_fig8.json)), and checks that S(0) of `dirac_pair.py` is their single-sector coefficient. `--all` runs all 21 points of the paper. | 5 min (`--all`: 1.5 h) |
| `song_check.py` | Sec. 8.2 | The closed form of Song, You, Cheng, He and Chen (2611.23220) for the leading squeezed coefficient of the fermion triangle (doubled for a Dirac fermion), compared with the values of record of `fermion_triangle.py`. `--all` adds κ = 20 at 80 digits. | 5 min (`--all`: 30 min) |
| `fermion_triangle.py` | Sec. 8.2 | The leading squeezed coefficient C of the Aoki–Strumia fermion triangle (2604.05548), together with its late-time versions (`nl`, `all0`). The hard part is in closed form (Eq. (B.8)); `--hard contour` uses the earlier contour evaluation. `--point κ m` runs one point, and `--all` runs the four points of the paper. | κ = 1.3: 3 min; 5: 5 min; 10: 7 min; 20: 15 min |
| `fermion_triangle_shape.py` | Sec. 8.3, Eqs. (8.6)–(8.8) | The pair-frequency signal of the fermion triangle to all orders in k₃/k₁ and at every shape, in R = 2k₃/(k₁+k₂) and δ = (k₁−k₂)/k₃ (`ext/fermitri.py`, `ext/shapevars.py`). It checks the exchange k₁ ↔ k₂ order by order and the leading coefficient against twice the closed form of Song et al. See `README_fermion_triangle_shape.md`. `direct/` holds the direct time-domain integration of the full triangle, its fits against the series, and the runs behind Sec. 8.3. | κ = 1.3 through n = 2: 3 min; `--all`: hours |

Two statements of Sec. 6.3:

- The vanishing of the leading pair coefficient of ψ̄ψ at κ = 0 (the top-quark loop) is checked in `dirac_pair.py`, where p̄₀₀ = 0 for every helicity pair.
- The comparison with C_H of Chen, Wang and Xianyu (1612.08122) is in `higgs_portal.py` (vertices and normalization from their Eqs. (4.2)–(4.4), coefficient from their Eq. (4.15)).

Shared code is in `ext/`:

| Module | Content |
|---|---|
| `bubble.py` | The complete bubble as single sums: single-time moments, the ordered tree and its remainder (Gegenbauer ladder), tree weights and tree sums for the four sign patterns, and the closed form of the cut (the spectral density of the pair at imaginary argument). Valid on the bispectrum locus x₂ = k₃ and at complex vertex powers. |
| `blocks.py` | Loop block C₂ (2.4), two-line soft moment (3.17), hard-line moments (B.3) and (7.14), Whittaker branch series (D.6), Mellin and Laplace transforms of W (D.8), (8.4) |
| `softvertex.py` | Position-space soft vertex: radial integral of two Laplace-transformed Whittaker functions [Eqs. (8.5), (D.11)]. The large-w tail is integrated term by term from its convergent 1/w series, so no fit is involved. |
| `cheb.py` | Chebyshev panels in ln t with a spectral cumulative-integration operator (the nested time integrals of the hard parts) |
| `dirac.py`, `fermion.py` | Dirac line with an axial chemical potential, in the conventions of `dirac_pair.py` and of Aoki–Strumia, respectively |
| `helical.py` | Helical vector line: soft late-time ends; Schwinger–Keldysh hard moments in closed form, transverse and longitudinal |
| `record.py` | Comparison with `values_of_record.json` |

## Methods and conventions worth knowing

- **Soft vertex in position space.** At large chemical potential the radial integrand reaches 5e11 (κ = 10) or 6e26 (κ = 20) and cancels down to O(10–100). Everything is therefore done in mpmath at 30–60 digits, with exact quadrature nodes.
- **Hard part of the triangle.** By default it is in closed form: Gamma functions for the cut sectors and the ₄F₃(−1) moments of Eq. (B.8) for the ordered ones. The earlier evaluation (`--hard contour`) uses the rays η = −t e^{∓iθ}, with small θ at large κ (0.25 at κ = 10, 0.15 at κ = 20), since on the fully rotated contour terms of size e^{πλ} cancel. It is cached in `cache/`.
- **Overall sign of the triangle.** The in-in sign is ⟨δτ³⟩′ = +Σ a₁a₂a₃ F F F Tr[…]. Each vertex contributes i a L_int with L_int = +∂δτ Ψ̄ΓΨ, the closed loop gives −Tr, and S = ⟨TΨΨ̄⟩ = iD. This is the opposite of the leading minus printed in Aoki–Strumia's Eq. (45). Moduli do not depend on it.
- **Late-time estimates.** In the all-branch estimates (`all0`, and the all-branch Dirac estimate), the branch pairs whose ν's cancel give integer line powers. Their loop integral is ultraviolet-divergent or scaleless, so these terms are dropped.
- **Parity-odd fraction.** It is quoted as max_n \|c_n − c₋ₙ\| / (2\|c₀\|). The sin(nβ) term itself has twice this amplitude.

## Tests

`tests/test_loops.py` has minute-scale checkpoints of every script:

    pytest tests/test_loops.py            # fast tier, about 2 min
    pytest -m slow tests/test_loops.py    # also the triangle at κ = 1.3, the hard-moment quadrature, and the W/Z soft factor
