# Code for "Massive de Sitter correlators at one loop"

These scripts reproduce the results of the squeezed and collapsed limits of the companion paper (Sections 7.2, 8.4, 8.5 and 9.2):

- the universal factor between the exact and the late-time soft vertex, and the W/Z loop;
- the charged-scalar bubble with a chemical potential on the vertex;
- the parity-odd collapsed trispectrum of a helical vector loop;
- the Dirac-pair bubble with an axial chemical potential, and its comparison with Han, Lei, Qin and Yin (2611.20919);
- the fermion triangle with an axial chemical potential, and its comparison with the closed form of Song, You, Cheng, He and Chen (2611.23220).

Like `examples/Letter/batch/`, they are scripts rather than library calls, and they do not import `dscorr`. They need only Python ≥ 3.10 and mpmath ≥ 1.3; the Letter's tests also use numpy, scipy and pytest. Every script compares its output with the numbers quoted in the paper, which are kept in [`values_of_record.json`](values_of_record.json). It prints one line per value and exits with a nonzero status if any value is not reproduced.

Run the scripts from this folder:

    cd examples/Loops
    python universal.py

## Scripts

| Script | Paper | What it computes | Time (one core) |
|---|---|---|---|
| `universal.py` | Sec. 7.2, Eq. (7.1) | U(Δ, r₂). It checks U against the ratio of the exact and late-time moments of Eq. (3.17) (to 1e-31) and against direct quadrature of the position-space (EAdS) soft vertex (to 1e-16). It also gives the values quoted for heavy and light lines. | 10 s |
| `gauge_loop.py` | Sec. 7.3 | The W/Z loop (f_A(X)F², Proca with all three polarizations). It compares the exact soft moment, with E and B kept at the soft vertex, against its late-time version. Summed over the nine helicity pairs, the ratio equals U(3+2ν, 2) (to 1e-23); the individual pairs do not. | 10 min |
| `charged_scalars.py` | Sec. 8.5 | Squeezed families of the charged-scalar bubble in the normalization of Bodas, Broadberry, Sundrum and Xu (2507.22978). The script compares them with their Eq. (3.35), near and below threshold. It also computes the resonance residue and the difference line. | 1 min |
| `parity_box.py` | Sec. 8.4, Eqs. (8.15)–(8.16) | Hard-line moments in closed form, checked against quadrature to 1e-20. Box harmonics c_n in the dihedral angle. Chirality ratios \|c₋₂/c₊₂\|. Parity-odd fraction of the full collapsed trispectrum. | 5 min |
| `dirac_pair.py` | Sec. 9.2 | The squeezed coefficient M_D(n) of the Dirac-pair bubble, with the soft moment in position space. The script compares it with the late-time pair term and with the all-branch late-time estimate. `--quick` runs only m = κ = 1. | 4 min |
| `han_check.py` | Sec. 9.2 | The squeezed bispectrum of the fermion bubble of Han, Lei, Qin and Yin (2611.20919), in their conventions, with our position-space soft vertex. It compares the amplitude and phase with their Fig. 8 (digitized in [`han_fig8.json`](han_fig8.json)), and checks that S(0) of `dirac_pair.py` is their single-sector coefficient. `--all` runs all 21 points of the paper. | 5 min (`--all`: 1.5 h) |
| `song_check.py` | Sec. 9.2 | The closed form of Song, You, Cheng, He and Chen (2611.23220) for the leading squeezed coefficient of the fermion triangle (doubled for a Dirac fermion), compared with the values of record of `fermion_triangle.py`. `--all` adds κ = 20 at 80 digits. | 5 min (`--all`: 30 min) |
| `fermion_triangle.py` | Sec. 9.2 | The leading squeezed coefficient C of the Aoki–Strumia fermion triangle (2604.05548), together with its late-time versions (`nl`, `all0`). `--point κ m` runs one point, and `--all` runs the four points of the paper. | κ = 1.3: 3 min; 5: 5 min; 10: 7 min; 20: 15 min |

Two statements of Sec. 7.3 have no script of their own:

- The vanishing of the leading pair coefficient of ψ̄ψ at κ = 0 (the top-quark loop) is checked in `dirac_pair.py`, where p̄₀₀ = 0 for every helicity pair.
- The comparison with C_H of Chen, Wang and Xianyu (1612.08122) is not included here, because it needs their Eq. (4.4).

Shared code is in `ext/`:

| Module | Content |
|---|---|
| `blocks.py` | Loop block C₂ (2.4), two-line soft moment (3.17), hard-line moments (B.4) and (8.14), Whittaker branch series (D.6), Mellin and Laplace transforms of W (D.8), (9.8) |
| `softvertex.py` | Position-space soft vertex: radial integral of two Laplace-transformed Whittaker functions [Eqs. (9.9), (D.11)]. The large-w tail is integrated term by term from its convergent 1/w series, so no fit is involved. |
| `cheb.py` | Chebyshev panels in ln t with a spectral cumulative-integration operator (the nested time integrals of the hard parts) |
| `dirac.py`, `fermion.py` | Dirac line with an axial chemical potential, in the conventions of `dirac_pair.py` and of Aoki–Strumia, respectively |
| `helical.py` | Helical vector line: soft late-time ends; Schwinger–Keldysh hard moments in closed form, transverse and longitudinal |
| `record.py` | Comparison with `values_of_record.json` |

## Methods and conventions worth knowing

- **Soft vertex in position space.** At large chemical potential the radial integrand reaches 5e11 (κ = 10) or 6e26 (κ = 20) and cancels down to O(10–100). Everything is therefore done in mpmath at 30–60 digits, with exact quadrature nodes.
- **Hard part of the triangle.** The rays are η = −t e^{∓iθ}, with small θ at large κ (0.25 at κ = 10, 0.15 at κ = 20), since on the fully rotated contour terms of size e^{πλ} cancel. The hard part is cached in `cache/`.
- **Overall sign of the triangle.** The in-in sign is ⟨δτ³⟩′ = +Σ a₁a₂a₃ F F F Tr[…]. Each vertex contributes i a L_int with L_int = +∂δτ Ψ̄ΓΨ, the closed loop gives −Tr, and S = ⟨TΨΨ̄⟩ = iD. This is the opposite of the leading minus printed in Aoki–Strumia's Eq. (45). Moduli do not depend on it.
- **Late-time estimates.** In the all-branch estimates (`all0`, and the all-branch Dirac estimate), the branch pairs whose ν's cancel give integer line powers. Their loop integral is ultraviolet-divergent or scaleless, so these terms are dropped.
- **Parity-odd fraction.** It is quoted as max_n \|c_n − c₋ₙ\| / (2\|c₀\|). The sin(nβ) term itself has twice this amplitude.

## Tests

`tests/test_loops.py` has minute-scale checkpoints of every script:

    pytest tests/test_loops.py            # fast tier, about 2 min
    pytest -m slow tests/test_loops.py    # also the triangle at κ = 1.3 and the hard-moment quadrature
