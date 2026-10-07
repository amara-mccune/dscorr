# One-loop batch pipelines

These are the production pipelines behind the one-loop results of the Letter. They are batch scripts, not library calls. They do not import `dscorr` and need only numpy, scipy and (bubble) mpmath.

`triangle_pidot/` and `triangle_pidot_correlator/` import the brute-force modules of `triangle/`, so they must stay beside it. Their results are the tables in `../data/` (see `../data/README.md`).

**Full regeneration takes about 45 to 730 core-hours per pipeline.** Run the commands from inside each folder. Times are for one core unless a worker count is given.

| Pipeline | Commands | Cost |
|---|---|---|
| **(i) bubble** (Fig. S3): S(r) at μ = 1.472, 50 ratios; problem name `fig3c` | see below | ~1e2 core-hours; one ratio ~1 h |
| **(ii) loop triangle, conformal legs** (Fig. 3, solid lines and circles): 33 configurations | 1. `python mb_point.py tasks.json out/mb` (A at h_s = 0.075; R at h_s = 0.05 and 0.04). 2. `python b_worker.py task.json out/b/<tag>_P1.json`. 3. `python far_patch.py out/b/<tag>_P1.json`. 4. `python assemble_scan.py --mb out/mb --b out/b --out triangle_scan.json` | ~570 core-hours (MB, including calibrations) + ~160 core-hours (brute force) |
| **(iii) loop triangle, π̇ legs** (Fig. 3, dashed lines and squares): 14 configurations | 1. `python run_mb.py tasks_mb_primary.json out/mb [--blas 1]`, and likewise `tasks_mb_crosscheck.json` and `tasks_mb_checks.json`. 2. `python run_bf.py tasks_bf/bf_task_<tag>.json out/bf/bf_<tag>.json --nproc 16`. 3. `python assemble_pidot.py --mb out/mb --bf out/bf --out assembled.json`. 4. `python make_table.py assembled.json --json table.json --csv table.csv` | ~45 core-hours (MB ~6 min, brute force ~2.5 core-hours per configuration) |
| **(iv) triangle correlator, π̇ legs** (Fig. S2(c)): μ = 0.5, 0.7, 17 configurations each, grid levels Q0–Q3 | `python run_triangle_correlator.py tasks_correlator.json --out out --workers 16 [--grid Q0,Q1] [--tags ...]` | Q0 ~6 min per point, Q1 ~4× that, Q2/Q3 hours; ~36 core-hours at Q0 and Q1 |

**Bubble commands, in order:**
1. Stage 1: `python launch_stage1.py --rfile rlist_fig3.txt --pieces pieces`. This runs 122 tasks of `letter_dii_worker.py`.
2. Merge: `python merge_dii.py --problem fig3c --pieces pieces --out cache_fig3c_dii.json`.
3. Stage 2: `python launch_stage2.py`. This runs 10 chunks of `stage2_worker.py` with the sentinel ratio 0.55 and writes `cache_fig3c_rest.json`.
4. Join the caches with `emit_results.join_caches`.
5. Results: `PYTHONPATH=. python emit_results.py --cache cache_fig3c.json`.
6. Analysis: `PYTHONPATH=. python fig3c_analysis.py --cache cache_fig3c.json`.

To compute a single ratio in one process, run `PYTHONPATH=. python letter_msbar.py --r 0.1 --dps 20 --Nmax 24 --Mmax 24 --mpmax 80 --h 0.05 --hU 0.025 --T 12 --tailrule --tailscale 1.3 --Ktail 3 --npconv`. The defaults of `letter_msbar.py` are *not* the production parameters.

`fig3c_analysis.py` also compares with the spectral decomposition when the optional file `data/fig3_dense_data.json` is present. That file is not shipped, but the spectral values are in `../data/bubble_spectral_xianyu_zhang.csv`.

**Minute-scale checkpoints** of each pipeline (single rungs, residues, fixed-loop-momentum values, assemblies) are the tests in `tests/test_oneloop.py`. The slow ones are marked `@pytest.mark.slow`.
