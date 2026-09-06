# Updated dataset: parameter tests

The updated initial angles are −22.5° and +18°; the old data used −22.5° and −18°. The file has 2,002 rows over 0–20.01 seconds with uniform 0.01-second spacing. input.dat preserves the tested file; old_input.dat reconstructs the old reference from the previously saved prediction CSV.

The production FPINN now uses DATA_STEP=30 (10 measurements in the first 3 seconds), whereas the previous successful benchmark used DATA_STEP=10 (30 measurements). The current learning rate, warmup, ramp and schedule also differ. `current_sparse` reproduces those current parameters; `current_script` restores 30 measurements only; `previous_winner` reproduces the full old winning configuration.

All tests retain the existing fpinn2.py algorithm. Only parameter values and benchmark observations change. Production scripts and Outputs were not edited. Each result.json records parameter overrides, source and input hashes, and all 1,000-update observations; source_snapshot.py captures the remaining defaults. Training logs and predictions are in each run folder.

Success requires both angle R² values ≥0.999 over the full record and the held-out region, plus normalized physics residual ≤1e-5, for three consecutive observations. For most runs the held-out region starts at row 300 (3 s); window10s runs start it at row 1000 (10 s). Those window runs still use 30 data points, spaced every 34 rows, but change the forecasting task and are not directly comparable.

| Run | Updates | Worst full R² | Worst held-out R² | Physics | Status |
|---|---:|---:|---:|---:|---|
| current_script | 30,000 | 0.097678 | -0.065224 | 0.00105 | budget_exhausted |
| current_sparse | 30,000 | 0.100127 | -0.062252 | 0.00106 | budget_exhausted |
| cutoff12 | 30,000 | 0.119184 | -0.033541 | 0.005254 | budget_exhausted |
| cutoff16 | 30,000 | 0.205866 | 0.062536 | 0.00161 | budget_exhausted |
| cutoff30 | 30,000 | -0.072371 | -0.266535 | 0.000209 | budget_exhausted |
| cutoff50 | 30,000 | 0.038406 | -0.135710 | 0.0004979 | budget_exhausted |
| cutoff50_physics10000 | 30,000 | -0.090920 | -0.288410 | 0.0003373 | budget_exhausted |
| cutoff8 | 30,000 | 0.118489 | -0.033816 | 0.005687 | budget_exhausted |
| early_physics | 30,000 | 0.097921 | -0.064939 | 0.001016 | budget_exhausted |
| energy1 | 30,000 | -0.256319 | -0.483823 | 0.0002454 | budget_exhausted |
| energy10 | 30,000 | -0.228859 | -0.451132 | 0.0002596 | budget_exhausted |
| period4_cutoff30 | 30,000 | -0.179738 | -0.393351 | 0.0002804 | budget_exhausted |
| physics10000 | 30,000 | 0.027608 | -0.143621 | 0.0006299 | budget_exhausted |
| previous_winner | 30,000 | 0.097796 | -0.065085 | 0.001014 | budget_exhausted |
| ridge_small | 30,000 | 0.091536 | -0.072464 | 0.00104 | budget_exhausted |
| spectrum_fast | 30,000 | -0.069839 | -0.263536 | 0.0007013 | budget_exhausted |
| window10s | 30,000 | 0.626254 | 0.248012 | 0.0003007 | budget_exhausted |
| window10s_cutoff12 | 30,000 | 0.448980 | -0.014619 | 0.01125 | budget_exhausted |
| window10s_energy | 30,000 | 0.697090 | 0.400284 | 0.000333 | budget_exhausted |

A failed 30,000-update screen does not prove the configuration can never converge. These results do show that the old winning settings are not robust to the initial-condition change. CPU runs were concurrent, so their timings are not GPU benchmarks. No TPINN-versus-FPINN convergence advantage is established on this new data.

Figures: [fit comparison](fit_comparison.png), [convergence](convergence.png), [best screened run animation](best_training.gif), [losses](best_loss.png). “Best” means highest final worst R² among screened runs; it does not imply successful convergence.
