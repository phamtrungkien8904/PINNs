# Parameter-only convergence test

These are fresh CPU reruns. Earlier test artifacts were unavailable, so conclusions below use only the saved results in this folder. Production scripts were not edited.

## Method

FPINN uses the existing fpinn2.py algorithm: four real spectral outputs, trainable Fourier-bin coefficients plus the original MLP correction, FFT reconstruction and spectral derivatives. TPINN uses tpinn2_ver2.py: time input, two angle outputs and second derivatives from autograd. The current tpinn2.py has a four-state model, so it was not used for this requested two-output comparison.

Both use the same 30 observations (rows 0:300:10). R² is evaluated separately for each angle over the entire reference and the extrapolation region (rows 300 onward). Success requires all four R² values ≥ 0.999 and mean((f1/10)²+(f2/10)²) ≤ 1e-5 at three consecutive checks, spaced 1,000 optimizer updates apart. This is an evaluation stop, not a changed training algorithm. Epoch counts below mean optimizer updates; the original loops number their first update as epoch zero.

## Results

| Run | Status | Updates | First target | Worst R² | Physics |
|---|---|---:|---:|---:|---:|
| fourier_baseline | budget_exhausted | 20,000 | not reached | 0.28541764 | 0.00023289 |
| fourier_seed0 | converged | 17,000 | 15000 | 0.99956563 | 4.9738e-06 |
| fourier_seed1 | converged | 17,000 | 15000 | 0.99953595 | 5.2183e-06 |
| fourier_seed2 | converged | 22,000 | 15000 | 0.99970804 | 5.7722e-06 |
| time_baseline | budget_exhausted | 25,000 | not reached | 0.01518613 | 0.0019363 |

## Tested Fourier candidate

```python
FOURIER_PERIOD_FACTOR = 2
MAX_ANGULAR_FREQUENCY = 20.0
LAMBDA_PHYSICS = 1000.0
WARMUP_EPOCHS = 1000
PHYSICS_RAMP_EPOCHS = 4000
LEARNING_RATE_NETWORK = 2e-4
LEARNING_RATE_SPECTRUM = 2e-4
WEIGHT_DECAY = 1e-8
LAMBDA_DATA = 1000.0
LAMBDA_INITIAL = 500.0
# Original MultiStepLR, milestones: [10000, 30000, 60000], gamma=0.3
```

The Fourier baseline uses the same settings except period factor 4, cutoff 12 and physics weight 10. Therefore the within-Fourier improvement isolates these three parameter changes together. The full candidate above differs in additional parameters from the current production file. TPINN uses width 128, three hidden tanh layers, 1,024 physics points, AdamW learning rate 2e-4, weight decay 1e-8, data weight 1000, initial weight 500, physics weight 0.1, energy weight 0.001. All original loss equations and optimizer algorithms are retained.

## Limits and artifacts

This is a candidate versus one TPINN parameter baseline, not proof that every optimized TPINN needs more epochs. A budget-exhausted run has no measured convergence epoch. Runtime values in summary.csv include observation overhead and concurrent CPU contention; they are not an A5000 speed comparison. HPC scripts and wider GPU models were not tested.

Each run includes source_snapshot.py, result.json (parameters and source hash), training.log, prediction.csv and snapshots.npz. run_trial.py applies parameter overrides and adds an observation callback to a copy of the source in memory. No production file is imported under its main name or edited.

Plots: [convergence](convergence.png), [final fit](fourier_prediction.png), [training animation](fourier_training.gif), [loss](fourier_loss.png).
