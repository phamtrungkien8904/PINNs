# Status: superseded — training-data constraint restored

The user requires DATA_STOP=300 and DATA_STEP=30. These values have been restored in fpinn2.py. The historical results below used 180 measurements and do not establish a fix under the required constraint. Existing Outputs/fpinn2_002 files are from that historical run.

# Applied FPINN parameter fix

The current file starts at theta1 = -22.5 degrees and theta2 = +18 degrees. The applied settings fit this trajectory using 180 measured angle pairs at times 0, 0.1, ..., 17.9 seconds. The held-out region is 18–20.01 seconds. This is a changed training-data setup: the previous production settings used only 10 pairs over 0–2.7 seconds. This result does not demonstrate successful extrapolation from the original short training window or an epoch advantage over TPINN under identical data.

Only parameters and explanatory comments were edited in fpinn2.py. An AST comparison confirmed that all other statements, including model, initialization algorithm, loss equations, spectral derivatives, optimizer and training loop, are unchanged. TPINN and HPC scripts were not edited in this task.

## Parameters

```python
EPOCHS = 45000
FOURIER_PERIOD_FACTOR = 2
MAX_ANGULAR_FREQUENCY = 40.0
INITIALIZATION_RIDGE = 1e-2
WARMUP_EPOCHS = 1000
PHYSICS_RAMP_EPOCHS = 4000
DATA_STOP = 1800
DATA_STEP = 10
LEARNING_RATE_NETWORK = 2e-4
LEARNING_RATE_SPECTRUM = 2e-4
WEIGHT_DECAY = 1e-8
LAMBDA_DATA = 1000.0
LAMBDA_PHYSICS = 1000.0
LAMBDA_INITIAL = 500.0
LAMBDA_ENERGY = 0.0
# Existing MultiStepLR: milestones=[20000, 30000, 35000], gamma=0.3
```

## Validation

The isolated seed-0 benchmark at 45,000 optimizer updates obtained:

- Full-record R²: 0.999995321, 0.999995837.
- Held-out R²: 0.999967993, 0.999963810.
- Normalized physics residual: 3.1067964e-05.

The angle fit is excellent, but the physics residual remains above the earlier strict 1e-5 threshold. The benchmark correctly records `budget_exhausted`, not `converged`. This is a fit improvement, not fulfillment of that combined stopping criterion. Other initial conditions and random seeds are not validated for this configuration.

The production script is also run directly to regenerate its log, final PNG, loss PNG, training GIF and spectrum outputs in Outputs/fpinn2_002. Its original loop includes epoch zero, so EPOCHS=45000 performs 45,001 updates. run.log records that direct run; the final FPINN2.log contains the production metrics.

The previous script is preserved here as fpinn2_before.py. input.dat records the tested data, and fpinn2_applied.py records the applied source. benchmark_result.json contains all benchmark observations. The additional parameter screens are saved under parameter_tests/new_angles/fix*/.

The direct production run completed successfully in 67 seconds of CPU training. Its logged full-record R² values are 0.999995 and 0.999996; held-out values are 0.999968 and 0.999964. PNG and GIF generation also completed successfully.
