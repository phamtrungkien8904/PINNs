"""A5000 launcher for the first-order time-domain PINN in tpinn2.py.

The local and HPC variants share the model and stopping rule. This launcher
uses a wider network and denser physics grid, requires CUDA, enables A5000
optimizations, and writes independent outputs under Outputs/tpinn2_hpc.
"""

import tpinn2


tpinn2.RUN_NAME = "Double Pendulum Time PINN - A5000 HPC"
tpinn2.OUTPUT_DIR = tpinn2.SCRIPT_DIR / "Outputs/tpinn2_hpc"
tpinn2.OUTPUT_PREFIX = "tpinn2_hpc"
tpinn2.LOG_FILE = tpinn2.OUTPUT_DIR / "TPINN2_HPC.log"

tpinn2.REQUIRE_CUDA = True
tpinn2.GPU_INDEX = 0
tpinn2.USE_TF32 = True
tpinn2.USE_FUSED_ADAM = True
tpinn2.USE_TORCH_COMPILE = True

# The stopping rule normally ends training before this ceiling.
tpinn2.EPOCHS = 100_000
tpinn2.PRINT_EVERY = 1_000
tpinn2.EVALUATE_EVERY = 1_000
tpinn2.HISTORY_EVERY = 100
tpinn2.SNAPSHOT_EVERY = 5_000
tpinn2.GIF_FPS = 15

# Same 30 sparse observations as the Fourier benchmark.
tpinn2.DATA_STOP = 300
tpinn2.DATA_STEP = 10

# A5000-sized model and residual grid.
tpinn2.PHYSICS_POINTS = 4_096
tpinn2.NETWORK_WIDTH = 256
tpinn2.NETWORK_DEPTH = 5
tpinn2.FIRST_LAYER_OMEGA = 60.0

tpinn2.LEARNING_RATE = 2e-4
tpinn2.WEIGHT_DECAY = 1e-8
tpinn2.LR_MILESTONES = (50_000, 80_000, 95_000)
tpinn2.LR_DECAY = 0.3

tpinn2.WARMUP_EPOCHS = 2_000
tpinn2.PHYSICS_RAMP_EPOCHS = 10_000
tpinn2.PHYSICS_EXPANSION_EPOCHS = 30_000
tpinn2.LAMBDA_DATA = 1_000.0
tpinn2.LAMBDA_PHYSICS = 10.0

tpinn2.EARLY_STOP = True
tpinn2.EARLY_STOP_MIN_EPOCH = 35_000
tpinn2.EARLY_STOP_R2 = 0.999
tpinn2.EARLY_STOP_PHYSICS = 1e-5
tpinn2.EARLY_STOP_PATIENCE = 3


if __name__ == "__main__":
    tpinn2.main()
