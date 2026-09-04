"""A5000 configuration for the double-pendulum time PINN.

This uses the same model, losses, stopping rule, and sparse measurements as
tpinn2.py. It raises an error when CUDA is unavailable so an HPC job cannot
silently run on a CPU node.
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

# Keep the same one-million-epoch ceiling and convergence target as fpinn2.py.
tpinn2.EPOCHS = 1_000_000
tpinn2.PRINT_EVERY = 1_000
tpinn2.EVALUATE_EVERY = 1_000
tpinn2.SNAPSHOT_EVERY = 10_000
tpinn2.HISTORY_EVERY = 100
tpinn2.GIF_FPS = 20

# The A5000 can evaluate a denser residual grid and a wider time network.
tpinn2.PHYSICS_POINTS = 2_048
tpinn2.NETWORK_WIDTH = 256
tpinn2.NETWORK_DEPTH = 4
tpinn2.LEARNING_RATE = 2e-4


if __name__ == "__main__":
    tpinn2.main()
