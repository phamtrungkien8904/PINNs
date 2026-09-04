"""Time-domain PINN for the double-pendulum trajectory.

The network predicts the first-order state
    [theta1, theta2, omega1, omega2]
from time. The initial state is imposed exactly, and the ODE residual is
evaluated with a differentiable fourth-order finite-difference stencil.
This avoids the expensive and poorly conditioned second-order autograd graph
in tpinn2_ver2.py.

The run writes a text log, the final prediction, the loss history, and a
training animation under Outputs/tpinn2.
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

plt.style.use("classic")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
    }
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "double_pendulum_data.dat"
OUTPUT_DIR = SCRIPT_DIR / "Outputs/tpinn2"
OUTPUT_PREFIX = "tpinn2"
LOG_FILE = OUTPUT_DIR / "TPINN2.log"

RUN_NAME = "Double Pendulum Time PINN"
SEED = 0
EPOCHS = 100_000
PRINT_EVERY = 1_000
EVALUATE_EVERY = 1_000
SNAPSHOT_EVERY = 1_000
HISTORY_EVERY = 100
GIF_FPS = 20

# Use the same sparse-data experiment as fpinn2.py: 30 samples over 0-2.9 s.
DATA_STOP = 300
DATA_STEP = 10

# A 512-point grid resolves the highest relevant trajectory frequency while
# keeping the CPU comparison inexpensive.
PHYSICS_POINTS = 512
NETWORK_WIDTH = 128
NETWORK_DEPTH = 4
FIRST_LAYER_OMEGA = 60.0
STATE_SCALE = (0.6, 0.6, 2.0, 2.0)
IC_TIME_SCALE = 1.0

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-8
GRADIENT_CLIP = 1.0
LR_MILESTONES = (50_000, 80_000, 95_000)
LR_DECAY = 0.3

WARMUP_EPOCHS = 2_000
PHYSICS_RAMP_EPOCHS = 10_000
PHYSICS_EXPANSION_EPOCHS = 30_000

LAMBDA_DATA = 1_000.0
LAMBDA_PHYSICS = 10.0
VELOCITY_SCALE = np.sqrt(10.0)
ACCELERATION_SCALE = 10.0

# Stop when a stable, physics-consistent fit is reached.
EARLY_STOP = True
EARLY_STOP_MIN_EPOCH = 35_000
EARLY_STOP_R2 = 0.999
EARLY_STOP_PHYSICS = 1e-5
EARLY_STOP_PATIENCE = 3

CPU_THREADS = 4
REQUIRE_CUDA = False
GPU_INDEX = 0
USE_TF32 = True
USE_FUSED_ADAM = True
USE_TORCH_COMPILE = False

# Double-pendulum parameters, matching fpinn2.py and tpinn2_ver2.py.
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 10.0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_data(path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Expected columns: time, theta1, theta2.")

    t = data[:, 0]
    theta = data[:, 1:3]
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("Time values must be strictly increasing.")

    if data.shape[1] >= 5:
        omega = data[:, 3:5]
    else:
        omega = np.gradient(theta, t, axis=0, edge_order=2)
    return t, theta, omega


def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference, axis=0)) ** 2, axis=0)
    residual = np.sum((reference - prediction) ** 2, axis=0)
    return 1.0 - residual / total


# -----------------------------------------------------------------------------
# Time-domain state network
# -----------------------------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega0=1.0, first=False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / omega0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class DoublePendulumStatePINN(nn.Module):
    """Map time to [theta1, theta2, omega1, omega2] with exact initial state."""

    def __init__(self, time_min, time_max, initial_state):
        super().__init__()
        self.register_buffer("time_min", torch.tensor(float(time_min)))
        self.register_buffer("time_span", torch.tensor(float(time_max - time_min)))
        self.register_buffer(
            "initial_state",
            torch.tensor(initial_state, dtype=torch.float32)[None, :],
        )
        self.register_buffer(
            "state_scale",
            torch.tensor(STATE_SCALE, dtype=torch.float32)[None, :],
        )

        layers = [
            SineLayer(
                1,
                NETWORK_WIDTH,
                omega0=FIRST_LAYER_OMEGA,
                first=True,
            )
        ]
        for _ in range(NETWORK_DEPTH - 1):
            layers.append(SineLayer(NETWORK_WIDTH, NETWORK_WIDTH))

        final_layer = nn.Linear(NETWORK_WIDTH, 4)
        with torch.no_grad():
            bound = np.sqrt(6.0 / NETWORK_WIDTH)
            final_layer.weight.uniform_(-bound, bound)
            final_layer.bias.zero_()
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        normalized_time = 2.0 * (t - self.time_min) / self.time_span - 1.0
        initial_envelope = 1.0 - torch.exp(
            -(t - self.time_min) / IC_TIME_SCALE
        )
        correction = self.state_scale * self.network(normalized_time)
        return self.initial_state + initial_envelope * correction


# -----------------------------------------------------------------------------
# Double-pendulum physics
# -----------------------------------------------------------------------------
def state_rhs(state):
    theta1 = state[:, 0:1]
    theta2 = state[:, 1:2]
    omega1 = state[:, 2:3]
    omega2 = state[:, 3:4]

    delta = theta1 - theta2
    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)

    numerator1 = (
        m2 * g * torch.sin(theta2) * cos_delta
        - m2
        * sin_delta
        * (l1 * omega1**2 * cos_delta + l2 * omega2**2)
        - (m1 + m2) * g * torch.sin(theta1)
    )
    denominator1 = l1 * (m1 + m2 * sin_delta**2)

    numerator2 = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
    )
    denominator2 = l2 * (m1 + m2 * sin_delta**2)

    acceleration1 = numerator1 / denominator1
    acceleration2 = numerator2 / denominator2
    return torch.cat((omega1, omega2, acceleration1, acceleration2), dim=1)


def fourth_order_derivative(values, step):
    """Fourth-order centered derivative at grid indices 2:-2."""
    return (
        values[:-4]
        - 8.0 * values[1:-3]
        + 8.0 * values[3:-1]
        - values[4:]
    ) / (12.0 * step)


def current_physics_weight(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.0
    progress = min(
        1.0,
        (epoch - WARMUP_EPOCHS + 1) / PHYSICS_RAMP_EPOCHS,
    )
    return LAMBDA_PHYSICS * progress**2


def current_physics_stop(epoch, measured_stop, time_max):
    if epoch < WARMUP_EPOCHS:
        return measured_stop
    progress = min(
        1.0,
        (epoch - WARMUP_EPOCHS + 1) / PHYSICS_EXPANSION_EPOCHS,
    )
    return measured_stop + progress * (time_max - measured_stop)


def physics_loss_on_grid(model, time_grid, physics_stop):
    state = model(time_grid)
    step = time_grid[1] - time_grid[0]
    derivative = fourth_order_derivative(state, step)
    state_center = state[2:-2]
    target = state_rhs(state_center)

    kinematic_error = (
        derivative[:, :2] - target[:, :2]
    ) / VELOCITY_SCALE
    dynamic_error = (
        derivative[:, 2:] - target[:, 2:]
    ) / ACCELERATION_SCALE

    point_loss = torch.mean(kinematic_error**2, dim=1) + torch.mean(
        dynamic_error**2, dim=1
    )
    active = (time_grid[2:-2, 0] <= physics_stop).to(point_loss.dtype)
    return torch.sum(active * point_loss) / torch.clamp(torch.sum(active), min=1.0)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_log(
    device,
    epoch,
    runtime,
    total_loss,
    data_loss,
    physics_loss,
    r2_all,
    r2_extra,
):
    gpu_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "None"
    )
    lines = [
        f"Name: {RUN_NAME}",
        f"Using device: {device}",
        f"GPU: {gpu_name}",
        f"CPU threads: {torch.get_num_threads()}",
        f"Data file: {DATA_FILE}",
        f"Data stop: {DATA_STOP}",
        f"Data step: {DATA_STEP}",
        f"Physics points: {PHYSICS_POINTS}",
        f"Network width: {NETWORK_WIDTH}",
        f"Network depth: {NETWORK_DEPTH}",
        f"Learning rate: {LEARNING_RATE}",
        f"Lambda data: {LAMBDA_DATA}",
        f"Lambda physics: {LAMBDA_PHYSICS}",
        f"Maximum epochs: {EPOCHS}",
        f"Epoch: {epoch}",
        f"Runtime: {runtime}",
        f"Loss: {total_loss:.6e}",
        f"Data loss: {data_loss:.6e}",
        f"Physics loss: {physics_loss:.6e}",
        f"R2 theta1: {r2_all[0]:.6f}",
        f"R2 theta2: {r2_all[1]:.6f}",
        f"R2 mean: {np.mean(r2_all):.6f}",
        f"R2 theta1 extrapolation: {r2_extra[0]:.6f}",
        f"R2 theta2 extrapolation: {r2_extra[1]:.6f}",
        f"R2 extrapolation mean: {np.mean(r2_extra):.6f}",
    ]
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_prediction_axes(axes, time_reference, theta_reference, data_indices):
    labels = (r"$\theta_1$", r"$\theta_2$")
    colors = ("blue", "red")
    measured_stop = time_reference[min(DATA_STOP, len(time_reference)) - 1]
    margin = 0.05 * (np.max(theta_reference) - np.min(theta_reference))

    for component, axis in enumerate(axes):
        axis.plot(
            time_reference,
            theta_reference[:, component],
            color=colors[component],
            alpha=0.35,
            label=f"Numerical {labels[component]}",
        )
        axis.plot(
            time_reference[data_indices],
            theta_reference[data_indices, component],
            "o",
            color=colors[component],
            markersize=3,
            label=f"Data {labels[component]}",
        )
        axis.axvline(
            measured_stop,
            color="0.4",
            linestyle=":",
            linewidth=1,
            label="Prediction start" if component == 0 else None,
        )
        axis.set_ylabel("Angle (rad)")
        axis.set_ylim(
            np.min(theta_reference[:, component]) - margin,
            np.max(theta_reference[:, component]) + margin,
        )
        axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)")


def save_results(time_reference, theta_reference, data_indices, prediction):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    configure_prediction_axes(axes, time_reference, theta_reference, data_indices)
    colors = ("blue", "red")
    labels = (r"TPINN $\theta_1$", r"TPINN $\theta_2$")
    for component, axis in enumerate(axes):
        axis.plot(
            time_reference,
            prediction[:, component],
            "--",
            color=colors[component],
            label=labels[component],
        )
        axis.legend(loc="upper right", ncol=3)
    axes[0].set_title("Double Pendulum Time PINN")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=300)
    plt.close(fig)


def save_loss(history):
    epochs = np.asarray(history["epoch"])
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.semilogy(epochs, history["total"], color="black", label="Total loss")
    axis.semilogy(epochs, history["data"], color="blue", label="Data loss")
    axis.semilogy(epochs, history["physics"], color="red", label="Physics loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="TPINN loss convergence")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png", dpi=300)
    plt.close(fig)


def save_training_animation(
    time_reference,
    theta_reference,
    data_indices,
    snapshot_epochs,
    snapshots,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    configure_prediction_axes(axes, time_reference, theta_reference, data_indices)
    colors = ("blue", "red")
    lines = []
    for component, axis in enumerate(axes):
        line, = axis.plot(
            time_reference,
            snapshots[0][:, component],
            "--",
            color=colors[component],
            label=rf"TPINN $\theta_{component + 1}$",
        )
        lines.append(line)
        axis.legend(loc="upper right", ncol=3)
    title = axes[0].set_title("")

    def update(frame):
        for component, line in enumerate(lines):
            line.set_ydata(snapshots[frame][:, component])
        title.set_text(
            f"Double Pendulum Time PINN - Epoch {snapshot_epochs[frame]}"
        )
        return *lines, title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        blit=True,
    )
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def select_device():
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required, but PyTorch cannot see a GPU. Check the CUDA "
            "PyTorch build and the HPC GPU allocation."
        )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_INDEX}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(SEED)
        if USE_TF32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    torch.set_num_threads(CPU_THREADS)
    return torch.device("cpu")


def create_optimizer(model, device):
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            fused=USE_FUSED_ADAM and device.type == "cuda",
        )
        fused = USE_FUSED_ADAM and device.type == "cuda"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        fused = False
    return optimizer, fused


def evaluate(model, time_tensor, theta_reference, extrapolation_start):
    model.eval()
    with torch.inference_mode():
        theta_prediction = model(time_tensor)[:, :2].cpu().numpy()
    model.train()

    r2_all = coefficient_of_determination(theta_reference, theta_prediction)
    r2_extra = coefficient_of_determination(
        theta_reference[extrapolation_start:],
        theta_prediction[extrapolation_start:],
    )
    return r2_all, r2_extra, theta_prediction


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_reference, theta_reference, omega_reference = load_data(DATA_FILE)
    data_stop = min(DATA_STOP, len(t_reference))
    data_indices = np.arange(0, data_stop, DATA_STEP)
    measured_stop = float(t_reference[data_stop - 1])

    initial_state = np.concatenate((theta_reference[0], omega_reference[0]))
    model = DoublePendulumStatePINN(
        t_reference[0],
        t_reference[-1],
        initial_state,
    ).to(device)

    optimizer, fused_optimizer = create_optimizer(model, device)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
        except Exception:
            compiled = False
    else:
        compiled = False

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(LR_MILESTONES),
        gamma=LR_DECAY,
    )

    time_data = torch.tensor(
        t_reference[data_indices], dtype=torch.float32, device=device
    )[:, None]
    theta_data = torch.tensor(
        theta_reference[data_indices], dtype=torch.float32, device=device
    )
    time_physics = torch.linspace(
        float(t_reference[0]),
        float(t_reference[-1]),
        PHYSICS_POINTS,
        dtype=torch.float32,
        device=device,
    )[:, None]
    time_evaluation = torch.tensor(
        t_reference, dtype=torch.float32, device=device
    )[:, None]

    print(f"Name: {RUN_NAME}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {properties.total_memory / 2**30:.1f} GiB")
    else:
        print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Physics points: {PHYSICS_POINTS}")
    print(f"Network: 1 -> {NETWORK_WIDTH} x {NETWORK_DEPTH} -> 4")
    print(f"Fused AdamW: {fused_optimizer}")
    print(f"torch.compile: {compiled}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    stable_checks = 0
    last_r2_all = np.array([-np.inf, -np.inf])
    last_r2_extra = np.array([-np.inf, -np.inf])
    history = {"epoch": [], "total": [], "data": [], "physics": []}
    snapshot_epochs = []
    snapshots = []

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)

        theta_at_data = model(time_data)[:, :2]
        data_loss = torch.mean((theta_at_data - theta_data) ** 2)

        physics_stop = current_physics_stop(
            epoch,
            measured_stop,
            float(t_reference[-1]),
        )
        physics_loss = physics_loss_on_grid(
            model,
            time_physics,
            physics_stop,
        )
        physics_weight = current_physics_weight(epoch)
        total_loss = (
            LAMBDA_DATA * data_loss
            + physics_weight * physics_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()

        if epoch % HISTORY_EVERY == 0:
            history["epoch"].append(epoch)
            history["total"].append(total_loss.item())
            history["data"].append(data_loss.item())
            history["physics"].append(physics_loss.item())

        should_evaluate = (
            epoch % EVALUATE_EVERY == 0
            or epoch % SNAPSHOT_EVERY == 0
        )
        if should_evaluate:
            last_r2_all, last_r2_extra, prediction_now = evaluate(
                model,
                time_evaluation,
                theta_reference,
                data_stop,
            )
            if epoch % SNAPSHOT_EVERY == 0:
                snapshot_epochs.append(epoch)
                snapshots.append(prediction_now.copy())
            physics_value = physics_loss.item()
            r2_mean = float(np.mean(last_r2_all))

            if (
                EARLY_STOP
                and epoch >= EARLY_STOP_MIN_EPOCH
                and r2_mean >= EARLY_STOP_R2
                and physics_value <= EARLY_STOP_PHYSICS
            ):
                stable_checks += 1
            else:
                stable_checks = 0

            if epoch % PRINT_EVERY == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = format_time(time.perf_counter() - start_time)
                print(
                    f"\rEpoch {epoch:7d} | Loss {total_loss.item():.3e} | "
                    f"R2 {r2_mean:.6f} | Time {elapsed}",
                    end="",
                    flush=True
                )

            if stable_checks >= EARLY_STOP_PATIENCE:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - start_time
    runtime = format_time(runtime_seconds)

    last_r2_all, last_r2_extra, prediction_final = evaluate(
        model,
        time_evaluation,
        theta_reference,
        data_stop,
    )
    if not snapshot_epochs or snapshot_epochs[-1] != epoch:
        snapshot_epochs.append(epoch)
        snapshots.append(prediction_final.copy())
    if not history["epoch"] or history["epoch"][-1] != epoch:
        history["epoch"].append(epoch)
        history["total"].append(total_loss.item())
        history["data"].append(data_loss.item())
        history["physics"].append(physics_loss.item())
    r2_mean = float(np.mean(last_r2_all))
    r2_extra_mean = float(np.mean(last_r2_extra))

    print("\nTPINN result")
    print(f"Epoch: {epoch}")
    print(f"Runtime: {runtime}")
    print(f"Loss: {total_loss.item():.6e}")
    print(f"R2 theta1: {last_r2_all[0]:.6f}")
    print(f"R2 theta2: {last_r2_all[1]:.6f}")
    print(f"R2 mean: {r2_mean:.6f}")
    print(f"R2 extrapolation mean: {r2_extra_mean:.6f}")

    save_log(
        device,
        epoch,
        runtime,
        total_loss.item(),
        data_loss.item(),
        physics_loss.item(),
        last_r2_all,
        last_r2_extra,
    )
    save_results(
        t_reference,
        theta_reference,
        data_indices,
        prediction_final,
    )
    save_loss(history)
    save_training_animation(
        t_reference,
        theta_reference,
        data_indices,
        snapshot_epochs,
        snapshots,
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
