"""Low-epoch Fourier PINN for an NVIDIA RTX A5000.

The model predicts Fourier spectra of the complete first-order state
[theta1, theta2, omega1, omega2]. It uses exact first spectral derivatives,
an exact initial-state constraint, causal time and bandwidth curricula, and
an AdamW-to-L-BFGS optimizer. Every L-BFGS closure is counted as an objective
evaluation so its epoch count remains comparable with tpinn2.py.

Outputs are written under Outputs/fpinn2_hpc.
"""

from pathlib import Path
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# A5000 configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "double_pendulum_data.dat"
OUTPUT_DIR = SCRIPT_DIR / "Outputs/fpinn2_hpc"
OUTPUT_PREFIX = "fpinn2_hpc"
LOG_FILE = OUTPUT_DIR / "FPINN2_HPC.log"
RUN_NAME = "Double Pendulum First-Order Fourier PINN - A5000 HPC"

SEED = 0
REQUIRE_CUDA = True
GPU_INDEX = 0
USE_TF32 = True
USE_FUSED_ADAM = True

MAX_OBJECTIVE_EVALUATIONS = 52_000
ADAM_EVALUATIONS = 35_000
LBFGS_INNER_ITERATIONS = 5
LBFGS_HISTORY_SIZE = 50
PRINT_EVERY = 1_000
EVALUATE_EVERY = 1_000
HISTORY_EVERY = 100
SNAPSHOT_EVERY = 5_000
GIF_FPS = 15

# Same 30 sparse angle measurements used by tpinn2.py (0--2.9 s).
DATA_STOP = 300
DATA_STEP = 10

# cuFFT length is rounded to a power of two in main().
FOURIER_PERIOD_FACTOR = 4
MAX_ANGULAR_FREQUENCY = 30.0
BANDWIDTH_STAGES = ((0, 10.0), (8_000, 20.0), (16_000, 30.0))
INITIALIZATION_RIDGE = 1e-2

NETWORK_WIDTH = 512
NETWORK_DEPTH = 4
NETWORK_SCALE = 1e-2
LEARNING_RATE_NETWORK = 2e-4
LEARNING_RATE_SPECTRUM = 5e-4
WEIGHT_DECAY = 1e-7
GRADIENT_CLIP = 1.0
LR_MILESTONES = (28_000, 33_000)
LR_DECAY = 0.3

WARMUP_EVALUATIONS = 2_000
PHYSICS_RAMP_EVALUATIONS = 5_000
PHYSICS_EXPANSION_EVALUATIONS = 20_000
LAMBDA_DATA_WARMUP = 1_000.0
LAMBDA_DATA_PHYSICS = 100.0
LAMBDA_PHYSICS = 100.0
LAMBDA_SPECTRAL = 1e-6
VELOCITY_SCALE = math.sqrt(10.0)
ACCELERATION_SCALE = 10.0

EARLY_STOP = True
EARLY_STOP_MIN_EVALUATIONS = 25_000
EARLY_STOP_R2 = 0.999
EARLY_STOP_EXTRAPOLATION_R2 = 0.999
EARLY_STOP_PHYSICS = 1e-5
EARLY_STOP_PATIENCE = 3

m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 10.0

plt.style.use("classic")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})


# -----------------------------------------------------------------------------
# Data and initialization
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
    omega = np.gradient(theta, t, axis=0, edge_order=2)
    return t, theta, omega


def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference, axis=0)) ** 2, axis=0)
    residual = np.sum((reference - prediction) ** 2, axis=0)
    return 1.0 - residual / total


def next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def estimate_initial_theta_spectrum(t_data, theta_data, frequencies):
    nonzero = frequencies[1:]
    design = np.column_stack((
        np.ones(len(t_data)),
        np.cos(t_data[:, None] * nonzero[None, :]),
        np.sin(t_data[:, None] * nonzero[None, :]),
    ))
    gram = design.T @ design
    regularizer = INITIALIZATION_RIDGE * np.eye(gram.shape[0])
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer,
        design.T @ theta_data,
    )
    mode_count = len(nonzero)
    spectrum = np.zeros((len(frequencies), 2), dtype=np.complex64)
    spectrum[0] = coefficients[0]
    cosine = coefficients[1 : 1 + mode_count]
    sine = coefficients[1 + mode_count :]
    spectrum[1:] = 0.5 * (cosine - 1j * sine)
    return spectrum


def make_initial_state_spectrum(t_data, theta_data, frequencies):
    theta_spectrum = estimate_initial_theta_spectrum(
        t_data, theta_data, frequencies
    )
    omega_spectrum = (
        1j * frequencies[:, None] * theta_spectrum
    ).astype(np.complex64)
    state_spectrum = np.column_stack((theta_spectrum, omega_spectrum))
    output = np.empty((len(frequencies), 8), dtype=np.float32)
    output[:, 0::2] = state_spectrum.real
    output[:, 1::2] = state_spectrum.imag
    return output


# -----------------------------------------------------------------------------
# First-order Fourier state network
# -----------------------------------------------------------------------------
class FourierStatePINN(nn.Module):
    def __init__(self, frequencies, n_fourier, initial_state, initial_spectrum):
        super().__init__()
        self.n_fourier = n_fourier
        self.register_buffer("frequencies", frequencies)
        self.register_buffer(
            "initial_state",
            torch.as_tensor(initial_state, dtype=torch.float32)[None, :],
        )

        layers = [nn.Linear(1, NETWORK_WIDTH), nn.Tanh()]
        for _ in range(NETWORK_DEPTH - 1):
            layers.extend((nn.Linear(NETWORK_WIDTH, NETWORK_WIDTH), nn.Tanh()))
        layers.append(nn.Linear(NETWORK_WIDTH, 8))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.spectral_parameters = nn.Parameter(
            torch.as_tensor(initial_spectrum, dtype=torch.float32)
        )

    def coefficients(self, maximum_frequency):
        omega_max = torch.clamp(self.frequencies[-1], min=1e-12)
        omega_input = 2.0 * self.frequencies[:, None] / omega_max - 1.0
        raw = self.spectral_parameters + NETWORK_SCALE * self.network(omega_input)
        real = raw[:, 0::2]
        imaginary = raw[:, 1::2]
        mask = (self.frequencies <= maximum_frequency).to(real.dtype)[:, None]

        real_nonzero = real[1:] * mask[1:]
        imaginary_nonzero = imaginary[1:] * mask[1:]
        # irfft(n * spectrum) at t=0 is DC + 2*sum(real modes).
        dc = self.initial_state - 2.0 * torch.sum(
            real_nonzero, dim=0, keepdim=True
        )
        real = torch.cat((dc, real_nonzero), dim=0)
        imaginary = torch.cat((
            torch.zeros_like(imaginary[:1]),
            imaginary_nonzero,
        ), dim=0)
        return torch.complex(real, imaginary)

    def reconstruct(self, maximum_frequency, n_physical):
        spectrum = self.coefficients(maximum_frequency)
        derivative_spectrum = 1j * self.frequencies[:, None] * spectrum
        # One batched cuFFT call reconstructs all states and derivatives.
        packed = torch.cat((spectrum, derivative_spectrum), dim=1)
        reconstructed = torch.fft.irfft(
            self.n_fourier * packed,
            n=self.n_fourier,
            dim=0,
        )[:n_physical]
        return reconstructed[:, :4], reconstructed[:, 4:], spectrum


# -----------------------------------------------------------------------------
# Physics and curricula
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
        - m2 * sin_delta * (l1 * omega1**2 * cos_delta + l2 * omega2**2)
        - (m1 + m2) * g * torch.sin(theta1)
    )
    denominator1 = l1 * (m1 + m2 * sin_delta**2)
    numerator2 = (
        (m1 + m2) * (
            l1 * omega1**2 * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
    )
    denominator2 = l2 * (m1 + m2 * sin_delta**2)
    return torch.cat((
        omega1,
        omega2,
        numerator1 / denominator1,
        numerator2 / denominator2,
    ), dim=1)


def current_bandwidth(evaluations):
    bandwidth = BANDWIDTH_STAGES[0][1]
    for start, value in BANDWIDTH_STAGES:
        if evaluations >= start:
            bandwidth = value
    return bandwidth


def current_physics_weight(evaluations):
    if evaluations < WARMUP_EVALUATIONS:
        return 0.0
    progress = min(
        1.0,
        (evaluations - WARMUP_EVALUATIONS + 1) / PHYSICS_RAMP_EVALUATIONS,
    )
    return LAMBDA_PHYSICS * progress**2


def current_data_weight(evaluations):
    if evaluations < WARMUP_EVALUATIONS:
        return LAMBDA_DATA_WARMUP
    return LAMBDA_DATA_PHYSICS


def current_physics_stop(evaluations, measured_stop, time_max):
    if evaluations < WARMUP_EVALUATIONS:
        return measured_stop
    progress = min(
        1.0,
        (evaluations - WARMUP_EVALUATIONS + 1)
        / PHYSICS_EXPANSION_EVALUATIONS,
    )
    return measured_stop + progress * (time_max - measured_stop)


def compute_losses(
    model,
    evaluations,
    time_physical,
    data_indices,
    theta_data,
    measured_stop,
    time_max,
):
    bandwidth = current_bandwidth(evaluations)
    state, derivative, spectrum = model.reconstruct(
        bandwidth, len(time_physical)
    )
    data_loss = torch.mean((state[data_indices, :2] - theta_data) ** 2)

    target = state_rhs(state)
    kinematic_error = (
        derivative[:, :2] - target[:, :2]
    ) / VELOCITY_SCALE
    dynamic_error = (
        derivative[:, 2:] - target[:, 2:]
    ) / ACCELERATION_SCALE
    point_loss = torch.mean(kinematic_error**2, dim=1) + torch.mean(
        dynamic_error**2, dim=1
    )
    physics_stop = current_physics_stop(evaluations, measured_stop, time_max)
    active = (time_physical <= physics_stop).to(point_loss.dtype)
    physics_loss = torch.sum(active * point_loss) / torch.clamp(
        torch.sum(active), min=1.0
    )

    normalized_frequency = model.frequencies / torch.clamp(
        model.frequencies[-1], min=1e-12
    )
    spectral_loss = torch.mean(
        normalized_frequency[:, None] ** 4 * torch.abs(spectrum) ** 2
    )
    total_loss = (
        current_data_weight(evaluations) * data_loss
        + current_physics_weight(evaluations) * physics_loss
        + LAMBDA_SPECTRAL * spectral_loss
    )
    return total_loss, data_loss, physics_loss, spectral_loss


# -----------------------------------------------------------------------------
# Evaluation and output
# -----------------------------------------------------------------------------
def evaluate(model, theta_reference, extrapolation_start, bandwidth=None):
    if bandwidth is None:
        bandwidth = MAX_ANGULAR_FREQUENCY
    model.eval()
    with torch.inference_mode():
        state, _, spectrum = model.reconstruct(bandwidth, len(theta_reference))
        prediction = state[:, :2].cpu().numpy()
        spectrum_magnitude = torch.abs(spectrum).cpu().numpy()
    model.train()
    r2_all = coefficient_of_determination(theta_reference, prediction)
    r2_extra = coefficient_of_determination(
        theta_reference[extrapolation_start:],
        prediction[extrapolation_start:],
    )
    return r2_all, r2_extra, prediction, spectrum_magnitude


def configure_prediction_axes(axes, t, theta, data_indices):
    colors = ("blue", "red")
    symbols = (r"$\theta_1$", r"$\theta_2$")
    measured_stop = t[min(DATA_STOP, len(t)) - 1]
    for component, axis in enumerate(axes):
        margin = 0.05 * np.ptp(theta[:, component])
        axis.plot(
            t, theta[:, component], color=colors[component], alpha=0.35,
            label=f"Numerical {symbols[component]}",
        )
        axis.plot(
            t[data_indices], theta[data_indices, component], "o",
            color=colors[component], markersize=3,
            label=f"Data {symbols[component]}",
        )
        axis.axvline(
            measured_stop, color="0.4", linestyle=":", linewidth=1,
            label="Prediction start" if component == 0 else None,
        )
        axis.set_ylabel("Angle (rad)")
        axis.set_ylim(
            np.min(theta[:, component]) - margin,
            np.max(theta[:, component]) + margin,
        )
        axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)")


def save_results(t, theta, data_indices, prediction):
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )
    configure_prediction_axes(axes, t, theta, data_indices)
    for component, axis in enumerate(axes):
        axis.plot(
            t, prediction[:, component], "--",
            color=("blue", "red")[component],
            label=rf"FPINN $\theta_{component + 1}$",
        )
        axis.legend(loc="upper right", ncol=3)
    axes[0].set_title("Double Pendulum First-Order Fourier PINN")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=300)
    plt.close(fig)


def save_loss(history):
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.semilogy(history["epoch"], history["total"], "k-", label="Total loss")
    axis.semilogy(history["epoch"], history["data"], "b-", label="Data loss")
    axis.semilogy(
        history["epoch"], history["physics"], "r-", label="Physics loss"
    )
    axis.set(
        xlabel="Objective evaluations", ylabel="Loss",
        title="FPINN loss convergence",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png", dpi=300)
    plt.close(fig)


def save_spectrum(frequencies, spectrum):
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.semilogy(
        frequencies, spectrum[:, 0] + 1e-14,
        color="blue", label=r"$|\Theta_1|$",
    )
    axis.semilogy(
        frequencies, spectrum[:, 1] + 1e-14,
        color="red", label=r"$|\Theta_2|$",
    )
    axis.set(
        xlabel="Angular frequency (rad/s)", ylabel="Magnitude",
        xlim=(0.0, MAX_ANGULAR_FREQUENCY),
        title="Learned FPINN angle spectrum",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_spectrum.png", dpi=300)
    plt.close(fig)


def save_training_animation(t, theta, data_indices, epochs, snapshots):
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
    )
    configure_prediction_axes(axes, t, theta, data_indices)
    lines = []
    for component, axis in enumerate(axes):
        line, = axis.plot(
            t, snapshots[0][:, component], "--",
            color=("blue", "red")[component],
            label=rf"FPINN $\theta_{component + 1}$",
        )
        lines.append(line)
        axis.legend(loc="upper right", ncol=3)
    title = axes[0].set_title("")

    def update(frame):
        for component, line in enumerate(lines):
            line.set_ydata(snapshots[frame][:, component])
        title.set_text(f"Fourier PINN - Evaluation {epochs[frame]}")
        return *lines, title

    movie = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True)
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_log(
    device, gpu_name, n_fourier, active_modes, evaluations,
    adam_completed, lbfgs_evaluations, runtime, losses,
    r2_all, r2_extra, reached_target,
):
    total_loss, data_loss, physics_loss, spectral_loss = losses
    lines = [
        f"Name: {RUN_NAME}",
        f"Using device: {device}",
        f"GPU: {gpu_name}",
        f"Data file: {DATA_FILE}",
        f"Data stop: {DATA_STOP}",
        f"Data step: {DATA_STEP}",
        f"Fourier length: {n_fourier}",
        f"Active modes: {active_modes}",
        f"Maximum angular frequency: {MAX_ANGULAR_FREQUENCY}",
        f"Network width: {NETWORK_WIDTH}",
        f"Network depth: {NETWORK_DEPTH}",
        f"Maximum objective evaluations: {MAX_OBJECTIVE_EVALUATIONS}",
        f"Objective evaluations: {evaluations}",
        f"Epoch: {evaluations}",
        f"Adam evaluations: {adam_completed}",
        f"L-BFGS evaluations: {lbfgs_evaluations}",
        f"Runtime: {runtime}",
        f"Loss: {total_loss:.6e}",
        f"Data loss: {data_loss:.6e}",
        f"Physics loss: {physics_loss:.6e}",
        f"Spectral loss: {spectral_loss:.6e}",
        f"R2 theta1: {r2_all[0]:.6f}",
        f"R2 theta2: {r2_all[1]:.6f}",
        f"R2 mean: {np.mean(r2_all):.6f}",
        f"R2 theta1 extrapolation: {r2_extra[0]:.6f}",
        f"R2 theta2 extrapolation: {r2_extra[1]:.6f}",
        f"R2 extrapolation mean: {np.mean(r2_extra):.6f}",
        f"Early-stop target reached: {reached_target}",
    ]
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def select_device():
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by fpinn2_hpc.py, but PyTorch cannot see a GPU. "
            "Check the CUDA PyTorch build and the HPC GPU allocation."
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
    return torch.device("cpu")


def create_adam_optimizer(model, device):
    groups = [
        {
            "params": model.network.parameters(),
            "lr": LEARNING_RATE_NETWORK,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [model.spectral_parameters],
            "lr": LEARNING_RATE_SPECTRUM,
            "weight_decay": 0.0,
        },
    ]
    try:
        optimizer = torch.optim.AdamW(
            groups, fused=USE_FUSED_ADAM and device.type == "cuda"
        )
        fused = USE_FUSED_ADAM and device.type == "cuda"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(groups)
        fused = False
    return optimizer, fused


def target_reached(evaluations, r2_all, r2_extra, physics_loss):
    return (
        EARLY_STOP
        and evaluations >= EARLY_STOP_MIN_EVALUATIONS
        and float(np.mean(r2_all)) >= EARLY_STOP_R2
        and float(np.mean(r2_extra)) >= EARLY_STOP_EXTRAPOLATION_R2
        and physics_loss <= EARLY_STOP_PHYSICS
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device()

    t, theta_reference, omega_reference = load_data(DATA_FILE)
    n_time = len(t)
    dt = float(np.mean(np.diff(t)))
    data_stop = min(DATA_STOP, n_time)
    data_indices_np = np.arange(0, data_stop, DATA_STEP)
    measured_stop = float(t[data_stop - 1])

    n_fourier = next_power_of_two(FOURIER_PERIOD_FACTOR * n_time)
    all_frequencies = 2.0 * np.pi * np.fft.rfftfreq(n_fourier, d=dt)
    active_modes = int(np.searchsorted(
        all_frequencies, MAX_ANGULAR_FREQUENCY, side="right"
    ))
    frequencies_np = all_frequencies[:active_modes]
    initial_spectrum = make_initial_state_spectrum(
        t[data_indices_np], theta_reference[data_indices_np], frequencies_np
    )
    initial_state = np.concatenate((theta_reference[0], omega_reference[0]))

    frequencies = torch.tensor(
        frequencies_np, dtype=torch.float32, device=device
    )
    model = FourierStatePINN(
        frequencies, n_fourier, initial_state, initial_spectrum
    ).to(device)
    time_physical = torch.tensor(t, dtype=torch.float32, device=device)
    data_indices = torch.tensor(
        data_indices_np, dtype=torch.long, device=device
    )
    theta_data = torch.tensor(
        theta_reference[data_indices_np], dtype=torch.float32, device=device
    )

    adam, fused_optimizer = create_adam_optimizer(model, device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        adam, milestones=list(LR_MILESTONES), gamma=LR_DECAY
    )
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "None"
    print(f"Name: {RUN_NAME}")
    print(f"Using device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Fourier length: {n_fourier}")
    print(f"Active Fourier modes: {active_modes}")
    print(f"Network: 1 -> {NETWORK_WIDTH} x {NETWORK_DEPTH} -> 8")
    print(f"Fused AdamW: {fused_optimizer}")
    print(f"Objective-evaluation budget: {MAX_OBJECTIVE_EVALUATIONS}")

    history = {"epoch": [], "total": [], "data": [], "physics": []}
    snapshot_epochs = []
    snapshots = []
    stable_checks = 0
    reached_target = False
    objective_evaluations = 0
    adam_completed = 0
    lbfgs_evaluations = 0
    latest_losses = (math.inf, math.inf, math.inf, math.inf)

    r2_all, r2_extra, prediction, _ = evaluate(
        model, theta_reference, data_stop, current_bandwidth(0)
    )
    snapshot_epochs.append(0)
    snapshots.append(prediction.copy())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    # AdamW learns the causal trajectory and progressively unlocks bandwidth.
    for _ in range(min(ADAM_EVALUATIONS, MAX_OBJECTIVE_EVALUATIONS)):
        adam.zero_grad(set_to_none=True)
        losses = compute_losses(
            model, objective_evaluations, time_physical, data_indices,
            theta_data, measured_stop, float(t[-1]),
        )
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        adam.step()
        scheduler.step()
        objective_evaluations += 1
        adam_completed += 1
        latest_losses = tuple(value.item() for value in losses)

        if objective_evaluations % HISTORY_EVERY == 0:
            history["epoch"].append(objective_evaluations)
            history["total"].append(latest_losses[0])
            history["data"].append(latest_losses[1])
            history["physics"].append(latest_losses[2])

        should_evaluate = objective_evaluations % EVALUATE_EVERY == 0
        should_snapshot = objective_evaluations % SNAPSHOT_EVERY == 0
        if should_evaluate or should_snapshot:
            r2_all, r2_extra, prediction, _ = evaluate(
                model, theta_reference, data_stop,
                current_bandwidth(objective_evaluations),
            )
            if should_snapshot:
                snapshot_epochs.append(objective_evaluations)
                snapshots.append(prediction.copy())
            if should_evaluate:
                stable_checks = (
                    stable_checks + 1
                    if target_reached(
                        objective_evaluations, r2_all, r2_extra,
                        latest_losses[2],
                    )
                    else 0
                )
                if objective_evaluations % PRINT_EVERY == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = format_time(time.perf_counter() - start_time)
                    print(
                        f"Evaluation {objective_evaluations:6d} | "
                        f"Loss {latest_losses[0]:.3e} | "
                        f"R2 {np.mean(r2_all):.6f} | "
                        f"Bandwidth {current_bandwidth(objective_evaluations):.0f} | "
                        f"Time {elapsed}"
                    )
                if stable_checks >= EARLY_STOP_PATIENCE:
                    reached_target = True
                    break

    # L-BFGS sharpens the full-bandwidth solution. Closure calls are counted.
    if not reached_target and objective_evaluations < MAX_OBJECTIVE_EVALUATIONS:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=0.8,
            max_iter=LBFGS_INNER_ITERATIONS,
            max_eval=LBFGS_INNER_ITERATIONS + 1,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=LBFGS_HISTORY_SIZE,
            line_search_fn="strong_wolfe",
        )
        next_evaluation = (
            objective_evaluations // EVALUATE_EVERY + 1
        ) * EVALUATE_EVERY
        next_history = (
            objective_evaluations // HISTORY_EVERY + 1
        ) * HISTORY_EVERY
        next_snapshot = (
            objective_evaluations // SNAPSHOT_EVERY + 1
        ) * SNAPSHOT_EVERY

        while (
            objective_evaluations + LBFGS_INNER_ITERATIONS + 1
            <= MAX_OBJECTIVE_EVALUATIONS
        ):
            closure_losses = {}

            def closure():
                nonlocal objective_evaluations, lbfgs_evaluations
                lbfgs.zero_grad(set_to_none=True)
                values = compute_losses(
                    model, objective_evaluations, time_physical, data_indices,
                    theta_data, measured_stop, float(t[-1]),
                )
                values[0].backward()
                objective_evaluations += 1
                lbfgs_evaluations += 1
                closure_losses["values"] = tuple(
                    value.detach().item() for value in values
                )
                return values[0]

            lbfgs.step(closure)
            if "values" in closure_losses:
                latest_losses = closure_losses["values"]

            if objective_evaluations >= next_history:
                history["epoch"].append(objective_evaluations)
                history["total"].append(latest_losses[0])
                history["data"].append(latest_losses[1])
                history["physics"].append(latest_losses[2])
                next_history += HISTORY_EVERY

            should_evaluate = objective_evaluations >= next_evaluation
            should_snapshot = objective_evaluations >= next_snapshot
            if should_evaluate or should_snapshot:
                r2_all, r2_extra, prediction, _ = evaluate(
                    model, theta_reference, data_stop
                )
                if should_snapshot:
                    snapshot_epochs.append(objective_evaluations)
                    snapshots.append(prediction.copy())
                    next_snapshot += SNAPSHOT_EVERY
                if should_evaluate:
                    stable_checks = (
                        stable_checks + 1
                        if target_reached(
                            objective_evaluations, r2_all, r2_extra,
                            latest_losses[2],
                        )
                        else 0
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = format_time(time.perf_counter() - start_time)
                    print(
                        f"Evaluation {objective_evaluations:6d} | L-BFGS | "
                        f"Loss {latest_losses[0]:.3e} | "
                        f"R2 {np.mean(r2_all):.6f} | Time {elapsed}"
                    )
                    next_evaluation += EVALUATE_EVERY
                    if stable_checks >= EARLY_STOP_PATIENCE:
                        reached_target = True
                        break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = format_time(time.perf_counter() - start_time)

    final_losses = compute_losses(
        model, objective_evaluations, time_physical, data_indices,
        theta_data, measured_stop, float(t[-1]),
    )
    latest_losses = tuple(value.detach().item() for value in final_losses)
    r2_all, r2_extra, prediction, spectrum = evaluate(
        model, theta_reference, data_stop
    )
    reached_target = reached_target or target_reached(
        objective_evaluations, r2_all, r2_extra, latest_losses[2]
    )
    if snapshot_epochs[-1] != objective_evaluations:
        snapshot_epochs.append(objective_evaluations)
        snapshots.append(prediction.copy())
    if not history["epoch"] or history["epoch"][-1] != objective_evaluations:
        history["epoch"].append(objective_evaluations)
        history["total"].append(latest_losses[0])
        history["data"].append(latest_losses[1])
        history["physics"].append(latest_losses[2])

    print("\nFPINN result")
    print(f"Objective evaluations: {objective_evaluations}")
    print(f"Runtime: {runtime}")
    print(f"Loss: {latest_losses[0]:.6e}")
    print(f"Physics loss: {latest_losses[2]:.6e}")
    print(f"R2 theta1: {r2_all[0]:.6f}")
    print(f"R2 theta2: {r2_all[1]:.6f}")
    print(f"R2 mean: {np.mean(r2_all):.6f}")
    print(f"R2 extrapolation mean: {np.mean(r2_extra):.6f}")
    print(f"Early-stop target reached: {reached_target}")

    save_log(
        device, gpu_name, n_fourier, active_modes, objective_evaluations,
        adam_completed, lbfgs_evaluations, runtime, latest_losses,
        r2_all, r2_extra, reached_target,
    )
    save_results(t, theta_reference, data_indices_np, prediction)
    save_loss(history)
    save_spectrum(frequencies_np, spectrum)
    save_training_animation(
        t, theta_reference, data_indices_np, snapshot_epochs, snapshots
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
