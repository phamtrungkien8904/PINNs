from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
plt.style.use("classic")
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \usepackage[utf8]{inputenc}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{siunitx}
        \usepackage{sfmath}
        """,
        "figure.dpi": 300,
        "figure.figsize": (10 / 2.54, 6 / 2.54),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.labelcolor": "black",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "figure.constrained_layout.use": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.minor.width": 0,
        "ytick.minor.width": 0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.title_fontsize": 8,
        "legend.fontsize": 8,
        "legend.handlelength": 2,
        "legend.loc": "best",
        "legend.numpoints": 1,
        "lines.linewidth": 1,
        "lines.markersize": 4,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.5,
    }
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_FILE = Path("double_pendulum_data.dat")
OUTPUT_DIR = Path("./Outputs/fpinn2")
OUTPUT_PREFIX = "fpinn2"
LOG_FILE = OUTPUT_DIR / f"{OUTPUT_PREFIX}.log"

SEED = 0
EPOCHS = 200_000
SNAPSHOT_EVERY = 1_000
PRINT_EVERY = 100

# ------------------------------------------------------------------
# Fourier-domain settings
#
# IMPORTANT:
# The physical trajectory is only evaluated over its real time interval,
# but the Fourier representation is built on a longer padded period.
# This greatly reduces the artificial condition theta(0) = theta(T_phys).
# ------------------------------------------------------------------
FOURIER_PERIOD_FACTOR = 4
MAX_ANGULAR_FREQUENCY = 12.0

INITIALIZATION_OMEGA_MAX = 5.0
INITIALIZATION_RIDGE = 1e-2

# Physics is only evaluated inside the physical time interval.  A small
# endpoint margin is retained because Fourier differentiation is most
# sensitive near the boundaries.
PHYSICS_MARGIN_FRACTION = 0.05

# ------------------------------------------------------------------
# Training schedule
# ------------------------------------------------------------------
WARMUP_EPOCHS = 20_000
PHYSICS_RAMP_EPOCHS = 40_000

DATA_STOP = 500
DATA_STEP = 10

# Conservative learning rates are important because spectral coefficients
# directly affect the full reconstructed trajectory.
LEARNING_RATE_NETWORK = 5e-4
LEARNING_RATE_SPECTRUM = 2e-4

# Weighted losses
LAMBDA_DATA = 1e3
LAMBDA_PHYSICS = 5.0
LAMBDA_INITIAL = 5e2
LAMBDA_ENERGY = 0.0

GRADIENT_CLIP = 10.0

SPECTRUM_XMAX = 12.0
SPECTRUM_YMAX = None
GIF_FPS = 30

# Double-pendulum parameters
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 10.0


torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_data(path):
    """Load a uniformly sampled double-pendulum trajectory.

    Accepted columns:
        t, theta1, theta2
    or
        t, theta1, theta2, omega1, omega2
    """
    data = np.loadtxt(path, skiprows=1)

    t = data[:, 0]
    theta1 = data[:, 1]
    theta2 = data[:, 2]

    dt_all = np.diff(t)
    dt = float(np.mean(dt_all))

    if not np.allclose(dt_all, dt, rtol=1e-5, atol=1e-8):
        raise ValueError("Input time grid must be uniformly sampled.")

    if data.shape[1] >= 5:
        omega1 = data[:, 3]
        omega2 = data[:, 4]
    else:
        omega1 = np.gradient(theta1, t, edge_order=2)
        omega2 = np.gradient(theta2, t, edge_order=2)

    return t, theta1, theta2, omega1, omega2, dt


def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference)) ** 2)
    residual = np.sum((reference - prediction) ** 2)
    if total <= 1e-16:
        return float("nan")
    return 1.0 - residual / total


def estimate_initial_spectrum(t_data, theta_data, frequencies):
    """Ridge-fit a small set of low-frequency Fourier modes.

    We deliberately keep INITIALIZATION_OMEGA_MAX low.  With sparse data,
    trying to initialize too many Fourier coefficients makes the ridge
    regression underdetermined and noisy.
    """
    frequencies = np.asarray(frequencies)
    use = frequencies <= INITIALIZATION_OMEGA_MAX
    used_frequencies = frequencies[use]

    columns = [np.ones_like(t_data)]

    for w in used_frequencies[1:]:
        columns.append(np.cos(w * t_data))
        columns.append(np.sin(w * t_data))

    design = np.column_stack(columns)

    gram = design.T @ design
    penalty = INITIALIZATION_RIDGE * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(
        gram + penalty,
        design.T @ theta_data,
    )

    spectrum = np.zeros(len(frequencies), dtype=np.complex128)
    spectrum[0] = beta[0]

    j = 1
    for k in range(1, len(used_frequencies)):
        cosine = beta[j]
        sine = beta[j + 1]

        spectrum[k] = cosine / 2.0 - 1j * sine / 2.0
        j += 2

    return spectrum


# -----------------------------------------------------------------------------
# Fourier neural network
# -----------------------------------------------------------------------------
class DoubleFourierPINN(nn.Module):
    """Map angular frequency -> complex spectra [Theta1, Theta2].

    The directly trainable spectral coefficients do most of the detailed
    fitting, while the MLP adds a smooth frequency-dependent correction.
    """

    def __init__(self, frequencies, n_fourier, initial_parameters):
        super().__init__()

        self.register_buffer("frequencies", frequencies)
        self.n_fourier = n_fourier

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
        )

        initial_spectrum = torch.zeros(
            len(frequencies),
            4,
            dtype=torch.float32,
        )

        for angle_index, coefficients in enumerate(initial_parameters):
            coefficients = np.asarray(coefficients)

            real_col = 2 * angle_index
            imag_col = real_col + 1

            initial_spectrum[:, real_col] = torch.tensor(
                coefficients.real,
                dtype=torch.float32,
            )
            initial_spectrum[:, imag_col] = torch.tensor(
                coefficients.imag,
                dtype=torch.float32,
            )

        self.spectral_coefficients = nn.Parameter(initial_spectrum)

        # Start exactly from the ridge estimate.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

        self.network_scale = 1e-2

    def forward(self, omega):
        omega_max = torch.clamp(self.frequencies[-1], min=1e-12)
        omega_scaled = 2.0 * omega / omega_max - 1.0

        correction = self.network_scale * self.network(omega_scaled)
        output = self.spectral_coefficients + correction

        real = torch.stack(
            (output[:, 0], output[:, 2]),
            dim=1,
        )
        imag = torch.stack(
            (output[:, 1], output[:, 3]),
            dim=1,
        )

        # DC coefficient must be real.
        imaginary_mask = torch.ones(
            len(output),
            device=output.device,
            dtype=output.dtype,
        )
        imaginary_mask[0] = 0.0

        return torch.complex(
            real,
            imag * imaginary_mask[:, None],
        )


# -----------------------------------------------------------------------------
# Fourier reconstruction
# -----------------------------------------------------------------------------
def reconstruct(
    model,
    omega_input,
    total_modes,
    n_fourier,
    n_physical,
):
    """Reconstruct padded periodic trajectory and return physical interval."""

    theta_active = model(omega_input)

    theta_fourier = torch.cat(
        (
            theta_active,
            torch.zeros(
                total_modes - len(theta_active),
                2,
                dtype=theta_active.dtype,
                device=theta_active.device,
            ),
        ),
        dim=0,
    )

    # Padded Fourier-period signal.
    theta_extended = torch.fft.irfft(
        n_fourier * theta_fourier,
        n=n_fourier,
        dim=0,
    )

    # Only the real physical time interval participates in data/physics losses.
    theta_physical = theta_extended[:n_physical]

    return theta_fourier, theta_extended, theta_physical


def spectral_derivatives(
    theta_fourier,
    omega,
    n_fourier,
    n_physical,
):
    """Compute velocity and acceleration by exact spectral differentiation."""

    omega_column = omega[:, None]

    velocity_fourier = 1j * omega_column * theta_fourier
    acceleration_fourier = -(omega_column**2) * theta_fourier

    velocity_extended = torch.fft.irfft(
        n_fourier * velocity_fourier,
        n=n_fourier,
        dim=0,
    )

    acceleration_extended = torch.fft.irfft(
        n_fourier * acceleration_fourier,
        n=n_fourier,
        dim=0,
    )

    velocity = velocity_extended[:n_physical]
    acceleration = acceleration_extended[:n_physical]

    return velocity, acceleration


# -----------------------------------------------------------------------------
# Double-pendulum mechanics
# -----------------------------------------------------------------------------
def mechanics(theta1, theta2, omega1, omega2):
    y1 = -l1 * torch.cos(theta1)
    y2 = y1 - l2 * torch.cos(theta2)

    vx1 = l1 * torch.cos(theta1) * omega1
    vy1 = l1 * torch.sin(theta1) * omega1

    vx2 = vx1 + l2 * torch.cos(theta2) * omega2
    vy2 = vy1 + l2 * torch.sin(theta2) * omega2

    kinetic = (
        0.5 * m1 * (vx1**2 + vy1**2)
        + 0.5 * m2 * (vx2**2 + vy2**2)
    )

    potential = m1 * g * y1 + m2 * g * y2

    lagrangian = kinetic - potential
    energy = kinetic + potential

    return kinetic, potential, lagrangian, energy


def explicit_physics_residuals(theta, velocity, acceleration):
    """Explicit double-pendulum ODE residuals."""

    theta1 = theta[:, 0]
    theta2 = theta[:, 1]

    omega1 = velocity[:, 0]
    omega2 = velocity[:, 1]

    gamma1 = acceleration[:, 0]
    gamma2 = acceleration[:, 1]

    delta = theta1 - theta2
    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)

    numer1 = (
        m2 * g * torch.sin(theta2) * cos_delta
        - m2
        * sin_delta
        * (
            l1 * omega1**2 * cos_delta
            + l2 * omega2**2
        )
        - (m1 + m2) * g * torch.sin(theta1)
    )

    denom1 = l1 * (
        m1 + m2 * sin_delta**2
    )

    numer2 = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2
        * l2
        * omega2**2
        * sin_delta
        * cos_delta
    )

    denom2 = l2 * (
        m1 + m2 * sin_delta**2
    )

    f1 = gamma1 - numer1 / denom1
    f2 = gamma2 - numer2 / denom2

    return f1, f2


def current_physics_weight(epoch):
    """Smoothly activate nonlinear physics after data warmup."""

    if epoch < WARMUP_EPOCHS:
        return 0.0

    ramp = min(
        1.0,
        (epoch - WARMUP_EPOCHS + 1)
        / PHYSICS_RAMP_EPOCHS,
    )

    return LAMBDA_PHYSICS * ramp


# -----------------------------------------------------------------------------
# Output utilities
# -----------------------------------------------------------------------------
def save_log(
    device,
    data_total,
    n_fourier,
    active_modes,
    epoch,
    loss,
    r2_1,
    r2_2,
    r2_train_1,
    r2_train_2,
    r2_test_1,
    r2_test_2,
    runtime,
):
    log_lines = [
        "Name: Double Pendulum Fourier PINN - padded period",
        f"Using device: {device}",
        f"Thread: {torch.get_num_threads()}",
        f"Data_total: {data_total}",
        f"Fourier samples: {n_fourier}",
        f"Fourier period factor: {FOURIER_PERIOD_FACTOR}",
        f"Active Fourier modes: {active_modes}",
        f"Max angular frequency: {MAX_ANGULAR_FREQUENCY}",
        f"Initialization omega max: {INITIALIZATION_OMEGA_MAX}",
        f"Initialization ridge: {INITIALIZATION_RIDGE}",
        f"Physics margin fraction: {PHYSICS_MARGIN_FRACTION}",
        f"data_stop: {DATA_STOP}",
        f"data_step: {DATA_STEP}",
        f"m1: {m1}",
        f"m2: {m2}",
        f"l1: {l1}",
        f"l2: {l2}",
        f"g: {g}",
        f"Learning rate network: {LEARNING_RATE_NETWORK}",
        f"Learning rate spectrum: {LEARNING_RATE_SPECTRUM}",
        f"Lambda data: {LAMBDA_DATA}",
        f"Lambda physics: {LAMBDA_PHYSICS}",
        f"Lambda initial: {LAMBDA_INITIAL}",
        f"Lambda energy: {LAMBDA_ENERGY}",
        f"Warmup epochs: {WARMUP_EPOCHS}",
        f"Physics ramp epochs: {PHYSICS_RAMP_EPOCHS}",
        f"Epoch: {epoch}",
        f"Loss: {loss:.6e}",
        f"R2 theta1 all: {r2_1:.6f}",
        f"R2 theta2 all: {r2_2:.6f}",
        f"R2 mean all: {0.5 * (r2_1 + r2_2):.6f}",
        f"R2 theta1 train: {r2_train_1:.6f}",
        f"R2 theta2 train: {r2_train_2:.6f}",
        f"R2 theta1 extrapolation: {r2_test_1:.6f}",
        f"R2 theta2 extrapolation: {r2_test_2:.6f}",
        f"Runtime: {runtime}",
    ]

    LOG_FILE.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )


def save_time_animation(
    t,
    theta1_reference,
    theta2_reference,
    data_indices,
    snapshots,
    epochs,
):
    fig, ax = plt.subplots()

    ax.plot(
        t,
        theta1_reference,
        color="blue",
        alpha=0.35,
        label=r"Numerical $\theta_1$",
    )
    ax.plot(
        t,
        theta2_reference,
        color="red",
        alpha=0.35,
        label=r"Numerical $\theta_2$",
    )

    ax.plot(
        t[data_indices],
        theta1_reference[data_indices],
        "o",
        color="blue",
        label=r"Data $\theta_1$",
    )
    ax.plot(
        t[data_indices],
        theta2_reference[data_indices],
        "o",
        color="red",
        label=r"Data $\theta_2$",
    )

    line1, = ax.plot(
        t,
        snapshots[0][:, 0],
        "--",
        color="blue",
        label=r"FPINN $\theta_1$",
    )

    line2, = ax.plot(
        t,
        snapshots[0][:, 1],
        "--",
        color="red",
        label=r"FPINN $\theta_2$",
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Angle (rad)",
    )
    ax.legend(ncol=2)

    title = ax.set_title("")

    def update(frame):
        line1.set_ydata(
            snapshots[frame][:, 0]
        )
        line2.set_ydata(
            snapshots[frame][:, 1]
        )

        title.set_text(
            f"Double Pendulum Fourier PINN - Epoch {epochs[frame]}"
        )

        return line1, line2, title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        blit=True,
    )

    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(
            fps=GIF_FPS
        ),
    )

    plt.close(fig)


def save_spectrum_animation(
    frequencies,
    reference,
    snapshots,
    epochs,
):
    fig, ax = plt.subplots()

    ax.plot(
        frequencies,
        reference[:, 0],
        color="blue",
        alpha=0.35,
        label=r"FFT $|\Theta_1|$",
    )
    ax.plot(
        frequencies,
        reference[:, 1],
        color="red",
        alpha=0.35,
        label=r"FFT $|\Theta_2|$",
    )

    line1, = ax.plot(
        frequencies,
        snapshots[0][:, 0],
        "--",
        color="blue",
        label=r"FPINN $|\Theta_1|$",
    )

    line2, = ax.plot(
        frequencies,
        snapshots[0][:, 1],
        "--",
        color="red",
        label=r"FPINN $|\Theta_2|$",
    )

    ax.set(
        xlabel="Angular frequency (rad/s)",
        ylabel=r"$|\Theta(\omega)|$",
        xlim=(0, SPECTRUM_XMAX),
    )

    if SPECTRUM_YMAX is not None:
        ax.set_ylim(
            0,
            SPECTRUM_YMAX,
        )

    ax.legend(ncol=2)

    title = ax.set_title("")

    def update(frame):
        line1.set_ydata(
            snapshots[frame][:, 0]
        )
        line2.set_ydata(
            snapshots[frame][:, 1]
        )

        title.set_text(
            f"Spectrum Evolution - Epoch {epochs[frame]}"
        )

        return line1, line2, title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        blit=True,
    )

    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_spectrum_evolution.gif",
        writer=animation.PillowWriter(
            fps=GIF_FPS
        ),
    )

    plt.close(fig)


def save_figures(
    t,
    theta1_reference,
    theta2_reference,
    data_indices,
    theta_prediction,
    frequencies,
    spectrum_reference,
    spectrum_prediction,
    history,
):
    # Time-domain result
    fig, ax = plt.subplots()

    ax.plot(
        t,
        theta1_reference,
        color="blue",
        alpha=0.35,
        label=r"Numerical $\theta_1$",
    )
    ax.plot(
        t,
        theta2_reference,
        color="red",
        alpha=0.35,
        label=r"Numerical $\theta_2$",
    )

    ax.plot(
        t[data_indices],
        theta1_reference[data_indices],
        "o",
        color="blue",
        label=r"Data $\theta_1$",
    )
    ax.plot(
        t[data_indices],
        theta2_reference[data_indices],
        "o",
        color="red",
        label=r"Data $\theta_2$",
    )

    ax.plot(
        t,
        theta_prediction[:, 0],
        "--",
        color="blue",
        label=r"FPINN $\theta_1$",
    )
    ax.plot(
        t,
        theta_prediction[:, 1],
        "--",
        color="red",
        label=r"FPINN $\theta_2$",
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Angle (rad)",
        title="Double Pendulum Fourier PINN",
    )

    ax.legend(ncol=2)

    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png",
        dpi=600,
    )
    plt.close(fig)

    # Spectrum
    fig, ax = plt.subplots()

    ax.plot(
        frequencies,
        spectrum_reference[:, 0] + 1e-12,
        color="blue",
        alpha=0.35,
        label=r"FFT $|\Theta_1|$",
    )
    ax.plot(
        frequencies,
        spectrum_reference[:, 1] + 1e-12,
        color="red",
        alpha=0.35,
        label=r"FFT $|\Theta_2|$",
    )

    ax.plot(
        frequencies,
        spectrum_prediction[:, 0] + 1e-12,
        "--",
        color="blue",
        label=r"FPINN $|\Theta_1|$",
    )
    ax.plot(
        frequencies,
        spectrum_prediction[:, 1] + 1e-12,
        "--",
        color="red",
        label=r"FPINN $|\Theta_2|$",
    )

    ax.set(
        xlabel="Angular frequency (rad/s)",
        ylabel=r"$|\Theta(\omega)|$",
        xlim=(0, SPECTRUM_XMAX),
    )

    if SPECTRUM_YMAX is not None:
        ax.set_ylim(
            0,
            SPECTRUM_YMAX,
        )

    ax.legend(ncol=2)

    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_spectrum.png",
        dpi=600,
    )
    plt.close(fig)

    # Raw losses
    epoch_axis = np.arange(
        len(history["total"])
    )

    fig, ax = plt.subplots()

    ax.semilogy(
        epoch_axis,
        np.maximum(history["total"], 1e-20),
        color="black",
        label="Total Loss",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["data"], 1e-20),
        color="blue",
        label="Data Loss",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["physics"], 1e-20),
        color="red",
        label="Physics Loss",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["initial"], 1e-20),
        color="green",
        label="Initial Condition Loss",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["energy"], 1e-20),
        color="purple",
        label="Energy Loss",
    )

    ax.set(
        xlabel="Epochs",
        ylabel="Loss",
        title="Raw Loss Convergence",
    )
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png",
        dpi=600,
    )
    plt.close(fig)

    # Weighted loss contributions
    fig, ax = plt.subplots()

    ax.semilogy(
        epoch_axis,
        np.maximum(history["weighted_data"], 1e-20),
        label="Weighted Data",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["weighted_physics"], 1e-20),
        label="Weighted Physics",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["weighted_initial"], 1e-20),
        label="Weighted Initial",
    )
    ax.semilogy(
        epoch_axis,
        np.maximum(history["weighted_energy"], 1e-20),
        label="Weighted Energy",
    )

    ax.set(
        xlabel="Epochs",
        ylabel="Weighted contribution",
        title="Weighted Loss Contributions",
    )
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_weighted_loss.png",
        dpi=600,
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    elif device.type == "mps":
        print("GPU: Apple Metal Performance Shaders")
    else:
        print(
            f"CPU: {torch.get_num_threads()} threads"
        )

    (
        t,
        theta1_ref,
        theta2_ref,
        omega1_ref,
        omega2_ref,
        dt,
    ) = load_data(DATA_FILE)

    n_physical = len(t)

    # Longer Fourier period.
    n_fourier = FOURIER_PERIOD_FACTOR * n_physical

    frequencies = (
        2.0
        * np.pi
        * np.fft.rfftfreq(
            n_fourier,
            d=dt,
        )
    )

    total_modes = len(frequencies)

    active_modes = int(
        np.searchsorted(
            frequencies,
            MAX_ANGULAR_FREQUENCY,
            side="right",
        )
    )

    active_modes = max(
        2,
        min(
            active_modes,
            total_modes,
        ),
    )

    data_stop = min(
        DATA_STOP,
        n_physical,
    )

    data_indices = np.arange(
        0,
        data_stop,
        DATA_STEP,
    )

    if len(data_indices) < 2:
        raise ValueError(
            "Too few training data points. "
            "Increase DATA_STOP or reduce DATA_STEP."
        )

    # Initial Fourier coefficients from sparse observed data.
    initial_1 = estimate_initial_spectrum(
        t[data_indices],
        theta1_ref[data_indices],
        frequencies[:active_modes],
    )

    initial_2 = estimate_initial_spectrum(
        t[data_indices],
        theta2_ref[data_indices],
        frequencies[:active_modes],
    )

    physical_duration = (
        t[-1] - t[0] + dt
    )
    fourier_duration = (
        n_fourier * dt
    )

    print(
        f"Physical samples: {n_physical}"
    )
    print(
        f"Fourier samples: {n_fourier}"
    )
    print(
        f"Physical duration: {physical_duration:.3f} s"
    )
    print(
        f"Fourier period: {fourier_duration:.3f} s"
    )
    print(
        f"Training data points: {len(data_indices)}"
    )
    print(
        f"Active Fourier modes: {active_modes}/{total_modes} "
        f"(omega_max={frequencies[active_modes - 1]:.3f} rad/s)"
    )

    omega = torch.tensor(
        frequencies,
        dtype=torch.float32,
        device=device,
    )

    omega_active = omega[:active_modes]
    omega_input = omega_active[:, None]

    index_tensor = torch.tensor(
        data_indices,
        dtype=torch.long,
        device=device,
    )

    theta_data = torch.tensor(
        np.column_stack(
            (
                theta1_ref[data_indices],
                theta2_ref[data_indices],
            )
        ),
        dtype=torch.float32,
        device=device,
    )

    theta0_target = torch.tensor(
        [
            theta1_ref[0],
            theta2_ref[0],
        ],
        dtype=torch.float32,
        device=device,
    )

    omega0_target = torch.tensor(
        [
            omega1_ref[0],
            omega2_ref[0],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        _, _, _, energy0_target = mechanics(
            theta0_target[0],
            theta0_target[1],
            omega0_target[0],
            omega0_target[1],
        )

    model = DoubleFourierPINN(
        omega_active,
        n_fourier,
        initial_parameters=(
            initial_1,
            initial_2,
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.network.parameters(),
                "lr": LEARNING_RATE_NETWORK,
            },
            {
                "params": [
                    model.spectral_coefficients
                ],
                "lr": LEARNING_RATE_SPECTRUM,
            },
        ]
    )

    # Mild LR reduction after the nonlinear-physics ramp has largely settled.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            60_000,
            80_000,
        ],
        gamma=0.3,
    )

    history = {
        name: []
        for name in (
            "total",
            "data",
            "physics",
            "initial",
            "energy",
            "weighted_data",
            "weighted_physics",
            "weighted_initial",
            "weighted_energy",
        )
    }

    snapshot_epochs = []
    time_snapshots = []
    spectrum_snapshots = []

    spectrum_plot_mask = (
        frequencies <= SPECTRUM_XMAX
    )

    # Reference FFT is plotted on the same padded frequency grid.
    theta1_padded = np.zeros(
        n_fourier,
        dtype=np.float64,
    )
    theta2_padded = np.zeros(
        n_fourier,
        dtype=np.float64,
    )

    theta1_padded[:n_physical] = theta1_ref
    theta2_padded[:n_physical] = theta2_ref

    spectrum_reference = np.column_stack(
        (
            np.abs(
                np.fft.rfft(theta1_padded)
                / n_fourier
            ),
            np.abs(
                np.fft.rfft(theta2_padded)
                / n_fourier
            ),
        )
    )

    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------------------
        # Fourier reconstruction
        # -------------------------------------------------------------
        (
            spectrum,
            theta_extended,
            theta,
        ) = reconstruct(
            model,
            omega_input,
            total_modes,
            n_fourier,
            n_physical,
        )

        velocity, acceleration = spectral_derivatives(
            spectrum,
            omega,
            n_fourier,
            n_physical,
        )

        # -------------------------------------------------------------
        # 1) Sparse data loss
        # -------------------------------------------------------------
        data_loss = torch.mean(
            (
                theta[index_tensor]
                - theta_data
            ) ** 2
        )

        # -------------------------------------------------------------
        # 2) Explicit nonlinear physics loss
        # -------------------------------------------------------------
        f1, f2 = explicit_physics_residuals(
            theta,
            velocity,
            acceleration,
        )

        margin = max(
            1,
            int(
                PHYSICS_MARGIN_FRACTION
                * n_physical
            ),
        )

        if 2 * margin < n_physical:
            f1_physics = f1[
                margin:-margin
            ]
            f2_physics = f2[
                margin:-margin
            ]
        else:
            f1_physics = f1
            f2_physics = f2

        # Normalize residual by the natural acceleration scale.
        acceleration_scale = (
            g / min(l1, l2)
        )

        physics_loss = torch.mean(
            (
                f1_physics
                / acceleration_scale
            ) ** 2
            + (
                f2_physics
                / acceleration_scale
            ) ** 2
        )

        # -------------------------------------------------------------
        # 3) Initial condition loss
        # -------------------------------------------------------------
        initial_angle_loss = torch.mean(
            (
                theta[0]
                - theta0_target
            ) ** 2
        )

        initial_velocity_loss = torch.mean(
            (
                velocity[0]
                - omega0_target
            ) ** 2
        )

        initial_loss = (
            initial_angle_loss
            + initial_velocity_loss
        )

        # -------------------------------------------------------------
        # 4) Energy loss
        # -------------------------------------------------------------
        _, _, _, energy = mechanics(
            theta[:, 0],
            theta[:, 1],
            velocity[:, 0],
            velocity[:, 1],
        )

        energy_scale = torch.clamp(
            torch.abs(energy0_target),
            min=1.0,
        )

        energy_loss = torch.mean(
            (
                (
                    energy
                    - energy0_target
                )
                / energy_scale
            ) ** 2
        )

        physics_weight = current_physics_weight(
            epoch
        )

        weighted_data = (
            LAMBDA_DATA
            * data_loss
        )
        weighted_physics = (
            physics_weight
            * physics_loss
        )
        weighted_initial = (
            LAMBDA_INITIAL
            * initial_loss
        )
        weighted_energy = (
            LAMBDA_ENERGY
            * energy_loss
        )

        total_loss = (
            weighted_data
            + weighted_physics
            + weighted_initial
            + weighted_energy
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}. "
                "Reduce learning rates or MAX_ANGULAR_FREQUENCY."
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )

        optimizer.step()
        scheduler.step()

        history["total"].append(
            total_loss.item()
        )
        history["data"].append(
            data_loss.item()
        )
        history["physics"].append(
            physics_loss.item()
        )
        history["initial"].append(
            initial_loss.item()
        )
        history["energy"].append(
            energy_loss.item()
        )
        history["weighted_data"].append(
            weighted_data.item()
        )
        history["weighted_physics"].append(
            weighted_physics.item()
        )
        history["weighted_initial"].append(
            weighted_initial.item()
        )
        history["weighted_energy"].append(
            weighted_energy.item()
        )

        # -------------------------------------------------------------
        # Console diagnostics
        # -------------------------------------------------------------
        if epoch % PRINT_EVERY == 0:
            elapsed = format_time(
                time.time() - start_time
            )

            current_lrs = [
                group["lr"]
                for group in optimizer.param_groups
            ]

            print(
                f"\rEpoch {epoch:6d} | "
                f"Total {total_loss.item():.3e} | "
                f"D {data_loss.item():.2e}"
                f"[{weighted_data.item():.2e}] | "
                f"P {physics_loss.item():.2e}"
                f"[{weighted_physics.item():.2e}] | "
                f"IC {initial_loss.item():.2e}"
                f"[{weighted_initial.item():.2e}] | "
                f"wp={physics_weight:.3f} | "
                f"lrS={current_lrs[1]:.1e} | "
                f"Time {elapsed}",
                end="",
                flush=True,
            )

        # -------------------------------------------------------------
        # Snapshots
        # -------------------------------------------------------------
        if epoch % SNAPSHOT_EVERY == 0:
            model.eval()

            with torch.no_grad():
                (
                    spectrum_now,
                    _,
                    theta_now,
                ) = reconstruct(
                    model,
                    omega_input,
                    total_modes,
                    n_fourier,
                    n_physical,
                )

            snapshot_epochs.append(
                epoch
            )

            time_snapshots.append(
                theta_now
                .detach()
                .cpu()
                .numpy()
                .copy()
            )

            spectrum_snapshots.append(
                np.abs(
                    spectrum_now
                    .detach()
                    .cpu()
                    .numpy()
                )[
                    spectrum_plot_mask
                ].copy()
            )

            model.train()

    # -----------------------------------------------------------------
    # Final prediction
    # -----------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        (
            spectrum_final,
            _,
            theta_final,
        ) = reconstruct(
            model,
            omega_input,
            total_modes,
            n_fourier,
            n_physical,
        )

    theta_final = (
        theta_final
        .detach()
        .cpu()
        .numpy()
    )

    spectrum_final = np.abs(
        spectrum_final
        .detach()
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    r2_1 = coefficient_of_determination(
        theta1_ref,
        theta_final[:, 0],
    )

    r2_2 = coefficient_of_determination(
        theta2_ref,
        theta_final[:, 1],
    )

    r2_mean = 0.5 * (
        r2_1 + r2_2
    )

    train_slice = slice(
        0,
        data_stop,
    )

    r2_train_1 = coefficient_of_determination(
        theta1_ref[train_slice],
        theta_final[train_slice, 0],
    )

    r2_train_2 = coefficient_of_determination(
        theta2_ref[train_slice],
        theta_final[train_slice, 1],
    )

    if data_stop < n_physical:
        test_slice = slice(
            data_stop,
            n_physical,
        )

        r2_test_1 = coefficient_of_determination(
            theta1_ref[test_slice],
            theta_final[test_slice, 0],
        )

        r2_test_2 = coefficient_of_determination(
            theta2_ref[test_slice],
            theta_final[test_slice, 1],
        )

    else:
        r2_test_1 = float("nan")
        r2_test_2 = float("nan")

    runtime = format_time(
        time.time() - start_time
    )

    print()
    print(
        f"R^2 theta1 (all):   {r2_1:.6f}"
    )
    print(
        f"R^2 theta2 (all):   {r2_2:.6f}"
    )
    print(
        f"R^2 mean (all):     {r2_mean:.6f}"
    )
    print(
        f"R^2 theta1 (train): {r2_train_1:.6f}"
    )
    print(
        f"R^2 theta2 (train): {r2_train_2:.6f}"
    )
    print(
        f"R^2 theta1 (extra): {r2_test_1:.6f}"
    )
    print(
        f"R^2 theta2 (extra): {r2_test_2:.6f}"
    )
    print(
        f"Runtime: {runtime}"
    )

    save_log(
        device=device,
        data_total=n_physical,
        n_fourier=n_fourier,
        active_modes=active_modes,
        epoch=epoch,
        loss=total_loss.item(),
        r2_1=r2_1,
        r2_2=r2_2,
        r2_train_1=r2_train_1,
        r2_train_2=r2_train_2,
        r2_test_1=r2_test_1,
        r2_test_2=r2_test_2,
        runtime=runtime,
    )

    print(
        "Saving figures and animations..."
    )

    save_time_animation(
        t,
        theta1_ref,
        theta2_ref,
        data_indices,
        time_snapshots,
        snapshot_epochs,
    )

    save_spectrum_animation(
        frequencies[
            spectrum_plot_mask
        ],
        spectrum_reference[
            spectrum_plot_mask
        ],
        spectrum_snapshots,
        snapshot_epochs,
    )

    save_figures(
        t,
        theta1_ref,
        theta2_ref,
        data_indices,
        theta_final,
        frequencies,
        spectrum_reference,
        spectrum_final,
        history,
    )

    print(
        "Figures and animations saved successfully."
    )


if __name__ == "__main__":
    main()
