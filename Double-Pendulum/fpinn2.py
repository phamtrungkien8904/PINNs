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
LOG_FILE = OUTPUT_DIR / f"FPINN2.log"

SEED = 0
EPOCHS = 100_000
SNAPSHOT_EVERY = 1_000
PRINT_EVERY = 100

# The physical 20 s record is only the first quarter of the Fourier period.
# This removes the false theta(0) == theta(20 s) boundary condition while
# retaining exact spectral differentiation on the interval of interest.
FOURIER_PERIOD_FACTOR = 4
MAX_ANGULAR_FREQUENCY = 12.0
INITIALIZATION_RIDGE = 1e-2

# Train the Fourier representation on data/IC first, then introduce physics.
WARMUP_EPOCHS = 2_000
PHYSICS_RAMP_EPOCHS = 8_000

# Use sparse measurements from the first 10 s and predict the remaining 10 s.
DATA_STOP = 1000
DATA_STEP = 10

LEARNING_RATE_NETWORK = 1e-4
LEARNING_RATE_SPECTRUM = 2e-4
WEIGHT_DECAY = 1e-7

LAMBDA_DATA = 1e3
LAMBDA_PHYSICS = 1e1
LAMBDA_INITIAL = 5e2
LAMBDA_ENERGY = 0.0

GRADIENT_CLIP = 1.0

SPECTRUM_XMAX = 20.0
SPECTRUM_YMAX = None
GIF_FPS = 30

# Double-pendulum parameters: same convention as tpinn2_ver2.py.
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
    data = np.loadtxt(path, skiprows=1)

    t = data[:, 0]
    theta1 = data[:, 1]
    theta2 = data[:, 2]

    dt_all = np.diff(t)
    dt = float(np.mean(dt_all))

    omega1 = np.gradient(theta1, t, edge_order=2)
    omega2 = np.gradient(theta2, t, edge_order=2)

    return t, theta1, theta2, omega1, omega2, dt


def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference)) ** 2)
    residual = np.sum((reference - prediction) ** 2)
    return 1.0 - residual / total


def estimate_initial_spectrum(t_data, theta_data, frequencies, ridge):
    """Fit all active sine/cosine modes jointly to the sparse measurements."""
    nonzero_frequencies = frequencies[1:]
    design = np.column_stack(
        (
            np.ones(len(t_data)),
            np.cos(t_data[:, None] * nonzero_frequencies[None, :]),
            np.sin(t_data[:, None] * nonzero_frequencies[None, :]),
        )
    )
    gram = design.T @ design
    regularizer = ridge * np.eye(gram.shape[0])
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer,
        design.T @ theta_data,
    )

    mode_count = len(nonzero_frequencies)
    spectrum = np.zeros((len(frequencies), theta_data.shape[1]), dtype=np.complex64)
    spectrum[0] = coefficients[0]
    cosine = coefficients[1 : 1 + mode_count]
    sine = coefficients[1 + mode_count :]
    spectrum[1:] = 0.5 * (cosine - 1j * sine)
    return spectrum


# -----------------------------------------------------------------------------
# Fourier neural network
# -----------------------------------------------------------------------------
class DoubleFourierPINN(nn.Module):
    """Map angular frequency -> complex spectra [Theta1(omega), Theta2(omega)]."""

    def __init__(self, frequencies, n_fourier, initial_spectrum):
        super().__init__()
        self.register_buffer("frequencies", frequencies)
        self.n_fourier = n_fourier

        # Output columns:
        # [Re Theta1, Im Theta1, Re Theta2, Im Theta2]
        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
        )

        initial_output = torch.zeros(len(frequencies), 4, dtype=torch.float32)
        initial_output[:, 0] = torch.from_numpy(initial_spectrum[:, 0].real.copy())
        initial_output[:, 1] = torch.from_numpy(initial_spectrum[:, 0].imag.copy())
        initial_output[:, 2] = torch.from_numpy(initial_spectrum[:, 1].real.copy())
        initial_output[:, 3] = torch.from_numpy(initial_spectrum[:, 1].imag.copy())
        self.spectral_coefficients = nn.Parameter(initial_output)

        # Start from the sparse spectral estimate. The MLP learns a smooth
        # frequency-dependent correction while each Fourier bin remains trainable.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.network_scale = 1e-2

    def forward(self, omega):
        omega_max = torch.clamp(self.frequencies[-1], min=1e-12)
        omega_scaled = 2.0 * omega / omega_max - 1.0
        output = self.spectral_coefficients + self.network_scale * self.network(
            omega_scaled
        )

        real = torch.stack((output[:, 0], output[:, 2]), dim=1)
        imag = torch.stack((output[:, 1], output[:, 3]), dim=1)

        # DC must be real. Nyquist is also real if it is part of the active set.
        imaginary_mask = torch.ones(len(output), device=output.device, dtype=output.dtype)
        imaginary_mask[0] = 0.0
        if self.n_fourier % 2 == 0:
            total_rfft_modes = self.n_fourier // 2 + 1
            if len(self.frequencies) == total_rfft_modes:
                imaginary_mask[-1] = 0.0

        return torch.complex(real, imag * imaginary_mask[:, None])


def reconstruct(model, omega_input, total_modes, n_fourier, n_physical):
    """Return normalized spectra and reconstructed [theta1(t), theta2(t)]."""
    theta_active = model(omega_input)  # [active_modes, 2]

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

    theta_extended = torch.fft.irfft(
        n_fourier * theta_fourier, n=n_fourier, dim=0
    )
    return theta_fourier, theta_extended[:n_physical]


def spectral_derivatives(theta_fourier, omega, n_fourier, n_physical):
    """Compute angular velocity and acceleration directly in Fourier space.

    d/dt Theta  -> i*omega*Theta
    d2/dt2 Theta -> -omega^2*Theta
    """
    omega_column = omega[:, None]
    velocity_fourier = 1j * omega_column * theta_fourier
    acceleration_fourier = -(omega_column**2) * theta_fourier

    velocity_extended = torch.fft.irfft(
        n_fourier * velocity_fourier, n=n_fourier, dim=0
    )
    acceleration_extended = torch.fft.irfft(
        n_fourier * acceleration_fourier, n=n_fourier, dim=0
    )
    return velocity_extended[:n_physical], acceleration_extended[:n_physical]


# -----------------------------------------------------------------------------
# Double-pendulum mechanics and explicit physics residual
# -----------------------------------------------------------------------------
def mechanics(theta1, theta2, omega1, omega2):
    """Return T, V, L and E=T+V using the same coordinates as tpinn2."""
    y1 = -l1 * torch.cos(theta1)
    y2 = y1 - l2 * torch.cos(theta2)

    vx1 = l1 * torch.cos(theta1) * omega1
    vy1 = l1 * torch.sin(theta1) * omega1
    vx2 = vx1 + l2 * torch.cos(theta2) * omega2
    vy2 = vy1 + l2 * torch.sin(theta2) * omega2

    kinetic = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (
        vx2**2 + vy2**2
    )
    potential = m1 * g * y1 + m2 * g * y2
    lagrangian = kinetic - potential
    energy = kinetic + potential
    return kinetic, potential, lagrangian, energy


def explicit_physics_residuals(theta, velocity, acceleration):
    """Explicit double-pendulum residuals f1=f2=0.

    This uses exactly the explicit equations from tpinn2_ver2.py, but theta,
    theta_dot and theta_ddot are supplied by the Fourier reconstruction and
    spectral differentiation.
    """
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
        * (l1 * omega1**2 * cos_delta + l2 * omega2**2)
        - (m1 + m2) * g * torch.sin(theta1)
    )
    denom1 = l1 * (m1 + m2 * sin_delta**2)

    numer2 = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
    )
    denom2 = l2 * (m1 + m2 * sin_delta**2)

    f1 = gamma1 - numer1 / denom1
    f2 = gamma2 - numer2 / denom2
    return f1, f2


def current_physics_weight(epoch):
    """Introduce the nonlinear physics residual gradually after warmup."""
    if epoch < WARMUP_EPOCHS:
        return 0.0
    ramp = min(1.0, (epoch - WARMUP_EPOCHS + 1) / PHYSICS_RAMP_EPOCHS)
    return LAMBDA_PHYSICS * ramp**2


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_log(
    device,
    data_total,
    active_modes,
    epoch,
    loss,
    r2_1,
    r2_2,
    r2_1_extra,
    r2_2_extra,
    runtime,
):
    log_lines = [
        "Name: Double Pendulum Fourier PINN",
        f"Using device: {device}",
        f"Thread: {torch.get_num_threads()}",
        f"Data_total: {data_total}",
        f"Active Fourier modes: {active_modes}",
        f"Fourier period factor: {FOURIER_PERIOD_FACTOR}",
        f"Maximum angular frequency: {MAX_ANGULAR_FREQUENCY}",
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
        f"R2 theta1: {r2_1:.6f}",
        f"R2 theta2: {r2_2:.6f}",
        f"R2 mean: {0.5 * (r2_1 + r2_2):.6f}",
        f"R2 theta1 extrapolation: {r2_1_extra:.6f}",
        f"R2 theta2 extrapolation: {r2_2_extra:.6f}",
        f"R2 extrapolation mean: {0.5 * (r2_1_extra + r2_2_extra):.6f}",
        f"Runtime: {runtime}",
    ]
    LOG_FILE.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def save_time_animation(
    t,
    theta1_reference,
    theta2_reference,
    data_indices,
    snapshots,
    epochs,
):
    fig, ax = plt.subplots()
    ax.plot(t, theta1_reference, color="blue", alpha=0.35, label=r"Numerical $\theta_1$")
    ax.plot(t, theta2_reference, color="red", alpha=0.35, label=r"Numerical $\theta_2$")
    ax.plot(t[data_indices], theta1_reference[data_indices], "o", color="blue", label=r"Data $\theta_1$")
    ax.plot(t[data_indices], theta2_reference[data_indices], "o", color="red", label=r"Data $\theta_2$")

    line1, = ax.plot(t, snapshots[0][:, 0], "--", color="blue", label=r"FPINN $\theta_1$")
    line2, = ax.plot(t, snapshots[0][:, 1], "--", color="red", label=r"FPINN $\theta_2$")
    ax.set(xlabel="Time (s)", ylabel="Angle (rad)")
    ax.legend(ncol=2)
    title = ax.set_title("")

    def update(frame):
        line1.set_ydata(snapshots[frame][:, 0])
        line2.set_ydata(snapshots[frame][:, 1])
        title.set_text(f"Double Pendulum Fourier PINN - Epoch {epochs[frame]}")
        return line1, line2, title

    movie = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True)
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_spectrum_animation(frequencies, reference, snapshots, epochs):
    fig, ax = plt.subplots()
    ax.plot(frequencies, reference[:, 0], color="blue", alpha=0.35, label=r"FFT $|\Theta_1|$")
    ax.plot(frequencies, reference[:, 1], color="red", alpha=0.35, label=r"FFT $|\Theta_2|$")

    line1, = ax.plot(frequencies, snapshots[0][:, 0], "--", color="blue", label=r"FPINN $|\Theta_1|$")
    line2, = ax.plot(frequencies, snapshots[0][:, 1], "--", color="red", label=r"FPINN $|\Theta_2|$")

    ax.set(xlabel="Angular frequency (rad/s)", ylabel=r"$|\Theta(\omega)|$", xlim=(0, SPECTRUM_XMAX))
    if SPECTRUM_YMAX is not None:
        ax.set_ylim(0, SPECTRUM_YMAX)
    ax.legend(ncol=2)
    title = ax.set_title("")

    def update(frame):
        line1.set_ydata(snapshots[frame][:, 0])
        line2.set_ydata(snapshots[frame][:, 1])
        title.set_text(f"Spectrum Evolution - Epoch {epochs[frame]}")
        return line1, line2, title

    movie = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True)
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_spectrum_evolution.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
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
    fig, ax = plt.subplots()
    ax.plot(t, theta1_reference, color="blue", alpha=0.35, label=r"Numerical $\theta_1$")
    ax.plot(t, theta2_reference, color="red", alpha=0.35, label=r"Numerical $\theta_2$")
    ax.plot(t[data_indices], theta1_reference[data_indices], "o", color="blue", label=r"Data $\theta_1$")
    ax.plot(t[data_indices], theta2_reference[data_indices], "o", color="red", label=r"Data $\theta_2$")
    ax.plot(t, theta_prediction[:, 0], "--", color="blue", label=r"FPINN $\theta_1$")
    ax.plot(t, theta_prediction[:, 1], "--", color="red", label=r"FPINN $\theta_2$")
    ax.set(xlabel="Time (s)", ylabel="Angle (rad)", title="Double Pendulum Fourier PINN")
    ax.legend(ncol=2)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(frequencies, spectrum_reference[:, 0] + 1e-12, color="blue", alpha=0.35, label=r"FFT $|\Theta_1|$")
    ax.plot(frequencies, spectrum_reference[:, 1] + 1e-12, color="red", alpha=0.35, label=r"FFT $|\Theta_2|$")
    ax.plot(frequencies, spectrum_prediction[:, 0] + 1e-12, "--", color="blue", label=r"FPINN $|\Theta_1|$")
    ax.plot(frequencies, spectrum_prediction[:, 1] + 1e-12, "--", color="red", label=r"FPINN $|\Theta_2|$")
    ax.set(xlabel="Angular frequency (rad/s)", ylabel=r"$|\Theta(\omega)|$", xlim=(0, SPECTRUM_XMAX))
    if SPECTRUM_YMAX is not None:
        ax.set_ylim(0, SPECTRUM_YMAX)
    ax.legend(ncol=2)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_spectrum.png", dpi=600)
    plt.close(fig)

    epoch_axis = np.arange(len(history["total"]))
    fig, ax = plt.subplots()
    ax.semilogy(epoch_axis, history["total"], color="black", label="Total Loss")
    ax.semilogy(epoch_axis, history["data"], color="blue", label="Data Loss")
    ax.semilogy(epoch_axis, history["physics"], color="red", label="Physics Loss")
    ax.semilogy(epoch_axis, history["initial"], color="green", label="Initial Condition Loss")
    ax.semilogy(epoch_axis, history["energy"], color="purple", label="Energy Loss")
    ax.set(xlabel="Epochs", ylabel="Loss", title="Loss Convergence")
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png", dpi=600)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU: {torch.get_num_threads()} threads")

    t, theta1_ref, theta2_ref, omega1_ref, omega2_ref, dt = load_data(DATA_FILE)
    n_time = len(t)

    n_fourier = FOURIER_PERIOD_FACTOR * n_time
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(n_fourier, d=dt)
    total_modes = len(frequencies)
    active_modes = min(
        int(np.searchsorted(frequencies, MAX_ANGULAR_FREQUENCY, side="right")),
        total_modes,
    )

    data_stop = n_time if DATA_STOP is None else min(DATA_STOP, n_time)
    data_indices = np.arange(0, data_stop, DATA_STEP)

    initial_spectrum = estimate_initial_spectrum(
        t[data_indices],
        np.column_stack((theta1_ref[data_indices], theta2_ref[data_indices])),
        frequencies[:active_modes],
        INITIALIZATION_RIDGE,
    )

    print(
        f"Active Fourier modes: {active_modes}/{total_modes} "
        f"(omega_max={frequencies[active_modes - 1]:.3f} rad/s)"
    )
    print(
        f"Physical window: {t[-1] - t[0]:.2f} s; "
        f"Fourier period: {n_fourier * dt:.2f} s"
    )

    omega = torch.tensor(frequencies, dtype=torch.float32, device=device)
    omega_active = omega[:active_modes]
    omega_input = omega_active[:, None]

    index_tensor = torch.tensor(data_indices, dtype=torch.long, device=device)
    theta_data = torch.tensor(
        np.column_stack((theta1_ref[data_indices], theta2_ref[data_indices])),
        dtype=torch.float32,
        device=device,
    )

    theta0_target = torch.tensor(
        [theta1_ref[0], theta2_ref[0]], dtype=torch.float32, device=device
    )
    omega0_target = torch.tensor(
        [omega1_ref[0], omega2_ref[0]], dtype=torch.float32, device=device
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
        initial_spectrum,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.network.parameters(), "lr": LEARNING_RATE_NETWORK, "weight_decay": WEIGHT_DECAY},
            {"params": [model.spectral_coefficients], "lr": LEARNING_RATE_SPECTRUM, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40_000, 70_000, 90_000],
        gamma=0.3,
    )

    history = {
        name: [] for name in ("total", "data", "physics", "initial", "energy")
    }
    snapshot_epochs = []
    time_snapshots = []
    spectrum_snapshots = []

    plot_frequencies = 2.0 * np.pi * np.fft.rfftfreq(n_time, d=dt)
    spectrum_plot_mask = plot_frequencies <= SPECTRUM_XMAX
    spectrum_reference = np.column_stack(
        (
            np.abs(np.fft.rfft(theta1_ref) / n_time),
            np.abs(np.fft.rfft(theta2_ref) / n_time),
        )
    )

    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)

        # Frequency-domain model -> two reconstructed time-domain angles.
        spectrum, theta = reconstruct(
            model, omega_input, total_modes, n_fourier, n_time
        )

        # Exact Fourier differentiation of the represented signal.
        velocity, acceleration = spectral_derivatives(
            spectrum, omega, n_fourier, n_time
        )

        # 1) Sparse angle data loss.
        data_loss = torch.mean((theta[index_tensor] - theta_data) ** 2)

        # 2) Explicit nonlinear double-pendulum physics loss.
        f1, f2 = explicit_physics_residuals(theta, velocity, acceleration)
        acceleration_scale = g / min(l1, l2)
        physics_loss = torch.mean(
            (f1 / acceleration_scale) ** 2
            + (f2 / acceleration_scale) ** 2
        )

        # 3) Initial angle + angular-velocity loss.
        initial_loss = torch.mean((theta[0] - theta0_target) ** 2) + torch.mean(
            (velocity[0] - omega0_target) ** 2
        )

        # 4) Mechanical energy conservation.
        _, _, _, energy = mechanics(
            theta[:, 0], theta[:, 1], velocity[:, 0], velocity[:, 1]
        )
        energy_loss = torch.mean((energy - energy0_target) ** 2)

        total_loss = (
            LAMBDA_DATA * data_loss
            + current_physics_weight(epoch) * physics_loss
            + LAMBDA_INITIAL * initial_loss
            + LAMBDA_ENERGY * energy_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()

        history["total"].append(total_loss.item())
        history["data"].append(data_loss.item())
        history["physics"].append(physics_loss.item())
        history["initial"].append(initial_loss.item())
        history["energy"].append(energy_loss.item())

        if epoch % PRINT_EVERY == 0:
            elapsed = format_time(time.time() - start_time)
            physics_weight = current_physics_weight(epoch)
            print(
                f"\rEpoch {epoch:6d} | Total {total_loss.item():.3e} | "
                f"D {data_loss.item():.2e}[{LAMBDA_DATA * data_loss.item():.2e}] | "
                f"P {physics_loss.item():.2e}[{physics_weight * physics_loss.item():.2e}] | "
                f"IC {initial_loss.item():.2e}[{LAMBDA_INITIAL * initial_loss.item():.2e}] | "
                f"Time {elapsed}",
                end='',
                flush=True,
            )

        if epoch % SNAPSHOT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                spectrum_now, theta_now = reconstruct(
                    model, omega_input, total_modes, n_fourier, n_time
                )
            snapshot_epochs.append(epoch)
            time_snapshots.append(theta_now.cpu().numpy().copy())
            theta_snapshot = theta_now.cpu().numpy()
            spectrum_snapshots.append(
                np.abs(np.fft.rfft(theta_snapshot, axis=0) / n_time)[
                    spectrum_plot_mask
                ].copy()
            )
            model.train()

    model.eval()
    with torch.no_grad():
        spectrum_final, theta_final = reconstruct(
            model, omega_input, total_modes, n_fourier, n_time
        )

    theta_final = theta_final.cpu().numpy()
    spectrum_final = np.abs(np.fft.rfft(theta_final, axis=0) / n_time)

    r2_1 = coefficient_of_determination(theta1_ref, theta_final[:, 0])
    r2_2 = coefficient_of_determination(theta2_ref, theta_final[:, 1])
    r2_mean = 0.5 * (r2_1 + r2_2)
    extrapolation_start = min(data_stop, n_time - 1)
    r2_1_extra = coefficient_of_determination(
        theta1_ref[extrapolation_start:], theta_final[extrapolation_start:, 0]
    )
    r2_2_extra = coefficient_of_determination(
        theta2_ref[extrapolation_start:], theta_final[extrapolation_start:, 1]
    )

    runtime = format_time(time.time() - start_time)
    print(f"\nR^2 theta1: {r2_1:.6f}")
    print(f"R^2 theta2: {r2_2:.6f}")
    print(f"R^2 mean:   {r2_mean:.6f}")
    print(f"R^2 theta1 extrapolation: {r2_1_extra:.6f}")
    print(f"R^2 theta2 extrapolation: {r2_2_extra:.6f}")
    print(f"Runtime: {runtime}")

    save_log(
        device=device,
        data_total=n_time,
        active_modes=active_modes,
        epoch=epoch,
        loss=total_loss.item(),
        r2_1=r2_1,
        r2_2=r2_2,
        r2_1_extra=r2_1_extra,
        r2_2_extra=r2_2_extra,
        runtime=runtime,
    )

    print("Saving figures and animations...")
    save_time_animation(
        t,
        theta1_ref,
        theta2_ref,
        data_indices,
        time_snapshots,
        snapshot_epochs,
    )
    save_spectrum_animation(
        plot_frequencies[spectrum_plot_mask],
        spectrum_reference[spectrum_plot_mask],
        spectrum_snapshots,
        snapshot_epochs,
    )
    save_figures(
        t,
        theta1_ref,
        theta2_ref,
        data_indices,
        theta_final,
        plot_frequencies,
        spectrum_reference,
        spectrum_final,
        history,
    )
    print("Figures and animations saved successfully.")


if __name__ == "__main__":
    main()
