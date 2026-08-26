"""Fourier-output PINN for a nonlinear pendulum.

The network predicts Fourier coefficients, while the pendulum residual is
evaluated in the time domain after an inverse FFT. All plots and animations
are written only after training.
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_FILE = Path("pendulum_data.dat")
OUTPUT_DIR = Path(".")
VERSION = "ver7"

SEED = 0
EPOCHS = 100_000
SNAPSHOT_EVERY = 1_000
PRINT_EVERY = 1_000
MAX_PHYSICS_MODES = 128

WARMUP_EPOCHS = 5_000
PHYSICS_RAMP_EPOCHS = 20_000

LAMBDA_DATA = 1e1
LAMBDA_PHYSICS = 1e0
LAMBDA_INITIAL = 1e1
LAMBDA_ENERGY = 1e-3

ALPHA_INITIAL = 0.0
SPECTRUM_XMAX = 10.0
SPECTRUM_YMAX = 1.0
GIF_FPS = 30


plt.style.use("classic")
torch.manual_seed(SEED)
np.random.seed(SEED)


def format_time(seconds):
    """Convert elapsed seconds to HH:MM:SS."""

    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_data(path):
    """Load and validate a uniformly sampled pendulum trajectory."""

    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Data must contain time, angle, and velocity columns.")

    t, theta, velocity = data[:, 0], data[:, 1], data[:, 2]
    if len(t) < 3:
        raise ValueError("At least three time samples are required.")

    dt = t[1] - t[0]
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-5, atol=1e-8):
        raise ValueError("The Fourier reconstruction requires a uniform time grid.")

    return t, theta, velocity, dt


def estimate_initial_mode(t_data, theta_data, frequencies):
    """Fit one sinusoidal mode to the sparse measurements."""

    errors = np.full(len(frequencies), np.inf)
    coefficients = np.zeros((len(frequencies), 2))

    for mode, frequency in enumerate(frequencies[1:], start=1):
        design = np.column_stack(
            (np.cos(frequency * t_data), np.sin(frequency * t_data))
        )
        coefficient, *_ = np.linalg.lstsq(design, theta_data, rcond=None)
        coefficients[mode] = coefficient
        errors[mode] = np.mean((design @ coefficient - theta_data) ** 2)

    mode = int(np.argmin(errors))
    cosine, sine = coefficients[mode]
    return mode, cosine / 2.0, -sine / 2.0


class FourierPINN(nn.Module):
    """Map angular frequency to complex Fourier coefficients."""

    def __init__(
        self,
        frequencies,
        n_time,
        initial_mode,
        initial_real,
        initial_imaginary,
    ):
        super().__init__()
        self.register_buffer("frequencies", frequencies)
        self.n_time = n_time

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

        initial_spectrum = torch.zeros(len(frequencies), 2, dtype=torch.float32)
        initial_spectrum[initial_mode] = torch.tensor(
            [initial_real, initial_imaginary], dtype=torch.float32
        )
        self.spectral_coefficients = nn.Parameter(initial_spectrum)
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INITIAL, dtype=torch.float32))

        # Start exactly from the sparse single-mode estimate. The network then
        # learns only a small smooth correction between frequency bins.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.network_scale = 1e-2

    def forward(self, omega):
        omega_scaled = 2.0 * omega / self.frequencies[-1] - 1.0
        output = self.spectral_coefficients + self.network_scale * self.network(
            omega_scaled
        )

        # DC and, when present, Nyquist coefficients must be real for irfft.
        imaginary_mask = torch.ones_like(output[:, 1])
        imaginary_mask[0] = 0.0
        if self.n_time % 2 == 0:
            total_rfft_modes = self.n_time // 2 + 1
            if len(self.frequencies) == total_rfft_modes:
                imaginary_mask[-1] = 0.0

        return torch.complex(output[:, 0], output[:, 1] * imaginary_mask)


def reconstruct(model, omega_input, total_modes, n_time):
    """Return the normalized spectrum and reconstructed time signal."""

    theta_active = model(omega_input)
    theta_fourier = torch.cat(
        (
            theta_active,
            torch.zeros(
                total_modes - len(theta_active),
                dtype=theta_active.dtype,
                device=theta_active.device,
            ),
        )
    )
    theta_time = torch.fft.irfft(n_time * theta_fourier, n=n_time)
    return theta_fourier, theta_time


def time_derivatives(theta, dt):
    """Use non-periodic finite differences on the physical time grid."""

    velocity = torch.cat(
        (
            ((-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dt))[None],
            (theta[2:] - theta[:-2]) / (2.0 * dt),
            ((3.0 * theta[-1] - 4.0 * theta[-2] + theta[-3]) / (2.0 * dt))[
                None
            ],
        )
    )
    acceleration = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / dt**2
    return velocity, acceleration


def current_physics_weight(epoch):
    """Introduce the physics residual gradually after the data-only warmup."""

    if epoch < WARMUP_EPOCHS:
        return 0.0
    ramp = min(1.0, (epoch - WARMUP_EPOCHS + 1) / PHYSICS_RAMP_EPOCHS)
    return LAMBDA_PHYSICS * ramp


def save_time_animation(t, theta_reference, data_indices, snapshots, epochs):
    fig, ax = plt.subplots()
    ax.plot(t, theta_reference, color="orange", label="Numerical Solution")
    ax.plot(
        t[data_indices],
        theta_reference[data_indices],
        "o",
        color="blue",
        label="Training Data",
    )
    prediction_line, = ax.plot(t, snapshots[0], color="red", label="Fourier PINN")
    ax.set(xlabel="Time (s)", ylabel="Angle (rad)")
    ax.legend()
    title = ax.set_title("")

    def update(frame):
        prediction_line.set_ydata(snapshots[frame])
        title.set_text(f"Pendulum Fourier PINN - Epoch {epochs[frame]}")
        return prediction_line, title

    movie = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=True
    )
    movie.save(
        OUTPUT_DIR / f"fpinn_training_{VERSION}.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_spectrum_animation(frequencies, reference, snapshots, epochs):
    fig, ax = plt.subplots()
    ax.plot(
        frequencies,
        reference,
        color="orange",
        label="Numerical FFT",
    )
    prediction_line, = ax.plot(
        frequencies, snapshots[0], color="red", label="Fourier PINN"
    )
    ax.set(
        xlabel="Angular frequency (rad/s)",
        ylabel=r"$|\Theta(\omega)|$",
        xlim=(0, SPECTRUM_XMAX),
        ylim=(0, SPECTRUM_YMAX),
    )
    ax.legend()
    title = ax.set_title("")

    def update(frame):
        prediction_line.set_ydata(snapshots[frame])
        title.set_text(f"Spectrum Evolution - Epoch {epochs[frame]}")
        return prediction_line, title

    movie = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=True
    )
    movie.save(
        OUTPUT_DIR / f"fpinn_spectrum_evolution_{VERSION}.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_figures(
    t,
    theta_reference,
    data_indices,
    theta_prediction,
    frequencies,
    spectrum_reference,
    spectrum_prediction,
    history,
):
    """Save the final time, spectrum, loss, and alpha figures."""

    fig, ax = plt.subplots()
    ax.plot(t, theta_reference, color="orange", label="Numerical Solution")
    ax.plot(
        t[data_indices],
        theta_reference[data_indices],
        "o",
        color="blue",
        label="Training Data",
    )
    ax.plot(t, theta_prediction, color="red", label="Fourier PINN Prediction")
    ax.set(
        xlabel="Time (s)",
        ylabel="Angle (rad)",
        title="Pendulum Fourier PINN",
    )
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"fpinn_results_{VERSION}.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(frequencies, spectrum_reference + 1e-12, color="orange", label="Numerical FFT")
    ax.plot(frequencies, spectrum_prediction + 1e-12, color="red", label="Fourier PINN")
    ax.set(
        xlabel="Angular frequency (rad/s)",
        ylabel=r"$|\Theta(\omega)|$",
        xlim=(0, SPECTRUM_XMAX),
        ylim=(0, SPECTRUM_YMAX),
    )
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"fpinn_spectrum_{VERSION}.png", dpi=600)
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
    fig.savefig(OUTPUT_DIR / f"fpinn_loss_{VERSION}.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epoch_axis, history["alpha"], color="darkcyan")
    ax.set(
        xlabel="Epochs",
        ylabel=r"$\alpha$",
        title=r"Learned $\alpha$ During Training",
    )
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / f"fpinn_alpha_{VERSION}.png", dpi=600)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU: {torch.get_num_threads()} threads")

    t, theta_reference, velocity_reference, dt = load_data(DATA_FILE)
    n_time = len(t)
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(n_time, d=dt)
    active_modes = min(MAX_PHYSICS_MODES, len(frequencies))
    spectrum_plot_mask = frequencies <= SPECTRUM_XMAX

    # Preserve the sparse measurement selection used in ver6.
    data_indices = np.arange(0, min(1000, n_time), 10)
    initial_mode, initial_real, initial_imaginary = estimate_initial_mode(
        t[data_indices], theta_reference[data_indices], frequencies[:active_modes]
    )

    omega = torch.tensor(frequencies, dtype=torch.float32, device=device)
    omega_active = omega[:active_modes]
    omega_input = omega_active[:, None]
    index_tensor = torch.tensor(data_indices, dtype=torch.long, device=device)
    theta_data = torch.tensor(
        theta_reference[data_indices], dtype=torch.float32, device=device
    )
    theta_initial = torch.tensor(theta_reference[0], dtype=torch.float32, device=device)
    velocity_initial = torch.tensor(
        velocity_reference[0], dtype=torch.float32, device=device
    )

    model = FourierPINN(
        omega_active,
        n_time,
        initial_mode,
        initial_real,
        initial_imaginary,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.network.parameters(), "lr": 1e-4},
            {"params": [model.spectral_coefficients], "lr": 1e-3},
            {"params": [model.alpha], "lr": 2e-4},
        ]
    )

    history = {
        name: []
        for name in ("total", "data", "physics", "initial", "energy", "alpha")
    }
    snapshot_epochs = []
    time_snapshots = []
    spectrum_snapshots = []

    print("Physics residual: time domain (centered finite differences)")
    print(
        f"Initial dominant mode: k={initial_mode}, "
        f"omega={frequencies[initial_mode]:.6f} rad/s"
    )
    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        spectrum, theta = reconstruct(model, omega_input, len(omega), n_time)
        velocity, acceleration = time_derivatives(theta, dt)

        data_loss = torch.mean((theta[index_tensor] - theta_data) ** 2)
        physics_residual = acceleration + model.alpha * torch.sin(theta[1:-1])
        physics_loss = torch.mean(physics_residual**2)
        initial_loss = (
            (theta[0] - theta_initial) ** 2
            + (velocity[0] - velocity_initial) ** 2
        )

        energy = 0.5 * velocity**2 + model.alpha * (1.0 - torch.cos(theta))
        initial_energy = 0.5 * velocity_initial**2 + model.alpha * (
            1.0 - torch.cos(theta_initial)
        )
        energy_loss = torch.mean((energy - initial_energy) ** 2)

        total_loss = (
            LAMBDA_DATA * data_loss
            + current_physics_weight(epoch) * physics_loss
            + LAMBDA_INITIAL * initial_loss
            + LAMBDA_ENERGY * energy_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        history["total"].append(total_loss.item())
        history["data"].append(data_loss.item())
        history["physics"].append(physics_loss.item())
        history["initial"].append(initial_loss.item())
        history["energy"].append(energy_loss.item())
        history["alpha"].append(model.alpha.item())

        if epoch % PRINT_EVERY == 0:
            elapsed = format_time(time.time() - start_time)
            print(
                f"\rEpoch {epoch:6d} | Loss {total_loss.item():.6e} | "
                f"alpha {model.alpha.item():.6f} | Time {elapsed}",
                end="",
                flush=True,
            )

        if epoch % SNAPSHOT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                spectrum_now, theta_now = reconstruct(
                    model, omega_input, len(omega), n_time
                )
            snapshot_epochs.append(epoch)
            time_snapshots.append(theta_now.cpu().numpy().copy())
            spectrum_snapshots.append(
                np.abs(spectrum_now.cpu().numpy())[spectrum_plot_mask].copy()
            )
            model.train()

    model.eval()
    with torch.no_grad():
        spectrum_final, theta_final = reconstruct(
            model, omega_input, len(omega), n_time
        )
    spectrum_final = np.abs(spectrum_final.cpu().numpy())
    theta_final = theta_final.cpu().numpy()
    spectrum_reference = np.abs(np.fft.rfft(theta_reference) / n_time)

    print(f"\nLearned alpha: {model.alpha.item():.6f}")
    print(f"Runtime: {format_time(time.time() - start_time)}")
    print("Saving figures and animations...")

    save_time_animation(
        t, theta_reference, data_indices, time_snapshots, snapshot_epochs
    )
    save_spectrum_animation(
        frequencies[spectrum_plot_mask],
        spectrum_reference[spectrum_plot_mask],
        spectrum_snapshots,
        snapshot_epochs,
    )
    save_figures(
        t,
        theta_reference,
        data_indices,
        theta_final,
        frequencies,
        spectrum_reference,
        spectrum_final,
        history,
    )

    output_names = [
        f"fpinn_training_{VERSION}.gif",
        f"fpinn_spectrum_evolution_{VERSION}.gif",
        f"fpinn_results_{VERSION}.png",
        f"fpinn_spectrum_{VERSION}.png",
        f"fpinn_loss_{VERSION}.png",
        f"fpinn_alpha_{VERSION}.png",
    ]
    print("Saved outputs:")
    for name in output_names:
        print(f"  {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
