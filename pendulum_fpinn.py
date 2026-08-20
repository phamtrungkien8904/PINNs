#!/usr/bin/env python3
"""Fourier-domain PINN for the nonlinear pendulum.

This is a spectral version of the supplied time-domain MLP PINN. It preserves
the original pendulum_data.dat input, learned alpha, plots, loss histories, and
training GIF. The principal changes are:

1. theta(t) is represented by a trainable Fourier series.
2. theta(0) and omega(0) are satisfied exactly by eliminating coefficients.
3. The ODE residual is sampled uniformly over one phase period and transformed
   with an orthonormal FFT before its spectral energy is minimized.
4. alpha is constrained positive and the learned period is constrained near a
   user-supplied initial estimate.

Required packages:
    numpy matplotlib torch pillow
"""

import math
import os
import shutil
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Use a non-interactive backend on an HPC node without a display.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

DATA_FILE = "pendulum_data.dat"

# Discrete training subset, equivalent to data[:600:10] in the original code.
DATA_STOP_INDEX = 1300
DATA_STRIDE = 10

# Fourier representation and collocation grid.
N_MODES = 40
N_PHYSICS = 2048

# Initial estimates. For theta(0)=0.999*pi, L=1 m and g=9.81 m/s^2,
# the nonlinear period is approximately 10 s.
ALPHA_INIT = 10
PERIOD_INIT = 10
LEARN_ALPHA = True
LEARN_PERIOD = True

# The trainable period is restricted to
# PERIOD_INIT * exp(-PERIOD_LOG_RANGE) ... exp(+PERIOD_LOG_RANGE).
PERIOD_LOG_RANGE = 0.5

# Optimizer settings. The Fourier model normally needs far fewer epochs than
# the original 500000-epoch MLP.
EPOCHS = 100_000
LEARNING_RATE = 1.0e-4
WARMUP_EPOCHS = 2_000
PHYSICS_RAMP_EPOCHS = 5_000
PRINT_EVERY = 100
SNAPSHOT_EVERY = 100

# Normalized loss weights.
LAMB_DATA = 10.0
LAMB_PHYSICS = 1.0
LAMB_INIT = 100.0
LAMB_SMOOTH = 1.0e-6

# Set this to True if every supplied angular-velocity point should also be used
# as training data. With False, omega_data[0] is still used as the exact IC.
USE_OMEGA_DATA = False
LAMB_OMEGA_DATA = 1.0

# L-BFGS is useful for final full-batch refinement.
USE_LBFGS = True
LBFGS_MAX_ITER = 500

SEED = 7
USE_FLOAT64 = True
USE_TEX = False  # Set True only if LaTeX and all preamble packages are present.


# -----------------------------------------------------------------------------
# REPRODUCIBILITY, DEVICE, AND PLOTTING
# -----------------------------------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

dtype = torch.float64 if USE_FLOAT64 else torch.float32
torch.set_default_dtype(dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CPU: {torch.get_num_threads()} threads")

if USE_TEX and shutil.which("latex") is None:
    print("Warning: LaTeX was requested but not found; using Matplotlib math text.")
    USE_TEX = False

plt.style.use("classic")
plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
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
        "lines.linestyle": "-",
        "lines.linewidth": 1,
        "lines.markersize": 4,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.5,
    }
)


def time_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def positive_inverse_softplus(value):
    """Return x such that softplus(x) is approximately value."""
    if value <= 0.0:
        raise ValueError("The initial value must be positive.")
    return math.log(math.expm1(value))


# -----------------------------------------------------------------------------
# LOAD DISCRETE DATA
# -----------------------------------------------------------------------------

data = np.loadtxt(DATA_FILE, skiprows=1)
if data.ndim != 2 or data.shape[1] < 3:
    raise ValueError("pendulum_data.dat must contain t, theta, and omega columns.")

t_num = data[:, 0]
theta_num = data[:, 1]
omega_num = data[:, 2]

t_data_np = data[:DATA_STOP_INDEX:DATA_STRIDE, 0].copy()
theta_data_np = data[:DATA_STOP_INDEX:DATA_STRIDE, 1].copy()
omega_data_np = data[:DATA_STOP_INDEX:DATA_STRIDE, 2].copy()

if len(t_data_np) < 3:
    raise ValueError("At least three discrete training points are required.")

t_min = float(t_num.min())
t_max = float(t_num.max())
t_initial = float(t_data_np[0])
theta_initial = float(theta_data_np[0])
omega_initial = float(omega_data_np[0])

print(f"Training points: {len(t_data_np)}")
print(f"Initial condition: theta={theta_initial:.9f}, omega={omega_initial:.9f}")

t_data = torch.as_tensor(t_data_np, dtype=dtype, device=device).view(-1, 1)
theta_data = torch.as_tensor(theta_data_np, dtype=dtype, device=device).view(-1, 1)
omega_data = torch.as_tensor(omega_data_np, dtype=dtype, device=device).view(-1, 1)

fig = plt.figure()
plt.plot(
    t_data_np,
    theta_data_np,
    label=r"Training Data",
    color="blue",
    marker="o",
    ls="None",
)
plt.plot(t_num, theta_num, label=r"Numerical Solution", color="orange")
plt.xlabel(r"Time (s)")
plt.ylabel(r"Angle (rad)")
plt.title(r"Pendulum training data")
plt.legend()
# plt.savefig("fourier_pinn_training_data.png", dpi=600)
plt.close(fig)


# -----------------------------------------------------------------------------
# FOURIER PINN
# -----------------------------------------------------------------------------

class FourierPINN(nn.Module):
    """Periodic Fourier model with theta(t0) and omega(t0) enforced exactly.

    The general K-mode series is

        theta(phi) = a0 + sum_k [a_k cos(k phi) + b_k sin(k phi)],
        phi = Omega * (t-t0).

    a0 and b1 are eliminated using the two initial conditions. Therefore the
    trainable coefficients cannot violate theta(t0) or omega(t0).
    """

    def __init__(
        self,
        n_modes,
        theta0,
        omega0,
        time0,
        alpha_init,
        period_init,
        learn_alpha=True,
        learn_period=True,
        period_log_range=0.5,
    ):
        super().__init__()
        if n_modes < 2:
            raise ValueError("N_MODES must be at least 2.")
        if period_init <= 0.0:
            raise ValueError("PERIOD_INIT must be positive.")

        self.register_buffer("k", torch.arange(1, n_modes + 1).view(1, -1))
        self.register_buffer("k_tail", torch.arange(2, n_modes + 1).view(1, -1))
        self.register_buffer("theta0", torch.tensor(theta0, dtype=dtype))
        self.register_buffer("omega0", torch.tensor(omega0, dtype=dtype))
        self.register_buffer("time0", torch.tensor(time0, dtype=dtype))
        self.register_buffer("period0", torch.tensor(period_init, dtype=dtype))

        self.cosine_coefficients = nn.Parameter(
            0.01 * torch.randn(1, n_modes, dtype=dtype)
        )
        self.sine_tail_coefficients = nn.Parameter(
            0.01 * torch.randn(1, n_modes - 1, dtype=dtype)
        )

        self.raw_alpha = nn.Parameter(
            torch.tensor(positive_inverse_softplus(alpha_init), dtype=dtype),
            requires_grad=learn_alpha,
        )
        self.raw_period_shift = nn.Parameter(
            torch.tensor(0.0, dtype=dtype), requires_grad=learn_period
        )
        self.period_log_range = float(period_log_range)

    @property
    def alpha(self):
        # Positive alpha prevents an unphysical negative g/L.
        return F.softplus(self.raw_alpha) + 1.0e-8

    @property
    def period(self):
        shift = self.period_log_range * torch.tanh(self.raw_period_shift)
        return self.period0 * torch.exp(shift)

    @property
    def fundamental_frequency(self):
        return 2.0 * torch.pi / self.period

    def all_sine_coefficients(self):
        # omega(t0) = Omega * sum_k(k*b_k) = omega0.
        weighted_tail = torch.sum(
            self.k_tail * self.sine_tail_coefficients, dim=1, keepdim=True
        )
        b1 = self.omega0 / self.fundamental_frequency - weighted_tail
        return torch.cat((b1.view(1, 1), self.sine_tail_coefficients), dim=1)

    def fields_from_phase(self, phase):
        """Return theta, dtheta/dt, and d2theta/dt2 at phase values."""
        harmonic_phase = phase * self.k
        sine_coefficients = self.all_sine_coefficients()

        # theta(t0)=a0+sum(a_k)=theta0, so eliminate a0.
        constant_coefficient = self.theta0 - torch.sum(
            self.cosine_coefficients, dim=1, keepdim=True
        )

        theta = constant_coefficient + torch.sum(
            self.cosine_coefficients * torch.cos(harmonic_phase)
            + sine_coefficients * torch.sin(harmonic_phase),
            dim=1,
            keepdim=True,
        )

        theta_phase = torch.sum(
            -self.k * self.cosine_coefficients * torch.sin(harmonic_phase)
            + self.k * sine_coefficients * torch.cos(harmonic_phase),
            dim=1,
            keepdim=True,
        )

        theta_phase_phase = torch.sum(
            -(self.k**2) * self.cosine_coefficients * torch.cos(harmonic_phase)
            - (self.k**2) * sine_coefficients * torch.sin(harmonic_phase),
            dim=1,
            keepdim=True,
        )

        angular_frequency = self.fundamental_frequency
        theta_t = angular_frequency * theta_phase
        theta_tt = angular_frequency**2 * theta_phase_phase
        return theta, theta_t, theta_tt

    def fields(self, time_values):
        phase = self.fundamental_frequency * (time_values - self.time0)
        return self.fields_from_phase(phase)

    def forward(self, time_values):
        return self.fields(time_values)[0]


model = FourierPINN(
    n_modes=N_MODES,
    theta0=theta_initial,
    omega0=omega_initial,
    time0=t_initial,
    alpha_init=ALPHA_INIT,
    period_init=PERIOD_INIT,
    learn_alpha=LEARN_ALPHA,
    learn_period=LEARN_PERIOD,
    period_log_range=PERIOD_LOG_RANGE,
).to(device=device, dtype=dtype)


def initialize_fourier_coefficients(model, times, angles, ridge=1.0e-6):
    """Initialize Fourier coefficients by constrained ridge least squares."""
    k = np.arange(1, N_MODES + 1, dtype=np.float64)
    k_tail = np.arange(2, N_MODES + 1, dtype=np.float64)
    frequency = 2.0 * np.pi / PERIOD_INIT
    phase = frequency * (times - t_initial)

    cosine_design = np.cos(phase[:, None] * k[None, :]) - 1.0
    sine_design = (
        np.sin(phase[:, None] * k_tail[None, :])
        - k_tail[None, :] * np.sin(phase[:, None])
    )
    fixed_velocity_part = (omega_initial / frequency) * np.sin(phase)
    target = angles - theta_initial - fixed_velocity_part
    design = np.column_stack((cosine_design, sine_design))

    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ target)

    with torch.no_grad():
        model.cosine_coefficients.copy_(
            torch.as_tensor(
                coefficients[:N_MODES], dtype=dtype, device=device
            ).view(1, -1)
        )
        model.sine_tail_coefficients.copy_(
            torch.as_tensor(
                coefficients[N_MODES:], dtype=dtype, device=device
            ).view(1, -1)
        )


initialize_fourier_coefficients(model, t_data_np, theta_data_np)

# A uniform phase grid on [0, 2*pi) is required for the FFT. The endpoint is
# excluded because phase 0 and phase 2*pi are the same periodic point.
phase_physics = (
    2.0
    * torch.pi
    * torch.arange(N_PHYSICS, dtype=dtype, device=device).view(-1, 1)
    / N_PHYSICS
)

# Fixed normalization scales make the lambda values interpretable and prevent
# a learnable alpha from changing its own loss denominator.
ANGLE_SCALE = math.pi
VELOCITY_SCALE = max(float(np.std(omega_data_np)), 1.0)
PHYSICS_SCALE = max(ALPHA_INIT, 1.0)


def physics_ramp(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.01
    fraction = (epoch - WARMUP_EPOCHS) / max(PHYSICS_RAMP_EPOCHS, 1)
    return float(np.clip(0.01 + 0.99 * fraction, 0.01, 1.0))


def calculate_losses(epoch, full_physics_weight=False):
    theta_prediction, omega_prediction, _ = model.fields(t_data)

    angle_data_loss = torch.mean(
        ((theta_prediction - theta_data) / ANGLE_SCALE) ** 2
    )
    if USE_OMEGA_DATA:
        omega_data_loss = torch.mean(
            ((omega_prediction - omega_data) / VELOCITY_SCALE) ** 2
        )
    else:
        omega_data_loss = torch.zeros((), dtype=dtype, device=device)
    data_loss = angle_data_loss + LAMB_OMEGA_DATA * omega_data_loss

    # Physics residual evaluated over exactly one learned Fourier period.
    theta_physics, _, acceleration_physics = model.fields_from_phase(
        phase_physics
    )
    residual_physics = (
        acceleration_physics + model.alpha * torch.sin(theta_physics)
    ) / PHYSICS_SCALE

    # Fourier-domain residual. With norm='ortho', Parseval gives
    # mean(|FFT(r)|^2) = mean(|r|^2).
    residual_spectrum = torch.fft.fft(
        residual_physics.squeeze(-1), norm="ortho"
    )
    physics_loss = torch.mean(torch.abs(residual_spectrum) ** 2)

    # This is calculated for monitoring. It stays at round-off level because
    # the Fourier coefficient elimination enforces both ICs exactly.
    theta_ic, omega_ic, _ = model.fields(t_data[:1])
    init_loss = torch.mean(((theta_ic - theta_data[:1]) / ANGLE_SCALE) ** 2)
    init_loss = init_loss + torch.mean(
        ((omega_ic - omega_data[:1]) / VELOCITY_SCALE) ** 2
    )

    relative_mode = model.k / model.k.max()
    sine_coefficients = model.all_sine_coefficients()
    smooth_loss = torch.mean(
        relative_mode**4
        * (
            (model.cosine_coefficients / ANGLE_SCALE) ** 2
            + (sine_coefficients / ANGLE_SCALE) ** 2
        )
    )

    ramp = 1.0 if full_physics_weight else physics_ramp(epoch)
    total_loss = (
        LAMB_DATA * data_loss
        + ramp * LAMB_PHYSICS * physics_loss
        + LAMB_INIT * init_loss
        + LAMB_SMOOTH * smooth_loss
    )
    return (
        total_loss,
        data_loss,
        physics_loss,
        init_loss,
        smooth_loss,
        ramp,
    )


# -----------------------------------------------------------------------------
# TRAINING AND ANIMATION SNAPSHOTS
# -----------------------------------------------------------------------------

start_time = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

t_PINN = np.linspace(t_min, t_max, 1000)
t_PINN_tensor = torch.as_tensor(t_PINN, dtype=dtype, device=device).view(-1, 1)

fig_anim, ax_anim = plt.subplots()
ax_anim.plot(t_num, theta_num, label=r"Numerical Solution", color="orange")
ax_anim.plot(
    t_data_np,
    theta_data_np,
    "o",
    label=r"Training Data",
    color="blue",
)
pinn_line, = ax_anim.plot(
    t_PINN,
    np.zeros_like(t_PINN),
    label=r"Fourier-PINN Prediction",
    color="red",
)
ax_anim.set_xlabel(r"Time (s)")
ax_anim.set_ylabel(r"Angle (rad)")
ax_anim.legend()

pinn_snapshots = []
snapshot_epochs = []
loss_history = []
data_loss_history = []
physics_loss_history = []
init_loss_history = []
alpha_history = []
period_history = []

for epoch in range(EPOCHS + 1):
    optimizer.zero_grad(set_to_none=True)
    (
        loss,
        data_loss,
        physics_loss,
        init_loss,
        smooth_loss,
        ramp,
    ) = calculate_losses(epoch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
    optimizer.step()

    loss_history.append(loss.item())
    data_loss_history.append(data_loss.item())
    physics_loss_history.append(physics_loss.item())
    init_loss_history.append(init_loss.item())
    alpha_history.append(model.alpha.item())
    period_history.append(model.period.item())

    if epoch % PRINT_EVERY == 0:
        elapsed_time = time.time() - start_time
        print(
            f'\rEpoch {epoch:6d} | total={loss.item():.3e}, alpha={model.alpha.item():.6f} T={model.period.item():.6f}, Runtime={time_format(elapsed_time)}',
            end='',
            flush=True,
        )

    if epoch % SNAPSHOT_EVERY == 0:
        model.eval()
        with torch.no_grad():
            theta_PINN_now = model(t_PINN_tensor).cpu().numpy().flatten()
        pinn_snapshots.append(theta_PINN_now.copy())
        snapshot_epochs.append(epoch)
        model.train()


if USE_LBFGS and LBFGS_MAX_ITER > 0:
    print("\nStarting L-BFGS refinement...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=LBFGS_MAX_ITER,
        max_eval=int(1.25 * LBFGS_MAX_ITER),
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
        total_loss = calculate_losses(EPOCHS, full_physics_weight=True)[0]
        total_loss.backward()
        return total_loss

    optimizer_lbfgs.step(closure)


elapsed_time = time.time() - start_time
final_losses = calculate_losses(EPOCHS, full_physics_weight=True)
alpha_learned = model.alpha.item()
period_learned = model.period.item()

print(f"\nLearned alpha: {alpha_learned:.9f}")
print(f"Learned period: {period_learned:.9f} s")
print(f"Final data loss: {final_losses[1].item():.3e}")
print(f"Final Fourier physics loss: {final_losses[2].item():.3e}")
print(f"Final initial-condition loss: {final_losses[3].item():.3e}")
print(f"Runtime: {time_format(elapsed_time)}")


# -----------------------------------------------------------------------------
# SAVE PREDICTION, MODEL, GIF, AND DIAGNOSTIC PLOTS
# -----------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    theta_PINN, omega_PINN, _ = model.fields(t_PINN_tensor)
    theta_PINN = theta_PINN.cpu().numpy().flatten()
    omega_PINN = omega_PINN.cpu().numpy().flatten()

# np.savetxt(
#     "fourier_pinn_prediction.dat",
#     np.column_stack((t_PINN, theta_PINN, omega_PINN)),
#     header="t theta_fourier_pinn omega_fourier_pinn",
# )


if pinn_snapshots:
    # Add the final post-L-BFGS result as the last frame.
    pinn_snapshots.append(theta_PINN.copy())
    snapshot_epochs.append(EPOCHS)

    def update_frame(frame_index):
        pinn_line.set_ydata(pinn_snapshots[frame_index])
        ax_anim.set_title(
            f"Fourier-PINN training - Epoch {snapshot_epochs[frame_index]}"
        )
        return (pinn_line,)

    anim = animation.FuncAnimation(
        fig_anim,
        update_frame,
        frames=len(pinn_snapshots),
        blit=True,
    )
    anim.save("fourier_pinn_training.gif", writer=animation.PillowWriter(fps=20))
    print("Saved animation to fourier_pinn_training.gif")
plt.close(fig_anim)

fig = plt.figure()
plt.plot(t_num, theta_num, label=r"Numerical Solution", color="orange")
plt.plot(
    t_data_np,
    theta_data_np,
    label=r"Training Data",
    color="blue",
    marker="o",
    ls="None",
)
plt.plot(t_PINN, theta_PINN, label=r"Fourier-PINN Prediction", color="red")
plt.xlabel(r"Time (s)")
plt.ylabel(r"Angle (rad)")
plt.title(r"Nonlinear Pendulum Fourier-PINN")
plt.legend()
plt.savefig("fourier_pinn_pendulum_results.png", dpi=600)
plt.close(fig)

# Convert the predicted angle from the time domain to the frequency domain.
theta_PINN_frequency = np.fft.rfft(theta_PINN)
frequency = np.fft.rfftfreq(
    len(theta_PINN), d=float(t_PINN[1] - t_PINN[0])
)
theta_PINN_spectrum = np.abs(theta_PINN_frequency) / len(theta_PINN)
if len(theta_PINN_spectrum) > 1:
    theta_PINN_spectrum[1:-1] *= 2.0

fig = plt.figure()
plt.plot(frequency, theta_PINN_spectrum, color="red", marker="o", ls="-")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 2.0)
plt.ylabel(r"$|\hat{\theta}(f)|$")
plt.title("Fourier-PINN Prediction in the Frequency Domain")
plt.savefig("fourier_pinn_prediction_spectrum.png", dpi=600)
plt.close(fig)

epochs_arr = np.arange(len(loss_history))
fig = plt.figure()
plt.semilogy(epochs_arr, loss_history, label="Total Loss", color="black")
plt.semilogy(epochs_arr, data_loss_history, label="Data Loss", color="blue")
plt.semilogy(
    epochs_arr, physics_loss_history, label="Fourier Physics Loss", color="red"
)
plt.semilogy(
    epochs_arr,
    np.maximum(init_loss_history, np.finfo(float).tiny),
    label="Initial Condition Loss",
    color="green",
)
plt.xlabel("Epochs")
plt.ylabel("Normalized loss")
plt.title("Loss Convergence")
plt.legend()
plt.savefig("fourier_pinn_loss.png", dpi=600)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(10 / 2.54, 9 / 2.54))
axes[0].plot(epochs_arr, alpha_history, color="purple")
axes[0].set_ylabel(r"Learned $\alpha$")
axes[0].set_xlabel("Epochs")
axes[1].plot(epochs_arr, period_history, color="teal")
axes[1].set_ylabel(r"Period $T$ (s)")
axes[1].set_xlabel("Epochs")
# fig.savefig("fourier_pinn_parameters.png", dpi=600)
plt.close(fig)

# Final residual spectrum: which harmonics still violate the ODE?
with torch.no_grad():
    theta_phase, _, acceleration_phase = model.fields_from_phase(phase_physics)
    residual = (
        acceleration_phase + model.alpha * torch.sin(theta_phase)
    ) / PHYSICS_SCALE
    residual_fft = torch.fft.rfft(residual.squeeze(-1), norm="ortho")
    residual_amplitude = torch.abs(residual_fft).cpu().numpy()

fig = plt.figure()
harmonic_index = np.arange(len(residual_amplitude))
plt.semilogy(
    harmonic_index,
    np.maximum(residual_amplitude, np.finfo(float).tiny),
    color="darkred",
)
plt.xlim(0, min(4 * N_MODES, len(residual_amplitude) - 1))
plt.xlabel("Harmonic index")
plt.ylabel(r"$|\hat{r}_k|$")
plt.title("Final ODE Residual Spectrum")
# plt.savefig("fourier_pinn_residual_spectrum.png", dpi=600)
plt.close(fig)

print("Saved Fourier-PINN prediction, model, plots, and animation.")
