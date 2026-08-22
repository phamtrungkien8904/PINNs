import numpy as np
import matplotlib

matplotlib.use("Agg")  # Save files only

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Settings
# ============================================================

n_modes = 128
hidden_nodes = 128
pretrain_epochs = 2_000
epochs = 30_000
animation_every = 1000

alpha_init = 8.0
learning_rate = 1e-4
alpha_relaxation = 0.02

lambda_data = 1e2
lambda_physics = 1e1
lambda_init = 1e3

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data and Fourier grid
# ============================================================

data = np.loadtxt("pendulum_data.dat", skiprows=1)

# Use [0,T), because FFT must not contain both periodic endpoints.
data = data[:-1]

t_num = data[:, 0]
theta_num = data[:, 1]
velocity_num = data[:, 2]

N = len(t_num)
dt = t_num[1] - t_num[0]

if not np.allclose(np.diff(t_num), dt):
    raise ValueError("Time samples must be uniformly spaced.")

# Same sparse measurements as the original code.
idx = np.arange(0, min(600, N), 30)
t_data = t_num[idx]

idx_tensor = torch.tensor(idx, dtype=torch.long, device=device)
theta_data = torch.tensor(theta_num[idx], dtype=torch.float32, device=device)
theta_0 = torch.tensor(theta_num[0], dtype=torch.float32, device=device)
velocity_0 = torch.tensor(velocity_num[0], dtype=torch.float32, device=device)

omega_full_np = 2 * np.pi * np.fft.rfftfreq(N, d=dt)
omega_full = torch.tensor(omega_full_np, dtype=torch.float32, device=device)

# Learn only the low-frequency part containing the pendulum peak.
n_modes = min(n_modes, len(omega_full))
omega = omega_full[:n_modes].view(-1, 1)

# True FFT is only plotted after training; it is not a training target.
Theta_true = np.fft.rfft(theta_num) / N


def estimate_initial_mode():
    """Estimate the dominant FFT bin from the sparse measurements.

    This prevents alpha_init from choosing (and locking) the wrong Fourier
    mode.  No full-solution FFT values are used for training.
    """

    errors = np.full(n_modes, np.inf)
    coefficients = np.zeros((n_modes, 2))

    for k in range(1, n_modes):
        wk = omega_full_np[k]
        design = np.column_stack(
            (np.cos(wk * t_data), np.sin(wk * t_data))
        )
        coefficient, *_ = np.linalg.lstsq(
            design, theta_num[idx], rcond=None
        )
        coefficients[k] = coefficient
        errors[k] = np.mean(
            (design @ coefficient - theta_num[idx]) ** 2
        )

    k0 = int(np.argmin(errors))
    cosine_amplitude, sine_amplitude = coefficients[k0]
    return k0, cosine_amplitude, sine_amplitude


initial_mode, initial_cosine, initial_sine = estimate_initial_mode()

# For theta(t) = A*cos(wt) + B*sin(wt), the positive-frequency
# normalized FFT coefficient is (A - i*B)/2.
initial_real = initial_cosine / 2.0
initial_imaginary = -initial_sine / 2.0


# ============================================================
# Linear-Tanh Fourier neural network
# ============================================================

class FourierTanhPINN(nn.Module):
    """omega -> [Re(Theta), Im(Theta)] using Linear-Tanh-Linear."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 2),
        )

        raw_alpha = np.log(np.expm1(alpha_init))
        # alpha is updated by a stable one-dimensional least-squares
        # projection below.  It must not compete with all NN weights in Adam.
        self.raw_alpha = nn.Parameter(
            torch.tensor(raw_alpha, dtype=torch.float32),
            requires_grad=False,
        )

        self.initialize_physical_peak()

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def scale_frequency(self, w):
        # Important: scale by the maximum ACTIVE frequency, not Nyquist.
        return 2 * w / omega[-1] - 1

    def forward(self, w):
        return self.network(self.scale_frequency(w))

    def initialize_physical_peak(self):
        """Initialize two Tanh neurons as a narrow spectral bump."""

        first_layer = self.network[0]
        output_layer = self.network[2]

        nn.init.xavier_uniform_(first_layer.weight)
        nn.init.zeros_(first_layer.bias)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        k0 = initial_mode

        x_grid = 2 * omega.flatten() / omega[-1] - 1
        x0 = x_grid[k0].item()
        dx = (x_grid[1] - x_grid[0]).item()

        # 0.5*(tanh(s*(x-left))-tanh(s*(x-right))) is one narrow bump.
        left = x0 - dx / 2
        right = x0 + dx / 2
        steepness = 8.0 / dx

        real_amplitude = initial_real
        imaginary_amplitude = initial_imaginary

        with torch.no_grad():
            first_layer.weight[0, 0] = steepness
            first_layer.bias[0] = -steepness * left

            first_layer.weight[1, 0] = steepness
            first_layer.bias[1] = -steepness * right

            output_layer.weight[0, 0] = real_amplitude / 2
            output_layer.weight[0, 1] = -real_amplitude / 2

            output_layer.weight[1, 0] = imaginary_amplitude / 2
            output_layer.weight[1, 1] = -imaginary_amplitude / 2

model = FourierTanhPINN().to(device)


def predict():
    output = model(omega)

    # DC must have zero imaginary part.
    imaginary_mask = torch.ones(n_modes, device=device)
    imaginary_mask[0] = 0.0
    if N % 2 == 0 and n_modes == len(omega_full):
        imaginary_mask[-1] = 0.0

    Theta_active = torch.complex(
        output[:, 0],
        output[:, 1] * imaginary_mask,
    )

    # Frequencies above the active low-frequency band are exactly zero.
    zero_modes = torch.zeros(
        len(omega_full) - n_modes,
        dtype=torch.complex64,
        device=device,
    )
    Theta = torch.cat((Theta_active, zero_modes))

    # Network output is the normalized spectrum: Theta = FFT(theta)/N.
    theta = torch.fft.irfft(N * Theta, n=N)
    velocity = torch.fft.irfft(N * 1j * omega_full * Theta, n=N)
    acceleration = torch.fft.irfft(
        -N * omega_full**2 * Theta,
        n=N,
    )

    return Theta_active, Theta, theta, velocity, acceleration


def update_alpha(acceleration, theta):
    """Relax alpha toward the exact least-squares value for current theta.

    For fixed theta, mean((theta_ddot + alpha*sin(theta))**2) is a
    one-variable quadratic.  Solving that quadratic avoids alpha oscillation.
    """

    with torch.no_grad():
        sine_theta = torch.sin(theta)
        denominator = torch.mean(sine_theta**2).clamp_min(1e-12)
        alpha_target = -torch.mean(acceleration * sine_theta) / denominator
        alpha_target = alpha_target.clamp(1e-3, 50.0)

        alpha_new = (
            (1.0 - alpha_relaxation) * model.alpha
            + alpha_relaxation * alpha_target
        )

        # Stable inverse of softplus: x = y + log(1-exp(-y)).
        raw_alpha_new = alpha_new + torch.log(-torch.expm1(-alpha_new))
        model.raw_alpha.copy_(raw_alpha_new)


# ============================================================
# Short physical-spectrum pretraining
# ============================================================

# The peak is selected from the sparse measurements, not from alpha_init.
k0 = initial_mode
Theta_initial = torch.zeros(n_modes, 2, device=device)
Theta_initial[k0, 0] = initial_real
Theta_initial[k0, 1] = initial_imaginary

pretrain_optimizer = torch.optim.Adam(
    model.network.parameters(),
    lr=5e-4,
)

for epoch in range(pretrain_epochs):
    pretrain_optimizer.zero_grad(set_to_none=True)
    output = model(omega)
    pretrain_loss = torch.sum((output - Theta_initial) ** 2)
    pretrain_loss.backward()
    pretrain_optimizer.step()

# ============================================================
# Fourier PINN training
# ============================================================

optimizer = torch.optim.Adam(
    model.network.parameters(),
    lr=learning_rate,
)

total_history = []
data_history = []
physics_history = []
init_history = []
alpha_history = []

theta_snapshots = []
spectrum_snapshots = []
snapshot_epochs = []

for epoch in range(epochs + 1):
    optimizer.zero_grad(set_to_none=True)

    (
        Theta_active,
        Theta,
        theta_prediction,
        velocity_prediction,
        acceleration_prediction,
    ) = predict()

    # Alternating inverse step: update only alpha for the current Fourier
    # signal, then update only the Linear-Tanh network with Adam.
    update_alpha(acceleration_prediction, theta_prediction)

    # 1. Sparse time-domain measurements
    data_loss = torch.mean(
        (theta_prediction[idx_tensor] - theta_data) ** 2
    )

    # 2. Nonlinear pendulum physics.  theta'' is still calculated by
    # Fourier differentiation, but sin(theta) must be evaluated in time.
    # Keeping sin(theta) is necessary to identify the generating alpha=10.
    physics_residual = (
        acceleration_prediction
        + model.alpha * torch.sin(theta_prediction)
    )
    physics_loss = torch.mean(physics_residual**2)

    # 3. Initial conditions after inverse FFT
    init_loss = (
        (theta_prediction[0] - theta_0) ** 2
        + (velocity_prediction[0] - velocity_0) ** 2
    )

    loss = (
        lambda_data * data_loss
        + lambda_physics * physics_loss
        + lambda_init * init_loss
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.network.parameters(), max_norm=10.0
    )
    optimizer.step()

    total_history.append(loss.item())
    data_history.append(data_loss.item())
    physics_history.append(physics_loss.item())
    init_history.append(init_loss.item())
    alpha_history.append(model.alpha.item())

    # Store arrays only; render all figures after training.
    if epoch % animation_every == 0:
        theta_snapshots.append(
            theta_prediction.detach().cpu().numpy().copy()
        )
        spectrum_snapshots.append(
            torch.abs(Theta_active).detach().cpu().numpy().copy() + 1e-12
        )
        snapshot_epochs.append(epoch)


# ============================================================
# Final results
# ============================================================

model.eval()
with torch.no_grad():
    (
        Theta_active,
        Theta,
        theta_PINN,
        velocity_PINN,
        acceleration_PINN,
    ) = predict()

theta_PINN = theta_PINN.cpu().numpy()
velocity_PINN = velocity_PINN.cpu().numpy()
Theta_PINN = Theta.cpu().numpy()

alpha_learned = model.alpha.item()
period_learned = 2 * np.pi / np.sqrt(alpha_learned)
rmse = np.sqrt(np.mean((theta_PINN - theta_num) ** 2))

# Time-domain PNG
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
ax.plot(t_num, theta_PINN, label="Tanh Fourier PINN", color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_tanh_pinn_result.png", dpi=300)
plt.close(fig)


# Fourier-domain PNG
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    omega_full_np[:n_modes],
    np.abs(Theta_true[:n_modes]) + 1e-12,
    label="FFT of numerical solution",
    color="orange",
)
ax.plot(
    omega_full_np[:n_modes],
    np.abs(Theta_PINN[:n_modes]) + 1e-12,
    label="Tanh Fourier PINN",
    color="red",
)
ax.axvline(np.sqrt(alpha_learned), color="black", ls="--", label="sqrt(alpha)")
ax.set_xlabel("Angular frequency omega (rad/s)")
ax.set_ylabel("|Theta(omega)|")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_tanh_pinn_spectrum.png", dpi=300)
plt.close(fig)


# Loss PNG
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(total_history, label="Total", color="black")
ax.semilogy(data_history, label="Data", color="blue")
ax.semilogy(physics_history, label="Physics", color="red")
ax.semilogy(init_history, label="Initial condition", color="green")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_tanh_pinn_losses.png", dpi=300)
plt.close(fig)


# Learned-alpha PNG
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(alpha_history, color="purple", label="Identified alpha")
ax.axhline(10.0, color="black", ls="--", label="Reference alpha = 10")
ax.set_xlabel("Epoch")
ax.set_ylabel("alpha (s^-2)")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_tanh_pinn_alpha.png", dpi=300)
plt.close(fig)


# Time-domain GIF
fig_time, ax_time = plt.subplots(figsize=(8, 4))
ax_time.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax_time.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
time_line, = ax_time.plot(
    t_num,
    theta_snapshots[0],
    label="Tanh Fourier PINN",
    color="red",
)
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Angle (rad)")
ax_time.legend()


def update_time(frame):
    time_line.set_ydata(theta_snapshots[frame])
    ax_time.set_title(f"Time domain - Epoch {snapshot_epochs[frame]}")
    return time_line,


time_animation = animation.FuncAnimation(
    fig_time,
    update_time,
    frames=len(theta_snapshots),
    blit=False,
)
time_animation.save(
    "fourier_tanh_pinn_time.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_time)


# Fourier-domain GIF
fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
ax_spectrum.plot(
    omega_full_np[:n_modes],
    np.abs(Theta_true[:n_modes]) + 1e-12,
    label="FFT of numerical solution",
    color="orange",
)
spectrum_line, = ax_spectrum.plot(
    omega_full_np[:n_modes],
    spectrum_snapshots[0],
    label="Tanh Fourier PINN",
    color="red",
)
ax_spectrum.set_ylim(1e-8, 1)
ax_spectrum.set_xlabel("Angular frequency omega (rad/s)")
ax_spectrum.set_ylabel("|Theta(omega)|")
ax_spectrum.legend()


def update_spectrum(frame):
    spectrum_line.set_ydata(spectrum_snapshots[frame])
    ax_spectrum.set_title(f"Fourier domain - Epoch {snapshot_epochs[frame]}")
    return spectrum_line,


spectrum_animation = animation.FuncAnimation(
    fig_spectrum,
    update_spectrum,
    frames=len(spectrum_snapshots),
    blit=False,
)
spectrum_animation.save(
    "fourier_tanh_pinn_spectrum.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_spectrum)


# np.savetxt(
#     "fourier_tanh_pinn_prediction.dat",
#     np.column_stack((t_num, theta_PINN, velocity_PINN)),
#     header="time theta_PINN velocity_PINN",
# )

# np.savetxt(
#     "fourier_tanh_pinn_summary.dat",
#     np.array([[alpha_learned, period_learned, rmse]]),
#     header="alpha period_seconds theta_rmse",
# )