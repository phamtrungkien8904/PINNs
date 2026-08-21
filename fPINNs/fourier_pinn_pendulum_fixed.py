import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Never open plot windows

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Settings
# ============================================================

epochs = 100_000
animation_every = 1000
n_modes = 64

lambda_data = 1e2
lambda_physics = 1e1
lambda_init = 1e3

alpha_init = 10.0  # g/L used to generate pendulum_data.dat
learning_rate = 1e-3

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# Data
# ============================================================

data = np.loadtxt("pendulum_data.dat", skiprows=1)

# FFT assumes samples on [0,T), not both t=0 and t=T.
# The original data contain 5001 points from 0 to 50 s, so remove t=50 s.
data = data[:-1]

t_num = data[:, 0]
theta_num = data[:, 1]
velocity_num = data[:, 2]

N = len(t_num)
dt = t_num[1] - t_num[0]

if not np.allclose(np.diff(t_num), dt):
    raise ValueError("The time samples must be uniformly spaced for FFT.")

# Same 20 sparse training points as the original code
idx = np.arange(0, 600, 30)
t_data = t_num[idx]

idx_tensor = torch.tensor(idx, dtype=torch.long, device=device)
theta_data = torch.tensor(theta_num[idx], dtype=torch.float32, device=device)
theta_0 = torch.tensor(theta_num[0], dtype=torch.float32, device=device)
velocity_0 = torch.tensor(velocity_num[0], dtype=torch.float32, device=device)


# ============================================================
# Fourier grid
# ============================================================

omega_full_np = 2 * np.pi * np.fft.rfftfreq(N, d=dt)
omega_full = torch.tensor(omega_full_np, dtype=torch.float32, device=device)

n_modes = min(n_modes, len(omega_full))
omega = omega_full[:n_modes].view(-1, 1)

# True FFT is used only for plotting, never as training data.
Theta_true = np.fft.rfft(theta_num) / N


# ============================================================
# Fourier PINN
# ============================================================

class FourierPINN(nn.Module):
    """RBF network: omega_k -> [Re(Theta_k), Im(Theta_k)].

    A normal Tanh MLP produces a smooth spectrum and cannot represent the
    isolated peak of a free pendulum. The narrow RBF neurons allow every FFT
    frequency bin to have an independent complex coefficient.
    """

    def __init__(self, frequency_grid, theta_initial, velocity_initial):
        super().__init__()

        centers = frequency_grid.flatten().clone()
        self.register_buffer("centers", centers)

        delta_omega = centers[1] - centers[0]
        self.register_buffer("sigma", 0.20 * delta_omega)

        # Trainable real and imaginary amplitudes of the RBF neurons
        self.spectrum_weights = nn.Parameter(
            torch.zeros(len(centers), 2, dtype=torch.float32)
        )

        # Positive learnable alpha = g/L
        raw_alpha = np.log(np.expm1(alpha_init))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

        # Physically informed initialization at omega approximately sqrt(alpha)
        k0 = torch.argmin(torch.abs(centers - np.sqrt(alpha_init))).item()
        omega_0 = centers[k0].item()

        with torch.no_grad():
            self.spectrum_weights[k0, 0] = theta_initial / 2
            self.spectrum_weights[k0, 1] = -velocity_initial / (2 * omega_0)

        print(
            f"Initial mode: k={k0}, omega={omega_0:.6f} rad/s, "
            f"period={2*np.pi/omega_0:.6f} s"
        )

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, frequency):
        # One narrow RBF neuron is centered at every active FFT frequency.
        distance = frequency - self.centers.view(1, -1)
        rbf = torch.exp(-0.5 * (distance / self.sigma) ** 2)
        return rbf @ self.spectrum_weights


model = FourierPINN(omega, theta_0.item(), velocity_0.item()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def predict():
    """Return spectrum, angle, and angular velocity."""

    output = model(omega)

    # DC must be real for a real time-domain signal.
    imaginary_mask = torch.ones(n_modes, device=device)
    imaginary_mask[0] = 0.0

    Theta_active = torch.complex(
        output[:, 0],
        output[:, 1] * imaginary_mask,
    )

    # Frequencies above n_modes are fixed to zero.
    zero_modes = torch.zeros(
        len(omega_full) - n_modes,
        dtype=torch.complex64,
        device=device,
    )
    Theta = torch.cat((Theta_active, zero_modes))

    # Theta contains normalized coefficients: Theta = FFT(theta)/N.
    theta = torch.fft.irfft(N * Theta, n=N)
    velocity = torch.fft.irfft(N * 1j * omega_full * Theta, n=N)

    return Theta_active, Theta, theta, velocity


# One-sided Parseval weights for a real signal
parseval_weights = torch.full((n_modes,), 2.0, device=device)
parseval_weights[0] = 1.0


# ============================================================
# Training
# ============================================================

loss_history = []
data_history = []
physics_history = []
init_history = []
alpha_history = []

theta_snapshots = []
spectrum_snapshots = []
snapshot_epochs = []

start_time = time.time()

for epoch in range(epochs + 1):
    optimizer.zero_grad(set_to_none=True)

    Theta_active, Theta, theta_prediction, velocity_prediction = predict()

    # 1. Sparse measurement loss after inverse FFT
    data_loss = torch.mean(
        (theta_prediction[idx_tensor] - theta_data) ** 2
    )

    # 2. Linearized pendulum physics directly in Fourier space
    # theta'' + alpha*theta = 0
    # (alpha - omega^2)*Theta = 0
    residual_hat = (
        model.alpha - omega.flatten() ** 2
    ) * Theta_active

    physics_loss = torch.sum(
        parseval_weights * torch.abs(residual_hat) ** 2
    )

    # 3. Initial angle and angular velocity after inverse FFT
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
    optimizer.step()

    loss_history.append(loss.item())
    data_history.append(data_loss.item())
    physics_history.append(physics_loss.item())
    init_history.append(init_loss.item())
    alpha_history.append(model.alpha.item())

    if epoch % 100 == 0:
        elapsed = time.time() - start_time
        print(
            f"\rEpoch {epoch:6d} | total={loss.item():.3e} "
            f"| data={data_loss.item():.3e} "
            f"| physics={physics_loss.item():.3e} "
            f"| IC={init_loss.item():.3e} "
            f"| alpha={model.alpha.item():.6f} "
            f"| time={elapsed:.1f}s",
            end="",
            flush=True,
        )

    # Store arrays only. All figures and GIFs are rendered after training.
    if epoch % animation_every == 0:
        theta_snapshots.append(
            theta_prediction.detach().cpu().numpy().copy()
        )
        spectrum_snapshots.append(
            torch.abs(Theta_active).detach().cpu().numpy().copy() + 1e-12
        )
        snapshot_epochs.append(epoch)


# ============================================================
# Final prediction
# ============================================================

model.eval()
with torch.no_grad():
    Theta_active, Theta, theta_PINN, velocity_PINN = predict()

theta_PINN = theta_PINN.cpu().numpy()
velocity_PINN = velocity_PINN.cpu().numpy()
Theta_PINN = Theta.cpu().numpy()

alpha_learned = model.alpha.item()
period_learned = 2 * np.pi / np.sqrt(alpha_learned)
rmse = np.sqrt(np.mean((theta_PINN - theta_num) ** 2))

print()
print(f"Learned alpha:  {alpha_learned:.8f} s^-2")
print(f"Learned period: {period_learned:.8f} s")
print(f"Full-time RMSE: {rmse:.8e} rad")
print("Rendering PNG and GIF files...")


# ============================================================
# Final PNG files
# ============================================================

# Time-domain result
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
ax.plot(t_num, theta_PINN, label="Fourier PINN", color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_pinn_fixed_result.png", dpi=300)
plt.close(fig)

# Fourier-domain result: show the correct FFT for comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(
    omega_full_np[:n_modes],
    np.abs(Theta_true[:n_modes]) + 1e-12,
    label="FFT of numerical solution",
    color="orange",
)
ax.semilogy(
    omega_full_np[:n_modes],
    np.abs(Theta_PINN[:n_modes]) + 1e-12,
    label="Fourier PINN",
    color="red",
)
ax.axvline(np.sqrt(alpha_learned), color="black", ls="--", label="sqrt(alpha)")
ax.set_xlabel("Angular frequency omega (rad/s)")
ax.set_ylabel("|Theta(omega)|")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_pinn_fixed_spectrum.png", dpi=300)
plt.close(fig)

# Separate loss components
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(loss_history, label="Total", color="black")
ax.semilogy(data_history, label="Data", color="blue")
ax.semilogy(physics_history, label="Physics", color="red")
ax.semilogy(init_history, label="Initial condition", color="green")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
fig.savefig("fourier_pinn_fixed_losses.png", dpi=300)
plt.close(fig)


# ============================================================
# GIF files, rendered only after training
# ============================================================

# Time-domain animation
fig_time, ax_time = plt.subplots(figsize=(8, 4))
ax_time.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax_time.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
time_line, = ax_time.plot(
    t_num,
    theta_snapshots[0],
    label="Fourier PINN",
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
    "fourier_pinn_fixed_time.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_time)


# Fourier-domain animation with the true FFT shown as reference
fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
ax_spectrum.semilogy(
    omega_full_np[:n_modes],
    np.abs(Theta_true[:n_modes]) + 1e-12,
    label="FFT of numerical solution",
    color="orange",
)
spectrum_line, = ax_spectrum.semilogy(
    omega_full_np[:n_modes],
    spectrum_snapshots[0],
    label="Fourier PINN",
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
    "fourier_pinn_fixed_spectrum.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_spectrum)


# # Save numerical predictions and learned parameters
# np.savetxt(
#     "fourier_pinn_fixed_prediction.dat",
#     np.column_stack((t_num, theta_PINN, velocity_PINN)),
#     header="time theta_PINN velocity_PINN",
# )

# with open("fourier_pinn_fixed_parameters.txt", "w", encoding="utf-8") as file:
#     file.write(f"alpha = {alpha_learned:.12e}\n")
#     file.write(f"period = {period_learned:.12e}\n")
#     file.write(f"RMSE = {rmse:.12e}\n")

print("Finished.")
