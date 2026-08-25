# Fourier PINN for the pendulum
import time
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CPU: {torch.get_num_threads()} threads")

plt.style.use("classic")


def time_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


torch.manual_seed(0)
np.random.seed(0)

data = np.loadtxt("pendulum_data.dat", skiprows=1)
t_num = data[:, 0]
theta_num = data[:, 1]
velocity_num = data[:, 2]

N = len(t_num)
dt = t_num[1] - t_num[0]


# Keep the original sparse measurement selection.
data_indices = np.arange(0, min(1000, N), 10)
t_data_np = t_num[data_indices].copy()
theta_data_np = theta_num[data_indices].copy()
omega_data_np = velocity_num[data_indices].copy()

idx_tensor = torch.tensor(data_indices, dtype=torch.long, device=device)
theta_data = torch.tensor(theta_data_np, dtype=torch.float32, device=device)
theta_0 = torch.tensor(theta_num[0], dtype=torch.float32, device=device)
omega_0 = torch.tensor(velocity_num[0], dtype=torch.float32, device=device)

# The Fourier grid is the grid required by irfft. Theta is normalized by N.
omegafreq_np = 2 * np.pi * np.fft.rfftfreq(N, d=dt)
omegafreq = torch.tensor(omegafreq_np, dtype=torch.float32, device=device)
physics_modes = min(128, len(omegafreq_np))
omegafreq_active = omegafreq[:physics_modes]
omegafreq_input = omegafreq_active.view(-1, 1)

alpha_init = 0.0


def estimate_initial_mode():
    errors = np.full(physics_modes, np.inf)
    coefficients = np.zeros((physics_modes, 2))

    for mode in range(1, physics_modes):
        frequency = omegafreq_np[mode]
        design = np.column_stack(
            (np.cos(frequency * t_data_np), np.sin(frequency * t_data_np))
        )
        coefficient, *_ = np.linalg.lstsq(
            design, theta_data_np, rcond=None
        )
        coefficients[mode] = coefficient
        errors[mode] = np.mean(
            (design @ coefficient - theta_data_np) ** 2
        )

    mode = int(np.argmin(errors))
    return mode, coefficients[mode, 0], coefficients[mode, 1]


initial_mode, initial_cosine, initial_sine = estimate_initial_mode()
initial_real = initial_cosine / 2.0
initial_imaginary = -initial_sine / 2.0


class FourierPINN(nn.Module):
    """Map angular frequency to the real and imaginary parts of Theta."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init), dtype=torch.float32)
        )

        self.initialize_physical_peak()

    def forward(self, omega):
        omega_scaled = 2 * omega / omegafreq_active[-1] - 1
        output = self.network(omega_scaled)

        # DC and Nyquist coefficients must be real for a real irfft signal.
        imaginary_mask = torch.ones(
            output.shape[0], dtype=output.dtype, device=output.device
        )
        imaginary_mask[0] = 0.0
        if N % 2 == 0 and len(omega) == len(omegafreq):
            imaginary_mask[-1] = 0.0

        return torch.complex(output[:, 0], output[:, 1] * imaginary_mask)

    def initialize_physical_peak(self):
        first_layer = self.network[0]
        output_layer = self.network[-1]

        nn.init.xavier_uniform_(first_layer.weight)
        nn.init.zeros_(first_layer.bias)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        x_grid = 2 * omegafreq_active / omegafreq_active[-1] - 1
        x0 = x_grid[initial_mode].item()
        dx = (x_grid[1] - x_grid[0]).item()
        steepness = 8.0 / dx
        left = x0 - dx / 2
        right = x0 + dx / 2

        with torch.no_grad():
            first_layer.weight[0, 0] = steepness
            first_layer.bias[0] = -steepness * left
            first_layer.weight[1, 0] = steepness
            first_layer.bias[1] = -steepness * right
            output_layer.weight[0, 0] = initial_real / 2
            output_layer.weight[0, 1] = -initial_real / 2
            output_layer.weight[1, 0] = initial_imaginary / 2
            output_layer.weight[1, 1] = -initial_imaginary / 2


model = FourierPINN().to(device)


def predict_from_fourier():
    """Predict Theta and reconstruct theta, velocity, and acceleration."""

    Theta_active = model(omegafreq_input)
    zero_modes = torch.zeros(
        len(omegafreq) - physics_modes,
        dtype=Theta_active.dtype,
        device=device,
    )
    Theta = torch.cat((Theta_active, zero_modes))
    theta = torch.fft.irfft(N * Theta, n=N)
    velocity = torch.fft.irfft(N * 1j * omegafreq * Theta, n=N)
    acceleration = torch.fft.irfft(
        -N * omegafreq**2 * Theta,
        n=N,
    )
    return Theta, theta, velocity, acceleration


# Prepare the original time-domain animation, now fed by inverse-FFT output.
t_PINN = np.linspace(t_num.min(), t_num.max(), N)
fig_anim, ax_anim = plt.subplots()
ax_anim.plot(t_num, theta_num, label="Numerical Solution", color="orange")
ax_anim.plot(t_data_np, theta_data_np, "o", label="Training Data", color="blue")
pinn_line, = ax_anim.plot(
    t_PINN,
    np.zeros_like(t_PINN),
    label="Fourier PINN Prediction",
    color="red",
)
ax_anim.set_xlabel("Time (s)")
ax_anim.set_ylabel("Angle (rad)")
ax_anim.set_title("Pendulum Fourier PINN")
ax_anim.legend()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10000
animate_every = 100
print_every = 100
lamb_data = 1e-1
lamb_physics = 1e-2
lamb_init = 1e-3
lamb_energy = 1e-10

pinn_snapshots = []
loss_history = []
data_loss_history = []
physics_loss_history = []
init_loss_history = []
energy_loss_history = []

start_time = time.time()
for epoch in range(epochs + 1):
    optimizer.zero_grad()

    Theta, theta_prediction, velocity_prediction, _ = predict_from_fourier()

    # Data loss is computed from the inverse Fourier reconstruction.
    data_loss = torch.mean(
        (theta_prediction[idx_tensor] - theta_data) ** 2
    )

    # Fourier physics residual: (alpha - omega^2) Theta(omega).
    residual_physics = (
        model.alpha - omegafreq_active**2
    ) * Theta[:physics_modes]
    residual_scale = model.alpha + omegafreq_active[-1] ** 2
    residual_physics = residual_physics / residual_scale
    physics_loss = torch.mean(torch.abs(residual_physics) ** 2)

    # Initial conditions are also computed from the inverse Fourier reconstruction.
    residual_init = (
        (theta_prediction[0] - theta_0) ** 2
        + (velocity_prediction[0] - omega_0) ** 2
    )
    init_loss = torch.mean(residual_init)

    # Preserve the original optional energy regularization in time domain.
    energy_prediction = (
        0.5 * velocity_prediction**2
        + model.alpha * (1 - torch.cos(theta_prediction))
    )
    energy_initial = 0.5 * omega_0**2 + model.alpha * (1 - torch.cos(theta_0))
    energy_loss = torch.mean((energy_prediction - energy_initial) ** 2)

    loss = (
        lamb_data * data_loss
        + lamb_physics * physics_loss
        + lamb_init * init_loss
        + lamb_energy * energy_loss
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    loss_history.append(loss.item())
    data_loss_history.append(data_loss.item())
    physics_loss_history.append(physics_loss.item())
    init_loss_history.append(init_loss.item())
    energy_loss_history.append(energy_loss.item())

    elapsed_time = time.time() - start_time
    if epoch % print_every == 0:
        print(f'\rEpoch {epoch}, Loss: {loss.item():.6f}, alpha: {model.alpha.item():.6f}, Time: {time_format(elapsed_time)}',
              end='', flush=True)

    if epoch % animate_every == 0:
        model.eval()
        with torch.no_grad():
            theta_PINN_now = predict_from_fourier()[1].cpu().numpy()
        pinn_snapshots.append(theta_PINN_now.copy())
        pinn_line.set_ydata(theta_PINN_now)
        ax_anim.set_title(f"Pendulum Fourier PINN - Epoch {epoch}")
        model.train()


alpha_learned = model.alpha.item()
model.eval()
with torch.no_grad():
    Theta_PINN, theta_PINN_tensor, _, _ = predict_from_fourier()
theta_PINN = theta_PINN_tensor.cpu().numpy()
Theta_PINN = Theta_PINN.cpu().numpy()


print(f"\nLearned alpha: {alpha_learned:.6f}")
print(f"Runtime: {time_format(time.time() - start_time)}")


if pinn_snapshots:
    def update_frame(i):
        pinn_line.set_ydata(pinn_snapshots[i])
        ax_anim.set_title(f"Pendulum Fourier PINN - Epoch {i * print_every}")
        return pinn_line,

    anim = animation.FuncAnimation(
        fig_anim, update_frame, frames=len(pinn_snapshots), blit=True
    )
    anim.save(
        "fpinn_training_ver6.gif",
        writer=animation.PillowWriter(fps=30),
    )
plt.close(fig_anim)


plt.plot(t_num, theta_num, label="Numerical Solution", color="orange")
plt.plot(
    t_data_np,
    theta_data_np,
    label="Training Data",
    color="blue",
    marker="o",
    ls="None",
)
plt.plot(t_num, theta_PINN, label="Fourier PINN Prediction", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Pendulum Fourier PINN")
plt.legend()
plt.savefig("fpinn_results_ver6.png", dpi=600)
plt.close()

plt.figure()
plt.plot(
    omegafreq_np,
    np.abs(np.fft.rfft(theta_num) / N) + 1e-12,
    label="Numerical FFT",
    color="orange",
)
plt.plot(
    omegafreq_np,
    np.abs(Theta_PINN) + 1e-12,
    label="Fourier PINN",
    color="red",
)
plt.xlabel("Angular frequency (rad/s)")
plt.ylabel(r"$|\Theta(\omega)|$")
plt.xlim(0, 10)
plt.ylim(0, 0.5)
plt.legend()
plt.savefig("fpinn_spectrum_ver6.png", dpi=600)
plt.close()

plt.figure()
epochs_arr = np.arange(len(loss_history))
plt.semilogy(epochs_arr, loss_history, label="Total Loss", color="black")
plt.semilogy(epochs_arr, data_loss_history, label="Data Loss", color="blue")
plt.semilogy(epochs_arr, physics_loss_history, label="Physics Loss", color="red")
plt.semilogy(
    epochs_arr,
    init_loss_history,
    label="Initial Condition Loss",
    color="green",
)
plt.semilogy(epochs_arr, energy_loss_history, label="Energy Loss", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence")
plt.legend()
plt.savefig("fpinn_loss_ver6.png", dpi=600)
plt.close()
