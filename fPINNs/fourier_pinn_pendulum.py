import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Save files only; never open plot windows

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn

# Custom settings
plt.style.use('classic')

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'''
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage[utf8]{inputenc}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{siunitx}
    \usepackage{sfmath}
    '''
})
plt.rcParams.update({
    # Figure settings
    'figure.dpi': 300,
    'figure.figsize': (10/2.54, 6/2.54),  # 10x6 cm in inches (1 figure per line)
    # 'figure.figsize': (8/2.54, 6/2.54),  # 10x6 cm in inches (2 figures per line)
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.labelcolor': 'black',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    # 'mathtext.fontset': 'cm',
    'figure.constrained_layout.use': True,

    # Ticks
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
    "xtick.minor.width":0,
    "ytick.minor.width": 0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,  

    # Legend
    'legend.frameon': False,
    'legend.title_fontsize': 8,
    'legend.fontsize': 8,
    'legend.handlelength': 2,
    'legend.loc': 'best',
    'legend.numpoints': 1,

    # Line style
    'lines.linestyle': '-',
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'lines.markeredgecolor': 'white',
    'lines.markeredgewidth': 0.5,
})

# Time formatting for runtime display
def time_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CPU: {torch.get_num_threads()} threads")


# ============================================================
# Data
# ============================================================

data = np.loadtxt("pendulum_data.dat", skiprows=1)
t_num = data[:, 0]
theta_num = data[:, 1]
velocity_num = data[:, 2]

# Same 20 training points as the original code
idx = np.arange(0, 600, 30)
t_data = t_num[idx]
idx_tensor = torch.tensor(idx, dtype=torch.long, device=device)
theta_data = torch.tensor(theta_num[idx], dtype=torch.float32, device=device)

theta_0 = torch.tensor(theta_num[0], dtype=torch.float32, device=device)
velocity_0 = torch.tensor(velocity_num[0], dtype=torch.float32, device=device)


# ============================================================
# Fourier grid
# ============================================================

N = len(t_num)
dt = t_num[1] - t_num[0]

# rFFT gives the non-negative frequencies
omega = 2 * np.pi * np.fft.rfftfreq(N, d=dt)
omega = torch.tensor(omega, dtype=torch.float32, device=device).view(-1, 1)


# ============================================================
# Neural network: omega -> [real Theta, imaginary Theta]
# ============================================================

class FourierPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

        # alpha = g/L
        self.alpha = nn.Parameter(torch.tensor(9.0))

        # Begin with Theta(omega) = 0
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, w):
        w_scaled = 2 * w / omega[-1] - 1
        return self.network(w_scaled)


model = FourierPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ============================================================
# Training
# ============================================================

epochs = 100_000

lambda_data = 1e2
lambda_physics = 1e1
lambda_init = 1e3

loss_history = []

# Store snapshots during training; create animations only after training
animation_every = 500
theta_snapshots = []
spectrum_snapshots = []
snapshot_epochs = []


start_time = time.time()
for epoch in range(epochs + 1):
    optimizer.zero_grad()

    # Network prediction in the Fourier domain
    output = model(omega)
    Theta = torch.complex(output[:, 0], output[:, 1])

    # The network predicts normalized coefficients Theta = FFT(theta)/N
    theta_prediction = torch.fft.irfft(N * Theta, n=N)

    # F[theta'] = i*omega*Theta
    velocity_prediction = torch.fft.irfft(
        N * 1j * omega.flatten() * Theta,
        n=N,
    )

    # 1. Data loss in the time domain
    data_loss = torch.mean((theta_prediction[idx_tensor] - theta_data) ** 2)

    # 2. Physics loss directly in the Fourier domain
    # theta'' + alpha*theta = 0
    # (-omega^2 + alpha)*Theta = 0
    physics_residual = (model.alpha - omega.flatten() ** 2) * Theta
    physics_loss = torch.mean(torch.abs(physics_residual) ** 2)

    # 3. Initial-condition loss in the time domain
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

    elapsed_time = time.time() - start_time
    if epoch % 100 == 0:
        print(
            f'\rEpoch {epoch:6d}, loss = {loss.item():.6e}, alpha = {model.alpha.item():.6f}, Runtime = {time_format(elapsed_time)}',
            end='',
            flush=True
        )

    # Store time-domain and Fourier-domain frames without displaying anything
    if epoch % animation_every == 0:
        theta_snapshots.append(theta_prediction.detach().cpu().numpy().copy())
        spectrum_snapshots.append(
            torch.abs(Theta).detach().cpu().numpy().copy() + 1e-12
        )
        snapshot_epochs.append(epoch)


# ============================================================
# Prediction and plots
# ============================================================

model.eval()
with torch.no_grad():
    output = model(omega)
    Theta = torch.complex(output[:, 0], output[:, 1])
    theta_PINN = torch.fft.irfft(N * Theta, n=N).cpu().numpy()

print("\nLearned alpha =", model.alpha.item())
print(f'Runtime: {time_format(elapsed_time)}')

# Time-domain training animation
fig_time, ax_time = plt.subplots()
ax_time.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax_time.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
time_line, = ax_time.plot(t_num, theta_snapshots[0], label="Fourier PINN", color="red")
theta_min = min(theta_num.min(), min(x.min() for x in theta_snapshots))
theta_max = max(theta_num.max(), max(x.max() for x in theta_snapshots))
theta_padding = 0.05 * max(theta_max - theta_min, 1e-6)
ax_time.set_ylim(theta_min - theta_padding, theta_max + theta_padding)
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Angle (rad)")
ax_time.legend()


def update_time_animation(frame):
    time_line.set_ydata(theta_snapshots[frame])
    ax_time.set_title(f"Time domain - Epoch {snapshot_epochs[frame]}")
    return time_line,


time_animation = animation.FuncAnimation(
    fig_time,
    update_time_animation,
    frames=len(theta_snapshots),
    blit=False,
)
time_animation.save(
    "fourier_pinn_time_training.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_time)


# Fourier-domain training animation
omega_numpy = omega.cpu().numpy().flatten()
n_plot_modes = min(50, len(omega_numpy))
fig_spectrum, ax_spectrum = plt.subplots()
spectrum_line, = ax_spectrum.plot(
    omega_numpy[:n_plot_modes],
    spectrum_snapshots[0][:n_plot_modes],
    color="red",
)
ax_spectrum.set_yscale("log")
ax_spectrum.set_ylim(
    1e-12,
    1.2 * max(snapshot[:n_plot_modes].max() for snapshot in spectrum_snapshots),
)
ax_spectrum.set_xlabel("Angular frequency omega (rad/s)")
ax_spectrum.set_ylabel("|Theta(omega)|")


def update_spectrum_animation(frame):
    spectrum_line.set_ydata(spectrum_snapshots[frame][:n_plot_modes])
    ax_spectrum.set_title(f"Fourier domain - Epoch {snapshot_epochs[frame]}")
    return spectrum_line,


spectrum_animation = animation.FuncAnimation(
    fig_spectrum,
    update_spectrum_animation,
    frames=len(spectrum_snapshots),
    blit=False,
)
spectrum_animation.save(
    "fourier_pinn_spectrum_training.gif",
    writer=animation.PillowWriter(fps=15),
)
plt.close(fig_spectrum)

# Time-domain prediction
plt.plot(t_num, theta_num, label="Numerical solution", color="orange")
plt.plot(t_data, theta_num[idx], "o", label="Training data", color="blue")
plt.plot(t_num, theta_PINN, label="Fourier PINN", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.savefig("fourier_pinn_result.png", dpi=300)
plt.close()

# Fourier spectrum
plt.semilogy(
    omega.cpu().numpy().flatten(),
    torch.abs(Theta).cpu().numpy() + 1e-12,
)
plt.xlabel("Angular frequency omega (rad/s)")
plt.ylabel("|Theta(omega)|")
plt.savefig("fourier_pinn_spectrum.png", dpi=300)
plt.close()

# Training loss
plt.semilogy(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("fourier_pinn_loss.png", dpi=300)
plt.close()
