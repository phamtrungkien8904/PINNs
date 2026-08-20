"""True Fourier-feature PINN for a nonlinear pendulum.

Architecture
------------
    time t
      -> learned periodic phase phi = 2*pi*(t-t0)/T
      -> Fourier features [sin(k*phi), cos(k*phi)]
      -> fully connected neural network
      -> predicted angle theta(t)

Physics
-------
    theta_tt + alpha*sin(theta) = 0,     alpha = g/L

The residual is evaluated on a uniform phase grid covering one learned period,
transformed by FFT, and minimized in the Fourier domain. Unlike a direct
Fourier-series fit, this file contains an actual dense neural network with a
configurable number of hidden layers and nodes.

Input file
----------
``pendulum_data.dat`` must contain a header followed by three columns:

    t    theta    omega

Required packages: numpy, matplotlib, torch, pillow
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


if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_FILE = "pendulum_data.dat"

# Same sparse-data selection as data[:600:10] in the original program.
DATA_STOP_INDEX = 1200
DATA_STRIDE = 40

# Fourier feature layer.
N_FOURIER_FEATURES = 20

# Actual neural-network size.
HIDDEN_NODES = 64
HIDDEN_LAYERS = 3

# Uniform phase points used for the FFT physics loss.
N_PHYSICS = 2048

# Learnable physical parameter alpha=g/L and oscillation period.
ALPHA_INIT = 10.0
PERIOD_INIT = 10.0
LEARN_ALPHA = True
LEARN_PERIOD = True

# The learned period is restricted to
# PERIOD_INIT*exp(-PERIOD_LOG_RANGE) ... PERIOD_INIT*exp(+PERIOD_LOG_RANGE).
PERIOD_LOG_RANGE = 0.5

# Training.
EPOCHS = 50_000
LEARNING_RATE = 1.0e-4
WARMUP_EPOCHS = 2_000
PHYSICS_RAMP_EPOCHS = 5_000
PRINT_EVERY = 100
SNAPSHOT_EVERY = 100

# Three normalized loss weights.
LAMB_DATA = 10.0
LAMB_PHYSICS = 1.0
LAMB_INIT = 100.0

# The original program only used theta at all data points and omega at t=0.
# Set this True to include every supplied omega point in the data loss.
USE_OMEGA_DATA = False
LAMB_OMEGA_DATA = 1.0

# Optional full-batch refinement after Adam.
USE_LBFGS = True
LBFGS_MAX_ITER = 500

SEED = 7
USE_FLOAT64 = True
USE_TEX = False


# =============================================================================
# SETUP
# =============================================================================

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
    print("LaTeX was not found. Falling back to Matplotlib math text.")
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


def inverse_softplus(value):
    if value <= 0.0:
        raise ValueError("ALPHA_INIT must be positive.")
    return math.log(math.expm1(value))


# =============================================================================
# LOAD DATA
# =============================================================================

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

t_data = torch.as_tensor(t_data_np, dtype=dtype, device=device).view(-1, 1)
theta_data = torch.as_tensor(theta_data_np, dtype=dtype, device=device).view(-1, 1)
omega_data = torch.as_tensor(omega_data_np, dtype=dtype, device=device).view(-1, 1)

print(f"Training points: {len(t_data_np)}")
print(f"Data interval: [{t_data_np.min():.6f}, {t_data_np.max():.6f}] s")
print(f"Prediction interval: [{t_min:.6f}, {t_max:.6f}] s")
print(f"Initial theta={theta_initial:.9f}, omega={omega_initial:.9f}")


# =============================================================================
# TRUE FOURIER-FEATURE NEURAL NETWORK
# =============================================================================

class FourierFeaturePINN(nn.Module):
    """Fourier encoding followed by a fully connected neural network."""

    def __init__(
        self,
        n_fourier_features,
        hidden_nodes,
        hidden_layers,
        time0,
        theta0,
        alpha_init,
        period_init,
        learn_alpha=True,
        learn_period=True,
        period_log_range=0.5,
    ):
        super().__init__()
        if n_fourier_features < 1:
            raise ValueError("N_FOURIER_FEATURES must be positive.")
        if hidden_nodes < 1 or hidden_layers < 1:
            raise ValueError("HIDDEN_NODES and HIDDEN_LAYERS must be positive.")
        if period_init <= 0.0:
            raise ValueError("PERIOD_INIT must be positive.")

        self.register_buffer(
            "harmonics",
            torch.arange(1, n_fourier_features + 1, dtype=dtype).view(1, -1),
        )
        self.register_buffer("time0", torch.tensor(time0, dtype=dtype))
        self.register_buffer("period0", torch.tensor(period_init, dtype=dtype))

        self.raw_alpha = nn.Parameter(
            torch.tensor(inverse_softplus(alpha_init), dtype=dtype),
            requires_grad=learn_alpha,
        )
        self.raw_period_shift = nn.Parameter(
            torch.tensor(0.0, dtype=dtype), requires_grad=learn_period
        )
        self.period_log_range = float(period_log_range)

        input_nodes = 2 * n_fourier_features
        layers = [nn.Linear(input_nodes, hidden_nodes), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh()])
        layers.append(nn.Linear(hidden_nodes, 1))
        self.network = nn.Sequential(*layers)

        # Xavier initialization for hidden layers. Start the final prediction
        # near theta0 while still allowing the data to determine the waveform.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.network[-1].bias, theta0)

    @property
    def alpha(self):
        # alpha=g/L must remain positive.
        return F.softplus(self.raw_alpha) + 1.0e-8

    @property
    def period(self):
        # Bounded positive period prevents collapse toward zero or infinity.
        bounded_shift = self.period_log_range * torch.tanh(
            self.raw_period_shift
        )
        return self.period0 * torch.exp(bounded_shift)

    @property
    def fundamental_frequency(self):
        return 2.0 * torch.pi / self.period

    def fourier_encode_phase(self, phase):
        harmonic_phase = phase * self.harmonics
        return torch.cat(
            (torch.sin(harmonic_phase), torch.cos(harmonic_phase)), dim=1
        )

    def forward_phase(self, phase):
        features = self.fourier_encode_phase(phase)
        return self.network(features)

    def forward(self, time_values):
        phase = self.fundamental_frequency * (time_values - self.time0)
        return self.forward_phase(phase)


model = FourierFeaturePINN(
    n_fourier_features=N_FOURIER_FEATURES,
    hidden_nodes=HIDDEN_NODES,
    hidden_layers=HIDDEN_LAYERS,
    time0=t_initial,
    theta0=theta_initial,
    alpha_init=ALPHA_INIT,
    period_init=PERIOD_INIT,
    learn_alpha=LEARN_ALPHA,
    learn_period=LEARN_PERIOD,
    period_log_range=PERIOD_LOG_RANGE,
).to(device=device, dtype=dtype)

trainable_parameters = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)
print(
    "Architecture: "
    f"{2*N_FOURIER_FEATURES} Fourier inputs -> "
    f"{HIDDEN_LAYERS} hidden layers x {HIDDEN_NODES} nodes -> 1 output"
)
print(f"Trainable parameters: {trainable_parameters}")


# A fixed uniform phase grid on [0, 2*pi). We make a differentiable copy in
# each loss evaluation because second derivatives with respect to phase are
# required. The endpoint is omitted because 0 and 2*pi are the same point.
phase_physics_base = (
    2.0
    * torch.pi
    * torch.arange(N_PHYSICS, dtype=dtype, device=device).view(-1, 1)
    / N_PHYSICS
)

# Fixed scales keep all losses dimensionless. PHYSICS_SCALE must not use the
# current learned alpha, otherwise alpha could change its own normalization.
ANGLE_SCALE = math.pi
VELOCITY_SCALE = max(float(np.std(omega_data_np)), 1.0)
PHYSICS_SCALE = max(ALPHA_INIT, 1.0)


def current_physics_multiplier(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.01
    progress = (epoch - WARMUP_EPOCHS) / max(PHYSICS_RAMP_EPOCHS, 1)
    return float(np.clip(0.01 + 0.99 * progress, 0.01, 1.0))


def calculate_losses(epoch, full_physics_weight=False):
    # ------------------------------------------------------------------
    # 1. Data loss
    # ------------------------------------------------------------------
    if USE_OMEGA_DATA:
        time_for_data = t_data.detach().clone().requires_grad_(True)
        theta_prediction = model(time_for_data)
        omega_prediction = torch.autograd.grad(
            theta_prediction,
            time_for_data,
            grad_outputs=torch.ones_like(theta_prediction),
            create_graph=True,
        )[0]
        omega_data_loss = torch.mean(
            ((omega_prediction - omega_data) / VELOCITY_SCALE) ** 2
        )
    else:
        theta_prediction = model(t_data)
        omega_data_loss = torch.zeros((), dtype=dtype, device=device)

    theta_data_loss = torch.mean(
        ((theta_prediction - theta_data) / ANGLE_SCALE) ** 2
    )
    data_loss = theta_data_loss + LAMB_OMEGA_DATA * omega_data_loss

    # ------------------------------------------------------------------
    # 2. Fourier-domain physics loss
    # ------------------------------------------------------------------
    phase = phase_physics_base.detach().clone().requires_grad_(True)
    theta_physics = model.forward_phase(phase)

    theta_phase = torch.autograd.grad(
        theta_physics,
        phase,
        grad_outputs=torch.ones_like(theta_physics),
        create_graph=True,
    )[0]
    theta_phase_phase = torch.autograd.grad(
        theta_phase,
        phase,
        grad_outputs=torch.ones_like(theta_phase),
        create_graph=True,
    )[0]

    # d2theta/dt2 = Omega^2 * d2theta/dphi2.
    theta_tt = model.fundamental_frequency**2 * theta_phase_phase
    residual = (theta_tt + model.alpha * torch.sin(theta_physics)) / PHYSICS_SCALE

    residual_fft = torch.fft.fft(residual.squeeze(-1), norm="ortho")
    physics_loss = torch.mean(torch.abs(residual_fft) ** 2)

    # ------------------------------------------------------------------
    # 3. Initial-condition loss
    # ------------------------------------------------------------------
    phase_initial = torch.zeros(
        (1, 1), dtype=dtype, device=device, requires_grad=True
    )
    theta_at_initial = model.forward_phase(phase_initial)
    theta_phase_initial = torch.autograd.grad(
        theta_at_initial,
        phase_initial,
        grad_outputs=torch.ones_like(theta_at_initial),
        create_graph=True,
    )[0]
    omega_at_initial = model.fundamental_frequency * theta_phase_initial

    initial_loss = torch.mean(
        ((theta_at_initial - theta_data[:1]) / ANGLE_SCALE) ** 2
    )
    initial_loss = initial_loss + torch.mean(
        ((omega_at_initial - omega_data[:1]) / VELOCITY_SCALE) ** 2
    )

    multiplier = 1.0 if full_physics_weight else current_physics_multiplier(epoch)
    total_loss = (
        LAMB_DATA * data_loss
        + multiplier * LAMB_PHYSICS * physics_loss
        + LAMB_INIT * initial_loss
    )

    return total_loss, data_loss, physics_loss, initial_loss, multiplier


# =============================================================================
# TRAINING
# =============================================================================

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
initial_loss_history = []
alpha_history = []
period_history = []

for epoch in range(EPOCHS + 1):
    optimizer.zero_grad(set_to_none=True)
    loss, data_loss, physics_loss, initial_loss, multiplier = calculate_losses(
        epoch
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
    optimizer.step()

    loss_history.append(loss.item())
    data_loss_history.append(data_loss.item())
    physics_loss_history.append(physics_loss.item())
    initial_loss_history.append(initial_loss.item())
    alpha_history.append(model.alpha.item())
    period_history.append(model.period.item())

    if epoch % PRINT_EVERY == 0:
        elapsed = time.time() - start_time
        print(
            f'\rEpoch {epoch:6d} | Loss={loss.item():.3e}, alpha={model.alpha.item():.6f} T={model.period.item():.6f}, Runtime={time_format(elapsed)}',
            end='',
            flush=True,
        )

    if epoch % SNAPSHOT_EVERY == 0:
        model.eval()
        with torch.no_grad():
            theta_snapshot = model(t_PINN_tensor).cpu().numpy().flatten()
        pinn_snapshots.append(theta_snapshot.copy())
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
        closure_loss = calculate_losses(
            EPOCHS, full_physics_weight=True
        )[0]
        closure_loss.backward()
        return closure_loss

    optimizer_lbfgs.step(closure)


elapsed_time = time.time() - start_time
final_loss, final_data, final_physics, final_initial, _ = calculate_losses(
    EPOCHS, full_physics_weight=True
)

print(f"\nFinal total loss: {final_loss.item():.3e}")
print(f"Final data loss: {final_data.item():.3e}")
print(f"Final Fourier physics loss: {final_physics.item():.3e}")
print(f"Final initial-condition loss: {final_initial.item():.3e}")
print(f"Learned alpha: {model.alpha.item():.9f}")
print(f"Learned period: {model.period.item():.9f} s")
print(f"Runtime: {time_format(elapsed_time)}")


# =============================================================================
# PREDICTION AND OUTPUTS
# =============================================================================

model.eval()
t_prediction_grad = t_PINN_tensor.detach().clone().requires_grad_(True)
theta_prediction_tensor = model(t_prediction_grad)
omega_prediction_tensor = torch.autograd.grad(
    theta_prediction_tensor,
    t_prediction_grad,
    grad_outputs=torch.ones_like(theta_prediction_tensor),
    create_graph=False,
)[0]

theta_PINN = theta_prediction_tensor.detach().cpu().numpy().flatten()
omega_PINN = omega_prediction_tensor.detach().cpu().numpy().flatten()

# np.savetxt(
#     "true_fourier_pinn_prediction.dat",
#     np.column_stack((t_PINN, theta_PINN, omega_PINN)),
#     header="t theta_fourier_pinn omega_fourier_pinn",
# )

# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "alpha": model.alpha.item(),
#         "period": model.period.item(),
#         "n_fourier_features": N_FOURIER_FEATURES,
#         "hidden_nodes": HIDDEN_NODES,
#         "hidden_layers": HIDDEN_LAYERS,
#         "time_initial": t_initial,
#         "theta_initial": theta_initial,
#         "omega_initial": omega_initial,
#     },
#     "true_fourier_pinn_model.pt",
# )


# Training animation.
if pinn_snapshots:
    pinn_snapshots.append(theta_PINN.copy())
    snapshot_epochs.append(EPOCHS)

    def update_frame(frame_index):
        pinn_line.set_ydata(pinn_snapshots[frame_index])
        ax_anim.set_title(
            f"Fourier-PINN training - Epoch {snapshot_epochs[frame_index]}"
        )
        return (pinn_line,)

    training_animation = animation.FuncAnimation(
        fig_anim,
        update_frame,
        frames=len(pinn_snapshots),
        blit=True,
    )
    training_animation.save(
        "true_fourier_pinn_training.gif",
        writer=animation.PillowWriter(fps=20),
    )
    print("Saved true_fourier_pinn_training.gif")
plt.close(fig_anim)


# Prediction plot.
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
plt.savefig("true_fourier_pinn_results.png", dpi=600)
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
plt.savefig("true_fourier_pinn_spectrum.png", dpi=600)
plt.close(fig)


# Loss histories.
epochs_array = np.arange(len(loss_history))
fig = plt.figure()
plt.semilogy(epochs_array, loss_history, label="Total Loss", color="black")
plt.semilogy(epochs_array, data_loss_history, label="Data Loss", color="blue")
plt.semilogy(
    epochs_array,
    physics_loss_history,
    label="Fourier Physics Loss",
    color="red",
)
plt.semilogy(
    epochs_array,
    initial_loss_history,
    label="Initial Condition Loss",
    color="green",
)
plt.xlabel("Epochs")
plt.ylabel("Normalized loss")
plt.title("Fourier-PINN Loss Convergence")
plt.legend()
plt.savefig("true_fourier_pinn_losses.png", dpi=600)
plt.close(fig)


# # Learned alpha and period.
# fig, axes = plt.subplots(2, 1, figsize=(10 / 2.54, 9 / 2.54))
# axes[0].plot(epochs_array, alpha_history, color="purple")
# axes[0].set_xlabel("Epochs")
# axes[0].set_ylabel(r"Learned $\alpha$")
# axes[1].plot(epochs_array, period_history, color="teal")
# axes[1].set_xlabel("Epochs")
# axes[1].set_ylabel(r"Period $T$ (s)")
# fig.savefig("true_fourier_pinn_parameters.png", dpi=600)
# plt.close(fig)



print("Saved prediction, model, plots, residual spectrum, and animation.")
