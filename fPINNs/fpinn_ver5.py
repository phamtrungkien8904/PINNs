import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Do not open figures while the job is running.

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


# ============================================================
# User settings
# ============================================================

data_file = "pendulum_data.dat"

# theta(t) is represented by the odd harmonics 1, 3, ..., max_harmonic.
# For an undamped pendulum oscillating around zero, even harmonics and DC are
# zero.  Harmonics 1, 3 and 5 are therefore explicitly available to the PINN.
max_harmonic = 15
hidden_nodes = 64

# k = 0, ..., sine_taylor_order.  The default keeps theta through theta^15.
# Degree 7 is not accurate enough when |theta| is close to pi.
sine_taylor_order = 7

adam_epochs = 15_000
lbfgs_max_iterations = 500
learning_rate_network = 1.0e-3
learning_rate_frequency = 2.0e-4
alpha_relaxation = 0.05

# All losses below are nondimensionalized before applying these weights.
lambda_data = 20.0
lambda_physics = 1.0
lambda_init = 5.0
lambda_spectrum_decay = 1.0e-5

alpha_init = 10.0
training_end_time = 12.0
number_training_data = 50

animation_every = 200
print_every = 100
save_gifs = True

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def time_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ============================================================
# Data
# ============================================================

if max_harmonic < 5 or max_harmonic % 2 == 0:
    raise ValueError("max_harmonic must be an odd integer >= 5.")

data_path = Path(data_file)
if not data_path.exists():
    data_path = Path(__file__).resolve().parent / data_file
if not data_path.exists():
    raise FileNotFoundError(
        f"Cannot find {data_file}. Put it beside this script or run the "
        "script from the directory containing the data file."
    )

data = np.loadtxt(data_path, skiprows=1)
t_num = data[:, 0]
theta_num = data[:, 1]
velocity_num = data[:, 2]

if len(t_num) < 4:
    raise ValueError("The data file must contain at least four samples.")
if not np.all(np.diff(t_num) > 0.0):
    raise ValueError("Time values must be strictly increasing.")

last_training_index = np.searchsorted(
    t_num,
    min(training_end_time, t_num[-1]),
    side="right",
) - 1
number_training_data = min(number_training_data, last_training_index + 1)
idx = np.unique(
    np.linspace(0, last_training_index, number_training_data, dtype=int)
)

t_data_np = t_num[idx]
theta_data_np = theta_num[idx]

t_all = torch.tensor(t_num, dtype=torch.float32, device=device)
t_data = torch.tensor(t_data_np, dtype=torch.float32, device=device)
theta_data = torch.tensor(theta_data_np, dtype=torch.float32, device=device)
theta_0 = torch.tensor(theta_num[0], dtype=torch.float32, device=device)
velocity_0 = torch.tensor(velocity_num[0], dtype=torch.float32, device=device)

theta_scale_value = max(float(np.max(np.abs(theta_data_np))), 1.0e-3)
theta_scale = torch.tensor(theta_scale_value, dtype=torch.float32, device=device)
physics_scale = torch.tensor(
    max(alpha_init * theta_scale_value, 1.0e-3),
    dtype=torch.float32,
    device=device,
)


# ============================================================
# Continuous fundamental-frequency initialization
# ============================================================

def harmonic_design_matrix(t, fundamental_omega, harmonic_orders):
    columns = []
    for harmonic in harmonic_orders:
        columns.append(np.cos(harmonic * fundamental_omega * t))
        columns.append(np.sin(harmonic * fundamental_omega * t))
    return np.column_stack(columns)


def estimate_fundamental_frequency():
    """Fit a continuous omega_1; do not round it to an FFT bin.

    A 1+3+5 harmonic least-squares model is used during the scan.  Including
    the nonlinear harmonics prevents the distorted large-angle waveform from
    biasing the fundamental-frequency estimate.
    """

    time_span = t_data_np[-1] - t_data_np[0]
    if time_span <= 0.0:
        raise ValueError("Training data must cover a nonzero time interval.")

    sparse_dt = np.median(np.diff(t_data_np))
    omega_min = max(2.0 * np.pi / (4.0 * time_span), 1.0e-3)
    omega_nyquist = 0.98 * np.pi / sparse_dt
    omega_max = min(omega_nyquist, 2.0 * math.sqrt(max(alpha_init, 1.0e-6)))
    if omega_max <= omega_min:
        omega_max = 4.0 * omega_min

    scan_orders = np.array([1, 3, 5])

    def fitting_error(fundamental_omega):
        design = harmonic_design_matrix(
            t_data_np,
            fundamental_omega,
            scan_orders,
        )
        coefficient, *_ = np.linalg.lstsq(
            design,
            theta_data_np,
            rcond=None,
        )
        return np.mean((design @ coefficient - theta_data_np) ** 2)

    omega_candidates = np.linspace(omega_min, omega_max, 6000)
    errors = np.array([fitting_error(w) for w in omega_candidates])
    best = int(np.argmin(errors))

    left = omega_candidates[max(best - 1, 0)]
    right = omega_candidates[min(best + 1, len(omega_candidates) - 1)]
    result = minimize_scalar(
        fitting_error,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    return float(result.x)


omega1_estimate = estimate_fundamental_frequency()
positive_orders_np = np.arange(1, max_harmonic + 1, 2)

# Least-squares Fourier coefficients from sparse measurements only.
# A*cos(n*w*t) + B*sin(n*w*t) corresponds to c_n=(A-iB)/2.
initial_design = harmonic_design_matrix(
    t_data_np,
    omega1_estimate,
    positive_orders_np,
)
initial_ab, *_ = np.linalg.lstsq(
    initial_design,
    theta_data_np,
    rcond=None,
)
initial_positive_coefficients_np = (
    initial_ab[0::2] - 1j * initial_ab[1::2]
) / 2.0

positive_orders = torch.tensor(
    positive_orders_np,
    dtype=torch.float32,
    device=device,
)
initial_positive_coefficients = torch.tensor(
    np.column_stack(
        (
            initial_positive_coefficients_np.real,
            initial_positive_coefficients_np.imag,
        )
    ),
    dtype=torch.float32,
    device=device,
)

print(f"Training samples: {len(idx)} over 0 to {t_data_np[-1]:.3f} s")
print(f"Continuous initial omega_1: {omega1_estimate:.8f} rad/s")


# ============================================================
# Linear-Tanh Fourier PINN
# ============================================================

class HarmonicFourierTanhPINN(nn.Module):
    """omega_n -> [Re(c_n), Im(c_n)] on the odd harmonic lattice.

    The sparse-data least-squares spectrum is a fixed starting point.  The
    Linear-Tanh-Linear network learns its physics-informed correction.  The
    final layer starts at zero, so epoch zero reproduces that data estimate
    instead of a single artificial FFT-bin spike.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 2),
        )

        nn.init.xavier_uniform_(self.network[0].weight)
        nn.init.zeros_(self.network[0].bias)
        nn.init.zeros_(self.network[2].weight)
        nn.init.zeros_(self.network[2].bias)

        self.register_buffer(
            "initial_coefficients",
            initial_positive_coefficients.clone(),
        )

        # A bounded frequency correction avoids jumping to an alias during
        # training.  exp(0.25*tanh(q)) permits about +/-28% around the scan.
        self.raw_frequency_shift = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        raw_alpha = alpha_init + math.log(-math.expm1(-alpha_init))
        self.raw_alpha = nn.Parameter(
            torch.tensor(raw_alpha, dtype=torch.float32),
            requires_grad=False,
        )

    @property
    def omega1(self):
        return omega1_estimate * torch.exp(
            0.25 * torch.tanh(self.raw_frequency_shift)
        )

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def positive_coefficients(self):
        positive_omega = positive_orders * self.omega1
        scaled_omega = 2.0 * positive_omega / (
            max_harmonic * self.omega1
        ) - 1.0
        correction = self.network(scaled_omega[:, None])
        output = self.initial_coefficients + correction
        return torch.complex(output[:, 0], output[:, 1])


model = HarmonicFourierTanhPINN().to(device)


def centered_spectrum(positive_coefficients):
    """Create [c_-H, ..., c_0, ..., c_H] with Hermitian symmetry."""

    spectrum = torch.zeros(
        2 * max_harmonic + 1,
        dtype=torch.complex64,
        device=device,
    )
    center = max_harmonic
    integer_orders = positive_orders.to(torch.long)
    spectrum[center + integer_orders] = positive_coefficients
    spectrum[center - integer_orders] = torch.conj(positive_coefficients)
    return spectrum


def predict_at_time(t):
    positive_coefficients = model.positive_coefficients()
    harmonic_omega = positive_orders * model.omega1
    phase = torch.exp(1j * t[:, None] * harmonic_omega[None, :])

    theta = 2.0 * torch.real(
        torch.sum(phase * positive_coefficients[None, :], dim=1)
    )
    velocity = 2.0 * torch.real(
        torch.sum(
            phase
            * (1j * harmonic_omega * positive_coefficients)[None, :],
            dim=1,
        )
    )
    return theta, velocity, positive_coefficients


def full_complex_convolution_conv1d(left, right):
    """Full complex convolution implemented using torch.nn.functional.conv1d."""

    def real_convolution(signal, kernel):
        # conv1d is cross-correlation, so reverse the kernel for convolution.
        return F.conv1d(
            signal[None, None, :],
            torch.flip(kernel, dims=(0,))[None, None, :],
            padding=kernel.numel() - 1,
        )[0, 0]

    real_part = (
        real_convolution(left.real, right.real)
        - real_convolution(left.imag, right.imag)
    )
    imaginary_part = (
        real_convolution(left.real, right.imag)
        + real_convolution(left.imag, right.real)
    )
    return torch.complex(real_part, imaginary_part)


def nonlinear_sine_coefficients(theta_coefficients):
    r"""Fourier-series coefficients of truncated sin(theta), using conv1d.

    For the continuous convention

        F[theta^(2k+1)] = (Theta *)^(2k+1)/(2*pi)^(2k),

    insert Theta(omega)=2*pi*sum_n c_n*delta(omega-n*omega_1).
    The powers of 2*pi cancel, leaving the ordinary discrete convolution of
    the Fourier-series coefficients c_n used here.
    """

    sine_coefficients = torch.zeros_like(theta_coefficients)
    convolution_power = theta_coefficients
    half_width = max_harmonic

    for k in range(sine_taylor_order + 1):
        center = convolution_power.numel() // 2
        same_band = convolution_power[
            center - half_width : center + half_width + 1
        ]
        sine_coefficients = sine_coefficients + (
            (-1.0) ** k / math.factorial(2 * k + 1)
        ) * same_band

        if k < sine_taylor_order:
            # Advance theta^(2k+1) -> theta^(2k+3).  Cropping is deliberately
            # postponed until after the complete convolution power is formed.
            convolution_power = full_complex_convolution_conv1d(
                convolution_power,
                theta_coefficients,
            )
            convolution_power = full_complex_convolution_conv1d(
                convolution_power,
                theta_coefficients,
            )

    return sine_coefficients


integer_orders = torch.arange(
    -max_harmonic,
    max_harmonic + 1,
    dtype=torch.float32,
    device=device,
)


def update_alpha(theta_coefficients, sine_coefficients):
    """Relax alpha toward the least-squares Fourier-residual value."""

    with torch.no_grad():
        omega_squared_theta = (
            integer_orders * model.omega1
        ) ** 2 * theta_coefficients
        denominator = torch.sum(torch.abs(sine_coefficients) ** 2).clamp_min(
            1.0e-12
        )
        alpha_target = torch.real(
            torch.sum(
                torch.conj(sine_coefficients) * omega_squared_theta
            )
        ) / denominator
        alpha_target = alpha_target.clamp(1.0e-4, 1.0e3)

        alpha_new = (
            (1.0 - alpha_relaxation) * model.alpha
            + alpha_relaxation * alpha_target
        )
        raw_alpha_new = alpha_new + torch.log(-torch.expm1(-alpha_new))
        model.raw_alpha.copy_(raw_alpha_new)


def loss_components(update_alpha_first):
    theta_prediction_data, _, positive_coefficients = predict_at_time(t_data)
    theta_prediction_0, velocity_prediction_0, _ = predict_at_time(
        t_all[:1]
    )

    theta_coefficients = centered_spectrum(positive_coefficients)
    sine_coefficients = nonlinear_sine_coefficients(theta_coefficients)

    if update_alpha_first:
        update_alpha(theta_coefficients, sine_coefficients)

    physics_residual = (
        -(integer_orders * model.omega1) ** 2 * theta_coefficients
        + model.alpha * sine_coefficients
    )

    data_loss = torch.mean(
        ((theta_prediction_data - theta_data) / theta_scale) ** 2
    )
    physics_loss = torch.mean(
        torch.abs(physics_residual / physics_scale) ** 2
    )
    init_loss = (
        ((theta_prediction_0[0] - theta_0) / theta_scale) ** 2
        + (
            (velocity_prediction_0[0] - velocity_0)
            / (theta_scale * model.omega1)
        ) ** 2
    )
    spectrum_decay = torch.mean(
        (positive_orders / max_harmonic) ** 4
        * torch.abs(positive_coefficients / theta_scale) ** 2
    )

    total_loss = (
        lambda_data * data_loss
        + lambda_physics * physics_loss
        + lambda_init * init_loss
        + lambda_spectrum_decay * spectrum_decay
    )
    return (
        total_loss,
        data_loss,
        physics_loss,
        init_loss,
        positive_coefficients,
    )


# ============================================================
# Adam training
# ============================================================

optimizer = torch.optim.Adam(
    [
        {
            "params": model.network.parameters(),
            "lr": learning_rate_network,
        },
        {
            "params": [model.raw_frequency_shift],
            "lr": learning_rate_frequency,
        },
    ]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(adam_epochs, 1),
    eta_min=1.0e-5,
)

total_history = []
data_history = []
physics_history = []
init_history = []
alpha_history = []
omega1_history = []

theta_snapshots = []
spectrum_snapshots = []
frequency_snapshots = []
snapshot_epochs = []

start_time = time.time()
for epoch in range(adam_epochs + 1):
    optimizer.zero_grad(set_to_none=True)
    (
        loss,
        data_loss,
        physics_loss,
        init_loss,
        positive_coefficients,
    ) = loss_components(update_alpha_first=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.network.parameters(), max_norm=10.0)
    optimizer.step()
    scheduler.step()

    total_history.append(loss.item())
    data_history.append(data_loss.item())
    physics_history.append(physics_loss.item())
    init_history.append(init_loss.item())
    alpha_history.append(model.alpha.item())
    omega1_history.append(model.omega1.item())

    if epoch % print_every == 0:
        elapsed = time.time() - start_time
        print(
            f"\rAdam {epoch:6d}/{adam_epochs} | "
            f"Loss {loss.item():.3e} | "
            f"alpha {model.alpha.item():.6f} | "
            f"omega_1 {model.omega1.item():.6f} | "
            f"Time {time_format(elapsed)}",
            end="",
            flush=True,
        )

    if epoch % animation_every == 0:
        with torch.no_grad():
            theta_snapshot, _, coefficients_snapshot = predict_at_time(t_all)
        theta_snapshots.append(theta_snapshot.cpu().numpy().copy())
        spectrum_snapshots.append(
            torch.abs(coefficients_snapshot).cpu().numpy().copy()
        )
        frequency_snapshots.append(
            positive_orders_np * model.omega1.item()
        )
        snapshot_epochs.append(epoch)

print()


# ============================================================
# L-BFGS finishing stage
# ============================================================

# Adam's alternating projection gives a stable alpha.  L-BFGS then jointly
# polishes the network, omega_1 and alpha, which is especially useful for phase
# accuracy over a long extrapolation interval.
model.raw_alpha.requires_grad_(True)
lbfgs = torch.optim.LBFGS(
    list(model.network.parameters())
    + [model.raw_frequency_shift, model.raw_alpha],
    lr=0.5,
    max_iter=lbfgs_max_iterations,
    max_eval=int(1.25 * lbfgs_max_iterations),
    tolerance_grad=1.0e-10,
    tolerance_change=1.0e-12,
    history_size=100,
    line_search_fn="strong_wolfe",
)


def lbfgs_closure():
    lbfgs.zero_grad(set_to_none=True)
    loss, *_ = loss_components(update_alpha_first=False)
    loss.backward()
    return loss


if lbfgs_max_iterations > 0:
    print("Running L-BFGS finishing stage...")
    lbfgs.step(lbfgs_closure)


# ============================================================
# Final evaluation
# ============================================================

model.eval()
with torch.no_grad():
    theta_PINN_t, velocity_PINN_t, positive_coefficients_t = predict_at_time(
        t_all
    )
    final_losses = loss_components(update_alpha_first=False)

theta_PINN = theta_PINN_t.cpu().numpy()
velocity_PINN = velocity_PINN_t.cpu().numpy()
positive_coefficients_PINN = positive_coefficients_t.cpu().numpy()

alpha_learned = model.alpha.item()
omega1_learned = model.omega1.item()
period_learned = 2.0 * np.pi / omega1_learned
rmse_all = np.sqrt(np.mean((theta_PINN - theta_num) ** 2))
rmse_training = np.sqrt(np.mean((theta_PINN[idx] - theta_num[idx]) ** 2))

total_history.append(final_losses[0].item())
data_history.append(final_losses[1].item())
physics_history.append(final_losses[2].item())
init_history.append(final_losses[3].item())
alpha_history.append(alpha_learned)
omega1_history.append(omega1_learned)

theta_snapshots.append(theta_PINN.copy())
spectrum_snapshots.append(np.abs(positive_coefficients_PINN).copy())
frequency_snapshots.append(positive_orders_np * omega1_learned)
snapshot_epochs.append(adam_epochs + lbfgs_max_iterations)

print(f"Learned alpha:       {alpha_learned:.8f} s^-2")
print(f"Learned omega_1:     {omega1_learned:.8f} rad/s")
print(f"Learned period:      {period_learned:.8f} s")
print(f"Training-data RMSE:  {rmse_training:.6e} rad")
print(f"Full-interval RMSE:  {rmse_all:.6e} rad")
print(f"Runtime:             {time_format(time.time() - start_time)}")
print("Learned positive-frequency harmonics:")
for harmonic, coefficient in zip(
    positive_orders_np,
    positive_coefficients_PINN,
):
    print(
        f"  n={harmonic:2d} | omega={harmonic * omega1_learned:10.6f} "
        f"rad/s | |c_n|={abs(coefficient):.6e}"
    )


# Numerical FFT is used for comparison only.  The PINN itself is not tied to
# these bins, so it can put peaks exactly at n*omega_1.
dt = np.median(np.diff(t_num))
omega_fft = 2.0 * np.pi * np.fft.rfftfreq(len(t_num), d=dt)
theta_fft = np.fft.rfft(theta_num) / len(t_num)


# ============================================================
# Save figures after training
# ============================================================

print("Saving results...")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_num, theta_num, label="Numerical solution", color="orange")
ax.plot(t_data_np, theta_data_np, "o", label="Training data", color="blue")
ax.plot(t_num, theta_PINN, label="Harmonic Fourier PINN", color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
fig.tight_layout()
fig.savefig("fpinn_result_ver5_fixed.png", dpi=300)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 4))
plot_limit = min(
    omega_fft[-1],
    (max_harmonic + 1) * omega1_learned,
)
fft_mask = omega_fft <= plot_limit
ax.plot(
    omega_fft[fft_mask],
    np.abs(theta_fft[fft_mask]) + 1.0e-12,
    label="FFT of numerical solution",
    color="orange",
)
markerline, stemlines, baseline = ax.stem(
    positive_orders_np * omega1_learned,
    np.abs(positive_coefficients_PINN),
    linefmt="r-",
    markerfmt="ro",
    basefmt=" ",
    label="Harmonic Fourier PINN",
)
plt.setp(stemlines, linewidth=1.5)
ax.axvline(
    omega1_learned,
    color="red",
    ls="--",
    alpha=0.6,
    label=r"learned $\omega_1$",
)
ax.axvline(
    math.sqrt(alpha_learned),
    color="black",
    ls=":",
    label=r"small-angle $\sqrt{\alpha}$",
)
ax.set_xlim(0.0, plot_limit)
ax.set_xlabel(r"Angular frequency $\omega$ (rad/s)")
ax.set_ylabel(r"Positive-frequency coefficient $|c_n|$")
ax.legend()
fig.tight_layout()
fig.savefig("fpinn_spectrum_ver5_fixed.png", dpi=300)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(total_history, label="Total", color="black")
ax.semilogy(data_history, label="Data", color="blue")
ax.semilogy(physics_history, label="Physics", color="red")
ax.semilogy(init_history, label="Initial condition", color="green")
ax.set_xlabel("Recorded optimization step")
ax.set_ylabel("Nondimensional loss")
ax.legend()
fig.tight_layout()
fig.savefig("fpinn_loss_ver5_fixed.png", dpi=300)
plt.close(fig)


fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(alpha_history, color="purple", label=r"identified $\alpha$")
axes[0].set_ylabel(r"$\alpha$ (s$^{-2}$)")
axes[0].legend()
axes[1].plot(omega1_history, color="teal", label=r"identified $\omega_1$")
axes[1].set_xlabel("Recorded optimization step")
axes[1].set_ylabel(r"$\omega_1$ (rad/s)")
axes[1].legend()
fig.tight_layout()
fig.savefig("fpinn_parameters_ver5_fixed.png", dpi=300)
plt.close(fig)


if save_gifs:
    fig_time, ax_time = plt.subplots(figsize=(8, 4))
    ax_time.plot(t_num, theta_num, label="Numerical solution", color="orange")
    ax_time.plot(t_data_np, theta_data_np, "o", label="Training data", color="blue")
    (time_line,) = ax_time.plot(
        t_num,
        theta_snapshots[0],
        label="Harmonic Fourier PINN",
        color="red",
    )
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Angle (rad)")
    ax_time.legend()

    def update_time(frame):
        time_line.set_ydata(theta_snapshots[frame])
        ax_time.set_title(f"Time domain - step {snapshot_epochs[frame]}")
        return (time_line,)

    time_animation = animation.FuncAnimation(
        fig_time,
        update_time,
        frames=len(theta_snapshots),
        blit=False,
    )
    time_animation.save(
        "fpinn_time_ver5_fixed.gif",
        writer=animation.PillowWriter(fps=15),
    )
    plt.close(fig_time)

    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
    ax_spectrum.plot(
        omega_fft[fft_mask],
        np.abs(theta_fft[fft_mask]) + 1.0e-12,
        label="FFT of numerical solution",
        color="orange",
    )
    (spectrum_line,) = ax_spectrum.plot(
        frequency_snapshots[0],
        spectrum_snapshots[0],
        "ro",
        label="Harmonic Fourier PINN",
    )
    ax_spectrum.set_xlim(0.0, plot_limit)
    ax_spectrum.set_ylim(
        0.0,
        1.1
        * max(
            np.max(np.abs(theta_fft[fft_mask])),
            np.max(np.abs(positive_coefficients_PINN)),
        ),
    )
    ax_spectrum.set_xlabel(r"Angular frequency $\omega$ (rad/s)")
    ax_spectrum.set_ylabel(r"Positive-frequency coefficient $|c_n|$")
    ax_spectrum.legend()

    def update_spectrum(frame):
        spectrum_line.set_data(
            frequency_snapshots[frame],
            spectrum_snapshots[frame],
        )
        ax_spectrum.set_title(
            f"Fourier domain - step {snapshot_epochs[frame]}"
        )
        return (spectrum_line,)

    spectrum_animation = animation.FuncAnimation(
        fig_spectrum,
        update_spectrum,
        frames=len(spectrum_snapshots),
        blit=False,
    )
    spectrum_animation.save(
        "fpinn_spectrum_ver5_fixed.gif",
        writer=animation.PillowWriter(fps=15),
    )
    plt.close(fig_spectrum)

print("All results saved.")
