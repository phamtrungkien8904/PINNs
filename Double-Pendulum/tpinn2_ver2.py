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
OUTPUT_DIR = Path("./Outputs/tpinn_double001")
OUTPUT_PREFIX = "tpinn_double"
LOG_FILE = OUTPUT_DIR / "TPINN_double.log"

SEED = 0
EPOCHS = 100000
SNAPSHOT_EVERY = 1000
PRINT_EVERY = 100
PHYSICS_POINTS = 2000
PREDICTION_POINTS = 2000

DATA_STOP = 5500    
DATA_STEP = 110

LEARNING_RATE = 1e-3
LAMBDA_DATA = 1e1
LAMBDA_PHYSICS = 1e0
LAMBDA_INITIAL = 1e1
LAMBDA_ENERGY = 1e-3
GRADIENT_CLIP = 1.0
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
    """Load double-pendulum data.

    Accepted formats:
        t, theta1, theta2
    or
        t, theta1, theta2, omega1, omega2

    If angular velocities are absent, they are estimated from the numerical
    trajectories and are used only for the initial-condition/energy targets.
    """
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "DATA_FILE must contain at least 3 columns: t, theta1, theta2."
        )

    t = data[:, 0]
    theta1 = data[:, 1]
    theta2 = data[:, 2]

    if data.shape[1] >= 5:
        omega1 = data[:, 3]
        omega2 = data[:, 4]
    else:
        edge_order = 2 if len(t) >= 3 else 1
        omega1 = np.gradient(theta1, t, edge_order=edge_order)
        omega2 = np.gradient(theta2, t, edge_order=edge_order)

    return t, theta1, theta2, omega1, omega2


def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference)) ** 2)
    residual = np.sum((reference - prediction) ** 2)
    return 1.0 - residual / total


def derivative(y, x):
    """dy/dx with graph construction enabled for higher derivatives."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def split_angles(theta):
    return theta[:, 0:1], theta[:, 1:2]


# -----------------------------------------------------------------------------
# Neural network
# -----------------------------------------------------------------------------
class DoublePendulumPINN(nn.Module):
    """Map normalized time t -> [theta1(t), theta2(t)]."""

    def __init__(self, time_min, time_max):
        super().__init__()
        self.register_buffer("time_min", torch.tensor(float(time_min)))
        self.register_buffer("time_max", torch.tensor(float(time_max)))

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t):
        t_scaled = 2.0 * (t - self.time_min) / (self.time_max - self.time_min) - 1.0
        return self.network(t_scaled)


# -----------------------------------------------------------------------------
# Double-pendulum mechanics
# -----------------------------------------------------------------------------
def mechanics(theta1, theta2, omega1, omega2):
    """Return kinetic energy T, potential energy V, Lagrangian L and E=T+V.

    Coordinates follow exactly:
        x1 = l1 sin(theta1)
        y1 = -l1 cos(theta1)
        x2 = x1 + l2 sin(theta2)
        y2 = y1 - l2 cos(theta2)
    """
    y1 = -l1 * torch.cos(theta1)
    y2 = y1 - l2 * torch.cos(theta2)

    vx1 = l1 * torch.cos(theta1) * omega1
    vy1 = l1 * torch.sin(theta1) * omega1
    vx2 = vx1 + l2 * torch.cos(theta2) * omega2
    vy2 = vy1 + l2 * torch.sin(theta2) * omega2

    kinetic = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
    potential = m1 * g * y1 + m2 * g * y2
    lagrangian = kinetic - potential
    energy = kinetic + potential
    return kinetic, potential, lagrangian, energy


def explicit_physics_residuals(model, t):
    """Return the explicit double-pendulum ODE residuals f1 and f2.

    With Delta = theta1 - theta2, the Euler-Lagrange equations are

        f1 = (m1 + m2) l1 theta1_ddot
             + m2 l2 theta2_ddot cos(Delta)
             + m2 l2 theta2_dot^2 sin(Delta)
             + (m1 + m2) g sin(theta1) = 0

        f2 = m2 l2 theta2_ddot
             + m2 l1 theta1_ddot cos(Delta)
             - m2 l1 theta1_dot^2 sin(Delta)
             + m2 g sin(theta2) = 0

    These are the explicit differential equations obtained from the
    Euler-Lagrange equations after dividing the first equation by l1 and
    the second equation by l2.
    """
    theta = model(t)
    theta1, theta2 = split_angles(theta)

    omega1 = derivative(theta1, t)
    omega2 = derivative(theta2, t)
    alpha1 = derivative(omega1, t)
    alpha2 = derivative(omega2, t)

    delta = theta1 - theta2

    f1 = (
        (m1 + m2) * l1 * alpha1
        + m2 * l2 * alpha2 * torch.cos(delta)
        + m2 * l2 * omega2**2 * torch.sin(delta)
        + (m1 + m2) * g * torch.sin(theta1)
    )

    f2 = (
        m2 * l2 * alpha2
        + m2 * l1 * alpha1 * torch.cos(delta)
        - m2 * l1 * omega1**2 * torch.sin(delta)
        + m2 * g * torch.sin(theta2)
    )

    # Energy is still computed from T + V for the energy-conservation loss.
    _, _, _, energy = mechanics(theta1, theta2, omega1, omega2)

    return theta, omega1, omega2, energy, f1, f2


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_log(device, data_total, epoch, loss, r2_1, r2_2, runtime):
    log_lines = [
        "Name: Double Pendulum TPINN",
        f"Using device: {device}",
        f"Thread: {torch.get_num_threads()}",
        f"Data_total: {data_total}",
        f"data_stop: {DATA_STOP}",
        f"data_step: {DATA_STEP}",
        f"m1: {m1}",
        f"m2: {m2}",
        f"l1: {l1}",
        f"l2: {l2}",
        f"g: {g}",
        f"Learning rate: {LEARNING_RATE}",
        f"Lambda data: {LAMBDA_DATA}",
        f"Lambda physics: {LAMBDA_PHYSICS}",
        f"Lambda initial: {LAMBDA_INITIAL}",
        f"Lambda energy: {LAMBDA_ENERGY}",
        f"Epoch: {epoch}",
        f"Loss: {loss:.6e}",
        f"R2 theta1: {r2_1:.6f}",
        f"R2 theta2: {r2_2:.6f}",
        f"R2 mean: {0.5 * (r2_1 + r2_2):.6f}",
        f"Runtime: {runtime}",
    ]
    LOG_FILE.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def save_time_animation(
    t_reference,
    theta1_reference,
    theta2_reference,
    data_indices,
    t_prediction,
    snapshots,
    snapshot_epochs,
):
    """Save training evolution for both predicted angles."""
    fig, ax = plt.subplots()

    ax.plot(t_reference, theta1_reference, color="blue", alpha=0.35, label=r"Numerical $\theta_1$")
    ax.plot(t_reference, theta2_reference, color="red", alpha=0.35, label=r"Numerical $\theta_2$")
    ax.plot(t_reference[data_indices], theta1_reference[data_indices], "o", color="blue", label=r"Data $\theta_1$")
    ax.plot(t_reference[data_indices], theta2_reference[data_indices], "o", color="red", label=r"Data $\theta_2$")

    line1, = ax.plot(t_prediction, snapshots[0][:, 0], "--", color="blue", label=r"PINN $\theta_1$")
    line2, = ax.plot(t_prediction, snapshots[0][:, 1], "--", color="red", label=r"PINN $\theta_2$")

    ax.set(xlabel="Time (s)", ylabel="Angle (rad)")
    ax.legend(ncol=2)
    title = ax.set_title("")

    def update(frame):
        line1.set_ydata(snapshots[frame][:, 0])
        line2.set_ydata(snapshots[frame][:, 1])
        title.set_text(f"Double Pendulum Time PINN - Epoch {snapshot_epochs[frame]}")
        return line1, line2, title

    movie = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True)
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_figures(
    t_reference,
    theta1_reference,
    theta2_reference,
    data_indices,
    t_prediction,
    prediction,
    history,
):
    """Save final angle prediction and loss convergence."""
    theta1_prediction = prediction[:, 0]
    theta2_prediction = prediction[:, 1]

    fig, ax = plt.subplots()
    ax.plot(t_reference, theta1_reference, color="blue", alpha=0.35, label=r"Numerical $\theta_1$")
    ax.plot(t_reference, theta2_reference, color="red", alpha=0.35, label=r"Numerical $\theta_2$")
    ax.plot(t_reference[data_indices], theta1_reference[data_indices], "o", color="blue", label=r"Data $\theta_1$")
    ax.plot(t_reference[data_indices], theta2_reference[data_indices], "o", color="red", label=r"Data $\theta_2$")
    ax.plot(t_prediction, theta1_prediction, "--", color="blue", label=r"PINN $\theta_1$")
    ax.plot(t_prediction, theta2_prediction, "--", color="red", label=r"PINN $\theta_2$")
    ax.set(xlabel="Time (s)", ylabel="Angle (rad)", title="Double Pendulum Time PINN")
    ax.legend(ncol=2)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=600)
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

    t_ref, theta1_ref, theta2_ref, omega1_ref, omega2_ref = load_data(DATA_FILE)
    time_min, time_max = t_ref.min(), t_ref.max()
    if time_max <= time_min:
        raise ValueError("The time range must be greater than zero.")

    # Sparse measured angle data, matching the structure of the original tpinn.py.
    data_indices = np.arange(0, min(DATA_STOP, len(t_ref)), DATA_STEP)
    t_data = torch.tensor(t_ref[data_indices], dtype=torch.float32, device=device)[:, None]
    theta_data = torch.tensor(
        np.column_stack((theta1_ref[data_indices], theta2_ref[data_indices])),
        dtype=torch.float32,
        device=device,
    )

    # Initial-condition targets: theta1(0), theta2(0), omega1(0), omega2(0).
    theta0_target = torch.tensor(
        [[theta1_ref[0], theta2_ref[0]]], dtype=torch.float32, device=device
    )
    omega0_target = torch.tensor(
        [[omega1_ref[0], omega2_ref[0]]], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        _, _, _, energy0_target = mechanics(
            theta0_target[:, 0:1],
            theta0_target[:, 1:2],
            omega0_target[:, 0:1],
            omega0_target[:, 1:2],
        )

    t_physics = torch.linspace(
        float(time_min),
        float(time_max),
        PHYSICS_POINTS,
        dtype=torch.float32,
        device=device,
    )[:, None].requires_grad_(True)

    t_prediction = np.linspace(time_min, time_max, PREDICTION_POINTS)
    t_prediction_tensor = torch.tensor(
        t_prediction, dtype=torch.float32, device=device
    )[:, None]

    model = DoublePendulumPINN(time_min, time_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        name: [] for name in ("total", "data", "physics", "initial", "energy")
    }
    snapshot_epochs = []
    snapshots = []

    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)

        # 1) Data loss: both measured angles.
        theta_at_data = model(t_data)
        data_loss = torch.mean((theta_at_data - theta_data) ** 2)

        # Physics trajectory and derivatives.
        theta_physics, omega1_physics, omega2_physics, energy_physics, f1, f2 = (
            explicit_physics_residuals(model, t_physics)
        )

        # 2) Physics loss: explicit double-pendulum differential equations.
        physics_loss = torch.mean(f1**2 + f2**2)

        # 3) Initial-condition loss: two angles + two angular velocities.
        theta0_prediction = theta_physics[0:1]
        omega0_prediction = torch.cat(
            (omega1_physics[0:1], omega2_physics[0:1]), dim=1
        )
        initial_loss = torch.mean((theta0_prediction - theta0_target) ** 2) + torch.mean(
            (omega0_prediction - omega0_target) ** 2
        )

        # 4) Energy-conservation loss: E(t) = E(0).
        energy_loss = torch.mean((energy_physics - energy0_target) ** 2)

        total_loss = (
            LAMBDA_DATA * data_loss
            + LAMBDA_PHYSICS * physics_loss
            + LAMBDA_INITIAL * initial_loss
            + LAMBDA_ENERGY * energy_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()

        history["total"].append(total_loss.item())
        history["data"].append(data_loss.item())
        history["physics"].append(physics_loss.item())
        history["initial"].append(initial_loss.item())
        history["energy"].append(energy_loss.item())

        if epoch % PRINT_EVERY == 0:
            elapsed = format_time(time.time() - start_time)
            print(
                f"\rEpoch {epoch:6d} | Loss {total_loss.item():.6e} "
                f"| Time {elapsed}",
                end="",
                flush=True,
            )

        if epoch % SNAPSHOT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                prediction_now = model(t_prediction_tensor).cpu().numpy()
            snapshot_epochs.append(epoch)
            snapshots.append(prediction_now.copy())
            model.train()

    model.eval()
    with torch.no_grad():
        prediction_final = model(t_prediction_tensor).cpu().numpy()

    theta1_ref_prediction = np.interp(t_prediction, t_ref, theta1_ref)
    theta2_ref_prediction = np.interp(t_prediction, t_ref, theta2_ref)
    r2_1 = coefficient_of_determination(theta1_ref_prediction, prediction_final[:, 0])
    r2_2 = coefficient_of_determination(theta2_ref_prediction, prediction_final[:, 1])

    runtime = format_time(time.time() - start_time)
    print(f"\nR^2 theta1: {r2_1:.6f}")
    print(f"R^2 theta2: {r2_2:.6f}")
    print(f"R^2 mean:   {0.5 * (r2_1 + r2_2):.6f}")
    print(f"Runtime: {runtime}")

    save_log(
        device=device,
        data_total=len(t_ref),
        epoch=epoch,
        loss=total_loss.item(),
        r2_1=r2_1,
        r2_2=r2_2,
        runtime=runtime,
    )

    print("Saving figures and animation...")
    save_time_animation(
        t_ref,
        theta1_ref,
        theta2_ref,
        data_indices,
        t_prediction,
        snapshots,
        snapshot_epochs,
    )
    save_figures(
        t_ref,
        theta1_ref,
        theta2_ref,
        data_indices,
        t_prediction,
        prediction_final,
        history,
    )
    print("All plots and animations saved successfully.")


if __name__ == "__main__":
    main()
