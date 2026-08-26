from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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
        "lines.linestyle": "-",
        "lines.linewidth": 1,
        "lines.markersize": 4,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.5,
    }
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_FILE = Path("pendulum_data.dat")
OUTPUT_DIR = Path("./Outputs")
OUTPUT_PREFIX = "tpinn"
LOG_FILE = OUTPUT_DIR / "TPINN.log"

SEED = 0
EPOCHS = 100000
SNAPSHOT_EVERY = 1000
PRINT_EVERY = 1000
PHYSICS_POINTS = 1000
PREDICTION_POINTS = 1000

DATA_STOP = 350
DATA_STEP = 35

LEARNING_RATE = 1e-3
LAMBDA_DATA = 1e1
LAMBDA_PHYSICS = 1e0
LAMBDA_INITIAL = 1e1
LAMBDA_ENERGY = 1e-3
GRADIENT_CLIP = 1.0

ALPHA_INITIAL = 0.0
GIF_FPS = 30


torch.manual_seed(SEED)
np.random.seed(SEED)


def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_data(path):
    """Load the time, angle, and angular-velocity columns."""

    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Data must contain time, angle, and velocity columns.")
    if len(data) < 2:
        raise ValueError("At least two time samples are required.")

    return data[:, 0], data[:, 1], data[:, 2]


def coefficient_of_determination(reference, prediction):
    """Return R^2 for a prediction against a reference trajectory."""
    total_sum_of_squares = np.sum((reference - np.mean(reference)) ** 2)
    residual_sum_of_squares = np.sum((reference - prediction) ** 2)
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


def save_log(
    device,
    thread_count,
    data_total,
    epoch,
    loss,
    learned_alpha,
    r2,
    runtime,
):
    """Save the training configuration and final results."""
    log_lines = [
        "Name: TPINN",
        f"Using device: {device}",
        f"Thread: {thread_count}",
        f"Data_total: {data_total}",
        f"data_stop: {DATA_STOP}",
        f"data_step: {DATA_STEP}",
        f"Learning rate: {LEARNING_RATE}",
        f"Lambda data: {LAMBDA_DATA}",
        f"Lambda physics: {LAMBDA_PHYSICS}",
        f"Lambda initial: {LAMBDA_INITIAL}",
        f"Lambda energy: {LAMBDA_ENERGY}",
        f"Epoch: {epoch}",
        f"Loss: {loss:.6e}",
        f"Learned alpha: {learned_alpha:.6f}",
        f"R2: {r2:.6f}",
        f"Runtime: {runtime}",
    ]
    LOG_FILE.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


class TimePINN(nn.Module):
    """Map normalized time directly to the pendulum angle theta(t)."""

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
            nn.Linear(64, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INITIAL, dtype=torch.float32))

    def forward(self, t):
        t_scaled = 2.0 * (t - self.time_min) / (self.time_max - self.time_min) - 1.0
        return self.network(t_scaled)


def time_derivatives(theta, t):
    """Calculate angular velocity and acceleration with autograd."""

    velocity = torch.autograd.grad(
        theta,
        t,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
    )[0]
    acceleration = torch.autograd.grad(
        velocity,
        t,
        grad_outputs=torch.ones_like(velocity),
        create_graph=True,
    )[0]
    return velocity, acceleration


def save_time_animation(
    t_reference,
    theta_reference,
    data_indices,
    t_prediction,
    snapshots,
    snapshot_epochs,
):
    """Save the time-domain prediction evolution."""

    fig, ax = plt.subplots()
    ax.plot(t_reference, theta_reference, color="orange", label="Numerical Solution")
    ax.plot(
        t_reference[data_indices],
        theta_reference[data_indices],
        "o",
        color="blue",
        label="Training Data",
    )
    prediction_line, = ax.plot(
        t_prediction, snapshots[0], color="red", label="Time PINN Prediction"
    )
    ax.set(xlabel="Time (s)", ylabel="Angle (rad)")
    ax.legend()
    title = ax.set_title("")

    def update(frame):
        prediction_line.set_ydata(snapshots[frame])
        title.set_text(f"Pendulum Time PINN - Epoch {snapshot_epochs[frame]}")
        return prediction_line, title

    movie = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=True
    )
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)


def save_figures(
    t_reference,
    theta_reference,
    data_indices,
    t_prediction,
    theta_prediction,
    history,
):
    """Save the final trajectory, loss history, and alpha history."""

    fig, ax = plt.subplots()
    ax.plot(t_reference, theta_reference, color="orange", label="Numerical Solution")
    ax.plot(
        t_reference[data_indices],
        theta_reference[data_indices],
        "o",
        color="blue",
        label="Training Data",
    )
    ax.plot(
        t_prediction,
        theta_prediction,
        color="red",
        label="Time PINN Prediction",
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Angle (rad)",
        title="Pendulum Time PINN",
    )
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=600)
    plt.close(fig)

    epoch_axis = np.arange(len(history["total"]))
    fig, ax = plt.subplots()
    ax.semilogy(epoch_axis, history["total"], color="black", label="Total Loss")
    ax.semilogy(epoch_axis, history["data"], color="blue", label="Data Loss")
    ax.semilogy(epoch_axis, history["physics"], color="red", label="Physics Loss")
    ax.semilogy(
        epoch_axis,
        history["initial"],
        color="green",
        label="Initial Condition Loss",
    )
    ax.semilogy(
        epoch_axis, history["energy"], color="purple", label="Energy Loss"
    )
    ax.set(xlabel="Epochs", ylabel="Loss", title="Loss Convergence")
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epoch_axis, history["alpha"], color="darkcyan")
    ax.set(
        xlabel="Epochs",
        ylabel=r"$\alpha$",
        title=r"Learned $\alpha$ During Training",
    )
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_alpha.png", dpi=600)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU: {torch.get_num_threads()} threads")

    t_reference, theta_reference, velocity_reference = load_data(DATA_FILE)
    time_min, time_max = t_reference.min(), t_reference.max()
    if time_max <= time_min:
        raise ValueError("The time range must be greater than zero.")

    # Preserve the sparse measurement selection from time_pinn_pendulum.py.
    data_indices = np.arange(0, min(DATA_STOP, len(t_reference)), DATA_STEP)
    t_data = torch.tensor(
        t_reference[data_indices], dtype=torch.float32, device=device
    )[:, None]
    theta_data = torch.tensor(
        theta_reference[data_indices], dtype=torch.float32, device=device
    )[:, None]
    velocity_data = torch.tensor(
        velocity_reference[data_indices], dtype=torch.float32, device=device
    )[:, None]

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

    model = TimePINN(time_min, time_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        name: []
        for name in ("total", "data", "physics", "initial", "energy", "alpha")
    }
    snapshot_epochs = []
    time_snapshots = []

    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        t_physics.grad = None

        theta_at_data = model(t_data)
        data_loss = torch.mean((theta_at_data - theta_data) ** 2)

        theta_physics = model(t_physics)
        velocity_physics, acceleration_physics = time_derivatives(
            theta_physics, t_physics
        )
        physics_residual = (
            acceleration_physics + model.alpha * torch.sin(theta_physics)
        )
        physics_loss = torch.mean(physics_residual**2)

        initial_loss = torch.mean(
            (theta_physics[0] - theta_data[0]) ** 2
            + (velocity_physics[0] - velocity_data[0]) ** 2
        )

        energy_physics = (
            0.5 * velocity_physics**2
            + model.alpha * (1.0 - torch.cos(theta_physics))
        )
        initial_energy = (
            0.5 * velocity_data[0] ** 2
            + model.alpha * (1.0 - torch.cos(theta_data[0]))
        )
        energy_loss = torch.mean((energy_physics - initial_energy) ** 2)

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
                theta_now = model(t_prediction_tensor).cpu().numpy().ravel()
            snapshot_epochs.append(epoch)
            time_snapshots.append(theta_now.copy())
            model.train()

    model.eval()
    with torch.no_grad():
        theta_final = model(t_prediction_tensor).cpu().numpy().ravel()
    theta_reference_at_prediction = np.interp(
        t_prediction, t_reference, theta_reference
    )
    coefficient_determination = coefficient_of_determination(
        theta_reference_at_prediction, theta_final
    )

    print(f"\nLearned alpha: {model.alpha.item():.6f}")
    print(f"Coefficient of determination (R^2): {coefficient_determination:.6f}")
    runtime = format_time(time.time() - start_time)
    print(f"Runtime: {runtime}")
    save_log(
        device=device,
        thread_count=torch.get_num_threads(),
        data_total=len(t_reference),
        epoch=epoch,
        loss=total_loss.item(),
        learned_alpha=model.alpha.item(),
        r2=coefficient_determination,
        runtime=runtime,
    )
    print("Saving figures and animation...")

    save_time_animation(
        t_reference,
        theta_reference,
        data_indices,
        t_prediction,
        time_snapshots,
        snapshot_epochs,
    )
    save_figures(
        t_reference,
        theta_reference,
        data_indices,
        t_prediction,
        theta_final,
        history,
    )
    print("All plots and animations saved successfully.")

    


if __name__ == "__main__":
    main()
