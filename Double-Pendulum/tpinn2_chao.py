"""Windowed sine-feature PINN for observed fitting and autonomous chaotic forecasting.

Fixed hidden sine features, conditioned by SVD, feed trainable linear output weights.
The first-order ODE residual is differentiated analytically and minimized using
Levenberg-Marquardt with an exact parameter Jacobian. Observed windows estimate
velocities from angle measurements; future windows inherit the entire predicted
state. No numerical ODE solver or future reference labels are used in training.
The reference trajectory is used only for final evaluation and plots.
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
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "double_pendulum_data.dat"
OUTPUT_DIR = SCRIPT_DIR / "Outputs/tpinn2_chao_001"
OUTPUT_PREFIX = "tpinn2_chao"
LOG_FILE = OUTPUT_DIR / "TPINN2_CHAO.log"
RUN_NAME = "Double Pendulum Windowed Sine-feature PINN"
SEED = 0
DATA_STOP = 500
DATA_STEP = 10
PREDICTION_END = 20.0
OBSERVATION_WINDOW_SAMPLES = 1  # One sparse observation interval (~0.1 s) per fit.
FORECAST_WINDOW = 0.025
PHYSICS_POINTS = 33
VALIDATION_POINTS = 65
FEATURE_COUNT = 16
BASIS_RCOND = 1e-11
LAMBDA_DATA = 100.0
LAMBDA_PHYSICS = 1.0
MAX_EVALUATIONS = 200
OPTIMIZER_TOLERANCE = 1e-13
PHYSICS_TOLERANCE = 1e-10
MAX_REFINEMENTS = 6
DTYPE = torch.float64
CPU_THREADS = 1
SNAPSHOT_EVERY = 25  # Completed windows, not optimizer iterations.
GIF_FPS = 10
m1 = m2 = l1 = l2 = 1.0
g = 10.0


# -----------------------------------------------------------------------------
# Utilities and double-pendulum physics
# -----------------------------------------------------------------------------
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def coefficient_of_determination(reference, prediction):
    total = np.sum((reference - np.mean(reference, axis=0)) ** 2, axis=0)
    residual = np.sum((reference - prediction) ** 2, axis=0)
    return 1.0 - residual / total

def state_rhs(state):
    theta1 = state[:, 0:1]
    theta2 = state[:, 1:2]
    omega1 = state[:, 2:3]
    omega2 = state[:, 3:4]

    delta = theta1 - theta2
    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)

    numerator1 = (
        m2 * g * torch.sin(theta2) * cos_delta
        - m2
        * sin_delta
        * (l1 * omega1**2 * cos_delta + l2 * omega2**2)
        - (m1 + m2) * g * torch.sin(theta1)
    )
    denominator1 = l1 * (m1 + m2 * sin_delta**2)

    numerator2 = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
    )
    denominator2 = l2 * (m1 + m2 * sin_delta**2)

    acceleration1 = numerator1 / denominator1
    acceleration2 = numerator2 / denominator2
    return torch.cat((omega1, omega2, acceleration1, acceleration2), dim=1)

def make_basis():
    """Condition fixed sine features using SVD; only output weights are trained."""
    rng = np.random.default_rng(SEED)
    frequency = torch.tensor(np.linspace(0.2, 2.8, FEATURE_COUNT), dtype=DTYPE)
    phase = torch.tensor(rng.uniform(-3, 3, FEATURE_COUNT), dtype=DTYPE)
    z = torch.linspace(-1, 1, VALIDATION_POINTS, dtype=DTYPE)[:, None]
    features = torch.cat((torch.ones_like(z), torch.sin(z * frequency + phase)), 1)
    _, singular, vh = torch.linalg.svd(features, full_matrices=False)
    keep = singular > singular[0] * BASIS_RCOND
    transform = vh[keep].T / singular[keep]
    return frequency, phase, transform


class DoublePendulumStatePINN(nn.Module):
    """Local time -> four-state PINN with fixed sine features and learned readout.

    The starting angles are imposed exactly. During observation assimilation only,
    starting velocities are estimated from observed angles plus the equations.
    All four starting components are fixed during autonomous forecasting.
    """
    def __init__(self, left, right, initial, basis, observed=False):
        super().__init__()
        self.left, self.right, self.observed = left, right, observed
        self.register_buffer("initial", torch.as_tensor(initial, dtype=DTYPE).clone())
        for name, value in zip(("frequency", "phase", "transform"), basis):
            self.register_buffer(name, value.clone())
        self.weights = nn.Parameter(torch.zeros((basis[2].shape[1], 4), dtype=DTYPE))
        self.initial_velocity = nn.Parameter(self.initial[2:].clone(), requires_grad=observed)

    def packed_parameters(self):
        weights = self.weights.flatten()
        return torch.cat((weights, self.initial_velocity)) if self.observed else weights

    def state_and_derivative(self, times, parameters=None):
        p = self.packed_parameters() if parameters is None else parameters
        weights = p[:self.weights.numel()].reshape_as(self.weights)
        initial = torch.cat((self.initial[:2], p[-2:])) if self.observed else self.initial
        dt = times.reshape(-1, 1) - self.left
        z = 2 * dt / (self.right - self.left) - 1
        argument = z * self.frequency + self.phase
        features = torch.cat((torch.ones_like(z), torch.sin(argument)), 1) @ self.transform
        derivative = torch.cat((
            torch.zeros_like(z),
            torch.cos(argument) * self.frequency * (2 / (self.right - self.left)),
        ), 1) @ self.transform
        state = initial + dt * (features @ weights)
        velocity = (features + dt * derivative) @ weights
        return state, velocity

    def forward(self, times):
        return self.state_and_derivative(times)[0]


def fit_window(left, right, initial, basis, times, angles):
    """This function receives only labels from the observed interval."""
    from scipy.optimize import least_squares
    observed = len(times) > 0
    model = DoublePendulumStatePINN(left, right, initial, basis, observed)
    grid = torch.linspace(left, right, PHYSICS_POINTS, dtype=DTYPE)
    t_data = torch.as_tensor(times, dtype=DTYPE)
    y_data = torch.as_tensor(angles, dtype=DTYPE)
    scale = np.sqrt(g / min(l1, l2))
    residual_scale = torch.tensor([scale, scale, scale**2, scale**2], dtype=DTYPE)

    def residual(parameters):
        state, derivative = model.state_and_derivative(grid, parameters)
        physical = ((derivative - state_rhs(state)) / residual_scale).flatten()
        terms = physical * np.sqrt(LAMBDA_PHYSICS / (4 * len(grid)))
        if observed:
            prediction = model.state_and_derivative(t_data, parameters)[0][:, :2]
            data = (prediction - y_data).flatten() * np.sqrt(LAMBDA_DATA / (2 * len(times)))
            terms = torch.cat((terms, data))
        return terms

    jacobian = torch.func.jacrev(residual)
    result = least_squares(
        lambda p: residual(torch.as_tensor(p, dtype=DTYPE)).detach().numpy(),
        model.packed_parameters().detach().numpy(),
        jac=lambda p: jacobian(torch.as_tensor(p, dtype=DTYPE)).detach().numpy(),
        method="lm", max_nfev=MAX_EVALUATIONS,
        ftol=OPTIMIZER_TOLERANCE, xtol=OPTIMIZER_TOLERANCE, gtol=OPTIMIZER_TOLERANCE,
    )
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError(f"Nonfinite coefficients in [{left}, {right}].")
    with torch.no_grad():
        parameters = torch.as_tensor(result.x, dtype=DTYPE)
        model.weights.copy_(parameters[:model.weights.numel()].reshape_as(model.weights))
        if observed:
            model.initial_velocity.copy_(parameters[-2:])
        check = torch.linspace(left, right, 2 * PHYSICS_POINTS, dtype=DTYPE)
        state, derivative = model.state_and_derivative(check)
        physics = float(((derivative - state_rhs(state)) / residual_scale).square().mean())
        data = float((model(t_data)[:, :2] - y_data).square().mean()) if observed else 0.0
        terminal = model(torch.tensor([right], dtype=DTYPE))[0].numpy().copy()
    return model, terminal, dict(
        left=left, right=right, data=data, physics=physics,
        total=float(np.dot(result.fun, result.fun)), evaluations=result.nfev,
        observed=observed, optimizer_status=result.status,
    )


def train_windows(observed_time, observed_angles, predict_end, progress=None):
    """Fit observed windows, then forecast without access to future labels."""
    torch.set_num_threads(CPU_THREADS)
    if len(observed_time) < 3 or predict_end <= observed_time[-1]:
        raise ValueError("Need at least three observations and a future time interval.")
    if not np.all(np.isfinite(observed_angles)) or not np.all(np.diff(observed_time) > 0):
        raise ValueError("Observed data must be finite and strictly ordered in time.")
    if OBSERVATION_WINDOW_SAMPLES < 1 or FORECAST_WINDOW <= 0:
        raise ValueError("Window sizes must be positive.")
    basis = make_basis()
    observed_velocity = np.gradient(observed_angles, observed_time, axis=0, edge_order=2)
    initial = np.r_[observed_angles[0], observed_velocity[0]]
    windows, diagnostics = [], []

    def accept(model, terminal, row):
        windows.append(model)
        diagnostics.append(row)
        if progress is not None:
            progress(windows, diagnostics)
        return terminal

    # Measured angles may update observed-window boundaries; velocities are fitted.
    # This assimilation ends strictly at the final supplied observation.
    for start in range(0, len(observed_time)-1, OBSERVATION_WINDOW_SAMPLES):
        stop = min(start + OBSERVATION_WINDOW_SAMPLES, len(observed_time)-1)
        initial = np.r_[observed_angles[start], observed_velocity[start]]
        model, terminal, row = fit_window(
            float(observed_time[start]), float(observed_time[stop]), initial, basis,
            observed_time[start:stop+1], observed_angles[start:stop+1],
        )
        initial = accept(model, terminal, row)

    def forecast(left, right, state, depth=0):
        model, terminal, row = fit_window(left, right, state, basis, np.empty(0), np.empty((0, 2)))
        if not np.isfinite(row["physics"]) or row["physics"] > PHYSICS_TOLERANCE:
            if depth >= MAX_REFINEMENTS:
                raise RuntimeError(f"Physics tolerance not reached in [{left:.6f}, {right:.6f}].")
            middle = (left + right) / 2
            state = forecast(left, middle, state, depth + 1)
            return forecast(middle, right, state, depth + 1)
        return accept(model, terminal, row)

    start = float(observed_time[-1])
    count = int(np.ceil((predict_end-start)/FORECAST_WINDOW))
    for i in range(count):
        left = start + i * FORECAST_WINDOW
        right = min(start + (i+1) * FORECAST_WINDOW, predict_end)
        initial = forecast(left, right, initial)
    return windows, diagnostics


def predict_windows(windows, times):
    """Use completed windows only; untrained future times remain NaN in animations."""
    prediction = np.full((len(times), 4), np.nan)
    ends = np.array([model.right for model in windows])
    indices = np.searchsorted(ends, times, side="left")
    with torch.no_grad():
        for i, model in enumerate(windows):
            mask = (indices == i) & (times >= model.left - 1e-12)
            if mask.any():
                prediction[mask] = model(torch.as_tensor(times[mask], dtype=DTYPE)).numpy()
    return prediction


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def configure_prediction_axes(axis, time_reference, theta_reference, data_indices):
    """Use FPINN's shared angle axes, colors, and measurement markers."""
    for component, color in enumerate(("blue", "red")):
        label = rf"$\theta_{component + 1}$"
        axis.plot(
            time_reference, theta_reference[:, component],
            color=color, alpha=0.35, label=f"Numerical {label}",
        )
        axis.plot(
            time_reference[data_indices], theta_reference[data_indices, component],
            "o", color=color, label=f"Data {label}",
        )
    axis.set(xlabel="Time (s)", ylabel="Angle (rad)")


def save_results(time_reference, theta_reference, data_indices, prediction):
    fig, axis = plt.subplots()
    configure_prediction_axes(axis, time_reference, theta_reference, data_indices)
    for component, color in enumerate(("blue", "red")):
        axis.plot(
            time_reference, prediction[:, component], "--", color=color,
            label=rf"TPINN $\theta_{component + 1}$",
        )
    axis.legend(ncol=2)
    axis.set_title("Double Pendulum Windowed PINN")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.png", dpi=600)
    plt.close(fig)


def save_loss(history):
    epochs = np.asarray(history["epoch"])
    fig, axis = plt.subplots()
    axis.semilogy(epochs, history["total"], color="black", label="Total Loss")
    axis.semilogy(epochs, history["data"], color="blue", label="Data Loss")
    axis.semilogy(epochs, history["physics"], color="red", label="Physics Loss")
    axis.set(xlabel="Completed windows", ylabel="Loss", title="Loss Convergence")
    axis.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_loss.png", dpi=600)
    plt.close(fig)


def save_training_animation(
    time_reference,
    theta_reference,
    data_indices,
    snapshot_epochs,
    snapshots,
):
    fig, axis = plt.subplots()
    configure_prediction_axes(axis, time_reference, theta_reference, data_indices)
    lines = []
    for component, color in enumerate(("blue", "red")):
        line, = axis.plot(
            time_reference, snapshots[0][:, component], "--", color=color,
            label=rf"TPINN $\theta_{component + 1}$",
        )
        lines.append(line)
    axis.legend(ncol=2)
    title = axis.set_title("")

    def update(frame):
        for component, line in enumerate(lines):
            line.set_ydata(snapshots[frame][:, component])
        title.set_text(
            f"Double Pendulum Windowed PINN - Window {snapshot_epochs[frame]}"
        )
        return *lines, title

    movie = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=True)
    movie.save(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_training.gif",
        writer=animation.PillowWriter(fps=GIF_FPS),
    )
    plt.close(fig)

def main():
    start = time.perf_counter()
    values = np.loadtxt(DATA_FILE, skiprows=1)
    values = values[values[:, 0] <= PREDICTION_END]
    times, reference = values[:, 0], values[:, 1:3]
    if not 3 <= DATA_STOP < len(times) or DATA_STEP < 1:
        raise ValueError("DATA_STOP must leave an unseen forecast interval; DATA_STEP must be positive.")
    indices = np.unique(np.r_[np.arange(0, DATA_STOP, DATA_STEP), DATA_STOP-1])
    # Only these arrays enter training. Reference values after DATA_STOP are inaccessible there.
    observed_time, observed_angles = times[indices].copy(), reference[indices].copy()
    history = {name: [] for name in ("epoch", "total", "data", "physics")}
    snapshots, snapshot_windows = [], []
    print(RUN_NAME, flush=True)
    print(f"Observations through {observed_time[-1]:g} s; forecast through {PREDICTION_END:g} s", flush=True)

    def progress(windows, rows):
        row = rows[-1]
        history["epoch"].append(len(windows))
        for name in ("total", "data", "physics"):
            history[name].append(row[name])
        if len(windows) % SNAPSHOT_EVERY == 0 or row["right"] >= PREDICTION_END:
            snapshots.append(predict_windows(windows, times)[:, :2])
            snapshot_windows.append(len(windows))
            print(f"Window {len(windows)} | t={row['right']:.3f} | data={row['data']:.2e} | "
                  f"physics={row['physics']:.2e} | {format_time(time.perf_counter()-start)}", flush=True)

    windows, rows = train_windows(observed_time, observed_angles, PREDICTION_END, progress)
    prediction = predict_windows(windows, times)[:, :2]
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("The prediction does not cover the evaluation interval.")
    observed = times <= observed_time[-1]
    future = times > observed_time[-1]
    observed_r2 = coefficient_of_determination(reference[observed], prediction[observed])
    future_r2 = coefficient_of_determination(reference[future], prediction[future])
    elapsed = time.perf_counter() - start
    lines = [
        f"Name: {RUN_NAME}", "Method: fixed sine-feature PINN; Levenberg-Marquardt; sequential windows",
        f"Data file: {DATA_FILE}", f"Observed through: {observed_time[-1]} s",
        f"Prediction end: {PREDICTION_END} s", f"Data step: {DATA_STEP}",
        f"Training samples: {len(indices)}",
        f"Observation intervals per window: {OBSERVATION_WINDOW_SAMPLES}",
        f"Forecast evaluation interval: ({observed_time[-1]}, {PREDICTION_END}] s",
        f"Seed: {SEED}", f"Precision: {DTYPE}", f"Feature count: {FEATURE_COUNT}",
        f"Retained feature rank: {windows[0].weights.shape[0]}",
        f"Forecast window: {FORECAST_WINDOW}", f"Physics points: {PHYSICS_POINTS}",
        f"Independent validation points: {2*PHYSICS_POINTS}",
        f"Physics tolerance: {PHYSICS_TOLERANCE}", f"Maximum evaluations per window: {MAX_EVALUATIONS}",
        f"Completed windows: {len(windows)}", f"Training runtime: {format_time(elapsed)}",
        f"Observed R2 theta1: {observed_r2[0]:.9f}", f"Observed R2 theta2: {observed_r2[1]:.9f}",
        f"R2 theta1 extrapolation: {future_r2[0]:.9f}", f"R2 theta2 extrapolation: {future_r2[1]:.9f}",
        "Future reference labels are used for evaluation/plots only.",
        "Observed windows assimilate measured boundary angles and estimate velocities.",
        "Forecast boundaries inherit all four predicted state components, without measurement resets.",
        "", "window start end observed evaluations data_loss validation_physics_loss optimizer_status",
    ]
    lines += [f"{i+1} {r['left']:.8f} {r['right']:.8f} {int(r['observed'])} {r['evaluations']} "
              f"{r['data']:.6e} {r['physics']:.6e} {r['optimizer_status']}" for i, r in enumerate(rows)]
    print(f"Fit interval R2: {observed_r2}; "
          f"{observed_time[-1]:g}-{PREDICTION_END:g} s forecast R2: {future_r2}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(lines)+"\n")
    save_results(times, reference, indices, prediction)
    save_loss(history)
    save_training_animation(times, reference, indices, snapshot_windows, snapshots)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    return prediction, rows


if __name__ == "__main__":
    main()
