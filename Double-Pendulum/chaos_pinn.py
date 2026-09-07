"""Standalone TPINN/FPINN chaotic double-pendulum experiment.

    python3 chaos_pinn.py                     # Run both, print results only
    python3 chaos_pinn.py --model fpinn        # Run Fourier model only
    python3 chaos_pinn.py --save comparison.png  # Optionally save ONE plot

Defaults: 51 angle observations over 0–10 s, physics-only continuation to 20 s,
0.1 s windows, float64, known zero initial velocities. Future labels never enter
training/stopping. This trains new windows beyond observations; it is not frozen
network extrapolation. No helper modules, checkpoints, logs or output folders.
Requires NumPy, PyTorch; Matplotlib is needed only with --save.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import numpy as np
import torch
from torch import nn

# Change these only if your data uses different physical parameters.
PARAMETERS = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=10.0)


@dataclass
class Config:
    train_end: float = 10.0
    predict_end: float = 20.0
    window: float = 0.1
    data_step: int = 20
    points: int = 65
    max_evaluations: int = 1000
    tolerance: float = 1e-10
    width: int = 16
    depth: int = 2
    modes: int = 4
    period_factor: float = 4.0
    seed: int = 0
    threads: int = 1
    data_weight: float = 100.0
    physics_weight: float = 1.0
    initial_velocity: tuple = (0.0, 0.0)
    angle_threshold: float = 0.1
    device: str = "cpu"


def load_record(path, config):
    record = np.loadtxt(path, comments="#", ndmin=2)
    if record.shape[1] not in (3, 5) or len(record) < 5:
        raise ValueError("Expected time, theta1, theta2 [optional omega1, omega2].")
    if not np.all(np.isfinite(record)) or not np.all(np.diff(record[:, 0]) > 0):
        raise ValueError("Data must be finite and times strictly increasing.")
    if not record[0, 0] < config.train_end < config.predict_end <= record[-1, 0] + 1e-10:
        raise ValueError("Require first time < train-end < predict-end <= last reference time.")
    record = record[record[:, 0] <= config.predict_end + 1e-10]
    observed = np.flatnonzero(record[:, 0] <= config.train_end + 1e-10)
    # Include the last available observed row explicitly, with no row past cutoff.
    indices = np.unique(np.r_[observed[::config.data_step], observed[-1]])
    if len(indices) < 3 or np.sum(record[:, 0] > config.train_end + 1e-10) < 2:
        raise ValueError("Need >=3 observations and >=2 future reference samples.")
    velocity = config.initial_velocity
    if velocity is None:
        if record.shape[1] != 5:
            raise ValueError("--initial-velocity from-data requires velocity columns.")
        velocity = record[0, 3:5]
    initial = np.r_[record[0, 1:3], velocity]
    return record, indices, initial


def rhs(state, parameters):
    m1, m2, l1, l2, g = (parameters[k] for k in ("m1", "m2", "l1", "l2", "g"))
    a, b, u, v = state.unbind(dim=1)
    s, c = torch.sin(a - b), torch.cos(a - b)
    d = m1 + m2 * s.square()
    du = (m2*g*torch.sin(b)*c - m2*s*(l1*u.square()*c + l2*v.square())
          - (m1+m2)*g*torch.sin(a)) / (l1*d)
    dv = ((m1+m2)*(l1*u.square()*s-g*torch.sin(b)+g*torch.sin(a)*c)
          + m2*l2*v.square()*s*c) / (l2*d)
    return torch.stack((u, v, du, dv), dim=1)


def energy(state, parameters):
    m1, m2, l1, l2, g = (parameters[k] for k in ("m1", "m2", "l1", "l2", "g"))
    a, b, u, v = state.unbind(dim=1)
    return (0.5*(m1+m2)*l1*l1*u.square() + 0.5*m2*l2*l2*v.square()
            + m2*l1*l2*u*v*torch.cos(a-b)
            - (m1+m2)*g*l1*torch.cos(a) - m2*g*l2*torch.cos(b))


class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega=1.0, first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(inputs, outputs)
        bound = 1.0/inputs if first else np.sqrt(6.0/inputs)/omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega*self.linear(x))


class TimeWindow(nn.Module):
    """Time -> four-state sine network, with exact starting state."""
    def __init__(self, duration, initial, config, device):
        super().__init__()
        self.duration = duration
        layers = [SineLayer(1, config.width, omega=3.0, first=True)]
        layers.extend(SineLayer(config.width, config.width) for _ in range(config.depth-1))
        final = nn.Linear(config.width, 4)
        with torch.no_grad():
            final.weight.uniform_(-np.sqrt(6/config.width), np.sqrt(6/config.width))
            final.bias.zero_()
        self.network = nn.Sequential(*layers, final).to(device=device, dtype=torch.float64)
        self.register_buffer('initial', torch.as_tensor(initial, dtype=torch.float64, device=device))
        self.register_buffer('scale', torch.tensor([2., 2., 8., 8.], dtype=torch.float64, device=device))

    def forward(self, local_time):
        t = local_time[:, None]
        return self.initial + (1-torch.exp(-t))*(self.scale*self.network(2*t/self.duration-1))

    def state_and_derivative(self, local_time):
        t = local_time.detach().clone().requires_grad_(True)
        state = self(t)
        derivative = torch.stack([
            torch.autograd.grad(state[:, i].sum(), t, create_graph=True, retain_graph=True)[0]
            for i in range(4)
        ], dim=1)
        return state, derivative


class FourierWindow(nn.Module):
    """Frequency MLP + learned Fourier coefficients and exact initial state."""
    def __init__(self, duration, initial, config, device):
        super().__init__()
        frequencies = 2*np.pi*np.arange(config.modes+1)/(config.period_factor*duration)
        self.register_buffer('frequencies', torch.as_tensor(frequencies, dtype=torch.float64, device=device))
        self.register_buffer('initial', torch.as_tensor(initial, dtype=torch.float64, device=device))
        layers = [nn.Linear(1, config.width), nn.Tanh()]
        for _ in range(config.depth-1):
            layers.extend((nn.Linear(config.width, config.width), nn.Tanh()))
        self.network = nn.Sequential(*layers, nn.Linear(config.width, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.network.to(device=device, dtype=torch.float64)
        self.coefficients = nn.Parameter(torch.zeros(config.modes+1, 4, dtype=torch.float64, device=device))

    def state_and_derivative(self, local_time):
        omega_input = 2*self.frequencies[:, None]/self.frequencies[-1]-1
        raw = (self.coefficients + 0.01*self.network(omega_input))[1:]
        a, b = raw[:, 0::2], raw[:, 1::2]
        omega = self.frequencies[1:]
        phase = local_time[:, None]*omega[None, :]
        cos, sin = torch.cos(phase), torch.sin(phase)
        # theta0 + omega0*t + F(t)-F(0)-t*F'(0): no periodic endpoint constraint.
        theta = (self.initial[:2] + local_time[:, None]*self.initial[2:]
                 + 2*((cos-1) @ a - (sin-phase) @ b))
        velocity = self.initial[2:] + 2*((-sin*omega) @ a - ((cos-1)*omega) @ b)
        acceleration = 2*((-cos*omega.square()) @ a + (sin*omega.square()) @ b)
        return torch.cat((theta, velocity), dim=1), torch.cat((velocity, acceleration), dim=1)

    def forward(self, local_time):
        return self.state_and_derivative(local_time)[0]


def make_windows(start, config):
    edges = [float(start)]
    # Always terminate a window at the observation cutoff.
    for stop in (config.train_end, config.predict_end):
        while edges[-1] < stop - 1e-10:
            edges.append(min(stop, edges[-1]+config.window))
    return list(zip(edges[:-1], edges[1:]))


def train_windows(kind, observed_t, observed_theta, initial, config):
    """Only observed labels enter this function; future reference is inaccessible."""
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    parameters = PARAMETERS.copy()
    if any(value <= 0 for value in parameters.values()):
        raise ValueError("Masses, lengths and gravity must be positive.")
    scale = np.sqrt(parameters["g"]/min(parameters["l1"], parameters["l2"]))
    residual_scale = torch.tensor([scale, scale, scale**2, scale**2], dtype=torch.float64, device=device)
    windows, diagnostics = [], []
    state0 = np.asarray(initial).copy()
    start = time.perf_counter()
    for number, (left, right) in enumerate(make_windows(observed_t[0], config)):
        duration = right-left
        model = (TimeWindow if kind == "tpinn2" else FourierWindow)(
            duration, state0, config, device)
        grid = torch.linspace(0, duration, config.points, dtype=torch.float64, device=device)
        # The boundary observation is already represented by the prior window.
        # Never re-inject a reference state at a future window boundary.
        mask = ((observed_t >= left-1e-10) if number == 0 else (observed_t > left+1e-10)) & (observed_t <= right+1e-10)
        t_data = torch.as_tensor(observed_t[mask]-left, dtype=torch.float64, device=device)
        theta_data = torch.as_tensor(observed_theta[mask], dtype=torch.float64, device=device)

        def objective(points):
            state, derivative = model.state_and_derivative(points)
            physical = ((derivative-rhs(state, parameters))/residual_scale).square().mean()
            data = ((model(t_data)[:, :2]-theta_data).square().mean()
                    if len(t_data) else physical.new_zeros(()))
            return config.physics_weight*physical + config.data_weight*data, physical, data

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=10,
                                     history_size=50, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-12, tolerance_change=1e-15)
        evaluations = 0
        stagnant = 0
        reason = "evaluation_budget"
        class BudgetReached(Exception):
            pass

        while evaluations < config.max_evaluations:
            accepted = [p.detach().clone() for p in model.parameters()]
            def closure():
                nonlocal evaluations
                if evaluations >= config.max_evaluations:
                    raise BudgetReached
                optimizer.zero_grad(set_to_none=True)
                loss = objective(grid)[0]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in window {number}.")
                loss.backward()
                evaluations += 1
                return loss
            try:
                optimizer.step(closure)
            except BudgetReached:
                with torch.no_grad():
                    for parameter, old in zip(model.parameters(), accepted):
                        parameter.copy_(old)
                break
            loss, _, _ = objective(grid)
            unchanged = all(torch.equal(p, old) for p, old in zip(model.parameters(), accepted))
            stagnant = stagnant+1 if unchanged else 0
            if float(loss.detach()) <= config.tolerance:
                # Independent half-grid points detect collocation overfitting.
                check = objective((grid[:-1]+grid[1:])/2)[0]
                if float(check.detach()) <= config.tolerance:
                    reason = "training_tolerance"
                    break
            if stagnant >= 3:
                reason = "optimizer_stalled"
                break

        loss, physical, data = [float(x.detach()) for x in objective(grid)]
        check = float(objective((grid[:-1]+grid[1:])/2)[1].detach())
        with torch.no_grad():
            state0 = model(grid[-1:])[0].cpu().numpy().copy()
        if not np.all(np.isfinite(state0)):
            raise FloatingPointError(f"Nonfinite endpoint in window {number}.")
        row = dict(window=number, start=left, end=right, observed_samples=int(mask.sum()),
                   objective_evaluations=evaluations, loss=loss, data_loss=data,
                   physics_loss=physical, independent_physics_loss=check, stop_reason=reason)
        diagnostics.append(row)
        windows.append((left, right, model))
        print(f"{kind} [{left:5.2f}, {right:5.2f}] observations={mask.sum():2d} "
              f"evals={evaluations:4d} data={data:.2e} physics={check:.2e} ({reason})", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return windows, diagnostics, parameters, time.perf_counter()-start


def predict(windows, times):
    result = np.empty((len(times), 4))
    covered = np.zeros(len(times), dtype=bool)
    for i, (left, right, model) in enumerate(windows):
        mask = (times >= left-1e-10) & (times <= right+1e-10) & ~covered
        if not np.any(mask):
            continue
        device = next(model.parameters()).device
        local = torch.as_tensor(times[mask]-left, dtype=torch.float64, device=device)
        with torch.no_grad():
            result[mask] = model(local).cpu().numpy()
        covered[mask] = True
    if not covered.all():
        raise ValueError("Prediction times must lie inside the solved windows.")
    return result


def score(reference, prediction):
    error = prediction-reference
    wrapped = np.arctan2(np.sin(error), np.cos(error))
    total = np.sum((reference-reference.mean(axis=0))**2, axis=0)
    r2 = np.divide(np.sum(error**2, axis=0), total, out=np.full(2, np.nan), where=total>0)
    return dict(r2=(1-r2).tolist(), rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                angular_rmse=np.sqrt(np.mean(wrapped**2, axis=0)).tolist())


def plot_results(record, indices, results, config, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = record[:, 0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for i, ax in enumerate(axes[:2]):
        ax.plot(t, record[:, i+1], color='0.4', label='Numerical reference')
        ax.scatter(t[indices], record[indices, i+1], s=10, color='black', label='Observed data', zorder=5)
        ax.set_ylabel(f'Angle {i+1} (rad)')
    for kind, states in results.items():
        for i, ax in enumerate(axes[:2]):
            ax.plot(t, states[:, i], '--', label=kind.upper())
        error = states[:, :2]-record[:, 1:3]
        angular = np.arctan2(np.sin(error), np.cos(error))
        axes[2].semilogy(t, np.maximum(np.sqrt(np.mean(angular**2, axis=1)), 1e-12), label=kind.upper())
    axes[0].set_title('Chaotic double pendulum: observed fit and physics-only continuation')
    axes[2].axhline(config.angle_threshold, color='black', linestyle=':', label=f'{config.angle_threshold:g} rad threshold')
    axes[2].set(xlabel='Time (s)', ylabel='Angular RMS error (rad)')
    for ax in axes:
        ax.axvline(config.train_end, color='black', linestyle='--', linewidth=1)
        ax.axvspan(config.train_end, config.predict_end, color='orange', alpha=.07)
        ax.set_xlim(t[0], config.predict_end)
        ax.legend(ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'Saved one plot: {path.resolve()}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('both', 'tpinn', 'fpinn'), default='both')
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('double_pendulum_data.dat'))
    parser.add_argument('--save', type=Path, help='Optional single comparison image; no files are created by default.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    defaults = Config()
    for name in ('train_end', 'predict_end', 'window'):
        parser.add_argument('--'+name.replace('_','-'), type=float, default=getattr(defaults,name))
    for name in ('data_step', 'max_evaluations', 'seed'):
        parser.add_argument('--'+name.replace('_','-'), type=int, default=getattr(defaults,name))
    parser.add_argument('--initial-velocity', type=float, nargs=2, default=(0., 0.), metavar=('W1','W2'))
    args = parser.parse_args(argv)
    config = Config(**{name: getattr(args, name) for name in
                       ('train_end','predict_end','window','data_step','max_evaluations','seed','device','initial_velocity')})
    if not np.all(np.isfinite([config.train_end, config.predict_end, config.window, *config.initial_velocity])):
        parser.error('Times and initial velocities must be finite.')
    if config.window <= 0 or config.data_step < 1 or config.max_evaluations < 1 or config.seed < 0:
        parser.error('Window, data step and budget must be positive; seed must be nonnegative.')
    if config.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    if args.save and (not args.save.parent.is_dir() or args.save.suffix.lower() not in ('.png', '.pdf', '.svg')):
        parser.error('--save requires an existing parent directory and a .png/.pdf/.svg filename.')
    record, indices, initial = load_record(args.data, config)
    results = {}
    names = ('tpinn2','fpinn2') if args.model == 'both' else (args.model+'2',)
    print(f'Using {len(indices)} observations through {config.train_end:g}s; physics-only continuation to {config.predict_end:g}s.')
    for kind in names:
        windows, diagnostics, parameters, runtime = train_windows(
            kind, record[indices,0].copy(), record[indices,1:3].copy(), initial.copy(), config)
        states = predict(windows, record[:,0])
        results[kind] = states
        fit = record[:,0] <= config.train_end+1e-10
        fit_score = score(record[fit,1:3], states[fit,:2])
        future_score = score(record[~fit,1:3], states[~fit,:2])
        difference = states[:, :2]-record[:,1:3]
        angular_error = np.sqrt(np.mean(np.arctan2(np.sin(difference), np.cos(difference))**2, axis=1))
        failed = np.flatnonzero((~fit) & (angular_error > config.angle_threshold))
        limit = record[failed[0],0] if len(failed) else record[-1,0]
        converged = sum(row['stop_reason']=='training_tolerance' for row in diagnostics)
        print(f'\n{kind.upper()} | runtime {runtime:.1f}s | converged windows {converged}/{len(windows)}')
        print(f'Fit R2 (theta1, theta2): {fit_score["r2"]}')
        print(f'Future R2: {future_score["r2"]}; angular RMSE: {future_score["angular_rmse"]}')
        print(f'{"First threshold crossing" if len(failed) else "No threshold crossing by"}: {limit:.2f}s; horizon {limit-config.train_end:.2f}s\n')
        del windows
    print('Scores compare with the supplied numerical record, not a certified exact trajectory.')
    if args.save:
        plot_results(record, indices, results, config, args.save)
    return results


if __name__ == '__main__':
    main()
