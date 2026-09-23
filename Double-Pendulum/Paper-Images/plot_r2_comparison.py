"""Plot 1 - mean(R1², R2²) against Epoch for FPINN and TPINN.

Run: python3 plot_r2_comparison.py
Requires matplotlib. Saves one figure in PNG and PDF beside this script.
Epoch uses the completed optimizer updates recorded in each history file.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

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
    # 'figure.figsize': (10/2.54, 6/2.54),  # 10x6 cm in inches (1 figure per line)
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

def load_history(path):
    """Read the two component R² scores and average before subtracting from 1."""
    epochs, errors = [], []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            epoch = int(row["optimizer_updates"])
            r1_squared = float(row["r2_theta1"])
            r2_squared = float(row["r2_theta2"])
            error = 1.0 - (r1_squared + r2_squared) / 2.0
            if not math.isfinite(error) or error <= 0:
                raise ValueError(f"{path}: nonpositive or nonfinite 1-R² at epoch {epoch}")
            if epochs and epoch <= epochs[-1]:
                raise ValueError(f"{path}: epochs must increase strictly")
            epochs.append(epoch)
            errors.append(error)
    if not epochs:
        raise ValueError(f"{path}: no history data")
    return epochs, errors


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpinn", type=Path, default=root / "fpinn2_003/fpinn2_r2_history.csv")
    parser.add_argument("--tpinn", type=Path, default=root / "tpinn2_001/tpinn2_r2_history.csv")
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10/2.54, 6/2.54), layout="constrained")
    for label, path, color in [
        ("FPINN", args.fpinn, "red"),
        ("TPINN", args.tpinn, "blue"),
    ]:
        epochs, errors = load_history(path)
        ax.semilogy(epochs, errors, color=color, linewidth=1.6, label=label)
        print(f"{label}: {len(epochs)} points, epochs {epochs[0]}–{epochs[-1]}, final 1-R²={errors[-1]:.6g}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$1-R^2$")
    ax.set_xlim(0, 50e3)
    ax.set_title(r"FPINN vs TPINN: $1-R^2$ comparison")
    ax.legend()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = args.output_dir / f"r2_comparison.{extension}"
        fig.savefig(output, dpi=600)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
