"""Generate the two dimension-scaling plots from the saved npz traces.

  plot 1: communication rounds R(d) for adaptive FA-HMC at W2^2 < 0.1
  plot 2: vanilla FA-HMC W2^2 vs d at matched compute budget (R = R_a)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent.parent
ADAPT_DIR = Path(os.environ.get(
    'DIMENSION_SCALING_ADAPT_DIR', str(HERE / 'results' / 'adaptive_hmc_scaling')
))
VANILLA_DIR = Path(os.environ.get(
    'DIMENSION_SCALING_VANILLA_DIR', str(HERE / 'results' / 'vanilla_baseline')
))
FIG_DIR = Path(os.environ.get('DIMENSION_SCALING_PLOT_DIR', str(HERE / 'plots')))
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [2, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
EPSILON_SQ = 0.1


def load_adaptive():
    rows = []
    for d in DIMS:
        z = np.load(ADAPT_DIR / f'scaling_d{d:03d}.npz')
        rows.append({
            'd': d, 'R': int(z['first_cross']),
            'K': int(z['K']), 'eta': float(z['eta']),
        })
    return rows


def load_vanilla():
    rows = []
    for d in DIMS:
        z = np.load(VANILLA_DIR / f'vanilla_d{d:03d}.npz')
        rows.append({
            'd': d, 'R': int(z['R_target']),
            'K': int(z['K']), 'eta': float(z['eta']),
            'w2_final': float(z['trace_w2'][-1]),
        })
    return rows


def fit_powerlaw(ds, ys):
    a, lc = np.polyfit(np.log(ds), np.log(ys), 1)
    return float(a), float(math.exp(lc))


def fit_log_corrected(ds, ys, epsilon_sq=EPSILON_SQ):
    """Fit y = C * (d / epsilon_sq)^alpha * log(d / epsilon_sq)."""
    scaled = ds / epsilon_sq
    a, lc = np.polyfit(np.log(scaled), np.log(ys / np.log(scaled)), 1)
    return float(a), float(math.exp(lc))


def fit_theory_scale(ds, ys, epsilon_sq=EPSILON_SQ):
    """Fit the multiplicative constant for (d/epsilon_sq)^(1/3) log(d/epsilon_sq)."""
    scaled = ds / epsilon_sq
    shape = scaled ** (1.0 / 3.0) * np.log(scaled)
    log_c = np.mean(np.log(ys) - np.log(shape))
    return float(math.exp(log_c))


def apply_plot_style(ax):
    """Apply a ggplot-style background and grid."""
    ax.set_facecolor('#EAEAF2')
    ax.grid(True, which='major', color='white', linewidth=1.2)
    ax.grid(False, which='minor')
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_rounds_vs_dimension(adapt_rows, out_path):
    ds = np.array([r['d'] for r in adapt_rows], dtype=float)
    Rs = np.array([r['R'] for r in adapt_rows], dtype=float)
    a_power, _ = fit_powerlaw(ds, Rs)
    a_log_all, _ = fit_log_corrected(ds, Rs)
    a_log_large, _ = fit_log_corrected(ds[ds > 2], Rs[ds > 2])
    theory_scale = fit_theory_scale(ds, Rs)
    print(f'rounds power-law alpha over all dimensions: {a_power:.3f}')
    print(f'rounds log-corrected alpha over all dimensions: {a_log_all:.3f}')
    print(f'rounds log-corrected alpha excluding d=2: {a_log_large:.3f}')

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(ds, Rs, 'o', color='#377EB8', markersize=7, label='measured')

    grid = np.geomspace(ds.min(), ds.max(), 100)
    scaled_grid = grid / EPSILON_SQ
    ax.loglog(grid, theory_scale * scaled_grid ** (1.0 / 3.0) * np.log(scaled_grid),
              ':', color='0.35', linewidth=1.8,
              label=r'theory: $(d/\epsilon^2)^{1/3}\log(d/\epsilon^2)$')

    ax.set_xlabel('dimension $d$')
    ax.set_ylabel('Communication rounds $R(d)$ to reach $W_2^2 < 0.1$')
    apply_plot_style(ax)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def plot_vanilla_error_vs_dimension(vanilla_rows, out_path):
    ds = np.array([r['d'] for r in vanilla_rows], dtype=float)
    Ws = np.array([r['w2_final'] for r in vanilla_rows], dtype=float)
    b_all, c_all = fit_powerlaw(ds, Ws)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(ds, Ws, 's', color='#E41A1C', markersize=7, label='vanilla FA-HMC')
    ax.axhline(0.1, color='#377EB8', linestyle=':', linewidth=1.8,
               label='adaptive threshold')

    grid = np.geomspace(ds.min(), ds.max(), 100)
    ax.loglog(grid, c_all * grid ** b_all, '--', color='#E41A1C', alpha=0.7,
              label=rf'empirical: $d^{{{b_all:.3f}}}$')

    ax.set_xlabel('dimension $d$')
    ax.set_ylabel(r'Final $W_2^2$ at matched budget')
    apply_plot_style(ax)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def main():
    adapt = load_adaptive()
    vanilla = load_vanilla()
    plot_rounds_vs_dimension(adapt, FIG_DIR / 'rounds_vs_dimension.png')
    plot_vanilla_error_vs_dimension(vanilla, FIG_DIR / 'vanilla_error_vs_dimension.png')


if __name__ == '__main__':
    main()
