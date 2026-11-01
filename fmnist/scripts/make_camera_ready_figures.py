"""Fashion-MNIST figures and tables on the equal server-exchange axis (paper Fig. 3).

FA-HMC/FA-LD use one server exchange per communication round and AFHMC uses
two, so FA-HMC round r is plotted at x = r and AFHMC round r at x = 2r. With a
3600-round run, the x-axis spans 0..3600 exchanges (AFHMC rounds 0..1800).

For each (metric, algo, K) the stepsize is the one with the best seed-mean value
of that metric at the last plotted point, matching the per-metric tuning
protocol of plot_fmnist_results.py.

Outputs:
  --out          figure with one panel per metric in --metrics
  --table        LaTeX tabular rows: final mean +/- SE per metric (appendix table)
  --trajectory   LaTeX tabular rows: accuracy at intermediate exchange counts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_fmnist_results import COL_ACC, COL_BRIER, COL_NLL, load_cells


ALGO_K = [('fa_hmc', 1), ('fa_hmc', 10), ('fa_hmc', 50), ('fa_hmc', 100),
          ('adaptive', 10), ('adaptive', 50), ('adaptive', 100)]
K_COLORS = {1: '#377EB8', 10: '#4DAF4A', 50: '#FF7F00', 100: '#E41A1C'}
ALGO_STYLE = {'fa_hmc': '-', 'adaptive': '--'}
EXCHANGES_PER_ROUND = {'fa_hmc': 1, 'adaptive': 2}
# name -> (test_poster column, axis label, lower is better, y-limits)
METRICS = {
    'accuracy': (COL_ACC, 'Accuracy', False, (0.76, 0.85)),
    'brier': (COL_BRIER, 'Brier score', True, (0.22, 0.34)),
    'nll': (COL_NLL, 'NLL', True, (4400, 7000)),
}
TABLE_NAMES = {'accuracy': 'Accuracy', 'brier': 'Brier', 'nll': 'NLL'}


def curve_label(algo, K):
    if algo == 'fa_hmc' and K == 1:
        return 'FA-LD (K=1)'
    return f"{'FA-HMC' if algo == 'fa_hmc' else 'AFHMC'} K={K}"


def method_name(algo, K):
    if algo == 'fa_hmc':
        return 'FA-LD ' if K == 1 else 'FA-HMC'
    return 'AFHMC '


def seed_values(runs, col, max_round):
    """(comm_round, values[n_seeds, n_points]) for comm_round <= max_round."""
    comm = None
    vals = []
    for _, cr, trace in sorted(runs, key=lambda r: r[0]):
        keep = cr <= max_round
        if comm is None:
            comm = cr[keep]
        elif not np.array_equal(comm, cr[keep]):
            raise ValueError('seeds have different comm_round axes')
        vals.append(trace[keep, col])
    return comm, np.stack(vals)


def select(cells, algo, K, col, lower, exchanges):
    """Best-eta (eta, comm_round, seed values) for one (algo, K) and metric."""
    max_round = exchanges // EXCHANGES_PER_ROUND[algo]
    best = None
    for (a, k, eta), runs in cells.items():
        if a != algo or k != K:
            continue
        comm, vals = seed_values(runs, col, max_round)
        if comm[-1] != max_round:
            raise ValueError(f'{algo} K={K} eta={eta} stops at round {comm[-1]}')
        final = vals[:, -1].mean()
        if best is None or (final < best[3] if lower else final > best[3]):
            best = (eta, comm, vals, final)
    return best[:3]


def fmt_eta(eta):
    mant, exp = f'{eta:.0e}'.split('e')
    return f'${mant}{{\\times}}10^{{{int(exp)}}}$'


def fmt_val(name, m, se):
    if name == 'nll':
        return f'${m:.2f} \\pm {se:.2f}$'
    return f'${m:.4f} \\pm {se:.4f}$'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--exchanges', type=int, default=3600)
    parser.add_argument('--metrics', default='accuracy')
    parser.add_argument('--out', default='fig3_cr_accuracy_equal_exchange.png')
    parser.add_argument('--table', default=None)
    parser.add_argument('--trajectory', default=None)
    args = parser.parse_args()
    names = [m.strip() for m in args.metrics.split(',') if m.strip()]
    cells = load_cells(Path(args.results).resolve())

    plt.style.use('ggplot')
    figsize = (6.8, 4.8) if len(names) == 1 else (11, 4.6)
    fig, axes = plt.subplots(1, len(names), figsize=figsize, squeeze=False)
    for ax, name in zip(axes[0], names):
        col, ylabel, lower, ylim = METRICS[name]
        for algo, K in ALGO_K:
            eta, comm, vals = select(cells, algo, K, col, lower, args.exchanges)
            x = comm * EXCHANGES_PER_ROUND[algo]
            ax.plot(x, vals.mean(axis=0), color=K_COLORS[K], linestyle=ALGO_STYLE[algo],
                    linewidth=1.6, label=curve_label(algo, K))
            print(f'{name:8s} {algo:9s} K={K:3d} eta={eta:.0e} seeds={len(vals)} '
                  f'final={vals[:, -1].mean():.4f}')
        ax.set_xlim(0, args.exchanges)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Server exchanges')
        ax.set_ylabel(ylabel)
        ax.set_facecolor('#EAEAF2')
        ax.grid(True, which='major', color='white', linewidth=1.0)
        ax.grid(False, which='minor')
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = 3 if len(names) == 1 else 4
    fig.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    out = Path(args.out).resolve()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')

    if args.table:
        lines = []
        for i, name in enumerate(['accuracy', 'brier', 'nll']):
            col, _, lower, _ = METRICS[name]
            if i:
                lines.append('\\midrule')
            for j, (algo, K) in enumerate(ALGO_K):
                eta, comm, vals = select(cells, algo, K, col, lower, args.exchanges)
                final = vals[:, -1]
                se = final.std(ddof=1) / np.sqrt(len(final))
                metric = TABLE_NAMES[name] if j == 0 else ''
                lines.append(f'{metric} & {method_name(algo, K)} & {K} & {comm[-1]} & '
                             f'{fmt_eta(eta)} & {fmt_val(name, final.mean(), se)} \\\\')
        Path(args.table).write_text('\n'.join(lines) + '\n')
        print(f'wrote {args.table}')

    if args.trajectory:
        points = [2000, 2400, 2800, 3200, 3600]
        lines = []
        for algo, K in ALGO_K:
            eta, comm, vals = select(cells, algo, K, COL_ACC, False, args.exchanges)
            x = comm * EXCHANGES_PER_ROUND[algo]
            row = [vals[:, np.where(x == p)[0][0]].mean() for p in points]
            lines.append(f'{method_name(algo, K)} & {K} & {fmt_eta(eta)} & '
                         + ' & '.join(f'{v:.4f}' for v in row) + ' \\\\')
        Path(args.trajectory).write_text('\n'.join(lines) + '\n')
        print(f'wrote {args.trajectory}')


if __name__ == '__main__':
    main()
