"""Compute ME = (1/d) * sum_i W_1(coord i, MHMC coord i) for every sweep cell,
pick the K minimizing ME at each (algo, eta), and write a summary plot.

Inputs:
  --sweep_dir : directory of sweep npz files (one per (algo, eta, K) cell)
  --mhmc      : path to MHMC samples.mat (key 'X' shape (d, n_mhmc))
  --out       : path to output PNG
  --title     : plot title
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance


def me_per_panel(samples_Md: np.ndarray, mhmc_dn: np.ndarray) -> float:
    """samples_Md: (n_samples, d), mhmc_dn: (d, n_mhmc). Returns ME = avg W1."""
    d = mhmc_dn.shape[0]
    assert samples_Md.shape[1] == d
    total = 0.0
    for i in range(d):
        total += wasserstein_distance(samples_Md[:, i], mhmc_dn[i, :])
    return total / d


def load_cell(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    samples = z['samples']  # (S, M, d)
    S, M, d = samples.shape
    # use last half across time, pool over chains -> (S/2 * M, d)
    half = samples[S // 2 :]
    pooled = half.reshape(-1, d)
    return {
        'algo': str(z['algo']),
        'eta': float(z['eta']),
        'K': int(z['K']),
        'T': int(z['T']),
        'pooled': pooled,
        'elapsed': float(z['elapsed_seconds']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', required=True)
    parser.add_argument('--mhmc', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--title', default='')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    mhmc = np.asarray(loadmat(args.mhmc)['X'], dtype=float)  # (d, n_mhmc)
    print(f'MHMC shape={mhmc.shape}')

    cells = []
    for npz in sorted(sweep_dir.glob('*.npz')):
        try:
            cell = load_cell(npz)
        except Exception as e:
            print(f'skip {npz}: {e}')
            continue
        me = me_per_panel(cell['pooled'], mhmc)
        cell['ME'] = me
        cells.append(cell)
        print(f'  {npz.name}: algo={cell["algo"]} eta={cell["eta"]} '
              f'K={cell["K"]} ME={me:.4f}  elapsed={cell["elapsed"]:.1f}s')

    # Pick best K per (algo, eta)
    best = {}  # (algo, eta) -> (ME, K)
    for c in cells:
        key = (c['algo'], c['eta'])
        if key not in best or c['ME'] < best[key][0]:
            best[key] = (c['ME'], c['K'])

    algos = sorted({a for a, _ in best.keys()})
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = {'fa_ld': '#E41A1C', 'fa_hmc': '#377EB8', 'adaptive': '#4DAF4A'}
    markers = {'fa_ld': 'o', 'fa_hmc': 's', 'adaptive': '^'}
    labels = {'fa_ld': 'FA-LD', 'fa_hmc': 'FA-HMC', 'adaptive': 'AFHMC'}
    for algo in algos:
        pts = sorted([(eta, me) for (a, eta), (me, _) in best.items() if a == algo])
        if not pts:
            continue
        etas, mes = zip(*pts)
        ax.loglog(etas, mes, marker=markers.get(algo, 'x'),
                  color=colors.get(algo, 'k'), label=labels.get(algo, algo),
                  linewidth=1.6, markersize=7)

    ax.set_xlabel(r'Stepsize $\eta$')
    ax.set_ylabel('Marginal Error (ME)')
    # ggplot-style: light grey background, white major gridlines, no minor grid.
    ax.set_facecolor('#EAEAF2')
    ax.grid(True, which='major', color='white', linewidth=1.2)
    ax.grid(False, which='minor')
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Force ticks at 10^-2, 10^-1, 10^0, 10^1 on the y-axis. Pad ylim so edge
    # ticks aren't clipped by matplotlib.
    from matplotlib.ticker import LogLocator, NullLocator
    all_mes = [me for (me, _) in best.values()]
    y_lo = min(min(all_mes) * 0.5, 7e-3)
    y_hi = max(max(all_mes) * 2.0, 1.5e1)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
