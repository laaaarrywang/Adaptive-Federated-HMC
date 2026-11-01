"""Generate the three panels of the dimension-scaling figure (paper Fig. 2).

  fig2_cr_rounds_vs_d.png:
      AFHMC first-crossing round R(d) on the heterogeneous Gaussian with the
      theory curve C (d/eps^2)^{1/3} log(d/eps^2).
  fig2_cr_lowhet_rounds_vs_d.png:
      AFHMC first-crossing round R(d) on the low-heterogeneity Gaussian
      (mean +/- sd over seeds) with the log fit R(d) = a + b log(d / eps^2).
  fig2_cr_fahmc_equal_exchange.png:
      FA-HMC W2^2 vs d for (i) the original matched-budget run (AFHMC eta(d),
      R(d) rounds) and (ii) the two FA-HMC schedules at the equal-exchange
      budget 2R(d).

Also writes the appendix table rows (LaTeX) for both experiments:
  fig2_cr_lowhet_table.tex, fig2_cr_fahmc_table.tex
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent.parent
# Default locations match the launchers in this folder.
RESULTS = Path(os.environ.get('SCRATCH_ROOT', str(HERE.parent / 'results'))) / 'dimension_scaling'
LOWHET_DIR = Path(os.environ.get('DIMENSION_SCALING_LOWHET_DIR', str(RESULTS / 'lowhet_adaptive_hmc_scaling')))
ADAPT_DIR = Path(os.environ.get('DIMENSION_SCALING_ADAPT_DIR', str(RESULTS / 'adaptive_hmc_scaling')))
VANILLA_DIR = Path(os.environ.get('DIMENSION_SCALING_VANILLA_DIR', str(RESULTS / 'vanilla_baseline')))
EQUAL_EXCHANGE_DIR = Path(os.environ.get('DIMENSION_SCALING_EQUAL_EXCHANGE_DIR', str(RESULTS / 'vanilla_equal_exchange')))
FIG_DIR = Path(os.environ.get('DIMENSION_SCALING_FIG_DIR', str(HERE / 'figures')))
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [2, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SEEDS = [0, 1, 2]
EPS2 = 0.1
# Three panels share a row at 0.32\linewidth; a smaller canvas keeps text legible.
FIGSIZE = (4.3, 3.4)
ROUNDS_YLABEL = r'Rounds $R(d)$ to reach $W_2^2 < 0.1$'


def apply_fig1_style(ax):
    """Match the Figure 1 ggplot-style background and grid."""
    ax.set_facecolor('#EAEAF2')
    ax.grid(True, which='major', color='white', linewidth=1.2)
    ax.grid(False, which='minor')
    for spine in ax.spines.values():
        spine.set_visible(False)


def fit_powerlaw(ds, ys):
    a, lc = np.polyfit(np.log(ds), np.log(ys), 1)
    return float(a), float(math.exp(lc))


def load_lowhet():
    means, sds = [], []
    for d in DIMS:
        rounds = []
        for s in SEEDS:
            z = np.load(LOWHET_DIR / f'seed{s}' / f'scaling_d{d:03d}.npz')
            fc = int(z['first_cross'])
            if fc <= 0:
                raise ValueError(f'd={d} seed={s} did not cross the threshold')
            rounds.append(fc)
        rounds = np.array(rounds, dtype=float)
        means.append(rounds.mean())
        sds.append(rounds.std(ddof=1))
    return np.array(DIMS, dtype=float), np.array(means), np.array(sds)


def w2_at_round(path, round_):
    z = np.load(path)
    rounds = np.asarray(z['trace_rounds']).astype(int)
    idx = np.where(rounds == round_)[0]
    if idx.size == 0:
        raise ValueError(f'round {round_} not in {path} ({rounds[0]}..{rounds[-1]})')
    return float(np.asarray(z['trace_w2'])[idx[0]])


def load_fahmc():
    orig, legacy, thm41 = [], [], []
    for d in DIMS:
        R = int(np.load(ADAPT_DIR / f'scaling_d{d:03d}.npz')['first_cross'])
        orig.append(w2_at_round(VANILLA_DIR / f'vanilla_d{d:03d}.npz', R))
        legacy.append(w2_at_round(EQUAL_EXCHANGE_DIR / 'legacy_k5_t10_eta_d14' / f'vanilla_d{d:03d}.npz', 2 * R))
        thm41.append(w2_at_round(EQUAL_EXCHANGE_DIR / 'theorem41_k13_eta_d712' / f'vanilla_d{d:03d}.npz', 2 * R))
    return np.array(DIMS, dtype=float), np.array(orig), np.array(legacy), np.array(thm41)


def plot_rounds(out_path):
    ds = np.array(DIMS, dtype=float)
    Rs = np.array([int(np.load(ADAPT_DIR / f'scaling_d{d:03d}.npz')['first_cross'])
                   for d in DIMS], dtype=float)
    # Theory shape with only the multiplicative constant fitted (log-residual mean).
    shape = (ds / EPS2) ** (1.0 / 3.0) * np.log(ds / EPS2)
    c_th = math.exp(np.mean(np.log(Rs) - np.log(shape)))
    grid = np.geomspace(ds.min(), ds.max(), 300)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(ds, Rs, 'o', color='#377EB8', markersize=6, label='measured')
    ax.loglog(grid, c_th * (grid / EPS2) ** (1.0 / 3.0) * np.log(grid / EPS2), ':',
              color='0.25', linewidth=2.0,
              label=r'theory: $(d/\epsilon^2)^{1/3}\log(d/\epsilon^2)$')
    ax.set_xlabel('dimension $d$')
    ax.set_ylabel(ROUNDS_YLABEL)
    apply_fig1_style(ax)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}  theory C={c_th:.4f}')


def plot_lowhet(out_path):
    ds, means, sds = load_lowhet()
    x = np.log(ds / EPS2)
    b, a = np.polyfit(x, means, 1)
    r2 = 1.0 - np.sum((means - (a + b * x)) ** 2) / np.sum((means - means.mean()) ** 2)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(ds, means, yerr=sds, fmt='o', color='#377EB8', markersize=6,
                capsize=3, label='mean $\\pm$ sd (3 seeds)')
    grid = np.geomspace(ds.min(), ds.max(), 100)
    ax.plot(grid, a + b * np.log(grid / EPS2), ':', color='0.25', linewidth=1.8,
            label=rf'fit: ${a:.1f}+{b:.1f}\log(d/\epsilon^2)$')
    ax.set_xscale('log')
    ax.set_xlabel('dimension $d$')
    ax.set_ylabel(ROUNDS_YLABEL)
    apply_fig1_style(ax)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}  fit a={a:.3f} b={b:.3f} R2={r2:.4f}')


def plot_fahmc(out_path):
    ds, orig, legacy, thm41 = load_fahmc()
    series = [
        (orig, 's', '#E41A1C', r'original setting, $R(d)$'),
        (legacy, '^', '#FF7F00', r'$K{=}5$, $\eta{=}0.02d^{-1/4}$, $2R(d)$'),
        (thm41, 'D', '#984EA3', r'Thm. 4.1 $\eta\propto d^{-7/12}$, $2R(d)$'),
    ]

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ys, marker, color, label in series:
        beta, _ = fit_powerlaw(ds, ys)
        ax.loglog(ds, ys, marker=marker, linestyle='-', color=color, markersize=5,
                  label=rf'{label} ($d^{{{beta:.2f}}}$)')
        print(f'  {label}: exponent {beta:.3f}, d=2 {ys[0]:.4g}, d=500 {ys[-1]:.4g}')
    ax.axhline(EPS2, color='#377EB8', linestyle=':', linewidth=1.8,
               label='AFHMC threshold')
    ax.set_xlabel('dimension $d$')
    ax.set_ylabel(r'FA-HMC $W_2^2$')
    apply_fig1_style(ax)
    ax.legend(loc='center right', framealpha=0.9, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}')


def write_lowhet_table(out_path):
    """Rows: d & K & eta & per-seed first-crossing rounds & mean +/- sd."""
    lines = []
    mean_devs, var_devs = [], []
    for d in DIMS:
        rounds, Ks, etas = [], set(), set()
        for s in SEEDS:
            z = np.load(LOWHET_DIR / f'seed{s}' / f'scaling_d{d:03d}.npz')
            rounds.append(int(z['first_cross']))
            Ks.add(int(z['K']))
            etas.add(round(float(z['eta']), 12))
            mean_devs.append(float(z['client_mean_rms_dev']))
            var_devs.append(float(z['client_var_rms_dev']))
        assert len(Ks) == 1 and len(etas) == 1
        r = np.array(rounds, dtype=float)
        lines.append(f'{d} & {Ks.pop()} & {etas.pop():.4g} & '
                     + ', '.join(str(x) for x in rounds)
                     + f' & ${r.mean():.2f} \\pm {r.std(ddof=1):.2f}$ \\\\')
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f'wrote {out_path}')
    print(f'  client mean RMS deviation: {min(mean_devs):.4g}..{max(mean_devs):.4g}; '
          f'client variance RMS deviation: {min(var_devs):.3g}..{max(var_devs):.3g}')


def write_fahmc_table(out_path):
    """Rows: d & R & 2R & K_F & (eta, W2^2) for original / legacy / theorem-4.1.

    The original and theorem-4.1 runs share K_F(d) = max(1, round(K(d)/10));
    the legacy schedule uses K_F = 5.
    """
    def fmt_w2(x):
        return f'{x:.1f}'

    lines = []
    for d in DIMS:
        R = int(np.load(ADAPT_DIR / f'scaling_d{d:03d}.npz')['first_cross'])
        zo = np.load(VANILLA_DIR / f'vanilla_d{d:03d}.npz')
        zl = np.load(EQUAL_EXCHANGE_DIR / 'legacy_k5_t10_eta_d14' / f'vanilla_d{d:03d}.npz')
        zt = np.load(EQUAL_EXCHANGE_DIR / 'theorem41_k13_eta_d712' / f'vanilla_d{d:03d}.npz')
        wo = w2_at_round(VANILLA_DIR / f'vanilla_d{d:03d}.npz', R)
        wl = w2_at_round(EQUAL_EXCHANGE_DIR / 'legacy_k5_t10_eta_d14' / f'vanilla_d{d:03d}.npz', 2 * R)
        wt = w2_at_round(EQUAL_EXCHANGE_DIR / 'theorem41_k13_eta_d712' / f'vanilla_d{d:03d}.npz', 2 * R)
        assert int(zo['K']) == int(zt['K']) and int(zl['K']) == 5
        lines.append(
            f'{d} & {R} & {2 * R} & {int(zo["K"])} & {float(zo["eta"]):.4g} & {fmt_w2(wo)} & '
            f'{float(zl["eta"]):.4g} & {fmt_w2(wl)} & '
            f'{float(zt["eta"]):.4g} & {fmt_w2(wt)} \\\\')
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f'wrote {out_path}')


def main():
    write_lowhet_table(FIG_DIR / 'fig2_cr_lowhet_table.tex')
    write_fahmc_table(FIG_DIR / 'fig2_cr_fahmc_table.tex')
    plot_rounds(FIG_DIR / 'fig2_cr_rounds_vs_d.png')
    plot_lowhet(FIG_DIR / 'fig2_cr_lowhet_rounds_vs_d.png')
    plot_fahmc(FIG_DIR / 'fig2_cr_fahmc_equal_exchange.png')


if __name__ == '__main__':
    main()
