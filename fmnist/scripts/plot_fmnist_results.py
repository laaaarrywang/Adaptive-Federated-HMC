"""Generate Fashion-MNIST metrics vs communication rounds.

For each (algo, K, metric) triple, pick the eta minimising / maximising that
metric's final-round value averaged over seeds. The same (algo, K) curve in
two different panels may correspond to different eta values.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Columns of the (n_chkpt, 6) test_poster array, per cal_poster_metrics_initial.evaluation.
# The anonymous release plots NLL, accuracy, and Brier score.
COL_NLL = 0
COL_ACC = 1
COL_BRIER = 2


def find_cells(base: Path):
    """Yield (algo, K, eta, seed, post_path) tuples for each existing cell."""
    pat = re.compile(r'(?P<algo>fa_hmc|adaptive)/K(?P<K>\d+)/eta(?P<eta>[0-9.eE+-]+)/seed(?P<seed>\d+)')
    for post in base.glob('*/K*/eta*/seed*/post.npz'):
        s = str(post.relative_to(base))
        m = pat.search(s)
        if not m:
            continue
        yield (m['algo'], int(m['K']), float(m['eta']), int(m['seed']), post)


def load_cells(base: Path, drop_diverged: bool = True):
    """Returns dict: (algo, K, eta) -> list of (seed, comm_round, test_poster) tuples.

    If drop_diverged=True, any (algo, K, eta) cell where any seed has a non-finite
    NLL at the final round is excluded because these cells diverged
    numerically and their other metrics are computed from a degenerate chain.
    """
    cells = defaultdict(list)
    for algo, K, eta, seed, path in find_cells(base):
        try:
            z = np.load(path)
        except Exception as e:
            print(f'  skip {path}: {e}')
            continue
        cells[(algo, K, eta)].append((seed, np.asarray(z['comm_round']),
                                      np.asarray(z['test_poster'])))
    if drop_diverged:
        n_dropped = 0
        for key in list(cells.keys()):
            traces = [t for (_, _, t) in cells[key]]
            # Drop the cell if ANY seed has a non-finite NLL anywhere in its
            # trace. A single mid-trajectory inf still poisons the mean trace
            # and the y-axis range, even if the final round happens to recover.
            if any(not np.all(np.isfinite(t[:, COL_NLL])) for t in traces):
                del cells[key]
                n_dropped += 1
        if n_dropped:
            print(f'  dropped {n_dropped} (algo, K, eta) cells with non-finite NLL anywhere in trace')
    return cells


def pick_best_eta(cells, algo: str, K: int, criterion=COL_NLL, lower_is_better=True):
    """Pick the eta minimising mean criterion at the final round across seeds."""
    candidates = {eta: data for (a, k, eta), data in cells.items() if a == algo and k == K}
    if not candidates:
        return None
    scores = {}
    for eta, runs in candidates.items():
        if not runs:
            continue
        # final-round metric per seed; average across seeds
        finals = [trace[-1, criterion] for (_, _, trace) in runs]
        scores[eta] = np.mean(finals)
    if not scores:
        return None
    if lower_is_better:
        best_eta = min(scores, key=scores.get)
    else:
        best_eta = max(scores, key=scores.get)
    return best_eta, scores[best_eta]


def aggregate_runs(runs):
    """Compute mean and std trace across seeds. Use the seed=0 comm_round axis."""
    runs = sorted(runs, key=lambda r: r[0])
    comm_round = runs[0][1]
    traces = [r[2] for r in runs]
    n_chkpt = min(t.shape[0] for t in traces)
    stack = np.stack([t[:n_chkpt] for t in traces], axis=0)  # (n_seeds, n_chkpt, 6)
    return comm_round[:n_chkpt], stack.mean(axis=0), stack.std(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='../results',
                        help='base directory containing {algo}/K{K}/eta{eta}/seed{seed}/post.npz')
    parser.add_argument('--out', default='../fmnist_k_sweep.png')
    args = parser.parse_args()

    base = Path(args.results).resolve()
    cells = load_cells(base)
    print(f'loaded {sum(len(v) for v in cells.values())} cells across '
          f'{len(cells)} (algo, K, eta) groups')

    # FA-HMC K=1 is the FA-LD limit (single leapfrog step). AFHMC K=1 is
    # algorithmically degenerate (T_a = K_a = 1 means sync every leapfrog), so
    # we omit it to keep the legend focused on the adaptive K ∈ {10, 50, 100}
    # comparison.
    algo_K = [
        ('fa_hmc',   1),
        ('fa_hmc',  10),
        ('fa_hmc',  50),
        ('fa_hmc', 100),
        ('adaptive', 10),
        ('adaptive', 50),
        ('adaptive', 100),
    ]
    K_values = [1, 10, 50, 100]
    algos = ['fa_hmc', 'adaptive']

    # --- pick best eta per (metric, algo, K), matching the paper protocol ---
    metric_specs = [
        (COL_ACC,   'Accuracy',    False),  # higher is better
        (COL_BRIER, 'Brier Score', True),
        (COL_NLL,   'NLL',         True),
    ]
    chosen = {}  # (col, algo, K) -> (best_eta, comm_round, mean_trace, std_trace)
    for col, mname, lower in metric_specs:
        print(f'\n--- panel: {mname} (lower_is_better={lower}) ---')
        for algo, K in algo_K:
            best = pick_best_eta(cells, algo, K, criterion=col, lower_is_better=lower)
            if best is None:
                print(f'  WARN: no cells for {algo} K={K}')
                continue
            best_eta, score = best
            runs = cells[(algo, K, best_eta)]
            comm, mean_t, std_t = aggregate_runs(runs)
            chosen[(col, algo, K)] = (best_eta, comm, mean_t, std_t)
            print(f'  {algo:9s} K={K:3d}  best eta={best_eta:.0e}  '
                  f'value={score:.4f}  ({len(runs)} seeds)')

    # --- plot selected metric panels ---
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    metric_info = [
        (axes[0], COL_ACC,   'Accuracy',    False),
        (axes[1], COL_BRIER, 'Brier Score', True),
        (axes[2], COL_NLL,   'NLL',         True),
    ]
    # axis_lookup: column index -> the chosen-dict key prefix used by that panel
    # (lets the per-panel loop find this panel's per-metric eta picks).
    K_colors = {1: '#377EB8', 10: '#4DAF4A', 50: '#FF7F00', 100: '#E41A1C'}
    algo_style = {'fa_hmc': '-', 'adaptive': '--'}
    algo_label = {'fa_hmc': 'FA-HMC', 'adaptive': 'AFHMC'}
    # Special-case label: FA-HMC at K=1 is FA-LD (single leapfrog ≡ Langevin).
    def curve_label(algo, K):
        if algo == 'fa_hmc' and K == 1:
            return 'FA-LD (K=1)'
        return f'{algo_label[algo]} K={K}'

    # Smooth (continuously-differentiable) y-axis transform that maps three
    # anchor data values (lo, mid, hi) to visual positions (0, 0.5, 1).
    # We use the power transform fwd(y) = ((y - lo) / (hi - lo))^p, with p
    # chosen so fwd(mid) = 0.5. This is monotonic and smooth (no slope
    # discontinuity at the mid anchor) yet still places the three ticks at
    # evenly-spaced visual positions.
    def make_smooth(lo, mid, hi):
        u = (mid - lo) / (hi - lo)
        p = np.log(0.5) / np.log(np.clip(u, 1e-12, 1 - 1e-12))
        def fwd(y):
            y = np.asarray(y, dtype=float)
            r = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
            return r ** p
        def inv(t):
            t = np.asarray(t, dtype=float)
            r = np.clip(t, 0.0, 1.0) ** (1.0 / p)
            return lo + r * (hi - lo)
        return fwd, inv

    for ax, col, title, _ in metric_info:
        # First pass: collect data ranges (using THIS panel's per-metric eta picks)
        all_min = float('inf')
        all_max = float('-inf')
        finals = []
        last_comm = 0
        for algo in algos:
            for K in K_values:
                if (col, algo, K) not in chosen:
                    continue
                _, comm, mean_t, std_t = chosen[(col, algo, K)]
                m = mean_t[:, col]
                all_min = min(all_min, float(m.min()))
                all_max = max(all_max, float(m.max()))
                finals.append(float(m[-1]))
                last_comm = max(last_comm, float(comm[-1]))
        conv_min = min(finals)
        conv_max = max(finals)
        # Three anchor values mapped to visual positions 0, 0.5, 1.
        # Use global data extrema for the lo/hi anchors so no curve gets
        # clipped, and the worst-converged value as the middle anchor —
        # this magnifies the converged region between the curves' finals.
        # Add a margin equal to the converged-final spread on the "good" side
        # of the axis. Without this margin, anchor_lo (or anchor_hi for Acc)
        # sits exactly on the best curve's converged value, and the power-law
        # transform's near-singular slope at the boundary visually amplifies
        # tiny tail wobbles (e.g. a 3-NLL drift of FA-HMC K=100 over the last
        # 50 rounds becomes a 30% visual jump). Pushing the anchor outward by
        # the spread relocates the converged region into the moderate-slope
        # zone of the transform.
        spread = max(conv_max - conv_min, 1e-12)
        # Per-metric natural bounds: BS/Acc live in [0, 1]; NLL is in
        # [0, inf). Clamp the +spread margin so it doesn't push past these.
        nat_lo = 0.0  # all four metrics are non-negative
        nat_hi = 1.0 if col != COL_NLL else float('inf')
        if col == COL_ACC:
            anchor_lo = all_min                  # initial accuracy (low)
            anchor_mid = conv_min                # worst converged
            anchor_hi = min(max(all_max, conv_max + spread), nat_hi)  # best obs + margin, capped at 1
        else:
            anchor_lo = max(min(all_min, conv_min - spread), nat_lo)  # best obs - margin, floored at 0
            anchor_mid = conv_max                # worst converged
            anchor_hi = all_max                  # initial (highest)
        fwd, inv = make_smooth(anchor_lo, anchor_mid, anchor_hi)
        ax.set_yscale('function', functions=(fwd, inv))
        ax.set_ylim(anchor_lo, anchor_hi)

        # Second pass: plot mean traces without confidence bands.
        # Apply a light moving-average smoothing so tiny data fluctuations
        # (~0.001 in absolute units) don't get visually amplified by the
        # non-linear y-axis transform into apparent dips/spikes.
        SMOOTH_WIN = 7
        for algo in algos:
            for K in K_values:
                if (col, algo, K) not in chosen:
                    continue
                _, comm, mean_t, _ = chosen[(col, algo, K)]
                m = mean_t[:, col]
                if len(m) >= SMOOTH_WIN:
                    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
                    m_smooth = np.convolve(m, kernel, mode='same')
                    # Fix edges: replace the half-window endpoints with the raw
                    # mean (convolve 'same' contaminates them with zeros).
                    half = SMOOTH_WIN // 2
                    m_smooth[:half] = m[:half]
                    m_smooth[-half:] = m[-half:]
                    m = m_smooth
                color = K_colors[K]
                ls = algo_style[algo]
                ax.plot(comm, m, color=color, linestyle=ls,
                        linewidth=1.6, label=curve_label(algo, K))

        ax.set_xlim(0, last_comm)
        ax.set_xlabel('Communication rounds')
        ax.set_ylabel(title)

        # Three ticks at the anchor data values.
        decimals = {COL_ACC: 3, COL_BRIER: 3, COL_NLL: 0}[col]
        ticks = [anchor_lo, anchor_mid, anchor_hi]
        fmt = '{:.' + str(decimals) + 'f}' if decimals > 0 else '{:.0f}'
        ax.set_yticks(ticks)
        ax.set_yticklabels([fmt.format(t) for t in ticks])
        ax.set_facecolor('#EAEAF2')
        ax.grid(True, which='major', color='white', linewidth=1.0)
        ax.grid(False, which='minor')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
