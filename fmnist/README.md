# Fashion-MNIST Heterogeneous Experiment

This folder contains the Fashion-MNIST federated HMC experiment code.

```bash
scripts/run_fmnist_heterogeneous_400_tuning.sh
```

The launcher runs label-skewed federated Fashion-MNIST with
$\mathrm{Dirichlet}(\alpha=0.1)$ client splits for:

- FA-HMC with $K \in \{1, 10, 50, 100\}$
- AFHMC with $K \in \{10, 50, 100\}$
- 5 seeds per cell
- 400 communication rounds per cell
- Full heterogeneous $\eta$ grid:
  $\{5{\times}10^{-4}, 2{\times}10^{-4}, 10^{-4}, 5{\times}10^{-5},
  2{\times}10^{-5}, 10^{-5}, 5{\times}10^{-6}, 2{\times}10^{-6},
  10^{-6}, 5{\times}10^{-7}, 2{\times}10^{-7}\}$

## Files

| file | purpose |
|---|---|
| `logistic_fashion.py` | training entrypoint |
| `cal_poster_metrics_initial.py` | posterior predictive metric accumulation |
| `model.py` | Fashion-MNIST linear model |
| `tools.py` | test-data loading utilities used by posterior evaluation |
| `tools_fast.py`, `trainer_fast.py` | GPU-preloaded fast path used by the launcher |
| `transforms.py` | torchvision transform helpers |
| `scripts/run_fmnist_heterogeneous_400_tuning.sh` | single-GPU full heterogeneous 400-round $\eta$ sweep |
| `scripts/plot_fmnist_results.py` | plot generator for Accuracy, Brier score, and NLL from saved `post.npz` files |

## Running

Set the repository and output roots, then submit:

```bash
export PROJECT_ROOT=/path/to/repository
export SCRATCH_ROOT=/path/to/output/root

cd "$PROJECT_ROOT/fmnist/scripts"
qsub run_fmnist_heterogeneous_400_tuning.sh
```

If unset, `PROJECT_ROOT` is inferred from the launcher location and `SCRATCH_ROOT` defaults to `$PROJECT_ROOT/results`.

## Plotting

After the sweep finishes:

```bash
# Load your Python environment here, for example:
# module use /path/to/modulefiles
# module load YOUR_PYTHON_MODULE
# conda activate YOUR_ENV

python "$PROJECT_ROOT/fmnist/scripts/plot_fmnist_results.py" \
  --results "$SCRATCH_ROOT/fmnist/heterogeneous_400" \
  --out "$PROJECT_ROOT/fmnist/fmnist_heterogeneous_400_k_sweep.png"
```

The plotter reports metric-specific $\eta$ choices and generates panels for
Accuracy, Brier score, and NLL.
