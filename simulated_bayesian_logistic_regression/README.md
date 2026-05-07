# Simulated Bayesian Logistic Regression

This folder contains the simulated Bayesian logistic regression experiment code.

Only two production sweep launchers are included in this anonymous export:

```bash
scripts/run_d1000_full_eta_grid.sh
scripts/run_d10_full_eta_grid.sh
```

They run the full $\eta$ grid for FA-LD, FA-HMC, and adaptive FA-HMC.

| launcher | dimension | gradient setting | output subfolder |
|---|---:|---|---|
| `run_d1000_full_eta_grid.sh` | $d=1000$ | exact gradients | `sweep/d1000_G` |
| `run_d10_full_eta_grid.sh` | $d=10$ | stochastic gradients with $\sigma=10$ | `sweep/d10_SG` |

## Files

| file | purpose |
|---|---|
| `scripts/run_logistic_regression_cell.py` | GPU runner for one $(\mathrm{dimension}, \mathrm{algorithm}, \eta, K)$ cell |
| `scripts/run_d1000_full_eta_grid.sh` | single-GPU full $\eta$-grid PBS sweep for $d=1000$ |
| `scripts/run_d10_full_eta_grid.sh` | single-GPU full $\eta$-grid PBS sweep for $d=10$ |
| `scripts/plot_logistic_regression_results.py` | computes marginal Wasserstein errors and generates plots |
| `scripts/alg_fed_logreg_gpu.py` | GPU implementations of FA-HMC and adaptive FA-HMC |
| `data/synthetic_data.mat` | $d=1000$ dataset |
| `data/synthetic_data_d10.mat` | $d=10$ dataset |

## Running

Set the repository and output roots, then submit the desired sweep:

```bash
export PROJECT_ROOT=/path/to/repository
export SCRATCH_ROOT=/path/to/output/root

cd "$PROJECT_ROOT/simulated_bayesian_logistic_regression/scripts"
qsub run_d1000_full_eta_grid.sh
qsub run_d10_full_eta_grid.sh
```

## Plotting

Use `plot_logistic_regression_results.py` after MHMC reference samples and sweep outputs are available. Example:

```bash
python plot_logistic_regression_results.py \
  --sweep_dir "$SCRATCH_ROOT/simulated_bayesian_logistic_regression/sweep/d10_SG" \
  --mhmc /path/to/mhmc_samples.mat \
  --out logistic_regression_d10_stochastic.png \
  --title "\$d=10\$, stochastic gradients"
```
