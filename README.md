# Reproducibility Code

This repository contains the runnable code for the three experiments:

| folder | experiment |
|---|---|
| `simulated_bayesian_logistic_regression/` | simulated Bayesian logistic regression |
| `dimension_scaling/` | dimension-scaling experiment |
| `fmnist/` | Fashion-MNIST experiment |

## Running on a cluster

The PBS launchers use two environment variables:

```bash
export PROJECT_ROOT=/path/to/this/repository
export SCRATCH_ROOT=/path/to/output/root
```

If unset, `PROJECT_ROOT` is inferred from the launcher location and
`SCRATCH_ROOT` defaults to `$PROJECT_ROOT/results`.

Replace `#PBS -A YOUR_ALLOCATION` in the launcher scripts with the appropriate
allocation name for your cluster.

## Plot Generation

Each experiment folder contains its own README with the relevant run and plot
commands. At a high level:

```bash
# Simulated Bayesian logistic regression
cd simulated_bayesian_logistic_regression/scripts
qsub run_d1000_full_eta_grid.sh
qsub run_d10_full_eta_grid.sh
python plot_logistic_regression_results.py --help

# Dimension scaling
cd dimension_scaling/scripts
qsub run_adaptive_hmc_scaling.sh
qsub run_vanilla_baseline.sh
python plot_dimension_scaling_results.py

# Fashion-MNIST
cd fmnist/scripts
qsub run_fmnist_heterogeneous_400_tuning.sh
python plot_fmnist_results.py --help
```
