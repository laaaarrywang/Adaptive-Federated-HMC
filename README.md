# Adaptive Federated Hamiltonian Monte Carlo (AFHMC)

Code for the NeurIPS 2026 paper

> **Breaking the $\sqrt{d}$ Communication Barrier in Federated Sampling with Adaptive Hamiltonian Monte Carlo**
> Jiajun Liang, Linxuan Wang, Guang Lin, Qifan Song

AFHMC adds a drift (gradient) correction to federated HMC so that local
trajectories do not drift toward the local posteriors. The repository contains
GPU implementations of AFHMC, FA-HMC and FA-LD and the scripts for every
experiment in the paper.

## Experiments

| folder | experiment | paper |
|---|---|---|
| `simulated_bayesian_logistic_regression/` | simulated Bayesian logistic regression ($d=1000$ exact gradients, $d=10$ noisy gradients) | Figure 2, appendix grid |
| `dimension_scaling/` | heterogeneous Gaussian dimension scaling | Figure 3 (left) |
| `dimension_scaling/` | low-heterogeneity Gaussian dimension scaling | Figure 3 (middle), appendix |
| `dimension_scaling/` | FA-HMC baselines, including FA-HMC-specific stepsizes at an equal number of server exchanges | Figure 3 (right), appendix |
| `fmnist/` | Bayesian logistic regression on heterogeneous Fashion-MNIST (400-round stepsize tuning, 3600-round runs) | Figure 4, appendix |

Each folder has its own README with run and plot commands.

**Communication accounting.** One AFHMC round consists of two server
exchanges (gradient aggregation for the drift correction, then parameter
aggregation), whereas one FA-HMC/FA-LD round consists of one. The
equal-exchange comparisons therefore read AFHMC at round $R$ against FA-HMC at
round $2R$.

## Requirements

Python 3.10 with

- `torch` and `torchvision` (tested with PyTorch 2.6; a CUDA GPU is required)
- `numpy`, `scipy`, `matplotlib`, `Pillow`

Fashion-MNIST is downloaded automatically by `torchvision` on first use.

## Running on a cluster

The PBS launchers use two environment variables:

```bash
export PROJECT_ROOT=/path/to/this/repository
export SCRATCH_ROOT=/path/to/output/root
```

If unset, `PROJECT_ROOT` is inferred from the launcher location and
`SCRATCH_ROOT` defaults to `$PROJECT_ROOT/results`.

Replace `#PBS -A YOUR_ALLOCATION` and `#PBS -l filesystems=YOUR_FILESYSTEMS`
in the launcher scripts with the values for your cluster, and load your Python
environment where indicated. The launchers run on one GPU (`GPU_ID`, default 0)
and skip cells whose results already exist, so they can be resubmitted after a
timeout.

## Quick reference

```bash
# Simulated Bayesian logistic regression (Figure 2)
cd simulated_bayesian_logistic_regression/scripts
qsub run_d1000_full_eta_grid.sh
qsub run_d10_full_eta_grid.sh
python plot_logistic_regression_results.py --help

# Dimension scaling (Figure 3)
cd dimension_scaling/scripts
qsub run_adaptive_hmc_scaling.sh          # heterogeneous target, AFHMC
qsub run_vanilla_baseline.sh              # FA-HMC, original setting (after the AFHMC sweep)
qsub run_vanilla_equal_exchange.sh        # FA-HMC-specific schedules at 2R(d) (after the AFHMC sweep)
qsub run_adaptive_hmc_lowhet_scaling.sh   # low-heterogeneity target, AFHMC
python make_camera_ready_figures.py

# Fashion-MNIST (Figure 4)
cd fmnist/scripts
qsub run_fmnist_heterogeneous_400_tuning.sh   # stepsize tuning
qsub run_fmnist_heterogeneous_3600.sh         # long runs at the selected stepsizes
python make_camera_ready_figures.py --results "$SCRATCH_ROOT/fmnist/heterogeneous_3600"
```

## Citation

```bibtex
@inproceedings{liang2026breaking,
  title     = {Breaking the $\sqrt{d}$ Communication Barrier in Federated Sampling
               with Adaptive {H}amiltonian {M}onte {C}arlo},
  author    = {Liang, Jiajun and Wang, Linxuan and Lin, Guang and Song, Qifan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT License; see `LICENSE`.
