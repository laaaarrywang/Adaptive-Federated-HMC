# Fashion-MNIST Heterogeneous Experiment

This folder contains the Fashion-MNIST federated HMC experiment code
(paper Figure 4 and appendix). There are two launchers:

```bash
scripts/run_fmnist_heterogeneous_400_tuning.sh   # stepsize tuning, 400 rounds
scripts/run_fmnist_heterogeneous_3600.sh         # selected stepsizes, 3600 rounds
```

The tuning launcher runs label-skewed federated Fashion-MNIST with
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
| `scripts/run_fmnist_heterogeneous_3600.sh` | 3600-round runs at the selected stepsizes (50 cells); supports `NUM_SHARDS`/`SHARD_ID` to split the cells across GPUs or jobs |
| `scripts/make_camera_ready_figures.py` | paper Figure 4 and the appendix figure/tables on the equal server-exchange axis |

## Running

Set the repository and output roots, then submit:

```bash
export PROJECT_ROOT=/path/to/repository
export SCRATCH_ROOT=/path/to/output/root

cd "$PROJECT_ROOT/fmnist/scripts"
qsub run_fmnist_heterogeneous_400_tuning.sh
```

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

## Long runs and equal-exchange plots

The 3600-round launcher runs the stepsizes selected by the tuning sweep for
5 seeds each. The full set takes roughly 135 GPU-hours on one GPU; to use a
4-GPU node:

```bash
cd "$PROJECT_ROOT/fmnist/scripts"
for s in 0 1 2 3; do
  NUM_SHARDS=4 SHARD_ID=$s GPU_ID=$s bash run_fmnist_heterogeneous_3600.sh &
done
wait
```

AFHMC uses two server exchanges per round and FA-HMC one, so the plots put
FA-HMC round $r$ at $x=r$ and AFHMC round $r$ at $x=2r$ (AFHMC rounds 0-1800):

```bash
python make_camera_ready_figures.py --results "$SCRATCH_ROOT/fmnist/heterogeneous_3600" \
  --metrics accuracy --out fmnist_accuracy_equal_exchange.png \
  --table final_table.tex --trajectory trajectory_table.tex
python make_camera_ready_figures.py --results "$SCRATCH_ROOT/fmnist/heterogeneous_3600" \
  --metrics brier,nll --out fmnist_brier_nll_equal_exchange.png
```
