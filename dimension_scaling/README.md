# Dimension Scaling Experiment

Heterogeneous Gaussian setup from the paper Section 5.2:
- $N=10$ clients; first half $\mathcal{N}(20\mathbf{1}_d, I_d)$, second half
  $\mathcal{N}(\mathbf{1}_d, 4I_d)$
  (paper has a typo: $\sigma^2=2$ should be $\sigma^2=4$)
- Aggregate target: $\mathcal{N}(16.2\mathbf{1}_d, 1.6I_d)$
- $W_2^2$ is computed against this aggregate.

Two experiments:
1. **Adaptive FA-HMC scaling** - communication rounds $R(d)$ to reach
   $W_2^2 < 0.1$.
   $K(d)=\mathrm{round}(50(d/100)^{1/3})$,
   $\eta(d)=0.006(d/100)^{-1/2}$, and $T=1$.
2. **Vanilla FA-HMC at matched compute** - same $\eta(d)$, same total
   leapfrog steps per round
   ($K_{\mathrm{v}}T_{\mathrm{v}}=K_{\mathrm{a}}$, $T_{\mathrm{v}}=10$),
   and same $R$ per $d$ as the adaptive experiment.
   Measures final $W_2^2$ (no SCAFFOLD correction, so bias does not vanish).

## Layout

```
dimension_scaling/
|-- README.md                       - this file
|-- scripts/                        - single-GPU runnable production pipeline
`-- results/                        - generated npz traces when `SCRATCH_ROOT` is left at its default
```

## scripts/

| file | role |
|---|---|
| `bench_local_step_gpu.py`         | Provides `build_client_params(N, d, device, dtype)` - the only piece imported by the GPU runners. Guarantees both algorithms see the same client parameters. |
| `run_adaptive_hmc_scaling_d.py`   | **GPU runner for experiment 1.** Adaptive FA-HMC with $d$-dependent $K$ and $\eta$, adaptive stopping when $W_2^2$ is below the threshold for `consecutive_below=20` consecutive rounds. Saves trace to npz. |
| `run_adaptive_hmc_scaling.sh`     | Single-GPU PBS submission for experiment 1 over $d \in \{2, 50, 100, \ldots, 500\}$. |
| `run_vanilla_baseline_d.py`       | **GPU runner for experiment 2.** Vanilla FA-HMC for exactly `--R_target` rounds, no stopping rule. Computes $K_{\mathrm{v}}=\mathrm{round}(K_{\mathrm{a}}/T_{\mathrm{vanilla}})$ so total leapfrog steps per round match adaptive. Records $W_2^2$ every round. |
| `run_vanilla_baseline.sh`         | Single-GPU PBS submission for experiment 2; per-d `R_target` is read from the adaptive sweep's first-cross rounds. |
| `plot_dimension_scaling_results.py` | Loads both result sets, reports power/log-corrected scaling fits, writes the two PNGs. |

## results/

`adaptive_hmc_scaling/scaling_d{002,050,...,500}.npz` (11 files)
Keys: `d, M, K, eta, T, threshold, target_mean, target_var, algo,
trace_rounds, trace_mu, trace_var, trace_w2, first_cross, final_round,
stopped_early`.

`vanilla_baseline/vanilla_d{002,050,...,500}.npz` (11 files)
Keys: `d, M, K, eta, T, R_target, target_mean, target_var, algo,
trace_rounds, trace_mu, trace_var, trace_w2, final_round`.

## Generated plots

`rounds_vs_dimension.png` - log-log plot of communication rounds $R(d)$ for adaptive FA-HMC.
- A direct power-law fit reports the empirical exponent from the saved runs.
- The plotting script also reports the log-corrected fit
  $R(d)=C(d/\epsilon^2)^\alpha \log(d/\epsilon^2)$, both over all dimensions
  and excluding $d=2$.
- The figure shows the measured rounds and the theoretical reference shape
  $R(d)\propto (d/\epsilon^2)^{1/3}\log(d/\epsilon^2)$, with
  $\epsilon^2=0.1$.

`vanilla_error_vs_dimension.png` - log-log plot of vanilla FA-HMC's final $W_2^2$ at
the **same** $R$, $\eta$, total-leapfrog/round budget that the adaptive version used to
reach the threshold.
- Vanilla $W_2^2$ grows as $d^\beta$ with $\beta \approx 0.86$-$0.92$,
  roughly linear in $d$.
- Adaptive remains pinned at threshold $\approx 0.1$ (horizontal reference).
- The vanilla error is **3-5 orders of magnitude** above adaptive, driven by a
  persistent mean bias ($\hat{\mu} \approx 6$-$9$ vs target $16.2$) - the heterogeneity bias that
  SCAFFOLD removes.

## Reproducing

From a cluster login node with the required Python environment:

```bash
qsub scripts/run_adaptive_hmc_scaling.sh    # experiment 1
qsub scripts/run_vanilla_baseline.sh        # experiment 2
python scripts/plot_dimension_scaling_results.py
```

The rerun scripts write npz files under
`$SCRATCH_ROOT/dimension_scaling/{adaptive_hmc_scaling,vanilla_baseline}/`.
Run the adaptive sweep first, then the vanilla baseline so it can read the
adaptive first-cross rounds. To plot directly from the rerun outputs, use
`DIMENSION_SCALING_ADAPT_DIR=$SCRATCH_ROOT/dimension_scaling/adaptive_hmc_scaling
DIMENSION_SCALING_VANILLA_DIR=$SCRATCH_ROOT/dimension_scaling/vanilla_baseline python scripts/plot_dimension_scaling_results.py`.
