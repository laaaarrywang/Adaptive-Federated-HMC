# Dimension Scaling Experiment

Gaussian dimension-scaling experiments (paper Figure 3 and appendix).

Heterogeneous Gaussian setup:
- $N=10$ clients; first half $\mathcal{N}(20*\mathbf{1}_d, I_d)$, second half
  $\mathcal{N}(\mathbf{1}_d, 4I_d)$
- Aggregate target: $\mathcal{N}(16.2*\mathbf{1}_d, 1.6I_d)$
- $W_2^2$ is computed against this aggregate.

Four experiments:
1. **Adaptive FA-HMC scaling** - communication rounds $R(d)$ to reach
   $W_2^2 < 0.1$.
   $K(d)=\mathrm{round}(50(d/100)^{1/3})$,
   $\eta(d)=0.006(d/100)^{-1/2}$, and $T=1$.
2. **Vanilla FA-HMC at matched compute** - same $\eta(d)$, same total
   leapfrog steps per round
   ($K_{\mathrm{v}}T_{\mathrm{v}}=K_{\mathrm{a}}$, $T_{\mathrm{v}}=10$),
   and same $R$ per $d$ as the adaptive experiment.
   Measures final $W_2^2$ (no gradient correction, so bias does not vanish).
3. **FA-HMC at an equal number of server exchanges** - FA-HMC is run for
   $2R(d)$ rounds (AFHMC uses two server exchanges per round) with two
   FA-HMC-specific stepsize schedules:
   $K=5$, $T=10$, $\eta(d)=0.02\,d^{-1/4}$; and the Theorem-4.1-motivated
   schedule $K(d)=\max\{1,\mathrm{round}(K_{\mathrm{a}}(d)/10)\}$, $T=10$,
   $\eta(d)=0.006(d/100)^{-7/12}$.
4. **Low-heterogeneity scaling** - every client estimates its local Gaussian
   from $10^5$ samples of $\mathcal{N}(16.2*\mathbf{1}_d, 1.6I_d)$ and the pooled
   samples define the target, so heterogeneity only comes from sampling error.
   AFHMC uses $K(d)=\mathrm{round}(50(d/100)^{1/2})$,
   $\eta(d)=0.006(d/100)^{-1/2}$, three seeds.

All runners estimate $W_2^2$ from $M(d)=\min\{100d(d-1), M_{\mathrm{budget}}\}$
parallel chains, where $M_{\mathrm{budget}}$ follows `--mem_budget_gb`
(default 28 GB per GPU; $\lfloor 10^8/d\rfloor$ for $N=10$, binding for
$d \ge 150$).

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
| `run_vanilla_equal_exchange.sh`   | Single-GPU PBS submission for experiment 3; reads $R(d)$ from the adaptive sweep and runs both FA-HMC schedules for $2R(d)$ rounds. |
| `run_adaptive_hmc_lowhet_scaling_d.py` | **GPU runner for experiment 4.** Builds the sampled low-heterogeneity clients and runs AFHMC with the same stopping rule. |
| `run_adaptive_hmc_lowhet_scaling.sh` | Single-GPU PBS submission for experiment 4 over all $d$ and seeds 0-2. |
| `make_camera_ready_figures.py`    | Writes the three panels of paper Figure 3 and the LaTeX rows of the two appendix tables. |

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
  the gradient correction removes.

## Reproducing

From a cluster login node with the required Python environment:

```bash
qsub scripts/run_adaptive_hmc_scaling.sh    # experiment 1
qsub scripts/run_vanilla_baseline.sh        # experiment 2
qsub scripts/run_vanilla_equal_exchange.sh  # experiment 3
qsub scripts/run_adaptive_hmc_lowhet_scaling.sh  # experiment 4
python scripts/plot_dimension_scaling_results.py
python scripts/make_camera_ready_figures.py
```

The rerun scripts write npz files under
`$SCRATCH_ROOT/dimension_scaling/{adaptive_hmc_scaling,vanilla_baseline}/`.
Run the adaptive sweep first, then experiments 2 and 3 so they can read the
adaptive first-cross rounds. Experiments 3 and 4 write to
`$SCRATCH_ROOT/dimension_scaling/{vanilla_equal_exchange,lowhet_adaptive_hmc_scaling}/`,
which is where `make_camera_ready_figures.py` looks by default (override with the
`DIMENSION_SCALING_*_DIR` variables listed in the script). To plot directly from the rerun outputs, use
`DIMENSION_SCALING_ADAPT_DIR=$SCRATCH_ROOT/dimension_scaling/adaptive_hmc_scaling
DIMENSION_SCALING_VANILLA_DIR=$SCRATCH_ROOT/dimension_scaling/vanilla_baseline python scripts/plot_dimension_scaling_results.py`.
