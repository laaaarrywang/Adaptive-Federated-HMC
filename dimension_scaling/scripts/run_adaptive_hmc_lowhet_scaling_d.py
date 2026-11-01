"""Low-heterogeneity AFHMC dimension-scaling runner.

Sampled-client construction used for the low-heterogeneity experiment:
for each client, draw `samples_per_client` samples from
N(parent_mean * 1_d, parent_var * I_d), estimate that client's Gaussian
parameters from those samples, and use the samples pooled over all clients to
define the W2 target.

The default variance mode is scalar, i.e. each local client target is
N(mu_hat_c, sigma_hat_c^2 I_d). A diagonal mode is also provided in case the
sample variance is later interpreted coordinate-wise.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Literal

import numpy as np
import torch


VarianceMode = Literal['scalar', 'diagonal']


def build_sampled_lowhet_params(
    *,
    N: int,
    d: int,
    samples_per_client: int,
    parent_mean: float,
    parent_var: float,
    stats_batch_size: int,
    variance_mode: VarianceMode,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    client_mu = torch.empty((N, d), device=device, dtype=dtype)
    if variance_mode == 'scalar':
        client_var = torch.empty((N,), device=device, dtype=dtype)
    else:
        client_var = torch.empty((N, d), device=device, dtype=dtype)

    total_sum = torch.zeros((d,), device=device, dtype=dtype)
    total_sumsq = torch.zeros((d,), device=device, dtype=dtype)
    parent_std = math.sqrt(parent_var)
    batch_size = min(stats_batch_size, samples_per_client)

    for c in range(N):
        local_sum = torch.zeros((d,), device=device, dtype=dtype)
        local_sumsq = torch.zeros((d,), device=device, dtype=dtype)
        remaining = samples_per_client
        while remaining > 0:
            n_batch = min(batch_size, remaining)
            samples = parent_mean + parent_std * torch.randn(
                (n_batch, d), device=device, dtype=dtype, generator=gen
            )
            local_sum += samples.sum(dim=0)
            local_sumsq += samples.square().sum(dim=0)
            remaining -= n_batch
            del samples

        mu_c = local_sum / samples_per_client
        var_c_vec = local_sumsq / samples_per_client - mu_c.square()
        var_c_vec = torch.clamp(var_c_vec, min=1e-12)

        client_mu[c].copy_(mu_c)
        if variance_mode == 'scalar':
            client_var[c] = var_c_vec.mean()
        else:
            client_var[c].copy_(var_c_vec)

        total_sum += local_sum
        total_sumsq += local_sumsq

    total_n = N * samples_per_client
    target_mu = total_sum / total_n
    target_var_vec = total_sumsq / total_n - target_mu.square()
    target_var_vec = torch.clamp(target_var_vec, min=1e-12)
    if variance_mode == 'scalar':
        target_var = target_var_vec.mean()
    else:
        target_var = target_var_vec

    return client_mu, client_var, target_mu, target_var


def compute_w2_to_sampled_target(
    X: torch.Tensor,
    target_mu: torch.Tensor,
    target_var: torch.Tensor,
):
    d = X.shape[1]
    sample_mu = X.mean(dim=0)
    sample_var_vec = X.var(dim=0, unbiased=False)
    mean_term = torch.sum((sample_mu - target_mu) ** 2)

    if target_var.ndim == 0:
        sample_var = sample_var_vec.mean()
        var_term = d * (torch.sqrt(torch.clamp(sample_var, min=0.0))
                        - torch.sqrt(target_var)) ** 2
        trace_var = sample_var
        final_sample_var = sample_var
    else:
        sample_var_vec = torch.clamp(sample_var_vec, min=0.0)
        var_term = torch.sum((torch.sqrt(sample_var_vec)
                              - torch.sqrt(target_var)) ** 2)
        trace_var = sample_var_vec.mean()
        final_sample_var = sample_var_vec

    w2_sq = mean_term + var_term
    return w2_sq, mean_term, var_term, sample_mu, final_sample_var, trace_var


def adaptive_hmc_lowhet(
    *,
    d: int,
    K: int,
    eta: float,
    M: int,
    N: int,
    T: int,
    threshold: float,
    max_rounds: int,
    samples_per_client: int,
    parent_mean: float,
    parent_var: float,
    stats_batch_size: int,
    variance_mode: VarianceMode,
    stats_seed: int,
    hmc_seed: int,
    check_every: int = 1,
    consecutive_below: int = 20,
    device: torch.device = torch.device('cuda'),
    dtype: torch.dtype = torch.float32,
    verbose_every: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 100,
):
    client_mu, client_var, target_mu, target_var = build_sampled_lowhet_params(
        N=N,
        d=d,
        samples_per_client=samples_per_client,
        parent_mean=parent_mean,
        parent_var=parent_var,
        stats_batch_size=stats_batch_size,
        variance_mode=variance_mode,
        device=device,
        dtype=dtype,
        seed=stats_seed,
    )

    hmc_gen = torch.Generator(device=device)
    hmc_gen.manual_seed(hmc_seed)

    mu_b = client_mu.unsqueeze(0).contiguous()
    if client_var.ndim == 1:
        inv_var_b = (1.0 / client_var).view(1, N, 1).contiguous()
    else:
        inv_var_b = (1.0 / client_var).unsqueeze(0).contiguous()

    mean_dev = torch.sqrt(torch.mean((client_mu - target_mu.unsqueeze(0)) ** 2)).item()
    if client_var.ndim == 1:
        var_dev = torch.sqrt(torch.mean((client_var - target_var) ** 2)).item()
        target_var_summary = target_var.item()
    else:
        var_dev = torch.sqrt(torch.mean((client_var - target_var.unsqueeze(0)) ** 2)).item()
        target_var_summary = target_var.mean().item()

    X0 = torch.randn((M, d), device=device, dtype=dtype, generator=hmc_gen)

    trace_rounds: list[int] = []
    trace_mu_avg: list[float] = []
    trace_var_avg: list[float] = []
    trace_w2: list[float] = []
    trace_mean_term: list[float] = []
    trace_var_term: list[float] = []
    first_cross = None
    streak = 0

    def as_numpy(x: torch.Tensor):
        return x.detach().cpu().numpy()

    def save(tag: str = ''):
        if checkpoint_path is None:
            return
        w2_sq, mean_term, var_term, sample_mu, sample_var, trace_var = (
            compute_w2_to_sampled_target(X0, target_mu, target_var)
        )
        tmp = checkpoint_path + '.tmp.npz'
        np.savez(
            tmp[:-4],
            d=np.array(d),
            N=np.array(N),
            M=np.array(M),
            K=np.array(K),
            T=np.array(T),
            eta=np.array(eta),
            threshold=np.array(threshold),
            max_rounds=np.array(max_rounds),
            samples_per_client=np.array(samples_per_client),
            parent_mean=np.array(parent_mean),
            parent_var=np.array(parent_var),
            variance_mode=np.array(variance_mode),
            stats_seed=np.array(stats_seed),
            hmc_seed=np.array(hmc_seed),
            algo=np.array('adaptive_hmc_lowhet_sampled'),
            client_mu=as_numpy(client_mu),
            client_var=as_numpy(client_var),
            target_mu=as_numpy(target_mu),
            target_var=as_numpy(target_var),
            target_var_summary=np.array(target_var_summary),
            client_mean_rms_dev=np.array(mean_dev),
            client_var_rms_dev=np.array(var_dev),
            final_sample_mu=as_numpy(sample_mu),
            final_sample_var=as_numpy(sample_var),
            final_w2=np.array(w2_sq.item()),
            final_mean_term=np.array(mean_term.item()),
            final_var_term=np.array(var_term.item()),
            trace_rounds=np.array(trace_rounds),
            trace_mu_avg=np.array(trace_mu_avg),
            trace_var_avg=np.array(trace_var_avg),
            trace_w2=np.array(trace_w2),
            trace_mean_term=np.array(trace_mean_term),
            trace_var_term=np.array(trace_var_term),
            first_cross=np.array(first_cross if first_cross is not None else -1),
            final_round=np.array(trace_rounds[-1] if trace_rounds else 0),
            stopped_early=np.array(int(tag == 'stopped')),
        )
        os.replace(tmp, checkpoint_path)

    t_start = time.time()
    print(f'[lowhet d={d}] M={M}  K={K}  eta={eta:.6g}  T={T}  '
          f'max_rounds={max_rounds}  variance_mode={variance_mode}',
          flush=True)
    print(f'[lowhet d={d}] samples/client={samples_per_client}  '
          f'parent=N({parent_mean}, {parent_var})  stats_seed={stats_seed}  '
          f'hmc_seed={hmc_seed}', flush=True)
    print(f'[lowhet d={d}] target_var_avg={target_var_summary:.6g}  '
          f'client_mean_rms_dev={mean_dev:.6g}  client_var_rms_dev={var_dev:.6g}',
          flush=True)

    for r in range(1, max_rounds + 1):
        X_sync_b = X0.unsqueeze(1)
        g_at_sync = (X_sync_b - mu_b) * inv_var_b
        g_global_sync = g_at_sync.mean(dim=1, keepdim=True)

        q = X_sync_b.expand(M, N, d).contiguous()
        for _ in range(T):
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype, generator=hmc_gen)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                g = (q - X_sync_b) * inv_var_b + g_global_sync
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g2 = (q_new - X_sync_b) * inv_var_b + g_global_sync
                p = p - 0.5 * eta * (g + g2)
                q = q_new
        X0 = q.mean(dim=1)

        if r % check_every == 0 or r == max_rounds:
            w2_sq, mean_term, var_term, sample_mu, _sample_var, trace_var = (
                compute_w2_to_sampled_target(X0, target_mu, target_var)
            )
            w2_value = w2_sq.item()
            trace_rounds.append(r)
            trace_mu_avg.append(sample_mu.mean().item())
            trace_var_avg.append(trace_var.item())
            trace_w2.append(w2_value)
            trace_mean_term.append(mean_term.item())
            trace_var_term.append(var_term.item())

            if w2_value < threshold:
                if first_cross is None:
                    first_cross = r
                    print(f'[lowhet d={d}] FIRST cross at round {r}: '
                          f'W2^2={w2_value:.4f}', flush=True)
                streak += 1
            else:
                streak = 0

            if r % verbose_every == 0:
                elapsed = time.time() - t_start
                print(f'[lowhet d={d}] round={r}  mu_avg={trace_mu_avg[-1]:.4f}  '
                      f'var_avg={trace_var_avg[-1]:.4f}  W2^2={w2_value:.4f}  '
                      f'mean_term={mean_term.item():.4f}  var_term={var_term.item():.4f}  '
                      f'streak={streak}  elapsed={elapsed:.1f}s',
                      flush=True)

            if r % checkpoint_every == 0:
                save()

            if streak >= consecutive_below:
                elapsed = time.time() - t_start
                print(f'[lowhet d={d}] STOP round {r}: W2^2 < {threshold} for '
                      f'{consecutive_below} consecutive checks. '
                      f'First cross = {first_cross}. Elapsed={elapsed:.1f}s',
                      flush=True)
                save('stopped')
                return

    save()
    print(f'[lowhet d={d}] Reached max_rounds={max_rounds} without stopping. '
          f'first_cross={first_cross}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--K_at_100', type=float, default=50.0)
    parser.add_argument('--K_exp', type=float, default=0.5)
    parser.add_argument('--eta_at_100', type=float, default=0.006)
    parser.add_argument('--eta_exp', type=float, default=-0.5)
    parser.add_argument('--M_cap', type=int, default=0)
    parser.add_argument('--mem_budget_gb', type=float, default=28.0)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--max_rounds', type=int, default=1000)
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--consecutive_below', type=int, default=20)
    parser.add_argument('--samples_per_client', type=int, default=100000)
    parser.add_argument('--parent_mean', type=float, default=16.2)
    parser.add_argument('--parent_var', type=float, default=1.6)
    parser.add_argument('--stats_batch_size', type=int, default=100000)
    parser.add_argument('--variance_mode', choices=('scalar', 'diagonal'), default='scalar')
    parser.add_argument('--stats_seed', type=int, default=8848)
    parser.add_argument('--seed', type=int, default=8848,
                        help='HMC chain/momentum seed')
    parser.add_argument('--verbose_every', type=int, default=50)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    K = max(1, round(args.K_at_100 * (args.d / 100.0) ** args.K_exp))
    eta = args.eta_at_100 * (args.d / 100.0) ** args.eta_exp
    M_full = max(100 * args.d * (args.d - 1), 1)
    M_budget = int(args.mem_budget_gb * 1e9 / (7 * args.N * args.d * 4))
    if args.M_cap > 0:
        M = min(M_full, args.M_cap)
    else:
        M = min(M_full, M_budget)

    adaptive_hmc_lowhet(
        d=args.d,
        K=K,
        eta=eta,
        M=M,
        N=args.N,
        T=args.T,
        threshold=args.threshold,
        max_rounds=args.max_rounds,
        samples_per_client=args.samples_per_client,
        parent_mean=args.parent_mean,
        parent_var=args.parent_var,
        stats_batch_size=args.stats_batch_size,
        variance_mode=args.variance_mode,
        stats_seed=args.stats_seed,
        hmc_seed=args.seed,
        check_every=args.check_every,
        consecutive_below=args.consecutive_below,
        device=device,
        verbose_every=args.verbose_every,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    print(f'[lowhet d={args.d}] saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
