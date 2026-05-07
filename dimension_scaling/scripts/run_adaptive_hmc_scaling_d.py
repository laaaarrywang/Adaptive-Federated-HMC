"""Adaptive (SCAFFOLD) FA-HMC driver with d-dependent K and eta.

Defaults follow the new theoretical scaling:
    K(d)   = round(K_at_100 * (d/100)^K_exp)        # K_exp = 1/3
    eta(d) = eta_at_100 * (d/100)^eta_exp           # eta_exp = -1/2
    M(d)   = 100 * d * (d-1)

Anchors at d=100 use K=50, eta=0.006.
At other d the new scaling differs, isolating the effect of the exponents.

Stopping rule: W2^2 against N(16.2*1_d, 1.6*I_d) below threshold for
`consecutive_below` consecutive rounds.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from bench_local_step_gpu import build_client_params


def adaptive_hmc(
    d: int,
    K: int,
    eta: float,
    M: int,
    N: int,
    T: int,
    threshold: float,
    max_rounds: int,
    check_every: int = 1,
    consecutive_below: int = 20,
    target_mean: float = 16.2,
    target_var: float = 1.6,
    device: torch.device = torch.device('cuda'),
    dtype=torch.float32,
    seed: int = 8848,
    verbose_every: int = 100,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 200,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mu_c, var_c = build_client_params(N, d, device, dtype)
    mu_b = mu_c.T.unsqueeze(0).contiguous()
    inv_var_b = (1.0 / var_c).view(1, N, 1).contiguous()

    X0 = torch.randn((M, d), device=device, dtype=dtype, generator=gen)

    trace_rounds: list[int] = []
    trace_mu: list[float] = []
    trace_var: list[float] = []
    trace_w2: list[float] = []
    first_cross = None
    streak = 0
    target_std = math.sqrt(target_var)

    def save(tag: str = ''):
        if checkpoint_path is None:
            return
        tmp = checkpoint_path + '.tmp.npz'
        np.savez(
            tmp[:-4],
            d=np.array(d), M=np.array(M), eta=np.array(eta),
            K=np.array(K), T=np.array(T), threshold=np.array(threshold),
            target_mean=np.array(target_mean), target_var=np.array(target_var),
            algo=np.array('adaptive_hmc_scaling'),
            trace_rounds=np.array(trace_rounds),
            trace_mu=np.array(trace_mu),
            trace_var=np.array(trace_var),
            trace_w2=np.array(trace_w2),
            first_cross=np.array(first_cross if first_cross is not None else -1),
            final_round=np.array(trace_rounds[-1] if trace_rounds else 0),
            stopped_early=np.array(int(tag == 'stopped')),
        )
        os.replace(tmp, checkpoint_path)

    t_start = time.time()
    print(f'[d={d}] M={M}  K={K}  eta={eta:.6g}  T={T}  '
          f'max_rounds={max_rounds}', flush=True)

    for r in range(1, max_rounds + 1):
        X_sync_b = X0.unsqueeze(1)  # (M, 1, d)
        g_at_sync = (X_sync_b - mu_b) * inv_var_b
        g_global_sync = g_at_sync.mean(dim=1, keepdim=True)

        q = X_sync_b.expand(M, N, d).contiguous()
        for _ in range(T):
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype, generator=gen)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                g = (q - X_sync_b) * inv_var_b + g_global_sync
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g2 = (q_new - X_sync_b) * inv_var_b + g_global_sync
                p = p - 0.5 * eta * (g + g2)
                q = q_new
        X0 = q.mean(dim=1)

        if r % check_every == 0 or r == max_rounds:
            mu_hat = X0.mean().item()
            var_hat = X0.var(dim=0, unbiased=False).mean().item()
            w2_sq = (d * (mu_hat - target_mean) ** 2
                     + d * (math.sqrt(max(var_hat, 0.0)) - target_std) ** 2)

            trace_rounds.append(r)
            trace_mu.append(mu_hat)
            trace_var.append(var_hat)
            trace_w2.append(w2_sq)

            if w2_sq < threshold:
                if first_cross is None:
                    first_cross = r
                    print(f'[d={d}] FIRST cross at round {r}: W2^2={w2_sq:.4f}',
                          flush=True)
                streak += 1
            else:
                streak = 0

            if r % verbose_every == 0:
                elapsed = time.time() - t_start
                print(f'[d={d}] round={r}  mu={mu_hat:.4f}  var={var_hat:.4f}  '
                      f'W2^2={w2_sq:.4f}  streak={streak}  elapsed={elapsed:.1f}s',
                      flush=True)

            if r % checkpoint_every == 0:
                save()

            if streak >= consecutive_below:
                elapsed = time.time() - t_start
                print(f'[d={d}] STOP round {r}: W2^2 < {threshold} for '
                      f'{consecutive_below} consecutive checks. '
                      f'First cross = {first_cross}. Elapsed={elapsed:.1f}s',
                      flush=True)
                save('stopped')
                return

    save()
    print(f'[d={d}] Reached max_rounds={max_rounds} without stopping. '
          f'first_cross={first_cross}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--K_at_100', type=float, default=50.0)
    parser.add_argument('--K_exp', type=float, default=1.0 / 3.0)
    parser.add_argument('--eta_at_100', type=float, default=0.006)
    parser.add_argument('--eta_exp', type=float, default=-0.5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--max_rounds', type=int, default=None,
                        help='default = max(500, 10 * round(c * d^(1/3))) with c=63')
    parser.add_argument('--check_every', type=int, default=1)
    parser.add_argument('--consecutive_below', type=int, default=20)
    parser.add_argument('--verbose_every', type=int, default=100)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=8848)
    parser.add_argument('--checkpoint_every', type=int, default=200)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    K = max(1, round(args.K_at_100 * (args.d / 100.0) ** args.K_exp))
    eta = args.eta_at_100 * (args.d / 100.0) ** args.eta_exp
    M = max(100 * args.d * (args.d - 1), 1)

    if args.max_rounds is None:
        # Generous slack: 10x predicted; floor 500
        predicted = round(63.0 * args.d ** (1.0 / 3.0))
        args.max_rounds = max(500, 10 * predicted)

    adaptive_hmc(
        d=args.d, K=K, eta=eta, M=M, N=args.N, T=args.T,
        threshold=args.threshold, max_rounds=args.max_rounds,
        check_every=args.check_every,
        consecutive_below=args.consecutive_below,
        device=device, seed=args.seed,
        verbose_every=args.verbose_every,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    print(f'[d={args.d}] saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
