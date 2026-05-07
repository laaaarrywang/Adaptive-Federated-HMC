"""Vanilla FA-HMC driver matched to the adaptive (SCAFFOLD) experiment.

Same eta(d), same total leapfrog per round, same R_target, same seed —
only difference is no SCAFFOLD gradient correction and momentum refresh
T times per round (vs once for adaptive).

K_v = max(1, round(K_a / T_vanilla)) where K_a is the adaptive K(d).
T_vanilla fixed (default 10).
Runs exactly --R_target rounds, records W2^2 every round.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from bench_local_step_gpu import build_client_params


def vanilla_fa_hmc(
    d: int,
    K: int,
    T: int,
    eta: float,
    M: int,
    N: int,
    R_target: int,
    target_mean: float = 16.2,
    target_var: float = 1.6,
    device: torch.device = torch.device('cuda'),
    dtype=torch.float32,
    seed: int = 8848,
    verbose_every: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 100,
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
    target_std = math.sqrt(target_var)

    def save():
        if checkpoint_path is None:
            return
        tmp = checkpoint_path + '.tmp.npz'
        np.savez(
            tmp[:-4],
            d=np.array(d), M=np.array(M), eta=np.array(eta),
            K=np.array(K), T=np.array(T), R_target=np.array(R_target),
            target_mean=np.array(target_mean), target_var=np.array(target_var),
            algo=np.array('vanilla_fa_hmc'),
            trace_rounds=np.array(trace_rounds),
            trace_mu=np.array(trace_mu),
            trace_var=np.array(trace_var),
            trace_w2=np.array(trace_w2),
            final_round=np.array(trace_rounds[-1] if trace_rounds else 0),
        )
        os.replace(tmp, checkpoint_path)

    t_start = time.time()
    print(f'[vanilla d={d}] M={M}  K={K}  T={T}  eta={eta:.6g}  '
          f'R_target={R_target}', flush=True)

    for r in range(1, R_target + 1):
        q = X0.unsqueeze(1).expand(M, N, d).contiguous()
        for _ in range(T):
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype, generator=gen)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                g = (q - mu_b) * inv_var_b
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g2 = (q_new - mu_b) * inv_var_b
                p = p - 0.5 * eta * (g + g2)
                q = q_new
        X0 = q.mean(dim=1)

        mu_hat = X0.mean().item()
        var_hat = X0.var(dim=0, unbiased=False).mean().item()
        w2_sq = (d * (mu_hat - target_mean) ** 2
                 + d * (math.sqrt(max(var_hat, 0.0)) - target_std) ** 2)
        trace_rounds.append(r)
        trace_mu.append(mu_hat)
        trace_var.append(var_hat)
        trace_w2.append(w2_sq)

        if r % verbose_every == 0:
            elapsed = time.time() - t_start
            print(f'[vanilla d={d}] round={r}  mu={mu_hat:.4f}  var={var_hat:.4f}  '
                  f'W2^2={w2_sq:.4f}  elapsed={elapsed:.1f}s', flush=True)

        if r % checkpoint_every == 0:
            save()

    save()
    elapsed = time.time() - t_start
    print(f'[vanilla d={d}] DONE R={R_target}: final W2^2={trace_w2[-1]:.4f}  '
          f'elapsed={elapsed:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--T_vanilla', type=int, default=10)
    parser.add_argument('--K_at_100', type=float, default=50.0)
    parser.add_argument('--K_exp', type=float, default=1.0 / 3.0)
    parser.add_argument('--eta_at_100', type=float, default=0.006)
    parser.add_argument('--eta_exp', type=float, default=-0.5)
    parser.add_argument('--R_target', type=int, required=True,
                        help='exact number of rounds to run (no stopping)')
    parser.add_argument('--verbose_every', type=int, default=50)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=8848)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    K_a = max(1, round(args.K_at_100 * (args.d / 100.0) ** args.K_exp))
    K_v = max(1, round(K_a / args.T_vanilla))
    eta = args.eta_at_100 * (args.d / 100.0) ** args.eta_exp
    M = max(100 * args.d * (args.d - 1), 1)

    print(f'[vanilla d={args.d}] K_adaptive={K_a}  K_vanilla={K_v}  '
          f'T={args.T_vanilla}  total_leapfrog/round={K_v*args.T_vanilla}', flush=True)

    vanilla_fa_hmc(
        d=args.d, K=K_v, T=args.T_vanilla, eta=eta, M=M, N=args.N,
        R_target=args.R_target,
        device=device, seed=args.seed,
        verbose_every=args.verbose_every,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    print(f'[vanilla d={args.d}] saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
