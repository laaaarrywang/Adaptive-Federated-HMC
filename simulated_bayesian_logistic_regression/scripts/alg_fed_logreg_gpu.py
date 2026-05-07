"""GPU implementations of FA-HMC for Bayesian logistic regression.

Three variants:
  - fa_hmc_logreg            — vanilla FA-HMC (matches alg_fed.py math)
  - fa_hmc_logreg_adaptive   — SCAFFOLD-corrected (matches alg_fed_adaptive.py math)
  - both accept `noise_sigma > 0` to inject N(0, noise_sigma^2) per gradient
    coordinate per client per chain (paper's stochastic-gradient simulation).

All three batch over M independent chains. Data layout:
  - U_stacked: shape (N, d, n_c) — column slices of the original (d, N*n_c) data
  - q, p: shape (M, N, d) during leapfrog
  - X0: shape (M, d) is the synced parameter at round end

Conventions mirror the Gaussian and image-data adaptive code:
  - Adaptive: T = 1 momentum draw / round, one gradient all-reduce at start,
    one parameter all-reduce at end.
  - Vanilla:  T momentum draws / round, parameter all-reduce only at round end
    (no mid-trajectory sync).
  - Shared momentum across clients within a chain (rho=1).
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def stack_clients(U: torch.Tensor, N: int) -> torch.Tensor:
    """Reshape U of shape (d, N*n_c) into (N, d, n_c)."""
    d, total = U.shape
    if total % N != 0:
        raise ValueError(f'U has {total} columns, not divisible by N={N}')
    n_c = total // N
    # reshape into (d, N, n_c) then permute to (N, d, n_c)
    return U.reshape(d, N, n_c).permute(1, 0, 2).contiguous()


def grad_client_batched(
    q: torch.Tensor,           # (M, N, d)
    U_stacked: torch.Tensor,   # (N, d, n_c)
    N: int,
    noise_sigma: float = 0.0,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-client negative log-posterior gradient: prior + N * likelihood.

    Returns (M, N, d). Matches `grad_client` in alg_fed.py / alg_fed_adaptive.py:
        g_c(x) = x + N * U_c @ sigmoid(U_c.T @ x)

    With `noise_sigma > 0`, adds independent N(0, noise_sigma^2) noise to every
    coordinate (paper's stochastic-gradient simulation, panel b/d).
    """
    # logits[m, c, i] = sum_k U_stacked[c, k, i] * q[m, c, k]
    logits = torch.einsum('cki,mck->mci', U_stacked, q)
    s = torch.sigmoid(logits)
    # g_lik[m, c, k] = sum_i U_stacked[c, k, i] * s[m, c, i]
    g_lik = torch.einsum('cki,mci->mck', U_stacked, s)
    g = q + N * g_lik
    if noise_sigma > 0.0:
        g = g + noise_sigma * torch.randn(
            g.shape, device=g.device, dtype=g.dtype, generator=gen
        )
    return g


def fa_hmc_logreg(
    U_stacked: torch.Tensor,   # (N, d, n_c)
    eta: float,
    R: int,                    # outer iterations / communication rounds
    K: int,                    # leapfrog_step (paper) - leapfrog per trajectory
    T: int,                    # local_step  (paper) - sync cadence inside trajectory
    M: int,
    *,
    noise_sigma: float = 0.0,
    init: str = 'normal',
    seed: int = 8848,
    record_every: int = 0,
    record_fn: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Vanilla FA-HMC matching alg_fed_noisy.py structure.

    Per outer iter: T trajectories from X0, each with K leapfrog and a
    mid-trajectory all-reduce of (q, p) every T grad evaluations. X0 is
    updated from the LAST trajectory's q at end of outer iter; sample
    callback fires once per outer iter (post-X0-update).

    Returns final X0 of shape (M, d).
    """
    N, d, n_c = U_stacked.shape
    device, dtype = U_stacked.device, U_stacked.dtype
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    if init == 'zero':
        X0 = torch.zeros((M, d), device=device, dtype=dtype)
    else:
        X0 = torch.randn((M, d), device=device, dtype=dtype, generator=gen)

    if record_fn is not None and record_every > 0:
        record_fn(0, X0)

    grad_count = 0  # global counter for sync cadence (matches alg_fed_noisy.py)
    for r in range(1, R + 1):
        for _ in range(T):
            q = X0.unsqueeze(1).expand(M, N, d).contiguous()
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype, generator=gen)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                grad_count += 1
                g = grad_client_batched(q, U_stacked, N, noise_sigma, gen)
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g2 = grad_client_batched(q_new, U_stacked, N, noise_sigma, gen)
                p = p - 0.5 * eta * (g + g2)
                q = q_new
                if grad_count % T == 0:
                    q_mean = q.mean(dim=1, keepdim=True)
                    p_mean = p.mean(dim=1, keepdim=True)
                    q = q_mean.expand(M, N, d).contiguous()
                    p = p_mean.expand(M, N, d).contiguous()
        X0 = q.mean(dim=1)  # X0 update from last trajectory's q
        if record_fn is not None and record_every > 0 and r % record_every == 0:
            record_fn(r, X0)

    return X0


def fa_hmc_logreg_adaptive(
    U_stacked: torch.Tensor,   # (N, d, n_c)
    eta: float,
    R: int,
    K: int,
    T: int,
    M: int,
    *,
    noise_sigma: float = 0.0,
    init: str = 'normal',
    seed: int = 8848,
    record_every: int = 0,
    record_fn: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Adaptive (SCAFFOLD) FA-HMC.

    At round start: snapshot g_c(X_sync) per client and the global average
    g_global(X_sync). During the round, leapfrog uses the corrected gradient
        g_corrected(q) = g_c(q) - g_c(X_sync) + g_global(X_sync)
    so the heterogeneity bias is removed at q = X_sync. T momentum draws per
    round; sync params at round end.

    With noise_sigma > 0, the snapshot g_c(X_sync) is computed *without* noise
    (the control variate must be a deterministic function of X_sync), but the
    in-trajectory g_c(q), g_c(q_new) evaluations are noisy.
    """
    N, d, n_c = U_stacked.shape
    device, dtype = U_stacked.device, U_stacked.dtype
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    if init == 'zero':
        X0 = torch.zeros((M, d), device=device, dtype=dtype)
    else:
        X0 = torch.randn((M, d), device=device, dtype=dtype, generator=gen)

    if record_fn is not None and record_every > 0:
        record_fn(0, X0)

    for r in range(1, R + 1):
        # Snapshot at X_sync = X0. Use *exact* gradient for the control variate.
        X_sync_b = X0.unsqueeze(1).expand(M, N, d).contiguous()
        g_c_sync = grad_client_batched(X_sync_b, U_stacked, N, noise_sigma=0.0)
        g_global_sync = g_c_sync.mean(dim=1, keepdim=True)  # (M, 1, d)

        q = X0.unsqueeze(1).expand(M, N, d).contiguous()
        for _ in range(T):
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype, generator=gen)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                g_c = grad_client_batched(q, U_stacked, N, noise_sigma, gen)
                g = g_c - g_c_sync + g_global_sync
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g_c_new = grad_client_batched(q_new, U_stacked, N, noise_sigma, gen)
                g2 = g_c_new - g_c_sync + g_global_sync
                p = p - 0.5 * eta * (g + g2)
                q = q_new
        X0 = q.mean(dim=1)
        if record_fn is not None and record_every > 0 and r % record_every == 0:
            record_fn(r, X0)

    return X0
