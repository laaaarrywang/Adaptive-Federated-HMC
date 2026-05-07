from __future__ import annotations

import time

import torch


def build_client_params(N: int, d: int, device, dtype=torch.float32):
    assert N % 2 == 0, 'N must be even'
    half = N // 2
    mu_clients = torch.empty((d, N), device=device, dtype=dtype)
    mu_clients[:, :half] = 20.0
    mu_clients[:, half:] = 1.0
    var_clients = torch.empty(N, device=device, dtype=dtype)
    var_clients[:half] = 1.0
    var_clients[half:] = 4.0
    return mu_clients, var_clients


def fa_hmc_gaussian(mu_clients, var_clients, eta, R, K, T, M, init='normal'):
    d, N = mu_clients.shape
    device, dtype = mu_clients.device, mu_clients.dtype
    mu_b = mu_clients.T.unsqueeze(0).contiguous()
    inv_var_b = (1.0 / var_clients).view(1, N, 1).contiguous()

    if init == 'zero':
        X0 = torch.zeros((M, d), device=device, dtype=dtype)
    else:
        X0 = torch.randn((M, d), device=device, dtype=dtype)

    for _ in range(R):
        q = X0.unsqueeze(1).expand(M, N, d).contiguous()
        for _ in range(T):
            p_shared = torch.randn((M, 1, d), device=device, dtype=dtype)
            p = p_shared.expand(M, N, d).contiguous()
            for _ in range(K):
                g = (q - mu_b) * inv_var_b
                q_new = q + eta * p - 0.5 * (eta * eta) * g
                g2 = (q_new - mu_b) * inv_var_b
                p = p - 0.5 * eta * (g + g2)
                q = q_new
        X0 = q.mean(dim=1)
    return X0


def bench(d, M, N=10, K=5, T=10, warmup_rounds=1, measure_rounds=3,
          dtype=torch.float32, device='cuda'):
    mu, var = build_client_params(N, d, device, dtype)
    eta = 0.02 / (d ** 0.25)
    torch.manual_seed(0)

    fa_hmc_gaussian(mu, var, eta, R=warmup_rounds, K=K, T=T, M=M)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    fa_hmc_gaussian(mu, var, eta, R=measure_rounds, K=K, T=T, M=M)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    per_local = total / (measure_rounds * T)
    return per_local, total


def main():
    device = 'cuda'
    print(f'PyTorch {torch.__version__}  device={torch.cuda.get_device_name(0)}')
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU memory: {total_mem_gb:.1f} GB')
    print(f'N=10  K=5  T=10   fp32   (time per one local step = K=5 leapfrog iters)')
    print(f'{"d":>6} {"M":>8} {"ms/local-step":>16} {"ns/chain/local":>18} {"total(s)":>10}', flush=True)

    N, K, T = 10, 5, 10
    configs = [
        (100,  [1000, 10000, 100000, 500000, 1000000]),
        (500,  [1000, 10000, 50000, 100000, 200000]),
        (1000, [1000, 10000, 30000, 50000, 100000]),
    ]
    for d, Ms in configs:
        for M in Ms:
            try:
                per_local, total = bench(d, M, N=N, K=K, T=T)
            except torch.cuda.OutOfMemoryError:
                print(f'{d:6d} {M:8d}  OOM', flush=True)
                torch.cuda.empty_cache()
                continue
            per_chain_ns = per_local / M * 1e9
            print(f'{d:6d} {M:8d} {per_local*1e3:16.3f} {per_chain_ns:18.2f} {total:10.2f}', flush=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
