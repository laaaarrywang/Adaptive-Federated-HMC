"""GPU driver for one logistic-regression (setting x algorithm x eta x K) cell.

Runs M chains for R rounds, collects samples every C rounds, saves
the resulting (M*S, d) sample matrix to npz alongside metadata. Used by the
PBS sweep script which loops over the cells.

Algorithms (matched-compute convention, K_adaptive = T * K_FA-HMC):
  fa_ld     : K=1, T=10, no gradient correction
  fa_hmc    : K=K_p(eta), T=10, no gradient correction
  adaptive  : K=10*K_p(eta), T=1, gradient-corrected
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent
sys.path.insert(0, str(SIM_DIR))

from alg_fed_logreg_gpu import (  # noqa: E402
    fa_hmc_logreg, fa_hmc_logreg_adaptive, stack_clients,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=['a', 'd'], required=True,
                        help='experiment setting label: a=d1000 exact, d=d10 stochastic')
    parser.add_argument('--algo', choices=['fa_ld', 'fa_hmc', 'adaptive'], required=True)
    parser.add_argument('--data', required=True, help='path to U.mat')
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--K', type=int, required=True,
                        help='leapfrog count for THIS algorithm (already includes the '
                             'matched-compute factor for adaptive)')
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--R', type=int, default=0,
                        help='outer iterations (overrides --iter_total if > 0)')
    parser.add_argument('--iter_total', type=int, default=0,
                        help='target total leapfrog count; computes R = iter_total/(K*T) '
                             'so different (K,T) settings get matched-leapfrog comparisons')
    parser.add_argument('--M', type=int, default=20, help='number of chains')
    parser.add_argument('--N', type=int, default=20, help='federated clients')
    parser.add_argument('--noise_sigma', type=float, default=0.0,
                        help='SG noise std (10 for panel d, 0 for panel a)')
    parser.add_argument('--n_samples_target', type=int, default=1000,
                        help='samples to collect per chain (so collect every R/this)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=8848)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    dtype = torch.float32

    U_np = np.asarray(loadmat(args.data)['U'], dtype=np.float32)
    d, n = U_np.shape
    assert n % args.N == 0, f'n={n} not divisible by N={args.N}'
    U = torch.from_numpy(U_np).to(device=device, dtype=dtype)
    U_stacked = stack_clients(U, args.N)  # (N, d, n_c)

    if args.R > 0:
        R = args.R
    elif args.iter_total > 0:
        R = max(1, args.iter_total // (args.K * args.T))
    else:
        raise ValueError('Pass either --R or --iter_total')
    record_every = max(1, R // args.n_samples_target)
    samples_buffer: list[np.ndarray] = []  # each entry (M, d)

    def record_fn(r: int, X0: torch.Tensor):
        # collect into CPU buffer to keep GPU memory bounded
        samples_buffer.append(X0.detach().cpu().numpy().copy())

    print(f'[{args.algo} panel={args.panel} eta={args.eta:.5g} K={args.K} '
          f'T={args.T} R={R} M={args.M} d={d} sigma={args.noise_sigma}] '
          f'record_every={record_every}  total_leapfrog={R*args.K*args.T}',
          flush=True)

    t0 = time.time()
    if args.algo == 'adaptive':
        runner = fa_hmc_logreg_adaptive
    else:
        runner = fa_hmc_logreg

    runner(
        U_stacked, eta=float(args.eta), R=int(R),
        K=int(args.K), T=int(args.T), M=int(args.M),
        noise_sigma=float(args.noise_sigma),
        seed=int(args.seed),
        record_every=record_every, record_fn=record_fn,
    )
    elapsed = time.time() - t0

    if not samples_buffer:
        raise RuntimeError('no samples were recorded; check record_every vs R')
    # samples_buffer[i] is (M, d); stack -> (S, M, d)
    samples = np.stack(samples_buffer, axis=0)  # (S, M, d)
    S = samples.shape[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        panel=np.array(args.panel),
        algo=np.array(args.algo),
        eta=np.array(args.eta), K=np.array(args.K), T=np.array(args.T),
        R=np.array(args.R), M=np.array(args.M), N=np.array(args.N),
        d=np.array(d), n=np.array(n),
        noise_sigma=np.array(args.noise_sigma),
        record_every=np.array(record_every),
        samples=samples.astype(np.float32),  # (S, M, d)
        elapsed_seconds=np.array(elapsed),
        seed=np.array(args.seed),
    )
    print(f'[{args.algo}] done in {elapsed:.1f}s  samples=(S={S}, M={args.M}, d={d})  '
          f'-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
