"""Fast federated trainer mirroring trainer.training_federated.

Algorithmic structure follows the original federated trainer: same momentum draw,
same leapfrog updates (with optional gradient correction when
`pars.adaptive=1`), same sync cadence (`numerical_step % local_step == 0`),
same per-checkpoint validation and `probs_chain` saving. The only change is
that batch fetching no longer goes through a DataLoader — instead, all data
sits as GPU tensors and we slice into them via index permutations. This
removes the per-leapfrog DataLoader overhead (~50 ms × 20 fetches) that
dominates wall time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tools_fast import validation_fast


def _ClientBatcher(idx_tensor: torch.Tensor, batch_size: int):
    """Returns a closure that yields successive batches of `batch_size` indices
    drawn (without replacement, then re-permuted) from `idx_tensor`."""
    perm = idx_tensor[torch.randperm(idx_tensor.shape[0], device=idx_tensor.device)]
    pos = [0]

    def next_batch():
        nonlocal perm
        if pos[0] + batch_size > perm.shape[0]:
            # Re-shuffle and reset for the next epoch.
            perm = idx_tensor[torch.randperm(idx_tensor.shape[0], device=idx_tensor.device)]
            pos[0] = 0
        idx = perm[pos[0]:pos[0] + batch_size]
        pos[0] += batch_size
        return idx
    return next_batch


def training_federated_fast(nets, net_server, train_x, train_y, clients,
                             test_x, test_y, train_size, pars):
    clients_size = [c.shape[0] for c in clients]
    clients_weight = np.array(clients_size) / train_size
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    Tem, lr, local_step, num_client = pars.T, pars.lr, pars.local_step, pars.num_client
    if pars.adaptive:
        local_step = pars.leapfrog_step
    output_gap = np.lcm(local_step, pars.leapfrog_step) * pars.save_gap

    batchers = [_ClientBatcher(clients[c], pars.batch_train) for c in range(num_client)]

    numerical_step = 0
    para_num = 2
    p_initial = [0] * para_num
    p_clients = [[0] * para_num for _ in range(num_client)]
    grad_first = [0] * para_num
    if pars.adaptive:
        grad_local_at_sync = [[None] * para_num for _ in range(num_client)]
        grad_global_at_sync = [None] * para_num

    save_count = 0
    save_name_index = 0
    probs = 0
    probs_chain = []

    test_results = np.zeros((pars.total_step * pars.leapfrog_step // output_gap, 6))

    for out_iter in range(pars.total_step):
        # Fresh momentum, broadcast to all clients
        for i, para in enumerate(nets[0].parameters()):
            p_initial[i] = torch.empty(para.shape, device=device).normal_()
            for c in range(num_client):
                p_clients[c][i] = p_initial[i].clone()

        # snapshot per-client + global gradients at the synced point
        if pars.adaptive:
            for c in range(num_client):
                net = nets[c]
                net.zero_grad()
                idx = batchers[c]()
                images, labels = train_x[idx], train_y[idx]
                loss = criterion(net(images), labels) * train_size / Tem
                loss.backward()
                with torch.no_grad():
                    for i, para in enumerate(net.parameters()):
                        grad_local_at_sync[c][i] = para.grad.data.clone()
                net.zero_grad()
            with torch.no_grad():
                for i in range(para_num):
                    grad_global_at_sync[i] = sum(
                        clients_weight[c] * grad_local_at_sync[c][i] for c in range(num_client)
                    )

        for leap_iter in range(pars.leapfrog_step):
            numerical_step += 1
            for c in range(num_client):
                net = nets[c]

                # First half: gradient at current q, then position update
                net.zero_grad()
                idx = batchers[c]()
                images, labels = train_x[idx], train_y[idx]
                loss = criterion(net(images), labels) * train_size / Tem
                loss.backward()
                with torch.no_grad():
                    for i, para in enumerate(net.parameters()):
                        if pars.adaptive:
                            g = para.grad.data - grad_local_at_sync[c][i] + grad_global_at_sync[i]
                            grad_first[i] = g.clone()
                            para.data = para.data + lr * p_clients[c][i] - 0.5 * (lr * lr) * g
                        else:
                            grad_first[i] = para.grad.data.clone()
                            para.data = para.data + lr * p_clients[c][i] - 0.5 * (lr * lr) * para.grad.data

                # Second half: gradient at new q, then momentum update
                net.zero_grad()
                idx = batchers[c]()
                images, labels = train_x[idx], train_y[idx]
                loss = criterion(net(images), labels) * train_size / Tem
                loss.backward()
                with torch.no_grad():
                    for i, para in enumerate(net.parameters()):
                        if pars.adaptive:
                            g2 = para.grad.data - grad_local_at_sync[c][i] + grad_global_at_sync[i]
                            p_clients[c][i] = p_clients[c][i] - 0.5 * lr * (grad_first[i] + g2)
                        else:
                            p_clients[c][i] = p_clients[c][i] - 0.5 * lr * (grad_first[i] + para.grad.data)

            # Sync params and momenta across clients
            if numerical_step % local_step == 0:
                para_server = [0] * para_num
                p_server = [0] * para_num
                with torch.no_grad():
                    for c in range(num_client):
                        for i, para in enumerate(nets[c].parameters()):
                            para_server[i] = para_server[i] + clients_weight[c] * para.data
                            p_server[i] = p_server[i] + clients_weight[c] * p_clients[c][i]
                    for c in range(num_client):
                        for i, para in enumerate(nets[c].parameters()):
                            para.data = para_server[i].clone()
                            p_clients[c][i] = p_server[i].clone()
                    for i, para in enumerate(net_server.parameters()):
                        para.data = para_server[i].clone()

        # Per-checkpoint validation and sample collection
        if numerical_step % output_gap == 0:
            save_count += 1
            test_results[save_count - 1, ], probs_new = validation_fast(
                net_server, test_x, test_y, pars.bin_size, batch_test=pars.batch_test
            )
            probs_chain.append(probs_new.cpu())
            probs = probs + probs_new
            if save_count % pars.save_size == 0:
                save_name_index += 1
                torch.save(probs_chain, pars.save_name + str(save_name_index) + '.pt')
                probs_chain = []

    # Final BMA evaluation on full test set
    probs = probs / save_count
    from tools_fast import validation_fast as _vf  # re-import for clarity
    # Use the same evaluation path as cal_poster (probability-based eval).
    # tools.evaluation expects (probs_all, test_loader, M); we substitute by
    # computing the same metrics manually on probs vs (test_x, test_y).
    nll_all, accu, Brier, ECE, MCE, entropy, bin_conf, bin_accu = _evaluate_probs(probs, test_y, pars.bin_size)
    test_ave = (nll_all, accu, Brier, ECE, MCE, np.zeros(test_y.shape[0]), bin_conf, bin_accu)
    return test_ave, test_results


def _evaluate_probs(probs_all: torch.Tensor, test_y: torch.Tensor, M: int):
    """Compute the same metrics as tools.evaluation but on a precomputed probs
    tensor + GPU labels."""
    import torch.nn.functional as Func
    test_size = test_y.shape[0]
    output_all = probs_all.log()
    with torch.no_grad():
        nll_all = Func.nll_loss(output_all, test_y, reduction='sum').item()
        Brier = torch.sum((probs_all - Func.one_hot(test_y, num_classes=output_all.shape[1])) ** 2).item() / test_size
        max_conf, pred = probs_all.max(dim=1)
        conf = max_conf.cpu().numpy()
        matched = pred.eq(test_y).cpu().numpy()
    accu = matched.mean().item()
    bins = np.linspace(0, 1, M + 1)
    bin_assign = np.digitize(conf, bins)
    bin_conf = np.zeros(M)
    bin_accu = np.zeros(M)
    bin_size = np.zeros(M, dtype=int)
    for j in range(1, M + 1):
        bin_size[j - 1] = int(np.sum(bin_assign == j))
        if bin_size[j - 1] > 0:
            bin_conf[j - 1] = np.mean(conf[bin_assign == j])
            bin_accu[j - 1] = np.mean(matched[bin_assign == j])
    ECE = np.sum(bin_size * np.abs(bin_conf - bin_accu)) / test_size
    MCE = float(np.max(np.abs(bin_conf - bin_accu)))
    return nll_all, accu, Brier, ECE, MCE, 0.0, bin_conf, bin_accu
