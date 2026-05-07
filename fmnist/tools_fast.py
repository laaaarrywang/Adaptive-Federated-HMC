"""GPU-resident loaders and validation, mirroring tools.py but with all data
preloaded as tensors. Eliminates the DataLoader-induced bottleneck (transform
+ CPU->GPU copy on every batch fetch) which dominates wall time on a tiny
linear model.

Reproducibility note: client splitting uses the same `torch.Generator(seed)`
seeded RNG as `tools.loader_federated`, so client indices are identical
between slow and fast paths. Batch ordering within a client uses `randperm`
on a per-epoch basis (vs. DataLoader's PyTorch sampler) — different RNG
stream, so trajectories are not bit-identical, but final-round metrics agree
within sample noise.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Func
import torchvision.datasets as datasets

import transforms


def _materialize_to_tensor(dataset):
    """Apply transforms once, stack to (N, 1, 28, 28) cuda tensor."""
    images = []
    labels = []
    for img, lbl in dataset:
        images.append(img)  # transform already applied: (1, 28, 28) tensor
        labels.append(lbl)
    X = torch.stack(images).cuda()              # (N, 1, 28, 28) float32
    y = torch.tensor(labels, dtype=torch.long).cuda()  # (N,) int64
    return X, y


def loader_federated_fast(args):
    """Pre-load FashionMNIST to GPU, return tensors + per-client index tensors.

    Returns:
      train_x  (60000, 1, 28, 28) float32 cuda
      train_y  (60000,)           int64   cuda
      clients  list of N tensors of int64 cuda — index lists into train_x
      test_x   (10000, 1, 28, 28) float32 cuda
      test_y   (10000,)           int64   cuda
      train_size = 60000

    Splitting strategy is selected by `args.split`:
      'iid'        — uniform random shuffle then equal-size shards (default)
      'dirichlet'  — Dirichlet(args.dirichlet_alpha) over labels-per-client.
                     For each class c, sample p_c ~ Dirichlet(alpha · 1_N)
                     and split class-c indices into proportions p_c. Smaller
                     alpha = more skewed (1-2 dominant classes per client at
                     alpha=0.1).
    """
    if not args.data.startswith('Fashion'):
        raise SystemExit('Unknown dataset')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    trainset = datasets.FashionMNIST(
        './data/' + args.data, train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root='./data/' + args.data, train=False, download=True, transform=transform
    )

    train_x, train_y = _materialize_to_tensor(trainset)
    test_x, test_y = _materialize_to_tensor(testset)

    train_size = train_x.shape[0]
    split = getattr(args, 'split', 'iid')

    if split == 'iid':
        # Mirrors tools.loader_federated's random_split with the same seed-generator.
        client_length = int(np.floor(train_size / args.num_client))
        sizes = [client_length] * (args.num_client - 1)
        sizes.append(train_size - sum(sizes))
        perm = torch.randperm(train_size, generator=torch.Generator().manual_seed(args.seed))
        clients = []
        start = 0
        for sz in sizes:
            clients.append(perm[start:start + sz].cuda())
            start += sz

    elif split == 'dirichlet':
        alpha = float(getattr(args, 'dirichlet_alpha', 0.1))
        rng = np.random.default_rng(args.seed)
        labels_cpu = train_y.cpu().numpy()
        num_classes = int(labels_cpu.max()) + 1
        N = args.num_client
        per_client_idx = [[] for _ in range(N)]
        for c in range(num_classes):
            idx_c = np.where(labels_cpu == c)[0]
            rng.shuffle(idx_c)
            # Dirichlet(alpha · 1_N) -> proportions of class c per client
            p = rng.dirichlet([alpha] * N)
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            chunks = np.split(idx_c, cuts)
            for i, chunk in enumerate(chunks):
                per_client_idx[i].extend(chunk.tolist())
        # Drop clients that ended up with zero samples (rare under Dir(0.1) for
        # the smallest classes, but possible). Their gradient is undefined.
        clients = []
        for i, idx in enumerate(per_client_idx):
            if len(idx) == 0:
                raise RuntimeError(
                    f'Client {i} got 0 samples under Dirichlet(alpha={alpha}), '
                    f'seed={args.seed}. Increase alpha or reseed.')
            clients.append(torch.tensor(sorted(idx), dtype=torch.long).cuda())

    else:
        raise SystemExit(f'Unknown split: {split!r}; use "iid" or "dirichlet"')

    return train_x, train_y, clients, test_x, test_y, train_size


def validation_fast(net, test_x: torch.Tensor, test_y: torch.Tensor, M: int,
                    batch_test: int = 2000):
    """Mirrors tools.validation but consumes GPU tensors directly. Returns the
    same 6-tuple of metrics plus the (test_size, num_classes) probability tensor."""
    test_size = test_x.shape[0]
    nll_all = 0.0
    accu = 0
    Brier = 0.0
    entropies_sum = 0.0
    conf_accu = np.zeros((test_size, 2))
    probs_all = torch.zeros(test_size, 10, device='cuda')

    with torch.no_grad():
        count = 0
        for start in range(0, test_size, batch_test):
            end = min(start + batch_test, test_size)
            images = test_x[start:end]
            labels = test_y[start:end]
            batch_size = end - start

            output = Func.log_softmax(net(images), dim=1)
            probs = output.exp()
            probs_all[start:end] = probs
            nll_all += Func.nll_loss(output, labels, reduction='sum').item()
            Brier += torch.sum((probs - Func.one_hot(labels, num_classes=output.shape[1])) ** 2).item()
            max_conf, pred = probs.max(dim=1)
            conf_accu[start:end, 0] = max_conf.cpu().numpy()
            matched = pred.eq(labels)
            accu += matched.sum().item()
            conf_accu[start:end, 1] = matched.cpu().numpy()
            entropies_sum += float(-torch.sum(output * probs))
            count += batch_size

    accu = accu / test_size
    Brier = Brier / test_size
    entropy = entropies_sum / test_size

    bins = np.linspace(0, 1, M + 1)
    bin_assign = np.digitize(conf_accu[:, 0], bins)
    bin_conf = np.zeros(M)
    bin_accu = np.zeros(M)
    bin_size = np.zeros(M, dtype=int)
    for j in range(1, M + 1):
        bin_size[j - 1] = np.sum(bin_assign == j)
        if bin_size[j - 1] > 0:
            bin_conf[j - 1] = np.mean(conf_accu[bin_assign == j, 0])
            bin_accu[j - 1] = np.mean(conf_accu[bin_assign == j, 1])

    ECE = np.sum(bin_size * np.abs(bin_conf - bin_accu)) / test_size
    MCE = np.max(np.abs(bin_conf - bin_accu))

    return (nll_all, accu, Brier, ECE, MCE, entropy), probs_all
