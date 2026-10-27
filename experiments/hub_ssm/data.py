"""Synthetic multivariate series with clustered lead-lag structure.

64 variables in 4 clusters of 16. Each cluster has a latent driver: AR(1) (phi=0.97)
plus, when season=1, a sinusoid with a cluster-specific period in [20, 40).
Variable j copies its driver with lag l_j in [0, 12], random sign (70% positive) and
scale in [0.5, 1.5], plus N(0, 0.3^2) noise, so a follower's next l_j steps are visible
in its leader's past.

static : fixed clusters; the first member of each cluster is its lag-0 leader;
         variable order is shuffled.
dynamic: two regimes alternate in segments of 100-300 steps; each variable belongs to
         a (generally different) cluster in each regime.
"""
import torch

N_CLUSTERS, PER, L, T = 4, 16, 48, 16
N = N_CLUSTERS * PER


def _drivers(g, length, season):
    t = torch.arange(length + 20).float()
    drivers = []
    for _ in range(N_CLUSTERS):
        eps = torch.randn(length + 20, generator=g)
        ar = torch.zeros(length + 20)
        for i in range(1, length + 20):
            ar[i] = 0.97 * ar[i - 1] + eps[i]
        period = 20 + 20 * torch.rand(1, generator=g)
        drivers.append(ar / ar.std() + season * torch.sin(2 * torch.pi * t / period))
    return drivers


def make_static(seed, length=12000, season=1.0):
    """Returns X (length, N) and meta with per-variable cluster label and leader flag."""
    g = torch.Generator().manual_seed(seed)
    drivers = _drivers(g, length, season)
    cols, labels, leader = [], [], []
    for c in range(N_CLUSTERS):
        for k in range(PER):
            lag = 0 if k == 0 else int(torch.randint(0, 13, (1,), generator=g))
            sign = 1.0 if torch.rand(1, generator=g) < 0.7 else -1.0
            scale = 0.5 + torch.rand(1, generator=g)
            s = drivers[c][20 - lag: 20 - lag + length]
            cols.append(sign * scale * s + 0.3 * torch.randn(length, generator=g))
            labels.append(c)
            leader.append(k == 0)
    perm = torch.randperm(N, generator=g)                       # arbitrary variable order
    X = torch.stack(cols, 1)[:, perm]
    meta = {"labels": torch.tensor(labels)[perm].unsqueeze(0),   # (1, N): one regime
            "leader": torch.tensor(leader)[perm]}
    return X, meta


def make_dynamic(seed, length=12000, season=1.0):
    """Returns X (length, N) and meta with the cluster label of each variable per regime."""
    g = torch.Generator().manual_seed(seed)
    D_ = torch.stack(_drivers(g, length, season))               # (C, length+20)
    regime = torch.zeros(length, dtype=torch.long)
    pos, r = 0, 0
    while pos < length:
        seg = int(torch.randint(100, 301, (1,), generator=g))
        regime[pos:pos + seg] = r
        pos, r = pos + seg, 1 - r
    assign = torch.stack([torch.randperm(N, generator=g) % N_CLUSTERS for _ in range(2)])  # (2, N)
    lag = torch.randint(0, 13, (N,), generator=g)
    sign = torch.where(torch.rand(N, generator=g) < 0.7, 1.0, -1.0)
    scale = 0.5 + torch.rand(N, generator=g)
    tt = torch.arange(length)
    X = torch.empty(length, N)
    for j in range(N):
        c = assign[regime, j]                                   # cluster of j at each time
        X[:, j] = sign[j] * scale[j] * D_[c, 20 + tt - lag[j]]
    X = X + 0.3 * torch.randn(length, N, generator=g)
    return X, {"labels": assign, "leader": None}


def windows(X, starts):
    idx = starts[:, None] + torch.arange(L + T)[None]
    w = X[idx]                                                  # (b, L+T, N)
    return w[:, :L], w[:, L:]
