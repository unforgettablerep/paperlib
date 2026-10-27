"""Property checks: permutation equivariance, temporal causality, lagged cross-variable
influence, coupling rank, and forward-time scaling with the number of variables N."""
import time
import torch
from models import HubSSMMixer, BiScanMixer, AttnMixer, SparseTopKMixer, LowRankSparseMixer

torch.manual_seed(0)
torch.set_num_threads(4)
b, N, P, D, K = 2, 50, 6, 32, 8


def permuted_output(m, x, perm):
    """Run m on x[:, perm] with every node-embedding table permuted the same way."""
    tables = [p for name, p in m.named_parameters() if name.endswith("node_emb")]
    saved = [t.data.clone() for t in tables]
    for t, s in zip(tables, saved):
        t.data = s[perm]
    y = m(x[:, perm])
    for t, s in zip(tables, saved):
        t.data = s
    return y


x = torch.randn(b, N, P, D)
perm = torch.randperm(N)
mixers = {"hub": HubSSMMixer(D, K, n_nodes=N, node_dim=8),
          "hub-v2": HubSSMMixer(D, K, n_nodes=N, node_dim=8, select=True),
          "sparse": SparseTopKMixer(D, N, node_dim=8, k=8),
          "lowrank+sparse": LowRankSparseMixer(D, K, N, node_dim=8, k=8),
          "bi-scan": BiScanMixer(D)}
print("[equivariance] max|f(Px) - P f(x)|")
for name, m in mixers.items():
    m.eval()
    with torch.no_grad():
        err = (permuted_output(m, x, perm) - m(x)[:, perm]).abs().max().item()
    print(f"  {name:15s} {err:.2e}")

hub = mixers["hub"]
x2 = x.clone()
x2[:, :, 4] += torch.randn(D)        # non-constant: LayerNorm would cancel a constant shift
with torch.no_grad():
    d = (hub(x2) - hub(x)).abs().amax(dim=(0, 1, 3))
print("[causality] hub output change per patch after perturbing patch 4:",
      " ".join(f"{v:.1e}" for v in d.tolist()))

x3 = x.clone()
x3[:, 7, 1] += torch.randn(D)
with torch.no_grad():
    d3 = (hub(x3) - hub(x))[:, 3].abs().amax(dim=(0, 2))
print("[lag] hub: effect of var7@patch1 on var3 at patches 0..5:",
      " ".join(f"{v:.1e}" for v in d3.tolist()))

with torch.no_grad():
    M = hub.coupling(x)[0, -1]
s = torch.linalg.svdvals(M)
off = M[~torch.eye(N, dtype=torch.bool)]
print(f"[coupling] hub numerical rank={int((s > s[0] * 1e-5).sum())} (K={K}), "
      f"off-diagonal CV={(off.std() / off.mean()).item():.2f} (pooled mean field: rank 1, CV 0)")

print("[scaling] forward time in ms, batch=4, P=12, D=64, K=16, k=8 (CPU)")
scale_mixers = lambda n: {"hub-v2": HubSSMMixer(64, 16, n_nodes=n, node_dim=8, select=True),
                          "lowrank+sparse": LowRankSparseMixer(64, 16, n, node_dim=8, k=8),
                          "attention": AttnMixer(64)}
for n in [256, 1024, 4096, 8192]:
    xx = torch.randn(4, n, 12, 64)
    cells = []
    for name, m in scale_mixers(n).items():
        m.eval()
        try:
            with torch.no_grad():
                m(xx)
                t0 = time.perf_counter()
                for _ in range(3):
                    m(xx)
            cells.append(f"{name}={(time.perf_counter() - t0) / 3 * 1e3:8.1f}")
        except RuntimeError:
            cells.append(f"{name}=     OOM")
    print(f"  N={n:5d}  " + "  ".join(cells))
