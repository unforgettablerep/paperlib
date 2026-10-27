"""Train one or more variable mixers on the synthetic data and print test MSE.

Example (round 4, no seasonality, static clusters):
    python train.py --setting static --season 0 --models ci,sparse8,lrs8_8 --seeds 0 1 2

Output: one tab-separated line per (model, seed): tag, model, seed, test MSE, seconds,
plus neighbour diagnostics for models with a sparse branch.
"""
import argparse
import time
import torch
import torch.nn as nn
from data import N, L, T, make_static, make_dynamic, windows
from models import (HubSSMMixer, MeanPoolMixer, AttnMixer, BiScanMixer, SparseTopKMixer,
                    LowRankSparseMixer, Forecaster)

D = 32
MODELS = {   # key: (display name, mixer factory)
    "ci": ("CI (no mixing)", lambda: None),
    "meanpool": ("MeanPool (homogeneous)", lambda: MeanPoolMixer(D, n_nodes=N, node_dim=8)),
    "hub1": ("Hub K=1 (pooled field)", lambda: HubSSMMixer(D, 1, n_nodes=N, node_dim=8)),
    "hub4_nomem": ("Hub K=4, no time memory",
                   lambda: HubSSMMixer(D, 4, n_nodes=N, node_dim=8, temporal=False)),
    "hub4": ("Hub K=4", lambda: HubSSMMixer(D, 4, n_nodes=N, node_dim=8)),
    "hub8": ("Hub K=8", lambda: HubSSMMixer(D, 8, n_nodes=N, node_dim=8)),
    "hub8_noemb": ("Hub K=8, no node emb", lambda: HubSSMMixer(D, 8)),
    "hubv2_8": ("Hub-v2 K=8", lambda: HubSSMMixer(D, 8, n_nodes=N, node_dim=8, select=True)),
    "hubv2_16": ("Hub-v2 K=16", lambda: HubSSMMixer(D, 16, n_nodes=N, node_dim=8, select=True)),
    "sparse8": ("Sparse k=8", lambda: SparseTopKMixer(D, N, node_dim=8, k=8)),
    "lrs8_8": ("LR+Sparse K=8 k=8", lambda: LowRankSparseMixer(D, 8, N, node_dim=8, k=8)),
    "lrs8_16": ("LR+Sparse K=8 k=16", lambda: LowRankSparseMixer(D, 8, N, node_dim=8, k=16)),
    "attn": ("Attention O(N^2)", lambda: AttnMixer(D)),
    "biscan": ("Bi-scan S6 (S-Mamba-like)", lambda: BiScanMixer(D)),
}


def neighbour_diagnostics(model, meta):
    """Share of learned top-k neighbours in the same cluster (any regime), and for the
    static setting the share of followers whose cluster leader is among their neighbours."""
    m = model.mixer
    sparse = m.sparse if isinstance(m, LowRankSparseMixer) else m
    if not isinstance(sparse, SparseTopKMixer):
        return ""
    with torch.no_grad():
        _, idx = sparse.neighbours()
    lab = meta["labels"]                                        # (R, N)
    same = (lab[:, idx] == lab[:, :, None]).any(0).float()      # (N, k)
    out = f"\tnb_same_cluster={same.mean():.2f}"
    chance = ((lab[:, None, :] == lab[:, :, None]).any(0).float().sum(1) - 1).mean() / (N - 1)
    out += f"(chance {chance:.2f})"
    if meta["leader"] is not None:
        lead = meta["leader"]
        lab0 = lab[0]
        hits = [bool(((lab0[idx[i]] == lab0[i]) & lead[idx[i]]).any())
                for i in range(N) if not lead[i]]
        out += f"\tleader_in_nb={sum(hits) / len(hits):.2f}(chance {sparse.k / (N - 1):.2f})"
    return out


def run(setting, key, seed, season=1.0, use_id=True, steps=800, bs=64):
    torch.manual_seed(seed)
    X, meta = (make_static if setting == "static" else make_dynamic)(seed, season=season)
    X = (X - X[:8000].mean(0)) / X[:8000].std(0)
    model = Forecaster(MODELS[key][1](), N, L, T, d_model=D, id_emb=use_id)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for _ in range(steps):
        xb, yb = windows(X, torch.randint(0, 8000 - L - T, (bs,)))
        loss = nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        starts = torch.arange(8000, X.shape[0] - L - T, 8)
        mse = sum(nn.functional.mse_loss(model(xb), yb, reduction="sum").item()
                  for xb, yb in (windows(X, s) for s in starts.split(256)))
    return mse / (len(starts) * T * N), time.time() - t0, neighbour_diagnostics(model, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["static", "dynamic"], required=True)
    ap.add_argument("--season", type=float, default=1.0)
    ap.add_argument("--id", type=int, default=1, help="add a per-variable identity embedding")
    ap.add_argument("--models", required=True, help="comma-separated keys of MODELS")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or f"{a.setting}-season{a.season:g}-id{a.id}"
    for key in a.models.split(","):
        for s in a.seeds:
            mse, sec, diag = run(a.setting, key, s, a.season, bool(a.id))
            print(f"{tag}\t{MODELS[key][0]}\tseed={s}\tmse={mse:.4f}\t{sec:.0f}s{diag}", flush=True)
