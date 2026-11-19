"""Pilot 3: does the variable scan order matter for a Mamba channel mixer (S-Mamba style)?

Same model, same seed, only the channel order differs. Orders: original, random(x3),
spectral (Fiedler vector of |corr| graph, as GPS-Mamba's Laplace scanning), spectral_dist
(Fiedler of road-distance kernel, PEMS only), olo (hierarchical clustering optimal leaf
order), anti (spectral order interleaved so that similar channels are far apart).
Order quality = normalised linear-arrangement cost on the top-k |corr| graph.
Usage: python pilot3_order.py DATASET H TRAIN_STRIDE EVAL_STRIDE EPOCHS VARIANTS [LAYERS]
  VARIANTS e.g. "bi16:original,random0,random1,random2,spectral,anti,olo;uni16:spectral,anti;seed:1,2;ci"
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.linalg import eigh
from scipy.spatial.distance import squareform

from common import Dataset, save_json
from models import Forecaster, train_eval

L = 96
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def fiedler_order(W):
    d = W.sum(1)
    Lap = np.diag(d) - W
    vals, vecs = eigh(Lap, np.diag(d + 1e-8))
    return np.argsort(vecs[:, 1])


def adversarial_order(W, seed=7, iters=30000):
    """Hill-climb random swaps to MAXIMISE linear-arrangement cost (similar channels far apart)."""
    rng = np.random.default_rng(seed)
    C = W.shape[0]
    order = rng.permutation(C)
    pos = np.empty(C, int)
    pos[order] = np.arange(C)
    for _ in range(iters):
        i, k = rng.integers(0, C, 2)
        if i == k:
            continue
        pi, pk = pos[i], pos[k]
        mask = np.ones(C, bool)
        mask[[i, k]] = False
        d = (W[i, mask] * (np.abs(pk - pos[mask]) - np.abs(pi - pos[mask]))).sum() + \
            (W[k, mask] * (np.abs(pi - pos[mask]) - np.abs(pk - pos[mask]))).sum()
        if d > 0:
            pos[i], pos[k] = pk, pi
    return np.argsort(pos)


def la_cost(order, W):
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    D = np.abs(pos[:, None] - pos[None, :])
    return float((W * D).sum() / (W.sum() * len(order)))


def main():
    name, H, s_tr, s_ev, epochs = sys.argv[1], *map(int, sys.argv[2:6])
    spec = sys.argv[6]
    layers = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    ds = Dataset(name)
    C = ds.C
    Xtr = ds.X[: ds.splits[0]].astype(np.float64)
    R = np.abs(np.nan_to_num(np.corrcoef(Xtr.T)))
    np.fill_diagonal(R, 0)
    k = min(10, C - 1)
    Wk = np.zeros_like(R)
    idx = np.argsort(-R, axis=1)[:, :k]
    Wk[np.repeat(np.arange(C), k), idx.ravel()] = R[np.repeat(np.arange(C), k), idx.ravel()]
    Wk = np.maximum(Wk, Wk.T)
    orders = {"original": np.arange(C)}
    for s in range(3):
        orders[f"random{s}"] = np.random.default_rng(100 + s).permutation(C)
    sp = fiedler_order(R + 1e-6)
    orders["spectral"] = sp
    orders["anti"] = adversarial_order(Wk, seed=7)
    Z = linkage(squareform(1 - np.clip(R + np.eye(C), 0, 1), checks=False), "average")
    orders["olo"] = leaves_list(optimal_leaf_ordering(Z, squareform(1 - np.clip(R + np.eye(C), 0, 1), checks=False)))
    if ds.dist is not None:
        dist = ds.dist.astype(np.float64)
        finite = dist[np.isfinite(dist) & (dist > 0)]
        Wd = np.where(np.isfinite(dist), np.exp(-(dist / finite.std()) ** 2), 0.0)
        np.fill_diagonal(Wd, 0)
        orders["spectral_dist"] = fiedler_order(Wd + 1e-6)
    costs = {o: la_cost(v, Wk) for o, v in orders.items()}
    print(f"[{name}] LA cost (lower = similar channels adjacent):", {o: round(c, 4) for o, c in costs.items()})

    tr = ds.starts("train", L, H, s_tr)
    va = ds.starts("val", L, H, s_ev)
    te = ds.starts("test", L, H, s_ev)
    fname = f"pilot3_order_{name}_H{H}.json"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", fname)
    res = {"dataset": name, "C": C, "L": L, "H": H, "layers": layers, "epochs": epochs, "train_stride": s_tr,
           "la_cost": costs, "runs": {}}
    if os.path.exists(path):
        import json
        res["runs"] = json.load(open(path)).get("runs", {})
    jobs = []
    for group in spec.split(";"):
        if group == "ci":
            jobs.append(("ci", "ci", None, 16, True, 0))
            continue
        head, rest = group.split(":")
        if head == "seed":
            for s in rest.split(","):
                jobs.append((f"bi16_original_seed{s}", "mamba", "original", 16, True, int(s)))
            continue
        bi = head.startswith("bi")
        N = int(head[2:] if bi else head[3:])
        for o in rest.split(","):
            if o in orders:
                jobs.append((f"{head}_{o}", "mamba", o, N, bi, 0))
    for key, mix, o, N, bi, seed in jobs:
        if key in res["runs"]:
            print("skip", key)
            continue
        torch.manual_seed(seed)
        model = Forecaster(L, H, C, 64, mix=mix, layers=layers, order=orders[o] if o else None, N=N, bidirectional=bi)
        out = train_eval(model, ds.X, tr, va, te, L, H, epochs=epochs, patience=2, lr=1e-3, bs=32, seed=seed,
                         log=lambda s: print(s, flush=True))
        out["order"] = o
        out["la_cost"] = costs.get(o) if o else None
        res["runs"][key] = out
        print(f"[{name}] {key:28s} test_mse={out['test_mse']:.4f} la={out['la_cost']} {out['seconds']:.0f}s", flush=True)
        save_json(res, fname)


if __name__ == "__main__":
    main()
