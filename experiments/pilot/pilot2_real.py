"""Pilot 2B/2C on real data.

2B diagnostics (train split only): PC1/PC3 variance share, raw vs factor-removed coupling,
   and cheap linear probes (ridge, shared over channels) for CI vs +graph vs +factor vs +factor+residual-graph.
2C small neural forecasters with one backbone and different channel-mixing blocks:
   ci, attn (dense, input-dependent), dense (static CxC), factor (r tokens, low rank),
   sparse (attention masked to top-k raw-corr graph), factor_sparse (factor + attention masked
   to top-k factor-removed-corr graph).
Usage: python pilot2_real.py DATASET TRAIN_STRIDE EVAL_STRIDE [variants comma-separated]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from common import Dataset, save_json, topk_adj
from models import Forecaster, topk_mask, train_eval

L, H = 96, 96
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def factor_basis(Xtr, r):
    Z = Xtr - Xtr.mean(0)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return Vt[:r].T, S ** 2 / (S ** 2).sum()


def coupling(R, k):
    M = np.abs(np.nan_to_num(R)).copy()
    np.fill_diagonal(M, 0)
    return float(np.sort(M, axis=1)[:, -k:].mean())


def ridge_probe(ds, Fac, A_raw, A_res, s_tr, s_ev):
    X = ds.X.astype(np.float64)
    tr = ds.starts("train", L, H, s_tr)
    va = ds.starts("val", L, H, s_ev)
    names = ["ci", "graph", "factor", "factor+resid_graph"]

    def feats(t):
        x = X[t - L:t].T
        mu, sd = x.mean(1, keepdims=True), np.maximum(x.std(1, keepdims=True), 0.05)
        xn = (x - mu) / sd
        f = Fac[t - L:t].T  # r x L
        f = (f - f.mean(1, keepdims=True)) / np.maximum(f.std(1, keepdims=True), 0.05)
        fr = np.repeat(f.reshape(1, -1), ds.C, 0)
        out = {"ci": xn, "graph": np.concatenate([xn, A_raw @ xn], 1),
               "factor": np.concatenate([xn, fr], 1),
               "factor+resid_graph": np.concatenate([xn, fr, A_res @ xn], 1)}
        y = (X[t:t + H].T - mu) / sd
        return out, y, mu, sd

    G = {n: 0 for n in names}
    B = {n: 0 for n in names}
    N = 0
    for t in tr:
        f, y, _, _ = feats(t)
        for n in names:
            G[n] = G[n] + f[n].T @ f[n]
            B[n] = B[n] + f[n].T @ y
        N += ds.C
    W = {n: np.linalg.solve(G[n] / N + 1e-2 * np.eye(G[n].shape[0]), B[n] / N) for n in names}
    se = {n: 0.0 for n in names}
    cnt = 0
    for t in va:
        f, _, mu, sd = feats(t)
        y = X[t:t + H].T
        for n in names:
            se[n] += float((((f[n] @ W[n]) * sd + mu - y) ** 2).sum())
        cnt += y.size
    mse = {n: se[n] / cnt for n in names}
    return {n: {"val_mse": mse[n], "gain_vs_ci_%": 100 * (mse["ci"] - mse[n]) / mse["ci"]} for n in names}


def main():
    name, s_tr, s_ev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ["ci", "attn", "dense", "factor", "sparse",
                                                                   "factor_sparse"]
    t0 = time.time()
    ds = Dataset(name)
    C = ds.C
    tr_end = ds.splits[0]
    Xtr = ds.X[:tr_end].astype(np.float64)
    r = min(3, C - 2)
    k = min(10, C - 1)
    V, var_share = factor_basis(Xtr, r)
    E = Xtr - (Xtr @ V) @ V.T
    R_raw = np.nan_to_num(np.corrcoef(Xtr.T))
    R_res = np.nan_to_num(np.corrcoef(E.T))
    Fac = ds.X.astype(np.float64) @ V
    diag = {"C": C, "pc1_share": float(var_share[0]), "pc3_share": float(var_share[:3].sum()),
            "raw_coupling_topk": coupling(R_raw, min(5, C - 1)),
            "resid_coupling_topk": coupling(R_res, min(5, C - 1)),
            "linear_probe": ridge_probe(ds, Fac, topk_adj(R_raw, k), topk_adj(R_res, k), s_tr, s_ev)}
    print(f"[{name}] diagnostics in {time.time()-t0:.0f}s:", {k2: v for k2, v in diag.items() if k2 != "linear_probe"})
    for n, v in diag["linear_probe"].items():
        print(f"   probe {n:20s} val_mse={v['val_mse']:.4f} gain={v['gain_vs_ci_%']:+.2f}%")
    tr = ds.starts("train", L, H, s_tr)
    va = ds.starts("val", L, H, s_ev)
    te = ds.starts("test", L, H, s_ev)
    m_raw, m_res = topk_mask(R_raw, k), topk_mask(R_res, k)
    res = {"dataset": name, "L": L, "H": H, "diagnostics": diag, "models": {},
           "config": {"train_stride": s_tr, "eval_stride": s_ev, "D": 64, "layers": 2, "k": k, "r_factor_tokens": min(4, C)}}
    for v in variants:
        torch.manual_seed(0)
        mask = m_raw if v == "sparse" else m_res if v == "factor_sparse" else None
        model = Forecaster(L, H, C, 64, mix=v, layers=2, mask=mask, r=min(4, C))
        out = train_eval(model, ds.X, tr, va, te, L, H, epochs=8, patience=2, lr=1e-3, bs=32, seed=0,
                         log=lambda s: print(s, flush=True))
        res["models"][v] = out
        print(f"[{name}] {v:14s} test_mse={out['test_mse']:.4f} test_mae={out['test_mae']:.4f} "
              f"ep={out['epochs_run']} {out['seconds']:.0f}s params={out['params']}", flush=True)
        save_json(res, f"pilot2_real_{name}.json")
    base = res["models"].get("ci", {}).get("test_mse")
    if base:
        for v in res["models"]:
            res["models"][v]["gain_vs_ci_%"] = 100 * (base - res["models"][v]["test_mse"]) / base
    res["seconds"] = time.time() - t0
    save_json(res, f"pilot2_real_{name}.json")


if __name__ == "__main__":
    main()
