"""Pilot 5: does cross-variable information still help on top of a time-series foundation model,
and can a cheap graph adapter recover it?

Base: Chronos-2 (120M) zero-shot, univariate mode (each channel alone, i.e. CI) vs its native
multivariate mode (all channels of a window form one group). Adapters (ridge, shared over channels,
fit on TSFM forecasts of training windows):
  ci_adapter    [p_c, x_c(last 96)]                         -> y_c
  graph_adapter [p_c, (A p)_c, x_c(last 96), (A x)_c(last 96)] -> y_c, A = static top-k train-corr graph
Usage: python pilot5_tsfm.py DATASET CTX H N_TRAIN_WIN N_TEST_WIN K
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from common import Dataset, save_json, topk_adj

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def pick(idx, n):
    return idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def main():
    name, ctx, H, n_tr, n_te, K = sys.argv[1], *map(int, sys.argv[2:7])
    from chronos import Chronos2Pipeline
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
    ds = Dataset(name)
    X, C = ds.X, ds.C
    A = topk_adj(np.nan_to_num(ds.train_corr()), min(K, C - 1))
    tr = pick(ds.starts("train", ctx, H, 1), n_tr)
    va = pick(ds.starts("val", ctx, H, 1), max(4, n_tr // 3))
    te = pick(ds.starts("test", ctx, H, 1), n_te)

    def forecast(idx, multivariate):
        ctxs = np.stack([X[t - ctx:t].T for t in idx])  # n, C, ctx
        if multivariate:
            inp = torch.as_tensor(ctxs, dtype=torch.float32)
        else:
            inp = torch.as_tensor(ctxs.reshape(-1, 1, ctx), dtype=torch.float32)
        out = pipe.predict(inp, prediction_length=H, batch_size=32 if multivariate else 256)
        q = pipe.quantiles
        mid = int(np.argmin(np.abs(np.asarray(q) - 0.5)))
        med = torch.stack([o[:, mid, :] for o in out]).numpy()  # n_items, n_variates, H
        return med.reshape(len(idx), C, H)

    res = {"dataset": name, "ctx": ctx, "H": H, "C": C, "n_train_windows": len(tr), "n_test_windows": len(te)}
    P = {}
    for mode in ["uni", "multi"]:
        t1 = time.time()
        P[mode] = {s: forecast(idx, mode == "multi") for s, idx in [("train", tr), ("val", va), ("test", te)]}
        res[f"seconds_{mode}"] = time.time() - t1
        print(f"[{name}] chronos-2 {mode} done in {res[f'seconds_{mode}']:.0f}s", flush=True)
    Y = {s: np.stack([X[t:t + H].T for t in idx]) for s, idx in [("train", tr), ("val", va), ("test", te)]}

    def mse(p, y):
        return float(((p - y) ** 2).mean()), float(np.abs(p - y).mean())

    res["zero_shot"] = {m: dict(zip(["mse", "mae"], mse(P[m]["test"], Y["test"]))) for m in P}

    def adapter_feats(split, idx, kind, base):
        rows, tg, mus, sds = [], [], [], []
        for w, t in enumerate(idx):
            x = X[t - 96:t].T.astype(np.float64)
            mu, sd = x.mean(1, keepdims=True), np.maximum(x.std(1, keepdims=True), 0.05)
            xn = (x - mu) / sd
            pn = (P[base][split][w] - mu) / sd
            f = [pn, xn]
            if kind == "graph":
                f += [A @ pn, A @ xn]
            rows.append(np.concatenate(f, 1))
            tg.append((Y[split][w] - mu) / sd)
            mus.append(mu)
            sds.append(sd)
        return np.concatenate(rows), np.concatenate(tg), np.concatenate(mus), np.concatenate(sds)

    res["adapters"] = {}
    for base in ["uni", "multi"]:
        for kind in ["ci", "graph"]:
            Ftr, Ytr, _, _ = adapter_feats("train", tr, kind, base)
            Fva, _, muv, sdv = adapter_feats("val", va, kind, base)
            Fte, _, mut, sdt = adapter_feats("test", te, kind, base)
            best = None
            for lam in [1e-2, 1e-1, 1.0, 10.0]:
                W = np.linalg.solve(Ftr.T @ Ftr / len(Ftr) + lam * np.eye(Ftr.shape[1]), Ftr.T @ Ytr / len(Ftr))
                v = mse((Fva @ W) * sdv + muv, Y["val"].reshape(-1, H))[0]
                if best is None or v < best[0]:
                    best = (v, lam, W)
            pt = (Fte @ best[2]) * sdt + mut
            m, a = mse(pt, Y["test"].reshape(-1, H))
            res["adapters"][f"{base}+{kind}_adapter"] = {"mse": m, "mae": a, "lambda": best[1]}
    base_mse = res["zero_shot"]["uni"]["mse"]
    res["gain_vs_uni_zero_shot_%"] = {k: 100 * (base_mse - v["mse"]) / base_mse
                                       for k, v in {**{f"{m}_zero_shot": res["zero_shot"][m] for m in P},
                                                    **res["adapters"]}.items()}
    res["seconds"] = time.time() - t0
    save_json(res, f"pilot5_{name}_ctx{ctx}_H{H}.json")
    for k, v in res["gain_vs_uni_zero_shot_%"].items():
        print(f"  {k:24s} gain vs chronos-2 univariate zero-shot: {v:+.2f}%")


if __name__ == "__main__":
    main()
