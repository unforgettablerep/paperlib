"""Pilot 6: is an LLM-written semantic prior graph (priors/semantic_priors.json, written from
variable names before any data analysis) a useful dependency structure?

(a) agreement: does prior edge strength rank pairs like the training-data |corr| does (AUROC)?
(b) forecasting: ridge graph-linear model y_c ~ [x_c, (A x)_c] where the EDGE SET comes from
    prior / data top-k (degree matched to the prior) / random (degree matched, 5 seeds), and edge
    WEIGHTS are always the signed train correlations (so only the structure differs).
    Full-data and few-shot (first 5% of the training span, graph also estimated from it) regimes.
Usage: python pilot6_prior.py DATASET PRIOR_KEY L H
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import roc_auc_score

from common import Dataset, save_json

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prior(key, names):
    pr = json.load(open(os.path.join(HERE, "priors", "semantic_priors.json")))[key]
    assert pr["variables"] == names, (pr["variables"], names)
    C = len(names)
    S = np.zeros((C, C))
    pos = {n: i for i, n in enumerate(names)}
    for a, b, w, *_ in pr["edges"]:
        S[pos[a], pos[b]] = S[pos[b], pos[a]] = max(S[pos[a], pos[b]], w)
    return S


def adj_from_mask(M, R):
    A = np.where(M, R, 0.0)
    return A / (np.abs(A).sum(1, keepdims=True) + 1e-8)


def ridge_eval(ds, A, tr, va, te, L, H):
    X = ds.X.astype(np.float64)
    C = ds.C

    def f(t):
        x = X[t - L:t].T
        mu, sd = x.mean(1, keepdims=True), np.maximum(x.std(1, keepdims=True), 0.05)
        xn = (x - mu) / sd
        F = xn if A is None else np.concatenate([xn, A @ xn], 1)
        return F, (X[t:t + H].T - mu) / sd, mu, sd, X[t:t + H].T

    G, B, N = 0, 0, 0
    for t in tr:
        F, y, *_ = f(t)
        G, B, N = G + F.T @ F, B + F.T @ y, N + C
    lams = [1e-3, 1e-2, 1e-1, 1.0]
    Ws = {l: np.linalg.solve(G / N + l * np.eye(G.shape[0]), B / N) for l in lams}

    def score(idx):
        se = {l: 0.0 for l in lams}
        n = 0
        for t in idx:
            F, _, mu, sd, y = f(t)
            for l in lams:
                se[l] += float((((F @ Ws[l]) * sd + mu - y) ** 2).sum())
            n += y.size
        return {l: se[l] / n for l in lams}

    v = score(va)
    lb = min(lams, key=lambda l: v[l])
    return score(te)[lb]


def main():
    name, key, L, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ds = Dataset(name)
    C = ds.C
    S = load_prior(key, ds.names)
    M_prior = S > 0
    deg = M_prior.sum(1)
    res = {"dataset": name, "L": L, "H": H, "C": C, "prior_edges": int(M_prior.sum() // 2),
           "prior_mean_degree": float(deg.mean()), "regimes": {}}
    for regime, frac in [("full", 1.0), ("fewshot_5pct", 0.05)]:
        tr_end = int(ds.splits[0] * frac)
        R = np.nan_to_num(np.corrcoef(ds.X[:tr_end].T.astype(np.float64)))
        offd = ~np.eye(C, dtype=bool)
        agree = float(roc_auc_score((np.abs(R) >= np.quantile(np.abs(R[offd]), 0.75))[offd], S[offd]))
        # data graph: per node, same degree as prior, top by |R|
        Rm = np.abs(R).copy()
        np.fill_diagonal(Rm, -1)
        M_data = np.zeros((C, C), bool)
        for i in range(C):
            if deg[i] > 0:
                M_data[i, np.argsort(-Rm[i])[:deg[i]]] = True
        tr = np.arange(L, tr_end - H + 1, 1 if C < 50 else 2)
        va = ds.starts("val", L, H, 2)
        te = ds.starts("test", L, H, 2)
        out = {"n_train_windows": int(len(tr)),
               "prior_vs_data_auroc(top-quartile |corr| pairs)": agree,
               "prior_edge_precision_in_data_topdeg": float((M_prior & M_data).sum() / max(M_prior.sum(), 1))}
        mse = {"ci": ridge_eval(ds, None, tr, va, te, L, H),
               "prior_graph": ridge_eval(ds, adj_from_mask(M_prior, R), tr, va, te, L, H),
               "data_graph": ridge_eval(ds, adj_from_mask(M_data, R), tr, va, te, L, H),
               "prior_or_data": ridge_eval(ds, adj_from_mask(M_prior | M_data, R), tr, va, te, L, H)}
        rnd = []
        for s in range(5):
            g = np.random.default_rng(s)
            M_r = np.zeros((C, C), bool)
            for i in range(C):
                if deg[i] > 0:
                    M_r[i, g.choice([j for j in range(C) if j != i], deg[i], replace=False)] = True
            rnd.append(ridge_eval(ds, adj_from_mask(M_r, R), tr, va, te, L, H))
        mse["random_graph_mean"] = float(np.mean(rnd))
        mse["random_graph_std"] = float(np.std(rnd))
        out["test_mse"] = mse
        out["gain_vs_ci_%"] = {k: 100 * (mse["ci"] - v) / mse["ci"] for k, v in mse.items() if not k.endswith("std")}
        res["regimes"][regime] = out
        print(name, regime, f"agreeAUROC={agree:.3f}", {k: f"{v:+.2f}%" for k, v in out["gain_vs_ci_%"].items()},
              f"random std={mse['random_graph_std']:.4f}", flush=True)
    save_json(res, f"pilot6_{name}_L{L}_H{H}.json")


if __name__ == "__main__":
    main()
