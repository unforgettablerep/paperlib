"""Pilot 1b: does the *use* of neighbour information depend on the calendar phase?

Ridge, shared over channels, static top-k graph A (train corr). Feature sets:
  shared      [x, Ax]
  nb_phase    [x, Ax (x) g]      neighbour weights specific to phase group g of the forecast start
  own_phase   [x (x) g, Ax]      control: own-history weights phase-specific instead
  both_phase  [x (x) g, Ax (x) g]
g = 6 four-hour blocks x weekday/weekend (12 groups). Gram matrices are assembled from per-group
accumulations of the 2L x 2L base design, so every variant is exact and cheap.
If nb_phase gains clearly more than own_phase, cross-variable dependence is phase-dependent in a
way that matters for prediction.
Usage: python pilot1b_phase_weights.py DATASET L H K TRAIN_STRIDE TEST_STRIDE
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from common import Dataset, save_json, topk_adj

NAMES = ["ci", "shared", "nb_phase", "own_phase", "both_phase"]
NG = 12


def main():
    name, L, H, K, s_tr, s_te = sys.argv[1], *map(int, sys.argv[2:7])
    t0 = time.time()
    ds = Dataset(name)
    X, C = ds.X.astype(np.float64), ds.C
    A = topk_adj(np.nan_to_num(ds.train_corr()), min(K, C - 1))
    tr, va, te = (ds.starts(s, L, H, st) for s, st in [("train", s_tr), ("val", s_te), ("test", s_te)])

    def group(t):
        tt = pd.to_datetime(ds.ts[np.asarray(t)], unit="s")
        return (tt.hour.to_numpy() // 4 + 6 * (tt.dayofweek.to_numpy() >= 5)).astype(int)

    def base(t):
        x = X[t - L:t].T
        mu, sd = x.mean(1, keepdims=True), np.maximum(x.std(1, keepdims=True), 0.05)
        xn = (x - mu) / sd
        return np.concatenate([xn, A @ xn], 1), (X[t:t + H].T - mu) / sd, mu, sd, X[t:t + H].T

    M = np.zeros((NG, 2 * L, 2 * L))
    R = np.zeros((NG, 2 * L, H))
    for t, g in zip(tr, group(tr)):
        F, y, *_ = base(t)
        M[g] += F.T @ F
        R[g] += F.T @ y
    N = len(tr) * C
    xs, ns = slice(0, L), slice(L, 2 * L)

    def solve(kind, lam):
        """Return per-group effective weights (NG, 2L, H)."""
        own_sp = kind in ("own_phase", "both_phase")
        nb_sp = kind in ("nb_phase", "both_phase")
        if kind == "ci":
            G = M[:, xs, xs].sum(0) / N
            W = np.linalg.solve(G + lam * np.eye(L), R[:, xs].sum(0) / N)
            out = np.zeros((NG, 2 * L, H))
            out[:, xs] = W
            return out
        # block layout: own part (1 or NG blocks of L) then neighbour part (1 or NG blocks of L)
        no, nn_ = (NG if own_sp else 1), (NG if nb_sp else 1)
        dim = L * (no + nn_)
        G = np.zeros((dim, dim))
        B = np.zeros((dim, H))

        def oi(g):
            return slice((g if own_sp else 0) * L, ((g if own_sp else 0) + 1) * L)

        def ni(g):
            o = no * L
            return slice(o + (g if nb_sp else 0) * L, o + ((g if nb_sp else 0) + 1) * L)

        for g in range(NG):
            for (a, sa), (b, sb) in [((oi(g), xs), (oi(g), xs)), ((oi(g), xs), (ni(g), ns)),
                                     ((ni(g), ns), (oi(g), xs)), ((ni(g), ns), (ni(g), ns))]:
                G[a, b] += M[g][sa, sb]
            B[oi(g)] += R[g][xs]
            B[ni(g)] += R[g][ns]
        W = np.linalg.solve(G / N + lam * np.eye(dim), B / N)
        out = np.zeros((NG, 2 * L, H))
        for g in range(NG):
            out[g, xs] = W[oi(g)]
            out[g, ns] = W[ni(g)]
        return out

    lams = [1e-3, 1e-2, 1e-1]
    Ws = {n: {l: solve(n, l) for l in lams} for n in NAMES}

    def score(idx):
        se = {n: {l: 0.0 for l in lams} for n in NAMES}
        cnt = 0
        for t, g in zip(idx, group(idx)):
            F, _, mu, sd, y = base(t)
            for n in NAMES:
                for l in lams:
                    se[n][l] += float((((F @ Ws[n][l][g]) * sd + mu - y) ** 2).sum())
            cnt += y.size
        return {n: {l: se[n][l] / cnt for l in lams} for n in NAMES}

    v = score(va)
    best = {n: min(lams, key=lambda l: v[n][l]) for n in NAMES}
    tst = score(te)
    res = {"dataset": name, "L": L, "H": H, "K": K, "test_mse": {n: tst[n][best[n]] for n in NAMES}}
    b = res["test_mse"]["shared"]
    res["gain_vs_shared_%"] = {n: 100 * (b - res["test_mse"][n]) / b for n in NAMES}
    res["seconds"] = time.time() - t0
    save_json(res, f"pilot1b_{name}_L{L}_H{H}.json")
    print(name, {n: f"{res['test_mse'][n]:.4f} ({res['gain_vs_shared_%'][n]:+.2f}%)" for n in NAMES}, flush=True)


if __name__ == "__main__":
    main()
