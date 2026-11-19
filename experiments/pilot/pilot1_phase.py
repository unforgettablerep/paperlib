"""Pilot 1: is the dependency structure of the *forecast window* better predicted by its
calendar phase than by the lookback window?  And how much would a better graph help?

(a) Statistics on test windows: compare candidate relation matrices with the realised
    future-window correlation R_fut (off-diagonal relative Frobenius error, top-k Jaccard).
    candidates: back (lookback corr), back_recent (last H lookback steps), static (global
    train corr), staticwin (mean of train future-window corrs), phase (mean of train
    future-window corrs with the same hour-of-day x weekend phase).
(b) Forecasting: ridge "graph-linear" model, shared over channels,
    y_c ~ [x_c, (A x)_c] with A = row-normalised signed top-k of each candidate,
    plus CI (x_c only) and the oracle A = topk(R_fut).
Usage: python pilot1_phase.py DATASET L H K TRAIN_STRIDE TEST_STRIDE
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import Dataset, save_json, topk_adj, window_corr

SOURCES = ["ci", "static", "staticwin", "phase", "back", "back_recent", "fut"]


def norm_win(X, t, L, H):
    x = X[t - L:t].T.astype(np.float64)  # C x L
    y = X[t:t + H].T.astype(np.float64)  # C x H
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    sd = np.maximum(sd, 0.05)
    return (x - mu) / sd, (y - mu) / sd, mu, sd, y


def offdiag_relfro(A, B):
    """Mean absolute difference of off-diagonal correlations (robust to near-zero R_fut)."""
    C = A.shape[0]
    m = ~np.eye(C, dtype=bool)
    return float(np.abs((A - B)[m]).mean())


def topk_sets(R, k):
    M = np.abs(R).copy()
    np.fill_diagonal(M, -1)
    return np.argpartition(-M, k - 1, axis=1)[:, :k]


def jaccard(I, J):
    s = 0.0
    for a, b in zip(I, J):
        sa, sb = set(a.tolist()), set(b.tolist())
        s += len(sa & sb) / len(sa | sb)
    return s / len(I)


def main():
    name, L, H, K, s_tr, s_te = sys.argv[1], *map(int, sys.argv[2:7])
    t0 = time.time()
    ds = Dataset(name)
    X, C = ds.X, ds.C
    K = min(K, C - 1)
    tr_idx = ds.starts("train", L, H, s_tr)
    va_idx = ds.starts("val", L, H, s_te)
    te_idx = ds.starts("test", L, H, s_te)

    # ---- relation matrices estimated from training data only
    R_static = ds.train_corr()
    R_static = np.nan_to_num(R_static)
    ph_tr = ds.phase(tr_idx)
    # relation matrices use a stride that visits every hour-of-day (stride 1 for hourly data)
    rel_stride = 1 if ds.spd <= 24 else max(1, min(s_tr, ds.spd // 24))
    rel_idx = ds.starts("train", L, H, rel_stride)
    R_ph = np.zeros((48, C, C))
    n_ph = np.zeros(48)
    R_sw = np.zeros((C, C))
    for t, p in zip(rel_idx, ds.phase(rel_idx)):
        R = window_corr(X[t:t + H].astype(np.float64))
        R_ph[p] += R
        n_ph[p] += 1
        R_sw += R
    R_sw /= len(rel_idx)
    for p in range(48):
        R_ph[p] = R_ph[p] / n_ph[p] if n_ph[p] > 0 else R_sw
    A_static = topk_adj(R_static, K)
    A_sw = topk_adj(R_sw, K)
    A_ph = np.stack([topk_adj(R_ph[p], K) for p in range(48)])
    print(f"[{name}] relation matrices built in {time.time()-t0:.1f}s; phases with data: {(n_ph>0).sum()}")

    def adj_for(src, t, p):
        if src == "static":
            return A_static
        if src == "staticwin":
            return A_sw
        if src == "phase":
            return A_ph[p]
        if src == "back":
            return topk_adj(window_corr(X[t - L:t].astype(np.float64)), K)
        if src == "back_recent":
            return topk_adj(window_corr(X[t - H:t].astype(np.float64)), K)
        if src == "fut":
            return topk_adj(window_corr(X[t:t + H].astype(np.float64)), K)
        raise ValueError(src)

    def feats(src, t, p, xn):
        if src == "ci":
            return xn
        A = adj_for(src, t, p)
        return np.concatenate([xn, A @ xn], axis=1)

    # ---- ridge accumulation on train windows
    G = {s: 0.0 for s in SOURCES}
    B = {s: 0.0 for s in SOURCES}
    N = 0
    for t, p in zip(tr_idx, ph_tr):
        xn, yn, *_ = norm_win(X, t, L, H)
        for s in SOURCES:
            F = feats(s, t, p, xn)
            G[s] = G[s] + F.T @ F
            B[s] = B[s] + F.T @ yn
        N += C
    print(f"[{name}] ridge stats accumulated in {time.time()-t0:.1f}s ({len(tr_idx)} train windows)")

    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    Ws = {s: {} for s in SOURCES}
    for s in SOURCES:
        Gs, Bs = G[s] / N, B[s] / N
        d = Gs.shape[0]
        for lam in lams:
            Ws[s][lam] = np.linalg.solve(Gs + lam * np.eye(d), Bs)

    def evaluate(idx, collect_stats=False):
        ph = ds.phase(idx)
        se = {s: {lam: 0.0 for lam in lams} for s in SOURCES}
        ae = {s: {lam: 0.0 for lam in lams} for s in SOURCES}
        cnt = 0
        st = {k: [] for k in ["back", "back_recent", "static", "staticwin", "phase"]}
        jac = {k: [] for k in st}
        for t, p in zip(idx, ph):
            xn, yn, mu, sd, y = norm_win(X, t, L, H)
            for s in SOURCES:
                F = feats(s, t, p, xn)
                for lam in lams:
                    pred = (F @ Ws[s][lam]) * sd + mu
                    dlt = pred - y
                    se[s][lam] += float((dlt ** 2).sum())
                    ae[s][lam] += float(np.abs(dlt).sum())
            cnt += y.size
            if collect_stats:
                Rf = window_corr(X[t:t + H].astype(np.float64))
                cands = {"back": window_corr(X[t - L:t].astype(np.float64)),
                         "back_recent": window_corr(X[t - H:t].astype(np.float64)),
                         "static": R_static, "staticwin": R_sw, "phase": R_ph[p]}
                Sf = topk_sets(Rf, K)
                for k2, Rc in cands.items():
                    st[k2].append(offdiag_relfro(Rc, Rf))
                    jac[k2].append(jaccard(topk_sets(Rc, K), Sf))
        mse = {s: {lam: se[s][lam] / cnt for lam in lams} for s in SOURCES}
        mae = {s: {lam: ae[s][lam] / cnt for lam in lams} for s in SOURCES}
        return mse, mae, st, jac

    vmse, _, _, _ = evaluate(va_idx)
    best = {s: min(lams, key=lambda l: vmse[s][l]) for s in SOURCES}
    tmse, tmae, st, jac = evaluate(te_idx, collect_stats=True)
    res = {"dataset": name, "L": L, "H": H, "K": K, "C": C,
           "n_train_windows": int(len(tr_idx)), "n_test_windows": int(len(te_idx)),
           "ridge_test": {s: {"mse": tmse[s][best[s]], "mae": tmae[s][best[s]], "lambda": best[s]}
                          for s in SOURCES}}
    ci = res["ridge_test"]["ci"]["mse"]
    for s in SOURCES:
        res["ridge_test"][s]["gain_vs_ci_%"] = 100 * (ci - res["ridge_test"][s]["mse"]) / ci
    res["relation_stats"] = {}
    st_arr = {k: np.array(v) for k, v in st.items()}
    jac_arr = {k: np.array(v) for k, v in jac.items()}
    for k2 in st:
        res["relation_stats"][k2] = {
            "mad_to_Rfut_mean": float(st_arr[k2].mean()),
            "topk_jaccard_with_Rfut_mean": float(jac_arr[k2].mean())}
    res["relation_stats"]["phase_beats_back_frac"] = float((st_arr["phase"] < st_arr["back"]).mean())
    res["relation_stats"]["phase_beats_staticwin_frac"] = float((st_arr["phase"] < st_arr["staticwin"]).mean())
    res["seconds"] = time.time() - t0
    save_json(res, f"pilot1_{name}_L{L}_H{H}.json")
    print(f"[{name}] done in {res['seconds']:.0f}s")
    for s in SOURCES:
        r = res["ridge_test"][s]
        print(f"  {s:12s} mse={r['mse']:.4f} gain_vs_ci={r['gain_vs_ci_%']:+.2f}%")
    for k2 in st:
        r = res["relation_stats"][k2]
        print(f"  {k2:12s} MAD={r["mad_to_Rfut_mean"]:.3f} jacc={r['topk_jaccard_with_Rfut_mean']:.3f}")


if __name__ == "__main__":
    main()
