"""Pilot 2A (synthetic, ground truth known): common factors + sparse lagged residual graph.

x_t = Lambda f_t + e_t ; f = daily cycle + slow trend (shared by all channels)
e_t = 0.5 e_{t-1} + sum_{parents} b * e_{j,t-lag} + noise  (each node has 2 parents)
Variant "switch": parents differ between day (07-18h) and night regimes.

Questions:
 1. Which graph estimator recovers the true sparse structure? raw corr (what most MTSF
    papers use) vs factor-removed corr vs lagged/VAR scores on factor-removed residuals.
 2. Forecasting with per-channel ridge: CI, CI+factors, dense VAR, sparse graphs from the
    different estimators, factor+sparse, oracle parents; and static vs regime-specific graphs.
Usage: python pilot2_synth.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import save_json

C, P, r_true = 40, 24, 2
DAYS = 800
L, H, K = 48, 24, 4


def simulate(seed, switch=False):
    rng = np.random.default_rng(seed)
    T = P * DAYS
    t = np.arange(T)
    f1 = np.sin(2 * np.pi * t / P)
    n = np.zeros(T)
    for i in range(1, T):
        n[i] = 0.9 * n[i - 1] + 0.2 * rng.standard_normal()
    f1 = f1 + n
    f2 = np.zeros(T)
    for i in range(1, T):
        f2[i] = 0.98 * f2[i - 1] + 0.1 * rng.standard_normal()
    F = np.stack([f1, f2], 1)
    Lam = np.stack([rng.normal(1.0, 0.3, C), rng.normal(0.0, 0.7, C)], 1)

    def parents():
        par = {}
        for i in range(C):
            js = rng.choice([j for j in range(C) if j != i], 2, replace=False)
            par[i] = [(int(j), float(rng.uniform(0.25, 0.4) * rng.choice([-1, 1])), int(rng.integers(1, 3)))
                      for j in js]
        return par

    par_day = parents()
    par_night = parents() if switch else par_day
    hour = t % P
    day = (hour >= 7) & (hour < 19)
    e = np.zeros((T, C))
    eps = 0.5 * rng.standard_normal((T, C))
    for s in range(2, T):
        par = par_day if day[s] else par_night
        v = 0.5 * e[s - 1] + eps[s]
        for i in range(C):
            for j, b, lag in par[i]:
                v[i] += b * e[s - lag, j]
        e[s] = v
    X = F @ Lam.T + e
    return X, par_day, par_night, day


def truth_matrix(par):
    G = np.zeros((C, C), bool)
    for i, lst in par.items():
        for j, _, _ in lst:
            G[i, j] = True  # j -> i
    return G


def remove_factors(Xtr, r):
    U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    V = Vt[:r].T
    return V


def residual(X, V):
    return X - (X @ V) @ V.T


def lagged_scores(E, maxlag=3):
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    S = np.zeros((C, C))
    n = len(E)
    for lag in range(1, maxlag + 1):
        M = np.abs(E[lag:].T @ E[:-lag] / (n - lag))  # [i, j] = corr(e_i(t), e_j(t-lag))
        S = np.maximum(S, M)
    np.fill_diagonal(S, 0)
    return S


def var_scores(E, maxlag=3, lam=1.0):
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    Z = np.concatenate([E[maxlag - l:len(E) - l] for l in range(1, maxlag + 1)], 1)
    Y = E[maxlag:]
    W = np.linalg.solve(Z.T @ Z / len(Z) + lam * 1e-2 * np.eye(Z.shape[1]), Z.T @ Y / len(Z))
    S = np.zeros((C, C))
    for l in range(maxlag):
        S += np.abs(W[l * C:(l + 1) * C].T)  # [i, j]: j's lag -> i
    np.fill_diagonal(S, 0)
    return S


def eval_structure(S, G):
    m = ~np.eye(C, dtype=bool)
    Gu = (G | G.T)[m]
    Su = np.maximum(S, S.T)[m]
    return {"auroc": float(roc_auc_score(Gu, Su)), "ap": float(average_precision_score(Gu, Su)),
            "base_rate": float(Gu.mean())}


def topk_from_scores(S, k):
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]


def lagmat(x, idx):
    """x (T,), idx (n,) forecast starts -> (n, L) lookback."""
    return np.stack([x[t - L:t] for t in idx])


def ridge_forecast(X, Fac, nbrs, tr, te, use_factor, dense=False, lam=1e-1, mask_tr=None, mask_te=None,
                   nbrs_alt=None):
    """Per-channel ridge. nbrs: (C,k) neighbour ids (or None). nbrs_alt: regime-specific neighbours for
    windows where mask is False (regime switching)."""
    se, n = 0.0, 0
    Ytr = np.stack([X[t:t + H] for t in tr])  # n, H, C
    Yte = np.stack([X[t:t + H] for t in te])
    lag_all_tr = np.stack([X[t - L:t] for t in tr])  # n, L, C
    lag_all_te = np.stack([X[t - L:t] for t in te])
    f_tr = np.stack([Fac[t - L:t].ravel() for t in tr]) if use_factor else None
    f_te = np.stack([Fac[t - L:t].ravel() for t in te]) if use_factor else None
    if dense:  # all channels' lags: identical design for every target -> one multi-output solve
        Ftr = lag_all_tr.reshape(len(tr), -1)
        Fte = lag_all_te.reshape(len(te), -1)
        if use_factor:
            Ftr, Fte = np.concatenate([Ftr, f_tr], 1), np.concatenate([Fte, f_te], 1)
        G = Ftr.T @ Ftr / len(Ftr)
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]) * np.trace(G) / G.shape[0],
                            Ftr.T @ Ytr.reshape(len(tr), -1) / len(Ftr))
        P_ = (Fte @ W).reshape(Yte.shape)
        return float(((P_ - Yte) ** 2).mean())
    for c in range(C):
        def feats(lag_all, f, mask):
            parts = [lag_all[:, :, c]]
            if dense:
                parts.append(lag_all[:, :, [j for j in range(C) if j != c]].reshape(len(lag_all), -1))
            elif nbrs is not None:
                if nbrs_alt is None:
                    parts.append(lag_all[:, :, nbrs[c]].reshape(len(lag_all), -1))
                else:
                    a = lag_all[:, :, nbrs[c]].reshape(len(lag_all), -1)
                    b = lag_all[:, :, nbrs_alt[c]].reshape(len(lag_all), -1)
                    parts.append(np.where(mask[:, None], a, 0.0))
                    parts.append(np.where(mask[:, None], 0.0, b))
            if use_factor:
                parts.append(f)
            return np.concatenate(parts, 1)

        Ftr = feats(lag_all_tr, f_tr if use_factor else None, mask_tr)
        Fte = feats(lag_all_te, f_te if use_factor else None, mask_te)
        G = Ftr.T @ Ftr / len(Ftr)
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]) * np.trace(G) / G.shape[0], Ftr.T @ Ytr[:, :, c] / len(Ftr))
        P_ = Fte @ W
        se += float(((P_ - Yte[:, :, c]) ** 2).sum())
        n += P_.size
    return se / n


def run(seed, switch):
    X, par_d, par_n, dayflag = simulate(seed, switch)
    T = len(X)
    tr_end, va_end = int(0.7 * T), int(0.8 * T)
    mu, sd = X[:tr_end].mean(0), X[:tr_end].std(0)
    X = (X - mu) / sd
    Xtr = X[:tr_end]
    V = remove_factors(Xtr, r_true)
    E_tr = residual(Xtr, V)
    Fac = X @ V  # factor scores (T, r)
    Gd, Gn = truth_matrix(par_d), truth_matrix(par_n)
    G_any = Gd | Gn
    out = {"seed": seed, "switch": switch}
    pc = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False) ** 2
    out["pc_var_explained_top2"] = float(pc[:2].sum() / pc.sum())
    est = {"raw_corr": np.abs(np.corrcoef(Xtr.T)),
           "factor_removed_corr": np.abs(np.corrcoef(E_tr.T)),
           "factor_removed_lagged": lagged_scores(E_tr),
           "factor_removed_var": var_scores(E_tr),
           "raw_var": var_scores(Xtr)}
    for k in est:
        np.fill_diagonal(est[k], 0)
    out["structure"] = {k: eval_structure(S, G_any) for k, S in est.items()}
    if switch:  # regime-specific estimation on factor-removed residuals
        day_tr = dayflag[:tr_end]
        S_day = var_scores(np.where(day_tr[:, None], E_tr, 0.0))
        S_night = var_scores(np.where(day_tr[:, None], 0.0, E_tr))
        out["structure_regime"] = {"static_vs_day_truth": eval_structure(est["factor_removed_var"], Gd),
                                   "day_specific_vs_day_truth": eval_structure(S_day, Gd),
                                   "static_vs_night_truth": eval_structure(est["factor_removed_var"], Gn),
                                   "night_specific_vs_night_truth": eval_structure(S_night, Gn)}
    tr = np.arange(L, tr_end - H, 2)
    te = np.arange(va_end, T - H, 3)
    nb = {k: topk_from_scores(S, K) for k, S in est.items()}
    oracle = np.array([[j for j, _, _ in par_d[i]] + [j for j, _, _ in par_d[i]] for i in range(C)])[:, :K]
    fc = {}
    fc["CI"] = ridge_forecast(X, Fac, None, tr, te, False)
    fc["CI+factor"] = ridge_forecast(X, Fac, None, tr, te, True)
    fc["dense_VAR"] = ridge_forecast(X, Fac, None, tr, te, False, dense=True)
    fc["dense_VAR+factor"] = ridge_forecast(X, Fac, None, tr, te, True, dense=True)
    fc["sparse(raw_corr)"] = ridge_forecast(X, Fac, nb["raw_corr"], tr, te, False)
    fc["factor+sparse(raw_corr)"] = ridge_forecast(X, Fac, nb["raw_corr"], tr, te, True)
    fc["factor+sparse(resid_var)"] = ridge_forecast(X, Fac, nb["factor_removed_var"], tr, te, True)
    fc["factor+oracle_parents"] = ridge_forecast(X, Fac, oracle, tr, te, True)
    if switch:
        day_tr = dayflag[:tr_end]
        nb_d = topk_from_scores(var_scores(np.where(day_tr[:, None], E_tr, 0.0)), K)
        nb_n = topk_from_scores(var_scores(np.where(day_tr[:, None], 0.0, E_tr)), K)
        # regime of the forecast window start
        m_tr, m_te = dayflag[tr], dayflag[te]
        fc["factor+sparse(regime-specific)"] = ridge_forecast(X, Fac, nb_d, tr, te, True, mask_tr=m_tr,
                                                              mask_te=m_te, nbrs_alt=nb_n)
    base = fc["CI"]
    out["forecast_mse"] = fc
    out["forecast_gain_vs_CI_%"] = {k: 100 * (base - v) / base for k, v in fc.items()}
    return out


def main():
    res = {"config": {"C": C, "period": P, "days": DAYS, "L": L, "H": H, "K": K}, "runs": []}
    for switch in [False, True]:
        for seed in [0, 1, 2]:
            o = run(seed, switch)
            res["runs"].append(o)
            print(f"switch={switch} seed={seed} pc2={o['pc_var_explained_top2']:.2f}")
            for k, v in o["structure"].items():
                print(f"   {k:24s} AUROC={v['auroc']:.3f} AP={v['ap']:.3f} (base {v['base_rate']:.3f})")
            for k, v in o["forecast_gain_vs_CI_%"].items():
                print(f"   {k:32s} mse={o['forecast_mse'][k]:.4f} gain={v:+.2f}%")
            if switch:
                for k, v in o["structure_regime"].items():
                    print(f"   {k:32s} AUROC={v['auroc']:.3f} AP={v['ap']:.3f}")
    save_json(res, "pilot2_synthetic.json")


if __name__ == "__main__":
    main()
