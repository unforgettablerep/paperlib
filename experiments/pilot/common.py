"""Shared utilities for the pilot experiments.

Data layout: raw downloads live under $PILOT_DATA (outside the repo); `prepare()`
converts them to <PILOT_DATA>/proc/<name>.npz with
  X (T, C) float32 raw values, ts (T,) int64 epoch-seconds (or empty), names (C,),
  dist (C, C) float32 road distances (PEMS only, inf = no edge), splits (2,) ints.
All experiments standardise with train statistics and report MSE/MAE on that scale.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

DATA = os.environ.get("PILOT_DATA", "")
PROC = os.path.join(DATA, "proc")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# steps per day, used for phase features
STEPS_PER_DAY = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "ECL": 24, "Traffic": 24,
                 "Solar": 144, "Weather": 144, "PEMS04": 288, "PEMS08": 288, "AirQuality": 24,
                 "Exchange": 1}


def _splits_ratio(T, r_train, r_test):
    n_train = int(T * r_train)
    n_test = int(T * r_test)
    return np.array([n_train, T - n_test], dtype=np.int64)


def _save(name, X, ts, names, splits, dist=None):
    os.makedirs(PROC, exist_ok=True)
    np.savez(os.path.join(PROC, name + ".npz"), X=X.astype(np.float32),
             ts=np.asarray(ts, dtype=np.int64), names=np.asarray(names),
             splits=np.asarray(splits, dtype=np.int64),
             dist=(dist.astype(np.float32) if dist is not None else np.zeros((0, 0), np.float32)))
    print(f"saved {name}: X={X.shape} splits={splits}")


def _epoch(index):
    # pandas >= 3 defaults to microsecond resolution, so convert the unit explicitly
    return pd.DatetimeIndex(index).as_unit("s").asi8.astype(np.int64)


def prepare():
    d = DATA
    # ETT (standard 12/4/4-month split)
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        df = pd.read_csv(os.path.join(d, "ett", name + ".csv"))
        ts = pd.to_datetime(df["date"])
        X = df.drop(columns=["date"]).to_numpy(np.float32)
        m = 1 if name.startswith("ETTh") else 4
        tr, va, te = 12 * 30 * 24 * m, 4 * 30 * 24 * m, 4 * 30 * 24 * m
        X, ts = X[: tr + va + te], ts[: tr + va + te]
        _save(name, X, _epoch(ts), list(df.columns[1:]), [tr, tr + va])
    # laiguokun collections (no timestamps in files; start times are documented)
    lai = {"ECL": ("electricity.txt.gz", "2012-01-01", "h"),
           "Traffic": ("traffic.txt.gz", "2015-01-01", "h"),
           "Solar": ("solar_AL.txt.gz", "2006-01-01", "10min"),
           "Exchange": ("exchange_rate.txt.gz", "1990-01-01", "D")}
    for name, (fn, start, freq) in lai.items():
        with gzip.open(os.path.join(d, "lai", fn), "rt") as f:
            X = np.loadtxt(f, delimiter=",", dtype=np.float32)
        ts = pd.date_range(start, periods=X.shape[0], freq=freq)
        _save(name, X, _epoch(ts), [f"{name}_{i}" for i in range(X.shape[1])],
              _splits_ratio(X.shape[0], 0.7, 0.2))
    # PEMS (traffic flow channel, 6:2:2 split, road-distance graph)
    for name, sub, start in [("PEMS04", "pems04", "2018-01-01"), ("PEMS08", "pems08", "2016-07-01")]:
        arr = np.load(os.path.join(d, sub, sub + ".npz"))["data"]
        X = arr[:, :, 0].astype(np.float32)
        C = X.shape[1]
        dist = np.full((C, C), np.inf, np.float32)
        np.fill_diagonal(dist, 0.0)
        e = pd.read_csv(os.path.join(d, sub, "distance.csv"))
        for a, b, c in e[["from", "to", "cost"]].to_numpy():
            a, b = int(a), int(b)
            if a < C and b < C:
                dist[a, b] = dist[b, a] = min(dist[a, b], c)
        ts = pd.date_range(start, periods=X.shape[0], freq="5min")
        _save(name, X, _epoch(ts), [f"{name}_{i}" for i in range(C)],
              _splits_ratio(X.shape[0], 0.6, 0.2), dist)
    # Weather (Jena roof station 2020, 21 variables, 10-min)
    parts = []
    for fn in ["mpi_roof_2020a.csv", "mpi_roof_2020b.csv"]:
        parts.append(pd.read_csv(os.path.join(d, "weather", "x", fn), encoding="latin1"))
    df = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    W = df.drop(columns=["Date Time"]).apply(pd.to_numeric, errors="coerce")
    W = W.mask(W <= -9998)  # station missing-value code
    n_missing = int(W.isna().sum().sum())
    W = W.interpolate(limit_direction="both")
    names = [c.strip() for c in W.columns]
    print("weather missing values interpolated:", n_missing)
    _save("Weather", W.to_numpy(np.float32), _epoch(ts), names, _splits_ratio(len(W), 0.7, 0.2))
    # Air Quality (UCI, hourly, -200 = missing; NMHC(GT) dropped, ~90% missing)
    df = pd.read_csv(os.path.join(d, "airq", "x", "AirQualityUCI.csv"), sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    A = df.drop(columns=["Date", "Time", "NMHC(GT)"]).apply(pd.to_numeric, errors="coerce")
    A = A.mask(A <= -199)
    print("airquality missing fraction per column:", A.isna().mean().round(3).to_dict())
    A = A.interpolate(limit_direction="both")
    _save("AirQuality", A.to_numpy(np.float32), _epoch(ts), list(A.columns),
          _splits_ratio(len(A), 0.7, 0.2))


class Dataset:
    def __init__(self, name):
        z = np.load(os.path.join(PROC, name + ".npz"), allow_pickle=False)
        self.name = name
        X = z["X"].astype(np.float64)
        self.splits = z["splits"]
        tr = self.splits[0]
        self.mu = X[:tr].mean(0)
        self.sd = X[:tr].std(0) + 1e-8
        self.X = ((X - self.mu) / self.sd).astype(np.float32)
        self.T, self.C = self.X.shape
        self.ts = z["ts"]
        self.names = [str(s) for s in z["names"]]
        self.dist = z["dist"] if z["dist"].size else None
        self.spd = STEPS_PER_DAY[name]

    def starts(self, split, L, H, stride=1):
        """Forecast start indices t: lookback X[t-L:t], target X[t:t+H]."""
        tr, va = self.splits
        lo, hi = {"train": (L, tr - H), "val": (tr, va - H), "test": (va, self.T - H)}[split]
        return np.arange(lo, hi + 1, stride)

    def phase(self, t):
        """Calendar phase of index t: hour-of-day (24) x weekend flag (2) -> 0..47."""
        tt = pd.to_datetime(np.asarray(self.ts)[np.asarray(t)], unit="s")
        return (tt.hour.to_numpy() + 24 * (tt.dayofweek.to_numpy() >= 5)).astype(np.int64)

    def train_corr(self):
        tr = self.splits[0]
        return np.corrcoef(self.X[:tr].T)


def window_corr(W):
    """Correlation of a window W (n, C); constant channels get zero correlation."""
    W = W - W.mean(0, keepdims=True)
    s = np.sqrt((W ** 2).sum(0))
    ok = s > 1e-6
    Z = np.zeros_like(W)
    Z[:, ok] = W[:, ok] / s[ok]
    R = Z.T @ Z
    np.fill_diagonal(R, 1.0)
    return R


def topk_adj(R, k, signed=True):
    """Row-normalised top-k (by |R|, self excluded) adjacency."""
    C = R.shape[0]
    k = min(k, C - 1)
    M = np.abs(R).copy()
    np.fill_diagonal(M, -1.0)
    idx = np.argpartition(-M, k - 1, axis=1)[:, :k]
    A = np.zeros_like(R)
    rows = np.repeat(np.arange(C), k)
    vals = R[rows, idx.ravel()] if signed else M[rows, idx.ravel()]
    A[rows, idx.ravel()] = vals
    norm = np.abs(A).sum(1, keepdims=True) + 1e-8
    return A / norm


def mse_mae(pred, true):
    d = pred - true
    return float(np.mean(d ** 2)), float(np.mean(np.abs(d)))


def save_json(obj, fname):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare()
