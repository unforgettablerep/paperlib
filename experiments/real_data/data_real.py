"""Benchmark loaders with the standard long-term forecasting splits (as in the Time-Series-Library).

ETTh1/ETTm1: 12/4/4 months for train/val/test. Weather, Electricity, Traffic: 70/10/20.
Values are standardised with statistics of the training split. Every window of the test
split is evaluated (no batches are dropped).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

FILES = {"ETTh1": "ETTh1.csv", "ETTm1": "ETTm1.csv", "Weather": "weather.csv",
         "Electricity": "electricity.csv", "Traffic": "traffic.csv"}


def borders(name, n, L):
    if name == "ETTh1":
        a, b, c = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
        return [0, a - L, b - L], [a, b, c]
    if name == "ETTm1":
        a, b, c = 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4
        return [0, a - L, b - L], [a, b, c]
    n_train, n_test = int(n * 0.7), int(n * 0.2)
    n_val = n - n_train - n_test
    return [0, n_train - L, n - n_test - L], [n_train, n_train + n_val, n]


class Split:
    """Holds one split as a (time, N) tensor and serves (lookback, horizon) windows."""

    def __init__(self, values, L, T):
        self.x, self.L, self.T = values, L, T
        self.n = len(values) - L - T + 1

    def batch(self, starts):
        idx = starts[:, None] + torch.arange(self.L + self.T)[None]
        w = self.x[idx]                                         # (b, L+T, N)
        return w[:, :self.L], w[:, self.L:]


def load(name, data_dir, L=96, T=96):
    df = pd.read_csv(Path(data_dir) / FILES[name])
    values = df.drop(columns=["date"]).values.astype(np.float32)
    b1, b2 = borders(name, len(values), L)
    train = values[b1[0]:b2[0]]
    mean, std = train.mean(0), train.std(0) + 1e-8
    values = torch.from_numpy((values - mean) / std)
    return [Split(values[s:e], L, T) for s, e in zip(b1, b2)]   # train, val, test
