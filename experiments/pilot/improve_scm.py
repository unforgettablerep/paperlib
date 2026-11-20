"""Improvement experiment: Static-Calendar Mixing (SCM) as a drop-in replacement for the
dynamic attention channel mixer of iTransformer-style forecasters.

Backbone (shared by every variant): RevIN -> Linear(L->D) per variable token -> 2 x [mixer + FFN]
-> Linear(D->H). Variants differ only in the channel mixer and the calendar inputs:
  ci            no mixing
  ci+time       no mixing, calendar embedding (hour-of-day + day-of-week) on every token
  attn          multi-head self-attention over variable tokens (iTransformer-like)
  attn+time     attention + calendar embedding (fair iTransformer analogue: it uses time features)
  dense+time    static learned CxC mixing (input-independent, zero-initialised) + calendar embedding
  scm           dense+time + calendar-conditioned low-rank modulation W_g = W + A_g B_g^T   <- proposed
  scm+drop      scm with DropConnect on W during training (regulariser for small/weakly-coupled data)
Calendar group g for the modulation: 6 four-hour blocks x weekday/weekend (12 groups).
Usage: python improve_scm.py DATASET H TRAIN_STRIDE EVAL_STRIDE [variants]   (env PILOT_SEED for seeds)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Dataset, save_json

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
L, D, NG, RANK = 96, 64, 12, 8
SEED = int(os.environ.get("PILOT_SEED", "0"))
ALL = ["ci", "ci+time", "attn", "attn+time", "dense+time", "scm", "scm+drop"]


class Mixer(nn.Module):
    def __init__(self, C, kind, drop_w=0.0):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.att = nn.MultiheadAttention(D, 4, batch_first=True)
            return
        self.W = nn.Parameter(torch.zeros(C, C))
        self.v = nn.Linear(D, D)
        self.drop_w = drop_w
        if kind == "mod":
            self.A = nn.Parameter(torch.zeros(NG, C, RANK))
            self.B = nn.Parameter(torch.randn(NG, C, RANK) * 0.02)

    def forward(self, h, g):
        if self.kind == "attn":
            return self.att(h, h, h, need_weights=False)[0]
        v = self.v(h)
        W = self.W
        if self.training and self.drop_w > 0:
            W = F.dropout(W, p=self.drop_w)  # DropConnect on the static mixing matrix
        out = torch.einsum("ij,bjd->bid", W, v)
        if self.kind == "mod":
            out = out + torch.einsum("bir,bjr,bjd->bid", self.A[g], self.B[g], v)
        return out


class Net(nn.Module):
    def __init__(self, C, H, variant):
        super().__init__()
        self.time = variant != "ci" and variant != "attn"
        kind = None if variant.startswith("ci") else "attn" if variant.startswith("attn") else \
            "dense" if variant == "dense+time" else "mod"
        self.emb = nn.Linear(L, D)
        if self.time:
            self.tod = nn.Embedding(24, D)
            self.dow = nn.Embedding(7, D)
            # zero-init: the model starts exactly as its no-calendar counterpart and only learns to use the
            # calendar when it helps (default N(0,1) init hurt ECL badly in a first run)
            nn.init.zeros_(self.tod.weight)
            nn.init.zeros_(self.dow.weight)
        self.mix = nn.ModuleList([Mixer(C, kind, 0.3 if variant == "scm+drop" else 0.0) if kind else None
                                  for _ in range(2)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * D, D))
                                  for _ in range(2)])
        self.n1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.n2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.head = nn.Linear(D, H)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, cal):
        hour, dow, g = cal
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True).clamp_min(1e-3)
        h = self.emb(((x - mu) / sd).transpose(1, 2))
        if self.time:
            h = h + (self.tod(hour) + self.dow(dow)).unsqueeze(1)
        for i in range(2):
            if self.mix[i] is not None:
                h = self.n1[i](h + self.drop(self.mix[i](h, g)))
            h = self.n2[i](h + self.drop(self.ffn[i](h)))
        return self.head(h).transpose(1, 2) * sd + mu


def main():
    name, H, s_tr, s_ev = sys.argv[1], *map(int, sys.argv[2:5])
    variants = sys.argv[5].split(",") if len(sys.argv) > 5 else ALL
    ds = Dataset(name)
    X = torch.as_tensor(ds.X)
    tr, va, te = (ds.starts(s, L, H, st) for s, st in [("train", s_tr), ("val", s_ev), ("test", s_ev)])
    tt_all = pd.to_datetime(ds.ts, unit="s")
    hour_all = torch.as_tensor(tt_all.hour.to_numpy(), dtype=torch.long)
    dow_all = torch.as_tensor(tt_all.day_of_week.to_numpy(), dtype=torch.long)
    ar = torch.arange(-L, H)

    def batch(t):
        t = torch.as_tensor(t)
        w = X[t.unsqueeze(1) + ar.unsqueeze(0)]
        hr, dw = hour_all[t], dow_all[t]
        g = hr // 4 + 6 * (dw >= 5).long()
        return w[:, :L], w[:, L:], (hr, dw, g)

    @torch.no_grad()
    def evaluate(model, idx):
        model.eval()
        se = ae = n = 0.0
        for i in range(0, len(idx), 64):
            x, y, cal = batch(idx[i:i + 64])
            p = model(x, cal)
            se += float(((p - y) ** 2).sum())
            ae += float((p - y).abs().sum())
            n += y.numel()
        return se / n, ae / n

    fname = f"improve_{name}_H{H}" + (f"_seed{SEED}" if SEED else "") + ".json"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", fname)
    res = {"dataset": name, "C": ds.C, "L": L, "H": H, "seed": SEED, "train_stride": s_tr, "eval_stride": s_ev,
           "runs": json.load(open(path))["runs"] if os.path.exists(path) else {}}
    for v in variants:
        if v in res["runs"]:
            continue
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = Net(ds.C, H, v)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, state, bad, t0 = 1e9, None, 0, time.time()
        for ep in range(8):
            model.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), 32):
                x, y, cal = batch(perm[i:i + 32])
                loss = F.mse_loss(model(x, cal), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vm = evaluate(model, va)[0]
            print(f"  {v} ep{ep} val={vm:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if vm < best - 1e-5:
                best, bad = vm, 0
                state = {k: p.detach().clone() for k, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
            for gr in opt.param_groups:
                gr["lr"] *= 0.7
        model.load_state_dict(state)
        mse, mae = evaluate(model, te)
        res["runs"][v] = {"val_mse": best, "test_mse": mse, "test_mae": mae, "seconds": time.time() - t0,
                          "params": sum(p.numel() for p in model.parameters())}
        print(f"[{name}] {v:11s} test_mse={mse:.4f} test_mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
        save_json(res, fname)


if __name__ == "__main__":
    main()
