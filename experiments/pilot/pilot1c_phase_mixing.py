"""Pilot 1c (neural check of the revised direction 1'): does a phase-conditioned channel-mixing matrix
beat a static one once the model already knows the calendar phase?

Backbone as in pilot 2 (RevIN, Linear(L->D) per channel, 2 x [mixing + FFN], head). Variants:
  ci+emb          no channel mixing, phase embedding added to every channel token
  dense           static CxC mixing, no phase information (pilot-2 'dense')
  dense+emb       static CxC mixing + phase embedding          (fair control)
  phasedense+emb  W_g = W + A_g B_g^T (rank r, one per phase group) + phase embedding
Phase group g of the forecast start: 6 four-hour blocks x weekday/weekend = 12.
Usage: python pilot1c_phase_mixing.py DATASET H TRAIN_STRIDE EVAL_STRIDE [variants]
"""
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
L, D, G, RANK = 96, 64, 12, 8


class Mix(nn.Module):
    def __init__(self, C, phase_specific):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(C, C))
        self.v = nn.Linear(D, D)
        self.ps = phase_specific
        if phase_specific:
            self.A = nn.Parameter(torch.zeros(G, C, RANK))
            self.B = nn.Parameter(torch.randn(G, C, RANK) * 0.02)

    def forward(self, h, g):
        v = self.v(h)
        out = torch.einsum("ij,bjd->bid", self.W, v)
        if self.ps:
            a, b = self.A[g], self.B[g]  # B,C,r
            out = out + torch.einsum("bir,bjr,bjd->bid", a, b, v)
        return out


class Net(nn.Module):
    def __init__(self, C, H, mix, emb):
        super().__init__()
        self.emb = nn.Linear(L, D)
        self.pe = nn.Embedding(G, D) if emb else None
        self.mix = nn.ModuleList([Mix(C, mix == "phase") if mix else None for _ in range(2)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * D, D))
                                  for _ in range(2)])
        self.n1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.n2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.head = nn.Linear(D, H)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, g):
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True).clamp_min(1e-3)
        h = self.emb(((x - mu) / sd).transpose(1, 2))
        if self.pe is not None:
            h = h + self.pe(g).unsqueeze(1)
        for i in range(2):
            if self.mix[i] is not None:
                h = self.n1[i](h + self.drop(self.mix[i](h, g)))
            h = self.n2[i](h + self.drop(self.ffn[i](h)))
        return self.head(h).transpose(1, 2) * sd + mu


def main():
    name, H, s_tr, s_ev = sys.argv[1], *map(int, sys.argv[2:5])
    variants = sys.argv[5].split(",") if len(sys.argv) > 5 else ["ci+emb", "dense", "dense+emb", "phasedense+emb"]
    ds = Dataset(name)
    X = torch.as_tensor(ds.X)
    tr, va, te = (ds.starts(s, L, H, st) for s, st in [("train", s_tr), ("val", s_ev), ("test", s_ev)])

    def grp(t):
        tt = pd.to_datetime(ds.ts[np.asarray(t)], unit="s")
        return torch.as_tensor(tt.hour.to_numpy() // 4 + 6 * (tt.dayofweek.to_numpy() >= 5), dtype=torch.long)

    ar = torch.arange(-L, H)

    def batch(t):
        t = torch.as_tensor(t)
        w = X[t.unsqueeze(1) + ar.unsqueeze(0)]
        return w[:, :L], w[:, L:], grp(t.numpy())

    @torch.no_grad()
    def evaluate(model, idx):
        model.eval()
        se = ae = n = 0.0
        for i in range(0, len(idx), 64):
            x, y, g = batch(idx[i:i + 64])
            p = model(x, g)
            se += float(((p - y) ** 2).sum())
            ae += float((p - y).abs().sum())
            n += y.numel()
        return se / n, ae / n

    fname = f"pilot1c_{name}_H{H}.json"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", fname)
    res = {"dataset": name, "L": L, "H": H, "phase_groups": G, "rank": RANK, "runs": {}}
    if os.path.exists(path):
        import json
        res["runs"] = json.load(open(path)).get("runs", {})
    for v in variants:
        if v in res["runs"]:
            continue
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        mix = None if v.startswith("ci") else ("phase" if v.startswith("phase") else "static")
        model = Net(ds.C, H, mix, emb=v.endswith("+emb"))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, state, bad, t0 = 1e9, None, 0, time.time()
        for ep in range(8):
            model.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), 32):
                x, y, g = batch(perm[i:i + 32])
                loss = F.mse_loss(model(x, g), y)
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
        res["runs"][v] = {"val_mse": best, "test_mse": mse, "test_mae": mae, "seconds": time.time() - t0}
        print(f"[{name}] {v:16s} test_mse={mse:.4f} test_mae={mae:.4f}", flush=True)
        save_json(res, fname)


if __name__ == "__main__":
    main()
