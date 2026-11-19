"""Pilot 4: is there headroom for *per-sample adaptive* test-time compute in coarse-to-fine forecasting?

Mini SCALER-like model without an LLM (channel independent): an MLP encoder gives a context
vector; step 1 predicts a coarse shape at H/8, steps 2..4 refine residuals at H/4, H/2, H
conditioned on the context and the upsampled previous prediction. Trained with multi-scale MSE.
At test time we may stop after step k (k = 1..4) and linearly upsample.
Measures:
  * test MSE of fixed-k stopping, and refinement FLOPs per step
  * how concentrated the benefit of refinement is across samples
  * oracle per-sample stopping (upper bound) and a *learned* halting policy (regressor trained on
    the validation split to predict the gain of continuing), swept over thresholds, vs fixed-k.
Usage: python pilot4_adaptive.py DATASET L H TRAIN_STRIDE EVAL_STRIDE
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Dataset, save_json

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
D, DH = 128, 256


def up(y, n):
    return F.interpolate(y.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)


def down(y, n):
    return F.adaptive_avg_pool1d(y.unsqueeze(1), n).squeeze(1)


class C2F(nn.Module):
    def __init__(self, L, H):
        super().__init__()
        self.scales = [H // 8, H // 4, H // 2, H]
        self.enc = nn.Sequential(nn.Linear(L, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.steps = nn.ModuleList()
        for i, s in enumerate(self.scales):
            inp = D if i == 0 else D + s
            self.steps.append(nn.Sequential(nn.Linear(inp, DH), nn.GELU(), nn.Linear(DH, s)))

    def forward(self, x):  # x: (n, L) normalised
        c = self.enc(x)
        outs = []
        prev = None
        for i, s in enumerate(self.scales):
            if i == 0:
                y = self.steps[0](c)
            else:
                base = up(prev, s)
                y = base + self.steps[i](torch.cat([c, base], -1))
            outs.append(y)
            prev = y
        return c, outs

    def step_flops(self):
        return [sum(m.weight.numel() for m in st if isinstance(m, nn.Linear)) for st in self.steps]


def series_batch(X, t, L, H):
    ar = torch.arange(-L, H)
    w = X[t.unsqueeze(1) + ar.unsqueeze(0)]  # B, L+H, C
    w = w.permute(0, 2, 1).reshape(-1, L + H)  # B*C, L+H
    x, y = w[:, :L], w[:, L:]
    mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True).clamp_min(1e-3)
    return (x - mu) / sd, (y - mu) / sd, mu, sd, y


def main():
    name, L, H, s_tr, s_ev = sys.argv[1], *map(int, sys.argv[2:6])
    t0 = time.time()
    torch.manual_seed(0)
    ds = Dataset(name)
    X = torch.as_tensor(ds.X)
    tr, va, te = (ds.starts(s, L, H, st) for s, st in [("train", s_tr), ("val", s_ev), ("test", s_ev)])
    model = C2F(L, H)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    bs = max(1, 256 // ds.C) if ds.C < 256 else 1

    def run_eval(idx):
        model.eval()
        errs, ctxs, preds = [], [], []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                t = torch.as_tensor(idx[i:i + 64])
                xn, yn, mu, sd, y = series_batch(X, t, L, H)
                c, outs = model(xn)
                e = torch.stack([(((up(o, H) * sd + mu) - y) ** 2).mean(1) for o in outs], 1)  # n, 4
                errs.append(e)
                ctxs.append(torch.cat([c] + [up(o, H) for o in outs[:3]], 1))
        return torch.cat(errs).numpy(), torch.cat(ctxs).numpy()

    best, best_state, bad = 1e9, None, 0
    for ep in range(12):
        model.train()
        for i in range(0, len(tr), bs):
            t = torch.as_tensor(tr[rng.integers(0, len(tr), bs)])
            xn, yn, *_ = series_batch(X, t, L, H)
            _, outs = model(xn)
            loss = sum(F.mse_loss(o, down(yn, o.shape[1])) for o in outs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ve = run_eval(va)[0][:, 3].mean()
        print(f"  ep{ep} val_final={ve:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ve < best - 1e-5:
            best, bad = ve, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    Ev, Zv = run_eval(va)
    Et, Zt = run_eval(te)
    flops = model.step_flops()
    enc = sum(m.weight.numel() for m in model.enc if isinstance(m, nn.Linear))
    cum = np.cumsum(flops) + enc
    fixed = {f"k={k+1}": {"mse": float(Et[:, k].mean()), "rel_flops": float(cum[k] / cum[-1])} for k in range(4)}
    gain = Et[:, 0] - Et[:, 3]  # benefit of refining from coarse to full
    pos = np.sort(np.clip(gain, 0, None))[::-1]
    share_top20 = float(pos[: max(1, len(pos) // 5)].sum() / max(pos.sum(), 1e-12))
    frac_hurt = float((gain < 0).mean())
    oracle_k = Et.argmin(1)
    oracle = {"mse": float(Et.min(1).mean()), "mean_steps": float((oracle_k + 1).mean()),
              "rel_flops": float(cum[oracle_k].mean() / cum[-1])}
    # learned halting: ridge on [context, upsampled partial predictions] -> gain of continuing to k=4
    Zs = (Zv - Zv.mean(0)) / (Zv.std(0) + 1e-6)
    Zts = (Zt - Zv.mean(0)) / (Zv.std(0) + 1e-6)
    Zs = np.concatenate([Zs, np.ones((len(Zs), 1))], 1)
    Zts = np.concatenate([Zts, np.ones((len(Zts), 1))], 1)
    pred_gain = []
    for k in range(3):
        target = Ev[:, k] - Ev[:, 3]
        w = np.linalg.solve(Zs.T @ Zs / len(Zs) + 1e-1 * np.eye(Zs.shape[1]), Zs.T @ target / len(Zs))
        pred_gain.append(Zts @ w)
    pred_gain = np.stack(pred_gain, 1)  # n, 3
    curve = []
    for tau in np.quantile(pred_gain[:, 0], np.linspace(0.0, 1.0, 21)):
        k_stop = np.full(len(Et), 3)
        for k in range(3):
            stop = (pred_gain[:, k] < tau) & (k_stop == 3)
            k_stop[stop] = k
        mse = Et[np.arange(len(Et)), k_stop].mean()
        curve.append({"tau": float(tau), "mse": float(mse), "rel_flops": float(cum[k_stop].mean() / cum[-1]),
                      "mean_steps": float((k_stop + 1).mean())})
    # random halting with matched compute, for reference
    rnd = []
    for p in [0.25, 0.5, 0.75]:
        ks = np.where(np.random.default_rng(1).random(len(Et)) < p, 0, 3)
        rnd.append({"p_stop_at_1": p, "mse": float(Et[np.arange(len(Et)), ks].mean()),
                    "rel_flops": float(cum[ks].mean() / cum[-1])})
    res = {"dataset": name, "L": L, "H": H, "n_test_series": int(len(Et)), "fixed_k": fixed,
           "refinement_gain": {"mean": float(gain.mean()), "share_of_total_gain_from_top20pct": share_top20,
                               "frac_samples_hurt_by_refinement": frac_hurt},
           "oracle_per_sample": oracle, "learned_halting_curve": curve, "random_halting": rnd,
           "seconds": time.time() - t0}
    save_json(res, f"pilot4_{name}_L{L}_H{H}.json")
    print(name, "fixed:", {k: round(v["mse"], 4) for k, v in fixed.items()}, "oracle:", round(oracle["mse"], 4),
          "top20 share:", round(share_top20, 3), "hurt frac:", round(frac_hurt, 3))
    full = fixed["k=4"]["mse"]
    for c in curve:
        if c["mse"] <= full * 1.005:
            print(f"  learned halting within +0.5% of k=4 at rel_flops={c['rel_flops']:.3f} (steps {c['mean_steps']:.2f})")
            break


if __name__ == "__main__":
    main()
