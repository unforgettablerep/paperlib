"""Train/evaluate variate-token forecasters on real benchmarks.

python train_real.py --data Electricity --data-dir DATA --models attn,bimamba,retrieve --seeds 0 1 2

Protocol: Adam, cosine learning-rate decay, early stopping on validation MSE (patience 3),
the best-validation checkpoint is evaluated on every test window. One line per run:
dataset, horizon, model, seed, val MSE, test MSE, test MAE, epochs, seconds.
"""
import argparse
import copy
import time
import torch
import torch.nn as nn
from data_real import load
from models_real import VariateForecaster

VARIANTS = {   # key: (vc layer, identity embedding, scan order)
    "none": ("none", False, None),
    "attn": ("attn", False, None),
    "attn+id": ("attn", True, None),
    "bimamba": ("bimamba", False, None),
    "bimamba+id": ("bimamba", True, None),
    "bimamba-shuffled": ("bimamba", False, "shuffle"),
    "dyntopk": ("dyntopk", False, None),
    "retrieve": ("retrieve", False, None),
    "retrieve+id": ("retrieve", True, None),
}


@torch.no_grad()
def evaluate(model, split, bs=128):
    model.eval()
    se = ae = 0.0
    starts = torch.arange(split.n)
    for s in starts.split(bs):
        x, y = split.batch(s)
        p = model(x)
        se += ((p - y) ** 2).sum().item()
        ae += (p - y).abs().sum().item()
    cnt = split.n * split.T * split.x.shape[1]
    return se / cnt, ae / cnt


def run(data, data_dir, key, seed, T=96, L=96, d=64, layers=2, k=16, lr=5e-4, bs=32,
        epochs=10, patience=3, max_steps=150):
    torch.manual_seed(seed)
    train, val, test = load(data, data_dir, L, T)
    n_vars = train.x.shape[1]
    vc, id_emb, order = VARIANTS[key]
    if order == "shuffle":
        order = torch.randperm(n_vars, generator=torch.Generator().manual_seed(1000 + seed))
    model = VariateForecaster(n_vars, L, T, vc=vc, d=d, layers=layers, d_ff=2 * d, k=k,
                              id_emb=id_emb, order=order)
    steps_per_epoch = train.n // bs if max_steps is None else min(train.n // bs, max_steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    best, best_state, bad, t0, ep_done = float("inf"), None, 0, time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(train.n)[: steps_per_epoch * bs].view(steps_per_epoch, bs)
        for idx in perm:
            x, y = train.batch(idx)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        ep_done = ep + 1
        v = evaluate(model, val)[0]
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    mse, mae = evaluate(model, test)
    return best, mse, mae, ep_done, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=150, help="steps per epoch (random windows)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for key in a.models.split(","):
        for s in a.seeds:
            v, mse, mae, ep, sec = run(a.data, a.data_dir, key, s, T=a.horizon, k=a.k,
                                       epochs=a.epochs, max_steps=a.max_steps)
            print(f"{a.data}\tT={a.horizon}\t{key}\tseed={s}\tval={v:.4f}\tmse={mse:.4f}"
                  f"\tmae={mae:.4f}\tepochs={ep}\t{sec:.0f}s", flush=True)
