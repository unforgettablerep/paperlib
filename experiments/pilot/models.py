"""Small forecasting models sharing one backbone so that only the channel-mixing block differs.

Backbone (iTransformer / S-Mamba style, one token per variable):
  RevIN -> Linear(L->D) per channel -> n_layers x [channel-mixing block + FFN] -> Linear(D->H)
Mixing blocks: ci (none), attn (dense self-attention), dense (static CxC matrix),
factor (r learned factor tokens, low-rank), sparse (attention restricted to a fixed top-k graph),
factor_sparse (both), mamba (selective SSM scanned over channels in a given order; uni/bi).
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixAttn(nn.Module):
    def __init__(self, D, heads=4, mask=None):
        super().__init__()
        self.att = nn.MultiheadAttention(D, heads, batch_first=True, dropout=0.0)
        self.register_buffer("mask", mask if mask is not None else torch.zeros(0), persistent=False)

    def forward(self, h):
        m = self.mask if self.mask.numel() else None
        out, _ = self.att(h, h, h, attn_mask=m, need_weights=False)
        return out


class MixDense(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(C, C))
        self.v = nn.Linear(D, D)

    def forward(self, h):
        return torch.einsum("ij,bjd->bid", self.W, self.v(h))


class MixFactor(nn.Module):
    """r learned queries pool the channels into r factor tokens; channels read them back."""

    def __init__(self, D, r=4, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(r, D) / math.sqrt(D))
        self.pool = nn.MultiheadAttention(D, heads, batch_first=True)
        self.read = nn.MultiheadAttention(D, heads, batch_first=True)

    def forward(self, h):
        q = self.q.unsqueeze(0).expand(h.shape[0], -1, -1)
        f, _ = self.pool(q, h, h, need_weights=False)
        out, _ = self.read(h, f, f, need_weights=False)
        return out


class MixFactorSparse(nn.Module):
    def __init__(self, D, mask, r=4, heads=4):
        super().__init__()
        self.factor = MixFactor(D, r, heads)
        self.sparse = MixAttn(D, heads, mask)

    def forward(self, h):
        return self.factor(h) + self.sparse(h)


class SelectiveSSM(nn.Module):
    """Minimal Mamba-1 block (pure PyTorch, sequential scan)."""

    def __init__(self, D, N=16, expand=2, conv=4):
        super().__init__()
        E = expand * D
        self.E, self.N = E, N
        self.R = max(1, math.ceil(D / 16))
        self.inp = nn.Linear(D, 2 * E)
        self.conv = nn.Conv1d(E, E, conv, groups=E, padding=conv - 1)
        self.xproj = nn.Linear(E, self.R + 2 * N, bias=False)
        self.dt = nn.Linear(self.R, E)
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(E, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.Dskip = nn.Parameter(torch.ones(E))
        self.out = nn.Linear(E, D)
        with torch.no_grad():  # dt init as in Mamba
            dt = torch.exp(torch.rand(E) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):  # x: B, S, D
        Bsz, S, _ = x.shape
        xz = self.inp(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :S].transpose(1, 2)
        u = F.silu(u)
        p = self.xproj(u)
        dt, Bm, Cm = torch.split(p, [self.R, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt(dt))  # B,S,E
        A = -torch.exp(self.A_log)  # E,N
        dBu = (dt * u).unsqueeze(-1) * Bm.unsqueeze(2)  # B,S,E,N
        la = (dt.unsqueeze(-1) * A).clamp_min(-60.0)  # log of the per-step decay, B,S,E,N
        y = self.scan(la, dBu, Cm) if not self.sequential else self.scan_seq(la, dBu, Cm)
        y = y + u * self.Dskip
        return self.out(y * F.silu(z))

    sequential = True  # sequential scan is faster on CPU; chunked float64 scan kept for reference

    @staticmethod
    def scan(la, dBu, Cm, chunk=8):
        """Chunked parallel scan in float64: h_t = exp(cum_t) * (h_0 + sum_{s<=t} exp(-cum_s) b_s)."""
        Bsz, S, E, N = dBu.shape
        h = torch.zeros(Bsz, E, N, dtype=torch.float64, device=dBu.device)
        ys = []
        for c0 in range(0, S, chunk):
            cum = torch.cumsum(la[:, c0:c0 + chunk].double(), dim=1)
            inner = torch.cumsum(torch.exp(-cum) * dBu[:, c0:c0 + chunk].double(), dim=1)
            hs = torch.exp(cum) * (h.unsqueeze(1) + inner)
            ys.append((hs * Cm[:, c0:c0 + chunk].double().unsqueeze(2)).sum(-1))
            h = hs[:, -1]
        return torch.cat(ys, 1).float()

    @staticmethod
    def scan_seq(la, dBu, Cm):
        Bsz, S, E, N = dBu.shape
        h = torch.zeros(Bsz, E, N, device=dBu.device)
        ys = []
        dA_s, dBu_s, C_s = torch.exp(la).unbind(1), dBu.unbind(1), Cm.unbind(1)
        for s in range(S):
            h = dA_s[s] * h + dBu_s[s]
            ys.append((h * C_s[s].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)


class MixMamba(nn.Module):
    def __init__(self, D, order, N=16, bidirectional=True):
        super().__init__()
        self.f = SelectiveSSM(D, N)
        self.b = SelectiveSSM(D, N) if bidirectional else None
        order = torch.as_tensor(order, dtype=torch.long)
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inv", torch.argsort(order), persistent=False)

    def forward(self, h):
        hp = h[:, self.order]
        out = self.f(hp)
        if self.b is not None:
            out = out + self.b(hp.flip(1)).flip(1)
        return out[:, self.inv]


class Block(nn.Module):
    def __init__(self, D, mixer, dropout=0.1):
        super().__init__()
        self.mixer = mixer
        self.n1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * D, D))
        self.n2 = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        if self.mixer is not None:
            h = self.n1(h + self.drop(self.mixer(h)))
        return self.n2(h + self.drop(self.ffn(h)))


class Forecaster(nn.Module):
    def __init__(self, L, H, C, D, mix="ci", layers=2, mask=None, order=None, N=16, bidirectional=True, r=4):
        super().__init__()
        self.emb = nn.Linear(L, D)
        blocks = []
        for _ in range(layers):
            if mix == "ci":
                m = None
            elif mix == "attn":
                m = MixAttn(D)
            elif mix == "dense":
                m = MixDense(C, D)
            elif mix == "factor":
                m = MixFactor(D, r)
            elif mix == "sparse":
                m = MixAttn(D, mask=mask)
            elif mix == "factor_sparse":
                m = MixFactorSparse(D, mask, r)
            elif mix == "mamba":
                m = MixMamba(D, order if order is not None else np.arange(C), N, bidirectional)
            else:
                raise ValueError(mix)
            blocks.append(Block(D, m))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(D, H)

    def forward(self, x):  # B, L, C
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True).clamp_min(1e-3)
        h = self.emb(((x - mu) / sd).transpose(1, 2))  # B, C, D
        for b in self.blocks:
            h = b(h)
        y = self.head(h).transpose(1, 2)  # B, H, C
        return y * sd + mu


def topk_mask(R, k):
    """Boolean attention mask for nn.MultiheadAttention: True = blocked. Keeps self + top-k |R|."""
    C = R.shape[0]
    k = min(k, C - 1)
    M = np.abs(np.nan_to_num(R)).copy()
    np.fill_diagonal(M, -1)
    idx = np.argpartition(-M, k - 1, axis=1)[:, :k]
    allow = np.zeros((C, C), dtype=bool)
    allow[np.repeat(np.arange(C), k), idx.ravel()] = True
    np.fill_diagonal(allow, True)
    return torch.as_tensor(~allow)


def windows(X, t, L, H):
    """X: torch (T, C); t: LongTensor (B,) -> x (B, L, C), y (B, H, C)."""
    ar = torch.arange(-L, H)
    idx = t.unsqueeze(1) + ar.unsqueeze(0)
    w = X[idx]
    return w[:, :L], w[:, L:]


@torch.no_grad()
def evaluate(model, X, idx, L, H, bs=64):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for i in range(0, len(idx), bs):
        t = torch.as_tensor(idx[i:i + bs])
        x, y = windows(X, t, L, H)
        p = model(x)
        se += float(((p - y) ** 2).sum())
        ae += float((p - y).abs().sum())
        n += y.numel()
    return se / n, ae / n


def train_eval(model, X, tr_idx, va_idx, te_idx, L, H, epochs=8, patience=2, lr=1e-3, bs=32, seed=0,
               max_minutes=None, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(X)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), bs):
            t = torch.as_tensor(perm[i:i + bs])
            x, y = windows(X, t, L, H)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, X, va_idx, L, H)[0]
        hist.append(v)
        if log:
            log(f"    ep{ep} val={v:.4f} ({time.time()-t0:.0f}s)")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        for g in opt.param_groups:
            g["lr"] *= 0.7
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            break
    model.load_state_dict(best_state)
    te = evaluate(model, X, te_idx, L, H)
    return {"val_mse": best, "test_mse": te[0], "test_mae": te[1], "epochs_run": len(hist),
            "seconds": time.time() - t0, "params": sum(p.numel() for p in model.parameters())}
