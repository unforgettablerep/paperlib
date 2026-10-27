"""Variate-token forecasters (iTransformer / S-Mamba skeleton) with interchangeable
variable-correlation (VC) layers.

Skeleton: instance norm -> Linear(L -> D) per variable (one token per variable)
          [+ learnable variable identity] -> E x [x = LN(x + VC(x)); x = LN(x + FFN(x))]
          -> Linear(D -> T) -> de-normalise.
VC layers: none (channel independent), attention (iTransformer), bi-Mamba (S-Mamba),
           dynamic top-K over content scores (MS-FLOW style, O(N^2)),
           retrieve-then-attend (top-k candidates from learned embeddings, O(N k) per sample).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoVC(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class AttnVC(nn.Module):
    """iTransformer: multi-head self-attention over variable tokens."""

    def __init__(self, d, heads=8, dropout=0.1):
        super().__init__()
        self.h, self.p = heads, dropout
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Mamba(nn.Module):
    """Mamba (S6) block as configured in S-Mamba (expand=1, d_conv=2). The selective scan
    h_t = exp(dA_t) * h_{t-1} + dBx_t is computed for all chunks of length `chunk` at once in
    log space (h_t = e^{S_t} * cumsum(e^{-S_s} dBx_s)), then the chunks are chained through
    the carried state. Per-step log-decays are clamped at -10 (a factor below 5e-5), which
    keeps e^{-S} finite in float32; otherwise it equals the sequential recurrence."""

    def __init__(self, d, d_state=16, d_conv=2, chunk=4):
        super().__init__()
        self.ds, self.chunk, self.r = d_state, chunk, math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)
        self.x_proj = nn.Linear(d, self.r + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.r, d)
        dt = torch.exp(torch.rand(d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d, 1))
        self.D = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Linear(d, d, bias=False)

    def scan(self, dA, dBx, C):                              # (b,L,d,s), (b,L,d,s), (b,L,s)
        b, L, d, s = dA.shape
        l = self.chunk
        pad = (-L) % l
        if pad:                                               # zero input, no decay
            dA, dBx = F.pad(dA, (0, 0, 0, 0, 0, pad)), F.pad(dBx, (0, 0, 0, 0, 0, pad))
        nc = (L + pad) // l
        S = torch.cumsum(dA.clamp(min=-10.0).view(b, nc, l, d, s), 2)    # (b, nc, l, d, s)
        local = torch.exp(S) * torch.cumsum(torch.exp(-S) * dBx.view(b, nc, l, d, s), 2)
        h, carry = dA.new_zeros(b, d, s), []
        for c in range(nc):                                   # chain chunks
            carry.append(h)
            h = local[:, c, -1] + torch.exp(S[:, c, -1]) * h
        hs = local + torch.exp(S) * torch.stack(carry, 1)[:, :, None]
        return torch.einsum("bclds,bcls->bcld", hs, F.pad(C, (0, 0, 0, pad)).view(b, nc, l, s)
                            ).reshape(b, nc * l, d)[:, :L]

    def forward(self, x):                                    # (b, L, d)
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, B, C = self.x_proj(u).split([self.r, self.ds, self.ds], -1)
        dt = F.softplus(self.dt_proj(dt))                    # (b, L, d)
        A = -torch.exp(self.A_log)                           # (d, s)
        y = self.scan(dt[..., None] * A, dt[..., None] * B[:, :, None] * u[..., None], C)
        return self.out_proj((y + u * self.D) * F.silu(z))


class BiMambaVC(nn.Module):
    """S-Mamba: a forward and a reversed Mamba block over the variable sequence, summed.

    vpt=True: scan the variables in a fresh random order at every training step.
    tta=M>1 (set at evaluation): average the layer output over M fixed random orders,
    a finite-sample estimate of the order-averaged (permutation-equivariant) mixer."""

    def __init__(self, d, d_state=2, order=None, vpt=False):
        super().__init__()
        self.fwd, self.bwd = Mamba(d, d_state), Mamba(d, d_state)
        self.register_buffer("order", order if order is not None else torch.empty(0, dtype=torch.long))
        self.vpt, self.tta = vpt, 0

    def scan_in(self, x, perm):
        inv = torch.argsort(perm)
        x = x[:, perm]
        return (self.fwd(x) + self.bwd(x.flip(1)).flip(1))[:, inv]

    def forward(self, x):
        if self.training and self.vpt:
            return self.scan_in(x, torch.randperm(x.shape[1], device=x.device))
        if not self.training and self.tta > 1:
            g = torch.Generator().manual_seed(12345)
            return torch.stack([self.scan_in(x, torch.randperm(x.shape[1], generator=g))
                                for _ in range(self.tta)]).mean(0)
        if len(self.order):                        # scan the variables in a given order
            inv = torch.argsort(self.order)
            x = x[:, self.order]
        y = self.fwd(x) + self.bwd(x.flip(1)).flip(1)
        return y[:, inv] if len(self.order) else y


class DynTopKVC(nn.Module):
    """MS-FLOW-style: content scores QK^T over all variables, keep the top-K per row."""

    def __init__(self, d, k, dropout=0.1):
        super().__init__()
        self.k = k
        self.q, self.kp, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, d = x.shape[1], x.shape[2]
        s = self.q(x) @ self.kp(x).transpose(1, 2) / math.sqrt(d)     # (b, N, N)
        k = min(self.k, n)
        thr = s.topk(k, -1).values[..., -1:]
        a = torch.softmax(s.masked_fill(s < thr, float("-inf")), -1)
        return self.out(self.drop(a) @ self.v(x))


class RetrieveAttendVC(nn.Module):
    """Candidates: top-k cosine neighbours of each variable in a learned embedding space
    (one O(N^2 d_e) computation per forward, shared by the batch; cacheable at inference).
    Weights: softmax over the candidates of q.k/sqrt(d) + s * cos(e_i, e_j), so they vary
    with the input. During training n_explore random candidates per variable are added so
    that useful variables outside the current top-k still receive gradient."""

    def __init__(self, d, n_vars, k, node_dim=16, n_explore=None, dropout=0.1):
        super().__init__()
        self.k = min(k, n_vars - 1)
        self.n_explore = (self.k + 1) // 2 if n_explore is None else n_explore
        self.emb = nn.Parameter(torch.randn(n_vars, node_dim) * 0.1)
        self.q, self.kp, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.sim_scale = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def neighbours(self):
        e = F.normalize(self.emb, dim=-1)
        sim = (e @ e.T).masked_fill(torch.eye(len(e), dtype=torch.bool, device=e.device), -1e4)
        return sim, sim.topk(self.k, -1).indices

    def forward(self, x):
        b, n, d = x.shape
        sim, idx = self.neighbours()
        if self.training and self.n_explore:
            idx = torch.cat([idx, torch.randint(0, n, (n, self.n_explore), device=x.device)], 1)
        q, kk, v = self.q(x), self.kp(x)[:, idx], self.v(x)[:, idx]   # kk, v: (b, N, k', d)
        s = torch.einsum("bnd,bnkd->bnk", q, kk) / math.sqrt(d) + self.sim_scale * sim.gather(1, idx)
        a = self.drop(torch.softmax(s, -1))
        return self.out(torch.einsum("bnk,bnkd->bnd", a, v))


class Layer(nn.Module):
    def __init__(self, d, vc, d_ff, dropout=0.1):
        super().__init__()
        self.vc = vc
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.vc(x)))
        return self.n2(x + self.drop(self.ffn(x)))


def make_vc(kind, d, n_vars, k, order=None, vpt=False):
    if kind == "none":
        return NoVC()
    if kind == "attn":
        return AttnVC(d)
    if kind == "bimamba":
        return BiMambaVC(d, order=order, vpt=vpt)
    if kind == "dyntopk":
        return DynTopKVC(d, k)
    if kind == "retrieve":
        return RetrieveAttendVC(d, n_vars, k)
    raise ValueError(kind)


class VariateForecaster(nn.Module):
    def __init__(self, n_vars, L, T, vc="attn", d=128, layers=2, d_ff=256, k=16,
                 id_emb=False, order=None, dropout=0.1, vpt=False):
        super().__init__()
        self.tok = nn.Linear(L, d)
        self.id = nn.Parameter(torch.zeros(n_vars, d)) if id_emb else None
        self.layers = nn.ModuleList(Layer(d, make_vc(vc, d, n_vars, k, order, vpt), d_ff, dropout)
                                    for _ in range(layers))
        self.proj = nn.Linear(d, T)

    def forward(self, x):                                    # x: (b, L, N)
        mu = x.mean(1, keepdim=True)
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        h = self.tok(((x - mu) / sd).transpose(1, 2))         # (b, N, d)
        if self.id is not None:
            h = h + self.id
        for layer in self.layers:
            h = layer(h)
        return self.proj(h).transpose(1, 2) * sd + mu        # (b, T, N)
