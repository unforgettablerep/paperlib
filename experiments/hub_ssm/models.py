"""Variable mixers for the Hub-SSM experiments: Hub-SSM (order-free low-rank selective
mixer), a sparse top-k mixer, their low-rank + sparse combination, and baselines.

Shapes: x is (batch, N variables, P patches, D model dim).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HubSSMMixer(nn.Module):
    """K hub states shared by all variables.

    write : variable j at patch p writes v_{j,p} into hub k with weight
            B_{j,p,k} * dt_{j,p,k}   (B = soft assignment, dt = selective step)
    state : H_p = exp(-dtbar_p * A) * H_{p-1} + sum_j w_{j,p} v_{j,p}   (per hub)
    read  : y_{i,p} = sum_k C_{i,p,k} * H_{p,k} / mass_{p,k}
    Every per-variable map shares weights and the write is a sum over variables,
    so permuting variables permutes the output (no scan order exists).
    """

    def __init__(self, d_model, n_hubs=8, n_nodes=None, node_dim=0, temporal=True,
                 select=False):
        super().__init__()
        self.K, self.temporal, self.select = n_hubs, temporal, select
        if select:
            # v2: hubs select variables (scores normalised over variables, not over hubs),
            # and a per-variable FiLM scale aligns sign/scale before writing
            self.film = nn.Linear(node_dim, d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.ones_(self.film.bias)
        self.norm = nn.LayerNorm(d_model)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1) if node_dim else None
        in_dim = d_model + node_dim
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_z = nn.Linear(d_model, d_model)
        self.proj_w = nn.Linear(in_dim, n_hubs)
        self.proj_r = nn.Linear(in_dim, n_hubs)
        self.proj_dt = nn.Linear(in_dim, n_hubs)
        # per-hub decay rates spread over timescales, as in Mamba's A initialisation
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.05, 2.0, n_hubs)))
        self.out = nn.Linear(d_model, d_model)

    def coupling(self, x):
        """Same-patch variable-to-variable mixing weights M[b,p,i,j] (for analysis)."""
        _, B, C, dt, _ = self._gates(x)
        w = B * dt
        mass = w.sum(1, keepdim=True)
        return torch.einsum("bipk,bjpk->bpij", C, w / (mass + 1e-6))

    def _feat(self, h):
        if self.node_emb is None:
            return h
        e = self.node_emb[None, :, None, :].expand(h.shape[0], -1, h.shape[2], -1)
        return torch.cat([h, e], -1)

    def _gates(self, x):
        h = self.norm(x)
        feat = self._feat(h)
        B = torch.softmax(self.proj_w(feat), -1)
        C = torch.softmax(self.proj_r(feat), -1)
        dt = F.softplus(self.proj_dt(feat))
        return h, B, C, dt, feat

    def forward(self, x):
        h, B, C, dt, _ = self._gates(x)
        v = self.proj_v(h)
        if self.select:
            s = self.proj_w(self._feat(h))                   # unbounded write scores
            w = torch.exp(s - s.amax(dim=(1, 2), keepdim=True))   # max over variables & patches is invariant
            v = v * self.film(self.node_emb)[None, :, None, :]
        else:
            w = B * dt                                       # (b,N,P,K)
        U = torch.einsum("bnpk,bnpd->bpkd", w, v)            # aggregated write
        m = w.sum(1)                                         # (b,P,K)
        if self.temporal:
            dtbar = (dt * B).sum(1) / (B.sum(1) + 1e-6)      # permutation-invariant step
            S = torch.cumsum(dtbar * torch.exp(self.A_log), 1)          # (b,P,K)
            diff = S[:, :, None, :] - S[:, None, :, :]                   # S_p - S_q
            P = x.shape[2]
            causal = torch.ones(P, P, dtype=torch.bool, device=x.device).tril()
            decay = torch.exp(-diff.masked_fill(~causal[None, :, :, None], float("inf")))
            U = torch.einsum("bpqk,bqkd->bpkd", decay, U)
            m = torch.einsum("bpqk,bqk->bpk", decay, m)
        Hn = U / (m[..., None] + 1e-6)                       # hub = decayed weighted centroid
        y = torch.einsum("bnpk,bpkd->bnpd", C, Hn)
        y = y * F.silu(self.proj_z(h))
        return x + self.out(y)


class SparseTopKMixer(nn.Module):
    """Sparse pairwise term S: each variable attends to k candidate variables.

    Candidates are the top-k cosine neighbours in a learned node-embedding space
    (computed once per forward, O(N^2 * node_dim), independent of batch and patches);
    weights over the candidates are content-dependent (softmax of q.k plus the
    embedding similarity, which is what gives the embeddings a gradient).
    During training `n_explore` random extra candidates per variable let useful
    variables that are not yet neighbours receive gradient and enter the top-k.
    Permuting variables together with their embeddings permutes the output.
    """

    def __init__(self, d_model, n_nodes, node_dim=8, k=8, n_explore=None):
        super().__init__()
        self.k = k
        self.n_explore = k // 2 if n_explore is None else n_explore
        self.norm = nn.LayerNorm(d_model)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1)
        self.q = nn.Linear(d_model + node_dim, d_model)
        self.kp = nn.Linear(d_model + node_dim, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.sim_scale = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(d_model, d_model)

    def neighbours(self):
        e = F.normalize(self.node_emb, dim=-1)
        sim = e @ e.T
        sim = sim.masked_fill(torch.eye(len(e), dtype=torch.bool, device=e.device), float("-inf"))
        return sim, sim.topk(self.k, -1).indices                  # (N, N), (N, k)

    def forward(self, x):
        b, n, p, d = x.shape
        h = self.norm(x)
        e = self.node_emb[None, :, None, :].expand(b, -1, p, -1)
        feat = torch.cat([h, e], -1)
        q, kk, v = self.q(feat), self.kp(feat), self.v(h)
        sim, idx = self.neighbours()
        if self.training and self.n_explore:
            rand = torch.randint(0, n, (n, self.n_explore), device=x.device)
            idx = torch.cat([idx, rand], 1)
        nb_sim = sim.gather(1, idx)                                # (N, k')
        nb_sim = torch.where(torch.isinf(nb_sim), torch.full_like(nb_sim, -1e4), nb_sim)  # self if sampled
        k_nb, v_nb = kk[:, idx], v[:, idx]                         # (b, N, k', P, D)
        score = torch.einsum("bnpd,bnkpd->bnpk", q, k_nb) / math.sqrt(d)
        score = score + self.sim_scale * nb_sim[None, :, None, :]
        alpha = torch.softmax(score, -1)
        y = torch.einsum("bnpk,bnkpd->bnpd", alpha, v_nb)
        return x + self.out(y)


class LowRankSparseMixer(nn.Module):
    """M ~ C B^T (Hub-SSM v2: global, order-free, low-rank) + S (sparse top-k pairs)."""

    def __init__(self, d_model, n_hubs, n_nodes, node_dim=8, k=8):
        super().__init__()
        self.hub = HubSSMMixer(d_model, n_hubs, n_nodes=n_nodes, node_dim=node_dim, select=True)
        self.sparse = SparseTopKMixer(d_model, n_nodes, node_dim=node_dim, k=k)

    def forward(self, x):
        return self.hub(x) + self.sparse(x) - x                    # both branches are residual


class MeanPoolMixer(nn.Module):
    """Homogeneous read baseline (DeepSets / VI-2D-SSM style global pooled field)."""

    def __init__(self, d_model, n_nodes=None, node_dim=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1) if node_dim else None
        in_dim = d_model + node_dim
        self.phi = nn.Sequential(nn.Linear(in_dim, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.rho = nn.Sequential(nn.Linear(in_dim + d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        h = self.norm(x)
        feat = h
        if self.node_emb is not None:
            e = self.node_emb[None, :, None, :].expand(x.shape[0], -1, x.shape[2], -1)
            feat = torch.cat([h, e], -1)
        g = self.phi(feat).mean(1, keepdim=True).expand_as(h)
        return x + self.rho(torch.cat([feat, g], -1))


class AttnMixer(nn.Module):
    """Full softmax attention over variables at each patch (iTransformer-style, O(N^2))."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        b, n, p, d = x.shape
        h = self.norm(x).permute(0, 2, 1, 3).reshape(b * p, n, d)
        y, _ = self.attn(h, h, h, need_weights=False)
        return x + y.reshape(b, p, n, d).permute(0, 2, 1, 3)


class S6(nn.Module):
    """Minimal selective scan (Mamba S6 without conv), scanning over the variable axis."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.x_proj = nn.Linear(d_model, 2 * d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))
        self.Dskip = nn.Parameter(torch.ones(d_model))
        self.out = nn.Linear(d_model, d_model)
        self.ds = d_state

    def forward(self, h):                                    # h: (b, L, d)
        xz = self.in_proj(h)
        u, z = xz.chunk(2, -1)
        u = F.silu(u)
        Bm, Cm = self.x_proj(u).split(self.ds, -1)
        dt = F.softplus(self.dt_proj(u))
        A = -torch.exp(self.A_log)
        state = torch.zeros(h.shape[0], h.shape[2], self.ds, device=h.device)
        ys = []
        for t in range(h.shape[1]):
            dA = torch.exp(dt[:, t, :, None] * A)
            state = dA * state + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((state * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + u * self.Dskip
        return self.out(y * F.silu(z))


class BiScanMixer(nn.Module):
    """S-Mamba-style bidirectional selective scan over variables (order-dependent)."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd, self.bwd = S6(d_model, d_state), S6(d_model, d_state)

    def forward(self, x):
        b, n, p, d = x.shape
        h = self.norm(x).permute(0, 2, 1, 3).reshape(b * p, n, d)
        y = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        return x + y.reshape(b, p, n, d).permute(0, 2, 1, 3)


class Forecaster(nn.Module):
    """RevIN -> patch embedding -> [variable mixer] -> FFN -> linear head."""

    def __init__(self, mixer, n_vars, L=48, T=16, plen=8, d_model=32, id_emb=False):
        super().__init__()
        self.plen, self.P = plen, L // plen
        self.embed = nn.Linear(plen, d_model)
        self.pos = nn.Parameter(torch.randn(self.P, d_model) * 0.02)
        # per-variable identity added to every token, so every mixer can tell variables apart
        self.id = nn.Parameter(torch.randn(n_vars, 1, d_model) * 0.1) if id_emb else None
        self.mixer = mixer
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(self.P * d_model, T)

    def forward(self, x):                                    # x: (b, L, N)
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        x = ((x - mu) / sd).transpose(1, 2)                  # (b, N, L)
        h = self.embed(x.unfold(-1, self.plen, self.plen)) + self.pos
        if self.id is not None:
            h = h + self.id
        if self.mixer is not None:
            h = self.mixer(h)
        h = h + self.ffn(h)
        y = self.head(h.flatten(2)).transpose(1, 2)          # (b, T, N)
        return y * sd + mu
