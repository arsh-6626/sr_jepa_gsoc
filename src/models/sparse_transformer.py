import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAttention(nn.Module):

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head  = n_head
        self.d_head  = d_model // n_head
        self.scale   = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

    def _build_index(self, seq_len: int, attend_fn, device) -> list[torch.Tensor]:
        return [
            attend_fn(q, seq_len).to(device)
            for q in range(seq_len)
        ]

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D)
        attend_fn,                
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_head, self.d_head
        Q = self.q_proj(x).view(B, T, H, Dh)   # (B, T, H, Dh)
        K = self.k_proj(x).view(B, T, H, Dh)
        V = self.v_proj(x).view(B, T, H, Dh)
        key_indices = self._build_index(T, attend_fn, x.device)
        out = torch.zeros(B, T, H, Dh, device=x.device, dtype=x.dtype)
        for q in range(T):
            ki = key_indices[q]
            nk = ki.size(0)
            if nk == 0:
                continue
            q_vec = Q[:, q, :, :].unsqueeze(2)
            k_vec = K[:, ki, :, :].permute(0, 2, 1, 3)
            scores = torch.matmul(q_vec, k_vec.transpose(-1, -2)) / self.scale
            attn   = torch.softmax(scores, dim=-1)
            attn   = self.drop(attn)
            v_vec = V[:, ki, :, :].permute(0, 2, 1, 3)
            out[:, q, :, :] = torch.matmul(attn, v_vec).squeeze(2)

        out = out.reshape(B, T, D)
        return self.out(out)


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_ff: int = None, dropout: float = 0.1):
        super().__init__()
        dim_ff = dim_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = SparseAttention(d_model, n_head, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attend_fn) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attend_fn)
        x = x + self.ff(self.norm2(x))
        return x


class SparseTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layers: int,
                 dim_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseTransformerLayer(d_model, n_head, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attend_fn) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attend_fn)
        return x

def causal_window_attend_fn(window: int, n_global: int, seq_len: int):
    indices = []
    for q in range(seq_len):
        lo = max(0, q - window)
        local = set(range(lo, q + 1))
        global_keys = set(range(min(n_global, seq_len)))
        if q < n_global:
            global_keys |= set(range(q + 1))
        allowed = sorted(local | global_keys)
        indices.append(torch.tensor(allowed, dtype=torch.long))

    def attend_fn(q_pos: int, _seq_len: int) -> torch.Tensor:
        return indices[q_pos]

    return attend_fn


def jepa_attend_fn(data_end_idx, window, seq_len):
    indices = []
    for q in range(seq_len):
        if q < data_end_idx:
            allowed = list(range(q + 1))
        else:
            # eq block: causal within eq block only — independent of data
            eq_q = q - data_end_idx
            eq_lo = max(0, eq_q - window)
            allowed = [data_end_idx + k for k in range(eq_lo, eq_q + 1)]
        indices.append(torch.tensor(allowed, dtype=torch.long))

    def attend_fn(q_pos, _seq_len):
        return indices[q_pos]
    return attend_fn