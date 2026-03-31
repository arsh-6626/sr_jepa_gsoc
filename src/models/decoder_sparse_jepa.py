import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embeddings.tnet_embeds import TNet
from src.embeddings.pos_embeds import SinusoidalPosEmbed
from src.models.sparse_transformer import (
    SparseTransformerEncoder,
    causal_window_attend_fn,
    jepa_attend_fn,
)


class SR_JEPA_Sparse_Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_size,
        n_head,
        n_layers,
        word_to_id,
        d_in=13,
        max_seq_len=1024,
        k_tokens=4,
        pred_token_id=3,
        window=8,
    ):
        super().__init__()

        self.dim_size      = dim_size
        self.vocab_size    = vocab_size
        self.k_tokens      = k_tokens
        self.pred_token_id = pred_token_id
        self.word_to_id    = word_to_id
        self.window        = window
        self.cond_proj   = nn.Linear(dim_size, dim_size)
        self.tnet        = TNet(d_in=d_in, e=dim_size)
        self.embedding   = nn.Embedding(vocab_size, dim_size)
        self.pos_encoder = SinusoidalPosEmbed(dim_size, max_seq_len)

        self.encoder = SparseTransformerEncoder(
            d_model=dim_size,
            n_head=n_head,
            n_layers=n_layers,
            dropout=0.1,
        )

        self.lm_head = nn.Linear(dim_size, vocab_size)

    def _build_sequence(self, raw_points, eq_token_ids, var_names_batch, pad_mask=None):
        device = raw_points.device
        B = raw_points.size(0)
        w_d      = self.tnet(raw_points, pad_mask)        # (B, D)
        w_d_proj = torch.tanh(self.cond_proj(w_d))        # (B, D)
        max_v = max(len(names) for names in var_names_batch)
        v_header_ids = torch.full(
            (B, max_v), self.word_to_id["[PAD]"], dtype=torch.long, device=device
        )
        for b, names in enumerate(var_names_batch):
            ids = [self.word_to_id.get(n, 0) for n in names]
            if ids:
                v_header_ids[b, :len(ids)] = torch.tensor(ids, device=device)
        v_header_embeds = self.embedding(v_header_ids)  # (B, max_v, D)
        pred_ids    = torch.full((B, self.k_tokens), self.pred_token_id,
                                 dtype=torch.long, device=device)
        pred_embeds = self.embedding(pred_ids)           # (B, k_tokens, D)
        eq_embeds = self.embedding(eq_token_ids)         # (B, seq, D)
        data_slot = w_d.unsqueeze(1)                     # (B, 1, D)
        x = torch.cat([data_slot, v_header_embeds, pred_embeds, eq_embeds], dim=1)
        x = x + w_d_proj.unsqueeze(1)
        x = self.pos_encoder(x)
        data_end_idx = 1 + max_v + self.k_tokens

        return x, data_end_idx

    def forward(self, raw_points, eq_token_ids, var_names, pad_mask=None):
        """
            logits    — (B, seq_len, vocab_size)  full sequence logits
            sy_tilde  — (B, D)  normalised predictor representation
            sy        — (B, D)  normalised target formula representation (detached)
        """
        x, data_end_idx = self._build_sequence(raw_points, eq_token_ids, var_names, pad_mask)
        seq_len = x.size(1)
        lm_fn = causal_window_attend_fn(self.window, data_end_idx, seq_len)
        z_lm  = self.encoder(x, lm_fn)
        logits = self.lm_head(z_lm)
        jp_fn  = jepa_attend_fn(data_end_idx, self.window, seq_len)
        z_jepa = self.encoder(x, jp_fn)
        #Pred(Enc(cloud))
        pred_repr = z_jepa[:, data_end_idx - 1, :]   # (B, D)
        eq_repr   = z_jepa[:, -1, :]                  # (B, D)
        sy_tilde = F.normalize(pred_repr, p=2, dim=-1)
        sy       = F.normalize(eq_repr.detach(), p=2, dim=-1)
        return logits, sy_tilde, sy, data_end_idx

    def decode_step(self, raw_points, curr_tokens, var_names, pad_mask=None):
        x, data_end_idx = self._build_sequence(raw_points, curr_tokens, var_names, pad_mask)
        seq_len = x.size(1)
        lm_fn   = causal_window_attend_fn(self.window, data_end_idx, seq_len)
        z       = self.encoder(x, lm_fn)
        return self.lm_head(z)