import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embeddings.tnet_embeds import TNet, OrthoTNet
from src.embeddings.pos_embeds import SinusoidalPosEmbed


class SR_JEPA_Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_size,
        n_head,
        n_layers,
        d_in=13,
        max_seq_len=1024,
        k_tokens=4,
        pred_token_id=3,
    ):
        super().__init__()

        self.dim_size = dim_size
        self.vocab_size = vocab_size
        self.k_tokens = k_tokens
        self.pred_token_id = pred_token_id

        self.tnet = TNet(d_in=d_in, e=dim_size)
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.pos_encoder = SinusoidalPosEmbed(dim_size, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_size,
            nhead=n_head,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.jepa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dim_size, vocab_size)

    def get_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return torch.zeros((sz, sz), device=device).masked_fill(mask, float("-inf"))

    def get_jepa_mask(self, total_sz, data_end_idx, device):
        mask = torch.zeros((total_sz, total_sz), device=device)
        mask[:data_end_idx, data_end_idx:] = float("-inf")
        mask[data_end_idx:, :data_end_idx] = float("-inf")
        target_sz = total_sz - data_end_idx
        if target_sz > 0:
            causal = torch.triu(
                torch.ones(target_sz, target_sz, device=device), diagonal=1
            ).bool()
            mask[data_end_idx:, data_end_idx:].masked_fill_(causal, float("-inf"))
        return mask

    def _build_sequence(self, raw_points, eq_token_ids, mask):

        device = raw_points.device
        B = raw_points.size(0)
        w_d = self.tnet(raw_points, mask)
        pred_token_ids = torch.full(
            (B, self.k_tokens), self.pred_token_id, dtype=torch.long, device=device
        )
        pred_embeds = self.embedding(pred_token_ids)
        eq_embeds = self.embedding(eq_token_ids)

        x = torch.cat([w_d.unsqueeze(1), pred_embeds, eq_embeds], dim=1)

        data_end_idx = 1 + self.k_tokens
        x = x + w_d.unsqueeze(1)  # broadcast to all positions
        x = self.pos_encoder(x)

        return x, data_end_idx

    def forward(self, raw_points, eq_token_ids, mask=None):
        x, data_end_idx = self._build_sequence(raw_points, eq_token_ids, mask)

        seq_len = x.size(1)
        device  = raw_points.device

        causal_mask = self.get_causal_mask(seq_len, device)
        z_lm  = self.jepa_encoder(x, mask=causal_mask)
        logits = self.lm_head(z_lm)

        jepa_mask = self.get_jepa_mask(seq_len, data_end_idx, device)
        z_jepa = self.jepa_encoder(x, mask=jepa_mask)

        pred_repr   = z_jepa[:, 1:data_end_idx, :]
        target_repr = z_jepa[:, data_end_idx:, :][:, :self.k_tokens, :]

        sy_tilde = F.normalize(pred_repr,   p=2, dim=-1)
        sy       = F.normalize(target_repr, p=2, dim=-1)

        return logits, sy_tilde, sy

    def decode_step(self, raw_points, curr_tokens, mask):
        device = raw_points.device
        B = raw_points.size(0)

        w_d= self.tnet(raw_points, mask)
        pred_token_ids = torch.full(
            (B, self.k_tokens), self.pred_token_id, dtype=torch.long, device=device
        )
        pred_embeds = self.embedding(pred_token_ids)
        eq_embeds   = self.embedding(curr_tokens)

        x = torch.cat([w_d.unsqueeze(1), pred_embeds, eq_embeds], dim=1)
        data_end_idx = 1 + self.k_tokens
        x[:, :data_end_idx] = x[:, :data_end_idx] + w_d.unsqueeze(1)
        x = self.pos_encoder(x)

        causal_mask = self.get_causal_mask(x.size(1), device)
        z = self.jepa_encoder(x, mask=causal_mask)

        return self.lm_head(z)