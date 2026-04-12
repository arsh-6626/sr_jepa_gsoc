import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embeddings.tnet_embeds import TNet
from src.embeddings.pos_embeds import SinusoidalPosEmbed

class SR_JEPA_EncDec(nn.Module):
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
        self.k_tokens = k_tokens
        self.pred_token_id = pred_token_id

        self.tnet = TNet(d_in=d_in, e=dim_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_size, nhead=n_head, batch_first=True, norm_first=True
        )
        self.data_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.pos_encoder = SinusoidalPosEmbed(dim_size, max_seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_size, nhead=n_head, batch_first=True, norm_first=True
        )
        self.equation_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(dim_size, vocab_size)

    def get_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return torch.zeros((sz, sz), device=device).masked_fill(mask, float("-inf"))

    def forward(self, raw_points, eq_token_ids, point_mask=None):
        device = raw_points.device
        B = raw_points.size(0)

        w_d = self.tnet(raw_points, point_mask).unsqueeze(1) 
        memory = self.data_encoder(w_d)

        pred_token_ids = torch.full(
            (B, self.k_tokens), self.pred_token_id, dtype=torch.long, device=device
        )
        tgt_embeds = self.embedding(torch.cat([pred_token_ids, eq_token_ids], dim=1))
        tgt_embeds = self.pos_encoder(tgt_embeds)

        tgt_mask = self.get_causal_mask(tgt_embeds.size(1), device)

        z_out = self.equation_decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(z_out[:, self.k_tokens:, :])

        pred_repr = z_out[:, :self.k_tokens, :]
        
        target_view_embeds = self.pos_encoder(self.embedding(eq_token_ids))
        target_repr = self.equation_decoder(target_view_embeds, memory)[:, :self.k_tokens, :]

        sy_tilde = F.normalize(pred_repr, p=2, dim=-1)
        sy = F.normalize(target_repr, p=2, dim=-1)

        return logits, sy_tilde, sy

    def generate(self, raw_points, max_len=50, start_token=1, point_mask=None):
        device = raw_points.device
        B = raw_points.size(0)

        memory = self.data_encoder(self.tnet(raw_points, point_mask).unsqueeze(1))
        
        curr_tokens = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        pred_token_ids = torch.full((B, self.k_tokens), self.pred_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt = self.pos_encoder(self.embedding(torch.cat([pred_token_ids, curr_tokens], dim=1)))
            tgt_mask = self.get_causal_mask(tgt.size(1), device)
            
            out = self.equation_decoder(tgt, memory, tgt_mask=tgt_mask)
            next_token_logits = self.lm_head(out[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            if (next_token == 2).all():
                break
                
        return curr_tokens
