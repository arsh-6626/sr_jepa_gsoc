import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim_size, max_seq_len=1024):
        super().__init__()
        self.dim_size = dim_size
        pe = torch.zeros(max_seq_len, dim_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_size, 2).float() * (-np.log(10000.0) / dim_size)
        )        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]