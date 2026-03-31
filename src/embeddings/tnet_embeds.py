import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, d_in, e):
        super().__init__()
        self.input_proj = nn.Linear(d_in, e)

        self.mlp1 = nn.Sequential(
            nn.Linear(e, e),
            nn.ReLU(),
            nn.LayerNorm(e)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(e, 2 * e),
            nn.ReLU(),
            nn.LayerNorm(2 * e)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(2 * e, 4 * e),
            nn.ReLU(),
            nn.LayerNorm(4 * e)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * e, 2 * e),
            nn.ReLU(),
            nn.Linear(2 * e, e)
        )

    def forward(self, x, mask=None):
        B, N, d_in = x.size()

        x = self.input_proj(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)  # (B, N, 4e)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask
            x_sum = x.sum(dim=1)
            counts = mask.sum(dim=1) + 1e-6
            x_mean = x_sum / counts
            x_max = (x - (1 - mask) * 1e9).max(dim=1).values

        else:
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1).values
        w_d = x_max + x_mean
        w_d = self.fc(w_d)
        w_d = F.normalize(w_d, dim=-1)
        return w_d

