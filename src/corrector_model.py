import torch
import torch.nn as nn


# ------------------------
# Simple neighbor-mean corrector
# ------------------------
class SimpleCorrector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(128,128,128), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim * 2
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        row, col = edge_index
        n = x.shape[0]
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.bincount(row, minlength=n).unsqueeze(1).to(x.dtype).to(x.device).clamp(min=1.0)
        agg = agg / deg
        h = torch.cat([x, agg], dim=1)
        return self.net(h)
