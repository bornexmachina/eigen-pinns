import torch
import torch.nn as nn
from typing import List, Tuple


# ------------------------
# Simple neighbor-mean corrector
# ------------------------
class SimpleCorrector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev = in_dim * 2
        for h in hidden_layers:
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


# ------------------------
# Spectral Corrector (GCN-like)
# GraphSAGE
# https://arxiv.org/pdf/1706.02216 
# ------------------------
class SpectralCorrector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: List[int], dropout: float):
        super().__init__()
        
        # Determine the input dimension for the first MLP layer
        # The MLP takes the concatenation of (current node features, aggregated features)
        mlp_input_dim = in_dim * 2
        
        layers = []
        prev = mlp_input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, A_norm_sparse: torch.Tensor) -> torch.Tensor:
        """
        Performs one GCN-like aggregation step followed by an MLP.

        Args:
            x (torch.Tensor): 
                Node features of shape (N, in_dim).
            A_norm_sparse (torch.Tensor):
                Graph structure 
                Pre-computed normalized adjacency matrix 
                (D_hat^-1/2 * A_hat * D_hat^-1/2) 
                of shape (N, N) as a sparse tensor.

        Returns:
            torch.Tensor: The predicted correction vector (N, out_dim).
        """
        
        # 1. GCN-like Aggregation (H^(l+1) = A_norm * H^(l))
        agg = torch.sparse.mm(A_norm_sparse, x)
        
        # 2. Residual connection: Concatenate and pass through MLP
        h = torch.cat([x, agg], dim=1)
        
        # 3. Output correction vector
        return self.net(h)