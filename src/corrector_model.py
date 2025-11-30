import torch
import torch.nn as nn


# ------------------------
# Simple neighbor-mean corrector
# ------------------------
class SimpleCorrectorOLD(nn.Module):
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


def gcn_norm(edge_index, num_nodes, device):
    """
    Compute symmetric normalized adjacency:
        D^{-1/2} A D^{-1/2}
    Returns:
        row, col (edges)
        norm (edge weights)
    """
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float().to(device)
    deg_inv_sqrt = 1.0 / torch.sqrt(deg + 1e-12)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return row, col, norm


class GraphConv(nn.Module):
    """
    Manual GCN-style graph convolution:
        H = A_norm @ X @ W
    where A_norm = D^{-1/2} A D^{-1/2}
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        num_nodes = x.size(0)
        device = x.device

        # Compute normalized adjacency weights
        row, col, norm = gcn_norm(edge_index, num_nodes, device)

        # Linear transform
        xW = self.linear(x)

        # Message passing: sum_j A_norm[i,j] * xW_j
        out = torch.zeros_like(xW)
        out.index_add_(0, row, xW[col] * norm.unsqueeze(-1))

        return out
        

class GraphCorrector(nn.Module):
    """
    Hand-made GCN Corrector:
        Input features: [x, U_init, res_feature]
        Output:        ΔU
    """
    def __init__(self, in_dim, out_dim, hidden_layers, dropout):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(GraphConv(prev, h))
            prev = h

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Final graph conv to map to eigenmode corrections ΔU
        self.out = GraphConv(prev, out_dim)

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)

        return self.out(h, edge_index)
    

class SimpleCorrector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(128,128,128), dropout=0.0):
        """
        Position-aware corrector:
        - node features: X, U_init, resolution
        - edge features: relative pos, distance, direction
        """
        super().__init__()

        # MLP for edge messages
        edge_in = 3 + 1 + 3   # dx, dy, dz, distance, unit direction (3)
        msg_layers = []
        prev = edge_in + in_dim   # concat(node_i_feats, edge_feats)
        for h in hidden_sizes:
            msg_layers.append(nn.Linear(prev, h))
            msg_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                msg_layers.append(nn.Dropout(dropout))
            prev = h
        self.msg_net = nn.Sequential(*msg_layers)

        # MLP to turn aggregated message + node feature into correction
        corr_layers = []
        prev = hidden_sizes[-1] + in_dim
        for h in hidden_sizes:
            corr_layers.append(nn.Linear(prev, h))
            corr_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                corr_layers.append(nn.Dropout(dropout))
            prev = h
        corr_layers.append(nn.Linear(prev, out_dim))
        self.corr_net = nn.Sequential(*corr_layers)

    def forward(self, x, edge_index):
        row, col = edge_index  # i = row (receiver), j = col (sender)
        
        # Coordinates are ALWAYS first 3 entries in x
        Xi = x[row, :3]
        Xj = x[col, :3]

        # Geometry
        rel = Xj - Xi                           # (E,3)
        dist = torch.norm(rel, dim=1, keepdim=True) + 1e-12
        unit = rel / dist

        edge_feat = torch.cat([rel, dist, unit], dim=1)

        # Messages = MLP([x_i || edge_feat])
        msg_in = torch.cat([x[row], edge_feat], dim=1)
        msg = self.msg_net(msg_in)

        # Aggregate messages (sum)
        n = x.shape[0]
        agg = torch.zeros(n, msg.shape[1], device=x.device)
        agg.index_add_(0, row, msg)

        # Correction MLP
        corr_in = torch.cat([x, agg], dim=1)
        return self.corr_net(corr_in)