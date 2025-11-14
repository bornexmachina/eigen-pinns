import numpy as np
from scipy.linalg import eigh
import torch
import torch.optim as optim
import robust_laplacian
from corrector_model import SimpleCorrector
import utils


class MultigridGNN:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None

    # ------------------------
    # Physics-informed GNN training
    # ------------------------
    def train_multiresolution(self, X_list, U_init_list, edge_index_list, epochs, lr, corr_scale, w_res, w_orth, w_proj, grad_clip, weight_decay, log_every):
        device = self.device
        n_modes = U_init_list[0].shape[1]

        # Build torch tensors and resolution indicators
        x_feats_all, U_all, edge_index_all = [], [], []
        node_offset = 0
        max_nodes = max([X.shape[0] for X in X_list])
        for X, U_init, edge_index in zip(X_list, U_init_list, edge_index_list):
            res_feat = np.full((X.shape[0], 1), X.shape[0]/max_nodes)
            x_feats_all.append(np.hstack([X, U_init, res_feat]))
            U_all.append(U_init)
            edge_index_all.append(edge_index + node_offset)
            node_offset += X.shape[0]

        x_feats_all = torch.FloatTensor(np.vstack(x_feats_all)).to(device)
        U_all_tensor = torch.FloatTensor(np.vstack(U_all)).to(device)
        edge_index_all = torch.cat(edge_index_all, dim=1).to(device)

        in_dim = x_feats_all.shape[1]
        if self.model is None:
            self.model = SimpleCorrector(in_dim, n_modes).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=lr, weight_decay=weight_decay)
        self.model.train()

        # Precompute Laplacians per level
        L_list, M_list = [], []
        node_offset = 0
        for X in X_list:
            L, M = robust_laplacian.point_cloud_laplacian(X)
            L_list.append(utils.scipy_sparse_to_torch_sparse(L).to(device))
            M_list.append(utils.scipy_sparse_to_torch_sparse(M).to(device))

        for ep in range(epochs):
            optimizer.zero_grad()
            corr_raw = self.model(x_feats_all, edge_index_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr

            # Physics-informed loss
            loss = 0.0
            node_offset = 0
            
            for i, (L_t, M_t, U_init) in enumerate(zip(L_list, M_list, U_init_list)):
                n_nodes = U_init.shape[0]
                U_level = U_pred[node_offset:node_offset+n_nodes]
                
                # Normalize each eigenvector
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_normalized = U_level / norms.unsqueeze(0)
                
                # Recompute with normalized vectors
                Mu_norm = torch.sparse.mm(M_t, U_level_normalized)
                Lu_norm = torch.sparse.mm(L_t, U_level_normalized)

                # Rayleigh residual
                num = torch.sum(U_level_normalized * Lu_norm, dim=0)
                den = torch.sum(U_level_normalized * Mu_norm, dim=0) + 1e-12
                lambdas = num / den
                res = Lu_norm - Mu_norm * lambdas.unsqueeze(0)
                L_res = torch.mean(res**2)
                
                # Orthonormality (should be near-perfect with normalization)
                Gram = U_level_normalized.t() @ Mu_norm
                L_orth = torch.mean((Gram - torch.eye(n_modes, device=device))**2)

                # Zero-mean constraint for modes 1+ (NOT mode 0)
                # Mode 0 is constant, modes 1+ should be zero-mean
                ones = torch.ones(n_nodes, 1, device=device)
                if n_modes > 1:
                    # Project out constant component from modes 1 onwards
                    mean_constraint = ones.t() @ Mu_norm[:, 1:]  # Shape: (1, n_modes-1)
                    L_mean = torch.mean(mean_constraint**2)
                else:
                    L_mean = torch.tensor(0.0, device=device)

                loss += w_res * L_res + w_orth * L_orth + w_proj * L_mean
                node_offset += n_nodes

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, 
                                                    self.model.parameters()), grad_clip)
            optimizer.step()

            if ep % log_every == 0 or ep == epochs-1:
                print(f"Epoch {ep:4d}: Loss={loss.item():.6f} | "
                    f"Res={L_res.item():.6f} | Orth={L_orth.item():.6f} | "
                    f"Mean={L_mean.item():.6f}")

        # === POST-TRAINING: Final normalization ===
        with torch.no_grad():
            corr_raw = self.model(x_feats_all, edge_index_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr
            
            # Normalize per level
            node_offset = 0
            U_pred_normalized = []
            for i, M_t in enumerate(M_list):
                n_nodes = U_init_list[i].shape[0]
                U_level = U_pred[node_offset:node_offset+n_nodes]
                
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_norm = U_level / norms.unsqueeze(0)
                U_pred_normalized.append(U_level_norm)
                
                node_offset += n_nodes
            
            U_pred_final = torch.cat(U_pred_normalized, dim=0)

        return U_pred_final.detach().cpu().numpy()

    # ------------------------
    # Rayleigh-Ritz refinement
    # ------------------------
    def refine_eigenvectors(self, U_pred, L, M):
        U = torch.FloatTensor(U_pred).to(self.device)
        L_t = utils.scipy_sparse_to_torch_sparse(L).to(self.device)
        M_t = utils.scipy_sparse_to_torch_sparse(M).to(self.device)
        A = (U.t() @ torch.sparse.mm(L_t, U)).cpu().numpy()
        B = (U.t() @ torch.sparse.mm(M_t, U)).cpu().numpy()
        vals, C = eigh(A, B)
        U_refined = U.cpu().numpy() @ C
        return vals, U_refined
