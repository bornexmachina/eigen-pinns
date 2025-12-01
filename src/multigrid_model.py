import numpy as np
from scipy.linalg import eigh
import torch
import torch.optim as optim
from corrector_model import SimpleCorrector
from corrector_model import SpectralCorrector
import utils


class MultigridGNN:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None

    # ------------------------
    # Physics-informed GNN training
    # ------------------------
    def train_multiresolution(self, X_list, K_list, M_list, U_init_list, lambda_coarse, 
                              edge_index_list, epochs, lr, corr_scale, 
                              w_res, w_orth, w_proj, w_trace, w_order, w_eigen, 
                              grad_clip, weight_decay, log_every, hidden_layers, dropout):
        device = self.device
        n_modes = U_init_list[0].shape[1]

        # Build torch tensors with physics-informed features
        x_feats_all, U_all, edge_index_all = [], [], []
        node_offset = 0
        max_nodes = max([X.shape[0] for X in X_list])
        
        print("\nBuilding physics-informed features...")
        for i, (X, U_init, edge_index) in enumerate(zip(X_list, U_init_list, edge_index_list)):
            n_nodes = X.shape[0]
            
            # 1. Resolution indicator
            res_feat = np.full((n_nodes, 1), n_nodes/max_nodes)
            
            # 2. Compute physics operators applied to initial guess
            # Ku = K @ U_init (stiffness applied to modes)
            # Mu = M @ U_init (mass applied to modes)
            Ku_init = K_list[i] @ U_init  # (n_nodes, n_modes)
            Mu_init = M_list[i] @ U_init  # (n_nodes, n_modes)
            
            # 3. Compute residual: R = Ku - lambda * Mu
            # This tells the model WHERE the eigenvalue equation is violated
            if lambda_coarse is not None:
                lambda_expanded = lambda_coarse.reshape(1, -1)  # (1, n_modes)
                residual = Ku_init - lambda_expanded * Mu_init  # (n_nodes, n_modes)
            else:
                residual = np.zeros_like(Ku_init)
            
            # 4. Diagonal features (local stiffness/mass coefficients)
            K_diag_feat = K_list[i].diagonal().reshape(-1, 1)
            M_diag_feat = M_list[i].diagonal().reshape(-1, 1)
            
            # 5. Local geometric features
            # Vertex degree (valence) from K-NN graph
            row_indices = edge_index[0].numpy()
            valence = np.bincount(row_indices, minlength=n_nodes).reshape(-1, 1).astype(np.float32)
            
            # Concatenate all features:
            # [coords(3), K_diag(1), M_diag(1), Ku(n_modes), Mu(n_modes), residual(n_modes), U_init(n_modes), valence(1), resolution(1)]
            x_feats = np.hstack([
                X,              # (n_nodes, 3) - spatial coordinates
                K_diag_feat,    # (n_nodes, 1) - diagonal stiffness
                M_diag_feat,    # (n_nodes, 1) - diagonal mass
                Ku_init,        # (n_nodes, n_modes) - stiffness applied to modes
                Mu_init,        # (n_nodes, n_modes) - mass applied to modes
                residual,       # (n_nodes, n_modes) - eigenvalue residual (KEY!)
                U_init,         # (n_nodes, n_modes) - initial guess
                valence,        # (n_nodes, 1) - vertex connectivity
                res_feat        # (n_nodes, 1) - resolution indicator
            ])
            
            print(f"  Level {i}: {n_nodes} nodes, feature dim = {x_feats.shape[1]}")
            
            x_feats_all.append(x_feats)
            U_all.append(U_init)
            edge_index_all.append(edge_index + node_offset)
            node_offset += n_nodes

        x_feats_all = torch.FloatTensor(np.vstack(x_feats_all)).to(device)
        U_all_tensor = torch.FloatTensor(np.vstack(U_all)).to(device)
        edge_index_all = torch.cat(edge_index_all, dim=1).to(device)

        in_dim = x_feats_all.shape[1]
        print(f"Total input feature dimension: {in_dim}")
        
        if self.model is None:
            self.model = SpectralCorrector(in_dim, n_modes, hidden_layers, dropout).to(device)
            # self.model = SimpleCorrector(in_dim, n_modes, hidden_layers, dropout).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=lr, weight_decay=weight_decay)
        
        self.model.train()

        # Precompute K/M sparse tensors and A_norm (GCN operator) per level
        K_t_list, M_t_list, A_norm_list = [], [], []

        for i, (K_scipy, M_scipy, edge_index) in enumerate(zip(K_list, M_list, edge_index_list)):
            n_nodes = K_scipy.shape[0]
 
            # 1. Convert K and M to torch sparse tensors
            K_t_list.append(utils.scipy_sparse_to_torch_sparse(K_scipy).to(device))
            M_t_list.append(utils.scipy_sparse_to_torch_sparse(M_scipy).to(device))

            # 2. Build and normalize Adjacency matrix (A_norm) for GCN
            edge_index_t = edge_index.to(device)
            # Create A_hat (A + I) from edge_index
            adj = torch.sparse_coo_tensor(edge_index_t, torch.ones(edge_index_t.shape[1], device=device), (n_nodes, n_nodes))
            A_hat = (adj + torch.sparse_coo_tensor(torch.arange(n_nodes, device=device).repeat(2, 1), 
                                                   torch.ones(n_nodes, device=device), (n_nodes, n_nodes))).coalesce()

            # Calculate D_hat^-1/2
            row = A_hat.indices()[0]
            deg_hat = torch.bincount(row, minlength=n_nodes).float()
            D_hat_invsqrt_diag = torch.pow(deg_hat.clamp(min=1e-12), -0.5)
            D_hat_invsqrt = torch.diag(D_hat_invsqrt_diag).to_sparse_coo()
 
            # A_norm = D_hat^-1/2 * A_hat * D_hat^-1/2
            A_norm = torch.sparse.mm(D_hat_invsqrt, torch.sparse.mm(A_hat, D_hat_invsqrt))
            A_norm_list.append(A_norm)

        if lambda_coarse is not None:
            lambda_target = torch.FloatTensor(lambda_coarse).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()

            # Pass the block-diagonal normalized operator (A_norm_all) to the SpectralCorrector
            A_norm_all = utils.sparse_block_diag(A_norm_list).to(device)
            corr_raw = self.model(x_feats_all, A_norm_all)

            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr

            # Physics-informed loss
            loss = 0.0
            loss_residual = 0.0
            loss_orthogonal = 0.0
            loss_surface_integral = 0.0
            loss_trace = 0.0
            loss_eigenvalue_order = 0.0
            loss_coarse_eigenvalues = 0.0
            node_offset = 0
            
            for i, (K_t, M_t, U_init) in enumerate(zip(K_t_list, M_t_list, U_init_list)):
                n_nodes = U_init.shape[0]
                U_level = U_pred[node_offset:node_offset+n_nodes]
                
                # Normalize each eigenvector
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_normalized = U_level / norms.unsqueeze(0)
                
                # Recompute with normalized vectors
                Mu_norm = torch.sparse.mm(M_t, U_level_normalized)
                Ku_norm = torch.sparse.mm(K_t, U_level_normalized)

                # Rayleigh quotient
                lambdas = torch.sum(U_level_normalized * Ku_norm, dim=0) / (torch.sum(U_level_normalized * Mu_norm, dim=0) + 1e-12)
                res = Ku_norm - Mu_norm * lambdas.unsqueeze(0)
                loss_residual += torch.mean(res**2)
                
                # Orthonormality
                Gram = U_level_normalized.t() @ Mu_norm
                loss_orthogonal += torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes

                # Zero-mean constraint for modes 1+ (NOT mode 0)
                ones = torch.ones(n_nodes, 1, device=device)
                if n_modes > 1:
                    mean_constraint = ones.t() @ Mu_norm[:, 1:]
                    loss_surface_integral += torch.mean(mean_constraint**2)
                else:
                    loss_surface_integral += torch.tensor(0.0, device=device)

                # Trace minimization
                loss_trace += torch.mean(lambdas)

                # Eigenvalue ordering
                lambda_diffs = lambdas[1:] - lambdas[:-1]
                loss_eigenvalue_order += torch.sum(torch.relu(-lambda_diffs))

                # Coarse eigenvalue matching (only level 0)
                if i == 0 and lambda_coarse is not None:
                    loss_coarse_eigenvalues += torch.mean((lambdas - lambda_target)**2)
                else:
                    loss_coarse_eigenvalues += torch.tensor(0.0, device=device)

                node_offset += n_nodes

            loss = w_res * loss_residual + w_orth * loss_orthogonal + w_proj * loss_surface_integral + w_trace * loss_trace + w_order * loss_eigenvalue_order + w_eigen * loss_coarse_eigenvalues
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, 
                                                    self.model.parameters()), grad_clip)
            optimizer.step()

            if ep % log_every == 0 or ep == epochs-1:
                print(f"Epoch {ep:4d}: Loss={loss.item():.6f} | "
                      f"Res={(w_res * loss_residual).item():.6f} | Orth={(w_orth * loss_orthogonal).item():.6f} | "
                      f"Mean={(w_proj * loss_surface_integral).item():.6f} | Trace={(w_trace * loss_trace).item():.6f} | "
                      f"Order={(w_order * loss_eigenvalue_order).item():.6f} | Eigen={(w_eigen * loss_coarse_eigenvalues).item():.6f}")

        # === POST-TRAINING: Final normalization ===
        with torch.no_grad():
            corr_raw = self.model(x_feats_all, A_norm_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr
            
            # Normalize per level
            node_offset = 0
            U_pred_normalized = []
            for i, M_t in enumerate(M_t_list):
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
    def refine_eigenvectors(self, U_pred, K, M):
        U = torch.FloatTensor(U_pred).to(self.device)
        K_t = utils.scipy_sparse_to_torch_sparse(K).to(self.device)
        M_t = utils.scipy_sparse_to_torch_sparse(M).to(self.device)
        A = (U.t() @ torch.sparse.mm(K_t, U)).cpu().numpy()
        B = (U.t() @ torch.sparse.mm(M_t, U)).cpu().numpy()
        vals, C = eigh(A, B)
        U_refined = U.cpu().numpy() @ C
        return vals, U_refined