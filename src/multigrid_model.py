import math
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

    def schedule_curriculum(self, epoch, epochs):
        """
        Implements a robust curriculum by very slowly and smoothly increasing 
        the weights of the 'Orthogonal' and 'Surface Integral' terms 
        (which appear to be the sources of instability).

        All other successful loss terms are maintained at a constant 1.0 weight.
        """
        
        # --- Define Ramp Parameters ---
        
        # We want a very slow ramp over a large fraction of the total epochs (e.g., 80%)
        Ramp_End_Epoch = int(0.80 * epochs) 
        Ramp_Start_Epoch = int(0.05 * epochs)  # Start ramp early

        # The steepness parameter (tau) for the Tanh function. 
        # A larger tau means a shallower, slower ramp.
        # W is the transition period length
        W = Ramp_End_Epoch - Ramp_Start_Epoch
        tau = W / 10.0 

        # --- Smooth Weight Calculation ---

        def _calculate_tanh_weight(current_epoch, start_epoch, end_epoch, ramp_tau):
            """Calculates a smooth Tanh weight that transitions from 0 to 1."""
            
            # Calculate the midpoint of the transition
            e_mid = start_epoch + (end_epoch - start_epoch) / 2
            
            input_val = (current_epoch - e_mid) / ramp_tau
            
            # Clip to prevent numerical overflow
            input_val = np.clip(input_val, -10.0, 10.0) 
            
            # Tanh-based Sigmoid: transitions from 0 to 1
            weight = 0.5 * (1.0 + math.tanh(input_val))
            
            # Ensure exact 0 or 1 boundaries for stability
            if current_epoch <= start_epoch:
                return 0.0
            if current_epoch >= end_epoch:
                return 1.0

            return weight

        # --- Apply Ramps ---

        # Start the orthogonal loss ramp immediately (or very early)
        w_orthogonal = _calculate_tanh_weight(
            epoch, 
            Ramp_Start_Epoch, # Start 5% into training
            Ramp_End_Epoch,   # End 80% into training
            tau
        )
        
        # Start the surface integral loss ramp later, but still gradually
        w_surface_integral = _calculate_tanh_weight(
            epoch, 
            int(0.33 * epochs), # Start 33% into training
            epochs,             # End at the end of training
            tau
        )

        # Assuming 'Mean' is related to 'Surface_Integral' and should ramp up similarly, 
        # or perhaps even later if it's the most sensitive term.
        w_mean = _calculate_tanh_weight(
            epoch, 
            int(0.50 * epochs), # Start 50% into training
            epochs,             # End at the end of training
            tau
        )
        
        # --- Return Final Weights ---
        
        # The other loss terms are maintained at a constant 1.0 (or their successfully used weights)
        return {
            'loss_residual': 1.0, 
            'loss_orthogonal': w_orthogonal, 
            'loss_surface_integral': w_surface_integral, 
            'loss_trace': 1.0, 
            'loss_eigenvalue_order': 1.0, 
            'loss_coarse_eigenvalues': 1.0,
            'loss_mean': w_mean # Include the mean term which was missing from your initial plan
        }

    # ------------------------
    # Physics-informed GNN training
    # ------------------------
    def train_multiresolution(self, X_list, U_init_list, edge_index_list, lambda_coarse, epochs, lr, corr_scale, w_res, w_orth, w_proj, w_trace, w_order, w_eigen, grad_clip, weight_decay, log_every, hidden_layers, dropout):
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
            self.model = SimpleCorrector(in_dim, n_modes, hidden_layers, dropout).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=lr, weight_decay=weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',         # Monitor the loss (minimized)
            factor=0.5,         # Halve the learning rate
            patience=20,        # Wait 20 epochs for no improvement
            min_lr=1e-6         # Don't let LR go below this value
        )
        
        self.model.train()

        # Precompute Laplacians per level
        L_list, M_list = [], []
        node_offset = 0
        for X in X_list:
            L, M = robust_laplacian.point_cloud_laplacian(X)
            L_list.append(utils.scipy_sparse_to_torch_sparse(L).to(device))
            M_list.append(utils.scipy_sparse_to_torch_sparse(M).to(device))

        if lambda_coarse is not None:
            lambda_target = torch.FloatTensor(lambda_coarse).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            corr_raw = self.model(x_feats_all, edge_index_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr

            # Physics-informed loss
            loss_residual = 0.0
            loss_orthogonal = 0.0
            loss_surface_integral = 0.0
            loss_trace = 0.0
            loss_eigenvalue_order = 0.0
            loss_coarse_eigenvalues = 0.0

            loss_curriculum = self.schedule_curriculum(ep, epochs)

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
                #num = torch.sum(U_level_normalized * Lu_norm, dim=0)
                #den = torch.sum(U_level_normalized * Mu_norm, dim=0) + 1e-12
                #lambdas = num / den
                lambdas = torch.sum(U_level_normalized * Lu_norm, dim=0) / (torch.sum(U_level_normalized * Mu_norm, dim=0) + 1e-12)
                res = Lu_norm - Mu_norm * lambdas.unsqueeze(0)
                loss_residual += torch.mean(res**2) * loss_curriculum['loss_residual']
                
                # Orthonormality (should be near-perfect with normalization)
                Gram = U_level_normalized.t() @ Mu_norm
                # L_orth = torch.mean((Gram - torch.eye(n_modes, device=device))**2)
                loss_orthogonal += (torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes) * loss_curriculum['loss_orthogonal']

                # Zero-mean constraint for modes 1+ (NOT mode 0)
                # Mode 0 is constant, modes 1+ should be zero-mean
                ones = torch.ones(n_nodes, 1, device=device)
                if n_modes > 1:
                    # Project out constant component from modes 1 onwards
                    mean_constraint = ones.t() @ Mu_norm[:, 1:]  # Shape: (1, n_modes-1)
                    loss_surface_integral += torch.mean(mean_constraint**2) * loss_curriculum['loss_surface_integral']
                else:
                    loss_surface_integral += torch.tensor(0.0, device=device) * loss_curriculum['loss_surface_integral']

                # Trace minimalization to force smaller modes
                loss_trace += torch.mean(lambdas) * loss_curriculum['loss_trace']

                lambda_diffs = lambdas[1:] - lambdas[:-1]
                loss_eigenvalue_order += torch.sum(torch.relu(-lambda_diffs)) * loss_curriculum['loss_eigenvalue_order']

                if i == 0 and lambda_coarse is not None:
                    loss_coarse_eigenvalues += torch.mean((lambdas - lambda_target)**2) * loss_curriculum['loss_coarse_eigenvalues']
                else:
                    loss_coarse_eigenvalues += torch.tensor(0.0, device=device) * loss_curriculum['loss_coarse_eigenvalues']

                #loss += w_res * L_res + w_orth * L_orth + w_proj * L_mean + w_trace * L_trace + w_order * L_order + w_eigen * L_eigen
                node_offset += n_nodes

            loss = w_res * loss_residual + w_orth * loss_orthogonal + w_proj * loss_surface_integral + w_trace * loss_trace + w_order * loss_eigenvalue_order + w_eigen * loss_coarse_eigenvalues
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, 
                                                    self.model.parameters()), grad_clip)
            optimizer.step()
            scheduler.step(loss)

            if ep % log_every == 0 or ep == epochs-1:
                print(f"Epoch {ep:4d}: Loss={loss.item():.6f} | "
                      f"Res={(w_res * loss_residual).item():.6f} | Orth={(w_orth * loss_orthogonal).item():.6f} | "
                      f"Mean={(w_proj * loss_surface_integral).item():.6f} | Trace={(w_trace * loss_trace).item():.6f} | "
                      f"Order={(w_order * loss_eigenvalue_order).item():.6f} | Eigen={(w_eigen * loss_coarse_eigenvalues).item():.6f}")

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