import numpy as np
from scipy.linalg import eigh
import torch
import torch.nn as nn # Added for init
import torch.optim as optim
from corrector_model import SimpleCorrector
from corrector_model import SpectralCorrector
import utils


class MultigridGNN:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None

    # ------------------------
    # Physics-informed GNN training (V-Cycle enhanced)
    # ------------------------
    def train_multiresolution(self, X_list, K_list, M_list, U_init_list, P_list, # P_list replaces lambda_coarse
                              edge_index_list, epochs, lr, corr_scale, 
                              w_res, w_orth, w_proj, w_trace, w_order, w_eigen, 
                              grad_clip, weight_decay, log_every, hidden_layers, dropout):
        device = self.device
        n_modes = U_init_list[0].shape[1]
        node_offset_list = [0] + list(np.cumsum([X.shape[0] for X in X_list[:-1]]))

        # 1. Initialize U_CGC_list and calculate all needed eigenvalues
        # Coarsest level (level 0) is the exact solution U_exact, used as U_CGC_0
        U_CGC_list = [torch.FloatTensor(U_init_list[0]).to(device)] 
        lambda_list = []

        # --- Apply Coarse Grid Correction (CGC) to all fine levels (i > 0) ---
        print("\nApplying Coarse Grid Correction (CGC) to all fine levels...")
        for i in range(1, len(K_list)): # Start from level 1 (the first fine mesh)
            U_fine = torch.FloatTensor(U_init_list[i]).to(device)
            
            # The coarse level for the current fine level 'i' is 'i-1'.
            K_coarse = K_list[i-1]
            P_np = P_list[i-1] # Prolongation from i-1 to i (P_{i-1 -> i})
            
            # U_CGC is the Multigrid-corrected vector. lambda_f is its Rayleigh quotient.
            U_CGC, lambda_f = self.apply_coarse_grid_correction(U_fine, K_list[i], M_list[i], K_coarse, P_np)
            U_CGC_list.append(U_CGC)
            lambda_list.append(lambda_f)

        # Get lambda for the coarsest level (U_exact)
        lambda_coarse, _ = self.refine_eigenvectors(U_init_list[0], K_list[0], M_list[0])
        lambda_list.insert(0, torch.FloatTensor(lambda_coarse).to(device))
        
        # Concatenate all corrected vectors (U_CGC)
        U_all_CGC = torch.cat(U_CGC_list, dim=0).to(device)
        
        # 2. Feature Construction using U_CGC as the base solution
        x_feats_all, edge_index_all = [], []
        
        print("Building simplified physics-informed features (Local Residuals) from U_CGC...")
        for i, (X, U_base, lambda_mode, edge_index) in enumerate(zip(X_list, U_CGC_list, lambda_list, edge_index_list)):
            
            # Features 1-3: Geometry and Local Connectivity (Keep)
            X_t = torch.FloatTensor(X).to(device)
            res_feat = np.full((X.shape[0], 1), len(K_list) - 1 - i).astype(np.float32)
            res_feat = torch.FloatTensor(res_feat).to(device)
            
            row, col = edge_index
            deg = torch.bincount(row, minlength=X.shape[0]).unsqueeze(1).to(device).to(X_t.dtype)
            deg_feat = deg / (deg.max() + 1e-12) # Normalized degree feature

            # Features 4-5: Diagonal K and M (Keep)
            K_t = utils.scipy_sparse_to_torch_sparse(K_list[i]).to(device)
            M_t = utils.scipy_sparse_to_torch_sparse(M_list[i]).to(device)

            K_diag = torch.FloatTensor(K_list[i].diagonal()).to(device).unsqueeze(1)
            M_diag = torch.FloatTensor(M_list[i].diagonal()).to(device).unsqueeze(1)
            
            # --- Feature 6: Core Feature - Local Residual Magnitude ---
            # Residual matrix: R_f = (K U_CGC - M U_CGC Lambda)
            res_vectors = K_t @ U_base - M_t @ (U_base @ torch.diag(lambda_mode))
            
            # IMPROVEMENT: Use per-node residual magnitude instead of global mean
            # This allows the GNN to see WHERE the error is high.
            # Shape (N_nodes, 1)
            res_mag_feat = torch.norm(res_vectors, dim=1, keepdim=True)
            # ============================================

            # Feature 7: The base mode values (U_CGC) - Input for correction
            U_base_feat = U_base.to(device) 
            
            # Combined Feature Vector: DIM = 3 + 1 + 1 + 2 + 1 + n_modes = 8 + n_modes
            features = [
                X_t, 
                res_feat, 
                deg_feat, 
                K_diag, M_diag, 
                res_mag_feat,
                U_base_feat
            ]
            x_feats = torch.cat(features, dim=1)
            x_feats_all.append(x_feats)
            edge_index_all.append(edge_index)

        # Concatenate features and build normalized adjacency matrix
        x_feats_all = torch.cat(x_feats_all, dim=0)
        edge_index_all_tensor = torch.cat(edge_index_all, dim=1)
        A_norm_all = utils.build_A_norm(edge_index_all_tensor, x_feats_all.shape[0], device)

        # --- Initialize Model ---
        input_dim = x_feats_all.shape[1]
        if self.model is None:
            # Assuming SpectralCorrector is the GNN architecture
            from corrector_model import SpectralCorrector 
            self.model = SpectralCorrector(input_dim, n_modes, hidden_layers, dropout).to(device)
            
            # === CRITICAL FIX: ZERO INITIALIZATION ===
            # Initialize the last output layer to zero. 
            # This ensures the model starts by predicting 0 correction,
            # preserving the U_CGC baseline and avoiding high-frequency noise explosion.
            nn.init.zeros_(self.model.net[-1].weight)
            nn.init.zeros_(self.model.net[-1].bias)
            
            print(f"Model initialized with input dim: {input_dim}. Target dim: {n_modes}")
            print("Applied Zero-Initialization to output layer to prevent mode collapse.")

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 3. Training Loop
        print("\nStarting training loop (GNN learns High-Frequency Correction on U_CGC)...")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Predict raw correction
            corr_raw = self.model(x_feats_all, A_norm_all)
            corr = corr_scale * corr_raw
            
            # Final prediction: U_pred = U_CGC + correction
            U_pred = U_all_CGC + corr
            
            # Normalize U_pred columns (crucial for loss calculation)
            U_pred_norm, _ = utils.normalize_columns_torch(U_pred, eps=1e-12) # Assumes utils.normalize_columns_torch exists

            # --- Calculate Losses ---
            # These loss functions are assumed to be defined as they were in the original code,
            # now using U_pred_norm and the derived lambda_list/U_CGC_list
            
            # L_Residual and L_Orthogonal (over all levels)
            loss_res, loss_orth, lambda_pred = self.calculate_residual_and_ortho_loss(
                U_pred_norm, K_list, M_list, node_offset_list, w_res, w_orth, n_modes
            )
            
            # L_Projection, L_Trace, L_Order, L_Eigen (coarsest level)
            loss_proj, loss_trace, loss_order, loss_eigen = self.calculate_projection_and_eigen_loss(
                U_pred_norm, M_list, node_offset_list, lambda_list[0], U_CGC_list[0], 
                w_proj, w_trace, w_order, w_eigen, n_modes
            )

            loss = loss_res + loss_orth + loss_proj + loss_trace + loss_order + loss_eigen
            
            # Backpropagation (Keep existing gradient clipping logic)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            # Logging... (your existing logging logic should be here)
            if epoch % log_every == 0 or epoch == epochs-1:
                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f} | "
                      f"Res={loss_res.item():.6f} | Orth={loss_orth.item():.6f} | "
                      f"Mean={loss_proj.item():.6f} | Trace={loss_trace.item():.6f} | "
                      f"Order={loss_order.item():.6f} | Eigen={loss_eigen.item():.6f}")
        
        # --- POST-TRAINING: Final prediction and normalization ---
        with torch.no_grad():
            corr_raw = self.model(x_feats_all, A_norm_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_CGC + corr # Base solution is U_CGC
            
            # Normalize per level
            node_offset = 0
            U_pred_normalized = []
            for i, M_np in enumerate(M_list):
                M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
                n_nodes = U_CGC_list[i].shape[0] 
                U_level = U_pred[node_offset:node_offset+n_nodes]
                
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_norm = U_level / norms.unsqueeze(0)
                U_pred_normalized.append(U_level_norm)
                
                node_offset += n_nodes
            
            U_pred_final = torch.cat(U_pred_normalized, dim=0)

        return U_pred_final.detach().cpu().numpy()
    

    # ------------------------
    # Coarse Grid Correction (CGC) step
    # ------------------------
    def apply_coarse_grid_correction(self, U_fine, K_fine, M_fine, K_coarse, P_np):
        """
        Performs a single Multigrid Coarse Grid Correction step.
        U_new = U_fine - P * K_coarse_inv * P^T * Residual(U_fine)
        
        Args:
            U_fine (torch.Tensor): Current eigenvector estimate on the fine mesh (N_fine x N_modes).
            K_fine (scipy.sparse.csr_matrix): Stiffness matrix on the fine mesh.
            M_fine (scipy.sparse.csr_matrix): Mass matrix on the fine mesh.
            K_coarse (scipy.sparse.csr_matrix): Stiffness matrix on the coarse mesh.
            P_np (scipy.sparse.coo_matrix): Prolongation operator (N_fine x N_coarse).
            
        Returns:
            torch.Tensor: The corrected eigenvector estimate U_CGC (N_fine x N_modes).
            torch.Tensor: The Rayleigh quotients lambda_f used for residual calculation.
        """
        device = self.device
        
        # 1. Compute fine-grid residual R_f = K_f U_f - lambda_f M_f U_f
        # Calculate the best lambda for the current U_fine guess (Rayleigh-Ritz)
        lambda_f, _ = self.refine_eigenvectors(U_fine.cpu().numpy(), K_fine, M_fine)
        lambda_f = torch.FloatTensor(lambda_f).to(device)
        
        K_f_t = utils.scipy_sparse_to_torch_sparse(K_fine).to(device)
        M_f_t = utils.scipy_sparse_to_torch_sparse(M_fine).to(device)
        
        # Residual matrix R_f (N_fine x N_modes)
        R_f = K_f_t @ U_fine - M_f_t @ U_fine @ torch.diag(lambda_f)

        # 2. Convert Prolongation operator P to sparse torch tensor
        P_t = utils.scipy_sparse_to_torch_sparse(P_np).to(device)
        
        # 3. Restriction R_c = P^T @ R_f (Galerkin Restriction)
        R_c = torch.sparse.mm(P_t.t(), R_f) 

        # 4. Coarse Solve: K_c * delta_u_c = R_c. Solve for delta_u_c
        # Note: Direct solution (linalg.solve) assumes K_coarse is small and non-singular.
        K_c_dense = torch.FloatTensor(K_coarse.todense()).to(device)
        delta_u_c = torch.linalg.solve(K_c_dense, R_c) # (N_coarse x N_modes)

        # 5. Prolongation: delta_u_f = P @ delta_u_c
        delta_u_f = torch.sparse.mm(P_t, delta_u_c) # (N_fine x N_modes)

        # 6. Apply correction: U_CGC = U_fine - delta_u_f
        U_CGC = U_fine - delta_u_f
        
        return U_CGC.detach(), lambda_f.detach()
    
    # ------------------------
    # Loss Calculation Helpers (Refactored)
    # ------------------------
    def calculate_residual_and_ortho_loss(self, U_pred, K_t_list, M_t_list, node_offset_list, w_res, w_orth, n_modes):
        device = self.device
        loss_residual = torch.tensor(0.0, device=device)
        loss_orthogonal = torch.tensor(0.0, device=device)
        lambda_pred_list = []
        
        for i, (K_t, M_t) in enumerate(zip(K_t_list, M_t_list)):
            n_start = node_offset_list[i]
            n_end = n_start + K_t.shape[0]
            U_level = U_pred[n_start:n_end]
            
            # --- STRUCTURAL FIX: M-NORMALIZE U_level BEFORE CALCULATING LOSS ---
            
            # 1. Calculate M-norm of the current modes (for normalization)
            # Mu is needed for both norm calculation and the Gram matrix
            Mu = torch.sparse.mm(utils.scipy_sparse_to_torch_sparse(M_t).to(device), U_level)
            
            # Calculate norms: sqrt(U^T M U) (diagonal terms)
            norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12) 
            
            # Normalize U_level (U_level_norm^T M U_level_norm should be close to I)
            U_level_norm = U_level / norms.unsqueeze(0)
            
            # Recalculate Mu and Ku with the M-normalized vectors
            Mu_norm = torch.sparse.mm(utils.scipy_sparse_to_torch_sparse(M_t).to(device), U_level_norm)
            Ku_norm = torch.sparse.mm(utils.scipy_sparse_to_torch_sparse(K_t).to(device), U_level_norm)

            # -------------------------------------------------------------------
            
            # Rayleigh quotient
            # Note: Since U_level_norm is M-normalized, the denominator U^T M U is ~1.
            lambdas = torch.sum(U_level_norm * Ku_norm, dim=0) / (torch.sum(U_level_norm * Mu_norm, dim=0) + 1e-12)
            lambda_pred_list.append(lambdas)
            
            # Residual (Calculated using M-normalized vectors)
            res = Ku_norm - Mu_norm * lambdas.unsqueeze(0)
            loss_residual += torch.mean(res**2) # Summing over all levels
            
            # Orthonormality (Crucial: Now uses U_level_norm)
            # Gram = U_norm^T M U_norm. We expect this to be close to Identity.
            Gram = U_level_norm.t() @ Mu_norm
            loss_orthogonal += torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes

        return w_res * loss_residual, w_orth * loss_orthogonal, lambda_pred_list

    def train_multiresolution(self, X_list, K_list, M_list, U_init_list, P_list,
                            edge_index_list, epochs, lr, corr_scale, 
                            w_res, w_orth, w_proj, w_trace, w_order, w_eigen, 
                            grad_clip, weight_decay, log_every, hidden_layers, dropout):
        device = self.device
        n_modes = U_init_list[0].shape[1]
        node_offset_list = [0] + list(np.cumsum([X.shape[0] for X in X_list[:-1]]))

        # 1. Initialize U_CGC_list and calculate all needed eigenvalues
        U_CGC_list = [torch.FloatTensor(U_init_list[0]).to(device)] 
        lambda_list = []

        print("\nApplying Coarse Grid Correction (CGC) to all fine levels...")
        for i in range(1, len(K_list)):
            U_fine = torch.FloatTensor(U_init_list[i]).to(device)
            K_coarse = K_list[i-1]
            P_np = P_list[i-1]
            
            U_CGC, lambda_f = self.apply_coarse_grid_correction(U_fine, K_list[i], M_list[i], K_coarse, P_np)
            U_CGC_list.append(U_CGC)
            lambda_list.append(lambda_f)

        lambda_coarse, _ = self.refine_eigenvectors(U_init_list[0], K_list[0], M_list[0])
        lambda_list.insert(0, torch.FloatTensor(lambda_coarse).to(device))
        
        # === CRITICAL FIX 1: Pre-normalize U_CGC with M-norm ===
        U_CGC_normalized_list = []
        for i, (U_base, M_np) in enumerate(zip(U_CGC_list, M_list)):
            M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
            Mu = torch.sparse.mm(M_t, U_base)
            norms = torch.sqrt(torch.sum(U_base * Mu, dim=0) + 1e-12)
            U_base_norm = U_base / norms.unsqueeze(0)
            U_CGC_normalized_list.append(U_base_norm)
        
        U_all_CGC_norm = torch.cat(U_CGC_normalized_list, dim=0).to(device)
        
        # 2. Feature Construction with IMPROVED scaling
        x_feats_all, edge_index_all = [], []
        
        print("Building physics-informed features from normalized U_CGC...")
        for i, (X, U_base_norm, lambda_mode, edge_index) in enumerate(zip(X_list, U_CGC_normalized_list, lambda_list, edge_index_list)):
            
            X_t = torch.FloatTensor(X).to(device)
            res_feat = np.full((X.shape[0], 1), len(K_list) - 1 - i).astype(np.float32)
            res_feat = torch.FloatTensor(res_feat).to(device)
            
            row, col = edge_index
            deg = torch.bincount(row, minlength=X.shape[0]).unsqueeze(1).to(device).to(X_t.dtype)
            deg_feat = deg / (deg.max() + 1e-12)

            K_t = utils.scipy_sparse_to_torch_sparse(K_list[i]).to(device)
            M_t = utils.scipy_sparse_to_torch_sparse(M_list[i]).to(device)

            K_diag = torch.FloatTensor(K_list[i].diagonal()).to(device).unsqueeze(1)
            M_diag = torch.FloatTensor(M_list[i].diagonal()).to(device).unsqueeze(1)
            
            # === CRITICAL FIX 2: Use normalized vectors for residual calculation ===
            Ku = K_t @ U_base_norm
            Mu = M_t @ U_base_norm
            res_vectors = Ku - Mu * lambda_mode.unsqueeze(0)
            
            # Normalize residual magnitude to [0, 1] range for better gradient flow
            res_mag = torch.norm(res_vectors, dim=1, keepdim=True)
            res_mag_normalized = res_mag / (res_mag.max() + 1e-12)
            
            # === CRITICAL FIX 3: Add Rayleigh quotient per node as feature ===
            # This tells the network WHERE eigenvalues are wrong
            rayleigh_per_node = torch.sum(U_base_norm * Ku, dim=1, keepdim=True) / \
                            (torch.sum(U_base_norm * Mu, dim=1, keepdim=True) + 1e-12)
            rayleigh_normalized = rayleigh_per_node / (lambda_mode.max() + 1e-12)

            U_base_feat = U_base_norm.to(device) 
            
            features = [
                X_t, 
                res_feat, 
                deg_feat, 
                K_diag, M_diag, 
                res_mag_normalized,  # Normalized residual magnitude
                rayleigh_normalized,  # NEW: Local Rayleigh quotient
                U_base_feat
            ]
            x_feats = torch.cat(features, dim=1)
            x_feats_all.append(x_feats)
            edge_index_all.append(edge_index)

        x_feats_all = torch.cat(x_feats_all, dim=0)
        edge_index_all_tensor = torch.cat(edge_index_all, dim=1)
        A_norm_all = utils.build_A_norm(edge_index_all_tensor, x_feats_all.shape[0], device)

        # --- Initialize Model ---
        input_dim = x_feats_all.shape[1]
        if self.model is None:
            from corrector_model import SpectralCorrector 
            self.model = SpectralCorrector(input_dim, n_modes, hidden_layers, dropout).to(device)
            
            # === CRITICAL FIX 4: Small random initialization instead of zero ===
            # This helps the network escape the "do nothing" local minimum
            nn.init.normal_(self.model.net[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.model.net[-1].bias)
            
            print(f"Model initialized with input dim: {input_dim}. Target dim: {n_modes}")
            print("Applied small random initialization to output layer.")

        # === CRITICAL FIX 5: Use a learning rate scheduler ===
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                        patience=2000, min_lr=1e-6)

        # 3. Training Loop
        print("\nStarting training loop...")
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            corr_raw = self.model(x_feats_all, A_norm_all)
            
            # === CRITICAL FIX 6: Adaptive correction scale ===
            # Start with smaller corrections and gradually increase
            adaptive_scale = corr_scale * min(1.0, epoch / 5000.0)
            corr = adaptive_scale * corr_raw
            
            # Final prediction: U_pred = U_CGC_norm + correction
            U_pred = U_all_CGC_norm + corr
            
            # === CRITICAL FIX 7: No additional normalization during training ===
            # Let the network learn to produce already-normalized corrections
            
            # Calculate Losses
            loss_res, loss_orth, lambda_pred = self.calculate_residual_and_ortho_loss_v2(
                U_pred, K_list, M_list, node_offset_list, w_res, w_orth, n_modes
            )
            
            loss_proj, loss_trace, loss_order, loss_eigen = self.calculate_projection_and_eigen_loss(
                U_pred, M_list, node_offset_list, lambda_list[0], lambda_pred, 
                w_proj, w_trace, w_order, w_eigen, n_modes
            )

            loss = loss_res + loss_orth + loss_proj + loss_trace + loss_order + loss_eigen
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > 5000:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for 5000 epochs)")
                break

            if epoch % log_every == 0 or epoch == epochs-1:
                print(f"Epoch {epoch:4d}: Loss={loss.item():.6f} | "
                    f"Res={loss_res.item():.6f} | Orth={loss_orth.item():.6f} | "
                    f"Mean={loss_proj.item():.6f} | Trace={loss_trace.item():.6f} | "
                    f"Order={loss_order.item():.6f} | Eigen={loss_eigen.item():.6f} | "
                    f"Scale={adaptive_scale:.4f}")
        
        # POST-TRAINING: Final prediction and normalization
        with torch.no_grad():
            corr_raw = self.model(x_feats_all, A_norm_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_CGC_norm + corr
            
            # Final normalization per level
            node_offset = 0
            U_pred_normalized = []
            for i, M_np in enumerate(M_list):
                M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
                n_nodes = U_CGC_normalized_list[i].shape[0] 
                U_level = U_pred[node_offset:node_offset+n_nodes]
                
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_norm = U_level / norms.unsqueeze(0)
                U_pred_normalized.append(U_level_norm)
                
                node_offset += n_nodes
            
            U_pred_final = torch.cat(U_pred_normalized, dim=0)

        return U_pred_final.detach().cpu().numpy()


    # === NEW: Improved loss calculation without internal normalization ===
    def calculate_residual_and_ortho_loss_v2(self, U_pred, K_list, M_list, node_offset_list, w_res, w_orth, n_modes):
        """
        Calculate losses WITHOUT normalizing U_pred inside the loop.
        Assumes U_pred is already close to M-normalized.
        """
        device = self.device
        loss_residual = torch.tensor(0.0, device=device)
        loss_orthogonal = torch.tensor(0.0, device=device)
        lambda_pred_list = []
        
        for i, (K_np, M_np) in enumerate(zip(K_list, M_list)):
            n_start = node_offset_list[i]
            n_end = n_start + K_np.shape[0]
            U_level = U_pred[n_start:n_end]
            
            K_t = utils.scipy_sparse_to_torch_sparse(K_np).to(device)
            M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
            
            Mu = torch.sparse.mm(M_t, U_level)
            Ku = torch.sparse.mm(K_t, U_level)
            
            # Rayleigh quotient (no normalization needed if U is already M-normalized)
            lambdas = torch.sum(U_level * Ku, dim=0) / (torch.sum(U_level * Mu, dim=0) + 1e-12)
            lambda_pred_list.append(lambdas)
            
            # Residual
            res = Ku - Mu * lambdas.unsqueeze(0)
            loss_residual += torch.mean(res**2)
            
            # Orthonormality
            Gram = U_level.t() @ Mu
            loss_orthogonal += torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes

        return w_res * loss_residual, w_orth * loss_orthogonal, lambda_pred_list
    

    def calculate_projection_and_eigen_loss(self, U_pred_norm, M_t_list, node_offset_list, lambda_target, lambda_pred_list, w_proj, w_trace, w_order, w_eigen, n_modes):
        device = self.device
        loss_surface_integral = torch.tensor(0.0, device=device)
        loss_trace = torch.tensor(0.0, device=device)
        loss_eigenvalue_order = torch.tensor(0.0, device=device)
        loss_coarse_eigenvalues = torch.tensor(0.0, device=device)
        i = 0
        lambdas = lambda_pred_list[i]
        n_nodes = M_t_list[i].shape[0]
        loss_trace += torch.mean(lambdas)
        lambda_diffs = lambdas[1:] - lambdas[:-1]
        loss_eigenvalue_order += torch.sum(torch.relu(-lambda_diffs))
        if lambda_target is not None:
            loss_coarse_eigenvalues += torch.mean((lambdas - lambda_target)**2)
        return (w_proj * loss_surface_integral, w_trace * loss_trace, w_order * loss_eigenvalue_order, w_eigen * loss_coarse_eigenvalues)

    # ------------------------
    # Rayleigh-Ritz refinement
    # ------------------------
    def refine_eigenvectors(self, U_pred, K, M):
        U = torch.FloatTensor(U_pred).to(self.device)
        K_t = utils.scipy_sparse_to_torch_sparse(K).to(self.device)
        M_t = utils.scipy_sparse_to_torch_sparse(M).to(self.device)
        A = (U.t() @ torch.sparse.mm(K_t, U)).cpu().numpy()
        B = (U.t() @ torch.sparse.mm(M_t, U)).cpu().numpy()

        print(f"------- A: {A.shape} -- B: {B.shape} ------")

        vals, C = eigh(A, B)
        U_refined = U.cpu().numpy() @ C
        return vals, U_refined