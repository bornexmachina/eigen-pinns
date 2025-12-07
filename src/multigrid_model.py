import numpy as np
from scipy.linalg import eigh
import torch
import torch.nn as nn
import torch.optim as optim
from corrector_model import SimpleCorrector, SpectralCorrector
import utils


class MultigridGNN:
    """Multigrid Graph Neural Network for eigenvector refinement."""
    
    def __init__(self, config):
        """
        Initialize the MultigridGNN.
        
        Args:
            model_type: Type of corrector model to use ('simple' or 'spectral')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = config.model_type.lower()
        self.epochs = config.epochs
        self.lr = config.learning_rate
        self.corr_scale = config.corrector_scale
        self.w_res = config.weight_residual
        self.w_orth = config.weight_orthogonal
        self.w_proj = config.weight_projection
        self.w_trace = config.weight_trace
        self.w_order = config.w_order
        self.w_eigen = config.w_eigen
        self.grad_clip = config.gradient_clipping
        self.weight_decay = config.weight_decay
        self.log_every = config.log_every
        self.hidden_layers = config.hidden_layers
        self.dropout = config.dropout
        self.n_modes = config.n_modes
        
        if self.model_type not in ['simple', 'spectral']:
            raise ValueError(f"model_type must be 'simple' or 'spectral', got '{self.model_type}'")

    def train_multiresolution(self, sampler):
        """
        Train the GNN-based multigrid correction model.
        
        Args:
            Sampler class has the following lists
                X_list: List of node feature matrices (coordinates) for each resolution
                K_list: List of stiffness matrices for each resolution
                M_list: List of mass matrices for each resolution
                U_init_list: List of initial eigenvector estimates for each resolution
                P_list: List of prolongation operators between resolutions
                edge_index_list: List of edge indices for graph structure
            epochs: Number of training epochs
            lr: Learning rate
            corr_scale: Scaling factor for corrections
            w_*: Loss weights for different components
            grad_clip: Gradient clipping threshold
            weight_decay: L2 regularization parameter
            log_every: Logging frequency
            hidden_layers: Number of hidden layers in the network
            dropout: Dropout rate
            
        Returns:
            Refined eigenvectors (numpy array)
        """
        node_offset_list = self._compute_node_offsets(sampler.X_list)
        
        # Step 1: Apply coarse grid correction and compute eigenvalues
        U_CGC_list, lambda_list = self._initialize_cgc_hierarchy(sampler.U_list, sampler.K_list, sampler.M_list, sampler.P_list)
        
        # Step 2: Normalize base eigenvectors with M-norm
        U_CGC_normalized_list = self._normalize_eigenvectors(U_CGC_list, sampler.M_list)
        U_all_CGC_norm = torch.cat(U_CGC_normalized_list, dim=0)
        
        # Step 3: Build physics-informed node features
        x_feats_all, edge_index_all_tensor, A_norm_all = self._build_features(sampler.X_list, U_CGC_normalized_list, lambda_list, sampler.edge_index_list, sampler.K_list, sampler.M_list)
        
        # Step 4: Initialize model and optimizer
        self._initialize_model(x_feats_all.shape[1], self.n_modes, self.hidden_layers, self.dropout)
        optimizer, scheduler = self._create_optimizer(self.lr, self.weight_decay)
        
        # Step 5: Training loop
        self._training_loop(
            x_feats_all, edge_index_all_tensor, A_norm_all, U_all_CGC_norm,
            sampler.K_list, sampler.M_list, lambda_list, node_offset_list,
            optimizer, scheduler)
        
        # Step 6: Generate final predictions
        U_pred_all = self._generate_final_predictions(x_feats_all, edge_index_all_tensor, A_norm_all, U_all_CGC_norm, U_CGC_normalized_list, sampler.M_list)

        return self._refine_final_predictions(sampler, U_pred_all)
    

    def _compute_node_offsets(self, X_list):
        """Compute cumulative node offsets for multi-resolution graphs."""
        return [0] + list(np.cumsum([X.shape[0] for X in X_list[:-1]]))

    def _initialize_cgc_hierarchy(self, U_init_list, K_list, M_list, P_list):
        """Apply coarse grid correction to all fine levels and compute eigenvalues."""
        print("\nApplying Coarse Grid Correction (CGC) to all fine levels...")
        
        U_CGC_list = [torch.FloatTensor(U_init_list[0]).to(self.device)]
        lambda_list = []

        for i in range(1, len(K_list)):
            U_fine = torch.FloatTensor(U_init_list[i]).to(self.device)
            U_CGC, lambda_f = self.apply_coarse_grid_correction(
                U_fine, K_list[i], M_list[i], K_list[i-1], P_list[i-1]
            )
            U_CGC_list.append(U_CGC)
            lambda_list.append(lambda_f)

        # Compute eigenvalues for coarsest level
        lambda_coarse, _ = self.refine_eigenvectors(U_init_list[0], K_list[0], M_list[0])
        lambda_list.insert(0, torch.FloatTensor(lambda_coarse).to(self.device))
        
        return U_CGC_list, lambda_list

    def _normalize_eigenvectors(self, U_CGC_list, M_list):
        """Normalize eigenvectors with respect to M-inner product."""
        U_normalized_list = []
        
        for U_base, M_np in zip(U_CGC_list, M_list):
            M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(self.device)
            Mu = torch.sparse.mm(M_t, U_base)
            norms = torch.sqrt(torch.sum(U_base * Mu, dim=0) + 1e-12)
            U_normalized_list.append(U_base / norms.unsqueeze(0))
        
        return U_normalized_list

    def _build_features(self, X_list, U_normalized_list, lambda_list, edge_index_list, K_list, M_list):
        """Construct physics-informed node features from normalized eigenvectors."""
        print("Building physics-informed features from normalized U_CGC...")
        
        x_feats_all = []
        edge_index_all = []
        
        for i, (X, U_norm, lambdas, edge_index) in enumerate(
            zip(X_list, U_normalized_list, lambda_list, edge_index_list)
        ):
            features = self._compute_level_features(
                X, U_norm, lambdas, edge_index, K_list[i], M_list[i], i, len(K_list)
            )
            print(f"--- features have shape: {features.shape} ---")
            x_feats_all.append(features)
            edge_index_all.append(edge_index)
        
        x_feats_tensor = torch.cat(x_feats_all, dim=0)
        edge_index_tensor = torch.cat(edge_index_all, dim=1)
        
        # Build normalized adjacency matrix for SpectralCorrector
        A_norm = None
        if self.model_type == 'spectral':
            A_norm = utils.build_A_norm(edge_index_tensor, x_feats_tensor.shape[0], self.device)
        
        return x_feats_tensor, edge_index_tensor, A_norm

    def _compute_level_features(self, X, U_norm, lambdas, edge_index, K_np, M_np, level_idx, n_levels):
        """Compute node features for a single resolution level."""
        device = self.device
        
        # Coordinate features
        X_t = torch.FloatTensor(X).to(device)
        
        # Resolution level indicator (coarse=0, fine=n_levels-1)
        res_feat = torch.full((X.shape[0], 1), n_levels - 1 - level_idx, 
                             dtype=torch.float32, device=device)
        
        # Node degree features (normalized)
        row, col = edge_index
        deg = torch.bincount(row, minlength=X.shape[0]).unsqueeze(1).float().to(device)
        deg_feat = deg / (deg.max() + 1e-12)
        
        # Matrix diagonal features
        K_diag = torch.FloatTensor(K_np.diagonal()).to(device).unsqueeze(1)
        M_diag = torch.FloatTensor(M_np.diagonal()).to(device).unsqueeze(1)
        
        # Residual magnitude (normalized)
        K_t = utils.scipy_sparse_to_torch_sparse(K_np).to(device)
        M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
        
        Ku = K_t @ U_norm
        Mu = M_t @ U_norm
        res_vectors = Ku - Mu * lambdas.unsqueeze(0)
        res_mag = torch.norm(res_vectors, dim=1, keepdim=True)
        res_mag_norm = res_mag / (res_mag.max() + 1e-12)
        
        # Local Rayleigh quotient (normalized) - indicates where eigenvalues are wrong
        rayleigh_per_node = (torch.sum(U_norm * Ku, dim=1, keepdim=True) / 
                            (torch.sum(U_norm * Mu, dim=1, keepdim=True) + 1e-12))
        rayleigh_norm = rayleigh_per_node / (lambdas.max() + 1e-12)
        
        # Concatenate all features
        # TO-DO: make features "selectable"
        return torch.cat([
            X_t, res_feat, deg_feat, 
            K_diag, M_diag, 
            res_mag_norm, rayleigh_norm, 
            U_norm
        ], dim=1)

    def _initialize_model(self, input_dim, n_modes, hidden_layers, dropout):
        """Initialize the correction model with small random weights."""
        if self.model is None:
            if self.model_type == 'simple':
                self.model = SimpleCorrector(input_dim, n_modes, hidden_layers, dropout).to(self.device)
            else:  # spectral
                self.model = SpectralCorrector(input_dim, n_modes, hidden_layers, dropout).to(self.device)
            
            # Small random initialization helps escape "do nothing" local minimum
            nn.init.normal_(self.model.net[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.model.net[-1].bias)
            
            print(f"Model initialized ({self.model_type}): input_dim={input_dim}, output_dim={n_modes}")
            print("Applied small random initialization to output layer.")

    def _create_optimizer(self, lr, weight_decay):
        """Create optimizer with learning rate scheduler."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2000, min_lr=1e-6
        )
        return optimizer, scheduler

    def _training_loop(
        self, x_feats, edge_index, A_norm, U_base, K_list, M_list, lambda_list, node_offsets,
        optimizer, scheduler
    ):
        """Execute the main training loop with early stopping."""
        print("\nStarting training loop...")
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 5000
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass with adaptive correction scaling
            corr_raw = self._forward_pass(x_feats, edge_index, A_norm)
            adaptive_scale = self.corr_scale * min(1.0, epoch / 5000.0)
            corr = adaptive_scale * corr_raw
            U_pred = U_base + corr
            
            # Compute losses
            loss_res, loss_orth, lambda_pred = self._compute_residual_ortho_loss(
                U_pred, K_list, M_list, node_offsets, self.w_res, self.w_orth, self.n_modes
            )
            loss_proj, loss_trace, loss_order, loss_eigen = self._compute_eigenvalue_losses(
                M_list, lambda_list[0], lambda_pred, self.w_proj, self.w_trace, self.w_order, self.w_eigen
            )
            
            total_loss = loss_res + loss_orth + loss_proj + loss_trace + loss_order + loss_eigen
            
            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step(total_loss.item())
            
            # Early stopping logic
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > max_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {max_patience} epochs)")
                break
            
            # Logging
            if epoch % self.log_every == 0 or epoch == self.epochs - 1:
                self._log_training_progress(
                    epoch, total_loss, loss_res, loss_orth, 
                    loss_proj, loss_trace, loss_order, loss_eigen, adaptive_scale
                )

    def _forward_pass(self, x_feats, edge_index, A_norm):
        """Execute forward pass through the model (handles both model types)."""
        x_feats = x_feats.to(self.device)
        edge_index = edge_index.to(self.device)
        
        if self.model_type == 'simple':
            return self.model(x_feats, edge_index)
        else:  # spectral
            return self.model(x_feats, A_norm)

    def _compute_residual_ortho_loss(self, U_pred, K_list, M_list, node_offsets, w_res, w_orth, n_modes):
        """
        Compute residual and orthonormality losses.
        Assumes U_pred is already close to M-normalized.
        """
        device = self.device
        loss_residual = torch.tensor(0.0, device=device)
        loss_orthogonal = torch.tensor(0.0, device=device)
        lambda_pred_list = []
        
        for i, (K_np, M_np) in enumerate(zip(K_list, M_list)):
            n_start = node_offsets[i]
            n_end = n_start + K_np.shape[0]
            U_level = U_pred[n_start:n_end]
            
            K_t = utils.scipy_sparse_to_torch_sparse(K_np).to(device)
            M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(device)
            
            Mu = torch.sparse.mm(M_t, U_level)
            Ku = torch.sparse.mm(K_t, U_level)
            
            # Rayleigh quotient
            lambdas = torch.sum(U_level * Ku, dim=0) / (torch.sum(U_level * Mu, dim=0) + 1e-12)
            lambda_pred_list.append(lambdas)
            
            # Residual: ||Ku - λMu||²
            res = Ku - Mu * lambdas.unsqueeze(0)
            loss_residual += torch.mean(res**2)
            
            # Orthonormality: ||U^T M U - I||²
            Gram = U_level.t() @ Mu
            loss_orthogonal += torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes
        
        return w_res * loss_residual, w_orth * loss_orthogonal, lambda_pred_list

    def _compute_eigenvalue_losses(self, M_list, lambda_target, lambda_pred_list, 
                                   w_proj, w_trace, w_order, w_eigen):
        """Compute eigenvalue-related losses (trace, ordering, target matching)."""
        device = self.device
        
        lambdas = lambda_pred_list[0]  # Use coarsest level
        
        # Trace minimization (sum of eigenvalues)
        loss_trace = torch.mean(lambdas)
        
        # Eigenvalue ordering constraint (λ₁ ≤ λ₂ ≤ ... ≤ λₙ)
        lambda_diffs = lambdas[1:] - lambdas[:-1]
        loss_order = torch.sum(torch.relu(-lambda_diffs))
        
        # Match target eigenvalues
        loss_eigen = (torch.mean((lambdas - lambda_target)**2) 
                     if lambda_target is not None else torch.tensor(0.0, device=device))
        
        # Surface integral loss (currently unused, set to zero)
        loss_proj = torch.tensor(0.0, device=device)
        
        return (w_proj * loss_proj, w_trace * loss_trace, 
                w_order * loss_order, w_eigen * loss_eigen)

    def _log_training_progress(self, epoch, total_loss, loss_res, loss_orth, 
                               loss_proj, loss_trace, loss_order, loss_eigen, scale):
        """Print training progress."""
        print(f"Epoch {epoch:4d}: Loss={total_loss.item():.6f} | "
              f"Res={loss_res.item():.6f} | Orth={loss_orth.item():.6f} | "
              f"Mean={loss_proj.item():.6f} | Trace={loss_trace.item():.6f} | "
              f"Order={loss_order.item():.6f} | Eigen={loss_eigen.item():.6f} | "
              f"Scale={scale:.4f}")

    def _generate_final_predictions(self, x_feats, edge_index, A_norm, U_base, U_normalized_list, M_list):
        """Generate final normalized predictions after training."""
        with torch.no_grad():
            corr_raw = self._forward_pass(x_feats, edge_index, A_norm)
            corr = self.corr_scale * corr_raw
            U_pred = U_base + corr
            
            # Normalize each resolution level separately
            node_offset = 0
            U_pred_normalized = []
            
            for i, M_np in enumerate(M_list):
                M_t = utils.scipy_sparse_to_torch_sparse(M_np).to(self.device)
                n_nodes = U_normalized_list[i].shape[0]
                U_level = U_pred[node_offset:node_offset + n_nodes]
                
                Mu = torch.sparse.mm(M_t, U_level)
                norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
                U_level_norm = U_level / norms.unsqueeze(0)
                U_pred_normalized.append(U_level_norm)
                
                node_offset += n_nodes
            
            U_pred_final = torch.cat(U_pred_normalized, dim=0)
        
        return U_pred_final.detach().cpu().numpy()

    def refine_eigenvectors(self, U_pred, K, M):
        """
        Perform Rayleigh-Ritz refinement on eigenvectors.
        
        Args:
            U_pred: Predicted eigenvectors
            K: Stiffness matrix
            M: Mass matrix
            
        Returns:
            Refined eigenvalues and eigenvectors
        """
        U = torch.FloatTensor(U_pred).to(self.device)
        K_t = utils.scipy_sparse_to_torch_sparse(K).to(self.device)
        M_t = utils.scipy_sparse_to_torch_sparse(M).to(self.device)
        
        # Solve generalized eigenvalue problem in reduced space
        A = (U.t() @ torch.sparse.mm(K_t, U)).cpu().numpy()
        B = (U.t() @ torch.sparse.mm(M_t, U)).cpu().numpy()
        vals, C = eigh(A, B)
        
        U_refined = U.cpu().numpy() @ C
        return vals, U_refined

    def apply_coarse_grid_correction(self, U_fine, K_fine, M_fine, K_coarse, P_np):
        """
        Apply single multigrid coarse grid correction step.
        Uses: U_new = U_fine - P * K_coarse⁻¹ * P^T * Residual(U_fine)
        
        Args:
            U_fine: Current eigenvector estimate on fine mesh (N_fine × N_modes)
            K_fine: Fine mesh stiffness matrix
            M_fine: Fine mesh mass matrix
            K_coarse: Coarse mesh stiffness matrix
            P_np: Prolongation operator (N_fine × N_coarse)
            
        Returns:
            Corrected eigenvectors U_CGC and corresponding Rayleigh quotients
        """
        device = self.device
        
        # Compute optimal eigenvalues for current guess (Rayleigh-Ritz)
        lambda_f, _ = self.refine_eigenvectors(U_fine.cpu().numpy(), K_fine, M_fine)
        lambda_f = torch.FloatTensor(lambda_f).to(device)
        
        # Convert matrices to torch sparse format
        K_f_t = utils.scipy_sparse_to_torch_sparse(K_fine).to(device)
        M_f_t = utils.scipy_sparse_to_torch_sparse(M_fine).to(device)
        P_t = utils.scipy_sparse_to_torch_sparse(P_np).to(device)
        
        # Compute fine-grid residual: R_f = K_f U_f - λ_f M_f U_f
        R_f = K_f_t @ U_fine - M_f_t @ U_fine @ torch.diag(lambda_f)
        
        # Restrict residual to coarse grid: R_c = P^T R_f
        R_c = torch.sparse.mm(P_t.t(), R_f)
        
        # Solve coarse problem: K_c δu_c = R_c
        K_c_dense = torch.FloatTensor(K_coarse.todense()).to(device)
        delta_u_c = torch.linalg.solve(K_c_dense, R_c)
        
        # Prolongate correction and apply: U_CGC = U_fine - P δu_c
        delta_u_f = torch.sparse.mm(P_t, delta_u_c)
        U_CGC = U_fine - delta_u_f
        
        return U_CGC.detach(), lambda_f.detach()
    
    def _refine_final_predictions(self, sampler, U_pred_all):
        # Extract finest level correctly ===
        hierarchy = sampler.actual_hierarchy
        node_offset = sum(hierarchy[:-1])
        print(f"\n--- Extracting finest level ---")
        print(f"Node offset: {node_offset}")
        print(f"Total nodes in U_pred_all: {U_pred_all.shape[0]}")
        print(f"Expected finest level nodes: {hierarchy[-1]}")
        
        U_finest = U_pred_all[node_offset:node_offset + hierarchy[-1]]
        print(f"Extracted U_finest shape: {U_finest.shape}")
        
        # Verify extraction
        assert U_finest.shape[0] == hierarchy[-1], f"Mismatch! Got {U_finest.shape[0]}, expected {hierarchy[-1]}"

        K_finest = sampler.K_list[-1]
        M_finest = sampler.M_list[-1]

        # Perform Rayleigh-Ritz refinement on finest level ===
        print("\n--- Rayleigh-Ritz refinement on finest level ---")
        vals_refined, U_refined = self.refine_eigenvectors(U_finest, K_finest, M_finest)
        print(f"Refined eigenvalues (first 10): {np.round(vals_refined[:10], 6)}")

        return U_refined
