import numpy as np
import torch
import torch.optim as optim


class CurriculumScheduler:
    """
    Manages curriculum learning phases for eigenmode learning.
    
    Phase 1 (0-20%): PDE residual only - learn to satisfy Lu = λMu
    Phase 2 (20-60%): Add orthonormality gradually - make modes independent  
    Phase 3 (60-100%): Full physics - balance all objectives
    """
    
    def __init__(self, total_epochs, 
                 w_res_base=1000.0, w_orth_base=100.0, 
                 w_proj_base=10.0, w_order_base=1.0, w_eigen_base=1.0):
        self.total_epochs = total_epochs
        self.w_res_base = w_res_base
        self.w_orth_base = w_orth_base
        self.w_proj_base = w_proj_base
        self.w_order_base = w_order_base
        self.w_eigen_base = w_eigen_base
        
        # Phase boundaries
        self.phase1_end = int(0.2 * total_epochs)
        self.phase2_end = int(0.6 * total_epochs)
    
    def get_weights(self, epoch):
        """Return loss weights for current epoch."""
        
        # Phase 1: ONLY PDE residual and ordering
        # Learn: "what does an eigenmode look like?"
        if epoch < self.phase1_end:
            alpha = epoch / self.phase1_end
            return {
                'w_res': self.w_res_base * 10.0,  # MUCH STRONGER residual weight
                'w_orth': 0.0,  # No orthonormality yet!
                'w_proj': 0.0,  # No projection yet!
                'w_order': self.w_order_base * (10.0 + 10.0 * alpha),  # Strong ordering
                'w_eigen': self.w_eigen_base * 50.0  # MUCH stronger coarse matching
            }
        
        # Phase 2: Introduce orthonormality
        # Learn: "make modes independent while staying eigenmodes"
        elif epoch < self.phase2_end:
            alpha = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return {
                'w_res': self.w_res_base * (10.0 - 8.0 * alpha),  # Keep strong, decay slowly
                'w_orth': alpha**0.5 * self.w_orth_base,  # Sqrt ramp (gradual)
                'w_proj': alpha**0.5 * self.w_proj_base,  # Sqrt ramp (gradual)
                'w_order': self.w_order_base * (20.0 - 15.0 * alpha),  # Still strong
                'w_eigen': self.w_eigen_base * (50.0 - 40.0 * alpha)  # Decay slowly
            }
        
        # Phase 3: Full balance
        # Refine: "perfect eigenmodes with orthonormality"
        else:
            return {
                'w_res': self.w_res_base * 5.0,  # Keep residual strong!
                'w_orth': self.w_orth_base * 1.5,  # Stronger in final phase
                'w_proj': self.w_proj_base,
                'w_order': self.w_order_base * 5.0,  # Keep modes separated
                'w_eigen': self.w_eigen_base * 10.0  # Keep matching coarse eigenvalues
            }
    
    def get_phase(self, epoch):
        """Return current phase name."""
        if epoch < self.phase1_end:
            return "Phase 1: PDE Learning"
        elif epoch < self.phase2_end:
            return "Phase 2: Orthogonalization"
        else:
            return "Phase 3: Refinement"


class AdaptiveLRScheduler:
    """
    Learning rate scheduler with warmup and plateau detection.
    NO eigenvalue computation - purely loss-based.
    """
    
    def __init__(self, optimizer, base_lr=0.001, warmup_epochs=500, 
                 patience=1000, factor=0.7, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        self.current_lr = base_lr
    
    def step(self, epoch, loss):
        """Update learning rate based on epoch and loss."""
        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Check for improvement (0.05% threshold)
            if loss < self.best_loss * 0.9995:
                self.best_loss = loss
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
            
            # Reduce LR on plateau
            if self.epochs_since_improvement >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.epochs_since_improvement = 0
                print(f"\n>>> LR plateau detected! Reducing to {self.current_lr:.2e}")
            
            lr = self.current_lr
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def eigenvalue_separation_loss(lambdas, min_gap=0.05):
    """
    Encourage separation between consecutive eigenvalues.
    Prevents mode collapse without computing eigenvalues externally.
    """
    if len(lambdas) < 2:
        return torch.tensor(0.0, device=lambdas.device)
    
    gaps = lambdas[1:] - lambdas[:-1]
    
    # Penalize small gaps (soft constraint)
    penalty = torch.sum(torch.relu(min_gap - gaps) ** 2)
    
    return penalty


def eigenvalue_smoothness_loss(lambdas):
    """
    Encourage smooth growth in eigenvalue spectrum.
    Penalizes sudden jumps or irregularities.
    """
    if len(lambdas) < 3:
        return torch.tensor(0.0, device=lambdas.device)
    
    # Second derivative of eigenvalue sequence
    first_diff = lambdas[1:] - lambdas[:-1]
    second_diff = first_diff[1:] - first_diff[:-1]
    
    # Penalize large second derivatives
    smoothness = torch.mean(second_diff ** 2)
    
    return smoothness


class ImprovedMultigridGNN:
    """Enhanced training with curriculum learning - NO eigenvalue shortcuts."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def train_multiresolution(self, X_list, U_init_list, edge_index_list, lambda_coarse,
                            epochs, lr, corr_scale, w_res, w_orth, w_proj, w_trace, 
                            w_order, w_eigen, grad_clip, weight_decay, log_every,
                            hidden_layers, dropout):
        
        device = self.device
        n_modes = U_init_list[0].shape[1]
        
        # Build tensors
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
        
        # Initialize model
        from corrector_model import SimpleCorrector
        in_dim = x_feats_all.shape[1]
        if self.model is None:
            self.model = SimpleCorrector(in_dim, n_modes, hidden_layers, dropout).to(device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize schedulers
        curriculum = CurriculumScheduler(epochs, w_res, w_orth, w_proj, w_order, w_eigen)
        lr_scheduler = AdaptiveLRScheduler(optimizer, base_lr=lr, warmup_epochs=500,
                                          patience=1000, factor=0.7)
        
        # Precompute Laplacians
        import robust_laplacian
        import utils
        L_list, M_list = [], []
        for X in X_list:
            L, M = robust_laplacian.point_cloud_laplacian(X)
            L_list.append(utils.scipy_sparse_to_torch_sparse(L).to(device))
            M_list.append(utils.scipy_sparse_to_torch_sparse(M).to(device))
        
        if lambda_coarse is not None:
            lambda_target = torch.FloatTensor(lambda_coarse).to(device)
        
        self.model.train()
        
        # Training loop
        for ep in range(epochs):
            # Get current curriculum weights
            weights = curriculum.get_weights(ep)
            phase = curriculum.get_phase(ep)
            
            optimizer.zero_grad()
            corr_raw = self.model(x_feats_all, edge_index_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr
            
            # Compute losses (NO external eigenvalue computation)
            loss_dict = self._compute_physics_losses(
                U_pred, L_list, M_list, U_init_list, 
                lambda_target if lambda_coarse is not None else None,
                n_modes
            )
            
            # Add eigenvalue quality losses
            loss_dict['separation'] = loss_dict.get('separation', torch.tensor(0.0, device=device))
            loss_dict['smoothness'] = loss_dict.get('smoothness', torch.tensor(0.0, device=device))
            
            # Weighted total loss
            total_loss = (
                weights['w_res'] * loss_dict['residual'] +
                weights['w_orth'] * loss_dict['orthogonal'] +
                weights['w_proj'] * loss_dict['projection'] +
                weights['w_order'] * (loss_dict['order'] + 0.1 * loss_dict['separation']) +
                weights['w_eigen'] * loss_dict['eigen'] +
                0.01 * loss_dict['smoothness']  # Small weight for smoothness
            )
            
            total_loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Update learning rate
            current_lr = lr_scheduler.step(ep, total_loss.item())
            
            # Logging
            if ep % log_every == 0 or ep == epochs - 1:
                print(f"Epoch {ep:5d} [{phase}] LR={current_lr:.2e}: "
                      f"Loss={total_loss.item():.4f} | "
                      f"Res={loss_dict['residual'].item():.4f} | "
                      f"Orth={loss_dict['orthogonal'].item():.4f} | "
                      f"Proj={loss_dict['projection'].item():.4f} | "
                      f"Order={loss_dict['order'].item():.4f} | "
                      f"Eigen={loss_dict['eigen'].item():.4f}")
        
        # Final normalization
        with torch.no_grad():
            corr_raw = self.model(x_feats_all, edge_index_all)
            corr = corr_scale * corr_raw
            U_pred = U_all_tensor + corr
            
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
    
    def _compute_physics_losses(self, U_pred, L_list, M_list, U_init_list, 
                               lambda_target, n_modes):
        """Compute all physics-informed losses - pure neural learning."""
        device = U_pred.device
        
        loss_residual = 0.0
        loss_orthogonal = 0.0
        loss_projection = 0.0
        loss_order = 0.0
        loss_eigen = 0.0
        loss_separation = 0.0
        loss_smoothness = 0.0
        
        node_offset = 0
        
        for i, (L_t, M_t, U_init) in enumerate(zip(L_list, M_list, U_init_list)):
            n_nodes = U_init.shape[0]
            U_level = U_pred[node_offset:node_offset+n_nodes]
            
            # Normalize each eigenvector
            Mu = torch.sparse.mm(M_t, U_level)
            norms = torch.sqrt(torch.sum(U_level * Mu, dim=0) + 1e-12)
            U_level_norm = U_level / norms.unsqueeze(0)
            
            Mu_norm = torch.sparse.mm(M_t, U_level_norm)
            Lu_norm = torch.sparse.mm(L_t, U_level_norm)
            
            # Rayleigh quotient (learned eigenvalues)
            lambdas = torch.sum(U_level_norm * Lu_norm, dim=0) / \
                     (torch.sum(U_level_norm * Mu_norm, dim=0) + 1e-12)
            
            # 1. Residual loss: || Lu - λMu ||²
            res = Lu_norm - Mu_norm * lambdas.unsqueeze(0)
            loss_residual += torch.mean(res**2)
            
            # 2. Orthonormality: ||U^T M U - I||²
            Gram = U_level_norm.t() @ Mu_norm
            loss_orthogonal += torch.sum((Gram - torch.eye(n_modes, device=device))**2) / n_modes
            
            # 3. Zero-mean for modes 1+ (mode 0 is constant)
            if n_modes > 1:
                ones = torch.ones(n_nodes, 1, device=device)
                mean_constraint = ones.t() @ Mu_norm[:, 1:]
                loss_projection += torch.mean(mean_constraint**2)
            
            # 4. Eigenvalue ordering: λ_i < λ_{i+1}
            lambda_diffs = lambdas[1:] - lambdas[:-1]
            loss_order += torch.sum(torch.relu(-lambda_diffs))
            
            # 5. Match coarse eigenvalues (only at coarsest level)
            if i == 0 and lambda_target is not None:
                # Use BOTH absolute and relative error
                # Absolute error: prevents scaling drift
                abs_error = torch.mean((lambdas - lambda_target)**2)
                # Relative error: ensures correct ratios
                relative_error = (lambdas - lambda_target) / (lambda_target + 1e-6)
                rel_error = torch.mean(relative_error**2)
                
                loss_eigen += abs_error + rel_error
            
            # 6. Eigenvalue separation (prevent mode collapse)
            loss_separation += eigenvalue_separation_loss(lambdas, min_gap=0.05)
            
            # 7. Eigenvalue smoothness (natural spectrum growth)
            loss_smoothness += eigenvalue_smoothness_loss(lambdas)
            
            node_offset += n_nodes
        
        return {
            'residual': loss_residual,
            'orthogonal': loss_orthogonal,
            'projection': loss_projection,
            'order': loss_order,
            'eigen': loss_eigen,
            'separation': loss_separation,
            'smoothness': loss_smoothness
        }