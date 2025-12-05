import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd, norm
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment
import robust_laplacian
import matplotlib.pyplot as plt
from mesh_helpers import compute_stiffness_and_mass_matrices


# =============================================================================
# MATRIX CONVERSION AND NORMALIZATION
# =============================================================================

def scipy_sparse_to_torch_sparse(A):
    """Convert scipy sparse matrix to torch sparse tensor."""
    A = A.tocoo()
    indices = np.vstack((A.row, A.col)).astype(np.int64)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(i, v, A.shape).coalesce()


def normalize_columns_np(U, eps=1e-12):
    """Normalize columns of numpy array."""
    norms = np.linalg.norm(U, axis=0) + eps
    return U / norms, norms


def normalize_columns_torch(U, eps=1e-12):
    """Normalize columns of torch tensor."""
    norms = torch.norm(U, dim=0) + eps
    return U / norms, norms


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_prolongation(X_coarse, X_fine, k):
    """
    Build interpolation matrix from coarse to fine points using k-NN weighting.
    
    Returns sparse matrix P where P[i,j] represents the weight of coarse point j
    in the interpolation of fine point i.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_coarse)
    distances, indices = nbrs.kneighbors(X_fine)
    
    n_fine, n_coarse = X_fine.shape[0], X_coarse.shape[0]
    rows, cols, vals = [], [], []
    
    for i in range(n_fine):
        weights = 1.0 / (distances[i] + 1e-12)
        weights /= weights.sum()
        for j, idx in enumerate(indices[i]):
            rows.append(i)
            cols.append(idx)
            vals.append(weights[j])
    
    return coo_matrix((vals, (rows, cols)), shape=(n_fine, n_coarse))


def build_knn_graph(X, k):
    """Build k-NN graph edge indices from point cloud."""
    n_points = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, neighbors = nbrs.kneighbors(X)
    
    rows, cols = [], []
    for i in range(n_points):
        for j in neighbors[i][1:]:  # Skip self-connection
            rows.append(i)
            cols.append(j)
    
    return torch.LongTensor([rows, cols])


def build_A_norm(edge_index, n_nodes, device):
    """
    Build normalized adjacency matrix for GCN: A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    
    Args:
        edge_index: (2, E) edge indices
        n_nodes: Total number of nodes
        device: torch device
        
    Returns:
        Normalized adjacency matrix as sparse torch tensor
    """
    edge_index = edge_index.to(device)
    
    # Create adjacency matrix A
    adj = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.shape[1], device=device),
        (n_nodes, n_nodes)
    ).coalesce()
    
    # Add self-loops: A_hat = A + I
    diag_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
    I = torch.sparse_coo_tensor(
        diag_indices,
        torch.ones(n_nodes, device=device),
        (n_nodes, n_nodes)
    ).coalesce()
    
    A_hat = (adj + I).coalesce()
    
    # Compute degree and D^(-1/2)
    row = A_hat.indices()[0]
    deg = torch.bincount(row, minlength=n_nodes).float()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    
    # Build D^(-1/2) as sparse diagonal
    D_inv_sqrt = torch.sparse_coo_tensor(
        diag_indices,
        deg_inv_sqrt,
        (n_nodes, n_nodes)
    ).coalesce()
    
    # A_norm = D^(-1/2) * A_hat * D^(-1/2)
    A_norm = torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(A_hat, D_inv_sqrt))
    
    return A_norm


def sparse_block_diag(sparse_tensor_list):
    """Combine list of sparse tensors into block-diagonal sparse tensor."""
    if not sparse_tensor_list:
        return None
    
    # Calculate dimensions
    n_rows_list = [t.shape[0] for t in sparse_tensor_list]
    n_cols_list = [t.shape[1] for t in sparse_tensor_list]
    total_rows = sum(n_rows_list)
    total_cols = sum(n_cols_list)
    
    # Collect and offset indices/values
    all_indices = []
    all_values = []
    row_offset = 0
    col_offset = 0
    
    for t in sparse_tensor_list:
        indices = t.indices().clone()
        indices[0, :] += row_offset
        indices[1, :] += col_offset
        
        all_indices.append(indices)
        all_values.append(t.values())
        
        row_offset += t.shape[0]
        col_offset += t.shape[1]
    
    # Concatenate and create block-diagonal tensor
    final_indices = torch.cat(all_indices, dim=1)
    final_values = torch.cat(all_values, dim=0)
    device = sparse_tensor_list[0].device
    
    return torch.sparse_coo_tensor(
        final_indices,
        final_values,
        (total_rows, total_cols),
        device=device
    ).coalesce()


# =============================================================================
# EIGENVALUE SOLVERS
# =============================================================================

def solve_eigenvalue_point_cloud(X, n_modes):
    """Solve eigenvalue problem for point cloud using robust Laplacian."""
    L, M = robust_laplacian.point_cloud_laplacian(X)
    vals, vecs = eigsh(L, k=n_modes, M=M, which='SM')
    return vals, np.array(vecs), L, M


def solve_eigenvalue_mesh(mesh, n_modes):
    """Solve eigenvalue problem for mesh using stiffness and mass matrices."""
    K, M = compute_stiffness_and_mass_matrices(mesh)
    vals, vecs = eigsh(K, k=n_modes, M=M, which='SM')
    return vals, np.array(vecs), K, M


# =============================================================================
# ORTHONORMALIZATION
# =============================================================================

def orthonormalize(U, M):
    """
    Gram-Schmidt orthonormalization with respect to mass matrix M.
    
    Args:
        U: (N, k) eigenvector matrix
        M: (N, N) mass matrix
        
    Returns:
        U_orth: Orthonormalized eigenvector matrix
    """
    n_modes = U.shape[1]
    U_orth = np.zeros_like(U)
    
    for i in range(n_modes):
        v = U[:, i].copy()
        
        # Orthogonalize against previous modes
        for j in range(i):
            u_j = U_orth[:, j]
            overlap = u_j.T @ M @ v
            v -= overlap * u_j
        
        # Normalize with respect to M
        norm_v = np.sqrt(v.T @ M @ v)
        U_orth[:, i] = v / (norm_v + 1e-12)
    
    return U_orth


def jacobi_smooth(M, L, U_rough, alpha=0.05, n_iters=5):
    """
    Iterative Jacobi smoothing as cheap alternative to direct solve.
    Solves (M + alpha*L) @ U = M @ U_rough approximately.
    """
    U = U_rough.copy()
    D_inv = 1.0 / (M.diagonal() + alpha * L.diagonal() + 1e-12)
    
    for _ in range(n_iters):
        residual = M @ U_rough - (M + alpha * L) @ U
        U += D_inv[:, None] * residual
    
    return U


# =============================================================================
# EIGENVECTOR ALIGNMENT
# =============================================================================

def align_eigenvectors(U_pred, U_exact, M=None, verbose=False):
    """
    Align predicted eigenvectors to exact eigenvectors using Hungarian algorithm.
    Finds optimal permutation and sign flips for mode-by-mode comparison.
    
    Args:
        U_pred: (n, k) predicted eigenvectors
        U_exact: (n, k) exact eigenvectors
        M: Mass matrix for M-orthogonality (optional)
        verbose: Print alignment details
        
    Returns:
        U_aligned: Aligned and sign-corrected eigenvectors
        permutation: Index mapping from predicted to exact modes
        signs: Sign flips applied to each mode
    """
    n_modes = U_pred.shape[1]
    
    # Compute overlap matrix
    overlap = U_pred.T @ M @ U_exact if M is not None else U_pred.T @ U_exact
    
    if verbose:
        print("\n=== Alignment Debug ===")
        print(f"Overlap shape: {overlap.shape}")
        print(f"Overlap (first 5x5):\n{overlap[:5, :5]}")
        print(f"Max abs overlap per row: {np.max(np.abs(overlap), axis=1)[:10]}")
    
    # Hungarian algorithm on absolute overlap values
    overlap_abs = np.abs(overlap)
    row_ind, col_ind = linear_sum_assignment(-overlap_abs)
    
    if verbose:
        print(f"\nHungarian matching (first 10):")
        for i in range(min(10, len(row_ind))):
            print(f"  Exact {col_ind[i]} <- Predicted {row_ind[i]} "
                  f"(overlap: {overlap_abs[row_ind[i], col_ind[i]]:.4f})")
    
    # Build permutation array
    permutation = np.zeros(n_modes, dtype=int)
    for i in range(n_modes):
        permutation[col_ind[i]] = row_ind[i]
    
    # Apply permutation
    U_aligned = U_pred[:, permutation].copy()
    
    # Determine and apply sign flips
    signs = np.zeros(n_modes)
    for i in range(n_modes):
        pred_idx = permutation[i]
        signs[i] = np.sign(overlap[pred_idx, i])
        signs[i] = 1 if signs[i] == 0 else signs[i]
    
    U_aligned *= signs
    
    if verbose:
        print(f"\nSign flips: {signs[:10]}")
        print("=" * 50)
    
    return U_aligned, permutation, signs


def get_subspace_error_and_alignment(U_pred, U_exact, M=None):
    """
    Compute optimal subspace alignment using Orthogonal Procrustes Analysis.
    Finds rotation R that minimizes ||U_pred @ R - U_exact||_F.
    
    Args:
        U_pred: (n, k) predicted eigenvectors
        U_exact: (n, k) exact eigenvectors
        M: Mass matrix for M-orthogonality (optional)
        
    Returns:
        U_aligned: U_pred @ R_optimal
        subspace_error: Frobenius norm of alignment error
    """
    # Compute overlap matrix
    W = U_pred.T @ M @ U_exact if M is not None else U_pred.T @ U_exact
    
    # SVD: W = V * Sigma * D^T
    V, _, D_T = svd(W)
    
    # Optimal rotation
    R_optimal = V @ D_T
    
    # Apply rotation and compute error
    U_aligned = U_pred @ R_optimal
    subspace_error = norm(U_aligned - U_exact, 'fro')
    
    return U_aligned, subspace_error


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def compute_rayleigh_quotients(U, L, M):
    """Compute eigenvalues via Rayleigh quotient: λ_i = u_i^T L u_i / u_i^T M u_i"""
    lambdas = []
    for i in range(U.shape[1]):
        u = U[:, i]
        lambda_i = (u.T @ L @ u) / (u.T @ M @ u + 1e-12)
        lambdas.append(lambda_i)
    return np.array(lambdas)


def comprehensive_diagnostics(U_pred, U_exact, X, config, L, M):
    """
    Perform comprehensive eigenmode analysis including:
    - Eigenvalue comparison
    - Eigenvector alignment quality
    - Subspace error (Procrustes)
    - Orthonormality check
    - Visualization
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EIGENMODE DIAGNOSTICS")
    print("="*80)
    
    # Convert sparse matrices to dense
    L_dense = L.toarray() if hasattr(L, 'toarray') else L
    M_dense = M.toarray() if hasattr(M, 'toarray') else M
    
    # Compute eigenvalues
    lambda_pred = compute_rayleigh_quotients(U_pred, L_dense, M_dense)
    lambda_exact = compute_rayleigh_quotients(U_exact, L_dense, M_dense)
    
    print("\n--- BEFORE ALIGNMENT ---")
    print(f"Predicted eigenvalues (first 10): {np.round(lambda_pred[:10], 4)}")
    print(f"Exact eigenvalues (first 10): {np.round(lambda_exact[:10], 4)}")
    
    # Align for mode-by-mode tracking
    U_aligned_ps, permutation, signs = align_eigenvectors(
        U_pred, U_exact, M_dense, verbose=True
    )
    lambda_pred_ordered = lambda_pred[permutation]
    
    # Align for subspace error
    _, subspace_error = get_subspace_error_and_alignment(U_pred, U_exact, M_dense)
    
    # -------------------------------------------------------------------------
    # 1. EIGENVALUE COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("1. EIGENVALUE COMPARISON")
    print("-" * 80)
    print(f"{'Mode':<6} {'λ_exact':<12} {'λ_pred':<12} {'Abs Err':<12} {'Rel Err':<12}")
    print("-" * 80)
    
    max_display = min(20, len(lambda_exact))
    for i in range(max_display):
        abs_err = abs(lambda_pred_ordered[i] - lambda_exact[i])
        rel_err = abs_err / (abs(lambda_exact[i]) + 1e-12)
        print(f"{i:<6} {lambda_exact[i]:<12.6f} {lambda_pred_ordered[i]:<12.6f} "
              f"{abs_err:<12.6f} {rel_err:<12.6f}")
    
    abs_errors = np.abs(lambda_pred_ordered - lambda_exact)
    rel_errors = abs_errors / (np.abs(lambda_exact) + 1e-12)
    print(f"\nSummary: Mean Abs Error = {np.mean(abs_errors):.6f}, "
          f"Mean Rel Error = {np.mean(rel_errors):.6f}")
    
    # -------------------------------------------------------------------------
    # 2. EIGENVECTOR ALIGNMENT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("2. EIGENVECTOR ALIGNMENT")
    print("-" * 80)
    print(f"{'Mode':<6} {'Perm':<10} {'Sign':<6} {'Cos Sim':<12} {'L2 Err':<12}")
    print("-" * 80)
    
    cos_sims = []
    l2_errors = []
    
    for i in range(max_display):
        u_aligned = U_aligned_ps[:, i]
        u_exact = U_exact[:, i]
        
        # M-normalized cosine similarity
        inner = u_aligned.T @ M_dense @ u_exact
        norm_aligned = np.sqrt(u_aligned.T @ M_dense @ u_aligned)
        norm_exact = np.sqrt(u_exact.T @ M_dense @ u_exact)
        cos_sim = abs(inner) / (norm_aligned * norm_exact + 1e-12)
        
        # L2 error
        l2_err = np.linalg.norm(u_aligned - u_exact)
        
        cos_sims.append(cos_sim)
        l2_errors.append(l2_err)
        
        print(f"{i:<6} {i}->{permutation[i]:<7} {signs[i]:>4.0f}   "
              f"{cos_sim:<12.6f} {l2_err:<12.6f}")
    
    # Compute all similarities
    for i in range(max_display, len(lambda_exact)):
        u_aligned = U_aligned_ps[:, i]
        u_exact = U_exact[:, i]
        inner = u_aligned.T @ M_dense @ u_exact
        norm_aligned = np.sqrt(u_aligned.T @ M_dense @ u_aligned)
        norm_exact = np.sqrt(u_exact.T @ M_dense @ u_exact)
        cos_sim = abs(inner) / (norm_aligned * norm_exact + 1e-12)
        l2_err = np.linalg.norm(u_aligned - u_exact)
        cos_sims.append(cos_sim)
        l2_errors.append(l2_err)
    
    print(f"\nSummary: Mean Cos Sim = {np.mean(cos_sims):.6f}, "
          f"Mean L2 Error = {np.mean(l2_errors):.6f}")
    
    # -------------------------------------------------------------------------
    # 3. SUBSPACE ALIGNMENT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("3. SUBSPACE ALIGNMENT (Procrustes)")
    print("-" * 80)
    print(f"Subspace Error (Frobenius): {subspace_error:.6e}")
    
    # -------------------------------------------------------------------------
    # 4. ORTHONORMALITY CHECK
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("4. ORTHONORMALITY CHECK")
    print("-" * 80)
    UMU = U_pred.T @ M_dense @ U_pred
    diag = np.diag(UMU)
    off_diag_max = np.max(np.abs(UMU - np.diag(diag)))
    print(f"Diagonal (first 10, should be ~1.0): {diag[:10]}")
    print(f"Max off-diagonal: {off_diag_max:.6e}")
    
    # -------------------------------------------------------------------------
    # 5. VISUALIZATIONS
    # -------------------------------------------------------------------------
    if config.diagnostics_viz:
        _create_diagnostic_plots(lambda_exact, lambda_pred_ordered, rel_errors, cos_sims, UMU, config)
    
    print("\n" + "="*80)


def _create_diagnostic_plots(lambda_exact, lambda_pred, rel_errors, cos_sims, UMU, config):
    """Create comprehensive diagnostic visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Eigenvalue comparison
    axes[0, 0].plot(lambda_exact[1:30], 'o-', label='Exact', 
                   markersize=8, linewidth=2)
    axes[0, 0].plot(lambda_pred[1:30], 's-', label='Predicted', 
                   markersize=6, alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Mode Index', fontsize=12)
    axes[0, 0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0, 0].set_title('Eigenvalue Comparison', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Relative errors
    axes[0, 1].semilogy(rel_errors[1:30], 'o-', linewidth=2)
    axes[0, 1].set_xlabel('Mode Index', fontsize=12)
    axes[0, 1].set_ylabel('Relative Error', fontsize=12)
    axes[0, 1].set_title('Eigenvalue Relative Errors', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cosine similarities
    axes[1, 0].plot(cos_sims[1:30], 'o-', linewidth=2)
    # axes[1, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (0.95)')
    axes[1, 0].set_xlabel('Mode Index', fontsize=12)
    axes[1, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1, 0].set_title('Eigenvector Alignment Quality', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Gram matrix
    im1 = axes[1, 1].imshow(np.abs(UMU), cmap='magma', vmin=0, vmax=1.1)
    axes[1, 1].set_title('|U^T M U| - Orthonormality', fontsize=14)
    plt.colorbar(im1, ax=axes[1, 1])
    
    plt.tight_layout()
    save_path = config.diagnostics_viz.replace('.png', '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plots saved to: {save_path}")


def post_training_diagnostics(UMU, n_modes, output_file):
    """Quick orthonormality check after training."""
    print("\nOrthonormality check (should be Identity):")
    print("Diagonal:", np.diag(UMU)[:10])
    print("Off-diagonal max:", np.max(np.abs(UMU - np.eye(n_modes))))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(UMU, cmap='magma', vmin=-0.1, vmax=1.1)
    axes[0].set_title('U^T M U (should be Identity)')
    plt.colorbar(axes[0].images[0], ax=axes[0])
    
    axes[1].imshow(np.abs(UMU - np.eye(n_modes)), cmap='magma')
    axes[1].set_title('|U^T M U - I|')
    plt.colorbar(axes[1].images[0], ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')