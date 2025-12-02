import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import torch
from scipy.sparse.linalg import eigsh
import robust_laplacian
import matplotlib.pyplot as plt
from mesh_helpers import compute_stiffness_and_mass_matrices
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd, norm


def scipy_sparse_to_torch_sparse(A):
    A = A.tocoo()
    indices = np.vstack((A.row, A.col)).astype(np.int64)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(i, v, A.shape).coalesce()


def normalize_columns_np(U, eps):
    norms = np.linalg.norm(U, axis=0) + eps
    return U / norms, norms


def normalize_columns_torch(U, eps=1e-12):
    norms = torch.norm(U, dim=0) + eps
    return U / norms, norms


def build_prolongation(X_coarse, X_fine, k):
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
    n_points = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, neighbors = nbrs.kneighbors(X)
    rows, cols = [], []
    for i in range(n_points):
        for j in neighbors[i][1:]:
            rows.append(i)
            cols.append(j)
    return torch.LongTensor([rows, cols]).to(torch.long)


def solve_eigenvalue_point_cloud(X, n_modes):
    L, M = robust_laplacian.point_cloud_laplacian(X)
    vals, vecs = eigsh(L, k=n_modes, M=M, which='SM')
    return vals, np.array(vecs), L, M


def solve_eigenvalue_mesh(mesh, n_modes):
    K, M = compute_stiffness_and_mass_matrices(mesh)
    vals, vecs = eigsh(K, k=n_modes, M=M, which='SM')
    return vals, np.array(vecs), K, M


def sparse_block_diag(sparse_tensor_list: list) -> torch.Tensor:
    """
    Combines a list of square sparse tensors into a single block-diagonal sparse tensor.
    Assumes all tensors are torch.sparse_coo_tensor.
    """
    if not sparse_tensor_list:
        return None

    # Calculate total dimensions
    n_rows_list = [t.shape[0] for t in sparse_tensor_list]
    n_cols_list = [t.shape[1] for t in sparse_tensor_list]
    total_rows = sum(n_rows_list)
    total_cols = sum(n_cols_list)

    # Collect and offset indices/values
    all_indices = []
    all_values = []
    
    current_row_offset = 0
    current_col_offset = 0

    for t in sparse_tensor_list:
        indices = t.indices()
        values = t.values()
        
        # Add offset to indices
        offset_indices = indices.clone()
        offset_indices[0, :] += current_row_offset
        offset_indices[1, :] += current_col_offset
        
        all_indices.append(offset_indices)
        all_values.append(values)

        current_row_offset += t.shape[0]
        current_col_offset += t.shape[1]

    # Concatenate all indices and values
    final_indices = torch.cat(all_indices, dim=1)
    final_values = torch.cat(all_values, dim=0)

    # Create the final block-diagonal sparse tensor
    # The device should be inferred from the components, but we enforce it
    device = sparse_tensor_list[0].device
    
    return torch.sparse_coo_tensor(
        final_indices, 
        final_values, 
        (total_rows, total_cols), 
        device=device
    ).coalesce()


def post_training_diagnostics(UMU, n_modes, output_file):
    print("\nOrthonormality check (should be Identity):")
    print("Diagonal:", np.diag(UMU)[:10])
    print("Off-diagonal max:", np.max(np.abs(UMU - np.eye(n_modes))))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(UMU, cmap='magma', vmin=-0.1, vmax=1.1)
    plt.colorbar()
    plt.title('U^T M U (should be Identity)')
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(UMU - np.eye(n_modes)), cmap='magma')
    plt.colorbar()
    plt.title('|U^T M U - I|')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


# ----------------------------------------------------------------------
# 1. Permutation + Sign Alignment (Used for mode-by-mode error tracking)
# ----------------------------------------------------------------------

def align_eigenvectors(U_pred, U_exact, M=None):
    """
    Align predicted eigenvectors to exact eigenvectors using Hungarian algorithm
    for permutation and sign flip.
    
    Args:
        U_pred: (n, k) predicted eigenvectors (columns are modes)
        U_exact: (n, k) exact eigenvectors (columns are modes)
        M: optional mass matrix for M-orthogonality
        
    Returns:
        U_aligned_ps: (n, k) permutation/sign aligned eigenvectors
        permutation: best permutation found (index in U_pred corresponding to U_exact order)
        signs: best signs found
    """
    n_modes = U_pred.shape[1]
    
    # Compute overlap matrix W = U_pred^T M U_exact
    if M is not None:
        overlap_pos = U_pred.T @ M @ U_exact
    else:
        overlap_pos = U_pred.T @ U_exact
    
    overlap_abs = np.abs(overlap_pos)
    
    # Hungarian algorithm: Maximize overlap <==> Minimize negative overlap
    # row_ind gives the column index in U_pred (source), col_ind in U_exact (target)
    row_ind, col_ind = linear_sum_assignment(-overlap_abs)
    
    # Apply permutation to predicted modes
    # U_aligned_ps[:, i] = U_pred[:, row_ind[i]]
    U_aligned_ps = U_pred[:, row_ind].copy()
    
    # Signs are determined from the overlap of the permuted pair
    signs = np.sign(overlap_pos[row_ind, col_ind])
    signs[signs == 0] = 1  # Handle near-zero overlaps
    
    # Apply sign flip
    U_aligned_ps = U_aligned_ps * signs
    
    # The permutation maps the original column index of U_exact (i) to the
    # column index of U_pred (row_ind[i]).
    return U_aligned_ps, row_ind, signs


def orthonormalize(U, M):
    """
    Gram-Schmidt orthonormalization of U with respect to mass matrix M.
    U: (N, k) numpy array
    M: (N, N) scipy sparse matrix
    """
    n_modes = U.shape[1]
    U_orth = np.zeros_like(U)

    for i in range(n_modes):
        v = U[:, i].copy()
        # Project out previous modes
        for j in range(i):
            u_j = U_orth[:, j]
            # overlap = <u_j, v>_M = u_j.T @ M @ v
            overlap = u_j.T @ M @ v
            v -= overlap * u_j

        # Normalize
        norm_v = np.sqrt(v.T @ M @ v)
        U_orth[:, i] = v / (norm_v + 1e-12)

    return U_orth


def jacobi_smooth(M, L, U_rough, alpha=0.05, n_iters=5):
    """Cheap iterative smoothing instead of direct solve"""
    U = U_rough.copy()
    D_inv = 1.0 / (M.diagonal() + alpha * L.diagonal() + 1e-12)

    for _ in range(n_iters):
        residual = M @ U_rough - (M + alpha * L) @ U
        U += D_inv[:, None] * residual

    return U


# This function implements the normalized adjacency matrix for a Graph Convolutional Network (GCN) layer.
# A_norm = D^(-1/2) * (A + I) * D^(-1/2)
def build_A_norm(edge_index, n_nodes, device):
    """
    Builds the normalized adjacency matrix (A_norm) for a GCN layer as a torch sparse tensor.

    Args:
        edge_index (torch.LongTensor): Edge indices (2 x E) of the graph.
        n_nodes (int): Total number of nodes in the graph (or combined graph blocks).
        device (torch.device): The device (e.g., 'cuda', 'cpu') to place the tensor on.

    Returns:
        torch.sparse_coo_tensor: The normalized adjacency matrix A_norm.
    """
    
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(device)

    # 1. Create the base adjacency matrix A
    # The values are 1 for all edges
    adj = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.shape[1], device=device), 
        (n_nodes, n_nodes)
    ).coalesce()

    # 2. Add self-loops (I) to form A_hat = A + I
    # Create identity matrix I (sparse)
    diag_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
    I = torch.sparse_coo_tensor(
        diag_indices, 
        torch.ones(n_nodes, device=device), 
        (n_nodes, n_nodes)
    ).coalesce()

    # A_hat = A + I
    A_hat = (adj + I).coalesce()
    
    # 3. Calculate D_hat (degree of A_hat) and D_hat^(-1/2)
    row = A_hat.indices()[0]
    
    # Count degrees for each node (row is a vector of row indices of non-zero elements in A_hat)
    deg_hat = torch.bincount(row, minlength=n_nodes).float()
    
    # Calculate D_hat^(-1/2) diagonal values
    D_hat_invsqrt_diag = torch.pow(deg_hat.clamp(min=1e-12), -0.5)
    
    # 4. Create the sparse diagonal matrix D_hat^(-1/2)
    D_hat_invsqrt_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
    D_hat_invsqrt = torch.sparse_coo_tensor(
        D_hat_invsqrt_indices, 
        D_hat_invsqrt_diag, 
        (n_nodes, n_nodes)
    ).coalesce()
    
    # 5. Calculate A_norm = D_hat^(-1/2) * A_hat * D_hat^(-1/2)
    # Using torch.sparse.mm for sparse matrix multiplication
    A_norm = torch.sparse.mm(D_hat_invsqrt, torch.sparse.mm(A_hat, D_hat_invsqrt))

    return A_norm


# ----------------------------------------------------------------------
# 2. Subspace Alignment (Orthogonal Procrustes Analysis via SVD)
# ----------------------------------------------------------------------

def get_subspace_error_and_alignment(U_pred, U_exact, M=None):
    """
    Finds the optimal orthogonal rotation R to align U_pred to U_exact using 
    Orthogonal Procrustes Analysis. This minimizes the subspace error,
    correcting for rotation, permutation, and sign-flip simultaneously.
    
    Args:
        U_pred: (n, k) predicted eigenvectors
        U_exact: (n, k) exact eigenvectors
        M: optional mass matrix for M-orthogonality
        
    Returns:
        U_aligned_procrustes: (n, k) U_pred @ R_optimal
        subspace_error: Frobenius norm of the difference || U_aligned_procrustes - U_exact ||_F
    """
    # 1. Compute the overlap matrix W = U_pred^T M U_exact
    if M is not None:
        W = U_pred.T @ M @ U_exact
    else:
        W = U_pred.T @ U_exact

    # 2. Perform SVD on the overlap matrix W = V * Sigma * D^T
    V, _, D_T = svd(W)
    
    # 3. Compute the optimal orthogonal rotation matrix R = V @ D^T
    R_optimal = V @ D_T

    # 4. Apply the optimal rotation
    U_aligned_procrustes = U_pred @ R_optimal
    
    # 5. Calculate the subspace error
    subspace_error = norm(U_aligned_procrustes - U_exact, 'fro')
    
    return U_aligned_procrustes, subspace_error


# ----------------------------------------------------------------------
# 3. Comprehensive Diagnostics (The main function with improvements)
# ----------------------------------------------------------------------

def comprehensive_diagnostics(U_pred, U_exact, X, config, L=None, M=None):
    """
    Comprehensive diagnostics for comparing predicted and exact eigenvectors,
    including both permutation/sign alignment (for mode-by-mode tracking)
    and Orthogonal Procrustes alignment (for best-case subspace error).
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EIGENMODE DIAGNOSTICS (SVD-ROBUST)")
    print("="*80)
    
    # --- 0. Setup: Compute Laplacian and Mass matrix ---
    if L is None or M is None:
        L, M = robust_laplacian.point_cloud_laplacian(X)


    # --- 1. Compute Eigenvalues via Rayleigh Quotient ---
    def compute_rayleigh_quotient(U, L, M):
        lambda_list = []
        for i in range(U.shape[1]):
            u = U[:, i]
            # Rayleigh Quotient: lambda = (u^T L u) / (u^T M u)
            Lu = L @ u
            Mu = M @ u
            lambda_i = (u.T @ Lu) / (u.T @ Mu + 1e-12) 
            lambda_list.append(lambda_i)
        return np.array(lambda_list)

    lambda_pred = compute_rayleigh_quotient(U_pred, L, M)
    lambda_exact = compute_rayleigh_quotient(U_exact, L, M)
    
    # --- 2. Alignment for Mode-by-Mode Tracking (Permutation + Sign) ---
    U_aligned_ps, permutation, signs = align_eigenvectors(U_pred, U_exact, M)
    lambda_pred_ordered = lambda_pred[permutation]
    
    # --- 3. Alignment for Subspace Error (Procrustes Analysis) ---
    U_aligned_procrustes, subspace_error = get_subspace_error_and_alignment(U_pred, U_exact, M)
    
    
    # =================================================================
    #   DIAGNOSTIC OUTPUT
    # =================================================================
    
    # --- 1. EIGENVALUE COMPARISON (using Permutation-aligned modes) ---
    print("\n1. EIGENVALUE COMPARISON (Permutation-Aligned)")
    print("-" * 80)
    print(f"{'Mode':<6} {'λ_exact':<12} {'λ_pred (aligned)':<18} {'Rel Error':<12}")
    print("-" * 80)
    
    for i in range(min(64, len(lambda_exact))):
        lambda_exact_i = lambda_exact[i]
        lambda_aligned_i = lambda_pred_ordered[i] 
        rel_error = abs(lambda_aligned_i - lambda_exact_i) / (abs(lambda_exact_i) + 1e-12)
        
        print(f"{i:<6} {lambda_exact_i:<12.6f} "
              f"{lambda_aligned_i:<18.6f} {rel_error:<12.6f}")

    # --- 2. EIGENVECTOR ALIGNMENT (Mode-by-Mode using Permutation/Sign) ---
    print("\n2. EIGENVECTOR ALIGNMENT (Mode-by-Mode Permutation/Sign)")
    print("-" * 80)
    print(f"{'Mode':<6} {'Permutation':<12} {'Sign':<6} {'Cosine Sim':<12} {'L2 Error (PS)':<14}")
    print("-" * 80)
    
    for i in range(min(64, U_exact.shape[1])):
        perm_i = permutation[i] # U_pred[:, perm_i] aligns to U_exact[:, i]
        sign_i = signs[i]
        
        u_aligned = U_aligned_ps[:, i]
        u_exact = U_exact[:, i]
        
        # M-normalized Cosine similarity
        inner_prod = u_aligned.T @ M @ u_exact
        norm_u_aligned_sq = u_aligned.T @ M @ u_aligned
        norm_u_exact_sq = u_exact.T @ M @ u_exact
        
        cos_sim = abs(inner_prod) / (np.sqrt(norm_u_aligned_sq * norm_u_exact_sq) + 1e-12)
        
        # L2 error (standard Euclidean norm) for Permutation/Sign alignment
        l2_error_ps = norm(u_aligned - u_exact)
        
        print(f"{i:<6} {i}->{perm_i:<9} {sign_i:>4.0f}    {cos_sim:<12.6f} {l2_error_ps:<14.6f}")

    # --- 3. SUBSPACE ALIGNMENT (Procrustes Analysis) ---
    print("\n3. SUBSPACE ALIGNMENT (Procrustes Analysis - Most Robust Check)")
    print("-" * 80)
    print(f"Subspace Error (Frobenius Norm, ||U_aligned_proc - U_exact||_F): {subspace_error:.6e}")
    print("Lower value indicates better alignment of the entire k-dimensional eigenspace.")
    print("-" * 80)

    # --- 4. ORTHONORMALITY CHECKS ---
    print("\n4. ORTHONORMALITY CHECK (Predicted)")
    print("-" * 80)
    UMU_pred = U_pred.T @ M @ U_pred
    diag_pred = np.diag(UMU_pred)
    off_diag_max_pred = np.max(np.abs(UMU_pred - np.diag(diag_pred)))
    print(f"Diagonal (should be ~1.0): {diag_pred[:64]}")
    print(f"Max off-diagonal: {off_diag_max_pred:.6e}")
    
    # --- 5. VISUALIZATIONS ---
    if config.diagnostics_viz:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Eigenvalue comparison
            axes[0, 0].plot(lambda_exact[:20], 'o-', label='Exact', markersize=8)
            axes[0, 0].plot(lambda_pred[:20], 's-', label='Predicted (Original Order)', markersize=6, alpha=0.7)
            axes[0, 0].plot(lambda_pred_ordered[:20], 'x-', label='Predicted (Permutation Aligned)', markersize=6, alpha=0.9)
            axes[0, 0].set_xlabel('Mode Index')
            axes[0, 0].set_ylabel('Eigenvalue')
            axes[0, 0].set_title('Eigenvalue Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Eigenvalue errors
            rel_errors = np.abs(lambda_pred_ordered - lambda_exact) / (np.abs(lambda_exact) + 1e-12)
            axes[0, 1].semilogy(rel_errors[:20], 'o-')
            axes[0, 1].set_xlabel('Mode Index')
            axes[0, 1].set_ylabel('Relative Error')
            axes[0, 1].set_title('Eigenvalue Relative Errors (Permutation Aligned)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Gram matrix (predicted) - Orthonormality check
            im1 = axes[1, 0].imshow(np.abs(UMU_pred), cmap='magma', vmin=0, vmax=1.1)
            axes[1, 0].set_title('|U^T M U| - Predicted (Orthonormality)')
            plt.colorbar(im1, ax=axes[1, 0])
            
            # Plot 4: Procrustes Aligned Overlap - Subspace quality check
            # This shows the maximal overlap possible after optimal rotation
            U_overlap_proc = U_aligned_procrustes.T @ M @ U_exact
            im2 = axes[1, 1].imshow(np.abs(U_overlap_proc), cmap='magma', vmin=0, vmax=1.1)
            axes[1, 1].set_title('|U_aligned_proc^T M U_exact| (Subspace Overlap)')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            save_path = config.diagnostics_viz.replace('.png', '_comprehensive_v2.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nDiagnostic plots saved to: {save_path}")
            
        except Exception as e:
            print(f"\nVisualization Error: Could not generate plots. Ensure dependencies are installed. Error: {e}")
            
    print("\n" + "="*80)
    

# This function implements the normalized adjacency matrix for a Graph Convolutional Network (GCN) layer.
# A_norm = D^(-1/2) * (A + I) * D^(-1/2)
def build_A_norm(edge_index, n_nodes, device):
    """
    Builds the normalized adjacency matrix (A_norm) for a GCN layer as a torch sparse tensor.

    Args:
        edge_index (torch.LongTensor): Edge indices (2 x E) of the graph.
        n_nodes (int): Total number of nodes in the graph (or combined graph blocks).
        device (torch.device): The device (e.g., 'cuda', 'cpu') to place the tensor on.

    Returns:
        torch.sparse_coo_tensor: The normalized adjacency matrix A_norm.
    """
    
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(device)

    # 1. Create the base adjacency matrix A
    # The values are 1 for all edges
    adj = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.shape[1], device=device), 
        (n_nodes, n_nodes)
    ).coalesce()

    # 2. Add self-loops (I) to form A_hat = A + I
    # Create identity matrix I (sparse)
    diag_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
    I = torch.sparse_coo_tensor(
        diag_indices, 
        torch.ones(n_nodes, device=device), 
        (n_nodes, n_nodes)
    ).coalesce()

    # A_hat = A + I
    A_hat = (adj + I).coalesce()
    
    # 3. Calculate D_hat (degree of A_hat) and D_hat^(-1/2)
    row = A_hat.indices()[0]
    
    # Count degrees for each node (row is a vector of row indices of non-zero elements in A_hat)
    deg_hat = torch.bincount(row, minlength=n_nodes).float()
    
    # Calculate D_hat^(-1/2) diagonal values
    D_hat_invsqrt_diag = torch.pow(deg_hat.clamp(min=1e-12), -0.5)
    
    # 4. Create the sparse diagonal matrix D_hat^(-1/2)
    D_hat_invsqrt_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
    D_hat_invsqrt = torch.sparse_coo_tensor(
        D_hat_invsqrt_indices, 
        D_hat_invsqrt_diag, 
        (n_nodes, n_nodes)
    ).coalesce()
    
    # 5. Calculate A_norm = D_hat^(-1/2) * A_hat * D_hat^(-1/2)
    # Using torch.sparse.mm for sparse matrix multiplication
    A_norm = torch.sparse.mm(D_hat_invsqrt, torch.sparse.mm(A_hat, D_hat_invsqrt))

    return A_norm


# ============================================================================
# FIX 2: Improved alignment function with better Hungarian matching
# ============================================================================

def align_eigenvectors_improved(U_pred, U_exact, M=None, verbose=False):
    """
    Improved alignment with better overlap calculation and debugging.
    
    Args:
        U_pred: (n, k) predicted eigenvectors
        U_exact: (n, k) exact eigenvectors
        M: mass matrix for M-orthogonality
        verbose: print debug info
        
    Returns:
        U_aligned_ps: aligned eigenvectors
        permutation: permutation array
        signs: sign flips
    """
    n_modes = U_pred.shape[1]
    
    # Compute overlap matrix
    if M is not None:
        overlap = U_pred.T @ M @ U_exact  # (k, k) matrix
    else:
        overlap = U_pred.T @ U_exact
    
    if verbose:
        print("\n=== Alignment Debug Info ===")
        print(f"Overlap matrix shape: {overlap.shape}")
        print(f"Overlap matrix (first 5x5):\n{overlap[:5, :5]}")
        print(f"Max absolute overlap per row: {np.max(np.abs(overlap), axis=1)[:10]}")
    
    # Use absolute value for matching (ignore signs initially)
    overlap_abs = np.abs(overlap)
    
    if verbose:
        print(f"Max abs overlap: {np.max(overlap_abs):.4f}")
        print(f"Min abs overlap: {np.min(overlap_abs):.4f}")
    
    # Hungarian algorithm: maximize overlap = minimize negative overlap
    # row_ind[i] tells us which predicted mode matches exact mode i
    row_ind, col_ind = linear_sum_assignment(-overlap_abs)
    
    if verbose:
        print(f"\nHungarian matching (first 10):")
        for i in range(min(10, len(row_ind))):
            print(f"  Exact mode {col_ind[i]} <- Predicted mode {row_ind[i]} "
                  f"(overlap: {overlap_abs[row_ind[i], col_ind[i]]:.4f})")
    
    # Create permutation: perm[i] = which column of U_pred corresponds to exact mode i
    permutation = np.zeros(n_modes, dtype=int)
    for i in range(n_modes):
        permutation[col_ind[i]] = row_ind[i]
    
    # Apply permutation
    U_aligned = U_pred[:, permutation].copy()
    
    # Determine signs from the original overlap matrix
    signs = np.zeros(n_modes)
    for i in range(n_modes):
        pred_idx = permutation[i]
        signs[i] = np.sign(overlap[pred_idx, i])
        if signs[i] == 0:
            signs[i] = 1
    
    # Apply signs
    U_aligned_ps = U_aligned * signs
    
    if verbose:
        print(f"\nSign flips: {signs[:10]}")
        print("=== End Alignment Debug ===\n")
    
    return U_aligned_ps, permutation, signs


# ============================================================================
# FIX 3: Update comprehensive_diagnostics to use improved alignment
# ============================================================================

def comprehensive_diagnostics_improved(U_pred, U_exact, X, config, L=None, M=None):
    """
    Improved diagnostics with better alignment and clearer output.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EIGENMODE DIAGNOSTICS (IMPROVED)")
    print("="*80)
    
    # Setup: Compute Laplacian and Mass matrix
    if L is None or M is None:
        from mesh_helpers import compute_stiffness_and_mass_matrices
        from Mesh import Mesh
        mesh = Mesh(verts=X, connectivity=None)
        L, M = compute_stiffness_and_mass_matrices(mesh)
    
    # Make matrices dense for easier computation
    L_dense = L.toarray() if hasattr(L, 'toarray') else L
    M_dense = M.toarray() if hasattr(M, 'toarray') else M

    # 1. Compute Eigenvalues via Rayleigh Quotient
    def compute_rayleigh_quotient(U, L, M):
        lambda_list = []
        for i in range(U.shape[1]):
            u = U[:, i]
            Lu = L @ u
            Mu = M @ u
            lambda_i = (u.T @ Lu) / (u.T @ Mu + 1e-12)
            lambda_list.append(lambda_i)
        return np.array(lambda_list)

    lambda_pred = compute_rayleigh_quotient(U_pred, L_dense, M_dense)
    lambda_exact = compute_rayleigh_quotient(U_exact, L_dense, M_dense)
    
    print("\n--- BEFORE ALIGNMENT ---")
    print(f"Predicted eigenvalues (first 10): {np.round(lambda_pred[:10], 4)}")
    print(f"Exact eigenvalues (first 10): {np.round(lambda_exact[:10], 4)}")
    
    # 2. Alignment for Mode-by-Mode Tracking
    U_aligned_ps, permutation, signs = align_eigenvectors_improved(
        U_pred, U_exact, M_dense, verbose=True
    )
    lambda_pred_ordered = lambda_pred[permutation]
    
    # 3. Alignment for Subspace Error (Procrustes)
    U_aligned_procrustes, subspace_error = get_subspace_error_and_alignment(
        U_pred, U_exact, M_dense
    )
    
    # =================================================================
    #   DIAGNOSTIC OUTPUT
    # =================================================================
    
    # 1. EIGENVALUE COMPARISON
    print("\n" + "="*80)
    print("1. EIGENVALUE COMPARISON (Permutation-Aligned)")
    print("-" * 80)
    print(f"{'Mode':<6} {'λ_exact':<12} {'λ_pred':<12} {'Abs Error':<12} {'Rel Error':<12}")
    print("-" * 80)
    
    max_display = min(20, len(lambda_exact))
    for i in range(max_display):
        lambda_exact_i = lambda_exact[i]
        lambda_aligned_i = lambda_pred_ordered[i]
        abs_error = abs(lambda_aligned_i - lambda_exact_i)
        rel_error = abs_error / (abs(lambda_exact_i) + 1e-12)
        
        print(f"{i:<6} {lambda_exact_i:<12.6f} {lambda_aligned_i:<12.6f} "
              f"{abs_error:<12.6f} {rel_error:<12.6f}")
    
    # Summary statistics
    abs_errors = np.abs(lambda_pred_ordered - lambda_exact)
    rel_errors = abs_errors / (np.abs(lambda_exact) + 1e-12)
    print(f"\nSummary: Mean Abs Error = {np.mean(abs_errors):.6f}, "
          f"Mean Rel Error = {np.mean(rel_errors):.6f}")

    # 2. EIGENVECTOR ALIGNMENT
    print("\n" + "="*80)
    print("2. EIGENVECTOR ALIGNMENT (Mode-by-Mode)")
    print("-" * 80)
    print(f"{'Mode':<6} {'Perm':<8} {'Sign':<6} {'Cos Sim':<12} {'L2 Error':<12}")
    print("-" * 80)
    
    for i in range(max_display):
        perm_i = permutation[i]
        sign_i = signs[i]
        
        u_aligned = U_aligned_ps[:, i]
        u_exact = U_exact[:, i]
        
        # M-normalized cosine similarity
        inner_prod = u_aligned.T @ M_dense @ u_exact
        norm_u_aligned = np.sqrt(u_aligned.T @ M_dense @ u_aligned)
        norm_u_exact = np.sqrt(u_exact.T @ M_dense @ u_exact)
        cos_sim = abs(inner_prod) / (norm_u_aligned * norm_u_exact + 1e-12)
        
        # L2 error
        l2_error = np.linalg.norm(u_aligned - u_exact)
        
        print(f"{i:<6} {i}->{perm_i:<5} {sign_i:>4.0f}   "
              f"{cos_sim:<12.6f} {l2_error:<12.6f}")
    
    # Summary statistics
    cos_sims = []
    l2_errors = []
    for i in range(len(lambda_exact)):
        u_aligned = U_aligned_ps[:, i]
        u_exact = U_exact[:, i]
        inner_prod = u_aligned.T @ M_dense @ u_exact
        norm_u_aligned = np.sqrt(u_aligned.T @ M_dense @ u_aligned)
        norm_u_exact = np.sqrt(u_exact.T @ M_dense @ u_exact)
        cos_sim = abs(inner_prod) / (norm_u_aligned * norm_u_exact + 1e-12)
        l2_error = np.linalg.norm(u_aligned - u_exact)
        cos_sims.append(cos_sim)
        l2_errors.append(l2_error)
    
    print(f"\nSummary: Mean Cos Sim = {np.mean(cos_sims):.6f}, "
          f"Mean L2 Error = {np.mean(l2_errors):.6f}")

    # 3. SUBSPACE ALIGNMENT
    print("\n" + "="*80)
    print("3. SUBSPACE ALIGNMENT (Procrustes Analysis)")
    print("-" * 80)
    print(f"Subspace Error (Frobenius): {subspace_error:.6e}")
    print("-" * 80)

    # 4. ORTHONORMALITY
    print("\n" + "="*80)
    print("4. ORTHONORMALITY CHECK")
    print("-" * 80)
    UMU_pred = U_pred.T @ M_dense @ U_pred
    diag_pred = np.diag(UMU_pred)
    off_diag_max = np.max(np.abs(UMU_pred - np.diag(diag_pred)))
    print(f"Diagonal (first 10, should be ~1.0): {diag_pred[:10]}")
    print(f"Max off-diagonal: {off_diag_max:.6e}")
    print("-" * 80)
    
    # 5. VISUALIZATIONS
    if config.diagnostics_viz:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Eigenvalue comparison
        axes[0, 0].plot(lambda_exact[:30], 'o-', label='Exact', markersize=8, linewidth=2)
        axes[0, 0].plot(lambda_pred_ordered[:30], 's-', label='Predicted (Aligned)', 
                       markersize=6, alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Mode Index', fontsize=12)
        axes[0, 0].set_ylabel('Eigenvalue', fontsize=12)
        axes[0, 0].set_title('Eigenvalue Comparison', fontsize=14)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Relative errors
        axes[0, 1].semilogy(rel_errors[:30], 'o-', linewidth=2)
        axes[0, 1].set_xlabel('Mode Index', fontsize=12)
        axes[0, 1].set_ylabel('Relative Error', fontsize=12)
        axes[0, 1].set_title('Eigenvalue Relative Errors', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cosine similarities
        axes[0, 2].plot(cos_sims[:30], 'o-', linewidth=2)
        axes[0, 2].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (0.95)')
        axes[0, 2].set_xlabel('Mode Index', fontsize=12)
        axes[0, 2].set_ylabel('Cosine Similarity', fontsize=12)
        axes[0, 2].set_title('Eigenvector Alignment Quality', fontsize=14)
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Gram matrix
        im1 = axes[1, 0].imshow(np.abs(UMU_pred), cmap='RdYlGn_r', vmin=0, vmax=1.1)
        axes[1, 0].set_title('|U^T M U| - Orthonormality', fontsize=14)
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Plot 5: Overlap matrix
        overlap_matrix = U_aligned_ps.T @ M_dense @ U_exact
        im2 = axes[1, 1].imshow(np.abs(overlap_matrix), cmap='RdYlGn', vmin=0, vmax=1.1)
        axes[1, 1].set_title('|U_aligned^T M U_exact|', fontsize=14)
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Plot 6: Permutation visualization
        perm_matrix = np.zeros((config.n_modes, config.n_modes))
        for i in range(config.n_modes):
            perm_matrix[permutation[i], i] = 1
        im3 = axes[1, 2].imshow(perm_matrix[:30, :30], cmap='binary', vmin=0, vmax=1)
        axes[1, 2].set_title('Permutation Matrix (30x30)', fontsize=14)
        axes[1, 2].set_xlabel('Exact Mode', fontsize=12)
        axes[1, 2].set_ylabel('Predicted Mode', fontsize=12)
        
        plt.tight_layout()
        save_path = config.diagnostics_viz.replace('.png', '_improved.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDiagnostic plots saved to: {save_path}")
    
    print("\n" + "="*80)