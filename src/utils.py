import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import robust_laplacian
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
