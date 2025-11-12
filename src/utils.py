import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import torch


def sp_to_torch_sparse(A):
    A = A.tocoo()
    indices = np.vstack((A.row, A.col)).astype(np.int64)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(i, v, A.shape).coalesce()


def normalize_columns_np(U, eps=1e-12):
    norms = np.linalg.norm(U, axis=0) + eps
    return U / norms, norms


def normalize_columns_torch(U, eps=1e-12):
    norms = torch.norm(U, dim=0) + eps
    return U / norms, norms


def build_prolongation(X_coarse, X_fine, k=1):
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


def build_knn_graph(X, k=4):
    n_points = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, neighbors = nbrs.kneighbors(X)
    rows, cols = [], []
    for i in range(n_points):
        for j in neighbors[i][1:]:
            rows.append(i)
            cols.append(j)
    return torch.LongTensor([rows, cols]).to(torch.long)
