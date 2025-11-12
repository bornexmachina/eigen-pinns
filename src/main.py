# multigrid_gnn_multires_physics.py
"""
Physics-informed Multigrid + GNN eigen-refinement
- Exact solve only on coarsest mesh
- Multiresolution GNN with residual + orthonormality + projection loss
- Coarse-to-fine prolongation only
"""

import os
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from Mesh import Mesh
import robust_laplacian



def main():
    mesh_path = "bunny.obj"
    n_modes = 128
    hierarchy = [256, 512, 1024]  # final level is full mesh
    k_neighbors = 4
    epochs = 10_000

    print("Loading mesh...")
    mesh = Mesh(mesh_path)
    mesh = MultigridGNN.normalize_mesh(mesh)
    X_full = mesh.verts
    n_total = X_full.shape[0]
    hierarchy = [n for n in hierarchy if n <= n_total]
    if hierarchy[-1] != n_total:
        hierarchy.append(n_total)
    print("Hierarchy:", hierarchy)

    # Use Voxel Grid Downsampling for nested hierarchy
    print("Computing voxel grid downsampling hierarchy...")
    indices_per_level = voxel_downsampling(X_full, hierarchy)
    
    for i, n_points in enumerate(hierarchy):
        actual_count = len(indices_per_level[i])
        print(f"  Level {i}: {actual_count} points (voxel downsampling, target: {n_points})")
    

    #for level_idx_vis, n_points in enumerate(hierarchy):
    #    # Visualize selected points for this level
    #    visualize_mesh(mesh, 
    #                  title=f'Level {level_idx_vis}: {len(indices_per_level[level_idx_vis])} Voxel Downsampled Points',
    #                  highlight_indices=indices_per_level[level_idx_vis])
    #print()

    solver = MultigridGNN()

    # ------------------------
    # Level 0: exact coarse solve
    # ------------------------
    idx0 = indices_per_level[0]
    X0 = X_full[idx0]
    print(f"\nLEVEL 0: exact solve on {X0.shape[0]} points...")
    lambda0, U0, L0, M0 = solver.solve_eigenvalue_problem(X0, n_modes)
    print("Coarse eigenvalues:", np.round(lambda0,6))

    # ------------------------
    # Coarse-to-fine prolongation
    # ------------------------
    U_prev = U0.copy()
    X_list, U_init_list, edge_index_list = [X0], [U0], [solver.build_knn_graph(X0, k=k_neighbors)]
    for level in range(1, len(hierarchy)):
        idx_coarse = indices_per_level[level-1]
        idx_fine = indices_per_level[level]
        Xc = X_full[idx_coarse]
        Xf = X_full[idx_fine]

        P = solver.build_prolongation(Xc, Xf, k=1)
        U_init = P @ U_prev
        edge_index = solver.build_knn_graph(Xf, k=k_neighbors)

        X_list.append(Xf)
        U_init_list.append(U_init)
        edge_index_list.append(edge_index)

        U_prev = U_init.copy()

    # ------------------------
    # Train physics-informed GNN
    # ------------------------
    print("\nTraining physics-informed multiresolution GNN...")
    U_pred_all = solver.train_multiresolution(X_list, U_init_list, edge_index_list,
                                              epochs=epochs)

    # ------------------------
    # Rayleigh-Ritz refinement per level
    # ------------------------
    node_offset = 0
    for level, X in enumerate(X_list):
        n_nodes = X.shape[0]
        U_pred = U_pred_all[node_offset:node_offset+n_nodes]
        node_offset += n_nodes
        L, M = robust_laplacian.point_cloud_laplacian(X)
        vals_refined, _ = solver.refine_eigenvectors(U_pred, L, M)
        print(f"Level {level} refined eigenvalues: {np.round(vals_refined,3)}")

    finest_idx = len(hierarchy) - 1
    node_offset = sum(X.shape[0] for X in X_list[:-1])
    U_finest = U_pred_all[node_offset:]

    X_finest = X_list[-1]
    L_finest, M_finest = robust_laplacian.point_cloud_laplacian(X_finest)

    # Check orthonormality
    tmp_mat = U_finest.T @ M_finest @ U_finest
    print("\nOrthonormality check (should be Identity):")
    print("Diagonal:", np.diag(tmp_mat)[:10])
    print("Off-diagonal max:", np.max(np.abs(tmp_mat - np.eye(n_modes))))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tmp_mat, cmap='magma', vmin=-0.1, vmax=1.1)
    plt.colorbar()
    plt.title('U^T M U (should be Identity)')
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(tmp_mat - np.eye(n_modes)), cmap='magma')
    plt.colorbar()
    plt.title('|U^T M U - I|')
    plt.show()

    return U_pred