# multigrid_gnn_multires_physics.py
"""
Physics-informed Multigrid + GNN eigen-refinement
- Exact solve only on coarsest mesh
- Multiresolution GNN with residual + orthonormality + projection loss
- Coarse-to-fine prolongation only
"""

import numpy as np
import robust_laplacian
import utils
import mesh_helpers
import samplers
from multigrid_model import MultigridGNN
from config import PINNConfig


def main():
    config = PINNConfig.from_yaml()

    print("Loading mesh...")
    mesh = mesh_helpers.load_mesh(config.mesh_file, normalize=True)
    X_full = mesh.verts
    n_total = X_full.shape[0]
    hierarchy = [n for n in config.hierarchy if n <= n_total]
    if hierarchy[-1] != n_total:
        hierarchy.append(n_total)
    print("Hierarchy:", hierarchy)

    # Use Voxel Grid Downsampling for nested hierarchy
    print("Computing voxel grid downsampling hierarchy...")
    indices_per_level = samplers.voxel_downsampling(X_full, hierarchy)
    
    for i, n_points in enumerate(hierarchy):
        actual_count = len(indices_per_level[i])
        print(f"  Level {i}: {actual_count} points (voxel downsampling, target: {n_points})")

    if config.do_extensive_visuals:
        for level_idx_vis, n_points in enumerate(hierarchy):
            # Visualize selected points for this level
            mesh_helpers.visualize_mesh(mesh, title=f'Level {level_idx_vis}: {len(indices_per_level[level_idx_vis])} Voxel Downsampled Points', highlight_indices=indices_per_level[level_idx_vis])
        print()

    
    # ------------------------
    # Level 0: exact coarse solve
    # ------------------------
    idx0 = indices_per_level[0]
    X0 = X_full[idx0]
    print(f"\nLEVEL 0: exact solve on {X0.shape[0]} points...")
    lambda0, U0, L0, M0 = utils.solve_eigenvalue_problem(X0, config.n_modes)
    if config.verbose:
        print("Coarse eigenvalues:", np.round(lambda0,6))

    # ------------------------
    # Coarse-to-fine prolongation
    # ------------------------
    U_prev = U0.copy()
    X_list, U_init_list, edge_index_list = [X0], [U0], [utils.build_knn_graph(X0, k=config.k_neighbors)]
    for level in range(1, len(hierarchy)):
        idx_coarse = indices_per_level[level-1]
        idx_fine = indices_per_level[level]
        Xc = X_full[idx_coarse]
        Xf = X_full[idx_fine]

        P = utils.build_prolongation(Xc, Xf, k=config.prolongation_neighbors)
        U_init = P @ U_prev
        edge_index = utils.build_knn_graph(Xf, k=config.k_neighbors)

        X_list.append(Xf)
        U_init_list.append(U_init)
        edge_index_list.append(edge_index)

        U_prev = U_init.copy()

    # ------------------------
    # Train physics-informed GNN
    # ------------------------
    print("\nTraining physics-informed multiresolution GNN...")
    solver = MultigridGNN()
    U_pred_all = solver.train_multiresolution(X_list, U_init_list, edge_index_list, 
                                              epochs=config.epochs, lr=config.learning_rate, corr_scale=config.corrector_scale,
                                              w_res=config.weight_residual, w_orth=config.weight_orthogonal, w_proj=config.weight_projection,
                                              grad_clip=config.gradient_clipping, weight_decay=config.weight_decay, log_every=config.log_every,
                                              hidden_layers=config.hidden_layers, dropout=config.dropout)

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
        if config.verbose:
            print(f"Level {level} refined eigenvalues: {np.round(vals_refined,3)}")

    node_offset = sum(X.shape[0] for X in X_list[:-1])
    U_finest = U_pred_all[node_offset:]

    X_finest = X_list[-1]
    _, M_finest = robust_laplacian.point_cloud_laplacian(X_finest)

    # Check orthonormality
    UMU = U_finest.T @ M_finest @ U_finest
    utils.post_training_diagnostics(UMU, config.n_modes, config.diagnostics_viz)
    mesh_helpers.save_eigenfunctions(mesh, U_pred, config.n_modes, config.vtu_file)


    lambda_exact, U_exact, _, _ = utils.solve_eigenvalue_problem(X_full, config.n_modes)

    utils.comprehensive_diagnostics(U_pred, U_exact, X, config)

    return U_pred


if __name__ == "__main__":
    main()
