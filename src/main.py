# multigrid_gnn_multires_physics.py
"""
Physics-informed Multigrid + GNN eigen-refinement
- Exact solve only on coarsest mesh
- Multiresolution GNN with residual + orthonormality + projection loss
- Coarse-to-fine prolongation only
"""

import numpy as np
import utils
import mesh_helpers
from multigrid_model import MultigridGNN
from config import PINNConfig


def main():
    config = PINNConfig.from_yaml()

    print("Loading mesh...")
    mesh = mesh_helpers.load_mesh(config.mesh_file, normalize=True)
    X_full = mesh.verts
    n_total = X_full.shape[0]
    K_total, M_total = mesh_helpers.compute_stiffness_and_mass_matrices(mesh)

    print("Loading coarse meshes...")
    mesh_list = []
    X_list = []
    hierarchy = []
    K_list = []
    M_list = []

    for coarse_mesh_file in config.coarse_mesh_files:
        coarse_mesh = mesh_helpers.load_mesh(coarse_mesh_file, normalize=True)
        X_coarse = coarse_mesh.verts
        n_coarse = X_coarse.shape[0]
        K_coarse, M_coarse = mesh_helpers.compute_stiffness_and_mass_matrices(coarse_mesh)
        
        mesh_list.append(coarse_mesh)
        X_list.append(X_coarse)
        hierarchy.append(n_coarse)
        K_list.append(K_coarse)
        M_list.append(M_coarse)

    mesh_list.append(mesh)
    X_list.append(X_full)
    hierarchy.append(n_total)
    K_list.append(K_total)
    M_list.append(M_total)

    for i, n_points in enumerate(hierarchy):
        print(f"  Level {i}: {n_points} points")

    if config.do_extensive_visuals:
        for level_idx_vis, n_points in enumerate(hierarchy):
            # Visualize selected points for this level
            mesh_helpers.visualize_mesh(mesh, title=f'Level {level_idx_vis}: {n_points} coarse grid Points', highlight_indices=n_points)
        print()

    # ------------------------
    # Level 0: exact coarse solve
    # ------------------------
    print(f"\nLEVEL 0: exact solve on {hierarchy[0]} points...")
    lambda0, U0, _, _ = utils.solve_eigenvalue_mesh(mesh_list[0], config.n_modes)
    if config.verbose:
        print("Coarse eigenvalues:", np.round(lambda0,6))

    # ------------------------
    # Coarse-to-fine prolongation
    # ------------------------
    U_prev = U0.copy()
    U_init_list, edge_index_list = [U0], [utils.build_knn_graph(X_list[0], k=config.k_neighbors)]
    for level in range(1, len(hierarchy)):
        P = utils.build_prolongation(X_list[level-1], X_list[level], k=config.prolongation_neighbors)
        U_init = P @ U_prev
        edge_index = utils.build_knn_graph(X_list[level], k=config.k_neighbors)

        U_init_list.append(U_init)
        edge_index_list.append(edge_index)

        U_prev = U_init.copy()

    # ------------------------
    # Train physics-informed GNN
    # ------------------------
    print("\nTraining physics-informed multiresolution GNN...")
    solver = MultigridGNN()
    U_pred_all = solver.train_multiresolution(X_list, K_list, M_list, U_init_list, edge_index_list, 
                                              epochs=config.epochs, lr=config.learning_rate, corr_scale=config.corrector_scale,
                                              w_res=config.weight_residual, w_orth=config.weight_orthogonal, w_proj=config.weight_projection,
                                              grad_clip=config.gradient_clipping, weight_decay=config.weight_decay, log_every=config.log_every,
                                              hidden_layers=config.hidden_layers, dropout=config.dropout)

    # ------------------------
    # Rayleigh-Ritz refinement per level
    # ------------------------
    node_offset = 0
    for level, (K_, M_, n_nodes_) in enumerate(zip(K_list, M_list, hierarchy)):
        U_pred = U_pred_all[node_offset:node_offset+n_nodes_]
        node_offset += n_nodes_
        vals_refined, _ = solver.refine_eigenvectors(U_pred, K_, M_)
        if config.verbose:
            print(f"Level {level} refined eigenvalues: {np.round(vals_refined,3)}")

    node_offset = sum(X.shape[0] for X in X_list[:-1])

    print(f"--- node_offset: {node_offset} ---")
    print(f"--- sum of nodes: {sum(hierarchy[:-1])}")

    U_finest = U_pred_all[node_offset:]
    M_finest = M_list[-1]

    # Check orthonormality
    UMU = U_finest.T @ M_finest @ U_finest
    utils.post_training_diagnostics(UMU, config.n_modes, config.diagnostics_viz)
    mesh_helpers.save_eigenfunctions(mesh, U_pred, config.n_modes, config.vtu_file)

    return U_pred


if __name__ == "__main__":
    main()
