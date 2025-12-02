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


    #lambda_exact, _, _, _ = utils.solve_eigenvalue_mesh(mesh, config.n_modes)

    #print(lambda_exact)
    #print(f"*******************************")

    print("Loading coarse meshes...")
    mesh_list = []
    X_list = []
    hierarchy = []
    K_list = []
    M_list = []
    P_list = []

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

        P_list.append(P)

        U_init = P @ U_prev
        U_init = utils.jacobi_smooth(M_list[level], K_list[level], U_init, alpha=0.1, n_iters=10)
        # edge_index = utils.build_knn_graph(X_list[level], k=config.k_neighbors)
        edge_index = mesh_helpers.mesh_to_edge_index(mesh_list[level])

        U_init_list.append(U_init)
        edge_index_list.append(edge_index)

        U_prev = U_init.copy()

    # ------------------------
    # Train physics-informed GNN
    # ------------------------
    print("\nTraining physics-informed multiresolution GNN...")
    solver = MultigridGNN()
    U_pred_all = solver.train_multiresolution(X_list, K_list, M_list, U_init_list, P_list, edge_index_list, 
                                              epochs=config.epochs, lr=config.learning_rate, corr_scale=config.corrector_scale,
                                              w_res=config.weight_residual, w_orth=config.weight_orthogonal, w_proj=config.weight_projection, w_trace=config.weight_trace, w_order=config.w_order, w_eigen=config.w_eigen,
                                              grad_clip=config.gradient_clipping, weight_decay=config.weight_decay, log_every=config.log_every,
                                              hidden_layers=config.hidden_layers, dropout=config.dropout)

    # === CRITICAL FIX: Extract finest level correctly ===
    node_offset = sum(hierarchy[:-1])  # Sum of all coarse levels
    print(f"\n--- Extracting finest level ---")
    print(f"Node offset: {node_offset}")
    print(f"Total nodes in U_pred_all: {U_pred_all.shape[0]}")
    print(f"Expected finest level nodes: {hierarchy[-1]}")
    
    U_finest = U_pred_all[node_offset:node_offset + hierarchy[-1]]
    print(f"Extracted U_finest shape: {U_finest.shape}")
    
    # Verify extraction
    assert U_finest.shape[0] == hierarchy[-1], f"Mismatch! Got {U_finest.shape[0]}, expected {hierarchy[-1]}"
    
    K_finest = K_list[-1]
    M_finest = M_list[-1]

    # === CRITICAL FIX: Perform Rayleigh-Ritz refinement on finest level ===
    print("\n--- Rayleigh-Ritz refinement on finest level ---")
    vals_refined, U_refined = solver.refine_eigenvectors(U_finest, K_finest, M_finest)
    print(f"Refined eigenvalues (first 10): {np.round(vals_refined[:10], 6)}")

    # Check orthonormality
    UMU = U_refined.T @ M_finest @ U_refined
    utils.post_training_diagnostics(UMU, config.n_modes, config.diagnostics_viz)

    # Save eigenfunctions
    mesh_helpers.save_eigenfunctions(mesh, U_refined, config.n_modes, config.vtu_file)

    # === CRITICAL FIX: Compare refined eigenvectors against exact solution ===
    print("\n--- Computing exact solution for comparison ---")
    lambda_exact, U_exact, _, _ = utils.solve_eigenvalue_mesh(mesh, config.n_modes)
    print(f"Exact eigenvalues (first 10): {np.round(lambda_exact[:10], 6)}")

    # Run comprehensive diagnostics with REFINED eigenvectors
    utils.comprehensive_diagnostics_improved(U_refined, U_exact, X_full, config, K_finest, M_finest)

    return U_refined


if __name__ == "__main__":
    main()
