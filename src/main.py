# multigrid_gnn_multires_physics.py
"""
Physics-informed Multigrid + GNN eigen-refinement
- Exact solve only on coarsest mesh
- Multiresolution GNN with residual + orthonormality + projection loss
- Coarse-to-fine prolongation only
"""

import numpy as np
from scipy.sparse.linalg import spsolve
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

    actual_hierarchy = []
    
    for i, n_points in enumerate(hierarchy):
        actual_count = len(indices_per_level[i])
        actual_hierarchy.append(actual_count)
        print(f"  Level {i}: {actual_count} points (voxel downsampling, target: {n_points})")

    print("Actual hierarchy:", actual_hierarchy)

    if config.do_extensive_visuals:
        for level_idx_vis, n_points in enumerate(indices_per_level):
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
        U_init_rough = P @ U_prev

        # --- SMOOTHING STEP ---
        # Prolongation introduces high-freq noise that looks like high eigenvalues.
        # We smooth it using implicit diffusion: (M + alpha*L) u = M u_rough
        print(f"  Level {level}: Prolongation & Smoothing...")
        L_fine, M_fine = robust_laplacian.point_cloud_laplacian(Xf)
        alpha = 0.01  # Smoothing strength (0.01 is usually good for point clouds)

        # Solve linear system for smoother U
        # Compute cost estimate
        if Xf.shape[0] < 5000:
            # Small enough for direct solve
            L_fine, M_fine = robust_laplacian.point_cloud_laplacian(Xf)
            A_smooth = M_fine + alpha * L_fine
            B_smooth = M_fine @ U_init_rough
            U_init = spsolve(A_smooth, B_smooth)
        else:
            # Too large - use cheap Jacobi smoothing or skip
            L_fine, M_fine = robust_laplacian.point_cloud_laplacian(Xf)
            U_init = utils.jacobi_smooth(M_fine, L_fine, U_init_rough, alpha=0.05, n_iters=5)

        # Normalize again just in case
        # (Optional, but good practice as smoothing can dampen amplitude)
        # norm = np.sqrt(np.diag(U_init.T @ M_fine @ U_init))
        # U_init = U_init / norm
        # ----------------------

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
    U_pred_all = solver.train_multiresolution(X_list, U_init_list, edge_index_list, lambda0,
                                              epochs=config.epochs, lr=config.learning_rate, corr_scale=config.corrector_scale,
                                              w_res=config.weight_residual, w_orth=config.weight_orthogonal, w_proj=config.weight_projection, w_trace=config.weight_trace, w_order=config.w_order, w_eigen=config.w_eigen,
                                              grad_clip=config.gradient_clipping, weight_decay=config.weight_decay, log_every=config.log_every,
                                              hidden_layers=config.hidden_layers, dropout=config.dropout)

    # === CRITICAL FIX: Extract finest level correctly ===

    node_offset = sum(actual_hierarchy[:-1])  # Sum of all coarse levels
    print(f"\n--- Extracting finest level ---")
    print(f"Node offset: {node_offset}")
    print(f"Total nodes in U_pred_all: {U_pred_all.shape[0]}")
    print(f"Expected finest level nodes: {actual_hierarchy[-1]}")
    
    U_finest = U_pred_all[node_offset:node_offset + actual_hierarchy[-1]]
    print(f"Extracted U_finest shape: {U_finest.shape}")
    
    # Verify extraction
    assert U_finest.shape[0] == actual_hierarchy[-1], f"Mismatch! Got {U_finest.shape[0]}, expected {actual_hierarchy[-1]}"
    
    X_finest = X_list[-1]
    L_finest, M_finest = robust_laplacian.point_cloud_laplacian(X_finest)

    # === CRITICAL FIX: Perform Rayleigh-Ritz refinement on finest level ===
    print("\n--- Rayleigh-Ritz refinement on finest level ---")
    vals_refined, U_refined = solver.refine_eigenvectors(U_finest, L_finest, M_finest)
    print(f"Refined eigenvalues (first 10): {np.round(vals_refined[:10], 6)}")

    # Check orthonormality
    UMU = U_refined.T @ M_finest @ U_refined
    utils.post_training_diagnostics(UMU, config.n_modes, config.diagnostics_viz)

    # Save eigenfunctions
    mesh_helpers.save_eigenfunctions(mesh, U_refined, config.n_modes, config.vtu_file)

    # === CRITICAL FIX: Compare refined eigenvectors against exact solution ===
    print("\n--- Computing exact solution for comparison ---")
    lambda_exact, U_exact, _, _ = utils.solve_eigenvalue_problem(X_full, config.n_modes)
    print(f"Exact eigenvalues (first 10): {np.round(lambda_exact[:10], 6)}")

    utils.comprehensive_diagnostics(U_finest, U_exact, X_finest, config)

    # Run comprehensive diagnostics with REFINED eigenvectors
    #utils.comprehensive_diagnostics_improved(U_refined, U_exact, X_full, config, K_finest, M_finest)

    return U_finest


if __name__ == "__main__":
    main()