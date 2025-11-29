# main.py - UPDATED with pure learning (no eigenvalue shortcuts)
"""
Physics-informed Multigrid + GNN eigen-refinement
- Exact solve only on coarsest mesh  
- Pure neural learning of eigenmodes (no eigh shortcuts)
- Curriculum learning for efficient training
"""

import numpy as np
from scipy.sparse.linalg import spsolve
import robust_laplacian
import utils
import mesh_helpers
import samplers
from config import PINNConfig
# Import the improved training class
from curriculum_training import ImprovedMultigridGNN


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

    # Voxel Grid Downsampling
    print("Computing voxel grid downsampling hierarchy...")
    indices_per_level = samplers.voxel_downsampling(X_full, hierarchy)
    
    for i, n_points in enumerate(hierarchy):
        actual_count = len(indices_per_level[i])
        print(f"  Level {i}: {actual_count} points (voxel downsampling, target: {n_points})")

    if config.do_extensive_visuals:
        for level_idx_vis, n_points in enumerate(hierarchy):
            mesh_helpers.visualize_mesh(
                mesh, 
                title=f'Level {level_idx_vis}: {len(indices_per_level[level_idx_vis])} Voxel Downsampled Points',
                highlight_indices=indices_per_level[level_idx_vis]
            )
        print()

    # ------------------------
    # Level 0: exact coarse solve (only for initialization!)
    # ------------------------
    idx0 = indices_per_level[0]
    X0 = X_full[idx0]
    print(f"\nLEVEL 0: exact solve on {X0.shape[0]} points (initialization only)...")
    lambda0, U0, L0, M0 = utils.solve_eigenvalue_problem(X0, config.n_modes)

    if config.verbose:
        print("Coarse eigenvalues (for reference):", np.round(lambda0, 6))

    # ------------------------
    # Coarse-to-fine prolongation with smoothing
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

        # Smoothing step
        print(f"  Level {level}: Prolongation & Smoothing...")
        L_fine, M_fine = robust_laplacian.point_cloud_laplacian(Xf)
        alpha = 0.01
        
        if Xf.shape[0] < 5000:
            A_smooth = M_fine + alpha * L_fine
            B_smooth = M_fine @ U_init_rough
            U_init = spsolve(A_smooth, B_smooth)
        else:
            U_init = utils.jacobi_smooth(M_fine, L_fine, U_init_rough, alpha=0.05, n_iters=5)

        edge_index = utils.build_knn_graph(Xf, k=config.k_neighbors)

        X_list.append(Xf)
        U_init_list.append(U_init)
        edge_index_list.append(edge_index)

        U_prev = U_init.copy()

    # ------------------------
    # Train physics-informed GNN (pure learning - no eigh!)
    # ------------------------
    print("\n" + "="*80)
    print("TRAINING PHASE: Pure Neural Learning of Eigenmodes")
    print("No eigenvalue computation shortcuts - everything is learned!")
    print("="*80 + "\n")
    

    
    solver = ImprovedMultigridGNN()
    U_pred_all = solver.train_multiresolution(
        X_list, U_init_list, edge_index_list, lambda0,
        epochs=config.epochs, 
        lr=config.learning_rate, 
        corr_scale=config.corrector_scale,
        w_res=config.weight_residual, 
        w_orth=config.weight_orthogonal, 
        w_proj=config.weight_projection, 
        w_trace=config.weight_trace,  # This should be VERY small now
        w_order=config.w_order, 
        w_eigen=config.w_eigen,
        grad_clip=config.gradient_clipping, 
        weight_decay=config.weight_decay, 
        log_every=config.log_every,
        hidden_layers=config.hidden_layers, 
        dropout=config.dropout
    )

    # ------------------------
    # Post-training analysis (no refinement - pure learned modes!)
    # ------------------------
    node_offset = 0
    print("\n" + "="*80)
    print("LEARNED EIGENVALUES (via Rayleigh Quotient)")
    print("="*80)
    
    for level, X in enumerate(X_list):
        n_nodes = X.shape[0]
        U_pred = U_pred_all[node_offset:node_offset+n_nodes]
        node_offset += n_nodes
        
        # Compute learned eigenvalues
        L, M = robust_laplacian.point_cloud_laplacian(X)
        lambda_learned = []
        for i in range(config.n_modes):
            u = U_pred[:, i]
            Lu = L @ u
            Mu = M @ u
            lam = (u.T @ Lu) / (u.T @ Mu + 1e-12)
            lambda_learned.append(lam)
        
        lambda_learned = np.array(lambda_learned)
        
        if config.verbose or level == len(X_list) - 1:
            print(f"\nLevel {level} learned eigenvalues: {np.round(lambda_learned[:10], 3)}")

    # Get finest level predictions
    node_offset = sum(X.shape[0] for X in X_list[:-1])
    U_finest = U_pred_all[node_offset:]
    X_finest = X_list[-1]
    
    # Check orthonormality
    _, M_finest = robust_laplacian.point_cloud_laplacian(X_finest)
    UMU = U_finest.T @ M_finest @ U_finest
    utils.post_training_diagnostics(UMU, config.n_modes, config.diagnostics_viz)
    
    # Save results
    mesh_helpers.save_eigenfunctions(mesh, U_finest, config.n_modes, config.vtu_file)

    # Full diagnostic comparison
    print("\n" + "="*80)
    print("COMPARING LEARNED vs EXACT MODES")
    print("="*80)
    lambda_exact, U_exact, _, _ = utils.solve_eigenvalue_problem(X_full, config.n_modes)
    utils.comprehensive_diagnostics(U_finest, U_exact, X_finest, config)

    return U_finest


if __name__ == "__main__":
    main()