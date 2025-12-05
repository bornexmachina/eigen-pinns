import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from Mesh import Mesh
import mesh_helpers
import utils


def _voxel_downsampling(mesh, hierarchy):
    """
    Voxel grid downsampling for nested hierarchical sampling.
    Args:
        mesh: Mesh object
        hierarchy: list of target point counts for each level
    Returns:
        indices_per_level: dict mapping level index to array of selected point indices
    """
    points = mesh.verts

    n_points = points.shape[0]
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    extent = max_bound - min_bound
    
    indices_per_level = {}
    
    for level_idx, target_count in enumerate(hierarchy):
        if target_count >= n_points:
            indices_per_level[level_idx] = np.arange(n_points)
            continue
        
        # Iteratively adjust voxel size to hit target
        # Start with a conservative estimate (assuming ~50% occupancy)
        volume = np.prod(extent)
        voxel_size = (volume / (target_count * 2)) ** (1/3)
        
        best_indices = None
        best_diff = float('inf')
        
        # Try multiple voxel sizes to get close to target
        for scale in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
            current_voxel_size = voxel_size * scale
            
            # Compute voxel grid
            grid_dims = np.ceil(extent / current_voxel_size).astype(int) + 1
            voxel_indices = ((points - min_bound) / current_voxel_size).astype(int)
            voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)
            
            # Flatten to 1D
            voxel_ids = (voxel_indices[:, 0] * grid_dims[1] * grid_dims[2] + 
                        voxel_indices[:, 1] * grid_dims[2] + 
                        voxel_indices[:, 2])
            
            # Select one point per voxel (closest to center)
            unique_voxels = np.unique(voxel_ids)
            selected_indices = []
            
            for voxel_id in unique_voxels:
                mask = (voxel_ids == voxel_id)
                voxel_points_idx = np.where(mask)[0]
                
                # Compute voxel center
                voxel_3d = np.array([
                    voxel_id // (grid_dims[1] * grid_dims[2]),
                    (voxel_id // grid_dims[2]) % grid_dims[1],
                    voxel_id % grid_dims[2]
                ])
                voxel_center = min_bound + (voxel_3d + 0.5) * current_voxel_size
                
                # Find closest point to center
                voxel_points = points[voxel_points_idx]
                distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
                closest_idx = voxel_points_idx[np.argmin(distances)]
                selected_indices.append(closest_idx)
            
            # Check if this is closer to target
            diff = abs(len(selected_indices) - target_count)
            if diff < best_diff:
                best_diff = diff
                best_indices = np.array(selected_indices)
            
            # If we're very close or over target, we can stop
            if len(selected_indices) >= target_count * 0.95:
                break
        
        indices_per_level[level_idx] = best_indices[:target_count] if len(best_indices) > target_count else best_indices

    # finally add the full mesh
    indices_per_level[level_idx+1] = np.arange(n_points)
    
    return indices_per_level


def _farthest_point_sampling(mesh, hierarchy):
    """
    Farthest Point Sampling (FPS) for nested hierarchical sampling.
    Args:
        mesh: Mesh object
        hierarchy: list of target point counts for each level
    Returns:
        indices_per_level: dict mapping level index to array of selected point indices
    """
    points = mesh.verts
    n_samples = hierarchy[-1]

    n_points = points.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)
    
    rng = np.random.default_rng()
    
    # Start with a random point
    selected_indices = [rng.integers(0, n_points)]
    distances = np.full(n_points, np.inf)
    
    for _ in range(n_samples - 1):
        # Update distances to nearest selected point
        last_selected = selected_indices[-1]
        dists_to_last = np.linalg.norm(points - points[last_selected], axis=1)
        distances = np.minimum(distances, dists_to_last)
        
        # Select point farthest from all selected points
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)


    all_sampled_indices = np.array(selected_indices)
    
    # Create nested levels by taking first n points from FPS result
    indices_per_level = {}
    for i, n_points in enumerate(hierarchy):
        indices_per_level[i] = all_sampled_indices[:n_points].copy()

    # finally add the full mesh
    indices_per_level[i+1] = all_sampled_indices.copy()
    
    return indices_per_level


def _simplify_mesh_decimation(mesh, hierarchy):
    """Simplify mesh using quadric decimation algorithm."""
    # Read input mesh
    vertices = mesh.verts
    faces = mesh.connectivity
    
    if faces is None:
        raise ValueError("No triangle or polygon cells found in mesh")
    
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to PyVista mesh
    faces_pv = np.hstack([[3] + list(face) for face in faces])
    pv_mesh = pv.PolyData(vertices, faces_pv)
    
    simplified_meshes = []
    
    for target_v in hierarchy:
        print(f"\nSimplifying to ~{target_v} vertices...")
        
        # Calculate target reduction ratio
        reduction = 1.0 - (target_v / len(vertices))
        reduction = max(0.0, min(0.99, reduction))  # Clamp between 0 and 0.99
        
        print(f"  Target reduction: {reduction*100:.1f}%")
        
        # Apply decimation
        simplified = pv_mesh.decimate(
            reduction,
            volume_preservation=True
        )
        
        # Extract vertices and faces
        verts_simplified = simplified.points
        faces_simplified = simplified.faces.reshape(-1, 4)[:, 1:4]
        
        output_mesh = Mesh(verts=verts_simplified, connectivity=faces_simplified)
        simplified_meshes.append(output_mesh)
    
    return simplified_meshes


class Sampler:
    def __init__(self, config):
        self.sampler_type = config.sampler_type
        self.edge_computation_type = config.edge_computation_type
        self.k_neighbors = config.k_neighbors
        self.prolongation_neighbors = config.prolongation_neighbors
        self.n_modes = config.n_modes
        self.hierarchy = config.hierarchy
        self.meshes = []
        self.X_list = []
        self.K_list = []
        self.M_list = []
        self.P_list = []
        self.U_list = []
        self.actual_hierarchy = []
        self.edge_index_list = []
        self.indices_per_level = []

        if self.edge_computation_type != 'connectivity_based':
            self.edge_computation_type = 'knn_based'

        if self.sampler_type not in ['farthest_point', 'voxel_downsampling', 'graph_coarsening']:
            raise ValueError(f"sampler_type must be 'farthest_point', 'voxel_downsampling' or 'graph_coarsening', got '{self.sampler_type}'")
        
    def _grid_coarsening(self, mesh, hierarchy):
        if self.sampler_type == 'farthest_point':
            return _farthest_point_sampling(mesh, hierarchy)
        if self.sampler_type == 'voxel_downsampling':
            return _voxel_downsampling(mesh, hierarchy)
        
    def _assemble_X_K_M(self, mesh, hierarchy):

        if self.sampler_type == 'graph_coarsening':
            coarse_meshes = _simplify_mesh_decimation(mesh, hierarchy)
            coarse_meshes.append(mesh)

            self.meshes = coarse_meshes

            for coarse_mesh in coarse_meshes:
                X_coarse = coarse_mesh.verts
                n_coarse = X_coarse.shape[0]
                K_coarse, M_coarse = mesh_helpers.compute_stiffness_and_mass_matrices(coarse_mesh)

                self.X_list.append(X_coarse)
                self.K_list.append(K_coarse)
                self.M_list.append(M_coarse)
                self.actual_hierarchy.append(n_coarse)

        if self.sampler_type in ['farthest_point', 'voxel_downsampling']:
            self.indices_per_level = self._grid_coarsening(mesh, hierarchy)
            self.meshes.append(mesh)

            for idx in self.indices_per_level:
                X_coarse = mesh.verts[idx]
                n_coarse = X_coarse.shape[0]
                K_coarse, M_coarse = mesh_helpers.compute_laplacian_and_mass_matrices(X_coarse)

                self.X_list.append(X_coarse)
                self.K_list.append(K_coarse)
                self.M_list.append(M_coarse)
                self.actual_hierarchy.append(n_coarse)

    def _assemble_edge_list(self):
        if self.sampler_type == 'graph_coarsening':
            if self.edge_computation_type == 'knn_based':
                for X in self.X_list:
                    self.edge_index_list.append(utils.build_knn_graph(X, k=self.k_neighbors))

            if self.edge_computation_type == 'connectivity_based':
                for mesh in self.meshes:
                    self.edge_index_list.append(mesh_helpers.mesh_to_edge_index(mesh))
        
        if self.sampler_type in ['farthest_point', 'voxel_downsampling']:
            for X in self.X_list:
                self.edge_index_list.append(utils.build_knn_graph(X, k=self.k_neighbors))

    def _assemble_P_U(self):
        if self.sampler_type == 'graph_coarsening':
            _, U_0, _, _ = utils.solve_eigenvalue_mesh(self.meshes[0], self.n_modes)
        if self.sampler_type in ['farthest_point', 'voxel_downsampling']:
            _, U_0, _, _ = utils.solve_eigenvalue_point_cloud(self.X_list[0], self.n_modes)

        self.U_list.append((U_0))

        U_prev = U_0.copy()
        for level in range(1, len(self.X_list)):
            P = utils.build_prolongation(self.X_list[level-1], self.X_list[level], k=self.prolongation_neighbors)
            self.P_list.append(P)

            U_init = P @ U_prev
            U_init = utils.jacobi_smooth(self.M_list[level], self.K_list[level], U_init, alpha=0.1, n_iters=10)
            self.U_list.append(U_init)

            U_prev = U_init.copy()

    def preprocess_mesh(self, mesh):
        self._assemble_X_K_M(mesh, self.hierarchy)
        self._assemble_edge_list()
        self._assemble_P_U()


    def _visualize_decimation(self, output_prefix):
        # Plot the simplified meshes
        print("\nGenerating plots...")
        n_meshes = len(self.meshes)
        fig = plt.figure(figsize=(5*n_meshes, 5))

        # Plot simplified meshes
        for idx, mesh in enumerate(self.meshes):
            cmap = 'plasma'
            if idx == n_meshes-1:
                cmap = 'viridis'
            faces = mesh.connectivity
            ax = fig.add_subplot(1, n_meshes, idx, projection='3d')
            ax.plot_trisurf(mesh.points[:, 0], 
                            mesh.points[:, 1], 
                            mesh.points[:, 2],
                            triangles=faces,
                            cmap=cmap, 
                            alpha=0.8, 
                            edgecolor='none')
            ax.set_title(f'Mesh with \n{len(mesh.points)} vertices')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=95, azim=-90)

        plt.tight_layout()
        plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_prefix}_comparison.png")
        plt.show()

    def _visualize_point_sampling(self, output_prefix):
        n_meshes = len(self.indices_per_level)
        mesh = self.meshes[0]
        fig = plt.figure(figsize=(5*n_meshes, 5))
        for idx, highlight_indices in enumerate(self.indices_per_level):
            ax = fig.add_subplot(1, n_meshes, idx, projection='3d')
            
            # Plot full mesh with transparency
            ax.plot_trisurf(mesh.verts[:, 0], mesh.verts[:, 1], mesh.verts[:, 2], triangles=mesh.connectivity, alpha=0.3)
            
            # Highlight specific points if provided
            if highlight_indices is not None:
                highlighted_verts = mesh.verts[highlight_indices]
                ax.scatter(highlighted_verts[:, 0], highlighted_verts[:, 1], highlighted_verts[:, 2], c='fuchsia', s=10, alpha=0.8, label=f'{len(highlight_indices)} selected points')
                ax.legend()
            
            ax.set_title("Mesh with highlighted points used for downsampling")
            ax.view_init(elev=95, azim=-90)

        plt.tight_layout()
        plt.savefig(f'{output_prefix}_downsampling.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_prefix}_downsampling.png")
        plt.show()

    def visualize(self, output_prefix):
        if self.sampler_type == 'graph_coarsening':
            self._visualize_decimation(self, output_prefix)
        if self.sampler_type in ['farthest_point', 'voxel_downsampling']:
            self._visualize_point_sampling(self, output_prefix)
